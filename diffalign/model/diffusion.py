from collections import defaultdict
from collections import Counter
from itertools import compress
import itertools
import wandb
import time
import logging
import os
import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_batch
import torch_geometric.data
from rdkit import Chem

from diffalign.model.diffusion_abstract import DiscreteDenoisingDiffusion
from diffalign.data import graph, mol
from diffalign.data.helpers import rxn_vs_sample_plot
from diffalign.helpers import average_rxn_scores, accumulate_rxn_scores
from diffalign.data.helpers import dense_from_pyg_file_data_for_reaction_list, dense_from_pyg_file_data, add_supernodes

log = logging.getLogger(__name__)

MAX_NODES = 300
MAX_NODES_FOR_EVALUATION = 200

class DiscreteDenoisingDiffusionRxn(DiscreteDenoisingDiffusion):
    def __init__(self, cfg, dataset_infos, 
                 node_type_counts_unnormalized=None, 
                 edge_type_counts_unnormalized=None, 
                 save_as_smiles=False, use_data_parallel=False,
                 denoiser=None):
        super().__init__(cfg=cfg, dataset_infos=dataset_infos, node_type_counts_unnormalized=node_type_counts_unnormalized, 
                         edge_type_counts_unnormalized=edge_type_counts_unnormalized, use_data_parallel=use_data_parallel,
                         denoiser=denoiser)
        
    def product_smiles_to_reactant_smiles(self, product_smiles):
        """Takes in as input one product SMILES string and returns a list of predicted reactant SMILES strings."""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # add atom mapping to product_smiles
        product_mol = Chem.MolFromSmiles(product_smiles)
        [atom.SetAtomMapNum(i+1) for i, atom in enumerate(product_mol.GetAtoms())]
        product_smiles_with_am = Chem.MolToSmiles(product_mol)

        # Crate a dummy reactant SMILES string with atom mapping
        reactant_smiles_with_am = "".join([f"[C:{i+1}]" for i in range(len(product_mol.GetAtoms()))]) + "C"*(self.cfg.dataset.nb_rct_dummy_nodes) #+ ('SuNo' if self.cfg.dataset.add_supernodes else '') # dummy reactants, will be set to noise in sampling
        
        # Turn to pyg object and then to a dense Placeholder object with full adjacency matrix etc. (dense_data)
        data = graph.turn_reactants_and_product_smiles_into_graphs(self.cfg, reactant_smiles_with_am.split('.'), product_smiles_with_am.split('.'), 0) # (this part can loose the stereochemistry, will be added back in the end)
        if self.cfg.dataset.add_supernodes:
            data = add_supernodes(self.cfg, data)

        # Create a DataBatch object
        data = torch_geometric.data.Batch.from_data_list([data])
        dense_data = graph.to_dense(data=data).to_device(device)
        
        # Do sampling
        final_samples = self.sample_for_condition(dense_data=dense_data, n_samples=self.cfg.test.n_samples_per_condition, 
                                                      inpaint_node_idx=None, inpaint_edge_idx=None, device=device)
        gen_rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=final_samples, cfg=self.cfg, return_dict=False) # TODO: CHECK THAT THIS IS PROPERLY DONE, NO KEKULIZED BUGS
        unique_indices, counts, is_unique, same_reaction_groups = graph.get_unique_indices_from_reaction_list(gen_rxn_smiles)
        gen_rxn_smiles_with_atom_mapping = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=final_samples, cfg=self.cfg, with_atom_mapping=True)
        
        # Choose the right atom mapping for each unique reaction (fancy method)
        final_samples_copy = copy.deepcopy(final_samples)
        final_samples = final_samples.select_subset(is_unique).get_new_object()
        for i in range(final_samples.X.shape[0]):
            am = graph.choose_highest_probability_atom_mapping_from_placeholder(final_samples_copy.subset_by_index_list(same_reaction_groups[i]),
                                                                            [s for j,s in enumerate(gen_rxn_smiles_with_atom_mapping) if j in same_reaction_groups[i]])
            final_samples.atom_map_numbers[i] = am
        gen_rxn_smiles_with_atom_mapping = [x for x,u in zip(gen_rxn_smiles_with_atom_mapping,is_unique) if u]
            
        # Evaluate the ELBO and sort the results
        elbos, loss_t, loss_0 = self.estimate_elbo_with_repeats(final_samples_one_hot = final_samples.to_one_hot(self.cfg))
        gen_rct_smiles_with_atom_mapping, gen_prod_smiles_with_atom_mapping= graph.split_reactions_to_reactants_and_products(gen_rxn_smiles_with_atom_mapping)
        elbo_sorted_rxns = self.sort_by_elbo(elbos, loss_t, loss_0, gen_rct_smiles_with_atom_mapping, gen_prod_smiles_with_atom_mapping, is_unique=is_unique, # TODO: This results in the bug where the product is split
                                                counts=counts, bs=1, n_samples=self.cfg.test.n_samples_per_condition, k=self.cfg.test.topks)
        weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_rxns, self.cfg.test.sort_lambda_value)

        # extract the SMILES from weighted_prob_sorted_rxns. weighted_prob_sorted_rxns is a dict with the key being the product smiles, which returns a list of dicts that contain the reactant smiles and the probability in order
        sorted_reactant_smiles = [".".join(r['rcts']) for r in list(weighted_prob_sorted_rxns.values())[0]]
        probs = [r['weighted_prob'] for r in list(weighted_prob_sorted_rxns.values())[0]]
        counts = [r['count'] for r in list(weighted_prob_sorted_rxns.values())[0]]
                
        # Transfer original stereochemistry for each unique reactant set. These are extracted from weighted_prob_sorted_rxns. Original product_smiles has the stereochemistry
        sorted_reactant_smiles_with_stereo = [mol.transfer_stereo_from_product_to_reactant(reactant_smiles, product_smiles_with_am) for reactant_smiles in sorted_reactant_smiles]

        return sorted_reactant_smiles_with_stereo, probs
    
    def sample_n_conditions(self, dataloader, inpaint_node_idx, inpaint_edge_idx, device_to_use, epoch_num=None):
        assert epoch_num is not None, f'Need to provide epoch_num to use this function. Got epoch_num={epoch_num}.'
        
        batch_size = graph.get_batch_size_of_dataloader(dataloader)
        len_ = graph.get_dataset_len_of_dataloader(dataloader)

        # Note: below we might take more samples than we intended => if n_conditions%batch_size>0, take a full additional batch
        self.cfg.test.n_conditions = min(len_, self.cfg.test.n_conditions)
        num_dataloader_batches = max(int(self.cfg.test.n_conditions/batch_size+int(self.cfg.test.n_conditions%batch_size>0)), 1)
        assert num_dataloader_batches<=len(dataloader), f'Requesting {num_dataloader_batches} batches of conditions when we have a total of {len(dataloader)} batches in dataloader.'
        
        # For figuring out the batch with the largest graph size in the data
        # ... can be used to 
        # biggest_batch = graph.PlaceHolder(X=torch.zeros((1,1)),E=torch.zeros((1,1)),y=torch.zeros((1,1)),node_mask=torch.zeros((1,1)))
        # biggest_batch_dense = biggest_batch
        # for batch in dataloader:
        #     dense_data = graph.duplicate_data(data=batch, n_samples=1, get_discrete_data=False)
        #     if dense_data.X.shape[1] > biggest_batch_dense.X.shape[1]:
        #         biggest_batch_dense = dense_data
        #         biggest_batch = batch
        # log.info(f"Size of the biggest graph in the data: {biggest_batch_dense.X.shape[1]}")

        # Get the relevant data into a single batch
        data_list = [] 
        dataiter = iter(dataloader)
        for _ in range(num_dataloader_batches):
            data_ = next(dataiter) # ... can replace this with biggest_batch to check whether we can handle the largest graph
            data_list.extend(data_.to_data_list())
            
        data_list = data_list[:self.cfg.test.n_conditions]
        assert inpaint_node_idx==None or len(inpaint_node_idx)==len(data_list), f'length of inpaint_node_idx={inpaint_node_idx} and len(data_list)={len(data_list)} are not equal.'
        assert inpaint_edge_idx==None or len(inpaint_edge_idx)==len(data_list), f'length of inpaint_edge_idx={inpaint_edge_idx} and len(data_list)={len(data_list)} are not equal.'
        
        log.info(f'Scoring n_conditions={min(self.cfg.test.n_conditions, len(data_list))} with batch_size={batch_size}. Total dataloader batches required={num_dataloader_batches}.\n')
        
        # For simplicity, iterate over each element in the batch
        # self.score_batch can already handle 1 > batch sizes, so we can simply
        # block the calculations into larger blocks based on the GPU count
        # Note that this does require that test.n_conditions is divisible by num_gpus to be exactly correct
        # Otherwise may skip the last few conditions
        if device_to_use == None: 
            num_gpus = torch.cuda.device_count() if isinstance(self.model, torch.nn.DataParallel) else 1
            print(f'num_gpus {num_gpus}\n')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: # This makes it possible to manually use this function to run with only a given device
            num_gpus = 1
            device = device_to_use

        all_gen_rxn_smiles = []
        all_true_rxn_smiles = []
        all_gen_rxn_pyg = []
        all_true_rxn_pyg = []
        # num_gpus = torch.cuda.device_count() if isinstance(self.model, torch.nn.DataParallel) else 1
        n_iterations = self.cfg.test.n_conditions//num_gpus+int(self.cfg.test.n_conditions%num_gpus!=0)
        log.info(f'n_iterations for num_gpus={num_gpus}: {n_iterations}\n')
        for i in range(n_iterations):
            # The torch.tensor([i]) could be changed to torch.tensor([i,j,k,...]) for batch sizes > 1
            s = i*num_gpus
            e = min((i+1)*num_gpus, self.cfg.test.n_conditions)
            data_ = torch_geometric.data.Batch.from_data_list(data_list[s:e])
            data_ = data_.to(device)
            log.info(f"Conditions {s}-{e} out of {self.cfg.test.n_conditions}: Samples per cond: {self.cfg.test.n_samples_per_condition}.\n")
            inpaint_node_idx_ = inpaint_node_idx[s:e] if inpaint_node_idx is not None else None
            inpaint_edge_idx_ = inpaint_edge_idx[s:e] if inpaint_edge_idx is not None else None
            
            dense_data = graph.to_dense(data_).to_device(device)
            
            if self.cfg.test.inpaint_on_one_reactant and inpaint_node_idx_==None:
                # This part assumes that we only sample one reaction at a time
                if dense_data.mol_assignment[0].max() > 2:
                    reactant_to_keep = random.randint(1,dense_data.mol_assignment[0].max().item()-1)
                    reactant_indices = (dense_data.mol_assignment[0] == reactant_to_keep).nonzero()
                    reactant_indices = [idx.item() for idx in reactant_indices]
                    all_indices = list(range(dense_data.X.shape[-2]))
                    inpaint_node_idx_ = [reactant_indices]
                    # inpaint_edge_idx_ = [[(i,j) for i in reactant_indices for j in all_indices] + [(j,i) for i in reactant_indices for j in all_indices]]
                    inpaint_edge_idx_ = ["NO_ADDITIONAL_CONNECTIONS"]
                else:
                    inpaint_node_idx_ = None
                    inpaint_edge_idx_ = None

            if 'full' in self.cfg.dataset.name:
                if self.dense_data.X.shape[-2] > MAX_NODES_FOR_EVALUATION: # For now skip really large reactions to get preliminary numbers
                    continue
            
            final_samples = self.sample_for_condition(dense_data=dense_data, n_samples=self.cfg.test.n_samples_per_condition, 
                                                      inpaint_node_idx=inpaint_node_idx_, inpaint_edge_idx=inpaint_edge_idx_, device=device)
            gen_rxns = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=final_samples, cfg=self.cfg, return_dict=False)
            dense_data_dup = graph.duplicate_data(dense_data, n_samples=self.cfg.test.n_samples_per_condition, get_discrete_data=False).to_device(device)
            dense_data_dup = dense_data_dup.mask(collapse=True)
            true_rxns = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=dense_data_dup, cfg=self.cfg, return_dict=False)

            # How many conditions actually were processed in this iteration? (can be more than one)
            actual_n_conditions_in_iter = len(true_rxns)//self.cfg.test.n_samples_per_condition
            # Split the processed conditions into singles
            for k in range(actual_n_conditions_in_iter):
                all_gen_rxn_smiles.append(gen_rxns[k*self.cfg.test.n_samples_per_condition:(k+1)*self.cfg.test.n_samples_per_condition])
                all_true_rxn_smiles.append(true_rxns[k*self.cfg.test.n_samples_per_condition:(k+1)*self.cfg.test.n_samples_per_condition])
                all_true_rxn_pyg.append(dense_data_dup.subset_by_idx(k*self.cfg.test.n_samples_per_condition, (k+1)*self.cfg.test.n_samples_per_condition)
                                            .to_cpu().pyg()) # TODO: Transfer to CPU as well
                all_gen_rxn_pyg.append(final_samples.subset_by_idx(k*self.cfg.test.n_samples_per_condition, (k+1)*self.cfg.test.n_samples_per_condition)
                                            .to_cpu().pyg())

        return all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg

    def sample_n_conditions_new(self, dataloader, inpaint_node_idx, inpaint_edge_idx, device_to_use, epoch_num=None):
        # TODO: This also may require some refactoring to do still
        assert epoch_num is not None, f'Need to provide epoch_num to use this function. Got epoch_num={epoch_num}.'
        
        batch_size = graph.get_batch_size_of_dataloader(dataloader)
        len_ = graph.get_dataset_len_of_dataloader(dataloader)

        # Note: below we might take more samples than we intended => if n_conditions%batch_size>0, take a full additional batch
        self.cfg.test.n_conditions = min(len_, self.cfg.test.n_conditions)
        num_dataloader_batches = max(int(self.cfg.test.n_conditions/batch_size+int(self.cfg.test.n_conditions%batch_size>0)), 1)
        assert num_dataloader_batches<=len(dataloader), f'Requesting {num_dataloader_batches} batches of conditions when we have a total of {len(dataloader)} batches in dataloader.'
        
        # For figuring out the batch with the largest graph size in the data
        # ... can be used to 
        # biggest_batch = graph.PlaceHolder(X=torch.zeros((1,1)),E=torch.zeros((1,1)),y=torch.zeros((1,1)),node_mask=torch.zeros((1,1)))
        # biggest_batch_dense = biggest_batch
        # for batch in dataloader:
        #     dense_data = graph.duplicate_data(data=batch, n_samples=1, get_discrete_data=False)
        #     if dense_data.X.shape[1] > biggest_batch_dense.X.shape[1]:
        #         biggest_batch_dense = dense_data
        #         biggest_batch = batch
        # log.info(f"Size of the biggest graph in the data: {biggest_batch_dense.X.shape[1]}")

        # Get the relevant data into a single batch
        data_list = [] 
        dataiter = iter(dataloader)
        for _ in range(num_dataloader_batches):
            data_ = next(dataiter) # ... can replace this with biggest_batch to check whether we can handle the largest graph
            data_list.extend(data_.to_data_list())
            
        data_list = data_list[:self.cfg.test.n_conditions]
        assert inpaint_node_idx==None or len(inpaint_node_idx)==len(data_list), f'length of inpaint_node_idx={inpaint_node_idx} and len(data_list)={len(data_list)} are not equal.'
        assert inpaint_edge_idx==None or len(inpaint_edge_idx)==len(data_list), f'length of inpaint_edge_idx={inpaint_edge_idx} and len(data_list)={len(data_list)} are not equal.'
        
        log.info(f'Scoring n_conditions={min(self.cfg.test.n_conditions, len(data_list))} with batch_size={batch_size}. Total dataloader batches required={num_dataloader_batches}.\n')
        
        # For simplicity, iterate over each element in the batch
        # self.score_batch can already handle 1 > batch sizes, so we can simply
        # block the calculations into larger blocks based on the GPU count
        # Note that this does require that test.n_conditions is divisible by num_gpus to be exactly correct
        # Otherwise may skip the last few conditions
        if device_to_use == None: 
            num_gpus = torch.cuda.device_count() if isinstance(self.model, torch.nn.DataParallel) else 1
            #print(f'num_gpus {num_gpus}\n')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: # This makes it possible to manually use this function to run with only a given device
            num_gpus = 1
            device = device_to_use

        all_gen_rxn_smiles = []
        all_true_rxn_smiles = []
        all_gen_rxn_pyg = []
        all_true_rxn_pyg = []
        # num_gpus = torch.cuda.device_count() if isinstance(self.model, torch.nn.DataParallel) else 1  
        n_iterations = self.cfg.test.n_conditions//num_gpus+int(self.cfg.test.n_conditions%num_gpus!=0)
        log.info(f'n_iterations for num_gpus={num_gpus}: {n_iterations}\n')
        for i in range(n_iterations):
            # The torch.tensor([i]) could be changed to torch.tensor([i,j,k,...]) for batch sizes > 1
            s = i*num_gpus
            e = min((i+1)*num_gpus, self.cfg.test.n_conditions)
            data_ = torch_geometric.data.Batch.from_data_list(data_list[s:e])
            data_ = data_.to(device)
            log.info(f"Conditions {s}-{e} out of {self.cfg.test.n_conditions}: Samples per cond: {self.cfg.test.n_samples_per_condition}.\n")
            inpaint_node_idx_ = inpaint_node_idx[s:e] if inpaint_node_idx is not None else None
            inpaint_edge_idx_ = inpaint_edge_idx[s:e] if inpaint_edge_idx is not None else None
            
            dense_data = graph.to_dense(data_).to_device(device)
            if self.cfg.test.inpaint_on_one_reactant and inpaint_node_idx_==None:
                # This part assumes that we only sample one reaction at a time
                if dense_data.mol_assignment[0].max() > 2:
                    reactant_to_keep = random.randint(1,dense_data.mol_assignment[0].max().item()-1)
                    reactant_indices = (dense_data.mol_assignment[0] == reactant_to_keep).nonzero()
                    reactant_indices = [idx.item() for idx in reactant_indices]
                    all_indices = list(range(dense_data.X.shape[-2]))
                    inpaint_node_idx_ = [reactant_indices]
                    # inpaint_edge_idx_ = [[(i,j) for i in reactant_indices for j in all_indices] + [(j,i) for i in reactant_indices for j in all_indices]]
                    inpaint_edge_idx_ = ["NO_ADDITIONAL_CONNECTIONS"]
                else:
                    inpaint_node_idx_ = None
                    inpaint_edge_idx_ = None

            final_samples = self.sample_for_condition(dense_data=dense_data, n_samples=self.cfg.test.n_samples_per_condition, 
                                                      inpaint_node_idx=inpaint_node_idx_, inpaint_edge_idx=inpaint_edge_idx_, device=device)
            #print(f'final_sampels {final_samples.X.shape}\n')
            gen_rxns = mol.get_cano_smiles_from_dense(X=final_samples.X, E=final_samples.E, mol_assignment=final_samples.mol_assignment, atom_types=self.dataset_info.atom_decoder,
                                                      bond_types=self.dataset_info.bond_decoder, return_dict=False)
            dense_data_dup = graph.duplicate_data(dense_data, n_samples=self.cfg.test.n_samples_per_condition)
            dense_data_dup = dense_data_dup.mask(dense_data_dup.node_mask, collapse=True)

            true_rxns = mol.get_cano_smiles_from_dense(X=dense_data_dup.X, E=dense_data_dup.E, mol_assignment=dense_data_dup.mol_assignment, atom_types=self.dataset_info.atom_decoder, 
                                                       bond_types=self.dataset_info.bond_decoder, return_dict=False)
            #print(f'true_rxns {true_rxns}\n')
            # How many conditions actually were processed in this iteration? (can be more than one)
            actual_n_conditions_in_iter = len(true_rxns)//self.cfg.test.n_samples_per_condition
            #print(f'actual_n_conditions_in_iter {actual_n_conditions_in_iter}\n')
            # Split the processed conditions into singles
            for k in range(actual_n_conditions_in_iter):
                all_gen_rxn_smiles.append(gen_rxns[k*self.cfg.test.n_samples_per_condition:(k+1)*self.cfg.test.n_samples_per_condition])
                all_true_rxn_smiles.append(true_rxns[k*self.cfg.test.n_samples_per_condition:(k+1)*self.cfg.test.n_samples_per_condition])
                all_true_rxn_pyg.append(dense_data_dup.subset_by_idx(k*self.cfg.test.n_samples_per_condition, (k+1)*self.cfg.test.n_samples_per_condition)
                                            .to_cpu().pyg()) # TODO: Transfer to CPU as well
                all_gen_rxn_pyg.append(final_samples.subset_by_idx(k*self.cfg.test.n_samples_per_condition, (k+1)*self.cfg.test.n_samples_per_condition)
                                            .to_cpu().pyg())

        return all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg

    def sample_for_condition(self, dense_data, n_samples, inpaint_node_idx, inpaint_edge_idx, device=None, return_chains=False):
        """
            Inputs:
            condition_idx: The index of the first conditioning product in the batch. Used for plotting.
        """
        # TODO: Division between eval_one_batch and score_one_batch could be a bit clearer
        # repeat the same data for n_samples times (to be able to generate n_samples per conditional object)
        bs = dense_data.X.shape[0]
        device = dense_data.X.device
        
        t0 = time.time()
        log.info(f'About to sample. Size of reaction: {dense_data.X.shape[1]}')
        num_nodes = dense_data.X.shape[1]
        # num_repetitions, num_batches = self.num_repetitions_func(n_samples, num_nodes, 
        #                                                          self.cfg.test.num_samples_per_condition_subrepetition_ranges, 
        #                                                          self.cfg.test.num_samples_per_condition_subrepetitions)
        
        # assert bs%100==0, f'Batch size must be divisible by 100. Got {bs}.'
        # num_repetitions = min(int(bs / 100), bs)
        # max batch size is 100, and usually we use a default batch size of 100 unless the total number of conditions is less
        max_batch_size = 100
        num_repetitions = min(max_batch_size, n_samples)
        num_batches = max(n_samples // max_batch_size + (n_samples%max_batch_size!=0), 1)
        
        # Split the repetition over num_samples per condition to subrepetitions, and aggregate results at the end (in case of memory limitations)
        final_samples_ = []
        actual_sample_chains_ = []
        prob_s_chains_ = []
        pred_0_chains_ = []
        true_rxns_ = []
        for i in range(num_batches):
            n_samples_in_sampling_iteration = min(num_repetitions, n_samples-i*num_repetitions)
            dense_data_dup = graph.duplicate_data(dense_data, n_samples=n_samples_in_sampling_iteration, get_discrete_data=False).to_device(device)
            # dense_data_dup = graph.duplicate_data(dense_data=dense_data, n_samples=n_samples, get_discrete_data=False).to_device(device)
            # duplicate the node/edge inpainting idx
            inpaint_node_idx_ = inpaint_node_idx * n_samples_in_sampling_iteration if inpaint_node_idx is not None else None
            inpaint_edge_idx_ = inpaint_edge_idx * n_samples_in_sampling_iteration if inpaint_edge_idx is not None else None

            #print(f'\t Starting sampling iteration {i} with X.shape: {dense_data_dup.X.shape}, E.shape: {dense_data_dup.E.shape}, y.shape: {dense_data_dup.y.shape}')

            if num_nodes <= 200:
                print(f'device passed to sample_one_batch: {device}')
                final_samples, actual_sample_chains, prob_s_chains, pred_0_chains, true_rxns = self.sample_one_batch(data=dense_data_dup, inpaint_node_idx=inpaint_node_idx_, 
                                                                                                                    inpaint_edge_idx=inpaint_edge_idx_, get_true_rxns=True, 
                                                                                                                    get_chains=True, device=device)
            else: # don't try these (uspto-full), more memory efficient for now
                print('Skipping sampling for large reaction')
                final_samples = copy.deepcopy(dense_data_dup.get_new_object())#.mask(dense_data_dup.node_mask, collapse=True)
                final_samples = final_samples.mask(final_samples.node_mask, collapse=True)
                n = 4
                final_samples.X = final_samples.X[:,:n]
                final_samples.E = final_samples.E[:,:n,:n]
                final_samples.node_mask = final_samples.node_mask[:,:n]
                final_samples.y = torch.zeros(final_samples.X.shape[0], 1)
                final_samples.atom_charges = final_samples.atom_charges[:, :n]
                final_samples.atom_chiral = final_samples.atom_chiral[:, :n]
                final_samples.bond_dirs = final_samples.bond_dirs[:, :n, :n]
                final_samples.node_mask = final_samples.node_mask[:, :n]
                final_samples.atom_map_numbers = torch.tensor([1,2,1,2], device=final_samples.X.device)[None,:].repeat(final_samples.X.shape[0],1)
                final_samples.mol_assignment = torch.tensor([0,0,1,1], device=final_samples.X.device)[None,:].repeat(final_samples.X.shape[0],1)
                final_samples.E[:] = 0 # no edges
                final_samples.bond_dirs[0] = 0
                dense_data_dup = copy.deepcopy(final_samples)
                dense_data_dup.X[:,0:2] = self.cfg.dataset.atom_types.index('U')
                true_rxns = dense_data_dup.get_new_object()

            actual_sample_chains, prob_s_chains, pred_0_chains = [(1,final_samples)],[(1,final_samples)],[(1,final_samples)]

            # ad-hoc fix for now for the case where the sample object size was cut to smaller in sample_one_batch (due to not enough dummy nodes for the reaction)
            # Don't extend to other places please! This is not good code but doing it better requires refactoring
            if final_samples.X.shape[1] != dense_data.X.shape[1]:
                dense_data.drop_n_first_nodes(dense_data.X.shape[1] - final_samples.X.shape[1])

            # final_samples.smiles = dense_data.smiles
            final_samples_.append(final_samples)
            actual_sample_chains_.append(actual_sample_chains)
            prob_s_chains_.append(prob_s_chains)
            pred_0_chains_.append(pred_0_chains)
            true_rxns_.append(true_rxns)
        
        final_samples = graph.concatenate_placeholders(final_samples_)
        true_rxns_ = graph.concatenate_placeholders(true_rxns_)
        actual_sample_chains = self.concatenate_sample_chains(actual_sample_chains_)
        prob_s_chains = self.concatenate_sample_chains(prob_s_chains_)
        pred_0_chains = self.concatenate_sample_chains(pred_0_chains_)

        log.info(f"Sampling time: {time.time()-t0}")
        # plot chains when sampling?
        
        if return_chains:
            return final_samples, actual_sample_chains, prob_s_chains, pred_0_chains, true_rxns
        
        return final_samples
    
    def evaluate_from_artifact(self, reaction_data, actual_n_conditions, epoch=None, device=None, condition_range=None):
        '''
            Evaluate samples read from a text file.
            
            final_samples: final_samples.X.shape=(bs, n_samples, n)
            dense_data: dense_data.X.shape=(bs, n_samples, n)
        '''        
        # 1. get dense_data to be of the right shape
        t0 = time.time()
        assert len(reaction_data['gen']) == len(reaction_data['true'])

        # try:
        #     from mpi4py import MPI
        #     comm = MPI.COMM_WORLD
        #     mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        #     mpi_rank = comm.Get_rank() # this will be 0
        #     log.warning(f" mpi_rank {mpi_rank}")
        # except:
        #     MPI = None
        #     log.warning("mpi4py not found. MPI will not be used.")
        # try:
        true_graph_data, sample_graph_data = dense_from_pyg_file_data_for_reaction_list(self.cfg, reaction_data)
        # except:
        #     log.warning("problem!")
        #     log.warning(f"Error happened at mpi_rank {mpi_rank}.")

        true_graph_data = true_graph_data.mask(collapse=True)
        sample_graph_data = sample_graph_data.mask(collapse=True)
        true_graph_data.reshape_bs_n_samples(bs=actual_n_conditions, n_samples=self.cfg.test.n_samples_per_condition, n=true_graph_data.X.shape[1])
        sample_graph_data.reshape_bs_n_samples(bs=actual_n_conditions, n_samples=self.cfg.test.n_samples_per_condition, n=sample_graph_data.X.shape[1])
        sample_graph_data = sample_graph_data.to_device(device)
        true_graph_data = true_graph_data.to_device(device)
        dense_data = true_graph_data
        final_samples = sample_graph_data
        bs, n_samples, n = dense_data.X.shape[0], dense_data.X.shape[1], dense_data.X.shape[2]
        num_gpus = 1

        # flatten bs and n_samples
        dense_data = dense_data.flatten(start_dim=0, end_dim=1)
        final_samples = final_samples.flatten(start_dim=0, end_dim=1)

        # 2. score the set of samples generated for each condition
        t0 = time.time()
        scores = []
        
        start = 0
        end = bs//num_gpus
        step = 1
        
        all_elbo_sorted_reactions = []
        all_weighted_prob_sorted_rxns = []
        placeholders_for_print = []
        for i in range(start, end, step):
            # if i*num_gpus<46:
            #     log.info(f'skipping condition {i*num_gpus}')
            #     continue
            log.info(f'First condition {i*num_gpus}')
            dense_data_ = dense_data.subset_by_idx(start_idx=n_samples*i*num_gpus, end_idx=n_samples*i*num_gpus+num_gpus*n_samples)
            log.info(f'len dense_data_ smiles: {len(dense_data_.smiles)}')
            final_samples_ = final_samples.subset_by_idx(start_idx=n_samples*i*num_gpus, end_idx=n_samples*i*num_gpus+num_gpus*n_samples)
            scores_, elbo_sorted_reactions, weighted_prob_sorted_rxns = self.score_one_batch(final_samples=final_samples_, true_data=dense_data_, 
                                                                                             bs=num_gpus, n_samples=n_samples, n=n, device=device, idx=i)
            
            if 'full' in self.cfg.dataset.name:
                if self.dense_data_.X.shape[-2] > MAX_NODES_FOR_EVALUATION: # For now skip really large reactions to get preliminary numbers
                    continue

            for key in scores_.keys(): # Make sure that no sneak in. This may be a bit paranoid, but trying to evade memory leaks
                if type(scores_[key]) == torch.Tensor:
                    scores_[key] = scores_[key].item()

            # take the true first data point from the batch for saving. Data is of shape (bs=n_samples, n, dx)
            original_placeholder_for_print = dense_data_.select_by_batch_idx(0).to_cpu()
            all_elbo_sorted_reactions.append(elbo_sorted_reactions)
            all_weighted_prob_sorted_rxns.append(weighted_prob_sorted_rxns)
            placeholders_for_print.append(original_placeholder_for_print)
            # self.save_reactions_to_text(original_placeholder_for_print, elbo_sorted_reactions, weighted_prob_sorted_rxns, epoch=epoch, condition_idx=i*num_gpus, start=condition_range[0])
            log.info(f"scores for condition {i*num_gpus}-{(i+1)*num_gpus}: {scores_}\n")
            #scores = helpers.accumulate_rxn_scores(acc_scores=scores, new_scores=scores_, total_iterations=(condition_range[1]-condition_range[0])//num_gpus)
            scores.append(scores_)
            
        log.info(f"Scoring time: {time.time()-t0}")
        
        return scores, all_elbo_sorted_reactions, all_weighted_prob_sorted_rxns, placeholders_for_print

    def evaluate_from_artifact_new(self, reaction_data, epoch=None, device=None, condition_range=None):
        '''
            TODO: Work in progress
            Evaluate samples read from a text file.
            
            Arguments:
            reaction_data: Format: dict {'gen':gen_rxns, 'true':true_rxns} where gen_rxns and true_rxns are lists of DataBatches that contain n_samples_per_condition each
        '''
            
        assert len(reaction_data['gen']) == len(reaction_data['true'])
        # 2. score the set of samples generated for each condition
        t0 = time.time()
        scores = []
        start = 0
        end = len(reaction_data['gen'])#//num_gpus
        
        all_elbo_sorted_reactions = []
        all_weighted_prob_sorted_rxns = []
        placeholders_for_print = []
        for i in range(len(reaction_data['gen'])):
            try:
                dense_data = dense_from_pyg_file_data(self.cfg, reaction_data['true'][i])
                final_samples = dense_from_pyg_file_data(self.cfg, reaction_data['gen'][i])
                dense_data = dense_data.mask(collapse=True).to_device(device)
                final_samples = final_samples.mask(collapse=True).to_device(device)

                # dense_data_ = dense_data.subset_by_idx(start_idx=n_samples*i*num_gpus, end_idx=n_samples*i*num_gpus+num_gpus*n_samples)
                # final_samples_ = final_samples.subset_by_idx(start_idx=n_samples*i*num_gpus, end_idx=n_samples*i*num_gpus+num_gpus*n_samples)
                
                # TODO: Change this part again to handle batches!
                num_nodes = dense_data.X.shape[1]
                num_samples_per_condition = final_samples.X.shape[0]
                num_repetitions, num_batches = self.num_repetitions_func(num_samples_per_condition, num_nodes, self.cfg.test.num_samples_per_condition_subrepetition_ranges, self.cfg.test.num_samples_per_condition_subrepetitions)
                log.info(f"Num nodes in molecule: {num_nodes}, num_repetitions: {num_repetitions}, num_batches: {num_batches}.\n")
                # Split the repetition over num_samples per condition to subrepetitions, and aggregate results at the end (in case of memory limitations)
                scores_all = []
                elbo_sorted_reactions_all = []
                weighted_prob_sorted_rxns_all = []
                counts_of_samples = []
                for j in range(num_batches):
                    n_samples_in_sampling_iteration = min(num_repetitions, num_samples_per_condition - j*num_repetitions)
                    start_idx = num_repetitions*j
                    end_idx = num_repetitions*j + n_samples_in_sampling_iteration
                    final_samples_subset = final_samples.subset_by_idx(start_idx=start_idx, end_idx=end_idx)
                    dense_data_subset = dense_data.subset_by_idx(start_idx=start_idx, end_idx=end_idx)
                    scores_, elbo_sorted_reactions, weighted_prob_sorted_rxns = self.score_one_batch(final_samples=final_samples_subset, true_data=dense_data_subset, 
                                                                                                    bs=1, n_samples=n_samples_in_sampling_iteration, n=num_nodes, device=device, idx=i)
                    scores_all.append(scores_)
                    elbo_sorted_reactions_all.append(elbo_sorted_reactions)
                    weighted_prob_sorted_rxns_all.append(weighted_prob_sorted_rxns)
                    counts_of_samples.append(n_samples_in_sampling_iteration)

                scores_ = average_rxn_scores(scores_all, counts_of_samples)
                elbo_sorted_reactions = self.concatenate_sorted_reactions(elbo_sorted_reactions_all, sorting_key='elbo', ascending=True)
                # TODO: This doesn't yet work -> fix locally
                weighted_prob_sorted_rxns = self.concatenate_sorted_reactions(weighted_prob_sorted_rxns_all, sorting_key='weighted_prob', ascending=False)
                
                for key in scores_.keys(): # Make sure that no sneak in. This may be a bit paranoid, but trying to evade memory leaks
                    if type(scores_[key]) == torch.Tensor:
                        scores_[key] = scores_[key].item()

                # log.info(scores_)
                # log.info(elbo_sorted_reactions)
                # log.info(weighted_prob_sorted_rxns)

                # take the true first data point from the batch for saving. Data is of shape (bs=n_samples, n, dx)
                original_placeholder_for_print = dense_data.select_by_batch_idx(0).to_cpu()
                all_elbo_sorted_reactions.append(elbo_sorted_reactions)
                all_weighted_prob_sorted_rxns.append(weighted_prob_sorted_rxns)
                placeholders_for_print.append(original_placeholder_for_print)
                # self.save_reactions_to_text(original_placeholder_for_print, elbo_sorted_reactions, weighted_prob_sorted_rxns, epoch=epoch, condition_idx=i*num_gpus, start=condition_range[0])
                log.info(f"scores for condition {i}-{(i+1)}: {scores_}\n")
                #scores = helpers.accumulate_rxn_scores(acc_scores=scores, new_scores=scores_, total_iterations=(condition_range[1]-condition_range[0])//num_gpus)
                scores.append(scores_)
            except Exception as err:
                log.info(f"Couldn't evaluate for sample {i}. Error {err}")
            
        log.info(f"Scoring time: {time.time()-t0}")
        
        return scores, all_elbo_sorted_reactions, all_weighted_prob_sorted_rxns, placeholders_for_print
         
    @torch.no_grad()  
    def evaluate(self, epoch, datamodule, device, inpaint_node_idx=None, inpaint_edge_idx=None):
        log.info(f"Evaluating for epoch {epoch}...\n") # This is now used also for the product-conditional sampling
        if self.cfg.diffusion.edge_conditional_set=='test':
            additional_dataloader = datamodule.test_dataloader()
        elif self.cfg.diffusion.edge_conditional_set=='val': 
            additional_dataloader = datamodule.val_dataloader()
        elif self.cfg.diffusion.edge_conditional_set=='train':    
            additional_dataloader = datamodule.train_dataloader()
        else:
            assert 'edge_conditional_set not recognized.'

        eval_start_time = time.time()
        elbo_of_data_time = time.time()
        ## TODO: UNCOMMENT THIS BACK
        if self.cfg.test.eval_elbo_during_training:
            log.info("calculating ELBO...")
            # TODO: Change this to a similar estimate that we would have during training
            test_elbo = self.get_elbo_of_data(datamodule.test_dataloader(), 
                                              n_samples=self.cfg.test.elbo_samples, 
                                              device=device)
            train_elbo = self.get_elbo_of_data(datamodule.train_dataloader(), 
                                               n_samples=self.cfg.test.elbo_samples, 
                                               device=device)
            log.info(f"ELBO train: {train_elbo}, ELBO test: {test_elbo}. Time taken: {time.time()-elbo_of_data_time}")
        else:
            test_elbo = 999
            train_elbo = 999

        scores = self.eval_n_conditions(dataloader=additional_dataloader, inpaint_node_idx=inpaint_node_idx, 
                                        inpaint_edge_idx=inpaint_edge_idx, epoch=epoch, device_to_use=device)
        scores['train_elbo'] = train_elbo
        scores['test_elbo'] = test_elbo
        
        log.info(f"Total evaluation time: {time.time()-eval_start_time}")
        
        return dict(scores)
    
    @torch.no_grad()
    def eval_full_dataset(self, dataloader, epoch=0, inpaint_node_idx=None, inpaint_edge_idx=None, device=None): 
        assert 'FUNCTION NOT TESTED!' 
        
        scores = defaultdict(lambda: 0) 
        for i, data in enumerate(dataloader):
            data = data.to(device)
            if 'cannot_generate' in data.keys and data.cannot_generate:
                scores_ = {'all_valid': 0., 'atleastone_valid': 0., 'all_coverage': 0., 'atleastone_coverage': 0.,
                           'atleastone_matching_atoms': 0., 'atleastone_matching_smiles':0., 'matching_atoms': 0., 
                           'matching_smiles':0., 'all_valid_unfiltered': 0., 'atleastone_valid_unfiltered': 0.}
                for k_ in self.cfg.test.topks:
                    scores_[f'top-{k_}'] = 0.
            else:
                dense_data = graph.to_dense_batch(data).to_device(device)
                scores_ = self.eval_one_batch(dense_data, n_samples=self.cfg.test.n_samples_per_condition, epoch=epoch, 
                                              inpaint_node_idx=inpaint_node_idx, inpaint_edge_idx=inpaint_edge_idx)
                scores = accumulate_rxn_scores(acc_scores=scores, new_scores=scores_, total_iterations=len(dataloader))
                
        return scores
    
    @torch.no_grad()
    def eval_n_conditions(self, dataloader, inpaint_node_idx=None, inpaint_edge_idx=None, epoch=0, device_to_use=None):
        batch_size = graph.get_batch_size_of_dataloader(dataloader)
        # Note: below we might take more samples than we intended => if n_conditions%batch_size>0, take a full additional batch
        num_dataloader_batches = max(int(self.cfg.test.n_conditions/batch_size+int(self.cfg.test.n_conditions%batch_size>0)),1)
        assert num_dataloader_batches<=len(dataloader), f'Requesting {num_dataloader_batches} batches of conditions when we have a total of {len(dataloader)} batches in dataloader.'
        
        # For figuring out the batch with the largest graph size in the data
        # ... can be used to 
        # biggest_batch = graph.PlaceHolder(X=torch.zeros((1,1)),E=torch.zeros((1,1)),y=torch.zeros((1,1)),node_mask=torch.zeros((1,1)))
        # biggest_batch_dense = biggest_batch
        # for batch in dataloader:
        #     dense_data = graph.duplicate_data(data=batch, n_samples=1, get_discrete_data=False)
        #     if dense_data.X.shape[1] > biggest_batch_dense.X.shape[1]:
        #         biggest_batch_dense = dense_data
        #         biggest_batch = batch
        # log.info(f"Size of the biggest graph in the data: {biggest_batch_dense.X.shape[1]}")

        # Get the relevant data into a single batch
        data_list = [] 
        dataiter = iter(dataloader)
        for _ in range(num_dataloader_batches):
            data_ = next(dataiter) # ... can replace this with biggest_batch to check whether we can handle the largest graph
            data_list.extend(data_.to_data_list())
            
        data_list = data_list[:self.cfg.test.n_conditions]
        assert inpaint_node_idx==None or len(inpaint_node_idx)==len(data_list), f'length of inpaint_node_idx={inpaint_node_idx} and len(data_list)={len(data_list)} are not equal.'
        assert inpaint_edge_idx==None or len(inpaint_edge_idx)==len(data_list), f'length of inpaint_edge_idx={inpaint_edge_idx} and len(data_list)={len(data_list)} are not equal.'
        
        log.info(f'Scoring n_conditions={min(self.cfg.test.n_conditions, len(data_list))} with batch_size={batch_size}. Total dataloader batches required={num_dataloader_batches}.\n')
        
        # For simplicity, iterate over each element in the batch
        # self.score_batch can already handle 1 > batch sizes, so we can simply
        # block the calculations into larger blocks based on the GPU count
        # Note that this does require that test.n_conditions is divisible by num_gpus to be exactly correct
        # Otherwise may skip the last few conditions
        # num_gpus = torch.cuda.device_count() if isinstance(self.model, torch.nn.DataParallel) else 1
        
        if device_to_use == None: 
            num_gpus = torch.cuda.device_count() if isinstance(self.model, torch.nn.DataParallel) else 1
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else: # This makes it possible to manually use this function to run with only a given device
            num_gpus = 1
            device = device_to_use

        log.info(f"We are using device: {device} with num_gpus: {num_gpus}. n_conditions: {self.cfg.test.n_conditions}")

        scores = defaultdict(lambda: 0) 
        for i in range(self.cfg.test.n_conditions//num_gpus):
            # The torch.tensor([i]) could be changed to torch.tensor([i,j,k,...]) for batch sizes > 1
            data_ = torch_geometric.data.Batch.from_data_list(data_list[(i*num_gpus):(i+1)*num_gpus])
            log.info(f"Device: {device}. conditions {i*num_gpus}-{(i+1)*num_gpus} out of {self.cfg.test.n_conditions}: Samples per cond: {self.cfg.test.n_samples_per_condition}.\n")
            inpaint_node_idx_ = inpaint_node_idx[(i*num_gpus):(i+1)*num_gpus] if inpaint_node_idx is not None else None
            inpaint_edge_idx_ = inpaint_edge_idx[(i*num_gpus):(i+1)*num_gpus] if inpaint_edge_idx is not None else None
            
            dense_data = graph.to_dense(data_).to_device(device)

            log.info("Device in use: {}".format(dense_data.X.device))
            scores_, elbo_sorted_reactions, weighted_prob_sorted_rxns = self.eval_one_batch(dense_data, self.cfg.test.n_samples_per_condition, device, condition_idx=i*num_gpus, epoch=epoch,
                                                                                            inpaint_node_idx=inpaint_node_idx_, inpaint_edge_idx=inpaint_edge_idx_)
            dense_data_discrete = dense_data.mask(dense_data.node_mask, collapse=True)

            self.save_reactions_to_text(dense_data_discrete, elbo_sorted_reactions, weighted_prob_sorted_rxns, epoch=epoch, condition_idx=i*num_gpus, start=i*num_gpus)

            for k_ in scores_.keys():
                if type(scores[k_])==torch.tensor:
                    scores_[k_] = scores_[k_].cpu().flatten() # Should be of shape (bs,) at this point, or (just 0-dim)
                    if 'cannot_generate' in data_.keys:
                        scores_[k_] *= ~data_.cannot_generate.cpu() #
            log.info(f"scores for condition {i*num_gpus}-{(i+1)*num_gpus}: {scores_}\n")
            scores = accumulate_rxn_scores(acc_scores=scores, new_scores=scores_, total_iterations=self.cfg.test.n_conditions//num_gpus)

        return scores

    def save_reactions_to_text(self, original_data_placeholder, elbo_sorted_reactions, weighted_prob_sorted_rxns, epoch, condition_idx, start=0):
        t0 = time.time()

        true_rxns = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=original_data_placeholder, cfg=self.cfg)
        
        graph.save_samples_to_file_without_weighted_prob(f'eval_epoch{epoch}_s{start}.txt', condition_idx, elbo_sorted_reactions, true_rxns)
        graph.save_samples_to_file(f'eval_epoch{epoch}_resorted_{self.cfg.test.sort_lambda_value}_s{start}.txt', condition_idx, weighted_prob_sorted_rxns, true_rxns)
        
        log.info(f"Saving samples to file time: {time.time()-t0}")

    @torch.no_grad()
    def eval_one_batch(self, dense_data, n_samples, device, epoch=None, condition_idx=None, inpaint_node_idx=None, inpaint_edge_idx=None):
        """
        inputs:
        condition_idx: The index of the first conditioning product in the batch. Used for plotting.
        """
        # 
        # TODO: Division between eval_one_batch and score_one_batch could be a bit clearer
        # repeat the same data for n_samples times (to be able to generate n_samples per conditional object)
        #bs = data.batch.max().item()+1
        assert dense_data.smiles is not None, "The dense_data object should have the ground-truth SMILES"
        bs = dense_data.X.shape[0]
        device = dense_data.X.device
        log.info("Device really in use: {}".format(dense_data.X.device))

        num_nodes = dense_data.X.shape[1]
        # The name should be something like: num_repetitions_per_condition_subrepetitions and num_repetitions_per_condition_subrepetition_ranges
        num_repetitions, num_batches = self.num_repetitions_func(n_samples, num_nodes, self.cfg.test.num_samples_per_condition_subrepetition_ranges, self.cfg.test.num_samples_per_condition_subrepetitions)

        max_batch_size = 5
        num_repetitions = min(max_batch_size, n_samples)
        num_batches = max(n_samples // max_batch_size + (n_samples%max_batch_size!=0), 1)

        # Split the repetition over num_samples per condition to subrepetitions, and aggregate results at the end (in case of memory limitations)
        actual_sample_chains_ = []
        elbo_sorted_reactions_ = []
        weighted_prob_sorted_rxns_ = []
        true_rxns_ = []
        for i in range(num_batches):
            n_samples_in_sampling_iteration = min(num_repetitions, n_samples - i*num_repetitions)

            dense_data_duplicated = graph.duplicate_data(dense_data, n_samples=n_samples_in_sampling_iteration, get_discrete_data=False)
            # duplicate the node/edge inpainting dx
            inpaint_node_idx_ = [item for item in inpaint_node_idx for ns in range(n_samples_in_sampling_iteration)] if inpaint_node_idx is not None else None
            inpaint_edge_idx_ = [item for item in inpaint_edge_idx for ns in range(n_samples_in_sampling_iteration)] if inpaint_edge_idx is not None else None

            t0 = time.time()
            log.info(f'About to sample {i}')
            # True reactions are here only for plotting purposes
            final_samples, actual_sample_chains, prob_s_chains, pred_0_chains, true_rxns = self.sample_one_batch(data=dense_data_duplicated, device=device, inpaint_node_idx=inpaint_node_idx_, 
                                                                                                                inpaint_edge_idx=inpaint_edge_idx_, get_true_rxns=True, 
                                                                                                                get_chains=True)

            # Final samples is a PlaceHolder object. actual_sample_chains is a list of PlaceHolder objects. true_rxns is a PlaceHolder object.
            # final_samples.X is of shape (bs*n_samples, max_nodes, x_features). 
            log.info(f"Sampling time: {time.time()-t0}")
            
            t0 = time.time()
            # dense_data = dense_data.mask(dense_data.node_mask, collapse=True)
            scores, elbo_sorted_reactions, weighted_prob_sorted_rxns = self.score_one_batch(final_samples=final_samples, true_data=dense_data_duplicated.mask(dense_data_duplicated.node_mask, collapse=True), 
                                                                                            bs=bs, n_samples=n_samples_in_sampling_iteration, n=dense_data_duplicated.X.shape[1], device=device, idx=i)
            log.info(f"Scoring time: {time.time()-t0}")

            actual_sample_chains_.append(actual_sample_chains)
            elbo_sorted_reactions_.append(elbo_sorted_reactions)
            weighted_prob_sorted_rxns_.append(weighted_prob_sorted_rxns)
            true_rxns_.append(true_rxns)

        # Concatenate the results together (also resort...)
        # TODO: Does this resorting actually combine the results correctly? Like add the counts etc.?
        actual_sample_chains = self.concatenate_sample_chains(actual_sample_chains_)
        elbo_sorted_reactions = self.concatenate_sorted_reactions(elbo_sorted_reactions_, sorting_key='elbo', ascending=True)
        weighted_prob_sorted_rxns = self.concatenate_sorted_reactions(weighted_prob_sorted_rxns_, sorting_key='weighted_prob', ascending=False)
        # weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_reactions, self.cfg.test.sort_lambda_value)
        true_rxns = graph.concatenate_placeholders(true_rxns_)

        # iterate over the batch size and plot the sample with the lowest elbo for chains_to_save chains samples.
        if self.cfg.test.plot_rxn_chains:
            t0 = time.time()
            rxn_plots = [] # Default value if no chains are saved for some reason
            for i, prod in enumerate(elbo_sorted_reactions.keys()): 
                if i+condition_idx+1<=self.cfg.test.chains_to_save: # handles the case of multiple conditions plotted at once
                    
                    '''
                        (true_rxns, sampled_rxns, cfg, chain_name='default', rxn_offset_nb=0)
                    '''
                    rxn_plots.extend(self.plot_diagnostics(true_rxns=true_rxns.select_by_batch_and_sample_idx(bs, n_samples, i, 0), 
                                                              sample_chains=graph.select_placeholder_from_chain_by_batch_and_sample(chains=actual_sample_chains, bs=bs, n_samples=n_samples,
                                                                                                                                 batch_idx=i, sample_idx=elbo_sorted_reactions[prod][0]['sample_idx']),
                                                              epoch=epoch, 
                                                              rxn_offset_nb=condition_idx+i))
            scores['rxn_plots'] = rxn_plots
            log.info(f"Plotting time: {time.time()-t0}")
        
        return scores, elbo_sorted_reactions, weighted_prob_sorted_rxns
    
    def num_repetitions_func(self, num_samples_per_condition, num_nodes, num_node_ranges_list, repetition_list):
        """
        Split the repetition over num_samples_per_condition to subrepetitions baed on the size of the graph (num_nodes) so that memory doesn't run out (especially on USPTO-Full)
        Args:
        num_samples_per_condition: the total number of samples we want in the end
        num_nodes: The number of nodes
        num_node_ranges_list: The cut-offs that specify the repetitions based on how many nodes we have in the entire graph
        repetition_list: How many times to repeat the sample in one go
        Outputs:
        The number of repetitions per batch and the number of batches necessary
        """
        assert len(num_node_ranges_list)+1 == len(repetition_list)
        num_node_ranges_list = num_node_ranges_list # augment with zero for easier handling
        num_node_range_idx = 0
        # After this loop, num_node_range_idx should point to the right repetition_list_element
        while num_node_range_idx < len(repetition_list) - 1 and num_nodes > num_node_ranges_list[num_node_range_idx]:
            num_node_range_idx += 1
        return repetition_list[num_node_range_idx], num_samples_per_condition // repetition_list[num_node_range_idx] + (num_samples_per_condition % repetition_list[num_node_range_idx] > 0)

    def concatenate_sample_chains(self, sample_chains_list):
        # TODO move things function to some utility place
        concatenated_sample_chains_list = []
        for i in range(len(sample_chains_list[0])):
            t = sample_chains_list[0][i][0]
            placeholder_list = []
            for j in range(len(sample_chains_list)):
                placeholder = sample_chains_list[j][i][1]
                placeholder_list.append(placeholder)
            concatenated_sample_chains_list.append((t, graph.concatenate_placeholders(placeholder_list)))
        return concatenated_sample_chains_list
    
    def concatenate_sorted_reactions(self, sorted_reactions, sorting_key, ascending):
        # concatenates together the 'sorted_reactions' datatype, which is a bit of a mess, so this is a bit complicated as well
        # Format of sorted reactions: List of dicts: {prod: list}. Always the same prod. The innter lists are contain dicts with the keys 'elbo', 'count', etc.
        # TODO: This function needs to combine the different lists together in a way that combines them in the counts as well, I think. 
        # sorting key, e.g., 'elbo'
        # Ascending: whether to sort ascending or not. "True" for elbo, "False" for weighted_prob
        prod = list(sorted_reactions[0].keys())[0]
        concatenated_reaction_info = list(itertools.chain.from_iterable([rxns[prod] for rxns in sorted_reactions]))

        # Need to take into account the counts as well here

        resorted_concatenated_reaction_info = sorted(concatenated_reaction_info, key=lambda x: x[sorting_key], reverse=not ascending)
        sorted_reactions = {prod: resorted_concatenated_reaction_info}
        return sorted_reactions
    
    @torch.no_grad()
    def score_one_batch(self, final_samples, true_data, bs, n_samples, n, device, idx=0):
        '''
            Compute various metrics on the generated samples.
            NOTE: 
                - the metrics are computed per sample, i.e. each scores entry has dim (bs, n_samples).
                - the exception is coverage metrics (e.g. any valid results per product) which have the shape (bs, n_samples).

            Input:
                final_samples: Placeholder object, with final_samples.X of shape (bs*n_samples, max_nodes, x_features)
                true_data: Placeholder object, with true_data.X of shape (bs*n_samples, max_nodes, x_features)
            Output:
                scores: dictionary with each entry of dimension (bs, n_samples) or (bs, 1) (see note above for info).
        '''

        t0 = time.time()
        # true_nodes = true_data.X.reshape(bs, n_samples, n)[:,0,...]
        # true_edges = true_data.E.reshape(bs, n_samples, n, n)[:,0,...]
        # take out the product from true_data and final_samples
        assert true_data.smiles is not None, "The true_data object should have the ground-truth SMILES"
        
        if self.cfg.test.return_smiles_with_atom_mapping:
            true_rxn_smiles_with_atom_mapping = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=true_data, cfg=self.cfg, with_atom_mapping=True) 
            gen_rxn_smiles_with_atom_mapping = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=final_samples, cfg=self.cfg, with_atom_mapping=True)
            # true_rxn_smiles_with_atom_mapping = mol.get_cano_smiles_with_atom_mapping_from_dense(X=true_data.X, E=true_data.E, atom_types=self.dataset_info.atom_decoder,
            #                                        bond_types=self.dataset_info.bond_decoder, return_dict=False, atom_map_numbers=true_data.atom_map_numbers)
            # gen_rxn_smiles_with_atom_mapping = mol.get_cano_smiles_with_atom_mapping_from_dense(X=final_samples.X, E=final_samples.E, atom_types=self.dataset_info.atom_decoder,
            #                                         bond_types=self.dataset_info.bond_decoder, return_dict=False, atom_map_numbers=final_samples.atom_map_numbers)
        
        # screw up the atom mapping a little bit on the reactant side. TODO: Remove for actual use
        # graph.noise_atom_mapping(final_samples)
        # gen_rxn_smiles_with_atom_mapping = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=final_samples, cfg=self.cfg, with_atom_mapping=True)

        # true_rxn_smiles = mol.get_cano_smiles_from_dense(X=true_data.X, E=true_data.E, mol_assignment=true_data.mol_assignment, atom_types=self.dataset_info.atom_decoder,
        #                                                  bond_types=self.dataset_info.bond_decoder, return_dict=False)
        true_rxn_smiles = true_data.smiles
        if not self.cfg.dataset.use_stereochemistry:
            true_rxn_smiles = [mol.remove_stereochem_from_smiles(rxn_smi) for rxn_smi in true_rxn_smiles]
        if not self.cfg.dataset.use_charges_as_features and not self.cfg.dataset.with_formal_charge_in_atom_symbols:
            true_rxn_smiles = [mol.remove_charges_from_smiles(rxn_smi) for rxn_smi in true_rxn_smiles] #TODO: MAKE SURE THAT THIS ACTUALLY IS CONSISTENT WITH THE DATASET CREATION
            #true_rxn_smiles = [mol.remove_stereochem_from_smiles(rxn_smi) for rxn_smi in true_rxn_smiles]
        # TODO: figure out canonicalization
        # gen_rxn_smiles = mol.get_cano_smiles_from_dense(X=final_samples.X, E=final_samples.E, mol_assignment=true_data.mol_assignment,
        #                                                 atom_types=self.dataset_info.atom_decoder,
        #                                                 bond_types=self.dataset_info.bond_decoder, return_dict=False)
        gen_rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=final_samples, cfg=self.cfg)
        
        ## TODO: remove true_rxn and n_samples from function below
        # if self.cfg.diffusion.mask_nodes=='reactant_and_sn' or self.cfg.diffusion.mask_edges=='reactant_and_sn':
        #     all_valid_unfiltered, atleastone_valid_unfiltered, _ = mol.check_valid_product_in_rxn(X=final_samples.X, E=final_samples.E, 
        #                                                                                           atom_types=self.dataset_info.atom_decoder, 
        #                                                                                           bond_types=self.dataset_info.bond_decoder,
        #                                                                                           true_rxn_smiles=true_rxn_smiles)
        # else:
        #     all_valid_unfiltered, atleastone_valid_unfiltered, _ = mol.check_valid_reactants_in_rxn(X=final_samples.X, E=final_samples.E, 
        #                                                                                             atom_types=self.dataset_info.atom_decoder, 
        #                                                                                             bond_types=self.dataset_info.bond_decoder,
        #                                                                                             true_rxn_smiles=true_rxn_smiles, n_samples=n_samples)
        
        unique_indices, counts, is_unique, same_reaction_groups = graph.get_unique_indices_from_reaction_list(gen_rxn_smiles)

        # select most likely atom mapping for each group of same reactions (list of lists)
        # wait a minute should I be using the final_samples object here instead for the atom mapping? -> Yes
        #top_atom_mappings = [graph.choose_highest_probability_atom_mapping([g for i,g in enumerate(gen_rxn_smiles_with_atom_mapping) if i in same_reaction_group]) for same_reaction_group in same_reaction_groups]
        # set the atom mapping of reactant side of final_samples to the most likely ones
        #graph.set_atom_mapping_to_reactants_of_placeholder(final_samples, top_atom_mappings)
        if self.cfg.test.return_smiles_with_atom_mapping:
            final_samples_copy = copy.deepcopy(final_samples)
            final_samples = final_samples.select_subset(is_unique).get_new_object()
            for i in range(final_samples.X.shape[0]):
                am = graph.choose_highest_probability_atom_mapping_from_placeholder(final_samples_copy.subset_by_index_list(same_reaction_groups[i]), 
                                                                                [s for j,s in enumerate(gen_rxn_smiles_with_atom_mapping) if j in same_reaction_groups[i]])
                final_samples.atom_map_numbers[i] = am
            # gen_rxn_smiles_with_atom_mapping = [x for x,u in zip(gen_rxn_smiles_with_atom_mapping,is_unique) if u]
            gen_rxn_smiles_with_atom_mapping = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=final_samples, cfg=self.cfg, with_atom_mapping=True)

            # add a check TODO remove in case we don't need it
            for i in range(len(gen_rxn_smiles_with_atom_mapping)):
                prod_assignment_idx = final_samples.mol_assignment[i].max()
                product_node_start_idx = (final_samples.mol_assignment[i] == prod_assignment_idx).nonzero()[0]
                rct_am = am[:product_node_start_idx].tolist()
                prod_am = am[product_node_start_idx:].tolist()
                # rct_am = graph.get_rct_atom_mapping_from_smiles(gen_rxn_smiles_with_atom_mapping[i])
                # prod_am = graph.get_prod_atom_mapping_from_smiles(gen_rxn_smiles_with_atom_mapping[i])
                if (rct_am!=None and prod_am!=None) and ((set(rct_am) - set([0])) != (set(prod_am) - set([0]))): # everything on prod side should be atom mapped, and there should be a corresponding atom mapping on the rct side too
                    log.info("---------------------------------------")
                    # log.info(f"None problem: {(rct_am==None or prod_am==None)}")
                    if not (rct_am==None or prod_am==None):
                        log.info(f"Atom mapping problem: {((set(rct_am) - set([0])) != (set(prod_am) - set([0])))}")
                        log.info(f"rct_am: {rct_am}, prod_am: {prod_am}")
                    try:
                        from mpi4py import MPI
                    except ImportError: # mpi4py is not installed, for local experimentation
                        MPI = None
                        log.warning("mpi4py not found. MPI will not be used.")
                    if MPI:
                        comm = MPI.COMM_WORLD
                        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
                        mpi_rank = comm.Get_rank()
                        log.info(f"MPI size: {mpi_size}, MPI rank: {mpi_rank}")
                    log.info(f"data batch idx: {idx}, gen idx: {i}")
                    log.info(gen_rxn_smiles_with_atom_mapping[i])
                    log.info(true_rxn_smiles_with_atom_mapping[unique_indices[i]])
                    log.info(final_samples.atom_map_numbers[i])
                    log.info("---------------------------------------")
            # graph.set_atom_mapping_to_reactants_of_multiple_reaction_smiles(gen_rxn_smiles_with_atom_mapping, top_atom_mappings)
        else:
            final_samples = final_samples.select_subset(is_unique).get_new_object()
        # deselect
        true_data = true_data.select_subset(is_unique).get_new_object()
        gen_rxn_smiles = [x for x,u in zip(gen_rxn_smiles,is_unique) if u]
        scores = {}

        '''
            Commenting out the other types of evaluation because: 1) not being reported in the paper atm, and 2) getting messed up by dummy valid molecules added when parsing samples from file.
        '''
        
        log.info(time.time()-t0)
        t1 = time.time()
        # TODO: remove true_rxn and n_samples from function below
        # if self.cfg.diffusion.mask_nodes=='reactant_and_sn' or self.cfg.diffusion.mask_edges=='reactant_and_sn':
        #     all_valid, atleastone_valid, _ = mol.check_valid_product_in_rxn(X=final_samples.X, E=final_samples.E, 
        #                                                                     atom_types=self.dataset_info.atom_decoder, 
        #                                                                     bond_types=self.dataset_info.bond_decoder,
        #                                                                     true_rxn_smiles=true_rxn_smiles)
        # else:
        #     all_valid, atleastone_valid, _ = mol.check_valid_reactants_in_rxn(X=final_samples.X, E=final_samples.E, 
        #                                                                       atom_types=self.dataset_info.atom_decoder, 
        #                                                                       bond_types=self.dataset_info.bond_decoder,
        #                                                                       true_rxn_smiles=true_rxn_smiles, n_samples=n_samples)
        log.info(time.time()-t1)
        
        matching_atoms = self.compare_true_and_gen_atoms(true_X=true_data.X, gen_X=final_samples.X, n=n, device=device)
        if self.cfg.test.smiles_accuracy:
            # already in (bs, n_samples) shape: might want to change for consistency
            try:
                matching_smiles = self.compare_true_and_gen_smiles([s for i,s in enumerate(true_rxn_smiles) if i in unique_indices], gen_rxn_smiles, device=device).to(device)
            except:
                raise ValueError(f"Error in comparing true and generated smiles. True: {true_rxn_smiles}, Gen: {gen_rxn_smiles}")
                #matching_smiles = torch.zeros((bs, n_samples)).float().to(device)
            # matching_smiles = torch.zeros((bs, n_samples)).float().to(device) # took this out because crashed randomly somewhere
        else:
            matching_smiles = torch.zeros((bs, n_samples)).float().to(device)
            # get tensors of: (product, n_samples)
        all_valid_bs, atleastone_valid_bs, all_coverage_bs, atleastone_coverage_bs = torch.zeros((bs,)), torch.zeros((bs,)), torch.zeros((bs,)), torch.zeros((bs,))
        matching_smiles_bs, matching_atoms_bs, all_valid_unfiltered_bs, atleastone_valid_unfiltered_bs = torch.zeros((bs,)), torch.zeros((bs,)), torch.zeros((bs,)), torch.zeros((bs,))
        for i in range(bs):
            unique_indices = np.array(unique_indices)
            nonrepeated_indices_in_batch = ((unique_indices>=i*n_samples)&(unique_indices<(i+1)*n_samples)).nonzero()[0]
            # get tensors of: (product, 1) => max over n_samples = at least one sample is valid (of value 1)
            # all_coverage_bs[i] = all_valid[nonrepeated_indices_in_batch].max(dim=-1)[0]
            # atleastone_coverage_bs[i] = atleastone_valid[nonrepeated_indices_in_batch].max(dim=-1)[0]
            # all_valid_bs[i] = all_valid[nonrepeated_indices_in_batch].mean(dim=-1)
            # atleastone_valid_bs[i] = atleastone_valid[nonrepeated_indices_in_batch].mean(dim=-1)
            matching_smiles_bs[i] = matching_smiles[nonrepeated_indices_in_batch].mean(dim=-1)
            matching_atoms_bs[i] = matching_atoms[nonrepeated_indices_in_batch].mean(dim=-1)
            # all_valid_unfiltered_bs[i] = all_valid_unfiltered[nonrepeated_indices_in_batch].mean(dim=-1)
            # atleastone_valid_unfiltered_bs[i] = atleastone_valid_unfiltered[nonrepeated_indices_in_batch].mean(dim=-1)
        
        scores = {'all_valid': all_valid_bs, 'atleastone_valid': atleastone_valid_bs, 
                  'all_coverage': all_coverage_bs, 'atleastone_coverage': atleastone_coverage_bs,
                  'matching_atoms': matching_atoms_bs, 'matching_smiles': matching_smiles_bs,
                  'all_valid_unfiltered': all_valid_unfiltered_bs, 'atleastone_valid_unfiltered': atleastone_valid_unfiltered_bs}
        
        # sorting reactions
        elbo_sorted_rxns, weighted_prob_sorted_rxns = self.sort_samples(final_samples, 
                                                                         gen_rxn_smiles, 
                                                                         gen_rxn_smiles_with_atom_mapping, 
                                                                         is_unique, 
                                                                         n_samples, 
                                                                         counts,
                                                                         bs,
                                                                         idx=0)

        scores = self.compute_topk_scores(elbo_sorted_rxns, weighted_prob_sorted_rxns, true_rxn_smiles, gen_rxn_smiles, is_unique, n_samples, scores)
        
        return scores, elbo_sorted_rxns, weighted_prob_sorted_rxns
    
    def compute_topk_scores(self, elbo_sorted_rxns, weighted_prob_sorted_rxns, true_rxn_smiles, gen_rxn_smiles, is_unique, n_samples, scores={}):
        '''
            Compute topk scores for the generated samples.
        '''
        if len(self.cfg.test.topks)>0:
            log.info("Computing topk....")
            t0 = time.time()
            true_rcts, true_prods = graph.split_reactions_to_reactants_and_products(true_rxn_smiles)
            #true_rcts, true_prods = true_rcts[::n_samples], true_prods[::n_samples] # We don't want duplicates going into the topk calculation
            topk = graph.calculate_top_k(self.cfg, elbo_sorted_rxns, true_rcts, true_prods)
            topk_weighted = graph.calculate_top_k(self.cfg, weighted_prob_sorted_rxns, true_rcts, true_prods)
            
            log.info(f"Done computing topk. Time: {time.time()-t0}")

            for j, k_ in enumerate(self.cfg.test.topks):
                scores[f'top-{k_}'] = topk[:,j]
                scores[f'top-{k_}_weighted_{self.cfg.test.sort_lambda_value}'] = topk_weighted[:,j]
        
        return scores
    
    def sort_samples(self,
                     final_samples,
                     gen_rxn_smiles,
                     gen_rxn_smiles_with_atom_mapping, 
                     is_unique,
                     n_samples, 
                     counts,
                     bs,
                     idx=0):
        log.info("Sorting samples by score_one_batch elbo (for topk and plotting)...")
        t0 = time.time()

        # Calculate the ELBOs of the filtered samples
        log.info(f"Device: {final_samples.X.device}")
        MPI = None

        # try:
        #     print('trying to import mpi')
        #     from mpi4py import MPI
        # except ImportError: # mpi4py is not installed, for local experimentation
        #     print('mpi4py not found')
        #     MPI = None
        #     log.warning("mpi4py not found. MPI will not be used.")
        # # NOTE: temporary fix since cannot use mpi on an interactive gpu session on puhti
        # # MPI = None
        # print(f'found mpi: {MPI}')
        # if MPI:
        #     comm = MPI.COMM_WORLD
        #     mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        #     mpi_rank = comm.Get_rank()
        #     log.info(f"Estimating ELBO for {idx}. MPI size: {mpi_size}, MPI rank: {mpi_rank}")

        elbos, loss_t, loss_0 = self.estimate_elbo_with_repeats(final_samples_one_hot=final_samples.to_one_hot(self.cfg))
        if MPI:
            log.info(f"Done estimating elbos for {idx}. MPI size {mpi_size}, rank {mpi_rank}. Time: {time.time()-t0}. Splitting SMILES to rct and prod")
            t0 = time.time()
        else:
            log.info(f"Done estimating elbos. Time: {time.time()-t0}. Splitting SMILES to rct and prod")
            t0 = time.time()
        gen_rct_smiles, gen_prod_smiles = graph.split_reactions_to_reactants_and_products(gen_rxn_smiles) # TODO: This wraps the sides in lists, so outputs are lists of lists for some reason
        log.info(f"Done splitting SMILES. Time: {time.time()-t0}. Sorting by ELBO")
        t0 = time.time()
        
        # ignore dummy samples of format [U]>>prod_smiles
        for i, rct in enumerate(gen_rct_smiles):
            # check that all atoms are '' (because U is ignored by the smiles conversion functions)
            if len([atom for atom in rct if atom==''])==len(rct): 
                elbos[i] = float('inf')
                counts[i] = 0
        
        if self.cfg.test.return_smiles_with_atom_mapping:
            gen_rct_smiles_with_atom_mapping, gen_prod_smiles_with_atom_mapping= graph.split_reactions_to_reactants_and_products(gen_rxn_smiles_with_atom_mapping)
            elbo_sorted_rxns = self.sort_by_elbo(elbos, loss_t, loss_0, gen_rct_smiles_with_atom_mapping, gen_prod_smiles_with_atom_mapping, is_unique=is_unique, # TODO: This results in the bug where the product is split
                                                 counts=counts, bs=bs, n_samples=n_samples, k=self.cfg.test.topks)
        else:
            elbo_sorted_rxns = self.sort_by_elbo(elbos, loss_t, loss_0, gen_rct_smiles, gen_prod_smiles, is_unique=is_unique, # TODO: This results in the bug where the product is split
                                                counts=counts, bs=bs, n_samples=n_samples, k=self.cfg.test.topks)
        log.info(f"Done sorting by elbo. Time: {time.time()-t0}")
        log.info("Resorting samples by elbo + counts (for topk and plotting)...")
        print(f'elbo_sorted_rxns: {elbo_sorted_rxns.keys()}')
        weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_rxns, self.cfg.test.sort_lambda_value)
        log.info(f"Done resorting. Time: {time.time()-t0}")

        return elbo_sorted_rxns, weighted_prob_sorted_rxns
    
    @torch.no_grad()
    def compare_true_and_gen_atoms(self, true_X, gen_X, n, device=None):
        '''
            Compare the atoms of batched test input with batched generated samples. 

            true_X: batched test input in discrete form, size: (bs, n)
            gen_X: batched generated samples, in discrete form, size: (bs*n_samples, n)
            
            returns: 
                bs_id, sample_id: indices of matched samples (used to generate smiles)
                correct_sample: number of correct matches per test input
        '''
        # compare sorted versions of true and generated discrete samples
        # compute the number of matching atoms between compared samples
        device = true_X.device
        sorted_discrete_true_X = true_X.sort(dim=-1)[0]
        sorted_discrete_gen_X = gen_X.sort(dim=-1)[0]
        n_correct_atoms = (sorted_discrete_true_X.to(device)==sorted_discrete_gen_X.to(device)).sum(dim=-1) # bs*n_samples

        # check that samples have all atoms matching
        matching_atoms = (n_correct_atoms==n).float() # bs

        # get all samples with n atom matching, for all input in batch
        # bs_id, sample_id = (n_correct_atoms==n).reshape(bs, n_samples).nonzero(as_tuple=True) 

        return matching_atoms
    
    @torch.no_grad()
    def compare_true_and_gen_smiles(self, all_true_smiles, all_gen_smiles, device=None):
        '''
            Compute and compare the smiles strings of batched true test and generated samples. 
            Takes into account the fact that the same molecule could be repeated multiple times in the reactants, 
            and that the same reactants can be permuted in many ways in the SMILES strings. 
        
            all_true_smiles: A list of all true smiles strings, size: (bs*n_samples)
            all_gen_smiles: A list of all generated smiles strings, size: (bs*n_samples)

            returns: 
                correct_smiles: boolean vector indicating if test input has smiles matches among the generated samples. size: (bs,n_samples)
        '''
        #matching_smiles = torch.zeros(bs,n_samples).float().to(device)

        # sanity check: the product smiles in true and generated samples should be the same
        true_prods = [r.split('>>')[-1] for r in all_true_smiles]
        gen_prods = [r.split('>>')[-1] for r in all_gen_smiles]
        if self.cfg.diffusion.mask_nodes=='product_and_sn' and self.cfg.diffusion.mask_edges=='product_and_sn':
            if (true_prods!=gen_prods):
                log.info(f'True and gen products not equal! true_pords={true_prods} and gen_prods={gen_prods}\n')

        # matching reactant smiles in this rep for current batch
        # The Counter acts as a multiset
        true_rcts = [r.split('>>')[0] for r in all_true_smiles]
        gen_rcts = [r.split('>>')[0] for r in all_gen_smiles]
        true_rcts = [Counter(r.split('.')) for r in true_rcts]
        gen_rcts = [Counter(r.split('.')) for r in gen_rcts]

        res = torch.tensor([t==g for (t,g) in zip(true_rcts, gen_rcts)], dtype=torch.float)
        #res = res.reshape(bs, n_samples)
        
        return res
    
    @torch.no_grad()
    def plot_diagnostics(self, true_rxns, sample_chains, epoch=0, rxn_offset_nb=0):
        rxn_plots = rxn_vs_sample_plot(true_rxns=true_rxns, sampled_rxns=sample_chains, cfg=self.cfg, chain_name=f'epoch{epoch}', rxn_offset_nb=rxn_offset_nb)
        
        return rxn_plots
    
    # def estimate_elbo_with_repeats(self, final_samples_one_hot):
    #     repeated_elbos = []
    #     repeated_loss_t = []
    #     repeated_loss_0 = []
    #     for rep_e in range(self.cfg.test.repeat_elbo):
    #         log.info(f"Repeat {rep_e}")
    #         with torch.no_grad():
    #             one_time_elbos, one_time_loss_t, one_time_loss_0 = self.elbo(dense_true=final_samples_one_hot, avg_over_batch=False)
    #         repeated_elbos.append(one_time_elbos.unsqueeze(0).float().detach())
    #         repeated_loss_t.append(one_time_loss_t.unsqueeze(0).float().detach())
    #         repeated_loss_0.append(one_time_loss_0.unsqueeze(0).float().detach())
            
    #     elbos = torch.cat(repeated_elbos, dim=0).mean(0) # shape (bs*n_samples)
    #     loss_t = torch.cat(repeated_loss_t, dim=0).mean(0)
    #     loss_0 = torch.cat(repeated_loss_0, dim=0).mean(0)
        
    #     return elbos, loss_t, loss_0
    
    def estimate_elbo_with_repeats(self, final_samples_one_hot):
        log.info(f'final_samples_one_hot.E.shape: {final_samples_one_hot.E.shape}')
        repeated_elbos = []
        repeated_loss_t = []
        repeated_loss_0 = []
        
        # Define the batch size limit
        max_nodes_in_batch_before_split = 110
        for rep_e in range(self.cfg.test.repeat_elbo):
            log.info(f'rep_e: {rep_e}')
            log.info(f"Repeat {rep_e}")
            with torch.no_grad():
                # Check if the batch size exceeds the limit
                if final_samples_one_hot.X.shape[1] > max_nodes_in_batch_before_split and final_samples_one_hot.X.shape[0] > 1:
                    # Split the PlaceHolder into two smaller batches
                    first_batch, second_batch = final_samples_one_hot.split(final_samples_one_hot.X.shape[0] // 2)

                    # Process the first batch
                    one_time_elbos_1, one_time_loss_t_1, one_time_loss_0_1 = self.elbo(dense_true=first_batch, avg_over_batch=False)
                    log.info(f'one_time_elbos_1.shape: {one_time_elbos_1.shape}')

                    # Process the second batch
                    one_time_elbos_2, one_time_loss_t_2, one_time_loss_0_2 = self.elbo(dense_true=second_batch, avg_over_batch=False)
                    log.info(f'one_time_elbos_2.shape: {one_time_elbos_2.shape}')
                    one_time_elbos = torch.cat([one_time_elbos_1, one_time_elbos_2], dim=0)
                    one_time_loss_t = torch.cat([one_time_loss_t_1, one_time_loss_t_2], dim=0)
                    one_time_loss_0 = torch.cat([one_time_loss_0_1, one_time_loss_0_2], dim=0)
                    
                    repeated_elbos.append(one_time_elbos.unsqueeze(0).float().detach())
                    repeated_loss_t.append(one_time_loss_t.unsqueeze(0).float().detach())
                    repeated_loss_0.append(one_time_loss_0.unsqueeze(0).float().detach())
                else:
                    # If the batch size is within the limit, process it directly
                    one_time_elbos, one_time_loss_t, one_time_loss_0 = self.elbo(dense_true=final_samples_one_hot, avg_over_batch=False)
                    log.info(f'one_time_elbos.shape: {one_time_elbos.shape}')
                    repeated_elbos.append(one_time_elbos.unsqueeze(0).float().detach())
                    repeated_loss_t.append(one_time_loss_t.unsqueeze(0).float().detach())
                    repeated_loss_0.append(one_time_loss_0.unsqueeze(0).float().detach())
                    
        elbos = torch.cat(repeated_elbos, dim=0).mean(0)  # shape (bs*n_samples)
        loss_t = torch.cat(repeated_loss_t, dim=0).mean(0)
        loss_0 = torch.cat(repeated_loss_0, dim=0).mean(0)

        return elbos, loss_t, loss_0

    @torch.no_grad()
    def sort_by_elbo(self, elbos, loss_t, loss_0, rct_smiles, prod_smiles, is_unique, counts, bs, n_samples, k=[10]):
        '''
            Check if true is in the top-k of samples (sorted by ELBO).
            
            elbos, loss_t, loss_0: (bs*n_samples)
            is_unique: (bs*n_samples): the samples to consider for ELBO evaluation and ranking later on
            counts: (sum(is_unique),): the number of times each sample was duplicated (assumes that first element corresponds to first non-duplicate sample in final_samples, second to second, etc.)
            k: list of k values to compute topk accuracy.
        '''
        
        if not torch.is_tensor(loss_t): loss_t = torch.zeros_like(elbos) # In case if T=1, we got a scalar instead of a tensor
        unique_indices = torch.tensor(is_unique).nonzero(as_tuple=True)[0] # another way to get the unique indices from the bool vector (old way was using two vectors and arange)
        elbos_sorted_list = []
        loss_t_sorted_list = []
        loss_0_sorted_list = []
        idx_sorted_list = []
        counts_list = []
        for i in range(bs):
            # The following lines choose the correct batch elements taking into consideration that the number of samples 
            # changes per batch element due to dropping of non-unique SMILES
            nonrepeated_indices_in_batch = np.array(((unique_indices>=i*n_samples)&(unique_indices<(i+1)*n_samples)).nonzero(as_tuple=True)[0]) 
            elbos_sorted, idx_sorted = elbos[nonrepeated_indices_in_batch].sort(-1)
            elbos_sorted_list.append(elbos_sorted)
            loss_t_sorted_list.append(loss_t[nonrepeated_indices_in_batch][idx_sorted])
            loss_0_sorted_list.append(loss_0[nonrepeated_indices_in_batch][idx_sorted])
            idx_sorted_list.append(idx_sorted)
            counts_list.append(torch.tensor(counts)[nonrepeated_indices_in_batch][idx_sorted])
        
        assert len(rct_smiles)==sum(is_unique) and len(prod_smiles)==sum(is_unique), 'sample_smiles is different than expected.'
        
        # sort sample-smiles by elbos-idx
        gen_rxns = {}
        unique_indices = torch.tensor(is_unique).nonzero(as_tuple=True)[0]
        # Transform rct_smiles and prod_smiles into numpy arrays of objects, so that we can index them with a list of indices
        rct_smiles_ = np.empty(len(rct_smiles), dtype=object) 
        for i in range(len(rct_smiles_)):
            rct_smiles_[i] = rct_smiles[i]
        prod_smiles_ = np.empty(len(prod_smiles), dtype=object)
        for i in range(len(prod_smiles_)):
            prod_smiles_[i] = prod_smiles[i]

        for i in range(bs):
            # The following lines choose the correct batch elements taking into consideration that the number of samples changes per batch element due to dropping of non-unique SMILES
            nonrepeated_indices_in_batch = np.array(((unique_indices>=i*n_samples)&(unique_indices<(i+1)*n_samples)).nonzero(as_tuple=True)[0]) # The np.array is surprisingly important because this is used for indexing another numpy array later on, and behaves differently in edge cases
            rcts_ = rct_smiles_[nonrepeated_indices_in_batch]
            prods_ = prod_smiles_[nonrepeated_indices_in_batch]
            # The following line gathers together the elbos, loss_t, loss_0, and idx for each sample in the batch that was unique, in sorted order. 
            # The results are in dictionaries for each sample.
            samples_and_elbo = [{'rcts': rcts_[s], 'prod': prods_[s], 'elbo': elbos_sorted_list[i][j].item(), 'loss_t': loss_t_sorted_list[i][j].item(), 
                                 'loss_0': loss_0_sorted_list[i][j].item(), 'sample_idx': s.item(), 'count': counts_list[i][j].item()} for j,s in enumerate(idx_sorted_list[i])]
            product = ".".join(prods_[0]) # product shld be the same for all samples
            if product in gen_rxns.keys():
                gen_rxns[product].extend(samples_and_elbo)
            else:
                gen_rxns[product] = samples_and_elbo
                
        # compute mean topk accuracy
        return gen_rxns