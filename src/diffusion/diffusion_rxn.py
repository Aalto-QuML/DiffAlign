import wandb
import time
import logging
import os
import copy

import torch
import torch.nn.functional as F
import numpy as np

from collections import defaultdict
from collections import Counter
from itertools import compress

from src.diffusion.diffusion_abstract import DiscreteDenoisingDiffusion
from src.utils import graph, mol, setup
from src.datasets import supernode_dataset_old
from src.utils.diffusion.helpers import rxn_diagnostic_chains, rxn_vs_sample_plot
from src.utils.diffusion import helpers
from torch_geometric.utils import to_dense_batch
import torch_geometric.data

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)

MAX_NODES = 300

def debug_get_dummy_samples(cfg, rep=0, device=None):
    assert cfg.general.task=='rxn', 'Code not tested for mol task.'
    assert 'debug' in cfg.dataset.dataset_nb, 'Expects a dataset called debug.'
    data_class = supernode_dataset_old

    cfg.test.testfile = rep
    test_dataloader = setup.load_testfile(cfg, data_class=data_class)
    dense_rxn = graph.to_dense(next(iter(test_dataloader)))
    
    return dense_rxn

class DiscreteDenoisingDiffusionRxn(DiscreteDenoisingDiffusion):
    def __init__(self, cfg, dataset_infos, node_type_counts_unnormalized=None, edge_type_counts_unnormalized=None, save_as_smiles=False, use_data_parallel=False):
        super().__init__(cfg=cfg, dataset_infos=dataset_infos, node_type_counts_unnormalized=node_type_counts_unnormalized, 
                         edge_type_counts_unnormalized=edge_type_counts_unnormalized, use_data_parallel=use_data_parallel)
        
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
            final_samples = self.sample_for_condition(dense_data=dense_data, n_samples=self.cfg.test.n_samples_per_condition, 
                                                      inpaint_node_idx=inpaint_node_idx_, inpaint_edge_idx=inpaint_edge_idx_, device=device)
            gen_rxns = mol.get_cano_smiles_from_dense(X=final_samples.X, E=final_samples.E, atom_types=self.dataset_info.atom_decoder,
                                                      bond_types=self.dataset_info.bond_decoder, return_dict=False)
            dense_data = dense_data.mask(collapse=True)
            true_rxns = mol.get_cano_smiles_from_dense(X=dense_data.X, E=dense_data.E, atom_types=self.dataset_info.atom_decoder, 
                                                       bond_types=self.dataset_info.bond_decoder, return_dict=False)

            # How many conditions actually were processed in this iteration? (can be more than one)
            actual_n_conditions_in_iter = len(true_rxns)//self.cfg.test.n_samples_per_condition
            # Split the processed conditions into singles
            for k in range(actual_n_conditions_in_iter):
                all_gen_rxn_smiles.append(gen_rxns[k*self.cfg.test.n_samples_per_condition:(k+1)*self.cfg.test.n_samples_per_condition])
                all_true_rxn_smiles.append(true_rxns[k*self.cfg.test.n_samples_per_condition:(k+1)*self.cfg.test.n_samples_per_condition])
                all_true_rxn_pyg.append(dense_data.subset_by_idx(k*self.cfg.test.n_samples_per_condition, (k+1)*self.cfg.test.n_samples_per_condition)
                                            .to_cpu().pyg()) # TODO: Transfer to CPU as well
                all_gen_rxn_pyg.append(final_samples.subset_by_idx(k*self.cfg.test.n_samples_per_condition, (k+1)*self.cfg.test.n_samples_per_condition)
                                            .to_cpu().pyg())

        return all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg

    def sample_for_condition(self, dense_data, n_samples, inpaint_node_idx, inpaint_edge_idx, device=None):
        """
            Inputs:
            condition_idx: The index of the first conditioning product in the batch. Used for plotting.
        """
        # TODO: Division between eval_one_batch and score_one_batch could be a bit clearer
        # repeat the same data for n_samples times (to be able to generate n_samples per conditional object)
        bs = dense_data.X.shape[0]
        device = dense_data.X.device
        dense_data_dup = graph.duplicate_data(dense_data=dense_data, n_samples=n_samples, get_discrete_data=False).to_device(device)
        # duplicate the node/edge inpainting idx
        inpaint_node_idx_ = [item for item in inpaint_node_idx for ns in range(n_samples)] if inpaint_node_idx is not None else None
        inpaint_edge_idx_ = [item for item in inpaint_edge_idx for ns in range(n_samples)] if inpaint_edge_idx is not None else None

        t0 = time.time()
        log.info(f'About to sample')
        final_samples, actual_sample_chains, prob_s_chains, pred_0_chains, true_rxns = self.sample_one_batch(data=dense_data_dup, inpaint_node_idx=inpaint_node_idx_, inpaint_edge_idx=inpaint_edge_idx_, 
                                                                                                             get_true_rxns=True, get_chains=True, device=device) 
        log.info(f"Sampling time: {time.time()-t0}")
        # plot chains when sampling?
        
        return final_samples
       
    def evaluate_from_artifact(self, dense_data, final_samples, epoch=None, device=None, condition_range=None):
        '''
            Evaluate samples read from a text file.
            
            final_samples: final_samples.X.shape=(bs, n_samples, n)
            dense_data: dense_data.X.shape=(bs, n_samples, n)
        '''
        assert dense_data.X.ndim==3, f'Expected the dense_data.X to be of shape=(bs, n_samples, n). Got {dense_data.X.shape}.\n'
        
        # 1. get dense_data to be of the right shape
        t0 = time.time()
        bs, n_samples, n = dense_data.X.shape[0], dense_data.X.shape[1], dense_data.X.shape[2]

        # flatten bs and n_samples
        dense_data = dense_data.flatten(start_dim=0, end_dim=1)
        final_samples = final_samples.flatten(start_dim=0, end_dim=1)
    
        # 2. score the set of samples generated for each condition
        t0 = time.time()
        # NOTE: the way this is used for evaluate_full_dataset, num_gpus should always be 1
        # NOTE: keeping the num_gpus code here in case we want to use the function elsewhere later 
        # NOTE: removed that part of the code, this makes the usage much easier
        num_gpus = 1 #torch.cuda.device_count() if isinstance(self.model, torch.nn.DataParallel) else 1
        #scores = defaultdict(lambda:0) 
        scores = []
        # start = condition_range[0] if condition_range else 0
        # end = condition_range[1] if condition_range else bs//num_gpus
        # step = num_gpus if condition_range else 1
        
        # NOTE: not condition range because we're reading samples one by one
        start = 0
        end = bs//num_gpus
        step = 1
        
        all_elbo_sorted_reactions = []
        all_weighted_prob_sorted_rxns = []
        placeholders_for_print = []
        for i in range(start, end, step):
            dense_data_ = dense_data.subset_by_idx(start_idx=n_samples*i*num_gpus, end_idx=n_samples*i*num_gpus+num_gpus*n_samples)
            final_samples_ = final_samples.subset_by_idx(start_idx=n_samples*i*num_gpus, end_idx=n_samples*i*num_gpus+num_gpus*n_samples)
            scores_, elbo_sorted_reactions, weighted_prob_sorted_rxns = self.score_one_batch(final_samples=final_samples_, true_data=dense_data_, 
                                                                                             bs=num_gpus, n_samples=n_samples, n=n, device=device)
            
            for key in scores_.keys(): # Make sure that no sneak in. This may be a bit paranoid, but trying to evade memory leaks
                if type(scores_[key]) == torch.Tensor:
                    scores_[key] = scores_[key].item()

            # log.info(scores_)
            # log.info(elbo_sorted_reactions)
            # log.info(weighted_prob_sorted_rxns)

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
        log.info("calculating ELBO...")
        # TODO: Change this to a similar estimate that we would have during training
        test_elbo = self.get_elbo_of_data(datamodule.test_dataloader(), n_samples=self.cfg.test.elbo_samples, device=device)
        train_elbo = self.get_elbo_of_data(datamodule.train_dataloader(), n_samples=self.cfg.test.elbo_samples, device=device)
        log.info(f"ELBO train: {train_elbo}, ELBO test: {test_elbo}. Time taken: {time.time()-elbo_of_data_time}")
        
        if self.cfg.test.full_dataset: 
            scores = self.eval_full_dataset(dataloader=additional_dataloader, inpaint_node_idx=inpaint_node_idx, 
                                            inpaint_edge_idx=inpaint_edge_idx, epoch=epoch, device_to_use=device)
        else: 
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
                scores = helpers.accumulate_rxn_scores(acc_scores=scores, new_scores=scores_, total_iterations=len(dataloader))
                
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

            self.save_reactions_to_text(dense_data, elbo_sorted_reactions, weighted_prob_sorted_rxns, epoch=epoch, condition_idx=i*num_gpus, start=i*num_gpus)

            for k_ in scores_.keys():
                if type(scores[k_])==torch.tensor:
                    scores_[k_] = scores_[k_].cpu().flatten() # Should be of shape (bs,) at this point, or (just 0-dim)
                    if 'cannot_generate' in data_.keys:
                        scores_[k_] *= ~data_.cannot_generate.cpu() #
            log.info(f"scores for condition {i*num_gpus}-{(i+1)*num_gpus}: {scores_}\n")
            scores = helpers.accumulate_rxn_scores(acc_scores=scores, new_scores=scores_, total_iterations=self.cfg.test.n_conditions//num_gpus)

        return scores
        
    def save_reactions_to_text(self, original_data_placeholder, elbo_sorted_reactions, weighted_prob_sorted_rxns, epoch, condition_idx, start=0):
        t0 = time.time()
        true_rcts, true_prods = mol.get_cano_list_smiles(X=original_data_placeholder.X, E=original_data_placeholder.E, atom_types=self.dataset_info.atom_decoder, 
                                                         bond_types=self.dataset_info.bond_decoder, plot_dummy_nodes=self.cfg.test.plot_dummy_nodes)
        # true_rcts = [true_rcts[i*n_samples] for i in range(len(gen_rxns))] # Filter out duplicates (each is repeated 100 times at the moment)
        # true_prods = [true_prods[i*n_samples] for i in range(len(gen_rxns))] # Filter out duplicates
        
        graph.save_samples_to_file_without_weighted_prob(f'eval_epoch{epoch}_s{start}.txt', condition_idx, elbo_sorted_reactions, true_rcts, true_prods)
        graph.save_samples_to_file(f'eval_epoch{epoch}_resorted_{self.cfg.test.sort_lambda_value}_s{start}.txt', condition_idx, weighted_prob_sorted_rxns, true_rcts, true_prods)
        
        log.info(f"Saving samples to file time: {time.time()-t0}")

    @torch.no_grad()
    def eval_one_batch(self, dense_data, n_samples, device, epoch=None, condition_idx=None, inpaint_node_idx=None, inpaint_edge_idx=None):
        """
        Inputs:
        condition_idx: The index of the first conditioning product in the batch. Used for plotting.
        """
        # 
        # TODO: Division between eval_one_batch and score_one_batch could be a bit clearer
        # repeat the same data for n_samples times (to be able to generate n_samples per conditional object)
        #bs = data.batch.max().item()+1
        bs = dense_data.X.shape[0]
        device = dense_data.X.device
        log.info("Device really in use: {}".format(dense_data.X.device))
        dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)
        # duplicate the node/edge inpainting idx
        inpaint_node_idx_ = [item for item in inpaint_node_idx for ns in range(n_samples)] if inpaint_node_idx is not None else None
        inpaint_edge_idx_ = [item for item in inpaint_edge_idx for ns in range(n_samples)] if inpaint_edge_idx is not None else None

        t0 = time.time()
        log.info(f'About to sample')
        final_samples, actual_sample_chains, prob_s_chains, pred_0_chains, true_rxns = self.sample_one_batch(data=dense_data, device=device, inpaint_node_idx=inpaint_node_idx_, 
                                                                                                             inpaint_edge_idx=inpaint_edge_idx_, get_true_rxns=True, 
                                                                                                             get_chains=True)

        # Final samples is a PlaceHolder object. actual_sample_chains is a list of PlaceHolder objects. true_rxns is a PlaceHolder object.
        # final_samples.X is of shape (bs*n_samples, max_nodes, x_features). 
        log.info(f"Sampling time: {time.time()-t0}")
        
        t0 = time.time()
        dense_data = dense_data.mask(dense_data.node_mask, collapse=True)
        scores, elbo_sorted_reactions, weighted_prob_sorted_rxns = self.score_one_batch(final_samples=final_samples, true_data=dense_data, bs=bs, n_samples=n_samples, n=dense_data.X.shape[1], device=device)
        log.info(f"Scoring time: {time.time()-t0}")
                
        # iterate over the batch size and plot the sample with the lowest elbo for chains_to_save chains samples.
        if self.cfg.test.plot_rxn_chains:
            t0 = time.time()
            rxn_plots = [] # Default value if no chains are saved for some reason
            for i, prod in enumerate(elbo_sorted_reactions.keys()): 
                if i+condition_idx+1<=self.cfg.test.chains_to_save: # handles the case of multiple conditions plotted at once
                    rxn_plots.extend(self.plot_diagnostics(true_rxns=true_rxns.select_by_batch_and_sample_idx(bs, n_samples, i, 0), 
                                                           sample_chains=graph.select_placeholder_from_chain_by_batch_and_sample(chains=actual_sample_chains, bs=bs, n_samples=n_samples,
                                                                                                                              batch_idx=i, sample_idx=elbo_sorted_reactions[prod][0]['sample_idx']),
                                                           epoch=epoch, inpaint_node_idx=inpaint_node_idx_, inpaint_edge_idx=inpaint_edge_idx_, rxn_offset_nb=condition_idx+i))
            scores['rxn_plots'] = rxn_plots
            log.info(f"Plotting time: {time.time()-t0}")
        
        return scores, elbo_sorted_reactions, weighted_prob_sorted_rxns
        
    @torch.no_grad()
    def score_one_batch(self, final_samples, true_data, bs, n_samples, n, device):
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
        true_nodes = true_data.X.reshape(bs, n_samples, n)[:,0,...]
        true_edges = true_data.E.reshape(bs, n_samples, n, n)[:,0,...]
        
        true_rcts, true_prods = mol.get_cano_list_smiles(X=true_nodes, E=true_edges, atom_types=self.dataset_info.atom_decoder, 
                                                         bond_types=self.dataset_info.bond_decoder, plot_dummy_nodes=self.cfg.test.plot_dummy_nodes)
        gen_rxn_smiles = mol.get_cano_smiles_from_dense(X=final_samples.X, E=final_samples.E, atom_types=self.dataset_info.atom_decoder,
                                                        bond_types=self.dataset_info.bond_decoder, return_dict=False)
        true_rxn_smiles = mol.get_cano_smiles_from_dense(X=true_data.X, E=true_data.E, atom_types=self.dataset_info.atom_decoder, 
                                                         bond_types=self.dataset_info.bond_decoder, return_dict=False)
        
        unique_indices, counts, is_unique = graph.get_unique_indices_from_reaction_list(gen_rxn_smiles)
        # deselect
        final_samples = final_samples.select_subset(is_unique).get_new_object()
        true_data = true_data.select_subset(is_unique).get_new_object()
        gen_rxn_smiles = [x for x,u in zip(gen_rxn_smiles,is_unique) if u]
        scores = {}
        
        '''
            Commenting out the other types of evaluation because: 1) not being reported in the paper atm, and 2) getting messed up by dummy valid molecules added when parsing samples from file.
        '''
        
        # TODO: remove true_rxn and n_samples from function below
        if self.cfg.diffusion.mask_nodes=='reactant_and_sn' or self.cfg.diffusion.mask_edges=='reactant_and_sn':
            all_valid_unfiltered, atleastone_valid_unfiltered, _ = mol.check_valid_product_in_rxn(X=final_samples.X, E=final_samples.E, 
                                                                                                  atom_types=self.dataset_info.atom_decoder, 
                                                                                                  bond_types=self.dataset_info.bond_decoder,
                                                                                                  true_rxn_smiles=true_rxn_smiles)
        else:
            all_valid_unfiltered, atleastone_valid_unfiltered, _ = mol.check_valid_reactants_in_rxn(X=final_samples.X, E=final_samples.E, 
                                                                                                    atom_types=self.dataset_info.atom_decoder, 
                                                                                                    bond_types=self.dataset_info.bond_decoder,
                                                                                                    true_rxn_smiles=true_rxn_smiles, n_samples=n_samples)
        log.info(time.time()-t0)
        t1 = time.time()
        # TODO: remove true_rxn and n_samples from function below
        if self.cfg.diffusion.mask_nodes=='reactant_and_sn' or self.cfg.diffusion.mask_edges=='reactant_and_sn':
            all_valid, atleastone_valid, _ = mol.check_valid_product_in_rxn(X=final_samples.X, E=final_samples.E, 
                                                                            atom_types=self.dataset_info.atom_decoder, 
                                                                            bond_types=self.dataset_info.bond_decoder,
                                                                            true_rxn_smiles=true_rxn_smiles)
        else:
            all_valid, atleastone_valid, _ = mol.check_valid_reactants_in_rxn(X=final_samples.X, E=final_samples.E, 
                                                                              atom_types=self.dataset_info.atom_decoder, 
                                                                              bond_types=self.dataset_info.bond_decoder,
                                                                              true_rxn_smiles=true_rxn_smiles, n_samples=n_samples)
        log.info(time.time()-t1)
        
        matching_atoms = self.compare_true_and_gen_atoms(true_X=true_data.X, gen_X=final_samples.X, n=n, device=device)
        if self.cfg.test.smiles_accuracy:
            # already in (bs, n_samples) shape: might want to change for consistency
            try:
                matching_smiles = self.compare_true_and_gen_smiles([s for i,s in enumerate(true_rxn_smiles) if i in unique_indices], gen_rxn_smiles, device=device).to(device)
            except:
                matching_smiles = torch.zeros((bs, n_samples)).float().to(device)
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
            all_coverage_bs[i] = all_valid[nonrepeated_indices_in_batch].max(dim=-1)[0]
            atleastone_coverage_bs[i] = atleastone_valid[nonrepeated_indices_in_batch].max(dim=-1)[0]
            all_valid_bs[i] = all_valid[nonrepeated_indices_in_batch].mean(dim=-1)
            atleastone_valid_bs[i] = atleastone_valid[nonrepeated_indices_in_batch].mean(dim=-1)
            matching_smiles_bs[i] = matching_smiles[nonrepeated_indices_in_batch].mean(dim=-1)
            matching_atoms_bs[i] = matching_atoms[nonrepeated_indices_in_batch].mean(dim=-1)
            all_valid_unfiltered_bs[i] = all_valid_unfiltered[nonrepeated_indices_in_batch].mean(dim=-1)
            atleastone_valid_unfiltered_bs[i] = atleastone_valid_unfiltered[nonrepeated_indices_in_batch].mean(dim=-1)
        
        scores = {'all_valid': all_valid_bs, 'atleastone_valid': atleastone_valid_bs, 
                  'all_coverage': all_coverage_bs, 'atleastone_coverage': atleastone_coverage_bs,
                  'matching_atoms': matching_atoms_bs, 'matching_smiles': matching_smiles_bs,
                  'all_valid_unfiltered': all_valid_unfiltered_bs, 'atleastone_valid_unfiltered': atleastone_valid_unfiltered_bs}

        log.info("Sorting samples bscore_one_y elbo (for topk and plotting)...")
        t0 = time.time()

        # Calculate the ELBOs of the filtered samples
        log.info(f"Device: {final_samples.X.device}")
        elbos, loss_t, loss_0 = self.estimate_elbo_with_repeats(final_samples_one_hot = final_samples.get_new_object(X=F.one_hot(final_samples.X.long(), len(self.dataset_info.atom_decoder)).float(),
                                                                                                                     E=F.one_hot(final_samples.E.long(), len(self.dataset_info.bond_decoder)).float()))
        
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
        
        elbo_sorted_rxns = self.sort_by_elbo(elbos, loss_t, loss_0, gen_rct_smiles, gen_prod_smiles, is_unique=is_unique, # TODO: This results in the bug where the product is split
                                             counts=counts, bs=bs, n_samples=n_samples, k=self.cfg.test.topks)
        log.info(f"Done sorting by elbo. Time: {time.time()-t0}")
        log.info("Resorting samples by elbo + counts (for topk and plotting)...")
        weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_rxns, self.cfg.test.sort_lambda_value)
        log.info(f"Done resorting. Time: {time.time()-t0}")

        if len(self.cfg.test.topks)>0:
            log.info("Computing topk....")
            t0 = time.time()
            true_rcts, true_prods = graph.split_reactions_to_reactants_and_products(true_rxn_smiles)
            true_rcts, true_prods = true_rcts[::n_samples], true_prods[::n_samples] # We don't want duplicates going into the topk calculation
            topk = graph.calculate_top_k(self.cfg, elbo_sorted_rxns, true_rcts, true_prods)
            topk_weighted = graph.calculate_top_k(self.cfg, weighted_prob_sorted_rxns, true_rcts, true_prods)
            
            log.info(f"Done computing topk. Time: {time.time()-t0}")

            for j, k_ in enumerate(self.cfg.test.topks):
                scores[f'top-{k_}'] = topk[:,j]
                scores[f'top-{k_}_weighted_{self.cfg.test.sort_lambda_value}'] = topk_weighted[:,j]
        
        return scores, elbo_sorted_rxns, weighted_prob_sorted_rxns
    
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
            assert (true_prods==gen_prods), 'True and gen products not equal!'

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
    def plot_diagnostics(self, true_rxns, sample_chains, epoch=0, rxn_nb=0, inpaint_node_idx=None, inpaint_edge_idx=None, rxn_offset_nb=0):
        # samples = next(iter(dataloader)).to(device)
        # samples_l = [s for s in samples.to_data_list() if 'cannot_generate' in s.keys and s.cannot_generate==False]
        # if len(samples_l)==0:
        #     return None
        # else:
        #     samples = torch_geometric.data.Batch.from_data_list(samples_l)
        
        # self.cfg.train.chains_to_save = min(len(samples_l), self.cfg.train.chains_to_save)
                    
        # dense_data = graph.to_dense(data=samples)
        # log.info(dense_data.X.shape)
        # assert self.cfg.train.chains_to_save<=dense_data.X.shape[0], 'Too many chains to plot. Consider plotting less than the batch_size for memory efficiency.'
        
        # data = dense_data.slice_by_idx(idx=self.cfg.train.chains_to_save)
        # inpaint_node_idx_ = inpaint_node_idx[:self.cfg.train.chains_to_save]
        # inpaint_edge_idx_ = inpaint_edge_idx[:self.cfg.train.chains_to_save]

        # final_samples, actual_sample_chains, prob_s_chains, pred_0_chains, true_rxns = self.sample_one_batch(data=data, get_chains=True, get_true_rxns=True, inpaint_node_idx=inpaint_node_idx_, inpaint_edge_idx=inpaint_edge_idx_)
        
        # plot the rxn generation from the sample
        rxn_plots = rxn_vs_sample_plot(true_rxns=true_rxns, sampled_rxns=sample_chains, atom_types=self.dataset_info.atom_decoder, bond_types=self.dataset_info.bond_decoder, 
                                       chain_name=f'epoch{epoch}', plot_dummy_nodes=self.cfg.test.plot_dummy_nodes, rxn_offset_nb=rxn_offset_nb)
        
        return rxn_plots
    
    def estimate_elbo_with_repeats(self, final_samples_one_hot):
        repeated_elbos = []
        repeated_loss_t = []
        repeated_loss_0 = []
        for rep_e in range(self.cfg.test.repeat_elbo):
            log.info(f"Repeat {rep_e}")
            one_time_elbos, one_time_loss_t, one_time_loss_0 = self.elbo(dense_true=final_samples_one_hot, avg_over_batch=False)
            repeated_elbos.append(one_time_elbos.unsqueeze(0).float())
            repeated_loss_t.append(one_time_loss_t.unsqueeze(0).float())
            repeated_loss_0.append(one_time_loss_0.unsqueeze(0).float())
            
        elbos = torch.cat(repeated_elbos, dim=0).mean(0) # shape (bs*n_samples)
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