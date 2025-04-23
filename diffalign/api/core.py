'''
This is the core API for the DiffAlign model.
'''
import os
import pathlib
import logging
import re
import random
import pickle
import time
import copy
import datetime
from os import listdir
from os.path import isfile, join
from datetime import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Any
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import wandb
import torch
import numpy as np
from torch import nn
from torch_geometric.loader import DataLoader  # PyG's DataLoader
from torch.utils.data.distributed import DistributedSampler  # PyTorch's DistributedSampler
from torch_geometric.data import Batch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from rdkit import Chem

from diffalign.neuralnet.transformer_model_with_pretraining import GraphTransformerWithPretraining, \
    AtomEncoder
from diffalign.neuralnet.transformer_model_with_y import GraphTransformerWithYAtomMapPosEmbInefficient
from diffalign.utils import setup, io_utils, graph, mol, math_utils
from diffalign.utils.setup import setup_logging, find_free_port, setup_multiprocessing, cleanup
from diffalign.utils.io_utils import create_evaluation_dataloader
from multidiff.filters import all_valid_molecules_in_smiles_list

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This ensures output to console
    ]
)

log = logging.getLogger(__name__)

@dataclass
class DiffAlignConfig:
    """Simple configuration class for DiffAlign predictions"""
    num_samples: int = 100
    num_steps: int = 10
    batch_size: int = 1
    array_job_index: int = 0
    device: str = 'cpu'
    checkpoint_path: Optional[str] = None
    config_file: str = 'multidiff-uspto190.yaml'
    project_root: str = ''
    output_folder: str = ''
    data_dir: str = ''
    set_to_eval: str = 'test'
    # Internal hydra config 
    _hydra_cfg: Optional[Any] = None

    def initialize_hydra(self):
        """Initialize hydra configuration."""
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
            
        with initialize(version_base=None,
                       job_name="predict",
                       config_path='../../configs/experiment'):
            cfg = compose(config_name=self.config_file)
            
        cfg.test.condition_index = self.array_job_index
        cfg.general.wandb.checkpoint_epochs = [620]
        #cfg.general.wandb.run_id = '7ckmnkvc'
        cfg.dataset.datadir = self.data_dir
        cfg.diffusion.edge_conditional_set = self.set_to_eval
        #cfg.dataset.shuffle = False
        #cfg.dataset.dataset_nb= ''
        cfg.dataset.datadist_dir = self.data_dir
        # cfg.test.num_samples_per_condition_subrepetitions = [100,100]
        # cfg.test.num_samples_per_condition_subrepetition_ranges = [10]
        cfg.general.wandb.eval_sampling_steps = [self.num_steps]
        #cfg.test.reassign_atom_map_nums = False
        # cfg.test.condition_first = 0
        # cfg.test.total_cond_eval = 22000
        print(f'self.batch_size: {self.batch_size}')
        cfg.test.batch_size = self.batch_size
        cfg.test.elbo_samples = self.batch_size
        cfg.test.n_conditions = self.batch_size # condition = reaction = product
        #cfg.test.n_conditions = 1 # condition = reaction = product
        cfg.test.n_samples_per_condition = self.num_samples # number of samples/predictions/reactants for each product/reaction
        cfg.diffusion.diffusion_steps = self.num_steps # the higher the better and the slower (training)
        cfg.diffusion.diffusion_steps_eval = self.num_steps # the higher the better and the slower (evaluation)
        #cfg.test.total_cond_eval = 1 # same as n_conditions, normally used when evaluation is parallelized
        #cfg.dataset.num_workers = 0
        
        self._hydra_cfg = cfg
        return self._hydra_cfg

    @property
    def hydra_cfg(self):
        """Get hydra configuration, initializing if needed."""
        if self._hydra_cfg is None:
            self.initialize_hydra()
        return self._hydra_cfg

class DiffAlignPredictor:
    """Main interface for making predictions with DiffAlign"""
    def __init__(self, config: DiffAlignConfig):
        self.config = config
        # Initialize model, load checkpoints etc.

    def initialize_model(self, dataset_infos=None, device_count=1, device='cpu'):
        """Initialize model, dataset, and load weights."""
        cfg = self.config.hydra_cfg
        if dataset_infos is None:
            dataset_infos = setup.get_dataset(cfg=cfg,
                                              dataset_class=setup.task_to_class_and_model[cfg.general.task]\
                                                                ['data_class'],
                                                shuffle=cfg.dataset.shuffle, 
                                                return_datamodule=False, 
                                                recompute_info=False)
        print('==== initializing model ====')
                
        pretrained_input_dims = {'X': 37, 'y': 13, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
        pretrained_output_dims = {'X': 29, 'y': 1, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
        pretrained_model = GraphTransformerWithYAtomMapPosEmbInefficient(n_layers=cfg.neuralnet.n_layers,
                                                                        input_dims=pretrained_input_dims,
                                                                        hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                                                        hidden_dims=cfg.neuralnet.hidden_dims,
                                                                        output_dims=pretrained_output_dims, 
                                                                        act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                                                        pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                                                        improved=cfg.neuralnet.improved, cfg=cfg, 
                                                                        dropout=cfg.neuralnet.dropout,
                                                                        p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                                                        p_to_r_init=cfg.neuralnet.p_to_r_init, 
                                                                        alignment_type=cfg.neuralnet.alignment_type,
                                                                        input_alignment=cfg.neuralnet.input_alignment)
        # TODO: make this dynamic
        atom_encoder_input_dims = 85 # the size of the atom labels + PEs
        # TODO: decide what is an appropriate encoder here
        new_encoder = AtomEncoder(input_dims=atom_encoder_input_dims,
                                hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims['X'],
                                out_dim=pretrained_input_dims['X'])
        # TODO: could also experiment with output layer sizes
        denoiser_with_pretraining = GraphTransformerWithPretraining(pretrained_model, 
                                                                    new_encoder,
                                                                    pretrained_model_out_dim_X=pretrained_output_dims['X'],
                                                                    output_hidden_dim_X=32,
                                                                    output_dim_X=dataset_infos.output_dims['X'])
        
        log.info("Getting the model and train objects...")
        print(f'device: {device}, device_count: {device_count}')
        model, optimizer, scheduler, scaler, _ = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'],
                                                                                            model_kwargs={'dataset_infos': dataset_infos,
                                                                                                        'denoiser': denoiser_with_pretraining,
                                                                                                        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                        'use_data_parallel': False},
                                                                                            parent_path=self.config.project_root, savedir=os.path.join(self.config.project_root, 'experiments'),
                                                                                            load_weights_bool=False, device=device, device_count=device_count)

        # 4. load the weights to the model
        savedir = os.path.join(self.config.project_root, 'checkpoints')
        epoch_num = 100
        weights_path = os.path.join(savedir, f'pretrained_epoch{epoch_num}.pt')
        # For your model
        print(f'loading weights from {weights_path} to cuda:{device}')
        state_dict = torch.load(weights_path, map_location='cpu')['model_state_dict']
        # if not setup.check_if_dataparallel_dict(state_dict):
        #     state_dict = setup.regular_state_dict_to_dataparallel_dict(state_dict)
        #state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        return model
            
    def predict_for_one_product(self,
                           product_smiles: str) -> List[str]:
        """
            Predict reactants for a single product
            reaction_str: atom-mapped reaction string, 
                        e.g. '[C:1][C:2][O:3]+[H:4][H:5]>>[C:6][O:7][O:8]+[C:9][H:10][H:11]'
            experiment_yaml_file: path to the experiment yaml file, 
                        e.g. 'RetroDiffuser/configs/experiment/severi_default_pe_skip.yaml'
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config.hydra_cfg.dataset.num_workers = 0 
        reactants = [product_smiles]
        products = [product_smiles]
        g = graph.turn_reactants_and_product_smiles_into_graphs(self.config.hydra_cfg,
                                                                reactants,
                                                                products,
                                                                0,
                                                                'test')
        dense_data = graph.to_dense(g)
        dense_data = dense_data.to_device(device)
        model = self.initialize_model()
        model = model.to(device)
        samples = model.sample_for_condition(dense_data,
                                                    n_samples=self.config.hydra_cfg.test.n_samples_per_condition,
                                                    inpaint_node_idx=None,
                                                    inpaint_edge_idx=None,
                                                    device=None,
                                                    return_chains=False)
        
        gen_rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=samples,
                                                                        cfg=self.config.hydra_cfg)
        _, counts, is_unique, _ = graph.get_unique_indices_from_reaction_list(gen_rxn_smiles)
        gen_rxn_smiles = [x for x,u in zip(gen_rxn_smiles, is_unique) if u]
         
        elbo_sorted_rxns, weighted_prob_sorted_rxns = model.sort_samples(final_samples=samples, 
                                                                              gen_rxn_smiles=gen_rxn_smiles, 
                                                                              gen_rxn_smiles_with_atom_mapping=gen_rxn_smiles, # don't need this, just passing it for consistency
                                                                              is_unique=is_unique, 
                                                                              n_samples=self.config.hydra_cfg.test.n_samples_per_condition, 
                                                                              counts=counts,
                                                                              bs=1,
                                                                              idx=0)
        
        return elbo_sorted_rxns, weighted_prob_sorted_rxns
        
    def predict(self,
                product_smiles: str,
                top_n: int = 10) -> List[str]:
        """
            Predict reactants for a single product to be used in a multi-step search algorithm.
            Note: the function is called 'predict' because this is what the desp implementation expects.
        """
        elbo_sorted_rxns, weighted_prob_sorted_rxns = self.predict_for_one_product(product_smiles)
        

        filtered_samples = []
        for sample in weighted_prob_sorted_rxns[mol.get_smiles_like_diffalign_output(product_smiles)]:
            if all_valid_molecules_in_smiles_list(sample['rcts']):
                filtered_samples.append(sample)
        
        output = []
        for res in filtered_samples:
            output.append({"rxn_smiles": ".".join(res['rcts']) + ">>" + product_smiles,
                         "score": res['weighted_prob'],
                         "template": '',
                         "reactants": res['rcts']})
        return output
    
    def predict_for_batch(self, batch_size, program_start=0) -> List[List[str]]:
        """
            Predict reactants for a batch of products to be used in a multi-step search algorithm.
        """
        self.config.hydra_cfg.dataset.shuffle = False
        self.config.hydra_cfg.dataset.num_workers = 0
        index_in_array_job = int(self.config.hydra_cfg.test.condition_index)
        print(f'self.config.hydra_cfg.test.n_conditions: {self.config.hydra_cfg.test.n_conditions}')
        
        MPI = None
        print(f'MPI: {MPI}')
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus > 1:
            try:
                from mpi4py import MPI
            except ImportError: # mpi4py is not installed, for local experimentation
                MPI = None
                print("mpi4py not found. MPI will not be used.")

            if MPI:
                comm = MPI.COMM_WORLD
                world_size = comm.Get_size() # if not --ntasks>1, this will be 1
                rank = comm.Get_rank() # this will be 0
            else:
                world_size = 1
                rank = 0
        else:
            world_size = 1
            rank = 0
        print(f'world_size: {world_size}, rank: {rank}')

        # Wrap model in DDP
        #print(f'rank {rank} out of {world_size}: ==== loading dataset ====')
        data_start_index_per_gpu = index_in_array_job*world_size*int(self.config.hydra_cfg.test.n_conditions)\
                                  + rank*int(self.config.hydra_cfg.test.n_conditions)
        data_slices = {'train': None, 'val': None, 'test': None}
        data_slices[self.config.hydra_cfg.diffusion.edge_conditional_set] =\
            [int(data_start_index_per_gpu), 
             int(data_start_index_per_gpu)+int(self.config.hydra_cfg.test.n_conditions)]
        print(f'rank {rank} out of {world_size}: data_slices: {data_slices[self.config.hydra_cfg.diffusion.edge_conditional_set]}')
        datamodule, dataset_infos = setup.get_dataset(cfg=self.config.hydra_cfg,
                                                      dataset_class=setup.task_to_class_and_model[self.config.hydra_cfg.general.task]['data_class'],
                                                      shuffle=self.config.hydra_cfg.dataset.shuffle,
                                                      return_datamodule=True,
                                                      recompute_info=False,
                                                      slices=data_slices)

        if self.config.hydra_cfg.diffusion.edge_conditional_set=='test':
            dataloader = datamodule.test_dataloader()
        elif self.config.hydra_cfg.diffusion.edge_conditional_set=='val':
            dataloader = datamodule.val_dataloader()
        elif self.config.hydra_cfg.diffusion.edge_conditional_set=='train':
            dataloader = datamodule.train_dataloader()
            
        start_model = time.time()
        print(f'rank {rank}: ==== initializing model ====')
        # Do heavy initialization once
        model = self.initialize_model(dataset_infos, device_count=world_size, device=rank)
        print(f"rank {rank}: Model initialization took {time.time() - start_model:.2f}s", flush=True)

        start_broadcast = time.time()
        model = model.cuda(rank)
        print(f"rank {rank}: Model broadcast took {time.time() - start_broadcast:.2f}s", flush=True)
        # Process batches
        results_to_gather = {
            'samples': [],
            'elbo_sorted': [], 
            'weighted_prob': [],
            'scores': []
        }
        print('==== processing batches ====')
        print(f'self.config.hydra_cfg.dataset.data_dir: {self.config.hydra_cfg.dataset.datadir}')
        print(f'self.config.hydra_cfg.diffusion.edge_conditional_set: {self.config.hydra_cfg.diffusion.edge_conditional_set}')
        print(f'len dataloader: {len(dataloader)}')
        for i, batch in enumerate(dataloader):
            batch = batch.to(rank)
            dense_data = graph.to_dense(batch)
            print(f'len batch: {len(batch)}, dense_data: {dense_data.X.shape}, dense_data.E.shape: {dense_data.E.shape}')
            dense_data = dense_data.to_device(rank)
            true_rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=copy.deepcopy(dense_data).mask(collapse=True),
                                                                             cfg=self.config.hydra_cfg)
            samples = model.sample_for_condition(dense_data,
                                                        n_samples=self.config.hydra_cfg.test.n_samples_per_condition,
                                                        inpaint_node_idx=None,
                                                        inpaint_edge_idx=None,
                                                        device=f'cuda:{rank}',
                                                        return_chains=False)
            gen_rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=samples,
                                                                            cfg=self.config.hydra_cfg)
            results_to_gather['samples'].append(gen_rxn_smiles)
            _, counts, is_unique, _ = graph.get_unique_indices_from_reaction_list(gen_rxn_smiles)
            gen_rxn_smiles = [x for x,u in zip(gen_rxn_smiles, is_unique) if u]
            
            batch_size = len(batch)
            elbo_sorted_rxns, weighted_prob_sorted_rxns = model.sort_samples(final_samples=samples, 
                                                                                    gen_rxn_smiles=gen_rxn_smiles, 
                                                                                    gen_rxn_smiles_with_atom_mapping=gen_rxn_smiles, # don't need this, just passing it for consistency
                                                                                    is_unique=is_unique, 
                                                                                    n_samples=self.config.hydra_cfg.test.n_samples_per_condition,
                                                                                    counts=counts,
                                                                                    bs=batch_size,
                                                                                    idx=0)
            results_to_gather['elbo_sorted'].append(elbo_sorted_rxns)
            results_to_gather['weighted_prob'].append(weighted_prob_sorted_rxns)
            
            start_scores = time.time()
            print(f'elbo_sorted_rxns: {elbo_sorted_rxns.keys()}')
            print(f'true_rxn_smiles: {true_rxn_smiles}')
            scores = model.compute_topk_scores(elbo_sorted_rxns,
                                                      weighted_prob_sorted_rxns,
                                                      true_rxn_smiles,
                                                      gen_rxn_smiles,
                                                      is_unique,
                                                      n_samples=self.config.hydra_cfg.test.n_samples_per_condition)
                        
            print(f'rank {rank}: compute_topk_scores batch {i} took {time.time() - start_scores:.2f}s')
            results_to_gather['scores'].append(scores)

        # Gather results from all GPUs
        print('* gathering results')
        #gathered_resutls = io_utils.gather_distributed_results(world_size, results_to_gather)
        gathered_resutls = results_to_gather
        print('* saving results')
        os.makedirs(os.path.join(self.config.output_folder, f'array{index_in_array_job}', f'rank{rank}'), exist_ok=True)
        io_utils.save_to_disk_distributed(gathered_resutls, os.path.join(self.config.output_folder, f'array{index_in_array_job}', f'rank{rank}'))

        return gathered_resutls
            