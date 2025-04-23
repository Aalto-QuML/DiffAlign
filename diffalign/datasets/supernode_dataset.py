"""
    This file contains the dataset class for the supernode dataset.
"""
import multiprocessing as mp
import os
import logging
import pathlib
import re

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import subgraph
from diffalign.datasets.abstract_dataset import AbstractDataModule, seed_worker
from diffalign.utils import data_utils
from omegaconf import OmegaConf

import pickle
from diffalign.utils import graph
from diffalign.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos, DistributionNodes

from diffalign.utils.mol import get_rdkit_chiral_tags, get_rdkit_bond_types, get_bond_orders, get_bond_orders_correct, get_rdkit_bond_dirs


log = logging.getLogger(__name__)

mp.set_start_method('spawn', force=True)

MAX_ATOMS_RXN = 1000

DUMMY_RCT_NODE_TYPE = 'U'

raw_files = ['train.csv', 'test.csv', 'val.csv']
processed_files = ['train.pt', 'test.pt', 'val.pt']

logging.basicConfig(filename='subprocess_log.txt', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')

class Dataset(InMemoryDataset):
    def __init__(self, stage, cfg):
        self.datadir = cfg.dataset.datadir
        self.datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.dataset_nb!='':
            self.datadir += '-'+str(cfg.dataset.dataset_nb)
            self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        self.stage = stage
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.root = os.path.join(base_path, self.datadir)
        self.cfg = cfg
        self.config_path = os.path.join(self.root, 'processed', 'dataset_config.yaml')
        if self.stage=='train': self.file_idx = 0
        elif self.stage=='test': self.file_idx = 1
        else: self.file_idx = 2
        super().__init__(self.root, transform=self.transform)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    def transform(self, x):
        transform_1 = lambda x: x
        if (self.cfg.neuralnet.pos_encoding_type != 'none' and self.cfg.neuralnet.pos_encoding_type != 'no_pos_enc'):
            pos_encoding_size = data_utils.get_pos_enc_size(self.cfg)
            transform_1 = lambda data: data_utils.positional_encoding_adding_transform(data, self.cfg.neuralnet.pos_encoding_type, pos_encoding_size)
        transform_2 = lambda x: x
        if self.cfg.dataset.drop_atom_maps_rate > 0:
            drop_transform = lambda data: data_utils.drop_atom_maps(data, self.cfg.dataset.drop_atom_maps_rate)
            transform_2 = lambda data: drop_transform(data)
        transform_3 = lambda x: x
        if self.cfg.neuralnet.add_product_identifier:
            product_id_transform = lambda data: data_utils.add_product_id(data)
            transform_3 =product_id_transform
        transform_4 = lambda x: x
        if self.cfg.dataset.add_supernodes:
            add_supernodes_transform = lambda data: data_utils.add_supernodes(self.cfg, data)
            transform_4 = add_supernodes_transform
        transform_5 = lambda x: x
        if self.cfg.dataset.add_supernode_edges:
            add_supernode_edges_transform = lambda data: data_utils.add_supernode_edges(self.cfg, data)
            transform_5 = add_supernode_edges_transform
        return transform_5(transform_4(transform_3(transform_2(transform_1(x)))))

    @property
    def raw_file_names(self):
        return raw_files

    @property
    def processed_file_names(self):
        return processed_files

    # Main function to process the dataset in parallel
    def process(self):
        all_graphs, all_filtered_raw = self.sub_process(open(self.raw_paths[self.file_idx],'r').readlines())
        print(f'===== Saving {len(all_graphs)} graphs to {self.processed_paths[self.file_idx]}.')

        pickle.dump(all_graphs, open(self.processed_paths[self.file_idx].replace('.pt', '_list.pickle'), 'wb'))    
        torch.save(self.collate(all_graphs), self.processed_paths[self.file_idx])
        with open(self.processed_paths[self.file_idx].split('.pt')[0]+'_filtered_raw.csv', 'w') as f:
            f.writelines(all_filtered_raw)
        
        # NOTE: what does this save?
        OmegaConf.save(self.cfg.dataset, self.config_path)

    def sub_process(self, dataset_chunk):
        assert DUMMY_RCT_NODE_TYPE in self.cfg.dataset.atom_types, 'DUMMY_RCT_NODE_TYPE not in atom_types.'
        
        def sub_function():
            graphs = []
            filtered_raw_rxns = []
            for i, rxn_ in enumerate(dataset_chunk):
                log.info(f'Processing reaction {i}')
                
                reactants = [r for r in rxn_.split('>>')[0].split('.')]
                products = [p for p in rxn_.split('>>')[1].split('.')]
                # skip
                if reactants == [''] or products == ['']:
                    continue
                g = graph.turn_reactants_and_product_smiles_into_graphs(self.cfg, reactants, products, i, self.stage)
                if g is not None:
                    graphs.append(g)
                    filtered_raw_rxns.append(rxn_)
                else:
                    log.info(f'No graph found for reaction {i}')
            
            if len(graphs) > 0:
                log.info(f'found {len(graphs)} graphs in subset {self.stage}\n')
            else:
                log.info(f'No graphs found for subset {self.stage}. Skipping this chunk.\n')
            
            return graphs, filtered_raw_rxns

        return sub_function()

class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)
    
    def prepare_data(self, shuffle=True, slices={'train':None, 'val':None, 'test':None}) -> None:
        if self.cfg.dataset.process_only=='train': 
            datasets = {'train': Dataset(stage='train', cfg=self.cfg)}
        elif self.cfg.dataset.process_only=='val': 
            datasets = {'val': Dataset(stage='val', cfg=self.cfg)}
        elif self.cfg.dataset.process_only=='test': 
            datasets = {'test': Dataset(stage='test', cfg=self.cfg)}
        else: 
            datasets = {'train': Dataset(stage='train', cfg=self.cfg),
                        'val': Dataset(stage='val', cfg=self.cfg),
                        'test': Dataset(stage='test', cfg=self.cfg)}
        
        for key, slice_ in slices.items():
            print(f'len(datasets[key]): {len(datasets[key])}\n')
            print(f'slice_: {slice_}\n')
            if slice_ is not None and key in datasets.keys():
                datasets[key] = datasets[key][slice_[0]:slice_[1]]
                
        for key, dataset in datasets.items():
            print(f'len {key} datasets {len(dataset)}\n')
        
        # TODO: Remove this super().prepare_data thing, and define a custom DataLoader class. 
        # TODO: Do we want to turn the data into dense format here? Do we want to calculate the Laplacian eigenvectors as well? -> probably. 
        # -> should probably have a legacy method to calculate the Laplacian eigenvector coefficients, as well as using pytorch_geometric to calculate the Laplacian eigenvector coefficients.
        
        super().prepare_data(datasets, shuffle=shuffle)

    def node_counts(self, max_nodes_possible=MAX_ATOMS_RXN):
        '''
            Number of nodes in a reaction.
        '''
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']: # over all datasets?
            for i, data in enumerate(self.dataloaders[split]):
                # batch_without_sn = data.batch[data.mask_sn] # No supernodes anymore
                _, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1

        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts/all_counts.sum()
        
        return all_counts

    def node_types(self):
        data = next(iter(self.dataloaders['train']))
        num_classes = data.x.shape[1] # including supernode 
        d = torch.zeros(num_classes)

        for data in self.dataloaders['train']:
            d += data.x.sum(dim=0) # supernode is at encoder index -1 => discard it
        d = d / d.sum()

        return d

    def edge_types(self):
        num_classes = None
        data = next(iter(self.dataloaders['train']))
        num_classes = data.edge_attr.shape[1]
        d = torch.zeros(num_classes)

        for i, data in enumerate(self.dataloaders['train']):
            # batch_without_sn = data.batch[data.mask_sn]
            unique, counts = torch.unique(data.batch, return_counts=True)
            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)
            non_sn_node_idx = (data.mask_sn==True).nonzero(as_tuple=True)[0]
            non_sn_edge_index, non_sn_edge_attr = subgraph(non_sn_node_idx, data.edge_index, data.edge_attr)

            num_edges = non_sn_edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = non_sn_edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:] 

        d = d/d.sum() 

        return d

    def node_types_unnormalized(self):
        #TODO: Can this be abstracted to the AbstractDataModule class?
        '''
            Return distribution over the of atom types in molecules.

            Output:
                counts: distribution over types of atoms in molecules.
        '''
        data = next(iter(self.dataloaders['train']))
        num_classes = data.x.shape[1] # get number of atom types from node encoding
        counts = torch.zeros(num_classes)

        for data in self.dataloaders['train']:
            counts += data.x.sum(dim=0)
            
        # ignore SuNo atom type 
        # (set frequencies to 0. because it does not appear in the data anyway)
        if 'SuNo' in self.cfg.dataset.atom_types:
            suno_idx = self.cfg.dataset.atom_types.index('SuNo')
            counts[suno_idx] = 0.

        return counts.long()
    
    def edge_types_unnormalized(self):
        #TODO: Can this be abstracted to the AbstractDataModule class?
        data = next(iter(self.dataloaders['train']))
        num_classes = data.edge_attr.shape[1]

        d = torch.zeros(num_classes)

        for i, data in enumerate(self.dataloaders['train']):
            _, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1) # all_pairs does not include edge from the node to itself

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges
            
            edge_types = data.edge_attr.sum(dim=0)
            edge_types += data.edge_attr.sum(dim=0)

            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]   
            
        # ignore SuNo edge types
        for t in ['mol', 'within', 'across']:
            if t in self.cfg.dataset.bond_types:
                suno_idx = self.cfg.dataset.bond_types.index(t)
                d[suno_idx] = 0.

        return d.long()
    
class DatasetInfos:
    def __init__(self, datamodule, cfg, recompute_info=False):
        '''
            zero_bond_order is a temporary fix to a bug in the creation of bond_orders that affects extra_features later.
            The fix is to accommodate models trained with the bug.
        '''
        self.datamodule = datamodule
        self.name = 'supernode_graphs'
        # self.atom_encoder = ['none']+atom_types # takes type (str) get idx (int)
        self.atom_decoder = cfg.dataset.atom_types
        self.bond_decoder = get_rdkit_bond_types(cfg.dataset.bond_types)
        self.atom_charges = cfg.dataset.atom_charges
        self.atom_chiral_tags = get_rdkit_chiral_tags(cfg.dataset.atom_chiral_tags)
        self.bond_dirs = get_rdkit_bond_dirs(cfg.dataset.bond_dirs)
        self.remove_h = cfg.dataset.remove_h
        # self.valencies = [0] + list(abs[0] for atom_type, abs in allowed_bonds.items() if atom_type in atom_types) + [0]
        if cfg.dataset.different_valencies:
            self.valencies = list(self.get_possible_valences(atom_type)[0] for atom_type in self.atom_decoder)
        else:
            self.valencies = [0] + list(abs[0] for atom_type, abs in cfg.dataset.allowed_bonds.items() if atom_type in cfg.dataset.atom_types) + [0]
        periodic_table = Chem.rdchem.GetPeriodicTable()
        atom_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in self.atom_decoder[1:-1]] + [0] # discard charge
        atom_weights = {atom_type: weight for atom_type, weight in zip(self.atom_decoder, atom_weights)}
        self.atom_weights = atom_weights
        self.max_weight = 390
        print(f'zero_bond_order {cfg.dataset.zero_bond_order}\n')
        if cfg.dataset.zero_bond_order: self.bond_orders = get_bond_orders(self.bond_decoder)
        else: self.bond_orders = get_bond_orders_correct(self.bond_decoder)
        print(f'self.bond_orders {self.bond_orders}\n')
        print(f"Base path: {os.path.realpath(__file__)}")
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.dataset_nb!='': self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        root_path = os.path.join(base_path, self.datadist_dir, 'processed')
        print(f"Root path: {root_path}")
        node_count_path = os.path.join(root_path, 'n_counts.txt')
        atom_type_path = os.path.join(root_path, 'atom_types.txt')
        edge_type_path = os.path.join(root_path, 'edge_types.txt')
        atom_type_unnorm_path = os.path.join(root_path, 'atom_types_unnorm_mol.txt')
        edge_type_unnorm_path = os.path.join(root_path, 'edge_types_unnorm_mol.txt')
        print(f"Checking if all the paths exits ({node_count_path, atom_type_path, edge_type_path, atom_type_unnorm_path, edge_type_unnorm_path})")
        paths_exist = os.path.exists(node_count_path) and os.path.exists(atom_type_path)\
                      and os.path.exists(edge_type_path) and os.path.exists(atom_type_unnorm_path)\
                      and os.path.exists(edge_type_unnorm_path)
        print(f"{paths_exist}")

        if not recompute_info and paths_exist:
            # use the same distributions for all subsets of the dataset
            print(f"Loading the info files...")
            self.n_nodes = torch.from_numpy(np.loadtxt(node_count_path)).float()
            self.node_types = torch.from_numpy(np.loadtxt(atom_type_path)).float()
            self.edge_types = torch.from_numpy(np.loadtxt(edge_type_path)).float()
            self.node_types_unnormalized = torch.from_numpy(np.loadtxt(atom_type_unnorm_path)).long()
            self.edge_types_unnormalized = torch.from_numpy(np.loadtxt(edge_type_unnorm_path)).long()
            print(f"Loaded the info files")
        else:
            print('Recomputing\n')
            np.set_printoptions(suppress=True, precision=5)

            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(node_count_path, self.n_nodes.cpu().numpy())

            self.node_types_unnormalized = datamodule.node_types_unnormalized()
            print("Counts of node types", self.node_types_unnormalized)
            np.savetxt(atom_type_unnorm_path, self.node_types_unnormalized.cpu().numpy())

            self.edge_types_unnormalized = datamodule.edge_types_unnormalized()
            print("Counts of edge types", self.edge_types_unnormalized)
            np.savetxt(edge_type_unnorm_path, self.edge_types_unnormalized.cpu().numpy())

            self.node_types = self.node_types_unnormalized / self.node_types_unnormalized.sum()
            print("Distribution of node types", self.node_types)
            np.savetxt(atom_type_path, self.node_types.cpu().numpy())

            self.edge_types = self.edge_types_unnormalized / self.edge_types_unnormalized.sum()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(edge_type_path, self.edge_types.cpu().numpy())

        print("Completing infos...")
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        print("Done completing infos!")
        
    def get_possible_valences(self, atom_type):
        # TODO: is this buggy somehow?
        pt = Chem.GetPeriodicTable()
        try:
            valence_list = pt.GetValenceList(atom_type)
        except:
            valence_list = [0]
    
        return list(valence_list)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, dx=None, de=None, dy=None, datamodule=None):
        assert datamodule is not None or dx is not None, f'Got datamodule={datamodule} and dx={dx}. One of the two should be specified.\n'
        
        log.info("Computing input/output dims (in function)")

        if dx is not None and de is not None and dy is not None:
            self.input_dims = {'X': dx, 'E': de, 'y': dy+1}  # + 1 due to time conditioning

            self.output_dims = {'X': dx, # output dim = # of features
                                'E': de,
                                'y': 0}
 
        else:
            log.info("Taking an example batch...")
            log.info(f"datamodule.train_dataloader(): {datamodule.train_dataloader()}")
            log.info(f"len(datamodule.train_dataloader()): {len(datamodule.train_dataloader())}")
            log.info(f"iter(datamodule.train_dataloader()): {iter(datamodule.train_dataloader())}")
            example_batch = next(iter(datamodule.train_dataloader()))
            log.info("Completed taking the example batch!")

            self.input_dims = {'X': example_batch.x.size(1), # n or dx?
                            'E': example_batch.edge_attr.size(1),
                            'y': example_batch.y.size(1) + 1,
                            'atom_charges': example_batch.atom_charges.size(1),
                            'atom_chiral': example_batch.atom_chiral.size(1),
                            'bond_dirs': example_batch.bond_dirs.size(1)}  # + 1 due to time conditioning
            self.output_dims = {'X': example_batch.x.size(1), # output dim = # of features
                                'E': example_batch.edge_attr.size(1),
                                'y': 0,
                                'atom_charges': example_batch.atom_charges.size(1),
                                'atom_chiral': example_batch.atom_chiral.size(1),
                                'bond_dirs': example_batch.bond_dirs.size(1)}
        log.info("Completed compute_input_output_dims! (in function)")

