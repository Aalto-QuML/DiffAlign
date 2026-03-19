import os
import os.path as osp
import pathlib
from typing import Any, Sequence
import pickle
import copy
import re

from rdkit.Chem import rdChemReactions
import torch
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph
from diffalign.datasets.abstract_dataset import AbstractDataModule, seed_worker
from torch_geometric.loader import DataLoader

from diffalign.utils import graph, mol, setup
from diffalign.utils.graph_builder import build_rxn_graph
from diffalign.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos, DistributionNodes
from diffalign.utils.rdkit import  mol2smiles, build_molecule_with_partial_charges
from diffalign.utils.rdkit import compute_molecular_metrics
from diffalign.constants import MAX_ATOMS_RXN, DUMMY_RCT_NODE_TYPE, BOND_TYPES as bond_types
bond_orders = [0, 1, 2, 3, 0, 0, 0]

raw_files = ['train.csv', 'test.csv', 'val.csv']
processed_files = ['train.pt', 'test.pt', 'val.pt']

class Dataset(InMemoryDataset):
    def __init__(self, stage, root, atom_types, size_test_splits=0, with_explicit_h=False, with_formal_charge=False, max_nodes_more_than_product=35,
                 canonicalize_molecule=True, add_supernode_edges=False, permute_mols=False):
        self.stage = stage
        self.root = root
        self.size_test_splits = size_test_splits
        self.with_explicit_h = with_explicit_h
        self.with_formal_charge = with_formal_charge
        self.max_nodes_more_than_product = max_nodes_more_than_product
        self.canonicalize_molecule = canonicalize_molecule
        self.add_supernode_edges = add_supernode_edges
        self.atom_types = atom_types
        self.permute_mols = permute_mols
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'test':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root)
        if 'test_' in self.stage:
            test_path = os.path.join(root, 'processed', self.stage+'.pt')
            self.data, self.slices = torch.load(test_path)
        else:
            self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return raw_files

    @property
    def processed_file_names(self):
        return processed_files

    def split_test_data(self, graphs, size_test_splits=100):
        '''
            (optional) Split data test file to smaller chunks to be used in parallel while testing (e.g. in slurm array jobs).

            Input:
                graphs: a list of processed graph objects (test data)
                size_test_splits: size of one test split to generate. Usually set in the config file.
            Output:
                size_test_splits saved to the dataset's processed directory.
        '''
        filepath = self.processed_paths[self.file_idx].split('.pt')[0]
        for i in range(0, len(graphs), size_test_splits):
            print(f'len(graphs[i:i+size_test_splits]) {len(graphs[i:i+size_test_splits])}\n')
            torch.save(self.collate(graphs[i:i+size_test_splits]), filepath+f'_{int(i/size_test_splits)}.pt')

    def process(self):
        assert DUMMY_RCT_NODE_TYPE in self.atom_types, 'DUMMY_RCT_NODE_TYPE not in atom_types.'
        graphs = []
        for i, rxn_ in enumerate(open(self.raw_paths[self.file_idx], 'r')):
            reactants = [r for r in rxn_.split('>>')[0].split('.')]
            products = [p for p in rxn_.split('>>')[1].split('.')]

            # Pre-check for cannot_generate to preserve train-mode skip behavior
            nb_product_nodes = sum([len(Chem.MolFromSmiles(p).GetAtoms()) for p in products])
            nb_rct_nodes = sum([len(Chem.MolFromSmiles(r).GetAtoms()) for r in reactants])
            nb_dummy_toadd = nb_product_nodes + self.max_nodes_more_than_product - nb_rct_nodes
            if nb_dummy_toadd < 0 and self.stage == 'train':
                # drop the rxns in the training set which we cannot generate
                continue

            graph, cannot_generate = build_rxn_graph(
                reactants=reactants,
                products=products,
                atom_types=self.atom_types,
                bond_types=bond_types,
                max_nodes_more_than_product=self.max_nodes_more_than_product,
                with_explicit_h=self.with_explicit_h,
                with_formal_charge=self.with_formal_charge,
                add_supernode_edges=self.add_supernode_edges,
                canonicalize_molecule=self.canonicalize_molecule,
                permute_mols=self.permute_mols,
                scramble_atom_mapping=True,
                idx=i,
            )

            graphs.append(graph)
        if self.stage=='test' and self.size_test_splits>0: self.split_test_data(graphs, size_test_splits=self.size_test_splits)
        list_path = os.path.join('/'.join(self.processed_paths[self.file_idx].split('/')[:-1]), self.stage+'.pickle')
        pickle.dump(graphs, open(list_path, 'wb'))
        torch.save(self.collate(graphs), self.processed_paths[self.file_idx])

class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.with_explicit_h = cfg.dataset.with_explicit_h
        self.with_formal_charge = cfg.dataset.with_formal_charge
        self.datadir = cfg.dataset.datadir
        self.datadist_dir = cfg.dataset.datadist_dir
        self.max_nodes_more_than_product = cfg.dataset.nb_rct_dummy_nodes
        self.canonicalize_molecule = cfg.dataset.canonicalize_molecule
        print(f'cfg.dataset.add_supernode_edges {cfg.dataset.add_supernode_edges}\n')
        self.add_supernode_edges = cfg.dataset.add_supernode_edges
        self.atom_types = cfg.dataset.atom_types
        self.permute_mols = cfg.dataset.permute_mols
        print(f'self.atom_types {self.atom_types}\n')
        if cfg.dataset.dataset_nb!='':
            self.datadir += '-'+str(cfg.dataset.dataset_nb)
            self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        super().__init__(cfg)
    
    def prepare_data(self, shuffle=True, slices={'train':None, 'val':None, 'test':None}) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': Dataset(stage='train', root=root_path, atom_types=self.atom_types, with_explicit_h=self.with_explicit_h, 
                                     with_formal_charge=self.with_formal_charge, add_supernode_edges=self.add_supernode_edges,
                                     max_nodes_more_than_product=self.max_nodes_more_than_product, canonicalize_molecule=self.canonicalize_molecule,
                                     permute_mols=self.permute_mols),
                    'val': Dataset(stage='val', root=root_path, atom_types=self.atom_types, with_explicit_h=self.with_explicit_h, 
                                   with_formal_charge=self.with_formal_charge, max_nodes_more_than_product=self.max_nodes_more_than_product, 
                                   canonicalize_molecule=self.canonicalize_molecule, add_supernode_edges=self.add_supernode_edges,
                                   permute_mols=self.permute_mols),
                    'test': Dataset(stage='test', root=root_path, atom_types=self.atom_types, size_test_splits=self.cfg.test.size_test_splits, 
                                    with_explicit_h=self.with_explicit_h, with_formal_charge=self.with_formal_charge,
                                    max_nodes_more_than_product=self.max_nodes_more_than_product, add_supernode_edges=self.add_supernode_edges,
                                    canonicalize_molecule=self.canonicalize_molecule,
                                    permute_mols=self.permute_mols)}
        
        for key in slices.keys():
            if slices[key] is not None:
                datasets[key] = datasets[key][slices[key][0]:slices[key][1]]
                
        print(f'datasets {len(datasets["test"])}\n')
                
        super().prepare_data(datasets, shuffle=shuffle)

    def node_counts(self, max_nodes_possible=300):
        '''
            Number of nodes in a rxn - supernodes.
        '''
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']: # over all datasets?
            for i, data in enumerate(self.dataloaders[split]):
                batch_without_sn = data.batch[data.mask_sn] # true everywhere but on sn nodes
                unique, counts = torch.unique(batch_without_sn, return_counts=True)
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
        suno_idx = self.atom_types.index('SuNo')
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

            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]   
            
        # ignore SuNo edge types
        for t in ['mol', 'within', 'across']:
            suno_idx = bond_types.index(t)
            d[suno_idx] = 0.

        return d.long()
    
class DatasetInfos:
    def __init__(self, datamodule, atom_types, allowed_bonds, recompute_info=False, remove_h=True):
        self.datamodule = datamodule
        self.name = 'supernode_graphs'
        # self.atom_encoder = ['none']+atom_types # takes type (str) get idx (int)
        self.atom_decoder = atom_types
        self.bond_decoder = bond_types
        self.remove_h = remove_h
        self.valencies = [0] + list(abs[0] for atom_type, abs in allowed_bonds.items() if atom_type in atom_types) + [0]
        periodic_table = Chem.rdchem.GetPeriodicTable()
        atom_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in atom_types[1:-1]] + [0] # discard charge
        atom_weights = {atom_type: weight for atom_type, weight in zip(atom_types, atom_weights)}
        self.atom_weights = atom_weights
        self.max_weight = 390
        self.bond_orders = bond_orders

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, datamodule.datadist_dir, 'processed')
        node_count_path = os.path.join(root_path, 'n_counts.txt')
        atom_type_path = os.path.join(root_path, 'atom_types.txt')
        edge_type_path = os.path.join(root_path, 'edge_types.txt')
        atom_type_unnorm_path = os.path.join(root_path, 'atom_types_unnorm_mol.txt')
        edge_type_unnorm_path = os.path.join(root_path, 'edge_types_unnorm_mol.txt')
        paths_exist = os.path.exists(node_count_path) and os.path.exists(atom_type_path)\
                      and os.path.exists(edge_type_path) and os.path.exists(atom_type_unnorm_path)\
                      and os.path.exists(edge_type_unnorm_path)

        if not recompute_info and paths_exist:
            # use the same distributions for all subsets of the dataset
            self.n_nodes = torch.from_numpy(np.loadtxt(node_count_path))
            self.node_types = torch.from_numpy(np.loadtxt(atom_type_path))
            self.edge_types = torch.from_numpy(np.loadtxt(edge_type_path))
            self.node_types_unnormalized = torch.from_numpy(np.loadtxt(atom_type_unnorm_path)).long()
            self.edge_types_unnormalized = torch.from_numpy(np.loadtxt(edge_type_unnorm_path)).long()
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

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, dx=None, de=None, dy=None, datamodule=None):
        assert datamodule is not None or dx is not None, f'Got datamodule={datamodule} and dx={dx}. One of the two should be specified.\n'
        
        if dx is not None and de is not None and dy is not None:
            self.input_dims = {'X': dx, 'E': de, 'y': dy+1}  # + 1 due to time conditioning

            self.output_dims = {'X': dx, # output dim = # of features
                                'E': de,
                                'y': 0}
 
        else:
            example_batch = next(iter(datamodule.train_dataloader()))

            self.input_dims = {'X': example_batch['x'].size(1), # n or dx?
                            'E': example_batch['edge_attr'].size(1),
                            'y': example_batch['y'].size(1) + 1}  # + 1 due to time conditioning

            self.output_dims = {'X': example_batch['x'].size(1), # output dim = # of features
                                'E': example_batch['edge_attr'].size(1),
                                'y': 0}


            
