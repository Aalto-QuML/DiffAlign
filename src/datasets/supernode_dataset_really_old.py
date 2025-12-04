import os
import os.path as osp
import pathlib
from typing import Any, Sequence
import pickle
import copy

from src.utils.setup import DistributionNodes

from rdkit.Chem import rdChemReactions
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph
from src.datasets.abstract_dataset import AbstractDataModule, seed_worker
from torch_geometric.loader import DataLoader

from src.utils import graph, mol, setup
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics

MAX_ATOMS_RXN = 300

# size_bins = { # [  7.   25.9  44.8  63.7  82.6 101.5 120.4 139.3 158.2 177.1 196. ]
#              'train': [7, 26, 50, 64, 83, 102],
#              # [ 18.   32.6  47.2  61.8  76.4  91.  105.6 120.2 134.8 149.4 164. ]
#              'test': [18, 33, 48, 77, 91, 106],
#              # [ 17.   29.8  42.6  55.4  68.2  81.   93.8 106.6 119.4 132.2 145. ]
#              'val': [17, 30, 43, 56, 69, 81]}

# batchsize_bins = { # [7.4200e+02 1.0731e+04 1.3928e+04 1.0869e+04 2.8270e+03 (5.6200e+02 1.1300e+02 1.9000e+01 1.1000e+01 1.0000e+00)]
#                   'train': [128, 1024, 1024, 1024, 512, 128],
#                   # [4.500e+02 1.259e+03 1.277e+03 1.242e+03 5.060e+02 1.440e+02 (5.000e+01, 1.500e+01, 5.000e+00, 1.000e+00)]
#                   'test': [64, 256, 256, 256, 64, 32], 
#                   # fixed batchsize for subset test files
#                   'test_i': [10, 10, 10, 10, 10, 10],  
#                   # [ 253.  985. 1132. 1327.  752.  332.  117.   32.   16.    5.]
#                   'val': [32, 64, 128, 128, 128, 128]}

atom_types =  ['Si', 'P', 'N', 'Mg', 'Se', 'Cu', 'S', 'Br', 'B', 'O', 'C', 'Zn', 'Sn', 'F', 'I', 'Cl', 'SuNo']
atom_type_offset = 1 # where to start atom type indexing. 1 if using no-node type, 0 otherwise

# starting from 1 because considering 0 an edge type (= no edge)
bond_types = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, 'mol', 'within', 'across']
bond_type_offset = 1

raw_files = ['train.csv', 'test.csv', 'val.csv']
processed_files = ['train.pt', 'test.pt', 'val.pt']

class Dataset(InMemoryDataset):
    def __init__(self, stage, root, size_test_splits=0):
        self.stage = stage
        self.root = root
        self.size_test_splits = size_test_splits
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'test':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root)
        if 'test_' in self.stage:
            test_path = os.path.join(root, 'processed', self.stage+'.pt')
            print(f'test_path {test_path}\n')
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
        graphs = []
        for i, rxn_ in enumerate(open(self.raw_paths[self.file_idx], 'r')):
            rxn = rdChemReactions.ReactionFromSmarts(rxn_.strip())

            reactants = [r for r in rxn.GetReactants()]
            products = [p for p in rxn.GetProducts()]
            offset = 0 
            # mask: (n), with n = nb of nodes
            mask_product_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only reactant nodes = True
            mask_reactant_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool)
            mask_sn = torch.ones(MAX_ATOMS_RXN, dtype=torch.bool) # only sn = False

            supernodes_rcts = []
            for j, r in enumerate(reactants):
                gi_nodes, gi_edge_index, gi_edge_attr = graph.get_mol_graph_with_supernode(mol=r, supernode_nb=offset+1)
                
                g_nodes = torch.cat((g_nodes, gi_nodes), dim=0) if j > 0 else gi_nodes # already a tensor
                g_edge_index = torch.cat((g_edge_index, gi_edge_index), dim=1) if j > 0 else gi_edge_index
                g_edge_attr = torch.cat((g_edge_attr, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr

                mask_product_and_sn[offset+1:gi_nodes.shape[0]+offset] = True # start from offset+1 because sn at offset

                si = gi_edge_index[0][0].item()
                supernodes_rcts.append(si)
                offset += gi_nodes.shape[0] 

                mask_sn[si] = False

            supernodes_prods = []
            for p in products:
                gi_nodes, gi_edge_index, gi_edge_attr = utils.get_mol_graph_with_supernode(mol=p, supernode_nb=offset+1)

                g_nodes = torch.cat((g_nodes, gi_nodes), dim=0) 
                g_edge_index = torch.cat((g_edge_index, gi_edge_index), dim=1)
                g_edge_attr = torch.cat((g_edge_attr, gi_edge_attr), dim=0)
                
                mask_reactant_and_sn[offset+1:gi_nodes.shape[0]+offset] = True # start from offset+1 because sn at offset

                si = gi_edge_index[0][0].item()
                supernodes_prods.append(si)
                offset += gi_nodes.shape[0]
                mask_sn[si] = False

            # add within edges to reactants
            within_edges = [[s, s_] for s in supernodes_rcts \
                                    for s_ in supernodes_rcts if s!=s_]  # assuming order of edges does not matter
            within_edge_index = torch.tensor(within_edges, dtype=torch.long)

            if len(within_edge_index.shape)==2: 
                within_edge_index = within_edge_index.mT.contiguous()
                nb_within_edges = within_edge_index.shape[1] 
            else: # empty list of within edges => only one reactant
                nb_within_edges = 0

            edge_type = torch.full(size=(nb_within_edges,), fill_value=bond_types.index('within')+bond_type_offset, dtype=torch.long) 
            within_edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)+bond_type_offset).float() # add 1 to len of bonds to account for no edge
            
            # no within edge attr for prods because assuming 1 prod per rxn

            # adding across edges between reactants and products
            across_edges = [[[rs, ps], [ps, rs]] for rs in supernodes_rcts \
                                                 for ps in supernodes_prods]  
            across_edge_index = torch.tensor(across_edges, dtype=torch.long).reshape(len(across_edges)*2, 2)
            
            if len(across_edge_index.shape)==2:
                across_edge_index = across_edge_index.mT.contiguous()
                nb_across_edges = across_edge_index.shape[1]  
            else: # case of one molecule (reactant) in whole reaction
                nb_across_edges = 0 

            edge_type = torch.full(size=(nb_across_edges,), fill_value=bond_types.index('across')+bond_type_offset, dtype=torch.long)
            across_edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)+bond_type_offset).float() # add 1 to len of bonds to account for no edge

            g_edge_index = torch.cat((g_edge_index, within_edge_index, across_edge_index), dim=1)
            g_edge_attr = torch.cat((g_edge_attr, within_edge_attr, across_edge_attr), dim=0)

            y = torch.zeros((1, 0), dtype=torch.float)
            
            # trim masks => one element per node in the rxn graph
            mask_product_and_sn = mask_product_and_sn[:g_nodes.shape[0]] # only reactant nodes = True
            mask_reactant_and_sn = mask_reactant_and_sn[:g_nodes.shape[0]]
            mask_sn = mask_sn[:g_nodes.shape[0]]

            assert mask_sn.shape[0]==g_nodes.shape[0] and \
                   mask_reactant_and_sn.shape[0]==g_nodes.shape[0] and \
                   mask_product_and_sn.shape[0]==g_nodes.shape[0]

            graph = Data(x=g_nodes, edge_index=g_edge_index, 
                         edge_attr=g_edge_attr, y=y, idx=i,
                         mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn, 
                         mask_product_and_sn=mask_product_and_sn)

            graphs.append(graph)
        if self.stage=='test' and self.size_test_splits>0: self.split_test_data(graphs, size_test_splits=self.size_test_splits)
        torch.save(self.collate(graphs), self.processed_paths[self.file_idx])

class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.dataset_nb!='':
            self.datadir += '-'+str(cfg.dataset.dataset_nb)
            self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        super().__init__(cfg)
    
    def prepare_data(self, shuffle=True, seed=0) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': Dataset(stage='train', root=root_path),
                    'val': Dataset(stage='val', root=root_path),
                    'test': Dataset(stage='test', root=root_path, size_test_splits=self.cfg.test.size_test_splits)}
        super().prepare_data(datasets, shuffle=shuffle, seed=seed)

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

class DatasetInfos:
    def __init__(self, datamodule, recompute_info=False):
        self.datamodule = datamodule
        self.name = 'supernode_graphs'
        # self.atom_encoder = ['none']+atom_types # takes type (str) get idx (int)
        self.atom_decoder = ['none']+atom_types
        self.bond_decoder = ['none']+bond_types

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, datamodule.datadist_dir, 'processed')
        node_count_path = os.path.join(root_path, 'n_counts.txt')
        atom_type_path = os.path.join(root_path, 'atom_types.txt')
        edge_type_path = os.path.join(root_path, 'edge_types.txt')
        paths_exist = os.path.exists(node_count_path) and os.path.exists(atom_type_path)\
                      and os.path.exists(edge_type_path)

        if not recompute_info and paths_exist:
            # use the same distributions for all subsets of the dataset
            self.n_nodes = torch.from_numpy(np.loadtxt(node_count_path))
            self.node_types = torch.from_numpy(np.loadtxt(atom_type_path))
            self.edge_types = torch.from_numpy(np.loadtxt(edge_type_path))
        else:
            print('Recomputing\n')
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(node_count_path, self.n_nodes.cpu().numpy())
            self.node_types = datamodule.node_types()                                   
            print("Distribution of node types", self.node_types)
            np.savetxt(atom_type_path, self.node_types.cpu().numpy())
            self.edge_types = datamodule.edge_types()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(edge_type_path, self.edge_types.cpu().numpy())

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule):
        example_batch = next(iter(datamodule.train_dataloader()))

        self.input_dims = {'X': example_batch['x'].size(1), # n or dx?
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}  # + 1 due to time conditioning

        self.output_dims = {'X': example_batch['x'].size(1), # output dim = # of features
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}


            
