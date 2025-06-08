import os
import os.path as osp
import pathlib
from typing import Any, Sequence
import pickle
import copy
import re

from diffalign_old.utils.setup import DistributionNodes

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
from diffalign_old.datasets.abstract_dataset import AbstractDataModule, seed_worker
from torch_geometric.loader import DataLoader

from diffalign_old.utils import graph, mol, setup
from diffalign_old.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from diffalign_old.utils.rdkit_deprecated import  mol2smiles, build_molecule_with_partial_charges
from diffalign_old.utils.rdkit_deprecated import compute_molecular_metrics

MAX_ATOMS_RXN = 500

# size_bins = { # [  7.   25.9  44.8  63.7  82.6 101.5 120.4 139.3 158.2 177.1 196. ]
#              'train': [7, 26, 50, 64, 83, 102],
#              # [ 18.   32.6  47.2  61.8  76.4  91.  105.6 120.2 134.8 149.4 164. ]
#              'test': [18, 33, 48, 77, 91, 106],
#              # [ 17.   29.8  42.6  55.4  68.2  81.   93.8 106.6 119.4 132.2 145. ]
#              'val': [17, 30, 43, 56, 69, 81]}

size_bins = {
    'train': [64,83,102],
    'test': [250],
    'val': [250]
}

batchsize_bins = { 
    'train': [32, 16, 8], # [128, 64, 16]
    'test': [32], # [64]
    'val': [32] # [64]
}

# batchsize_bins = { 
#     'train': [32, 16, 8], # [128, 64, 16]
#     'test': [32], # [64]
#     'val': [32] # [64]
# }

# batchsize_bins = { # [7.4200e+02 1.0731e+04 1.3928e+04 1.0869e+04 2.8270e+03 (5.6200e+02 1.1300e+02 1.9000e+01 1.1000e+01 1.0000e+00)]
#                   'train': [128, 128, 128, 64, 32, 16],
#                   # [4.500e+02 1.259e+03 1.277e+03 1.242e+03 5.060e+02 1.440e+02 (5.000e+01, 1.500e+01, 5.000e+00, 1.000e+00)]
#                   'test': [64, 256, 256, 256, 64, 32], 
#                   # fixed batchsize for subset test files
#                   'test_i': [10, 10, 10, 10, 10, 10],  
#                   # [ 253.  985. 1132. 1327.  752.  332.  117.   32.   16.    5.]
#                   'val': [32, 64, 128, 128, 128, 128]}

atom_types = ['none', 'O', 'C', 'N', 'I', 'H', 'Cl', 'Si', 'F', 'Br', 'N+1', 'O-1', 'S', 'B', 'N-1', 
              'Zn+1', 'Cu', 'Sn', 'P+1', 'Mg+1', 'C-1', 'P', 'S+1', 'S-1', 'Se', 'Zn', 'Mg', 'Au', 
              'SuNo']

allowed_bonds = {'O': [2], 'C': [4], 'N': [3], 'I': [1, 3, 5, 7], 'H': [1], 'Cl': [1], 'Si': [4, 6], 'F': [1],
                 'Br': [1], 'N+1': [4], 'O-1': [1], 'S': [2, 4, 6], 'B': [3], 'N-1': [2], 'Zn+1': [3], 'Cu': [1, 2], 
                 'Sn': [2, 4], 'P+1': [4, 6, 8], 'Mg+1': [3], 'C-1': [3], 'P': [3, 5, 7], 'S+1': [3, 5, 7], 'S-1': [1, 3, 5], 
                 'Se': [2, 4, 6], 'Zn': [2], 'Mg': [2], 'Au': [0]}

valencies = [0] + list(abs[0] for abs in allowed_bonds.values()) + [0]

periodic_table = Chem.rdchem.GetPeriodicTable()
atom_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in atom_types[1:-1]] + [0] # discard charge
atom_weights = {atom_type: weight for atom_type, weight in zip(atom_types, atom_weights)}

bond_types = ['none', BT.SINGLE, BT.DOUBLE, BT.TRIPLE, 'mol', 'within', 'across']
bond_orders = [0, 1, 2, 3, 0, 0, 0]

raw_files = ['train.csv', 'test.csv', 'val.csv']
processed_files = ['train.pt', 'test.pt', 'val.pt']

class Dataset(InMemoryDataset):
    def __init__(self, stage, root, size_test_splits=0, add_supernode_edges=True, 
                 with_explicit_h=False, with_formal_charge=False):
        self.stage = stage
        self.root = root
        self.size_test_splits = size_test_splits
        self.add_supernode_edges = add_supernode_edges
        self.with_explicit_h = with_explicit_h
        self.with_formal_charge = with_formal_charge
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
        graphs = []
        for i, rxn_ in enumerate(open(self.raw_paths[self.file_idx], 'r')):
            reactants = [r for r in rxn_.split('>>')[0].split('.')]
            products = [p for p in rxn_.split('>>')[1].split('.')]
            offset = 0 
            # mask: (n), with n = nb of nodes
            mask_product_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only reactant nodes = True
            mask_reactant_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool)
            mask_sn = torch.ones(MAX_ATOMS_RXN, dtype=torch.bool) # only sn = False
            mask_atom_mapping = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)

            supernodes_rcts = []
            for j, r in enumerate(reactants):
                gi_nodes, gi_edge_index, gi_edge_attr, atom_map = mol.rxn_to_graph_supernode(mol=r, atom_types=atom_types, 
                                                                                             bond_types=bond_types, supernode_nb=offset+1,
                                                                                             add_supernode_edges=self.add_supernode_edges,
                                                                                             with_explicit_h=self.with_explicit_h, 
                                                                                             with_formal_charge=self.with_formal_charge,
                                                                                             get_atom_mapping=True)
                
                g_nodes = torch.cat((g_nodes, gi_nodes), dim=0) if j > 0 else gi_nodes # already a tensor
                g_edge_index = torch.cat((g_edge_index, gi_edge_index), dim=1) if j > 0 else gi_edge_index
                g_edge_attr = torch.cat((g_edge_attr, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr

                mask_product_and_sn[offset+1:gi_nodes.shape[0]+offset] = True # start from offset+1 because sn at offset
                atom_mapped_idx = (atom_map!=0).nonzero()
                mask_atom_mapping[atom_mapped_idx+offset] = atom_map[atom_mapped_idx]
                si = len(g_nodes) - len(gi_nodes) # gi_edge_index[0][0].item()
                supernodes_rcts.append(si)
                offset += gi_nodes.shape[0] 
                mask_sn[si] = False

            supernodes_prods = []
            for p in products:
                gi_nodes, gi_edge_index, gi_edge_attr = mol.rxn_to_graph_supernode(mol=p, atom_types=atom_types, 
                                                                                   bond_types=bond_types, supernode_nb=offset+1,
                                                                                   with_explicit_h=self.with_explicit_h, 
                                                                                   with_formal_charge=self.with_formal_charge,
                                                                                   add_supernode_edges=self.add_supernode_edges)

                g_nodes = torch.cat((g_nodes, gi_nodes), dim=0) 
                g_edge_index = torch.cat((g_edge_index, gi_edge_index), dim=1)
                g_edge_attr = torch.cat((g_edge_attr, gi_edge_attr), dim=0)
                
                mask_reactant_and_sn[offset+1:gi_nodes.shape[0]+offset] = True # start from offset+1 because sn at offset

                si = len(g_nodes) - len(gi_nodes) # si = gi_edge_index[0][0].item()
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

            edge_type = torch.full(size=(nb_within_edges,), fill_value=bond_types.index('within'), dtype=torch.long) 
            within_edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)).float() # add 1 to len of bonds to account for no edge
        
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

            edge_type = torch.full(size=(nb_across_edges,), fill_value=bond_types.index('across'), dtype=torch.long)
            across_edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)).float() # add 1 to len of bonds to account for no edge

            g_edge_index = torch.cat((g_edge_index, within_edge_index, across_edge_index), dim=1)
            g_edge_attr = torch.cat((g_edge_attr, within_edge_attr, across_edge_attr), dim=0)

            y = torch.zeros((1, 0), dtype=torch.float)
            
            # trim masks => one element per node in the rxn graph
            mask_product_and_sn = mask_product_and_sn[:g_nodes.shape[0]] # only reactant nodes = True
            mask_reactant_and_sn = mask_reactant_and_sn[:g_nodes.shape[0]]
            mask_sn = mask_sn[:g_nodes.shape[0]]
            mask_atom_mapping = mask_atom_mapping[:g_nodes.shape[0]]

            assert mask_atom_mapping.shape[0]==g_nodes.shape[0] and mask_sn.shape[0]==g_nodes.shape[0] and \
                   mask_reactant_and_sn.shape[0]==g_nodes.shape[0] and \
                   mask_product_and_sn.shape[0]==g_nodes.shape[0], 'Wrong shapes for masks during data processing.\n'

            graph = Data(x=g_nodes, edge_index=g_edge_index, 
                         edge_attr=g_edge_attr, y=y, idx=i,
                         mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn, 
                         mask_product_and_sn=mask_product_and_sn, mask_atom_mapping=mask_atom_mapping)

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
        self.add_supernode_edges = cfg.dataset.add_supernode_edges
        if not self.add_supernode_edges:
            self.datadir += '_no_sn_edges'
            self.datadist_dir += '_no_sn_edges'
        if cfg.dataset.dataset_nb!='':
            self.datadir += '-'+str(cfg.dataset.dataset_nb)
            self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        super().__init__(cfg)
    
    def prepare_data(self, shuffle=True, seed=0) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': Dataset(stage='train', root=root_path, add_supernode_edges=self.add_supernode_edges, 
                                     with_explicit_h=self.with_explicit_h, with_formal_charge=self.with_formal_charge),
                    'val': Dataset(stage='val', root=root_path, add_supernode_edges=self.add_supernode_edges, 
                                   with_explicit_h=self.with_explicit_h, with_formal_charge=self.with_formal_charge),
                    'test': Dataset(stage='test', root=root_path, size_test_splits=self.cfg.test.size_test_splits, 
                                    add_supernode_edges=self.add_supernode_edges, with_explicit_h=self.with_explicit_h, 
                                    with_formal_charge=self.with_formal_charge)}
        super().prepare_data(datasets, shuffle=shuffle, seed=seed)

    def node_counts(self, max_nodes_possible=MAX_ATOMS_RXN):
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
        suno_idx = atom_types.index('SuNo')
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
    def __init__(self, datamodule, recompute_info=False, remove_h=True):
        self.datamodule = datamodule
        self.name = 'supernode_graphs'
        # self.atom_encoder = ['none']+atom_types # takes type (str) get idx (int)
        self.atom_decoder = atom_types
        self.bond_decoder = bond_types
        self.remove_h = remove_h
        self.valencies = valencies
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

    def compute_input_output_dims(self, datamodule):
        example_batch = next(iter(datamodule.train_dataloader()))

        self.input_dims = {'X': example_batch['x'].size(1), # n or dx?
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}  # + 1 due to time conditioning

        self.output_dims = {'X': example_batch['x'].size(1), # output dim = # of features
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}


            
