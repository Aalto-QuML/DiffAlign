import os
import pathlib
import torch
import re

from rdkit import Chem
import numpy as np
import pickle
from rdkit.Chem.rdchem import BondType as BT

from torch_geometric.data import Data, InMemoryDataset

from src.utils import graph, mol, setup
from src.datasets.abstract_dataset import AbstractDataModule, seed_worker, DistributionNodes

# size_bins = { # [ 1.  10.6 20.2 29.8 39.4 49.  (58.6 68.2 77.8 87.4 97.)]
#              'train': [1, 11, 21, 30, 40, 50],
#              # [ 1.   8.9 16.8 24.7 32.6 40.5 (48.4 56.3 64.2 72.1 80.)]
#              'test': [1, 9, 17, 25, 33, 41],
#              # [ 1.   8.1 15.2 22.3 29.4 36.5 (43.6 50.7 57.8 64.9 72.)]
#              'val': [1, 9, 16, 23, 30, 37]}

# batchsize_bins = { # [21725. 38317. 26596. 16702.  3590.   (805.   142.    25.    15.     3.)]
#                   # 'train': [1024, 1024, 1024, 1024, 512, 128],
#                   'train': [32, 32, 32, 32, 16, 16],
#                   # [1.727e+03 4.044e+03 3.369e+03 2.493e+03 1.332e+03 (3.410e+02 9.800e+01 1.900e+01 8.000e+00 2.000e+00)]
#                   'test': [128, 256, 256, 256, 128, 128], 
#                   # fixed batchsize for subset test files => CHANGE
#                   'test_i': [10, 10, 10, 10, 10, 10],  
#                   # [1738. 3430. 3158. 2480. 1684.  (675.  177.   55.   19.   15.)]
#                   'val': [128, 256, 256, 256, 128, 128]}

atom_types = ['none', 'O', 'C', 'N', 'I', 'H', 'Cl', 'Si', 'F', 'Br', 'N+1', 'O-1', 'S', 'B', 'N-1', 
              'Zn+1', 'Cu', 'Sn', 'P+1', 'Mg+1', 'C-1', 'P', 'S+1', 'S-1', 'Se', 'Zn', 'Mg', 'Au']

allowed_bonds = {'O': [2], 'C': [4], 'N': [3], 'I': [1, 3, 5, 7], 'H': [1], 'Cl': [1], 'Si': [4, 6], 'F': [1],
                 'Br': [1], 'N+1': [4], 'O-1': [1], 'S': [2, 4, 6], 'B': [3], 'N-1': [2], 'Zn+1': [3], 'Cu': [1, 2], 
                 'Sn': [2, 4], 'P+1': [4, 6, 8], 'Mg+1': [3], 'C-1': [3], 'P': [3, 5, 7], 'S+1': [3, 5, 7], 'S-1': [1, 3, 5], 
                 'Se': [2, 4, 6], 'Zn': [2], 'Mg': [2], 'Au': [0]}

valencies = [0] + list(abs[0] for abs in allowed_bonds.values())

periodic_table = Chem.rdchem.GetPeriodicTable()
atom_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in atom_types[1:-1]] # discard charge
atom_weights = {atom_type: weight for atom_type, weight in zip(atom_types, atom_weights)}

bond_types = ['none', BT.SINGLE, BT.DOUBLE, BT.TRIPLE]
bond_orders = [0, 1, 2, 3]

raw_files = ['train.csv', 'test.csv', 'val.csv']
processed_files = ['train_mols.pt', 'test_mols.pt', 'val_mols.pt']

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
            test_path = os.path.join(root, 'processed', f"{self.stage.split('_')[0]}_mols_{self.stage.split('_')[1]}.pt")
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

    def split_test_data(self, graphs):
        '''
            (optional) Split data test file to smaller chunks to be used in parallel while testing (e.g. in slurm array jobs).

            Input:
                graphs: a list of processed graph objects (test data)
                size_test_splits: size of one test split to generate. Usually set in the config file.
            Output:
                size_test_splits saved to the dataset's processed directory.
        '''
        filepath = self.processed_paths[self.file_idx].split('.pt')[0]
        for i in range(0, len(graphs), self.size_test_splits):
            torch.save(self.collate(graphs[i:i+self.size_test_splits ]), filepath+f'_{int(i/self.size_test_splits)}.pt')
    
    def get_molecules(self):
        in_path = self.raw_paths[self.file_idx]
        out_path = self.raw_paths[self.file_idx].split('.csv')[0]+'mols.csv'

        reactants, products = [], []
        for l in open(in_path, 'r').readlines():
            rxn = l.strip()
            reactants.extend(rxn.split('>>')[0].split('.'))
            products.extend(rxn.split('>>')[1].split('.'))
        
        open(out_path, 'w').writelines(f'{m}\n' for m in set(reactants+products))

        return out_path

    def process(self):
        mol_path = self.raw_paths[self.file_idx].split('.csv')[0]+'mols.csv'
        if not os.path.exists(mol_path):
            mol_path = self.get_molecules()
        graphs = []
        for i, smiles in enumerate(open(mol_path, 'r')):
            mol_ = Chem.MolFromSmiles(smiles.strip())
            nodes, edge_index, edge_attr = mol.mol_to_graph(mol_, atom_types=atom_types, 
                                                                      bond_types=bond_types, offset=0)
            y = torch.zeros((1,0), dtype=torch.float)
            graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
            graphs.append(graph)

        if self.stage=='test' and self.size_test_splits>0: self.split_test_data(graphs)
        list_path = os.path.join('/'.join(self.processed_paths[self.file_idx].split('/')[:-1]), self.stage+'.pickle')
        pickle.dump(graphs, open(list_path, 'wb'))
        torch.save(self.collate(graphs), self.processed_paths[self.file_idx])

class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.with_explicit_h = cfg.dataset.with_explicit_h
        self.with_formal_charge = cfg.dataset.with_formal_charge
        self.datadir = cfg.dataset.datadir
        self.datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.dataset_nb!='':
            self.datadir += '-'+str(cfg.dataset.dataset_nb)
            self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        super().__init__(cfg)
    
    def prepare_data(self, shuffle=True, seed=0):
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        print(root_path)
        datasets = {'train': Dataset(stage='train', root=root_path),
                    'val': Dataset(stage='val', root=root_path),
                    'test': Dataset(stage='test', root=root_path, size_test_splits=self.cfg.test.size_test_splits)}
        
        super().prepare_data(datasets, shuffle=shuffle, seed=seed)

    def node_counts(self, max_nodes_possible=300):
        #TODO: Can this be abstracted to the AbstractDataModule class?
        '''
            Return distribution over the number of atoms in molecules.

            Input:
                max_nodes_possible (optional): hypothetical maximum number of atoms in a molecule.
            Output:
                all_counts: distribution over number of atoms in molecules.
        '''
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                _, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1

        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index+1]
        all_counts = all_counts/all_counts.sum()
        
        return all_counts

    def node_types(self):
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

        counts = counts/counts.sum()

        return counts
    
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

        return counts.long()

    def edge_types(self):
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

        d = d/d.sum() 

        return d
    
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

        return d.long()

class DatasetInfos:
    #TODO: Can this be mostly abstracted to the AbstractDataModule class?
    # (maybe, e.g., the name of the dataset is not abstractable, but otherwise it seems like mostly this could be)
    def __init__(self, datamodule, recompute_info=False, remove_h=True):
        self.datamodule = datamodule
        self.name = 'uspto50k-mols'
        self.atom_decoder = atom_types
        self.bond_decoder = bond_types
        self.remove_h = remove_h
        self.valencies = valencies
        self.atom_weights = atom_weights
        self.max_weight = 390
        self.bond_orders = bond_orders
        
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, datamodule.datadist_dir, 'processed')
        node_count_path = os.path.join(root_path, 'n_counts_mol.txt')
        atom_type_path = os.path.join(root_path, 'atom_types_mol.txt')
        edge_type_path = os.path.join(root_path, 'edge_types_mol.txt')
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

        self.input_dims = {'X': example_batch['x'].size(1), # dx
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1)+1}  # + 1 due to time conditioning

        self.output_dims = {'X': example_batch['x'].size(1), # output dim = # of features
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}

