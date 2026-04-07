import os
import os.path as osp
import pathlib
from dataclasses import dataclass
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
from torch_geometric.loader import DataLoader

from diffalign.utils import graph, mol, setup
from diffalign.utils.graph_builder import build_rxn_graph
from diffalign.datasets.abstract_dataset import DistributionNodes
from diffalign.utils.rdkit import mol2smiles, build_molecule_with_partial_charges
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

            nb_product_nodes = sum([len(Chem.MolFromSmiles(p).GetAtoms()) for p in products])
            nb_rct_nodes = sum([len(Chem.MolFromSmiles(r).GetAtoms()) for r in reactants])
            nb_dummy_toadd = nb_product_nodes + self.max_nodes_more_than_product - nb_rct_nodes
            if nb_dummy_toadd < 0 and self.stage == 'train':
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


def _get_datadir(cfg):
    """Compute the data directory from config, handling dataset_nb suffix."""
    datadir = cfg.dataset.datadir
    if cfg.dataset.dataset_nb != '':
        datadir += '-' + str(cfg.dataset.dataset_nb)
    return datadir


def create_dataloaders(cfg, slices=None):
    """Create train/val/test DataLoaders directly from config.

    Args:
        cfg: Hydra config.
        slices: Optional dict like {'train': None, 'val': [0, 100], 'test': None}
                to take subsets of splits.
    Returns:
        dict mapping split name to DataLoader.
    """
    if slices is None:
        slices = {'train': None, 'val': None, 'test': None}

    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    root_path = os.path.join(base_path, _get_datadir(cfg))

    common = dict(
        atom_types=cfg.dataset.atom_types,
        with_explicit_h=cfg.dataset.with_explicit_h,
        with_formal_charge=cfg.dataset.with_formal_charge,
        max_nodes_more_than_product=cfg.dataset.nb_rct_dummy_nodes,
        canonicalize_molecule=cfg.dataset.canonicalize_molecule,
        add_supernode_edges=cfg.dataset.add_supernode_edges,
        permute_mols=cfg.dataset.permute_mols,
    )

    datasets = {
        'train': Dataset(stage='train', root=root_path, **common),
        'val': Dataset(stage='val', root=root_path, **common),
        'test': Dataset(stage='test', root=root_path, size_test_splits=cfg.test.size_test_splits, **common),
    }

    for key in slices:
        if slices[key] is not None:
            datasets[key] = datasets[key][slices[key][0]:slices[key][1]]

    train_bs = cfg.train.batch_size
    test_bs = cfg.test.batch_size
    batch_sizes = {'train': train_bs, 'val': test_bs, 'test': test_bs}
    num_workers = cfg.dataset.num_workers

    return {
        split: DataLoader(ds, batch_size=batch_sizes[split],
                          num_workers=num_workers,
                          shuffle=(split == 'train'))
        for split, ds in datasets.items()
    }


@dataclass
class DatasetInfos:
    """All dataset metadata needed by the model."""
    atom_decoder: list
    bond_decoder: list
    valencies: list
    atom_weights: dict
    max_weight: int
    bond_orders: list
    remove_h: bool
    n_nodes: torch.Tensor
    node_types: torch.Tensor
    edge_types: torch.Tensor
    node_types_unnormalized: torch.Tensor
    edge_types_unnormalized: torch.Tensor
    max_n_nodes: int
    nodes_dist: DistributionNodes
    input_dims: dict
    output_dims: dict


def build_dataset_info(cfg, datadist_dir=None):
    """Build DatasetInfos from config and cached statistics files.

    Args:
        cfg: Hydra config.
        datadist_dir: Override for the data distribution directory.
                      If None, computed from cfg.dataset.datadist_dir.
    """
    atom_types = cfg.dataset.atom_types
    allowed_bonds = cfg.dataset.allowed_bonds

    # Static config
    valencies = [0] + list(abs_[0] for atom_type, abs_ in allowed_bonds.items() if atom_type in atom_types) + [0]
    periodic_table = Chem.rdchem.GetPeriodicTable()
    raw_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', at)[0]) for at in atom_types[1:-1]] + [0]
    atom_weights = {at: w for at, w in zip(atom_types, raw_weights)}

    # Resolve datadist_dir
    if datadist_dir is None:
        datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.dataset_nb != '':
            datadist_dir += '-' + str(cfg.dataset.dataset_nb)

    # Load cached statistics
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    root_path = os.path.join(base_path, datadist_dir, 'processed')

    n_nodes = torch.from_numpy(np.loadtxt(os.path.join(root_path, 'n_counts.txt')))
    node_types = torch.from_numpy(np.loadtxt(os.path.join(root_path, 'atom_types.txt')))
    edge_types = torch.from_numpy(np.loadtxt(os.path.join(root_path, 'edge_types.txt')))
    node_types_unnormalized = torch.from_numpy(np.loadtxt(os.path.join(root_path, 'atom_types_unnorm_mol.txt'))).long()
    edge_types_unnormalized = torch.from_numpy(np.loadtxt(os.path.join(root_path, 'edge_types_unnorm_mol.txt'))).long()

    # Dims are deterministic from config
    dx = len(atom_types)
    de = len(bond_types)

    return DatasetInfos(
        atom_decoder=atom_types,
        bond_decoder=bond_types,
        valencies=valencies,
        atom_weights=atom_weights,
        max_weight=390,
        bond_orders=bond_orders,
        remove_h=cfg.dataset.remove_h,
        n_nodes=n_nodes,
        node_types=node_types,
        edge_types=edge_types,
        node_types_unnormalized=node_types_unnormalized,
        edge_types_unnormalized=edge_types_unnormalized,
        max_n_nodes=len(n_nodes) - 1,
        nodes_dist=DistributionNodes(n_nodes),
        input_dims={'X': dx, 'E': de, 'y': 1},
        output_dims={'X': dx, 'E': de, 'y': 0},
    )


def compute_dataset_statistics(dataloaders, atom_types, datadist_dir):
    """Compute and save dataset statistics from dataloaders. Only needed during data preprocessing."""
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    root_path = os.path.join(base_path, datadist_dir, 'processed')

    # Node counts
    all_counts = torch.zeros(300)
    for split in ['train', 'val', 'test']:
        for data in dataloaders[split]:
            batch_without_sn = data.batch[data.mask_sn]
            unique, counts = torch.unique(batch_without_sn, return_counts=True)
            for count in counts:
                all_counts[count] += 1
    max_index = max(all_counts.nonzero())
    all_counts = all_counts[:max_index + 1]
    n_nodes = all_counts / all_counts.sum()
    np.savetxt(os.path.join(root_path, 'n_counts.txt'), n_nodes.cpu().numpy())

    # Node types unnormalized
    data = next(iter(dataloaders['train']))
    num_classes = data.x.shape[1]
    node_counts = torch.zeros(num_classes)
    for data in dataloaders['train']:
        node_counts += data.x.sum(dim=0)
    suno_idx = atom_types.index('SuNo')
    node_counts[suno_idx] = 0.
    np.savetxt(os.path.join(root_path, 'atom_types_unnorm_mol.txt'), node_counts.long().cpu().numpy())

    node_types = node_counts / node_counts.sum()
    np.savetxt(os.path.join(root_path, 'atom_types.txt'), node_types.cpu().numpy())

    # Edge types unnormalized
    data = next(iter(dataloaders['train']))
    num_edge_classes = data.edge_attr.shape[1]
    d = torch.zeros(num_edge_classes)
    for data in dataloaders['train']:
        _, counts = torch.unique(data.batch, return_counts=True)
        all_pairs = sum(c * (c - 1) for c in counts)
        num_edges = data.edge_index.shape[1]
        edge_type_counts = data.edge_attr.sum(dim=0)
        d[0] += all_pairs - num_edges
        d[1:] += edge_type_counts[1:]
    for t in ['mol', 'within', 'across']:
        suno_idx = bond_types.index(t)
        d[suno_idx] = 0.
    np.savetxt(os.path.join(root_path, 'edge_types_unnorm_mol.txt'), d.long().cpu().numpy())

    edge_types = d / d.sum()
    np.savetxt(os.path.join(root_path, 'edge_types.txt'), edge_types.cpu().numpy())
