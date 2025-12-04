import copy
import numpy as np
import random 
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import rdChemReactions
from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import Counter

from collections import defaultdict

import pickle
import math
import os
import gzip
from diffalign.data import *

import logging
log = logging.getLogger(__name__)

from diffalign.data import graph
from diffalign.data import mol
from diffalign.helpers import PROJECT_ROOT
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops, dense_to_sparse
# from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import concurrent.futures

import signal

torch.multiprocessing.set_sharing_strategy('file_system')

from diffalign.data import mol

# bond_types = ['none', BT.SINGLE, BT.DOUBLE, BT.TRIPLE, 'mol', 'within', 'across']

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ATOMS_RXN = 300
DUMMY_RCT_NODE_TYPE = 'U'

# def timeout_wrapper(func, args=(), kwargs={}, timeout_duration=10):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(func, *args, **kwargs)
#         try:
#             return future.result(timeout=timeout_duration)
#         except concurrent.futures.TimeoutError:
#             print(f"Function {func.__name__} timed out after {timeout_duration} seconds")
#             return None  # or any other default value or special indicator

import multiprocessing
from multiprocessing import Pool

# Register an handler for the timeout
def handler(signum, frame):
    print("Forever is over!")
    raise Exception("end of time")

# def timeout_wrapper(func, args=(), kwargs={}, timeout_duration=10):
#     def wrapper(queue):
#         try:
#             result = func(*args, **kwargs)
#             queue.put(result)
#         except Exception as e:
#             queue.put(e)

#     queue = multiprocessing.Queue()
#     process = multiprocessing.Process(target=wrapper, args=(queue,))
#     process.start()

#     try:
#         result = queue.get(block=True, timeout=timeout_duration)
#         if isinstance(result, Exception):
#             raise result
#         return result
#     except multiprocessing.queues.Empty:
#         print(f"Function {func.__name__} timed out after {timeout_duration} seconds")
#         process.terminate()
#         process.join()
#         return None  # or any other default value or special indicator
#     finally:
#         if process.is_alive():
#             process.terminate()
#             process.join()

from multiprocessing import get_context
import time

def get_index_from_states(atom_decoder, bond_decoder, node_states_to_mask, edge_states_to_mask, device):
    if not 'none' in node_states_to_mask:
        node_states_to_mask.append('none')
    if node_states_to_mask!=None:
        not_in_list = [a for a in node_states_to_mask if a not in atom_decoder]
        assert len(not_in_list)==0, f'node_states_to_mask contains states not in atom_decoder: {not_in_list}'
        idx_of_states = [atom_decoder.index(a) for a in node_states_to_mask]
        node_idx_to_mask = torch.tensor(idx_of_states, dtype=torch.long, device=device)
    else:
        node_idx_to_mask = torch.empty((1,), dtype=torch.long, device=device)
        
    if edge_states_to_mask!=None:
        not_in_list = [a for a in edge_states_to_mask if a not in bond_decoder]
        assert len(not_in_list)==0, f'edge_states_to_mask contains states not in bond_decoder: {not_in_list}'
        idx_of_states = [bond_decoder.index(a) for a in edge_states_to_mask]
        edge_idx_to_mask = torch.tensor(idx_of_states, dtype=torch.long, device=device)  
    # don't mask none in edges, used as masking state
    else:
        #edge_idx_to_mask = torch.empty((1,), dtype=torch.long, device=device)
        edge_idx_to_mask = None
     
    return node_idx_to_mask, edge_idx_to_mask

# Permute the rows here to make sure that the NN can only process topological information
def permute_rows(nodes, mask_atom_mapping, mol_assignment, edge_index):
    # Permutes the graph specified by nodes, mask_atom_mapping, mol_assignment and edge_index
    # nodes: (n,d_x) node feature tensor
    # mask_atom_mapping (n,) tensor
    # mol_assignment: (n,) tensor
    # edge_index: (2,num_edges) tensor
    # does everything in-place
    rct_section_len = nodes.shape[0]
    perm = torch.randperm(rct_section_len)
    nodes[:] = nodes[perm]
    mask_atom_mapping[:rct_section_len] = mask_atom_mapping[:rct_section_len][perm]
    mol_assignment[:rct_section_len] = mol_assignment[:rct_section_len][perm]
    inv_perm = torch.zeros(rct_section_len, dtype=torch.long)
    inv_perm.scatter_(dim=0, index=perm, src=torch.arange(rct_section_len))
    edge_index[:] = inv_perm[edge_index]

def timeout_wrapper_OLD(func, args=(), kwargs={}, timeout_duration=10):
    def wrapper(result_queue, func, args, kwargs):
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            result_queue.put(e)

    ctx = get_context('fork')
    result_queue = ctx.Queue()
    process = ctx.Process(target=wrapper, args=(result_queue, func, args, kwargs))
    process.start()

    try:
        result = result_queue.get(timeout=timeout_duration)
        if isinstance(result, Exception):
            raise result
        return result
    except multiprocessing.queues.Empty:
        print(f"Function {func.__name__} timed out after {timeout_duration} seconds")
        return None  # or any other default value or special indicator
    finally:
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()

def turn_reactants_and_product_smiles_into_graphs(cfg, reactants, products, data_idx, stage='test'):
    nb_product_nodes = sum([len(Chem.MolFromSmiles(p.strip()).GetAtoms()) for p in products])
    nb_rct_nodes = sum([len(Chem.MolFromSmiles(r.strip()).GetAtoms()) for r in reactants])

    nb_dummy_toadd_to_rcts = nb_product_nodes + cfg.dataset.nb_rct_dummy_nodes - nb_rct_nodes
    if nb_dummy_toadd_to_rcts<0 and stage=='train':
        log.info(f'dropping rxn {data_idx} in {stage} set')
        return None

    offset = 0
    for j, r in enumerate(reactants):
        nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = mol.smiles_to_graph_with_stereochem(smi=r, cfg=cfg)
        edge_index += offset
        nodes_rct = torch.cat((nodes_rct, nodes), dim=0) if j > 0 else nodes # already a tensor
        edge_index_rct = torch.cat((edge_index_rct, edge_index), dim=1) if j > 0 else edge_index
        bond_types_rct = torch.cat((bond_types_rct, bond_types), dim=0) if j > 0 else bond_types
        atom_map_reactants = torch.cat((atom_map_reactants, atom_map), dim=0) if j > 0 else atom_map
        atom_charges_rct = torch.cat((atom_charges_rct, atom_charges), dim=0) if j > 0 else atom_charges
        atom_chiral_rct = torch.cat((atom_chiral_rct, atom_chiral), dim=0) if j > 0 else atom_chiral
        bond_dirs_rct = torch.cat((bond_dirs_rct, bond_dirs), dim=0) if j > 0 else bond_dirs
        mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.ones(nodes.shape[0], dtype=torch.long) * j+1], dim=0) if j > 0 else torch.ones(nodes.shape[0], dtype=torch.long) * j+1
        offset += nodes.shape[0]

    if nb_dummy_toadd_to_rcts > 0:
        nodes_dummy = torch.ones(nb_dummy_toadd_to_rcts, dtype=torch.long) * cfg.dataset.atom_types.index(DUMMY_RCT_NODE_TYPE)
        nodes_dummy = F.one_hot(nodes_dummy, num_classes=len(cfg.dataset.atom_types)).float() # This is hardcoded
        edges_idx_dummy = torch.zeros([2, 0], dtype=torch.long)
        bond_types_dummy = torch.zeros([0, len(cfg.dataset.bond_types)], dtype=torch.long)
        nodes_rct = torch.cat([nodes_rct, nodes_dummy], dim=0)
        edge_index_rct = torch.cat([edge_index_rct, edges_idx_dummy], dim=1)
        bond_types_rct = torch.cat([bond_types_rct, bond_types_dummy], dim=0)
        atom_charges_rct = torch.cat([atom_charges_rct, F.one_hot(torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long), num_classes=len(cfg.dataset.atom_charges))], dim=0)
        atom_chiral_rct = torch.cat([atom_chiral_rct, F.one_hot(torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long), num_classes=len(cfg.dataset.atom_chiral_tags))], dim=0)
        bond_dirs_rct = torch.cat([bond_dirs_rct, torch.zeros([0, len(cfg.dataset.bond_dirs)], dtype=torch.long)], dim=0)
        atom_map_reactants = torch.cat([atom_map_reactants, torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long)], dim=0)
        mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long)], dim=0)

    # Permute the rows here to make sure that the NN can only process topological information
    if cfg.dataset.permute_mols:
        permute_rows(nodes_rct, atom_map_reactants, mol_assignment_reactants, edge_index_rct)

    offset = 0
    for j, p in enumerate(products):
        # get the num of atoms in molecule
        # NOTE: set dataset.max_atoms_rxn_parse to None if you want to parse all reactions regardless of size
        nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map  = mol.smiles_to_graph_with_stereochem(smi=p, cfg=cfg)
        nodes_prod = torch.cat((nodes_prod, nodes), dim=0) if j > 0 else nodes # already a tensor
        edge_index_prod = torch.cat((edge_index_prod, edge_index), dim=1) if j > 0 else edge_index
        bond_types_prod = torch.cat((bond_types_prod, bond_types), dim=0) if j > 0 else bond_types
        atom_charges_prod = torch.cat((atom_charges_prod, atom_charges), dim=0) if j > 0 else atom_charges
        atom_chiral_prod = torch.cat((atom_chiral_prod, atom_chiral), dim=0) if j > 0 else atom_chiral
        bond_dirs_prod = torch.cat((bond_dirs_prod, bond_dirs), dim=0) if j > 0 else bond_dirs
        atom_map_products = torch.cat((atom_map_products, atom_map), dim=0) if j > 0 else atom_map
        mol_assignment_products = torch.cat([mol_assignment_products, torch.ones(nodes.shape[0], dtype=torch.long) * len(reactants)+j+1], dim=0) \
                                    if j > 0 else torch.ones(nodes.shape[0], dtype=torch.long) * len(reactants)+j+1
        offset += nodes.shape[0]

    y = torch.zeros((1, 0), dtype=torch.float)

    # Check that the one-hot encodings are correct
    assert (bond_types_rct.sum(-1) == 1).all() # Should be one-hot encodings here
    assert (bond_types_prod.sum(-1) == 1).all() # TODO: REPLACE WITH .all() for efficiency
    assert (nodes_rct.sum(-1) == 1).all()
    assert (nodes_prod.sum(-1) == 1).all()

    # Make sure that there are no duplicate edges
    assert len(set([(edge[0].item(), edge[1].item()) for edge in edge_index_rct.T])) == edge_index_rct.shape[1]
    assert len(set([(edge[0].item(), edge[1].item()) for edge in edge_index_prod.T])) == edge_index_prod.shape[1]

    # Make sure that there are no edges pointing to nodes that don't exist
    assert (edge_index_rct < nodes_rct.shape[0]).flatten().all()
    assert (edge_index_prod < nodes_prod.shape[0]).flatten().all()

    smiles_to_save = ".".join(reactants) + ">>" + ".".join(products) #data_utils.remove_atom_mapping_from_reaction(".".join(reactants) + ">>" + ".".join(products))

    graph = Data(x=torch.cat([nodes_rct, nodes_prod], dim=0), 
                 edge_index=torch.cat([edge_index_rct, edge_index_prod + len(nodes_rct)], dim=1),
                 edge_attr=torch.cat([bond_types_rct, bond_types_prod], dim=0), y=y, idx=data_idx,
                 mol_assignment=torch.cat([mol_assignment_reactants, mol_assignment_products], dim=0),
                 atom_map_numbers=torch.cat([atom_map_reactants, atom_map_products], dim=0),
                 smiles=smiles_to_save,
                 atom_charges=torch.cat([atom_charges_rct, atom_charges_prod], dim=0),
                 atom_chiral=torch.cat([atom_chiral_rct, atom_chiral_prod], dim=0),
                 bond_dirs=torch.cat([bond_dirs_rct, bond_dirs_prod], dim=0))
    
    return graph


def turn_reactants_and_product_smiles_into_graphs_for_all_datasets(cfg, reactants, products, data_idx, stage='test'):
    # preprocess: get total number of product nodes
    nb_product_nodes = sum([len(Chem.MolFromSmiles(p.strip()).GetAtoms()) for p in products])
    # if cfg.dataset.max_atoms_rxn_parse and nb_product_nodes > cfg.dataset.max_atoms_rxn_parse \
    #     or (data_idx == 37435 and stage == 'test') or (data_idx == 22827 and stage == 'val'):
    if cfg.dataset.max_atoms_rxn_parse and nb_product_nodes > cfg.dataset.max_atoms_rxn_parse:
        products = ['C']
        nb_product_nodes = 1
    nb_rct_nodes = sum([len(Chem.MolFromSmiles(r.strip()).GetAtoms()) for r in reactants])
    if cfg.dataset.max_atoms_rxn_parse and nb_rct_nodes > cfg.dataset.max_atoms_rxn_parse:
        reactants = ['O']
        nb_rct_nodes = 1

    nb_dummy_toadd_to_rcts = nb_product_nodes + cfg.dataset.nb_rct_dummy_nodes - nb_rct_nodes
    #print(f'nb_dummy_toadd_to_rcts={nb_dummy_toadd_to_rcts}')
    if nb_dummy_toadd_to_rcts<0 and stage=='train':
    # if nb_dummy_toadd_to_rcts<0 and stage=='train' \
    #     or (data_idx==45028 and stage=='train') \
    #     or (data_idx==497883 and stage=='train') \
    #     or (data_idx==528508 and stage=='train'): # add indices to ignore here
        # drop the rxns in the training set which we cannot generate
        log.info(f'dropping rxn {data_idx} in {stage} set')
        return None

    offset = 0
    for j, r in enumerate(reactants):
        # If the processing gets stuck (e.g., because of canonicalization getting stuck on really weird reactions), don't let it run forever
        #timeout_res = timeout_wrapper(mol.smiles_to_graph_with_stereochem, args=(r, cfg), timeout_duration=40)
        # NOTE: set dataset.max_atoms_rxn_parse to None if you want to parse all reactions regardless of size

        nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = mol.smiles_to_graph_with_stereochem(smi=r, cfg=cfg)
        # nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = mol.smiles_to_graph_with_stereochem(smi=r, cfg=cfg)
        edge_index += offset
        nodes_rct = torch.cat((nodes_rct, nodes), dim=0) if j > 0 else nodes # already a tensor
        edge_index_rct = torch.cat((edge_index_rct, edge_index), dim=1) if j > 0 else edge_index
        bond_types_rct = torch.cat((bond_types_rct, bond_types), dim=0) if j > 0 else bond_types
        atom_map_reactants = torch.cat((atom_map_reactants, atom_map), dim=0) if j > 0 else atom_map
        atom_charges_rct = torch.cat((atom_charges_rct, atom_charges), dim=0) if j > 0 else atom_charges
        atom_chiral_rct = torch.cat((atom_chiral_rct, atom_chiral), dim=0) if j > 0 else atom_chiral
        bond_dirs_rct = torch.cat((bond_dirs_rct, bond_dirs), dim=0) if j > 0 else bond_dirs
        mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.ones(nodes.shape[0], dtype=torch.long) * j+1], dim=0) if j > 0 else torch.ones(nodes.shape[0], dtype=torch.long) * j+1
        offset += nodes.shape[0]

    if nb_dummy_toadd_to_rcts > 0:
        nodes_dummy = torch.ones(nb_dummy_toadd_to_rcts, dtype=torch.long) * cfg.dataset.atom_types.index(DUMMY_RCT_NODE_TYPE)
        nodes_dummy = F.one_hot(nodes_dummy, num_classes=len(cfg.dataset.atom_types)).float() # This is hardcoded
        edges_idx_dummy = torch.zeros([2, 0], dtype=torch.long)
        bond_types_dummy = torch.zeros([0, len(cfg.dataset.bond_types)], dtype=torch.long)
        nodes_rct = torch.cat([nodes_rct, nodes_dummy], dim=0)
        edge_index_rct = torch.cat([edge_index_rct, edges_idx_dummy], dim=1)
        bond_types_rct = torch.cat([bond_types_rct, bond_types_dummy], dim=0)
        atom_charges_rct = torch.cat([atom_charges_rct, F.one_hot(torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long), num_classes=len(cfg.dataset.atom_charges))], dim=0)
        atom_chiral_rct = torch.cat([atom_chiral_rct, F.one_hot(torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long), num_classes=len(cfg.dataset.atom_chiral_tags))], dim=0)
        bond_dirs_rct = torch.cat([bond_dirs_rct, torch.zeros([0, len(cfg.dataset.bond_dirs)], dtype=torch.long)], dim=0)
        atom_map_reactants = torch.cat([atom_map_reactants, torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long)], dim=0)
        mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long)], dim=0)

    # Permute the rows here to make sure that the NN can only process topological information
    if cfg.dataset.permute_mols:
        permute_rows(nodes_rct, atom_map_reactants, mol_assignment_reactants, edge_index_rct)

    offset = 0
    for j, p in enumerate(products):
        # get the num of atoms in molecule
        # NOTE: set dataset.max_atoms_rxn_parse to None if you want to parse all reactions regardless of size
        nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map  = mol.smiles_to_graph_with_stereochem(smi=r, cfg=cfg)
        nodes_prod = torch.cat((nodes_prod, nodes), dim=0) if j > 0 else nodes # already a tensor
        edge_index_prod = torch.cat((edge_index_prod, edge_index), dim=1) if j > 0 else edge_index
        bond_types_prod = torch.cat((bond_types_prod, bond_types), dim=0) if j > 0 else bond_types
        atom_charges_prod = torch.cat((atom_charges_prod, atom_charges), dim=0) if j > 0 else atom_charges
        atom_chiral_prod = torch.cat((atom_chiral_prod, atom_chiral), dim=0) if j > 0 else atom_chiral
        bond_dirs_prod = torch.cat((bond_dirs_prod, bond_dirs), dim=0) if j > 0 else bond_dirs
        atom_map_products = torch.cat((atom_map_products, atom_map), dim=0) if j > 0 else atom_map
        mol_assignment_products = torch.cat([mol_assignment_products, torch.ones(nodes.shape[0], dtype=torch.long) * len(reactants)+j+1], dim=0) if j > 0 else torch.ones(nodes.shape[0], dtype=torch.long) * len(reactants)+j+1
        offset += nodes.shape[0]

    y = torch.zeros((1, 0), dtype=torch.float)
    
    # TODO : Clean some of this stuff up, less asserts, and have them somewhere else
    if cfg.dataset.fix_atom_mappings:
        #if stage == 'train':
        #assert (atom_map_reactants != 0).sum() == (atom_map_products != 0).sum()
        print(f'=== Before fixing: atom_map_reactants={atom_map_reactants}, atom_map_products={atom_map_products}')
        atom_map_reactants, atom_map_products = data_utils.fix_atom_mappings(atom_map_reactants, atom_map_products)
        print(f'=== After fixing: atom_map_reactants={atom_map_reactants}, atom_map_products={atom_map_products}')
        equal_number_of_atom_map_numbers = (atom_map_reactants != 0).sum() == (atom_map_products != 0).sum()
        #print(f'atom_map_reactants={(atom_map_reactants != 0).sum()}, atom_map_products={(atom_map_products != 0).sum()}')
        unique_reactants_atom_map_numbers = len(set(atom_map_reactants.tolist()) - set([0])) == (atom_map_reactants != 0).sum().item()
        unique_product_atom_map_numbers = len(set(atom_map_products.tolist())) == (atom_map_products != 0).sum().item()
        if not equal_number_of_atom_map_numbers or not unique_reactants_atom_map_numbers or \
            not unique_product_atom_map_numbers:
                log.info(f'skipping rxn {data_idx} in train. equal_number_of_atom_map_numbers={equal_number_of_atom_map_numbers}, unique_reactants_atom_map_numbers={unique_reactants_atom_map_numbers}, unique_product_atom_map_numbers={unique_product_atom_map_numbers}\n')
                print(f'atom_map_reactants={atom_map_reactants}, atom_map_products={atom_map_products}')
                print(f'equal_number_of_atom_map_numbers={equal_number_of_atom_map_numbers}, unique_reactants_atom_map_numbers={unique_reactants_atom_map_numbers}, unique_product_atom_map_numbers={unique_product_atom_map_numbers}')
                return None  
        if set(atom_map_reactants.tolist()) - set([0]) != set(atom_map_products.tolist()) - set([0]):
            print("SOMETHING WRONG HERE")
            print(atom_map_reactants)
            print(atom_map_products)
        # else:
        #     if cfg.dataset.name != 'uspto-50k':
        #         atom_map_reactants, atom_map_products = data_utils.fix_atom_mappings(atom_map_reactants, atom_map_products) # Just use the same for now
        #         equal_number_of_atom_map_numbers = (atom_map_reactants != 0).sum() == (atom_map_products != 0).sum()
        #         unique_reactants_atom_map_numbers = len(set(atom_map_reactants.tolist()) - set([0])) == (atom_map_reactants != 0).sum().item()
        #         unique_product_atom_map_numbers = len(set(atom_map_products.tolist())) == (atom_map_products != 0).sum().item()
        #         if not equal_number_of_atom_map_numbers or not unique_reactants_atom_map_numbers or \
        #             not unique_product_atom_map_numbers:
        #                 log.info(f'skipping rxn {data_idx} in test/val\n')
        #                 return None 
            # else:
            #     # atom_map_reactants, atom_map_products = data_utils.fix_atom_mappings(atom_map_reactants, atom_map_products)
            #     atom_map_reactants, atom_map_products = data_utils.fix_atom_mappings(atom_map_reactants, atom_map_products)
            #     if set(atom_map_reactants.tolist()) - set([0]) != set(atom_map_products.tolist()) - set([0]):
            #         log.info("SOMETHING WRONG HERE")
            #         log.info(f'atom_map_reactants={atom_map_reactants}')
            #         log.info(f'atom_map_products={atom_map_products}')
            #         log.info(f'reactants={reactants}')
            #         log.info(f'products={products}')
            #     #assert (set(atom_map_reactants.tolist()) - set([0])) == (set(atom_map_products.tolist()) - set([0]))
            #     return None # TODO: fix this

        # Align the graphs here according to the atom mapping, so that alignment is not necessary afterwards
        if cfg.dataset.pre_align_graphs:
            perm = torch.cat([torch.tensor([0], dtype=torch.long), 1+torch.randperm(atom_map_reactants.max())])
            atom_map_reactants, indices = torch.sort(atom_map_reactants, dim=0, descending=True) # put the non-atom-mapped stuff as last
            inverse_indices = torch.argsort(indices) #torch.tensor([idx[0].item() for idx in sorted(zip(torch.arange(len(indices)), indices), key=lambda x: x[1])])
            nodes_rct = nodes_rct[indices]
            edge_index_rct = inverse_indices[edge_index_rct]
            mol_assignment_reactants = mol_assignment_reactants[indices]
            atom_chiral_rct = atom_chiral_rct[indices]
            atom_charges_rct = atom_charges_rct[indices]
            atom_map_products, indices = torch.sort(atom_map_products, dim=0, descending=True) # put the non-atom-mapped stuff as last
            inverse_indices = torch.argsort(indices)
            nodes_prod = nodes_prod[indices]
            edge_index_prod = inverse_indices[edge_index_prod]
            mol_assignment_products = mol_assignment_products[indices]
            atom_chiral_prod = atom_chiral_prod[indices]
            atom_charges_prod = atom_charges_prod[indices]

    # Check that the one-hot encodings are correct
    assert all(bond_types_rct.sum(-1) == 1) # Should be one-hot encodings here
    assert all(bond_types_prod.sum(-1) == 1) # TODO: REPLACE WITH .all() for efficiency
    assert all(nodes_rct.sum(-1) == 1)
    assert all(nodes_prod.sum(-1) == 1)

    # Make sure that there are no duplicate edges
    assert len(set([(edge[0].item(), edge[1].item()) for edge in edge_index_rct.T])) == edge_index_rct.shape[1]
    assert len(set([(edge[0].item(), edge[1].item()) for edge in edge_index_prod.T])) == edge_index_prod.shape[1]

    # Make sure that there are no edges pointing to nodes that don't exist
    assert all((edge_index_rct < nodes_rct.shape[0]).flatten())
    assert all((edge_index_prod < nodes_prod.shape[0]).flatten())

    smiles_to_save = ".".join(reactants) + ">>" + ".".join(products) #data_utils.remove_atom_mapping_from_reaction(".".join(reactants) + ">>" + ".".join(products))

    graph = Data(x=torch.cat([nodes_rct, nodes_prod], dim=0), 
                edge_index=torch.cat([edge_index_rct, edge_index_prod + len(nodes_rct)], dim=1),
                edge_attr=torch.cat([bond_types_rct, bond_types_prod], dim=0), y=y, idx=data_idx,
                mol_assignment=torch.cat([mol_assignment_reactants, mol_assignment_products], dim=0),
                atom_map_numbers=torch.cat([atom_map_reactants, atom_map_products], dim=0),
                smiles=smiles_to_save,
                atom_charges=torch.cat([atom_charges_rct, atom_charges_prod], dim=0),
                atom_chiral=torch.cat([atom_chiral_rct, atom_chiral_prod], dim=0),
                bond_dirs=torch.cat([bond_dirs_rct, bond_dirs_prod], dim=0))
    
    return graph

def rxn_smi_to_graph_data(cfg, rxn_smi):
    cannot_generate = False
    offset = 0 
    reactants = [r for r in rxn_smi.split('>>')[0].split('.')]
    products = [p for p in rxn_smi.split('>>')[1].split('.')]
    
    # mask: (n), with n = nb of nodes
    mask_product_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only reactant nodes = True
    mask_reactant_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only product nodes = True
    mask_sn = torch.ones(MAX_ATOMS_RXN, dtype=torch.bool) # only sn = False
    mask_atom_mapping = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)
    mol_assignment = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)

    # preprocess: get total number of product nodes
    nb_product_nodes = sum([len(Chem.MolFromSmiles(p).GetAtoms()) for p in products])
    nb_rct_nodes = sum([len(Chem.MolFromSmiles(r).GetAtoms()) for r in reactants])
    
    # add dummy nodes: (nodes_in_product + max_added) - nodes_in_reactants
    nb_dummy_toadd = nb_product_nodes + cfg.dataset.nb_rct_dummy_nodes - nb_rct_nodes
    if nb_dummy_toadd<0:
        # cut the rct nodes
        nb_dummy_toadd = 0
        cannot_generate = True

    # TODO: Add here some kind of loop over the reactants and the products that erases the atom mapping order 
    # and replaces it with the SMILES-based order. 
    # Do we want to add it here or in a separate process thing? Maybe here but in a separate function that is called from here?
    # But for the reactant side: We would ideally also include the information about splitting the separate graphs
    # in this preprocessing stage. -> but that's a totally separate thing anyways, a bit later
    for j, r in enumerate(reactants):
        # NOTE: no supernodes for reactants (treated as one block)
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = mol.mol_to_graph(mol=r, atom_types=cfg.dataset.atom_types, 
                                                                           bond_types=cfg.dataset.bond_types, with_explicit_h=cfg.dataset.with_explicit_h,
                                                                           with_formal_charge=cfg.dataset.with_formal_charge,
                                                                           offset=offset, get_atom_mapping=True,
                                                                           canonicalize_molecule=cfg.dataset.canonicalize_molecule)
                        
        g_nodes_rct = torch.cat((g_nodes_rct, gi_nodes), dim=0) if j > 0 else gi_nodes # already a tensor
        g_edge_index_rct = torch.cat((g_edge_index_rct, gi_edge_index), dim=1) if j > 0 else gi_edge_index
        g_edge_attr_rct = torch.cat((g_edge_attr_rct, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr

        atom_mapped_idx = (atom_map!=0).nonzero()
        mask_atom_mapping[atom_mapped_idx+offset] = atom_map[atom_mapped_idx]
        mol_assignment[offset:offset+gi_nodes.shape[0]] = j+1
        offset += gi_nodes.shape[0] 

    g_nodes_dummy = torch.ones(nb_dummy_toadd, dtype=torch.long) * cfg.dataset.atom_types.index(DUMMY_RCT_NODE_TYPE)
    g_nodes_dummy = F.one_hot(g_nodes_dummy, num_classes=len(cfg.dataset.atom_types)).float()
    # edges: fully connected to every node in the rct side with edge type 'none'
    g_edges_idx_dummy = torch.zeros([2, 0], dtype=torch.long)
    g_edges_attr_dummy = torch.zeros([0, len(cfg.dataset.bond_types)], dtype=torch.long)
    mask_product_and_sn[:g_nodes_rct.shape[0]+g_nodes_dummy.shape[0]] = True
    mol_assignment[offset:offset+g_nodes_dummy.shape[0]] = 0
    offset += g_nodes_dummy.shape[0]
    
    supernodes_prods = []
    for j, p in enumerate(products):
        # NOTE: still need supernode for product to distinguish it from reactants
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = mol.rxn_to_graph_supernode(mol=p, atom_types=cfg.dataset.atom_types, bond_types=cfg.dataset.bond_types,
                                                                                     with_explicit_h=cfg.dataset.with_explicit_h, supernode_nb=offset+1,
                                                                                     with_formal_charge=cfg.dataset.with_formal_charge,
                                                                                     add_supernode_edges=cfg.dataset.add_supernode_edges, get_atom_mapping=True,
                                                                                     canonicalize_molecule=cfg.dataset.canonicalize_molecule)
        
        g_nodes_prod = torch.cat((g_nodes_prod, gi_nodes), dim=0) if j > 0 else gi_nodes # already a tensor
        g_edge_index_prod = torch.cat((g_edge_index_prod, gi_edge_index), dim=1) if j > 0 else gi_edge_index
        g_edge_attr_prod = torch.cat((g_edge_attr_prod, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr
        atom_mapped_idx = (atom_map!=0).nonzero()
        mask_atom_mapping[atom_mapped_idx+offset] = atom_map[atom_mapped_idx]
        mask_reactant_and_sn[offset:gi_nodes.shape[0]+offset] = True
        mol_assignment[offset] = 0 # supernode does not belong to any molecule
        mol_assignment[offset+1:offset+1+gi_nodes.shape[0]] = len(reactants)+j+1
        mask_sn[offset] = False
        mask_reactant_and_sn[offset] = False
        # supernode is always in the first position
        si = 0 # gi_edge_index[0][0].item()
        supernodes_prods.append(si)
        offset += gi_nodes.shape[0]
    
    # concatenate all types of nodes and edges
    g_nodes = torch.cat([g_nodes_rct, g_nodes_dummy, g_nodes_prod], dim=0)
    g_edge_index = torch.cat([g_edge_index_rct, g_edges_idx_dummy, g_edge_index_prod], dim=1)
    g_edge_attr = torch.cat([g_edge_attr_rct, g_edges_attr_dummy, g_edge_attr_prod], dim=0)
    y = torch.zeros((1, 0), dtype=torch.float)
    
    # trim masks => one element per node in the rxn graph
    mask_product_and_sn = mask_product_and_sn[:g_nodes.shape[0]] # only reactant nodes = True
    mask_reactant_and_sn = mask_reactant_and_sn[:g_nodes.shape[0]]
    mask_sn = mask_sn[:g_nodes.shape[0]]
    mask_atom_mapping = mask_atom_mapping[:g_nodes.shape[0]]
    mol_assignment = mol_assignment[:g_nodes.shape[0]]

    assert mask_atom_mapping.shape[0]==g_nodes.shape[0] and mask_sn.shape[0]==g_nodes.shape[0] and \
            mask_reactant_and_sn.shape[0]==g_nodes.shape[0] and mask_product_and_sn.shape[0]==g_nodes.shape[0] and \
            mol_assignment.shape[0]==g_nodes.shape[0]

    graph = Data(x=g_nodes, edge_index=g_edge_index, edge_attr=g_edge_attr, y=y, idx=0,
                 mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn, 
                 mask_product_and_sn=mask_product_and_sn, mask_atom_mapping=mask_atom_mapping,
                 mol_assignment=mol_assignment, cannot_generate=cannot_generate)
    
    return graph

def get_graph_data_from_product_smi(product_smi, cfg, return_pyg_batch=True):
    # mask: (n), with n = nb of nodes
    mask_product_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only reactant nodes = True
    mask_reactant_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only product nodes = True
    mask_sn = torch.ones(MAX_ATOMS_RXN, dtype=torch.bool) # only sn = False
    mask_atom_mapping = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)
    mol_assignment = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)

    # preprocess: get total number of product nodes
    nb_product_nodes = len(Chem.MolFromSmiles(product_smi).GetAtoms())
            
    # add dummy nodes: (nodes_in_product + max_added) - nodes_in_reactants
    nb_dummy_toadd = nb_product_nodes + cfg.dataset.nb_rct_dummy_nodes
    g_nodes_dummy = torch.ones(nb_dummy_toadd, dtype=torch.long) * cfg.dataset.atom_types.index(DUMMY_RCT_NODE_TYPE)
    g_nodes_dummy = F.one_hot(g_nodes_dummy, num_classes=len(cfg.dataset.atom_types)).float()
    # edges: fully connected to every node in the rct side with edge type 'none'
    g_edges_idx_dummy = torch.zeros([2, 0], dtype=torch.long)
    g_edges_attr_dummy = torch.zeros([0, len(cfg.dataset.bond_types)], dtype=torch.long)
    
    mask_product_and_sn[:g_nodes_dummy.shape[0]] = True
    mol_assignment[:g_nodes_dummy.shape[0]] = 0
    # mask_atom_mapping[:g_nodes_dummy.shape[0]] = MAX_ATOMS_RXN
    offset = g_nodes_dummy.shape[0]
    #offset = 0
            
    supernodes_prods = []
    # NOTE: still need supernode for product to distinguish it from reactants
    g_nodes_prod, g_edge_index_prod, g_edge_attr_prod, atom_map = mol.rxn_to_graph_supernode(mol=product_smi, atom_types=cfg.dataset.atom_types, bond_types=cfg.dataset.bond_types,
                                                                                             with_explicit_h=cfg.dataset.with_explicit_h, supernode_nb=offset+1,
                                                                                             with_formal_charge=cfg.dataset.with_formal_charge,
                                                                                             add_supernode_edges=cfg.dataset.add_supernode_edges, 
                                                                                             get_atom_mapping=True,
                                                                                             canonicalize_molecule=cfg.dataset.canonicalize_molecule)
    atom_mapped_idx = (atom_map!=0).nonzero()
    mask_atom_mapping[atom_mapped_idx+offset] = atom_map[atom_mapped_idx]
    mask_atom_mapping[atom_mapped_idx] = atom_map[atom_mapped_idx] # also add atom-mapping to the reactants
    mask_reactant_and_sn[offset:g_nodes_prod.shape[0]+offset] = True
    mol_assignment[offset] = 0 # supernode does not belong to any molecule
    mol_assignment[offset+1:offset+1+g_nodes_prod.shape[0]] = 1
    mask_sn[offset] = False
    mask_reactant_and_sn[offset] = False
    # supernode is always in the first position
    si = 0 # gi_edge_index[0][0].item()
    supernodes_prods.append(si)
    offset += g_nodes_prod.shape[0]

    # concatenate all types of nodes and edges
    g_nodes = torch.cat([g_nodes_dummy, g_nodes_prod], dim=0)
    g_edge_index = torch.cat([g_edges_idx_dummy, g_edge_index_prod], dim=1)
    g_edge_attr = torch.cat([g_edges_attr_dummy, g_edge_attr_prod], dim=0)

    y = torch.zeros((1, 0), dtype=torch.float)
            
    # trim masks => one element per node in the rxn graph
    mask_product_and_sn = mask_product_and_sn[:g_nodes.shape[0]] # only reactant nodes = True
    mask_reactant_and_sn = mask_reactant_and_sn[:g_nodes.shape[0]]
    mask_sn = mask_sn[:g_nodes.shape[0]]
    mask_atom_mapping = mask_atom_mapping[:g_nodes.shape[0]]
    mol_assignment = mol_assignment[:g_nodes.shape[0]]

    assert mask_atom_mapping.shape[0]==g_nodes.shape[0] and mask_sn.shape[0]==g_nodes.shape[0] and \
            mask_reactant_and_sn.shape[0]==g_nodes.shape[0] and mask_product_and_sn.shape[0]==g_nodes.shape[0] and \
            mol_assignment.shape[0]==g_nodes.shape[0]

    data = Data(x=g_nodes, edge_index=g_edge_index, 
                    edge_attr=g_edge_attr, y=y, idx=0,
                    mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn, 
                    mask_product_and_sn=mask_product_and_sn, mask_atom_mapping=mask_atom_mapping,
                    mol_assignment=mol_assignment, cannot_generate=False)
    
    graph = Batch.from_data_list([data])
    
    if return_pyg_batch:
        return graph
    
    return data

def duplicate_data(dense_data, n_samples=1, mask_=True, get_discrete_data=False):
    '''
        Turn test data to the format used by the test function:
          1) duplicate dense data tensors if n_samples>1, 
          2) (optional) turn one-hot encoded matrices to vectors of discrete values.
        
        Input:
            data: batched pyg graph data.
            n_samples: (optional, default=1) number of samples to duplicate the data to (only applies to dense data for now).
            get_discrete_data: (optional, default=False) whether to return dense data in discrete vector format or not.

        Output: 
            dense data (X, E), (optional) discrete vector versions of X and E.
    '''
    dense_data = copy.deepcopy(dense_data)
    if mask_: dense_data = dense_data.mask(dense_data.node_mask) #     
    dense_data.X = dense_data.X.repeat_interleave(n_samples, dim=0) # (bs, n, v) => (bs*n_samples, n, v)
    dense_data.E = dense_data.E.repeat_interleave(n_samples, dim=0) # (bs, n, n, e) => (bs*n_samples, n, n, e)
    dense_data.node_mask = dense_data.node_mask.repeat_interleave(n_samples, dim=0)
    dense_data.y = dense_data.y.repeat_interleave(n_samples, dim=0)
    if dense_data.atom_map_numbers is not None:
        dense_data.atom_map_numbers = dense_data.atom_map_numbers.repeat_interleave(n_samples, dim=0)
    if dense_data.mol_assignment is not None:
        dense_data.mol_assignment = dense_data.mol_assignment.repeat_interleave(n_samples, dim=0)
    if dense_data.pos_encoding is not None:
        dense_data.pos_encoding = dense_data.pos_encoding.repeat_interleave(n_samples, dim=0)
    if dense_data.smiles is not None: # This should be a list
        # do the repetition in a repeat_interleave fashion
        dense_data.smiles = [s for s in dense_data.smiles for _ in range(n_samples)]
    if dense_data.atom_charges is not None:
        dense_data.atom_charges = dense_data.atom_charges.repeat_interleave(n_samples, dim=0)
    if dense_data.atom_chiral is not None:
        dense_data.atom_chiral = dense_data.atom_chiral.repeat_interleave(n_samples, dim=0)
    if dense_data.bond_dirs is not None:
        dense_data.bond_dirs = dense_data.bond_dirs.repeat_interleave(n_samples, dim=0)

    if get_discrete_data:
        dense_data_discrete = copy.deepcopy(dense_data).mask(dense_data.node_mask, collapse=True)
        return dense_data, dense_data_discrete
    
    return dense_data

def noise_atom_mapping(samples_placeholder):
    """
    Change the atom mapping to something slightly different on the reactant side
    """
    max_am = samples_placeholder.atom_map_numbers.max().item()
    product_node_start_idx = (samples_placeholder.mol_assignment[0] == 0).nonzero()[0]
    num_possible_changes = 100
    change_maps = [{i:j, j:i} for i,j in  zip(np.random.randint(1,max_am,100), np.random.randint(1,max_am,num_possible_changes))]
    for i in range(len(samples_placeholder.atom_map_numbers)):
        # sample a random change map
        change_map = random.choice(change_maps)
        # change the atom mapping on the reactant side
        samples_placeholder.atom_map_numbers[i,:product_node_start_idx] = torch.tensor([(change_map[am] if am in change_map else am) for am in samples_placeholder.atom_map_numbers[i,:product_node_start_idx].tolist()]).unsqueeze(0)


def choose_most_common_atom_mapping(same_reaction_groups, gen_rxn_smiles_with_am):
    '''
        Choose the most common atom mapping for each reaction.'
        TODO: This probably doesn't work that great yet, see choose_highest_probability_atom_mapping_from_placeholder
    '''
    gen_rxn_smiles_with_am = copy.deepcopy(gen_rxn_smiles_with_am)
    for i, group in enumerate(same_reaction_groups):
        atom_mappings = [gen_rxn_smiles_with_am[j].split('>>')[0] for j in group]
        atom_mappings = [am.split(' ')[1:] for am in atom_mappings]
        atom_mappings = [tuple([int(a) for a in am]) for am in atom_mappings] # TODO: This doesn't take into account the fact that the atoms are not necessarily in the same order in the different generations
        most_common_am = Counter(atom_mappings).most_common(1)[0][0]
        gen_rxn_smiles_with_am[group[0]] = gen_rxn_smiles_with_am[group[0]].split(' ')[0] + ' ' + ' '.join([str(a) for a in most_common_am])
    return gen_rxn_smiles_with_am

def get_rct_atom_mapping_from_smiles(smiles):
    '''
        Get the atom mapping from the atom-mapped SMILES string. (only for the reactants)
    '''
    mol = Chem.MolFromSmiles(smiles.split('>>')[0])
    if mol is None: # Set None, don't do anything with this
        return None
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]

def get_prod_atom_mapping_from_smiles(smiles):
    """
        Get the atom mapping from the atom-mapped SMILES string. (only for the product)
    """
    mol = Chem.MolFromSmiles(smiles.split('>>')[1])
    if mol is None: # Set None, don't do anything with this
        return None
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]

def choose_highest_probability_atom_mapping_from_placeholder(gen_samples_placeholder_with_one_group, gen_rxn_smiles_with_am_with_one_group):
    """
    gen_samples_placeholder: A placeholder object containing the generated samples. All the graphs should be the same, but the atom mappings may be different
        Algorithm:
        We have a distribution of atom mapping assignments for each atom. To get this, we want to align the atom-mapped parts of the reactants together
        We choose the first set of reactants as the reference reactants. We then create a mapping from reference atoms to atoms in the other reactants sets using SMILES strings
    """

    # TODO: MAKE SURE THAT THE INDEXING IS THE SAME FOR THE ORIGINAL_ATOM_MAPPING AND TOP_ATOM_MAPPING
    original_atom_mapping = get_rct_atom_mapping_from_smiles(gen_rxn_smiles_with_am_with_one_group[0])
    if original_atom_mapping is None:
        return gen_samples_placeholder_with_one_group.slice_by_idx(1).atom_map_numbers[0]
    resolved_mols, top_atom_mapping = mol.resolve_atom_mappings([r.split('>>')[0] for r in gen_rxn_smiles_with_am_with_one_group])
    #top_atom_mapping = choose_highest_probability_atom_mapping_from_smiles(gen_rxn_smiles_with_am_with_one_group)
    
    # create a mapping from the original atom mapping in the first reactant set to the top atom mapping of the corresponding atoms
    original_am_to_top_am = {original_atom_mapping[i]: top_atom_mapping[i] for i in range(len(original_atom_mapping))}
    original_am_to_top_am[0] = 0
    gen_samples_reference_placeholder = gen_samples_placeholder_with_one_group.slice_by_idx(1) # select the first sample 

    # Then output the edited atom mapping (only edited for the reactant side)
    am = gen_samples_reference_placeholder.atom_map_numbers[0].clone()
    # only change up to product node
    product_mol_assignment_idx = gen_samples_reference_placeholder.mol_assignment[0].max()
    product_node_start_idx = (gen_samples_reference_placeholder.mol_assignment[0] == product_mol_assignment_idx).nonzero()[0]
    # In the following, account for the possibility that some of the atom mappings were placed on blank nodes (gen_rxn_smiles_with_am_with_one_group will not have that node)
    
    am[:product_node_start_idx] = torch.tensor([(original_am_to_top_am[a_m] if a_m in original_am_to_top_am else a_m) for a_m in gen_samples_reference_placeholder.atom_map_numbers[0,:product_node_start_idx].tolist()]).unsqueeze(0) 
    
    rct_am = am[:product_node_start_idx].tolist()
    prod_am = am[product_node_start_idx:].tolist()
    if set(rct_am) - set([0]) != set(prod_am) - set([0]): # if failed, just go back to original
        return gen_samples_reference_placeholder.atom_map_numbers[0].clone()

    return am
    # return torch.tensor([original_am_to_top_am[am] for am in gen_samples_reference_placeholder.atom_map_numbers[0].tolist()]).unsqueeze(0)
    # gen_samples_reference_placeholder.atom_map_numbers = torch.tensor([original_am_to_top_am[am] for am in gen_samples_reference_placeholder.atom_map_numbers[0].tolist()]).unsqueeze(0)

    # return gen_samples_reference_placeholder

    # Next we want to create the atom-mapping-to-atom-mapping matching, to transfer the matching to the atom-mapped SMILES
    
    # Let's think about how to do this. 
    # We have the mappings_from_reference_to_reactants, which is a list of lists of indices.
    
    # We can get the Counters for each atom in the reference reactants using the code below (factorize this code somehow to reuse it)
    # We can then change the keys of the counters to the atom map numbers of the reference reaction
    # -> then we can get the counters for the first reference placeholder reaction indices, and consequently we can 
    # change the placeholder reaction atom mappings to the top atom mappings, and return that placeholder object


def choose_highest_probability_atom_mapping_from_smiles(gen_rxn_smiles_with_am_with_one_group):
    """
    gen_rxn_smiles_with_am_with_one_group: List of atom-mapped SMILES representations of the reactions. All the reactions should be the same, but potentially with different atom mappings
        Algorithm: 
        We have a distribution of atom mapping assignments for each atom. To get this, we first have to do graph matching
        We put then sort the atoms based on the maximum probability of atom mapping assignment.
        We then assign the highest-probability atom mapping to the first atom in the list, and remove this from the atom mappings that are left to assign.
        In principle, at this points we should resort the list. But we can maybe skip that stage in the beginnning here
        We then assign the highest-probability atom mapping to the second atom in the list, and remove this from the atom mappings that are left to assign.
        We continue this until all atom mappings are assigned.

    Returns:
        output_atom_mapping: A list of atom map numbers that are assigned to the atoms in the reference reactants. Ordering based on the 
        molecule object atom ordering in the reference reactant.

        Hmmm symmetric groups inside the molecule can be a major problem for this approach actually... maybe shuold just stick to the 
        most common SMILES string overall. Or maybe the symmetric groups will be resolved in the same way for all the reactants anyways? 
        (if we do the matching to the reference reactants as well?) Nah Not sure about this
        -> could try breaking the symmetry with the atom mapping that we already have in the matching algorithm. 
        -> but that would require actually rewriting the graph matching algorithm to take into account the atom mappings. 
        -> and still unclear how exactly would that work. 

        -> something else to deal with symmetric groups? 

    """
    # turn the generated reactants into rdkit molecules
    reactants = [Chem.MolFromSmiles(rxn.split('>>')[0]) for rxn in gen_rxn_smiles_with_am_with_one_group]
    
    # Perform substructure matching for each reactant
    reference_reactants = reactants[0]
    mappings_from_reference_to_reactants = [tuple(range(len(reference_reactants.GetAtoms())))] # identity mapping from reference to itself
    
    for reactant in reactants[1:]:
        mapping_from_reference_to_reactant = reference_reactants.GetSubstructMatch(reactant)
        mappings_from_reference_to_reactants.append(mapping_from_reference_to_reactant)
    
    # For each atom index in the reference reactants, get the distribution of atom maps (represent as a Counter object)
    atom_map_distributions = []
    for i in range(len(reference_reactants.GetAtoms())): # loop over reference atoms
        corresponding_index_in_other_reactants = [mappings[i] for mappings in mappings_from_reference_to_reactants] # a list of indices
        corresponding_ams_in_other_reactants = [reactant.GetAtomWithIdx(index).GetAtomMapNum()for reactant, index in zip(reactants, corresponding_index_in_other_reactants)]
        # corresponding_ams_in_other_reactants += [reference_reactants.GetAtomWithIdx(i).GetAtomMapNum()] # add the reference atom map number <- this is taken care of the by the identity mapping
        atom_map_distributions.append(Counter(corresponding_ams_in_other_reactants))
    
    # Sort the atom_map_distributions based on the maximum probability of atom mapping assignment
    sorted_atom_map_distributions = sorted(atom_map_distributions, key=lambda x: x.most_common(1)[0][1], reverse=True)

    output_atom_mapping = []
    # Assign the atom mappings to the atoms in the reference reactants
    # We can do this in a greedy way, by assigning the highest-probability atom mapping to the first atom in the list, 
    # and remove this from the atom mappings that are left to assign. 
    assigned_atom_mappings = set()
    for i in range(len(reference_reactants.GetAtoms())):
        # Remove the atom mappings that have already been assigned from the Counter for the i:th atom
        for j in range(len(sorted_atom_map_distributions)):
            for am in assigned_atom_mappings:
                if am in sorted_atom_map_distributions[j]:
                    del sorted_atom_map_distributions[j][am]
        # Assign the highest-probability atom mapping to the i:th atom
        am = sorted_atom_map_distributions[i].most_common(1)[0][0]
        assigned_atom_mappings.add(am)
        output_atom_mapping.append(am)
    
    return output_atom_mapping

def set_atom_mapping_to_reactants_of_multiple_reaction_smiles(reaction_smiles, reactant_atom_map_numbers):
    """
    Here the assumption is that reaction_smiles is a list of SMILES strings, and reactant_atom_map_nubmers 
    is a list of lists of atom map numbers.
    """
    for i in range(len(reaction_smiles)):
        reaction_smiles[i] = set_atom_mapping_to_reactant_of_reaction_smiles(reaction_smiles[i], reactant_atom_map_numbers[i])

def set_atom_mapping_to_reactant_of_reaction_smiles(reaction_smiles, reactant_atom_map_numbers):
    """
    Set the atom mapping to the atom_map_numbers list in the reactant side for the reaction_smiles object.
    """
    # First loop over the reactants, turn them into rdkit molecule objects and get the atom mappings
    # reactants = [r for r in reaction_smiles.split('>>')[0].split('.')]
    reactants = reaction_smiles.split('>>')[0]
    reactant_mols = Chem.MolFromSmiles(reactants)
    # set the molecule atom mappings to the reactant_atom_map_numbers tensor
    for i in range(len(reactant_mols.GetAtoms())):
        reactant_mols.GetAtomWithIdx(i).SetAtomMapNum(reactant_atom_map_numbers[i])
    return Chem.MolToSmiles(reactant_mols) + '>>' + reaction_smiles.split('>>')[1]    

def set_atom_mapping_to_reactants_of_placeholder(placeholder, atom_mapping):
    """
    Set the atom mapping to the atom_map_numbers tensor in the reactant side for the placeholder object.
    """
    for i in range(len(placeholder.atom_map_numbers)):
        placeholder.atom_map_numbers[i, :len(atom_mapping[i])] = torch.tensor(atom_mapping[i]).to(placeholder.atom_map_numbers.device)
    return placeholder

def get_unique_indices_from_reaction_list(gen_rxn_smiles):
    """
        Remove duplicates from data.
    Input: 
        gen_rxn_smiles: list of SMILES strings.
    Output:
        - sorted_unique_indices: first indices of unique reactions in the sorted list of gen_rxn_smiles
        - sorted_counts: number of times each unique reaction appears in the list (indexed according to the sorted_unique_indices list)
        - is_unique: list of booleans indicating whether each reaction is unique or not. (according to the original gen_rxn_smiles sorting)
    """

    # get unique reactions
    rcts = [r.split('>>') for r in gen_rxn_smiles]
    # sort the reactants alphabetically in case rdkit doesn't do this automatically
    rcts = [".".join(sorted(r[0].split('.'))) + '>>' + r[1] for r in rcts]

    unique_indices, counts = np.unique(rcts, return_index=True, return_counts=True)[1:]
    is_unique = [i in unique_indices for i in range(len(gen_rxn_smiles))] # [0,0,0,0] = [u, u, u, ]

    # Pair the two arrays
    paired = list(zip(counts, unique_indices))
    # Sort the paired list based on unique_indices
    sorted_paired = sorted(paired, key=lambda x: x[1])
    # Unzip the sorted paired list
    sorted_counts, sorted_unique_indices = zip(*sorted_paired)
    # Convert them back to lists
    sorted_counts = list(sorted_counts)
    sorted_unique_indices = list(sorted_unique_indices)

    # Also return a list that contains lists of indices of the same reaction
    same_reaction_groups = []
    for i in range(len(sorted_unique_indices)):
        same_reaction_groups.append([j for j, x in enumerate(rcts) if x == rcts[sorted_unique_indices[i]]])    
    
    return sorted_unique_indices, sorted_counts, is_unique, same_reaction_groups

    # get unique rows
    # X_unique, idx_unique = torch.unique(data_placeholder.X, dim=0, return_inverse=True)
    # E_unique = data_placeholder.E[idx_unique]
    # node_mask_unique = data_placeholder.node_mask[idx_unique]
    # y_unique = data_placeholder.y[idx_unique]
    # atom_map_numbers_unique = data_placeholder.atom_map_numbers[idx_unique]
    # mol_assignment_unique = data_placeholder.mol_assignment[idx_unique]
    
    # return PlaceHolder(X=X_unique, E=E_unique, y=y_unique, node_mask=node_mask_unique, 
    #                    atom_map_numbers=atom_map_numbers_unique, mol_assignment=mol_assignment_unique)

def to_dense(data, cfg=None, smiles=None):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch)
    X = encode_no_element(X)
    
    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr) # these should, in principle, be also removed from the bond_dirs
    #try:
    E = to_dense_adj(edge_index=edge_index, batch=data.batch, 
                         edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    # except Exception as e:
    #     log.info(f"An error occurred: {e}")
        
    E = encode_no_element(E)

    #atom_map_numbers, mol_assignment, smiles, pos_encoding, atom_charges, atom_chiral, bond_dirs = None, None, None, None, None, None, None
    atom_map_numbers = torch.zeros(X.shape[0], X.shape[1], dtype=torch.int64)
    mol_assignment = torch.zeros(X.shape[0], X.shape[1], dtype=torch.int64)

    atom_charges = torch.zeros(X.shape[0], X.shape[1], 3, dtype=torch.float32)
    atom_chiral = torch.zeros(X.shape[0], X.shape[1], 3, dtype=torch.float32)
    bond_dirs = torch.zeros(E.shape[0], E.shape[1], E.shape[2], 3,dtype=torch.float32)
    keys =  data.keys if type(data.keys)==dict or type(data.keys)==list else data.keys() # TODO: This seems quite hacky at the moment
    pos_encoding = None
    smiles = ['']*X.shape[0]
    # if 'atom_map_numbers' in keys:
    #     # atom_map_numbers, _ = to_dense_batch(x=data.mask_atom_mapping, batch=data.batch)
    #     atom_map_numbers = data.atom_map_numbers # were these of this shape? -> let's check
    # if 'mol_assignment' in keys:
    #     mol_assignment = data.mol_assignment
    # TODO: should this match the one-hot encoding?
    if 'mol_assignment' in keys: # For the original pyg objects, it is called mol_assignment, and not mol_assignment
        mol_assignment, _ = to_dense_batch(x=data.mol_assignment, batch=data.batch) 
    if 'atom_map_numbers' in keys:
        atom_map_numbers, _ = to_dense_batch(x=data.atom_map_numbers, batch=data.batch)
    if 'pos_encoding' in keys:
        pos_encoding, _ = to_dense_batch(x=data.pos_encoding, batch=data.batch)
    if 'smiles' in keys:
        smiles = data.smiles
    if 'atom_charges' in keys:
        atom_charges, _ = to_dense_batch(x=data.atom_charges, batch=data.batch)
        atom_charges = encode_no_element(atom_charges)
    if 'atom_chiral' in keys:
        atom_chiral, _ = to_dense_batch(x=data.atom_chiral, batch=data.batch)
        atom_chiral = encode_no_element(atom_chiral)
    if 'bond_dirs' in keys:
        edge_index, bond_dirs = remove_self_loops(data.edge_index, data.bond_dirs) 
        bond_dirs = to_dense_adj(edge_index=edge_index, batch=data.batch, 
                                 edge_attr=bond_dirs, max_num_nodes=max_num_nodes)
        bond_dirs = encode_no_element(bond_dirs)
    
    return PlaceHolder(X=X, E=E, y=data.y, node_mask=node_mask, atom_map_numbers=atom_map_numbers, mol_assignment=mol_assignment, 
                        atom_charges=atom_charges, atom_chiral=atom_chiral, bond_dirs=bond_dirs, pos_encoding=pos_encoding, smiles=smiles)

def concatenate_databatches(cfg, list_of_databatches):
    # Concatenates a list of DataBatches together
    concatenated = []
    for i in range(len(list_of_databatches)):
        concatenated.extend(list_of_databatches[i].to_data_list())
    return Batch.from_data_list(concatenated)

def pyg_to_full_precision_expanded(data, cfg):
    """Reverses the encoding of the data to full precision after encoding to pyg format and saving to pickle.
    data is a DataBatch object. 
    Also expands out x to the one-hot encoding."""
    new_data = copy.deepcopy(data)
    new_data.x = F.one_hot(new_data.x.long(), len(cfg.dataset.atom_types))
    new_data.edge_attr = F.one_hot(new_data.edge_attr.long(), len(cfg.dataset.bond_types))
    # keys handles different pyg versions
    keys = new_data.keys if type(new_data.keys)==list else new_data.keys()
    if 'atom_charges' in keys:
        new_data.atom_charges = F.one_hot(new_data.atom_charges.long(), len(cfg.dataset.atom_charges))
    if 'atom_chiral' in keys:
        new_data.atom_chiral = F.one_hot(new_data.atom_chiral.long(), len(cfg.dataset.atom_chiral_tags))
    if 'bond_dirs' in keys:
        new_data.bond_dirs = F.one_hot(new_data.bond_dirs.long(), len(cfg.dataset.bond_dirs))
    # new_data.edge_attr = data.edge_attr.long()
    if 'pos_encoding' in keys:
        new_data.pos_encoding = new_data.pos_encoding.float()
    new_data.edge_index = new_data.edge_index.long()
    new_data.y = new_data.y.float()
    new_data.node_mask = new_data.node_mask.bool()
    new_data.atom_map_numbers = new_data.atom_map_numbers.long()
    new_data.mol_assignment = new_data.mol_assignment.long()
    return new_data
    
class PlaceHolder:
    # TODO: maybe rename X, E and y to smthg more informative?
    def __init__(self, X, E, y,
                 atom_charges=None, atom_chiral=None,
                 bond_dirs=None, node_mask=None, atom_map_numbers=None,
                 mol_assignment=None, pos_encoding=None, smiles=None):
        self.X = X
        self.atom_charges = atom_charges
        self.atom_chiral = atom_chiral
        self.E = E
        self.bond_dirs = bond_dirs
        self.y = y
        self.node_mask = node_mask
        self.atom_map_numbers = atom_map_numbers
        self.mol_assignment = mol_assignment
        self.pos_encoding = pos_encoding
        self.smiles = smiles

    def list_not_valid_attributes(self, attributes):
        return [at for at in attributes if not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None and not at=='smiles']
    
    def list_not_tensor_attributes(self, attributes):
        return [at for at in attributes if not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None]

    def flatten(self, start_dim, end_dim):
        '''
            return a placeholder object with feature tensors flattened from start_dim to end_dim.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
     
        new_kwargs = {at: getattr(self, at).flatten(start_dim=start_dim, end_dim=end_dim) if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}
        
        not_valid_attribute = self.list_not_valid_attributes(attributes)
        not_tensor_attribute = self.list_not_tensor_attributes(attributes)
        
        for at in not_tensor_attribute:
            if at=='smiles':
                new_kwargs[at] = getattr(self, at) #TODO: do we have to flatten this somehow?
            else:
                assert 'PlaceHolder object has none tensor attributes other than smiles.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj
    
    def reshape_bs_n_samples(self, bs, n_samples, n):
        # TODO: make sure the features are not one-hot encoded
        self.X = self.X.reshape(bs, n_samples, n)
        self.atom_charges = self.atom_charges.reshape(bs, n_samples, n)
        self.atom_chiral = self.atom_chiral.reshape(bs, n_samples, n)
        self.E = self.E.reshape(bs, n_samples, n, n)
        self.bond_dirs = self.bond_dirs.reshape(bs, n_samples, n, n)
        self.y = torch.empty((bs, n_samples))
        self.node_mask = self.node_mask.reshape(bs, n_samples, n)
        if self.atom_map_numbers is not None:
            self.atom_map_numbers = self.atom_map_numbers.reshape(bs, n_samples, n)
        self.mol_assignment = self.mol_assignment.reshape(bs, n_samples, n)
        if self.pos_encoding is not None:
            self.pos_encoding = self.pos_encoding.reshape(bs, n_samples, n, -1)
                                               
    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.type_as(x)
        if isinstance(self.mol_assignment, torch.Tensor):
            self.mol_assignment = self.mol_assignment.type_as(x)
        if isinstance(self.atom_charges, torch.Tensor):
            self.atom_charges = self.atom_charges.type_as(x)
        if isinstance(self.atom_chiral, torch.Tensor):
            self.atom_chiral = self.atom_chiral.type_as(x)
        if isinstance(self.node_mask, torch.Tensor):
            self.node_mask = self.node_mask.type_as(x)
        if isinstance(self.pos_encoding, torch.Tensor):
            self.pos_encoding = self.pos_encoding.type_as(x)
        if isinstance(self.bond_dirs, torch.Tensor):
            self.bond_dirs = self.bond_dirs.type_as(x)

        return self
    
    def to_device(self, device):
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device)
        self.node_mask = self.node_mask.to(device)
        
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.to(device)
        if isinstance(self.mol_assignment, torch.Tensor):
            self.mol_assignment = self.mol_assignment.to(device)
        if isinstance(self.atom_charges, torch.Tensor):
            self.atom_charges = self.atom_charges.to(device)
        if isinstance(self.atom_chiral, torch.Tensor):
            self.atom_chiral = self.atom_chiral.to(device)
        if isinstance(self.node_mask, torch.Tensor):
            self.node_mask = self.node_mask.to(device)
        if isinstance(self.pos_encoding, torch.Tensor):
            self.pos_encoding = self.pos_encoding.to(device)
        if isinstance(self.bond_dirs, torch.Tensor):
            self.bond_dirs = self.bond_dirs.to(device)
            
        return self
    
    def to_numpy(self):
        self.X = self.X.detach().cpu().numpy()
        self.E = self.E.detach().cpu().numpy()
        self.y = self.y.detach().cpu().numpy()
        self.node_mask = self.node_mask.detach().cpu().numpy()
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.detach().cpu().numpy()
        if isinstance(self.mol_assignment, torch.Tensor):
            self.mol_assignment = self.mol_assignment.detach().cpu().numpy()
        if isinstance(self.atom_charges, torch.Tensor):
            self.atom_charges = self.atom_charges.detach().cpu().numpy()
        if isinstance(self.atom_chiral, torch.Tensor):
            self.atom_chiral = self.atom_chiral.detach().cpu().numpy()
        if isinstance(self.node_mask, torch.Tensor):
            self.node_mask = self.node_mask.detach().cpu().numpy()
        if isinstance(self.pos_encoding, torch.Tensor):
            self.pos_encoding = self.pos_encoding.detach().cpu().numpy()
        if isinstance(self.bond_dirs, torch.Tensor):
            self.bond_dirs = self.bond_dirs.detach().cpu().numpy()
            
        return self
    
    def to_cpu(self):
        self.X = self.X.detach().cpu()
        self.E = self.E.detach().cpu()
        self.y = self.y.detach().cpu()
        self.node_mask = self.node_mask.detach().cpu()
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.detach().cpu()
        if isinstance(self.mol_assignment, torch.Tensor):
            self.mol_assignment = self.mol_assignment.detach().cpu()
        if isinstance(self.atom_charges, torch.Tensor):
            self.atom_charges = self.atom_charges.detach().cpu()
        if isinstance(self.atom_chiral, torch.Tensor):
            self.atom_chiral = self.atom_chiral.detach().cpu()
        if isinstance(self.node_mask, torch.Tensor):
            self.node_mask = self.node_mask.detach().cpu()
        if isinstance(self.pos_encoding, torch.Tensor):
            self.pos_encoding = self.pos_encoding.detach().cpu()
        if isinstance(self.bond_dirs, torch.Tensor):
            self.bond_dirs = self.bond_dirs.detach().cpu()
            
        return self

    def to_one_hot(self, cfg):
        assert len(self.X.shape)==2 and len(self.E.shape)==3, 'X and E should be 2D and 3D tensors, respectively.'
        X = F.one_hot(self.X.long(), len(cfg.dataset.atom_types)).float()
        E = F.one_hot(self.E.long(), len(cfg.dataset.bond_types)).float()
        atom_charge = F.one_hot(self.atom_charges.long(), len(cfg.dataset.atom_charges)).float() if self.atom_charges!=None else None
        atom_chiral = F.one_hot(self.atom_chiral.long(), len(cfg.dataset.atom_chiral_tags)).float() if self.atom_chiral!=None else None
        bond_dirs = F.one_hot(self.bond_dirs.long(), len(cfg.dataset.bond_dirs)).float() if self.bond_dirs!=None else None
        one_hot_version = PlaceHolder(X=X, E=E, y=self.y, atom_charges=atom_charge, atom_chiral=atom_chiral, bond_dirs=bond_dirs,
                                        node_mask=self.node_mask, atom_map_numbers=self.atom_map_numbers, mol_assignment=self.mol_assignment,
                                        pos_encoding=self.pos_encoding, smiles=self.smiles)
        one_hot_version.mask(one_hot_version.node_mask, collapse=False)
        return one_hot_version

    def mask(self, node_mask=None, collapse=False):
        if node_mask==None: node_mask = self.node_mask
            
        assert node_mask is not None, 'node_mask is None.'
            
        x_node_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_node_mask1 = x_node_mask.unsqueeze(2)            # bs, n, 1, 1
        e_node_mask2 = x_node_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1) # (bs, n)
            self.E = torch.argmax(self.E, dim=-1) # (bs, n, n)
            self.atom_charges = torch.argmax(self.atom_charges, dim=-1) if self.atom_charges!=None else None
            self.atom_chiral = torch.argmax(self.atom_chiral, dim=-1) if self.atom_chiral!=None else None
            self.bond_dirs = torch.argmax(self.bond_dirs, dim=-1) if self.bond_dirs!=None else None

            self.X[node_mask == 0] = 0
            self.E[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0
            self.atom_charges[node_mask == 0] = 0 if self.atom_charges!=None else None
            self.atom_chiral[node_mask == 0] = 0 if self.atom_chiral!=None else None
            self.bond_dirs[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0 if self.bond_dirs!=None else None
        else:
            # always mask by node, masking by subgraph is a subset of that
            ''' 
                X_0 = NN(noisy.X) => (bs, n_max, v) 
                => e.g.: true nodes: (0, n<n_max, v) = [0.9, 0.8, .....]
                         fake nodes: (0, n>n_max, v) = [0.9, 0.8, .....] => [1, 0, 0, 0....]
                        
                => how to get correct fake nodes? 
                => node_mask: (bs, n_max, 1)
                X_0 * node_mask => X_0' = (bs, n<n_max, v) = [0.9, 0.8, .....]
                                   X_0' = (bs, n>n_max, v) = [0, 0, .....] (doesn't exist)
                                   
                => last step: fix the [0, ...] to [1, 0, ...]
                => last step for other masks: X_0' = X_0 * node_mask + X_orig * (~node_mask)
                    => X_0': (bs, n<n_max, v) = [0.9, 0.8, .....] (e.g. output of NN)
                    => X_0': (bs, n>n_max, v) = [1, 0, .....] (e.g. orig one_hot) => perks: already proba dist, already one-hot...
            '''
            self.X = self.X * x_node_mask
            self.atom_charges = self.atom_charges * x_node_mask if self.atom_charges!=None else None
            self.atom_chiral = self.atom_chiral * x_node_mask if self.atom_charges!=None else None
            self.E = self.E * e_node_mask1 * e_node_mask2
            self.bond_dirs = self.bond_dirs * e_node_mask1 * e_node_mask2 if self.bond_dirs!=None else None
            diag = torch.eye(self.E.shape[1], dtype=torch.bool).unsqueeze(0).expand(self.E.shape[0], -1, -1)
            self.E[diag] = 0
            self.X = encode_no_element(self.X)
            self.E = encode_no_element(self.E)
            self.atom_charges = encode_no_element(self.atom_charges) if self.atom_charges!=None else None
            self.atom_chiral = encode_no_element(self.atom_chiral) if self.atom_chiral!=None else None
            self.bond_dirs = encode_no_element(self.bond_dirs) if self.bond_dirs!=None else None

            # adjacency matrix of undirected graph => mirrored over the diagonal
            # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self
    
    def get_new_object(self, **kwargs):
        '''
            returns a new placeholder object with X, E or y changed 
            and the other features copied from the current placeholder object.
        '''
        # get all attributes that are not functions
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
        # logic below: 
        # ... if we're given a new variable in the kwargs for the attribute, 
        # ...... clone it and use it as the value for the new object
        # ... if the current object has a value for att, clone that instead.
        # ... if no value for the at anywhere, assign None.
        # We assume all attributes are tensors
        new_kwargs = {at: kwargs.get(at).clone() if isinstance(kwargs.get(at), torch.Tensor) else \
                          getattr(self, at).clone() if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}
        
        not_valid_attribute = self.list_not_valid_attributes(attributes)
        not_tensor_attribute = self.list_not_tensor_attributes(attributes)

        for at in not_tensor_attribute:
            if at=='smiles':
                new_kwargs[at] = getattr(self, at) #TODO: do we have to flatten this somehow?
            else:
                assert 'PlaceHolder object has none tensor attributes other than smiles.'
        
        assert sum(not_valid_attribute)==0, 'PlaceHolder object has attributes that are not tensors or lists of strings.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj

    def is_valid_data(self,data):
        """Checks whether data is valid for an attribute of the Placeholder class. Only tensor"""
        return isinstance(data, torch.Tensor) or (isinstance(data, list) and isinstance(data[0], str))

    def select_subset(self, selection):
        '''
            return a placeholder object with the selection in the form of a boolean mask of shape (bs,)
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
     
        new_kwargs = {at: getattr(self, at).clone()[selection] if isinstance(getattr(self, at), torch.Tensor) else None for at in attributes}
        if self.smiles is not None:
            new_kwargs['smiles'] = [self.smiles[i] for i in range(len(self.smiles)) if selection[i]]

        not_valid_attribute = self.list_not_valid_attributes(attributes)
        not_tensor_attribute = self.list_not_tensor_attributes(attributes)

        for at in not_tensor_attribute:
            if at=='smiles':
                new_kwargs[at] = getattr(self, at) #TODO: do we have to flatten this somehow?
            else:
                assert 'PlaceHolder object has none tensor attributes other than smiles.'
        
        assert sum(not_valid_attribute)==0, 'PlaceHolder object has attributes that are not tensors  or lists of strings. These will be set to None in the new object.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj

    def slice_by_idx(self, idx):
        '''
            return a placeholder object with the first idx batch elements.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
     
        new_kwargs = {at: getattr(self, at).clone()[:idx] if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}
        
        not_valid_attribute = self.list_not_valid_attributes(attributes)
        not_tensor_attribute = self.list_not_tensor_attributes(attributes)

        for at in not_tensor_attribute:
            if at=='smiles':
                new_kwargs[at] = getattr(self, at) #TODO: do we have to flatten this somehow?
            else:
                assert 'PlaceHolder object has none tensor attributes other than smiles.'
        
        assert sum(not_valid_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj
    
    def subset_by_index_list(self, index_list):
        '''
            return a placeholder with the elements fo the batch specified in the index_list.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
     
        new_kwargs = {at: getattr(self, at).clone()[index_list] if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}
        
        not_valid_attribute = self.list_not_valid_attributes(attributes)
        not_tensor_attribute = self.list_not_tensor_attributes(attributes)

        for at in not_tensor_attribute:
            if at=='smiles':
                new_kwargs[at] = getattr(self, at) #TODO: do we have to flatten this somehow?
            else:
                assert 'PlaceHolder object has none tensor attributes other than smiles.'
        
        assert sum(not_valid_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj
    
    def subset_by_idx(self, start_idx, end_idx):
        '''
            return a placeholder object with the first idx batch elements.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
     
        new_kwargs = {at: getattr(self, at).clone()[start_idx:end_idx] if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}
        
        not_valid_attribute = self.list_not_valid_attributes(attributes)
        not_tensor_attribute = self.list_not_tensor_attributes(attributes)
        
        for at in not_tensor_attribute:
            if at=='smiles':
                new_kwargs[at] = getattr(self, at)[start_idx:end_idx]
            else:
                assert 'PlaceHolder object has none tensor attributes other than smiles.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj
    
    def cat_by_batchdim(self, placeh):
        self.X = torch.cat((self.X, placeh.X), dim=0)
        self.atom_charges = torch.cat((self.atom_charges, placeh.atom_charges), dim=0)
        self.atom_chiral = torch.cat((self.atom_chiral, placeh.atom_chiral), dim=0)
        self.E = torch.cat((self.E, placeh.E), dim=0)
        self.bond_dirs = torch.cat((self.bond_dirs, placeh.bond_dirs), dim=0)
        self.node_mask = torch.cat((self.node_mask, placeh.node_mask), dim=0)
        self.atom_map_numbers = torch.cat((self.atom_map_numbers, placeh.atom_map_numbers), dim=0)
        self.mol_assignment = torch.cat((self.mol_assignment, placeh.mol_assignment), dim=0)
        self.y = torch.cat((self.y, placeh.y), dim=0)
        
    def cat_by_batchdim_with_padding(self, placeh):
        # 1. choose which object to pad
        if self.X.shape[1] > placeh.X.shape[1]:
            to_pad = placeh
            ready = self
        else:
            to_pad = self
            ready = placeh
            
        # 2. pad object
        pad_size = ready.X.shape[1]-to_pad.X.shape[1]
        to_pad.pad_nodes(pad_size)
        
        # 3. cat
        ready.cat_by_batchdim(to_pad)
        
        return ready
        
    def pad_nodes(self, pad_size):
        padding_tuple_X = (0, pad_size) if self.X.ndim==2 else (0, 0, 0, pad_size)
        padding_tuple_E = (0, pad_size, 0, pad_size) if self.E.ndim==3 else (0, 0, 0, pad_size, 0, pad_size)
        padding_tuple_v = (0, pad_size)
        self.X = F.pad(self.X, padding_tuple_X, value=0)
        self.atom_charges = F.pad(self.atom_charges, padding_tuple_X, value=0)
        self.atom_chiral = F.pad(self.atom_chiral, padding_tuple_X, value=0)
        self.E = F.pad(self.E, padding_tuple_E, value=0)
        self.bond_dirs = F.pad(self.bond_dirs, padding_tuple_E, value=0)
        self.node_mask = F.pad(self.node_mask, padding_tuple_v, value=0)
        self.atom_map_numbers = F.pad(self.atom_map_numbers, padding_tuple_v, value=0)
        self.mol_assignment = F.pad(self.mol_assignment, padding_tuple_v, value=0)
        
    def select_by_batch_idx(self, idx):
        '''
            Return a placeholder graph specified by the batch idx given as input.
            The returned graph does not share same memory with the original graph. 
            idx: batch idx given
        '''
        return PlaceHolder(X=copy.deepcopy(self.X[idx:idx+1]),
                           atom_charges=copy.deepcopy(self.atom_charges[idx:idx+1]),
                           atom_chiral=copy.deepcopy(self.atom_chiral[idx:idx+1]),
                           E=copy.deepcopy(self.E[idx:idx+1]), y=copy.deepcopy(self.y[idx:idx+1]), 
                           bond_dirs=copy.deepcopy(self.bond_dirs[idx:idx+1]),
                           node_mask=copy.deepcopy(self.node_mask[idx:idx+1]), 
                           atom_map_numbers=copy.deepcopy(self.atom_map_numbers[idx:idx+1]), 
                           mol_assignment=copy.deepcopy(self.mol_assignment[idx:idx+1]))
    
    def select_by_batch_and_sample_idx(self, bs, n_samples, batch_idx, sample_idx):
        assert self.X.ndim==2, f'Expected X of shape (bs, n), got X.shape={self.X.shape}. Use mask(node_mask, collapse=True) before calling this function.'
        assert self.E.ndim==3, f'Expected E of shape (bs, n, n), got E.shape={self.E.shape}. Use mask(node_mask, collapse=True) before calling this function.'
        
        # TODO: For some reason, the one-hot encoding has been dropped out of X and E here. 
        # Makes the entire thing a bit confusing, to be honest
        # TODO: Real bug: the bond_dirs etc. should be encoded in an adjacency matrix format

        return PlaceHolder(X=self.X.reshape(bs, n_samples, self.X.shape[1])[batch_idx:batch_idx+1, sample_idx], 
                           atom_charges=self.atom_charges.reshape(bs, n_samples, self.atom_charges.shape[1])[batch_idx:batch_idx+1, sample_idx],
                           atom_chiral=self.atom_chiral.reshape(bs, n_samples, self.atom_chiral.shape[1])[batch_idx:batch_idx+1, sample_idx],
                           E=self.E.reshape(bs, n_samples, self.E.shape[2], -1)[batch_idx:batch_idx+1, sample_idx], 
                           bond_dirs=self.bond_dirs.reshape(bs, n_samples, self.bond_dirs.shape[2], -1)[batch_idx:batch_idx+1, sample_idx],
                           y=self.y.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx], 
                           node_mask=self.node_mask.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx], 
                           atom_map_numbers=self.atom_map_numbers.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx], 
                           mol_assignment=self.mol_assignment.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx])
 
    def serialize(self):
        return {"X": self.X.detach().cpu().numpy().tolist(),
                "atom_charges": self.atom_charges.detach().cpu().numpy().tolist(),
                "atom_chiral": self.atom_chiral.detach().cpu().numpy().tolist(),
                "E": self.E.detach().cpu().numpy().tolist(),
                "bond_dirs": self.bond_dirs.detach().cpu().numpy().tolist(),
                "y": self.y.detach().cpu().numpy().tolist(), 
                "node_mask": self.node_mask.detach().cpu().numpy().tolist(),
                "atom_map_numbers": self.atom_map_numbers.detach().cpu().numpy().tolist(), 
                "mol_assignment": self.mol_assignment.detach().cpu().numpy().tolist()}
    
    def pyg(self):
        """Turns back into a pytorch geometric DataBatch() object, also with lesser precision for easier saving.
        To turn back to higher precision, there exists a function for that. pyg_to_full_precision_expanded() """
        
        # NOTE: make sure edge_index for bond types and bond directions are consistent
        # i.e. only consider the bond directions for the edges that are present
        edge_bool_mask = self.E!=0
        self.bond_dirs = edge_bool_mask*self.bond_dirs
 
        # TODO: This or the other iteration are unnecessary, but it's okay
        return_data = []
        for i in range(self.X.shape[0]):
            # Concatenate the X as well
            E_idx, E_attr = dense_to_sparse(adj=self.E[i])
            # NOTE: here we take the bond_dirs of all existing edges even if the value is 0
            bond_dirs_attr = torch.zeros_like(E_attr, device=E_attr.device)
            for e in range(E_idx.shape[-1]):
                bond_dirs_attr[e] = self.bond_dirs[i,E_idx[0,e],E_idx[1,e]]
            
            X = self.X[i] if self.X.dim() == 2 else self.X[i].argmax(-1)
            assert X.dim() == 1
            
            atom_charges = self.atom_charges[i] if self.atom_charges.dim() == 2 else self.atom_charges[i].argmax(-1)
            assert atom_charges.dim() == 1
            atom_chiral = self.atom_chiral[i] if self.atom_chiral.dim() == 2 else self.atom_chiral[i].argmax(-1)
            assert atom_chiral.dim() == 1
            
            # X = self.X[i].reshape(-1, self.X.shape[1]) if self.X.dim() == 2 else self.X[i].argmax(-1).reshape(-1, self.X.shape[1])
            atom_map_numbers = self.atom_map_numbers[i]
            node_mask = self.node_mask[i]
            mol_assignment = self.mol_assignment[i]
            pos_encoding = self.pos_encoding[i] if self.pos_encoding is not None else None
            smiles = self.smiles[i] if self.smiles is not None else None

            # NOTE: atom mappings and mol_assignment have a different field names in the Data() objects and in the PlaceHolder objects. Needs to be accommodated here.
            return_data.append(Data(x=X.to(torch.uint8), 
                                    atom_charges=atom_charges.to(torch.uint8),
                                    atom_chiral=atom_chiral.to(torch.uint8),
                                    edge_index=E_idx.to(torch.int16), 
                                    edge_attr=E_attr.to(torch.uint8), 
                                    bond_dirs=bond_dirs_attr.to(torch.uint8),
                                    y=self.y.to(torch.uint8), 
                                    node_mask=node_mask.to(torch.uint8),
                                    atom_map_numbers=atom_map_numbers.to(torch.uint8),
                                    smiles=smiles,
                                    #pos_encoding=pos_encoding, # Don't add the positional encoding here, instead recalculate it later since the pos enc takes up a lot of space
                                    mol_assignment=mol_assignment.to(torch.uint8)))

        return Batch.from_data_list(return_data)

    def to_argmaxed(self):
        return_placeholder = copy.deepcopy(self)
        if len(self.X.shape) == 3:
            return_placeholder.X = return_placeholder.X.argmax(-1)
        if len(self.atom_charges.shape) == 3:
            return_placeholder.atom_charges = return_placeholder.atom_charges.argmax(-1)
        if len(self.atom_chiral.shape) == 3:
            return_placeholder.atom_chiral = return_placeholder.atom_chiral.argmax(-1)
        if len(self.E.shape) == 4:
            return_placeholder.E = return_placeholder.E.argmax(-1)
        if len(self.bond_dirs.shape) == 4:
            return_placeholder.bond_dirs = return_placeholder.bond_dirs.argmax(-1)
        
        return return_placeholder
    
    def add(self, placeholder):
        # Used for summing logits together
        new_placeholder = copy.deepcopy(self)
        new_placeholder.X += placeholder.X
        new_placeholder.atom_charges += placeholder.atom_charges
        new_placeholder.atom_chiral += placeholder.atom_chiral
        new_placeholder.E += placeholder.E
        new_placeholder.bond_dirs += placeholder.bond_dirs
        return new_placeholder
    
    def softmax(self):
        new_placeholder = copy.deepcopy(self)
        new_placeholder.X = F.softmax(new_placeholder.X, dim=-1)
        new_placeholder.atom_charges = F.softmax(new_placeholder.atom_charges, dim=-1)
        new_placeholder.atom_chiral = F.softmax(new_placeholder.atom_chiral, dim=-1)
        new_placeholder.E = F.softmax(new_placeholder.E, dim=-1)
        new_placeholder.bond_dirs = F.softmax(new_placeholder.bond_dirs, dim=-1)
        return new_placeholder

    def log(self):
        # used for taking the log of probabilities
        new_placeholder = copy.deepcopy(self)
        new_placeholder.X = torch.log(new_placeholder.X)
        new_placeholder.atom_charges = torch.log(new_placeholder.atom_charges)
        new_placeholder.atom_chiral = torch.log(new_placeholder.atom_chiral)
        new_placeholder.E = torch.log(new_placeholder.E)
        new_placeholder.bond_dirs = torch.log(new_placeholder.bond_dirs)
        return new_placeholder

    def exp(self):
        # used for exponentiating log-probabilities
        new_placeholder = copy.deepcopy(self)
        new_placeholder.X = torch.exp(new_placeholder.X)
        new_placeholder.atom_charges = torch.exp(new_placeholder.atom_charges)
        new_placeholder.atom_chiral = torch.exp(new_placeholder.atom_chiral)
        new_placeholder.E = torch.exp(new_placeholder.E)
        new_placeholder.bond_dirs = torch.exp(new_placeholder.bond_dirs)
        return new_placeholder
    
    def drop_n_first_nodes(self, n):
        self.X = self.X[:, n:]
        self.E = self.E[:, n:, n:]
        self.atom_charges = self.atom_charges[:, n:]
        self.atom_chiral = self.atom_chiral[:, n:]
        self.bond_dirs = self.bond_dirs[:, n:, n:]
        self.node_mask = self.node_mask[:, n:]
        self.y = self.y
        self.atom_map_numbers = self.atom_map_numbers[:, n:]
        self.mol_assignment = self.mol_assignment[:, n:]
        if self.pos_encoding is not None:
            self.pos_encoding = self.pos_encoding[:, n:]
            
    def split(self, split_size):
        """
        Splits the PlaceHolder object into two smaller PlaceHolder objects.
        Args:
            split_size (int): The size of the first split. The second split will contain the remaining elements.
        Returns:
            tuple: Two PlaceHolder objects.
        """
        assert hasattr(self, 'X') and split_size < self.X.shape[0], "Split size must be less than the batch size."

        # Define attributes to split if they exist
        attrs = ['X', 'E', 'y', 'atom_charges', 'atom_chiral', 'bond_dirs', 'node_mask', 
                'atom_map_numbers', 'mol_assignment', 'pos_encoding', 'smiles']
        
        # Create dictionaries for splits containing only existing attributes
        first_dict = {}
        second_dict = {}
        
        for attr in attrs:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    first_dict[attr] = val[:split_size]
                    second_dict[attr] = val[split_size:]

        return PlaceHolder(**first_dict), PlaceHolder(**second_dict)

def reassign_atom_map_nums(data_placeholder, atom_decoder):
    '''
        Reassign atom map numbers such that all atoms in the product have an atom mapping and
        they are assigned arbitraritly to the reactant side.
        Assumes that data_placeholder.X is in the one-hot format
    '''

    data_placeholder = copy.deepcopy(data_placeholder)
    data_placeholder.atom_map_numbers
    device = data_placeholder.X.device

    suno_idx_in_atom_types = atom_decoder.index('SuNo')
    match_rows, match_columns = (data_placeholder.X.argmax(-1)==suno_idx_in_atom_types).nonzero(as_tuple=True)
    # assert (match_rows == torch.arange(data_placeholder.X.shape[0], device=device)).all(), 'SuNo should be in all rows of the X matrix'
    suno_indices_in_graph = match_columns

    for i, suno_idx in enumerate(suno_indices_in_graph):
        num_atoms_in_product = data_placeholder.node_mask[i, suno_idx+1:].sum()
        data_placeholder.atom_map_numbers[i, suno_idx+1:] = torch.arange(1, data_placeholder.X.shape[1]-suno_idx, device=device)
        data_placeholder.atom_map_numbers[i, suno_idx+1:] = data_placeholder.atom_map_numbers[i, suno_idx+1:] * data_placeholder.node_mask[i, suno_idx+1:]
        data_placeholder.atom_map_numbers[i, :num_atoms_in_product] = torch.arange(1, num_atoms_in_product+1, device=device)
        data_placeholder.atom_map_numbers[i, num_atoms_in_product:suno_idx] = 0

    return data_placeholder

def concatenate_placeholders(placeholder_list):
    new_X = torch.concatenate([p.X for p in placeholder_list], 0)
    new_E = torch.concatenate([p.E for p in placeholder_list], 0)
    new_y = torch.concatenate([p.y for p in placeholder_list], 0)
    new_node_mask = torch.concatenate([p.node_mask for p in placeholder_list], 0)
    new_atom_map_numbers = torch.concatenate([p.atom_map_numbers for p in placeholder_list], 0)
    new_mol_assignment = torch.concatenate([p.mol_assignment for p in placeholder_list], 0)
    
    new_atom_chirals = torch.concatenate([p.atom_chiral for p in placeholder_list], 0)
    new_atom_charges = torch.concatenate([p.atom_charges for p in placeholder_list], 0)
    new_bond_dirs = torch.concatenate([p.bond_dirs for p in placeholder_list], 0)
    new_pos_encoding = torch.cat([p.pos_encoding for p in placeholder_list], 0) if placeholder_list[0].pos_encoding is not None else None
    new_smiles = [s for p in placeholder_list for s in p.smiles]
    
    return PlaceHolder(X=new_X, E=new_E, y=new_y, node_mask=new_node_mask, atom_map_numbers=new_atom_map_numbers, 
                       mol_assignment=new_mol_assignment, atom_charges=new_atom_charges, atom_chiral=new_atom_chirals, 
                       bond_dirs=new_bond_dirs, pos_encoding=new_pos_encoding, smiles=new_smiles)

def json_to_graph(json_dict, x_classes, e_classes):
    graph = PlaceHolder(X=torch.Tensor(json_dict["X"]).to(torch.float32), E=torch.Tensor(json_dict["E"]).to(torch.float32),
                        y=torch.Tensor(json_dict["y"]), node_mask=torch.Tensor(json_dict["node_mask"]).to(torch.bool), 
                        atom_map_numbers=torch.Tensor(json_dict["atom_map_numbers"]).int(), 
                        mol_assignment=torch.Tensor(json_dict["mol_assignment"]).int())   
    
    return graph
        
def select_placeholder_from_chain_by_idx(chains, idx):
    '''
        Select a single placeh object (i.e. single chain) from a batch of chains.
        Chain is of the form [(time_step (int), PlaceHolder obj), (...,...), ...].
        
        chains: chains to select from
        idx: the idx of the chain to select
        
        return: all_chains = acc_chain with placeh selected by batch idx.
    '''
    
    assert len(chains)>0, 'Chain is empty.'
    assert idx<chains[0][1].X.shape[0], f'Cannot choose idx={idx} from a chain of size={chains[0][1].X.shape[0]}.'
    
    return [(time_step, placeh.select_by_batch_idx(idx)) for time_step, placeh in chains]

def select_placeholder_from_chain_by_batch_and_sample(chains, bs, n_samples, batch_idx, sample_idx):
    '''
        Select a single placeh object (i.e. single chain) from a batch of chains.
        Chain is of the form [(time_step (int), PlaceHolder obj), (...,...), ...].
        
        chains: chains to select from
        idx: the idx of the chain to select
        
        return: all_chains = acc_chain with placeh selected by batch idx.
    '''
    
    assert len(chains)>0, 'Chain is empty.'
    
    return [(time_step, placeh.select_by_batch_and_sample_idx(bs, n_samples, batch_idx, sample_idx)) for time_step, placeh in chains]

def resolve_atom_mappings(smiles_list):
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    if all(mol is None for mol in mol_list):
        print("Error: No valid molecules found in the input list.")
        return None, None
    
    aggregated_mappings = aggregate_matchings(mol_list)
    most_likely_mappings = get_unique_most_likely_mappings(aggregated_mappings)
    
    # Apply the most likely mappings to all molecules
    for mol in mol_list:
        for i, mapping in enumerate(most_likely_mappings):
            mol.GetAtomWithIdx(i).SetAtomMapNum(mapping)
    
    return mol_list, most_likely_mappings

def encode_no_element(A):
    '''
        Turns no elements (e.g. from dense padding) to one-hot encoded vectors.
        Works on X and E.
    '''
    assert len(A.shape) >= 3 
    if A.shape[-1]==0: 
        return A
    no_elt = torch.sum(A, dim=-1) == 0
    first_elt = A[..., 0]
    first_elt[no_elt] = 1
    A[..., 0] = first_elt
    return A

def get_batch_subgraph(X, mask_X, E, mask_E):
    '''
        subgraph X and E according to given masks.
    '''
    X_sub = X*mask_X
    X_idx = ~(X_sub.sum(dim=-1)==0) # only take reactant nodes
    X_sub = X_sub[X_idx].reshape(X.shape[0], -1, X.shape[-1])

    E_sub = E*mask_E
    E_idx = ~(E_sub.sum(dim=-1)==0) # only take reactant nodes
    E_sub = E_sub[E_idx].reshape(E_sub.shape[0], X_sub.shape[1], X_sub.shape[1], E_sub.shape[-1])

    return X_sub, E_sub

def batch_graph_by_size(input_data_list, size_bins, batchsize_bins, get_batches=False):
    '''
        Function to generate graph batches depending on the sizes of the graphs. 
        Size bins and corresponding batch sizes are determined for each dataset during preprocessing.

        Input:
            data_list: a list of pyg data objects to load.
        Output:
            list of dataloader for each size bin with a corresponding batch size.
    '''
    assert len(input_data_list)>0, f'Empty data_list.'
    data_by_size = {}
    for data in input_data_list:
        size = data['x'].shape[0]
        upper_size = size_bins[-1] if size > size_bins[-1] else next(s for s in size_bins if size <= s)
        if upper_size in data_by_size.keys():
            data_by_size[upper_size].append(data)
        else:
            data_by_size[upper_size] = [data]
      
    if get_batches:      
        batches = [data for upper_size, data_list in data_by_size.items() for data in iter(DataLoader(data_list, batch_size=batchsize_bins[size_bins.index(upper_size)], shuffle=True))]
    else:
        dataloaders = [DataLoader(data_list, batch_size=batchsize_bins[size_bins.index(upper_size)], shuffle=True) for upper_size, data_list in data_by_size.items()]
   
    return batches, list(data_by_size.keys())

def extract_block_diagonal(a, bs, n, f):
    '''
        Returns the block diagonal matrices from a matrix of blocks of size: (bs*n_nodes, bs*n_nodes).
        Used to create a mask for the adjacency matrix E.

        Input: a(bs*n, bs*n, f)
        output: b(bs, n, n, f)
    '''
    s = (range(bs), np.s_[:], range(bs), np.s_[:], np.s_[:])  # the slices of the blocks
    b = a.reshape(bs, n, bs, n, f)[s]  # reshaping according to blocks and slicing

    return b  # reshape to desired output format

def get_batch_masks(mask_vec, batch_vec, bs, n, v, e, discrete=False):
    '''
        Turn a mask vector of shape (n) to batched masks for X (bs, n, v) and E (bs, n, n, e).
    '''
    # alternative to outer product masking for E
    # 2 masks: mask1.shape=(bs, n, 1, 1), mask2.shape=(bs, 1, n, 1)
    assert mask_vec.ndim==1, f'Expected mask_vec to have 1 dimension: (bs,). Got shape {mask_vec.shape}'
    
    mask, _ = to_dense_batch(x=mask_vec, batch=batch_vec)
    mask_X_discrete = mask.unsqueeze(dim=-1) # (bs, n, 1)
    mask_X = mask_X_discrete.expand(-1, -1, v) # (bs, n ,v)
    mask_X = mask_X.reshape(bs, n, v)

    mask_E = mask.flatten().outer(mask.flatten()) # (bs*n, bs*n)
    mask_E_discrete = extract_block_diagonal(a=mask_E, bs=bs, n=n, f=1) # (bs, n, n, 1)
    mask_E = mask_E_discrete.expand(-1, -1, -1, e) # (bs, n, n, e)
    mask_E = mask_E.reshape(bs, n, n, e)

    if discrete: 
        return mask_X_discrete.squeeze(), mask_E_discrete.squeeze()
    
    return mask_X, mask_E

def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)

def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)

def get_batch_size_of_dataloader(dataloader):
    dense_data = to_dense(data=next(iter(dataloader)))
    return dense_data.X.shape[0]

def get_dataset_len_of_dataloader(dataloader):
    l = len(dataloader.dataset)
    return l 

def permute_placeholder(dense_data):
    dense_data = copy.deepcopy(dense_data)
    for i in range(dense_data.X.shape[0]):
        product_mol_assignment = dense_data.mol_assignment[i].max().item()
        product_selection = (dense_data.mol_assignment[i] == product_mol_assignment)
        product_X = dense_data.X[i, product_selection]
        product_atom_map_numbers = dense_data.atom_map_numbers[i, product_selection]
        perm = torch.randperm(product_X.shape[0])

        # Find the indices where M is True
        indices = torch.nonzero(product_selection).squeeze()
        # Apply the permutation to the indices
        permuted_indices = indices[perm]

        # Easy for these
        dense_data.X[i, product_selection] = product_X[perm]
        dense_data.atom_map_numbers[i, product_selection] = product_atom_map_numbers[perm]

        # For edges, more involved: create a new tensor to store the permuted adjacency matrix
        new_E = dense_data.E[i].clone()
        # Create index grids for the original and permuted indices
        i_grid, j_grid = torch.meshgrid(indices, indices, indexing='ij')
        perm_i_grid, perm_j_grid = torch.meshgrid(permuted_indices, permuted_indices, indexing='ij')
        # Use advanced indexing to perform the permutation in one step
        new_E[i_grid, j_grid] = dense_data.E[i][perm_i_grid, perm_j_grid]
        dense_data.E[i] = new_E
    return dense_data, perm

def fix_others_than_reactant_to_original(cfg, transformed_data, original_data, as_logits=False, include_supernode=True):
    """
    Takes the transformed data and fixes everything but the reactant nodes and edges to the original data. The product nodes and edges are obtained
    with the mol_assignment part
    
    Arguments:
    transformed_data : PlaceHolder object
    original_data : PlaceHolder object
    """
    # assert all(transformed_data.mol_assignment.flatten()==original_data.mol_assignment.flatten()), "The product nodes are not the same in the two data objects"
    # assert original_data.atom_map_numbers is None or all(transformed_data.atom_map_numbers.flatten()==original_data.atom_map_numbers.flatten()), "The atom map numbers are not the same in the two data objects"
    # assert all(transformed_data.node_mask.flatten()==original_data.node_mask.flatten()), "The node masks are not the same in the two data objects"
    # assert original_data.pos_encoding is None or all(transformed_data.pos_encoding.flatten()==original_data.pos_encoding.flatten()), "The positional encodings are not the same in the two data objects"
    assert len(transformed_data.X.shape)==3, "The X tensor should have 3 dimensions"
    assert len(transformed_data.E.shape)==4, "The E tensor should have 4 dimensions"
    
    # TODO: Would be better if the mask that this outputs would not include the feature dimension. -> figure out how much work would it be to change this

    multiplier = 1
    if as_logits: # if we want to return logits for something, we just create a really peaked distribution here
        multiplier = 100
    
    device = transformed_data.node_mask.device
    transformed_data = copy.deepcopy(transformed_data)
    product_mol_indices = transformed_data.mol_assignment.max(dim=1).values
    product_indices = torch.tensor([(transformed_data.mol_assignment[i] == product_mol_indices[i]).nonzero()[0,0].item() for i in range(transformed_data.X.shape[0])], device=device)
    if include_supernode: # backwards compatibility with supernodes
        product_indices -= 1 # assume that the product supernode is before the first node in the product
        # suno_idx = cfg.dataset.atom_types.index('SuNo')
        # product_indices = product_indices | torch.tensor([(transformed_data.X[i].argmax(-1) == suno_idx).nonzero()[0,0].item() for i in range(transformed_data.X.shape[0])], device=device)
    n = transformed_data.X.shape[1]
    bs = transformed_data.X.shape[0]
    fdim_x = transformed_data.X.shape[2]
    fdim_e = transformed_data.E.shape[3]
    # print(f'torch.arange(n)[None,:,None].repeat(bs,1,fdim_x) {torch.arange(n)[None,:,None].repeat(bs,1,fdim_x).device}\n')
    # print(f'product_indices {product_indices.device}')
    arange = torch.arange(n, device=device)
    # print(f'transformed_data.node_mask {transformed_data.node_mask.device}\n')

    # log.info(f"bs:{bs}, product_indices.shape:{product_indices.shape}, product_indices[:,None].repeat(1,n).shape:{product_indices[:,None].repeat(1,n).shape}")
    # log.info(f"(arange[None,:].repeat(bs,1) >= product_indices[:,None].repeat(1,n)).shape:{(arange[None,:].repeat(bs,1) >= product_indices[:,None].repeat(1,n)).shape}")
    # log.info(f"outside_reactant_mask_nodes.shape: {((arange[None,:].repeat(bs,1) >= product_indices[:,None].repeat(1,n)) | (~transformed_data.node_mask)[:,:].repeat(1,1)).shape }")
    outside_reactant_mask_nodes = (arange[None,:].repeat(bs,1) >= product_indices[:,None].repeat(1,n)) \
                                        | (~transformed_data.node_mask)[:,:].repeat(1,1) # picks out everything not in reactants. shape (bs, n, fdim)
    transformed_data.X[outside_reactant_mask_nodes] = original_data.X[outside_reactant_mask_nodes] * multiplier
    transformed_data.atom_charges[outside_reactant_mask_nodes] = original_data.atom_charges[outside_reactant_mask_nodes] * multiplier
    transformed_data.atom_chiral[outside_reactant_mask_nodes] = original_data.atom_chiral[outside_reactant_mask_nodes] * multiplier
    outside_reactant_mask_edges = (arange[None,:,None].repeat(bs,1,n) >= product_indices[:,None,None].repeat(1,n,n)) \
        | (arange[None,None,:].repeat(bs,n,1) >= product_indices[:,None,None].repeat(1,n,n)) # shape (bs, n, n, fdim)
    outside_reactant_mask_edges = outside_reactant_mask_edges | (~transformed_data.node_mask[:,:,None] | ~transformed_data.node_mask[:,None,:])
    transformed_data.E[outside_reactant_mask_edges] = original_data.E[outside_reactant_mask_edges] * multiplier
    transformed_data.bond_dirs[outside_reactant_mask_edges] = original_data.bond_dirs[outside_reactant_mask_edges] * multiplier

    # outside_reactant_mask_nodes = (arange[None,:,None].repeat(bs,1,fdim_x) >= product_indices[:,None,None].repeat(1,n,fdim_x)) \
    #                                     | (~transformed_data.node_mask)[:,:,None].repeat(1,1,fdim_x) # picks out everything not in reactants. shape (bs, n, fdim)
    # transformed_data.X[outside_reactant_mask_nodes] = original_data.X[outside_reactant_mask_nodes] * multiplier
    # transformed_data.atom_charges[outside_reactant_mask_nodes[...,0]] = original_data.atom_charges[outside_reactant_mask_nodes[...,0]] * multiplier
    # transformed_data.atom_chiral[outside_reactant_mask_nodes[...,0]] = original_data.atom_chiral[outside_reactant_mask_nodes[...,0]] * multiplier
    # outside_reactant_mask_edges = (arange[None,:,None,None].repeat(bs,1,n,fdim_e) >= product_indices[:,None,None,None].repeat(1,n,n,fdim_e)) \
    #     | (arange[None,None,:,None].repeat(bs,n,1,fdim_e) >= product_indices[:,None,None,None].repeat(1,n,n,fdim_e)) # shape (bs, n, n, fdim)
    # outside_reactant_mask_edges = outside_reactant_mask_edges | (~transformed_data.node_mask[:,:,None,None] | ~transformed_data.node_mask[:,None,:,None])
    # transformed_data.E[outside_reactant_mask_edges] = original_data.E[outside_reactant_mask_edges] * multiplier
    # transformed_data.bond_dirs[outside_reactant_mask_edges[...,0]] = original_data.bond_dirs[outside_reactant_mask_edges[...,0]] * multiplier

    return transformed_data, outside_reactant_mask_nodes, outside_reactant_mask_edges

def apply_mask(cfg, orig, z_t, atom_decoder, bond_decoder, mask_nodes=None,
               mask_edges=None, node_states_to_mask=[], 
               edge_states_to_mask=[], as_logits=False, return_masks=False,
               include_supernode=True):
    '''
        Apply a mask to fix some of the values of z_t to the values of orig.
        
        input:
            orig: original data.
            z_t: data to be masked.
            mask: mask vector of shape (n).
            (optional) n_samples: number of samples if the data is duplicated.
            include_supernode: whether to include the supernode in the mask or not. 
                Used in classifier-free guidance to drop out parts of the reaction, 
                but not the supernode
            
        output:
            z_t: masked data.

        NOTE: This function used to have the side effect of changing z_t directly. Doesn't anymore,
        but this shouldn't cause bugs since it was always used with z_t = apply_mask(...)
    '''

    z_t_, mask_X, mask_E = fix_others_than_reactant_to_original(cfg, z_t, orig, as_logits=as_logits, include_supernode=include_supernode)
    if return_masks:
        return z_t_, mask_X, mask_E
    else:
        return z_t_

    device = orig.X.device
    mask_X, mask_E = get_mask(orig=orig, atom_decoder=atom_decoder, bond_decoder=bond_decoder, 
                              mask_nodes=mask_nodes, mask_edges=mask_edges, 
                              node_states_to_mask=node_states_to_mask, 
                              edge_states_to_mask=edge_states_to_mask,
                              include_supernode=include_supernode)
    #print(f'mask.device={mask_X.device}, z_t.X.device={z_t.X.device}, orig.X.device={orig.X.device}\n')
    z_t_ = z_t.get_new_object()
    
    # TODO: What do we need this mask for here? We don't really need to fix the products with this masking logic anymore
    # Right now this has the side effect that it may pick up edges that were incorrectly placed outside the graph (padding nodes)

   #z_t_ = graph.PlaceHolder(X=z_t.X.clone(), E=z_t.E.clone(), y=z_t.y.clone(), node_mask=z_t.node_mask.clone(), atom_map_numbers=z_t.atom_map_numbers)
    if as_logits:
        # if as_logits=True, then the orig.X one hot encodings are turned to logits
        # TODO: This is a bit hacky, it depends on the scale of the logits. Better if they were normalized in advance
        z_t_.X[~mask_X], z_t_.E[~mask_E] = orig.X[~mask_X]*100, orig.E[~mask_E]*100
        z_t_.X[mask_X], z_t_.E[mask_E] = z_t.X[mask_X], z_t.E[mask_E]
        # z_t_.X = z_t_.X*mask_X+orig.X*(~mask_X)*100
        # z_t_.E = z_t_.E*mask_E+orig.E*(~mask_E)*100
    else:
        # NOTE: In case of NaN values that we want to get rid of, we can't use summing X*mask_X + orig.X*(~mask_X)
        z_t_.X[~mask_X], z_t_.E[~mask_E] = orig.X[~mask_X], orig.E[~mask_E]
        z_t_.X[mask_X], z_t_.E[mask_E] = z_t.X[mask_X], z_t.E[mask_E]
        # z_t_.X = z_t_.X*mask_X+orig.X*(~mask_X)
        # z_t_.E = z_t_.E*mask_E+orig.E*(~mask_E)
        
    if return_masks:
        return z_t_, mask_X, mask_E
    
    return z_t_
        
def get_mask(orig, atom_decoder, bond_decoder, mask_nodes, mask_edges, node_states_to_mask,
             edge_states_to_mask, include_supernode, return_mask_nodes=False):
    '''
        Get a mask vector of shape (n) to fix some of the values of z_t to the values of orig. Picks out the parts of orig that should be fixed. 
        
        include_supernode: whether to include the supernode in the mask or not. 
                Used in classifier-free guidance to drop out parts of the reaction, 
                but not the supernode. (a special case even if we usually mask out the supernode)
    
        Outputs:
            mask_X: mask for nodes. Shape (bs, n, v)
            mask_E: mask for edges. Shape (bs, n, n, e)
    '''
    device = orig.X.device
    
    # get structure-based masks
    mask = orig.node_mask.clone()
    mask_x = mask.clone()
    mask_e = mask.clone()

    # if mask_nodes=='product' or mask_edges=='product':
    #     mask = get_mask_product(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
    #     mask_x = mask.clone()
    #     mask_e = mask.clone()
    # elif mask_nodes=='reactant' or mask_edges=='reactant':
    #     mask = get_mask_reactant(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
    #     mask_x = mask.clone()
    #     mask_e = mask.clone()
    # elif mask_nodes=='sn' or mask_edges=='sn':
    #     mask = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
    #     mask_x = mask.clone()
    #     mask_e = mask.clone()
    # elif mask_nodes=='product_and_sn' or mask_edges=='product_and_sn':
    #     mask_1 = get_mask_product(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
    #     mask_2 = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
    #     mask = mask_1 * mask_2
    #     mask_x = mask.clone()
    #     mask_e = mask.clone()
    # elif mask_nodes=='reactant_and_sn' or mask_edges=='reactant_and_sn':
    #     mask_1 = get_mask_reactant(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
    #     mask_2 = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
    #     mask = mask_1 * mask_2
    #     mask_x = mask.clone()
    #     mask_e = mask.clone()
    # elif mask_nodes=='atom_mapping' or mask_edges=='atom_mapping':
    #     # NOTE: it does not make sense to use the atom_mapping without conditioning on products and SN.
    #     assert 'atom_map_numbers' in orig.__dir__(), 'Masking atom mapping is None in orig.'
    #     mask_1 = orig.atom_map_numbers == 0 # noise out the ones that don't have atom mapping
    #     mask_2 = get_mask_product(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
    #     mask_3 = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
    #     mask = mask_1 * mask_2 * mask_3
    #     mask_x = mask.clone()
    #     mask_e = mask_2 * mask_3
    # else:
    #     # mask for padding nodes
    #     # TODO: Are we only masking the padding nodes in this case?
    #     mask = orig.node_mask.clone()
    
    assert mask.shape==(orig.X.shape[0], orig.X.shape[1]), 'Something is wrong with the mask. Should have shape (bs, n_max).'

    # TODO: The following may not really be needed now that we don't have supernodes or supernode edges
    node_idx_to_mask, edge_idx_to_mask = get_index_from_states(atom_decoder=atom_decoder, bond_decoder=bond_decoder, node_states_to_mask=node_states_to_mask, 
                                                               edge_states_to_mask=edge_states_to_mask, device=device)
    
    # get state-based masks
    # the logic here is that if a state is fixed
    # then all the nodes/edges with that state are fixed
    # fixing/masking = taking the value from the original data
    mask_states = torch.ones_like(mask_x, dtype=torch.bool)
    for i in node_idx_to_mask:
        mask_states *= torch.where(orig.X.argmax(-1)==i, False, True)
        
    mask_x = mask_x*mask_states
    mask_e = mask_e*mask_states
    
    if return_mask_nodes: return mask
    mask_X, mask_E = from_mask_to_maskX_maskE(mask_nodes=mask_x, mask_edges=mask_e, shape_X=orig.X.shape, shape_E=orig.E.shape)

    return mask_X, mask_E
    
def get_mask_sn(origX, atom_decoder, device):
    '''
        Get mask for SuNo nodes.
    '''
    
    if origX.dim()==3: origX = origX.argmax(-1)
        
    # get index of SuNo
    suno_idx = atom_decoder.index('SuNo')
    # get indices of the SuNo nodes => idx along 0 dim (batch): (total # SuNo nodes,)
    # idx along 1 dim (index of node in batch element): (total # SuNo nodes,)
    suno_indices = (origX==suno_idx).nonzero(as_tuple=True)
    # turn SuNo indices to batched format: (bs, max # of SuNo nodes)
    suno_indices_batched, _ = to_dense_batch(x=suno_indices[1], batch=suno_indices[0]) # (bs, n)
    # make padding node match the index of the last SuNo nodes
    suno_indices_batched[:,-1] = suno_indices_batched.max(-1)[0]
    # get suno_indices_batched to the same shape as orig.X
    repeats = torch.ones(suno_indices_batched.shape[-1]).int().to(device)
    repeats[-1] = origX.shape[1] - suno_indices_batched.shape[-1] + 1
    suno_indices_batched = suno_indices_batched.repeat_interleave(repeats=repeats, dim=-1)
    # get flat indices of suno_indices_batched to be able to index mask in specific locations
    flat_idx = torch.arange(origX.shape[0]).repeat_interleave(origX.shape[1], dim=-1).to(device)
    corr_idx = flat_idx*origX.shape[1] + suno_indices_batched.flatten()
    mask = torch.ones_like(origX, dtype=torch.bool).flatten().to(device)
    mask[corr_idx] = False
    mask = mask.reshape(origX.shape[0], -1)
    
    return mask
        
def get_mask_product(origX, atom_decoder, device):
    '''
        Get mask according to the property of the nodes: (bs, n_max).
    '''
    if origX.dim()==3: origX = origX.argmax(-1)
    # get index of SuNo
    suno_idx = atom_decoder.index('SuNo')
    # get indices of the SuNo nodes => idx along 0 dim (batch): (total # SuNo nodes,)
    # idx along 1 dim (index of node in batch element): (total # SuNo nodes,)
    suno_indices = (origX==suno_idx).nonzero(as_tuple=True)
    # turn SuNo indices to batched format: (bs, max # of SuNo nodes)
    suno_indices_batched, _ = to_dense_batch(x=suno_indices[1], batch=suno_indices[0]) # (bs, n)
    # get the index of the last SuNo corresponding to the product molecule
    last_suno, _ = suno_indices_batched.max(-1)
    # ignore the SuNo of the product (design choice)
    last_suno += 1
    # create mask: (bs, n_max)
    mask = torch.ones_like(origX, dtype=torch.bool, device=device)
    # mark last SuNo in mask as False
    mask[:,-1] = False
    # Get everything after last SuNo to be False as well
    # This implicitly also takes care of the padding node masking, since the padding nodes are at the end of the reaction graph
    # TODO: cleaner way to do this?
    rg = torch.arange(origX.shape[1]).unsqueeze(0).expand(origX.shape[0], -1).to(device)
    last_idx = (origX.shape[1]-1)*torch.ones_like(origX).to(device)
    v = last_suno.unsqueeze(-1).expand(-1, origX.shape[1]).to(device)
    idx = torch.where(rg<v, rg, last_idx)
    mask = torch.gather(mask, 1, idx)
    
    return mask

def get_mask_reactant(origX, atom_decoder, device):
    '''
        Mask for reactants.
    '''
    if origX.dim()==3: origX = origX.argmax(-1)
    
    mask_prod = get_mask_product(origX, atom_decoder, device)
    mask_sn = get_mask_sn(origX, atom_decoder, device)
    
    return ~(mask_prod*mask_sn)

def get_all_matches(mol1, mol2):
    return mol2.GetSubstructMatches(mol1, uniquify=False)
    
def score_matching(mol1, mol2, matching):
    score = 0
    for i, j in enumerate(matching):
        atom1 = mol1.GetAtomWithIdx(i)
        atom2 = mol2.GetAtomWithIdx(j)
        if atom1.GetAtomMapNum() == atom2.GetAtomMapNum() and atom1.GetAtomMapNum() != 0:
            score += 1
    return score
    
def aggregate_matchings(mol_list):
    if not mol_list:
        return []

    n_atoms = mol_list[0].GetNumAtoms()
    aggregated_mappings = [defaultdict(int) for _ in range(n_atoms)]
    
    # First, aggregate the atom mappings from all molecules
    for mol in mol_list:
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            map_num = atom.GetAtomMapNum()
            if map_num != 0:
                aggregated_mappings[i][map_num] += 1
    
    # Then, aggregate matchings between pairs of molecules
    for i in range(len(mol_list)):
        for j in range(i+1, len(mol_list)):
            matching = get_best_matching(mol_list[i], mol_list[j])
            if matching:
                for k, l in enumerate(matching):
                    atom_i = mol_list[i].GetAtomWithIdx(k)
                    atom_j = mol_list[j].GetAtomWithIdx(l)
                    map_i = atom_i.GetAtomMapNum()
                    map_j = atom_j.GetAtomMapNum()
                    
                    if map_i != 0:
                        aggregated_mappings[k][map_i] += 1
                    if map_j != 0:
                        aggregated_mappings[l][map_j] += 1  # Note the change from k to l here
    
    return aggregated_mappings

def get_best_matching(mol1, mol2):
    matches = get_all_matches(mol1, mol2)
    if not matches:
        return None
    scores = [score_matching(mol1, mol2, match) for match in matches]
    return matches[scores.index(max(scores))]

def from_mask_to_maskX_maskE(mask_nodes, mask_edges, shape_X, shape_E):
    mask_X = mask_nodes.unsqueeze(dim=-1) # (bs, n, 1)
    mask_X = mask_X.expand(-1, -1, shape_X[-1]) # (bs, n ,v)
    mask_X = mask_X.reshape(shape_X)
  
    mask_E = mask_edges.flatten().outer(mask_edges.flatten()) # (bs*n, bs*n)
    mask_E_discrete = extract_block_diagonal(a=mask_E, bs=shape_X[0], 
                                             n=shape_X[1], f=1) # (bs, n, n, 1)
    # shape the mask for E
    mask_E = mask_E_discrete.expand(-1, -1, -1, shape_E[-1]) # (bs, n, n, e)
    mask_E = mask_E.reshape(shape_E)
        
    return mask_X, mask_E

def create_node_and_edge_masks_by_idx(data, node_idx, edge_idx):
    '''
        Return a mask where all nodes and edges with specified indices are set to True and everything else to False.
        To be used in fixing the nodes/edges from the true data during generation (mainly with inpainting).
        
        data: placeholder data.
        node_idx: python list (of lists? for batches) of indices of nodes to fix.
        edge_idx: python list (of lists? for batches) of indicies of edges to fix.
        
        return:
            masks for X and E.
    '''
    assert type(data)==graph.PlaceHolder, 'Expected data to be of type graph.PlaceHolder.'
    assert data.X.ndim==3, f'Expected dense_data.X.shape=(bs, n, dx). Got dense_data.X.shape={data.X.shape}'
    assert data.E.ndim==4, f'Expected dense_data.E.shape=(bs, n, n, de). Got dense_data.E.shape={data.E.shape}'
    
    mask_X = torch.zeros_like(data.X, dtype=torch.bool)[...,0]
    mask_E = torch.zeros_like(data.E, dtype=torch.bool)[...,0]
    
    if node_idx!=None:
        for bs in range(mask_X.shape[0]):
            mask_X[bs, node_idx[bs]] = True
            
    if edge_idx!=None:
        for bs in range(mask_E.shape[0]):
            if type(edge_idx[bs]) == tuple:
                for bond in edge_idx[bs]:
                    mask_E[bs, bond[0], bond[1]] = True
                    mask_E[bs, bond[1], bond[0]] = True
            elif edge_idx[bs] == 'NO_ADDITIONAL_CONNECTIONS':
                mask_E[bs, node_idx] = True
                mask_E[bs, :, node_idx] = True
        
    return mask_X, mask_E

# def fix_nodes_and_edges_by_idx_old(data, node_idx, edge_idx):
#     '''
#         Return a mask where all nodes and edges with specified indices are set to True and everything else to False.
#         To be used in fixing the nodes/edges from the true data during generation (mainly with inpainting).
        
#         data: graph or placeholder data.
#         node_idx: python list (of lists? for batches) of indices of nodes to fix.
#         edge_idx: python list (of lists? for batches) of indicies of edges to fix.
        
#         return:
#             masks for X and E.
#     '''
#     if type(data)!=graph.PlaceHolder:
#         dense_data = graph.to_dense(data)
#     else:
#         dense_data = copy.deepcopy(data)
    
#     assert dense_data.X.ndim==3, f'Expected dense_data.X.shape=(bs, n, dx). Got dense_data.X.shape={dense_data.X.shape}'
#     assert dense_data.E.ndim==4, f'Expected dense_data.E.shape=(bs, n, n, de). Got dense_data.E.shape={dense_data.E.shape}'
    
#     dense_data.mask(dense_data.node_mask, collapse=True)
    
#     mask_X = torch.zeros_like(dense_data.X, dtype=torch.bool)
#     mask_E = torch.zeros_like(dense_data.E, dtype=torch.bool)
    
#     if node_idx!=None:
#         for bs in range(mask_X.shape[0]):
#             mask_X[bs, node_idx[bs]] = True
            
#     if edge_idx!=None:
#         for bs in range(mask_E.shape[0]):
#             for bond in edge_idx[bs]:
#                 mask_E[bs, bond[0], bond[1]] = True
#                 mask_E[bs, bond[1], bond[0]] = True
        
#     return mask_X, mask_E

def fix_nodes_and_edges_by_idx(data_to_fix, data, node_idx, edge_idx):
    '''
        Fix the elements of data_to_fix with specified indices to their values in data.
        Used in fixing the nodes/edges from the true data during generation (mainly with inpainting).
        
        data_to_fix: placeholder data, e.g., an intermediate sample from the model
        data: placeholder data.
        node_idx: python list (of lists? for batches) of indices of nodes to fix.
        edge_idx: python list (of lists? for batches) of indicies of edges to fix.
        
        return:
            masks for X and E.
    '''
    mask_X, mask_E = create_node_and_edge_masks_by_idx(data, node_idx, edge_idx)
    data_to_fix = copy.deepcopy(data_to_fix)
    if mask_X.sum() > 0: # check if there's anything to be masked
        data_to_fix.X[mask_X] = data.X[mask_X]
    if mask_E.sum() > 0:
        data_to_fix.E[mask_E] = data.E[mask_E]
    data_to_fix.X[~mask_X], data_to_fix.E[~mask_E] = data_to_fix.X[~mask_X], data_to_fix.E[~mask_E]
    return data_to_fix, mask_X, mask_E
        
def get_product_mask_old(orig, batch):
    # TODO: change for loop for efficiency reasons
    # think of how to make generating masks possible at a batch level
    suno_idx = 28
    for j in range(orig.X.shape[0]): 
        # get the indices of the last SuNo in the sequence => product
        suno_indices = (orig.X[j,...].argmax(dim=-1)==suno_idx).nonzero(as_tuple=True)[0]
        # set the product nodes to False
        mask_vec_ = torch.ones_like(orig.X[j,...].argmax(dim=-1), dtype=torch.bool)
        mask_vec_[suno_indices[-1]:] = False
        # remove padding if used
        padding_indices = (orig.X[j,...].argmax(dim=-1)==0).nonzero(as_tuple=True)[0]
        if padding_indices.shape[0]>0: mask_vec_ = mask_vec_[:padding_indices[0]]
        mask_vec = torch.cat((mask_vec, mask_vec_), dim=0) if j>0 else mask_vec_

    mask_X, mask_E = get_batch_masks(mask_vec=mask_vec, batch_vec=batch, bs=orig.X.shape[0], n=orig.X.shape[1], 
                                     v=orig.X.shape[-1], e=orig.E.shape[-1], discrete=False)

    return mask_X, mask_E

def get_index_from_states(atom_decoder, bond_decoder, node_states_to_mask, edge_states_to_mask, device):
    if not 'none' in node_states_to_mask:
        node_states_to_mask.append('none')
    if node_states_to_mask!=None:
        not_in_list = [a for a in node_states_to_mask if a not in atom_decoder]
        assert len(not_in_list)==0, f'node_states_to_mask contains states not in atom_decoder: {not_in_list}'
        idx_of_states = [atom_decoder.index(a) for a in node_states_to_mask]
        node_idx_to_mask = torch.tensor(idx_of_states, dtype=torch.long, device=device)
    else:
        node_idx_to_mask = torch.empty((1,), dtype=torch.long, device=device)
        
    if edge_states_to_mask!=None:
        not_in_list = [a for a in edge_states_to_mask if a not in bond_decoder]
        assert len(not_in_list)==0, f'edge_states_to_mask contains states not in bond_decoder: {not_in_list}'
        idx_of_states = [bond_decoder.index(a) for a in edge_states_to_mask]
        edge_idx_to_mask = torch.tensor(idx_of_states, dtype=torch.long, device=device)  
    # don't mask none in edges, used as masking state
    else:
        #edge_idx_to_mask = torch.empty((1,), dtype=torch.long, device=device)
        edge_idx_to_mask = None
     
    return node_idx_to_mask, edge_idx_to_mask

def filter_small_molecules(elbo_sorted_rxns, filter_limit=1):
    elbo_sorted_rxns = copy.deepcopy(elbo_sorted_rxns)
    generated_data = list(elbo_sorted_rxns.values())
    for generated_reactions_for_product in generated_data:
        for gen_reaction in generated_reactions_for_product:
            gen_reaction['rcts'] = [rct for rct in gen_reaction['rcts'] if len(rct) > filter_limit]
    return elbo_sorted_rxns

def reactions_sorted_with_weighted_prob(restructured_data, lambda_value):
    """
    Sorts generated reactions based on probability calculated from ELBO and counts
    input: restructured_data = {'prod': [{'rcts': ['ABC', 'CD', ...], 'prod': ['CC'], 'elbo': float, 'loss_t': float, 'loss_0': float, 'count': int/float}, {}, {}, ...]

    output: sorted version of the restructured_data data structure, where the list of reactions for each product is sorted based on the weighted probability of each reaction
    The weighted probability is obtained with the formula lambda * p_ELBO(x) + (1-lambda) * p_count(x), where p_ELBO(x) is 
    the normalized probability of the reaction based on the ELBO, and p_count(x) is the normalized probability of the reaction based on the counts
    """

    # Example usage:
    # lambda_value = 0.5  # Replace with your actual lambda value
    # restructured_data = {
    #     'CC': [{'rcts': 'A.A', 'prod': 'CC', 'elbo': 1, 'loss_t': 2, 'loss_0': 3, 'count': 4},
    #            {'rcts': 'B.B', 'prod': 'CC', 'elbo': 5, 'loss_t': 6, 'loss_0': 7, 'count': 8}],
    #     'DD': [{'rcts': 'X.X', 'prod': 'DD', 'elbo': 9, 'loss_t': 10, 'loss_0': 11, 'count': 12},
    #            {'rcts': 'Y.Y', 'prod': 'DD', 'elbo': 13, 'loss_t': 14, 'loss_0': 15, 'count': 16}]
    # }
    # output = calculate_weighted_prob(restructured_data, lambda_value)
    # print(output)

    restructured_data = copy.deepcopy(restructured_data)

    for product, reactions_list in restructured_data.items():
        # Calculate the sum of exp(-elbo) and sum of counts for the current list
        # use np.exp() because it handles overflow more gracefully\
        sum_exp_elbo = sum(np.exp(-( 0 if np.isnan(reaction['elbo']) else reaction['elbo'])) for reaction in reactions_list)
        sum_counts = sum(reaction['count'] for reaction in reactions_list)

        # Calculate the weighted probability for each reaction and add it to the dictionary
        for reaction in reactions_list:
            exp_elbo = np.exp(-(0 if np.isnan(reaction['elbo']) else reaction['elbo']))
            weighted_prob = (exp_elbo / sum_exp_elbo) * lambda_value + (reaction['count'] / sum_counts) * (1 - lambda_value)
            reaction['weighted_prob'] = weighted_prob

        # Sort the list of reactions for the current product based on weighted_prob
        restructured_data[product] = sorted(reactions_list, key=lambda x: x['weighted_prob'], reverse=True)

    return restructured_data

def calculate_top_k(cfg, elbo_sorted_rxns, true_rcts, true_prods):
    # TODO: 
    # Previously this thing was used in two ways:
    # 1. true_rcts and true_prods has an entry for each individiual sample (wrong)
    # 2. true_rcts and true_prods has an entry for each individual condition (correct)
    # elbo_sorted_rxns format: sorted list from most relevant to least relevant, with each element 
    # being a dictionary {'rcts': [rct1, rct2, ...], 'prod': [prod], 
    #                     'elbo': float, 'loss_t': float, 'loss_0': float, 'count': int}
    # -> need to create true_rcts, true_prods from somewhere
    topk = {}
    true_smiles = [set(r).union(set(p)) for r,p in zip(true_rcts,true_prods)] # why a set here?
    bs = len(elbo_sorted_rxns.keys())

    # compute topk accuracy

    topk = torch.zeros((bs, len(cfg.test.topks)), dtype=torch.float)
    for i, prod in enumerate(elbo_sorted_rxns.keys()): # This goes over the batch size
        for j, k_ in enumerate(cfg.test.topks):
            topk[i,j] = (set(true_smiles[i]) in [set(s['rcts']).union(s['prod']) for s in elbo_sorted_rxns[prod][:k_]]) # is the true smiles in the topk for each product?
    
    return topk

def split_reactions_to_reactants_and_products(reactions):
    rcts = []
    prods = []
    for r in reactions:
        reaction = rdChemReactions.ReactionFromSmarts(r, useSmiles=True)
        reactants = reaction.GetReactants()
        products = reaction.GetProducts()
        rcts.append([Chem.MolToSmiles(r) for r in reactants]) # shld handle invalid molecules automatically
        prods.append([Chem.MolToSmiles(p) for p in products])
    return rcts, prods

def save_samples_to_file_without_weighted_prob(filename, condition_idx, gen_rxns, true_rxns, overwrite=False):
    '''
    Save generated reactions to a txt file without weighted probability.    
    filename: name of txt file where to output the samples.
    condition_idx: the condition number
    gen_rxns: list of generated reactions
    true_rxns: list of true reactions
    overwrite: if True, overwrite the file if it exists, otherwise append to the file
    '''
    # This is used mainly to handle with some old evaluations, deprecated aside from that
    # TODO: Is n_samples really needed here? It seems that it may cause bugs if we are not careful when removing duplicate samples
    if overwrite:
        file = open(filename,'w')
    elif os.path.exists(filename):
        file = open(filename,'a') 
    else:
        file = open(filename,'w')
    # file = open(filename,'w') if condition_idx==0 else open(filename,'a') 
    for i, p in enumerate(gen_rxns):
        lines = [f'(cond {condition_idx + i}) {true_rxns[i]}:\n'] + \
                [f'\t{(mol.rxn_list_to_str(x["rcts"], x["prod"]), [x["elbo"], x["loss_t"], x["loss_0"], x["count"]])}\n' for x in gen_rxns[p]]
        file.writelines(lines)
        # file.write(f'(cond {condition_idx + i}) {mol.rxn_list_to_str(true_rcts[i], true_prods[i])}:\n') # *n_samples because we have n_samples per condition
        # file.writelines([f'\t{(mol.rxn_list_to_str(x["rcts"], x["prod"]), [x["elbo"], x["loss_t"], x["loss_0"], x["count"]])}\n' for x in gen_rxns[p]])
    file.close()
    
def save_gen_rxn_smiles_to_file(filename, condition_idx, gen_rxns, true_rcts, true_prods):
    '''
        save rxns to a txt file in the following format: (cond i) true_rxn:\n\tsample1\n\tsample2...
        
        true_rcts, true_prods: lists of the components of the true reaction split into rcts and prods
        gen_rxns: list of generated reactions
        condition_idx: the condition number
        filename: name of txt file where to output the samples.
    '''
    file = open(filename,'w') if condition_idx==0 else open(filename,'a') 
    file.write(f'(cond {condition_idx}) {mol.rxn_list_to_str(true_rcts, true_prods)}:\n') 
    file.writelines([f'\t{p}\n' for p in gen_rxns])
    file.close()

def save_gen_rxn_pyg_to_file(filename, gen_rxns_pyg, true_rxns_pyg):
    # file = open(filename,'wb')
    # gzip.compress(data, compresslevel=9, mtime=None)
    with gzip.open(filename, 'wb') as file:
        pickle.dump({"gen": gen_rxns_pyg, "true": true_rxns_pyg}, file)
    # file.close()

def save_samples_to_file(filename, condition_idx, gen_rxns, true_rxns, overwrite=False):
    '''
    Save generated reactions to a txt file without weighted probability.    
    filename: name of txt file where to output the samples.
    condition_idx: the condition number
    gen_rxns: list of generated reactions
    true_rxns: list of true reactions
    overwrite: if True, overwrite the file if it exists, otherwise append to the file
    '''
    # TODO: Is n_samples really needed here? It seems that it may cause bugs if we are not careful when removing duplicate samples
    # TODO: This shouldn't take as input condition_idx. I guess it is supposed to handle the case where batch size > 1?
    if overwrite:
        file = open(filename,'w')
    elif os.path.exists(filename):
        file = open(filename,'a')
    else:
        file = open(filename,'w')
    for i, p in enumerate(gen_rxns):
        lines = [f'(cond {condition_idx + i}) {true_rxns[i]}:\n'] + \
                [f'\t{(mol.rxn_list_to_str(x["rcts"], x["prod"]), [x["elbo"], x["loss_t"], x["loss_0"], x["count"], x["weighted_prob"]])}\n' for x in gen_rxns[p]]
        file.writelines(lines)
        # file.write(f'(cond {condition_idx + i}) {mol.rxn_list_to_str(true_rcts[i], true_prods[i])}:\n') # *n_samples because we have n_samples per condition
        # file.writelines([f'\t{(mol.rxn_list_to_str(x["rcts"], x["prod"]), [x["elbo"], x["loss_t"], x["loss_0"], x["count"], x["weighted_prob"]])}\n' for x in gen_rxns[p]])
    file.close()

def save_samples_to_file_with_overwrite(filename, gen_rxns, true_rcts, true_prods):
    file = open(filename,'w')
    for i, p in enumerate(gen_rxns):
        lines = [f'(cond {i}) {mol.rxn_list_to_str(true_rcts[i], true_prods[i])}:\n'] + \
                [f'\t{(mol.rxn_list_to_str(x["rcts"], x["prod"]), [x["elbo"], x["loss_t"], x["loss_0"], x["count"], x["weighted_prob"]])}\n' for x in gen_rxns[p]]
        file.writelines(lines)
    file.close()