import copy
import numpy as np
import random 
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from collections import Counter

import pickle
import math
import os
import gzip

import logging
log = logging.getLogger(__name__)

from src.utils import graph
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops, dense_to_sparse
# from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

torch.multiprocessing.set_sharing_strategy('file_system')

from src.utils import mol

bond_types = ['none', BT.SINGLE, BT.DOUBLE, BT.TRIPLE, 'mol', 'within', 'across']

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ATOMS_RXN = 300
DUMMY_RCT_NODE_TYPE = 'U'

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
                                                                           bond_types=bond_types, with_explicit_h=cfg.dataset.with_explicit_h,
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
    g_edges_attr_dummy = torch.zeros([0, len(bond_types)], dtype=torch.long)
    mask_product_and_sn[:g_nodes_rct.shape[0]+g_nodes_dummy.shape[0]] = True
    mol_assignment[offset:offset+g_nodes_dummy.shape[0]] = 0
    offset += g_nodes_dummy.shape[0]
    
    supernodes_prods = []
    for j, p in enumerate(products):
        # NOTE: still need supernode for product to distinguish it from reactants
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = mol.rxn_to_graph_supernode(mol=p, atom_types=cfg.dataset.atom_types, bond_types=bond_types,
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

def get_graph_data_from_product_smi(product_smi, cfg):
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
    g_edges_attr_dummy = torch.zeros([0, len(bond_types)], dtype=torch.long)
    
    mask_product_and_sn[:g_nodes_dummy.shape[0]] = True
    mol_assignment[:g_nodes_dummy.shape[0]] = 0
    # mask_atom_mapping[:g_nodes_dummy.shape[0]] = MAX_ATOMS_RXN
    offset = g_nodes_dummy.shape[0]
    #offset = 0
            
    supernodes_prods = []
    # NOTE: still need supernode for product to distinguish it from reactants
    g_nodes_prod, g_edge_index_prod, g_edge_attr_prod, atom_map = mol.rxn_to_graph_supernode(mol=product_smi, atom_types=cfg.dataset.atom_types, bond_types=bond_types,
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
    
    return graph

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
    # if type(data)!=PlaceHolder:
    #     dense_data = to_dense(data=data)
    # else:
    #     dense_data = data
        
    if mask_: dense_data = dense_data.mask(dense_data.node_mask) #     
    dense_data.X = dense_data.X.repeat_interleave(n_samples, dim=0) # (bs, n, v) => (bs*n_samples, n, v)
    dense_data.E = dense_data.E.repeat_interleave(n_samples, dim=0) # (bs, n, n, e) => (bs*n_samples, n, n, e)
    dense_data.node_mask = dense_data.node_mask.repeat_interleave(n_samples, dim=0)
    dense_data.y = dense_data.y.repeat_interleave(n_samples, dim=0)
    if dense_data.atom_map_numbers is not None:
        dense_data.atom_map_numbers = dense_data.atom_map_numbers.repeat_interleave(n_samples, dim=0)
    if dense_data.mol_assignments is not None:
        dense_data.mol_assignments = dense_data.mol_assignments.repeat_interleave(n_samples, dim=0)

    if get_discrete_data:
        dense_data_discrete = copy.deepcopy(dense_data).mask(dense_data.node_mask, collapse=True)
        return dense_data, dense_data_discrete
    
    return dense_data

def get_unique_indices_from_reaction_list(gen_rxn_smiles):
    """
        Remove duplicates from data.
    Input: 
        gen_rxn_smiles: list of SMILES strings.
    Output:
        data_placeholder: PlaceHolder object with duplicate reactions removed.
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

    return sorted_unique_indices, sorted_counts, is_unique

    # get unique rows
    # X_unique, idx_unique = torch.unique(data_placeholder.X, dim=0, return_inverse=True)
    # E_unique = data_placeholder.E[idx_unique]
    # node_mask_unique = data_placeholder.node_mask[idx_unique]
    # y_unique = data_placeholder.y[idx_unique]
    # atom_map_numbers_unique = data_placeholder.atom_map_numbers[idx_unique]
    # mol_assignments_unique = data_placeholder.mol_assignments[idx_unique]
    
    # return PlaceHolder(X=X_unique, E=E_unique, y=y_unique, node_mask=node_mask_unique, 
    #                    atom_map_numbers=atom_map_numbers_unique, mol_assignments=mol_assignments_unique)

def to_dense(data):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch)
    X = encode_no_element(X)
    
    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    try:
        E = to_dense_adj(edge_index=edge_index, batch=data.batch, 
                         edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    except Exception as e:
        print(f"An error occurred: {e}")
        
    E = encode_no_element(E)

    atom_map_numbers, mol_assignments = None, None
    keys =  data.keys if type(data.keys)==dict or type(data.keys)==list else data.keys() # TODO: This seems quite hacky at the moment
    if 'mask_atom_mapping' in keys:
        atom_map_numbers, _ = to_dense_batch(x=data.mask_atom_mapping, batch=data.batch)
    # if 'atom_map_numbers' in keys:
    #     # atom_map_numbers, _ = to_dense_batch(x=data.mask_atom_mapping, batch=data.batch)
    #     atom_map_numbers = data.atom_map_numbers # were these of this shape? -> let's check
    # if 'mol_assignment' in keys:
    #     mol_assignments = data.mol_assignment
    if 'mol_assignment' in keys: # For the original pyg objects, it is called mol_assignment, and not mol_assignments
        mol_assignments, _ = to_dense_batch(x=data.mol_assignment, batch=data.batch)
    
    return PlaceHolder(X=X, E=E, y=data.y, node_mask=node_mask, atom_map_numbers=atom_map_numbers, mol_assignments=mol_assignments)

def concatenate_databatches(cfg, list_of_databatches):
    # Concatenates a list of DataBatches together
    concatenated = []
    for i in range(len(list_of_databatches)):
        concatenated.extend(list_of_databatches[i].to_data_list())
    return Batch.from_data_list(concatenated)

def pyg_to_full_precision_expanded(data, atom_types):
    """Reverses the encoding of the data to full precision after encoding to pyg format and saving to pickle.
    data is a DataBatch object. 
    Also expands out x to the one-hot encoding."""
    new_data = copy.deepcopy(data)
    new_data.x = F.one_hot(new_data.x.long(), len(atom_types))
    new_data.edge_attr = F.one_hot(new_data.edge_attr.long(), len(graph.bond_types))
    # new_data.edge_attr = data.edge_attr.long()
    new_data.edge_index = new_data.edge_index.long()
    new_data.y = new_data.y.float()
    new_data.node_mask = new_data.node_mask.bool()
    new_data.mask_atom_mapping = new_data.mask_atom_mapping.long()
    new_data.mol_assignment = new_data.mol_assignment.long()
    return new_data

class PlaceHolderWithoutY:
    def __init__(self, X, E):
        self.X = X
        self.E = E

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)

        return self

    def mask(self, node_mask=None, collapse=False):
        x_node_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_node_mask1 = x_node_mask.unsqueeze(2)             # bs, n, 1, 1
        e_node_mask2 = x_node_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1) # (bs, n)
            self.E = torch.argmax(self.E, dim=-1) # (bs, n, n)

            self.X[node_mask == 0] = 0
            self.E[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0
        else:
            # always mask by node, masking by subgraph is a subset of that
            self.X = self.X * x_node_mask
            self.E = self.E * e_node_mask1 * e_node_mask2
            self.X = encode_no_element(self.X)
            self.E = encode_no_element(self.E)

            # adjacency matrix of undirected graph => mirrored over the diagonal
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self
    
class PlaceHolder:
    def __init__(self, X, E, y, node_mask=None, atom_map_numbers=None, mol_assignments=None):
        self.X = X
        self.E = E
        self.y = y
        self.node_mask = node_mask
        self.atom_map_numbers = atom_map_numbers
        self.mol_assignments = mol_assignments

    def flatten(self, start_dim, end_dim):
        '''
            return a placeholder object with the first idx batch elements.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
     
        new_kwargs = {at: getattr(self, at).flatten(start_dim=start_dim, end_dim=end_dim) if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}
        
        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]
        
        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj
    
    def reshape_bs_n_samples(self, bs, n_samples, n):
        self.X = self.X.reshape(bs, n_samples, n)
        self.E = self.E.reshape(bs, n_samples, n, n)
        self.y = torch.empty((bs, n_samples))
        self.node_mask = self.node_mask.reshape(bs, n_samples, n)
        self.atom_map_numbers = self.atom_map_numbers.reshape(bs, n_samples, n)
        self.mol_assignments = self.mol_assignments.reshape(bs, n_samples, n)
                                               
    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.type_as(x)
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.type_as(x)
        return self
    
    def to_device(self, device):
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device)
        self.node_mask = self.node_mask.to(device)
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.to(device)
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.to(device)
        return self
    
    def to_numpy(self):
        self.X = self.X.detach().cpu().numpy()
        self.E = self.E.detach().cpu().numpy()
        self.y = self.y.detach().cpu().numpy()
        self.node_mask = self.node_mask.detach().cpu().numpy()
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.detach().cpu().numpy()
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.detach().cpu().numpy()
        return self
    
    def to_cpu(self):
        self.X = self.X.detach().cpu()
        self.E = self.E.detach().cpu()
        self.y = self.y.detach().cpu()
        self.node_mask = self.node_mask.detach().cpu()
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.detach().cpu()
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.detach().cpu()
        return self

    def mask(self, node_mask=None, collapse=False):
        if node_mask==None:
            node_mask = self.node_mask
            
        assert node_mask is not None, 'node_mask is None.'
            
        x_node_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_node_mask1 = x_node_mask.unsqueeze(2)            # bs, n, 1, 1
        e_node_mask2 = x_node_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1) # (bs, n)
            self.E = torch.argmax(self.E, dim=-1) # (bs, n, n)

            self.X[node_mask == 0] = 0
            self.E[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0
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
            self.E = self.E * e_node_mask1 * e_node_mask2
            diag = torch.eye(self.E.shape[1], dtype=torch.bool).unsqueeze(0).expand(self.E.shape[0], -1, -1)
            self.E[diag] = 0
            self.X = encode_no_element(self.X)
            self.E = encode_no_element(self.E)

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
        
        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]
        
        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj

    def select_subset(self, selection):
        '''
            return a placeholder object with the selection in the form of a boolean mask of shape (bs,)
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
     
        new_kwargs = {at: getattr(self, at).clone()[selection] if isinstance(getattr(self, at), torch.Tensor) else None for at in attributes}
        
        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]
        
        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'
        
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
        
        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]
        
        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'
        
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
        
        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]
        
        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'
        
        new_obj = PlaceHolder(**new_kwargs)
        
        return new_obj
    
    def cat_by_batchdim(self, placeh):
        self.X = torch.cat((self.X, placeh.X), dim=0)
        self.E = torch.cat((self.E, placeh.E), dim=0)
        self.node_mask = torch.cat((self.node_mask, placeh.node_mask), dim=0)
        self.atom_map_numbers = torch.cat((self.atom_map_numbers, placeh.atom_map_numbers), dim=0)
        self.mol_assignments = torch.cat((self.mol_assignments, placeh.mol_assignments), dim=0)
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
        self.E = F.pad(self.E, padding_tuple_E, value=0)
        self.node_mask = F.pad(self.node_mask, padding_tuple_v, value=0)
        self.atom_map_numbers = F.pad(self.atom_map_numbers, padding_tuple_v, value=0)
        self.mol_assignments = F.pad(self.mol_assignments, padding_tuple_v, value=0)
        
    def select_by_batch_idx(self, idx):
        '''
            Return a placeholder graph specified by the batch idx given as input.
            The returned graph does not share same memory with the original graph. 
            idx: batch idx given
        '''
        return PlaceHolder(X=copy.deepcopy(self.X[idx:idx+1]), E=copy.deepcopy(self.E[idx:idx+1]), y=copy.deepcopy(self.y[idx:idx+1]), node_mask=copy.deepcopy(self.node_mask[idx:idx+1]), 
                           atom_map_numbers=copy.deepcopy(self.atom_map_numbers[idx:idx+1]), mol_assignments=copy.deepcopy(self.mol_assignments[idx:idx+1]))
    
    def select_by_batch_and_sample_idx(self, bs, n_samples, batch_idx, sample_idx):
        assert self.X.ndim==2, f'Expected X of shape (bs, n), got X.shape={self.X.shape}. Use mask(node_mask, collapse=True) before calling this function.'
        assert self.E.ndim==3, f'Expected E of shape (bs, n, n), got E.shape={self.E.shape}. Use mask(node_mask, collapse=True) before calling this function.'
        
        return PlaceHolder(X=self.X.reshape(bs, n_samples, self.X.shape[1])[batch_idx:batch_idx+1, sample_idx], 
                           E=self.E.reshape(bs, n_samples, self.E.shape[2], -1)[batch_idx:batch_idx+1, sample_idx], 
                           y=self.y.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx], 
                           node_mask=self.node_mask.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx], 
                           atom_map_numbers=self.atom_map_numbers.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx], 
                           mol_assignments=self.mol_assignments.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx])
 
    def serialize(self):
        return {"X": self.X.detach().cpu().numpy().tolist(), "E": self.E.detach().cpu().numpy().tolist(),
                "y": self.y.detach().cpu().numpy().tolist(), "node_mask": self.node_mask.detach().cpu().numpy().tolist(),
                "atom_map_numbers": self.atom_map_numbers.detach().cpu().numpy().tolist(), 
                "mol_assignments": self.mol_assignments.detach().cpu().numpy().tolist()}
    
    def pyg(self):
        """Turns back into a pytorch geometric DataBatch() object, also with lesser precision for easier saving.
        To turn back to higher precision, there exists a function for that. pyg_to_full_precision_expanded() """
        
        # TODO: This or the other iteration are unnecessary, but it's okay
        return_data = []
        for i in range(self.X.shape[0]):
            # Concatenate the X as well
            E_idx, E_attr = dense_to_sparse(adj=self.E[i])

            X = self.X[i] if self.X.dim() == 2 else self.X[i].argmax(-1)
            assert X.dim() == 1

            # X = self.X[i].reshape(-1, self.X.shape[1]) if self.X.dim() == 2 else self.X[i].argmax(-1).reshape(-1, self.X.shape[1])
            atom_map_numbers = self.atom_map_numbers[i]
            node_mask = self.node_mask[i]
            mol_assignment = self.mol_assignments[i]

            # NOTE: atom mappings and mol_assignment have a different field names in the Data() objects and in the PlaceHolder objects. Needs to be accommodated here.
            return_data.append(Data(x=X.to(torch.uint8), edge_index=E_idx.to(torch.int16), edge_attr=E_attr.to(torch.uint8), y=self.y.to(torch.uint8), node_mask=node_mask.to(torch.uint8),
                        mask_atom_mapping=atom_map_numbers.to(torch.uint8), mol_assignment=mol_assignment.to(torch.uint8)))
        
        return Batch.from_data_list(return_data)

def json_to_graph(json_dict, x_classes, e_classes):
    graph = PlaceHolder(X=torch.Tensor(json_dict["X"]).to(torch.float32), E=torch.Tensor(json_dict["E"]).to(torch.float32),
                        y=torch.Tensor(json_dict["y"]), node_mask=torch.Tensor(json_dict["node_mask"]).to(torch.bool), 
                        atom_map_numbers=torch.Tensor(json_dict["atom_map_numbers"]).int(), 
                        mol_assignments=torch.Tensor(json_dict["mol_assignments"]).int())   
    
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
        product_mol_assignment = dense_data.mol_assignments[i].max().item()
        product_selection = (dense_data.mol_assignments[i] == product_mol_assignment)
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

def apply_mask(orig, z_t, atom_decoder, bond_decoder, mask_nodes=None,
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
    device = orig.X.device
    mask_X, mask_E = get_mask(orig=orig, atom_decoder=atom_decoder, bond_decoder=bond_decoder, 
                              mask_nodes=mask_nodes, mask_edges=mask_edges, 
                              node_states_to_mask=node_states_to_mask, 
                              edge_states_to_mask=edge_states_to_mask,
                              include_supernode=include_supernode)
    
    z_t_ = z_t.get_new_object()
    
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
        Get a mask vector of shape (n) to fix some of the values of z_t to the values of orig.
        
        include_supernode: whether to include the supernode in the mask or not. 
                Used in classifier-free guidance to drop out parts of the reaction, 
                but not the supernode. (a special case even if we usually mask out the supernode)
    '''
    device = orig.X.device
    # if type(orig)!=graph.PlaceHolder:
    #     orig = graph.to_dense(orig)
        
    # get structure-based masks
    if mask_nodes=='product' or mask_edges=='product':
        mask = get_mask_product(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
        mask_x = mask.clone()
        mask_e = mask.clone()
    elif mask_nodes=='reactant' or mask_edges=='reactant':
        mask = get_mask_reactant(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
        mask_x = mask.clone()
        mask_e = mask.clone()
    elif mask_nodes=='sn' or mask_edges=='sn':
        mask = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
        mask_x = mask.clone()
        mask_e = mask.clone()
    elif mask_nodes=='product_and_sn' or mask_edges=='product_and_sn':
        mask_1 = get_mask_product(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
        mask_2 = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
        mask = mask_1 * mask_2
        mask_x = mask.clone()
        mask_e = mask.clone()
    elif mask_nodes=='reactant_and_sn' or mask_edges=='reactant_and_sn':
        mask_1 = get_mask_reactant(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
        mask_2 = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
        mask = mask_1 * mask_2
        mask_x = mask.clone()
        mask_e = mask.clone()
    elif mask_nodes=='atom_mapping' or mask_edges=='atom_mapping':
        # NOTE: it does not make sense to use the atom_mapping without conditioning on products and SN.
        assert 'atom_map_numbers' in orig.__dir__(), 'Masking atom mapping is None in orig.'
        mask_1 = orig.atom_map_numbers == 0 # noise out the ones that don't have atom mapping
        mask_2 = get_mask_product(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device)
        mask_3 = get_mask_sn(origX=orig.X.clone(), atom_decoder=atom_decoder, device=device) if include_supernode else orig.node_mask.clone()
        mask = mask_1 * mask_2 * mask_3
        mask_x = mask.clone()
        mask_e = mask_2 * mask_3
    else:
        # mask for padding nodes
        # TODO: Are we only masking the padding nodes in this case?
        mask = orig.node_mask.clone()
    
    assert mask.shape==(orig.X.shape[0], orig.X.shape[1]), 'Something is wrong with the mask. Should have shape (bs, n_max).'

    node_idx_to_mask, edge_idx_to_mask = get_index_from_states(atom_decoder=atom_decoder, bond_decoder=bond_decoder, node_states_to_mask=node_states_to_mask, 
                                                               edge_states_to_mask=edge_states_to_mask, device=device)
    
    # get state-based masks
    # the logic here is that if a state is fixed
    # then all the nodes/edges with that state are fixed
    # fixing/masking = taking the value from the original data
    # TODO: lose for loop
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

def fix_nodes_and_edges_by_idx(data, node_idx, edge_idx):
    '''
        Return a mask where all nodes and edges with specified indices are set to True and everything else to False.
        To be used in fixing the nodes/edges from the true data during generation (mainly with inpainting).
        
        data: graph or placeholder data.
        node_idx: python list (of lists? for batches) of indices of nodes to fix.
        edge_idx: python list (of lists? for batches) of indicies of edges to fix.
        
        return:
            masks for X and E.
    '''
    if type(data)!=graph.PlaceHolder:
        dense_data = graph.to_dense(data)
    else:
        dense_data = copy.deepcopy(data)
    
    assert dense_data.X.ndim==3, f'Expected dense_data.X.shape=(bs, n, dx). Got dense_data.X.shape={dense_data.X.shape}'
    assert dense_data.E.ndim==4, f'Expected dense_data.E.shape=(bs, n, n, de). Got dense_data.E.shape={dense_data.E.shape}'
    
    dense_data.mask(dense_data.node_mask, collapse=True)
    
    mask_X = torch.zeros_like(dense_data.X, dtype=torch.bool)
    mask_E = torch.zeros_like(dense_data.E, dtype=torch.bool)
    
    if node_idx!=None:
        for bs in range(mask_X.shape[0]):
            mask_X[bs, node_idx[bs]] = True
            
    if edge_idx!=None:
        for bs in range(mask_E.shape[0]):
            for bond in edge_idx[bs]:
                mask_E[bs, bond[0], bond[1]] = True
                mask_E[bs, bond[1], bond[0]] = True
        
    return mask_X, mask_E
        
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
        # use np.exp() because it handles overflow more gracefully
        sum_exp_elbo = sum(np.exp(-reaction['elbo']) for reaction in reactions_list)
        sum_counts = sum(reaction['count'] for reaction in reactions_list)

        # Calculate the weighted probability for each reaction and add it to the dictionary
        for reaction in reactions_list:
            exp_elbo = np.exp(-reaction['elbo'])
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
    rcts = [r.split('>>')[0].split('.') for r in reactions]
    prods = [[r.split('>>')[1]] for r in reactions]
    return rcts, prods

def save_samples_to_file_without_weighted_prob(filename, condition_idx, gen_rxns, true_rcts, true_prods):
    # This is used mainly to handle with some old evaluations, deprecated aside from that
    # TODO: Is n_samples really needed here? It seems that it may cause bugs if we are not careful when removing duplicate samples
    if os.path.exists(filename):
        file = open(filename,'a') 
    else:
        file = open(filename,'w')
    # file = open(filename,'w') if condition_idx==0 else open(filename,'a') 
    for i, p in enumerate(gen_rxns):
        lines = [f'(cond {condition_idx + i}) {mol.rxn_list_to_str(true_rcts[i], true_prods[i])}:\n'] + \
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

def save_samples_to_file(filename, condition_idx, gen_rxns, true_rcts, true_prods):
    # TODO: Is n_samples really needed here? It seems that it may cause bugs if we are not careful when removing duplicate samples
    # TODO: This shouldn't take as input condition_idx. I guess it is supposed to handle the case where batch size > 1?
    if os.path.exists(filename):
        file = open(filename,'a')
    else:
        file = open(filename,'w')
    # file = open(filename,'w') if condition_idx==0 else open(filename,'a') 
    for i, p in enumerate(gen_rxns):
        lines = [f'(cond {condition_idx + i}) {mol.rxn_list_to_str(true_rcts[i], true_prods[i])}:\n'] + \
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