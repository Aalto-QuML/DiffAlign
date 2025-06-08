import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
    scatter,
    to_edge_index,
    is_torch_sparse_tensor,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)
import copy
from scipy.sparse.linalg import eigsh
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data

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

# Keep the supernode intact here, others are permuted
def permute_rows_product(g_nodes_prod, mask_atom_mapping, g_edge_index_prod, suno_idx):
    # TODO: This is deprecated, we are not using supernodes anymore
    prod_indices = (suno_idx, suno_idx + g_nodes_prod.shape[0])
    perm = torch.cat([torch.tensor([0], dtype=torch.long), 1 + torch.randperm(g_nodes_prod.shape[0]-1)], 0)
    inv_perm = torch.zeros(len(perm), dtype=torch.long)
    inv_perm.scatter_(dim=0, index=perm, src=torch.arange(len(perm)))
    g_nodes_prod[:] = g_nodes_prod[perm]
    
    # sn_and_prod_selection = (prod_selection | suno_idx == torch.arange(len(prod_selection)))
    mask_atom_mapping[prod_indices[0]:prod_indices[1]] = mask_atom_mapping[prod_indices[0]:prod_indices[1]][perm]
    
    # The following because g_edge_index_prod are counted with their offset in the final graph
    offset_padded_perm = torch.cat([torch.zeros(suno_idx, dtype=torch.long), suno_idx + perm]) # for debugging
    offset_padded_inv_perm = torch.cat([torch.zeros(suno_idx, dtype=torch.long), suno_idx + inv_perm])
    
    g_edge_index_prod[:] = offset_padded_inv_perm[g_edge_index_prod]

def laplacian_eigenvectors_scipy(num_nodes, edge_index, k):
    """
    Adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/add_positional_encoding.html
    Computes the eigenvectors of the Laplacian matrix of a graph.

    Arguments:
    num_nodes (int): The number of nodes in the graph.
    edge_index (torch.Tensor): A tensor of shape (2, m) containing the indices of the edges in the graph.
    k (int): The number of eigenvectors to compute.

    Returns:
    torch.Tensor: A tensor of shape (n, k) containing the real parts of the k eigenvectors of the Laplacian matrix, excluding the first eigenvector.
    """
    # TODO: Change this to calculate the eigenvectors from the dense format
    # At the moment sometimes the eigenvector dimensionality is smaller than the 

    # print(f'edge_index.shape {edge_index.shape}\n')
    # print(f'num_nodes {num_nodes}\n')
    L_edge_index, L_edge_weight = get_laplacian(
        edge_index,
        normalization='sym',
        num_nodes=num_nodes
    )

    L = to_scipy_sparse_matrix(L_edge_index, L_edge_weight, num_nodes)

    eig_vals, eig_vecs = eigsh(
        L,
        k=min(k+1, num_nodes-1), # To make sure that we are not trying to calculate too many eigenvectors
        which='SA',
        return_eigenvectors=True,
        ncv=min(num_nodes, max(20*k + 1, 40))
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k + 1]) # Exclude the first eigenvector, since constant eigenvector is not useful
    if pe.shape[1] < k:
        pe = torch.cat([pe, torch.zeros((num_nodes, k - pe.shape[1]))], dim=1) # Zero pad if we have less than k eigenvectors
    # pe = torch.from_numpy(eig_vecs)
    sign = -1 + 2 * torch.randint(0, 2, (k, ))
    pe *= sign
    assert pe.shape == (num_nodes, k)
    return pe

def laplacian_eigenvectors_gpu(num_nodes, edge_index, k, device='cuda'):
    """
    Computes the eigenvectors of the Laplacian matrix of a graph using Pytorch and potentially CUDA.

    Arguments:
    num_nodes (int): The number of nodes in the graph.
    edge_index (torch.Tensor): A tensor of shape (2, m) containing the indices of the edges in the graph.
    k (int): The number of eigenvectors to compute.

    Returns:
    torch.Tensor: A tensor of shape (n, k) containing the real parts of the k eigenvectors of the Laplacian matrix, excluding the first eigenvector.
    """

    L_edge_index, L_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)

    # Convert to dense format for eigen decomposition
    L_dense = torch.zeros((num_nodes, num_nodes), device=device)
    L_dense[L_edge_index[0], L_edge_index[1]] = L_edge_weight

    # Compute eigenvalues and eigenvectors using PyTorch
    eig_vals, eig_vecs = torch.linalg.eigh(L_dense)

    # Sort eigenvectors based on eigenvalues and exclude the first eigenvector
    sorted_indices = torch.argsort(eig_vals)
    pe = eig_vecs[:, sorted_indices[1:k + 1]]

    # Apply random sign flipping
    sign = -1 + 2 * torch.randint(0, 2, (k, ), device=device)
    pe *= sign

    return pe.cpu()

def rw_positional_encoding(edge_index, num_nodes, walk_length):
    # Adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/add_positional_encoding.html
    row, col = edge_index
    num_edges = row.size(0)
    N = num_nodes
    assert N is not None

    value = torch.ones(num_edges, device=row.device)
    value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
    value = 1.0 / value

    if N <= 2_000:  # Dense code path for faster computation:
        adj = torch.zeros((N, N), device=row.device)
        adj[row, col] = value
        loop_index = torch.arange(N, device=row.device)
    elif torch_geometric.typing.NO_MKL:  # pragma: no cover
        adj = to_torch_coo_tensor(edge_index, value, size=num_nodes) # TODO: Check that size refers to num_nodes. Was originally data.size()
    else:
        adj = to_torch_csr_tensor(edge_index, value, size=num_nodes)

    def get_pe(out: torch.Tensor) -> torch.Tensor:
        if is_torch_sparse_tensor(out):
            return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
        return out[loop_index, loop_index]

    out = adj
    pe_list = [get_pe(out)]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(get_pe(out))

    pe = torch.stack(pe_list, dim=-1)
    return pe

def positional_encoding(num_nodes, g_edge_index, pos_encoding_type, num_lap_eig_vectors, random_walk_length):
    assert pos_encoding_type in ['laplacian_pos_enc', 'laplacian_pos_enc_gpu', 'random_walk', 'gaussian_pos_enc', 'unaligned_laplacian_pos_enc']
    if pos_encoding_type == 'laplacian_pos_enc':
        pe = laplacian_eigenvectors_scipy(num_nodes, g_edge_index, num_lap_eig_vectors)
    elif pos_encoding_type == 'laplacian_pos_enc_gpu': # This is probably not very useful
        pe = laplacian_eigenvectors_gpu(num_nodes, g_edge_index, num_lap_eig_vectors)
    elif pos_encoding_type == 'random_walk':
        pe = rw_positional_encoding(g_edge_index, num_nodes, random_walk_length)
    else: # This more efficient version is not implemented yet
        pe = torch.zeros(num_nodes, num_lap_eig_vectors)
    return pe

def get_num_product_atoms(mol_assignment):
    if len(mol_assignment.shape) == 1:
        product_mol_index = mol_assignment.max()
        num_nodes_product = (mol_assignment == product_mol_index).sum().item()
    elif len(mol_assignment.shape) == 2:
        product_mol_index = mol_assignment.max(dim=1, keepdim=True).values
        num_nodes_product = (mol_assignment == product_mol_index).sum(-1)
    else:
        raise ValueError("mol_assignment should be a 1D or 2D tensor")
    return num_nodes_product

def get_first_product_index(mol_assignment):
    if len(mol_assignment.shape) == 1:
        product_mol_index = mol_assignment.max()
        # print(f'mol_assignment {mol_assignment}\n')
        # print(f'product_mol_index {product_mol_index}\n')
        # print(f'{(mol_assignment == product_mol_index).nonzero().squeeze()}\n')
        # print(f'{type((mol_assignment == product_mol_index).nonzero().squeeze())}\n')
        # print(f'{(mol_assignment == product_mol_index).nonzero().squeeze().numel()}\n')
        product_start = (mol_assignment == product_mol_index).nonzero().squeeze()
        if product_start.numel()>1:
            product_start = product_start[0].item()
            
    elif len(mol_assignment.shape) == 2:
        product_mol_index = mol_assignment.max(dim=1).values
        product_start = (mol_assignment == product_mol_index[:,None]).int().argmax(dim=-1)
    else:
        raise ValueError("mol_assignment should be a 1D or 2D tensor")
    return product_start

def positional_encoding_for_reactant_product_graph(num_nodes_total, num_nodes_product, edge_index, mol_assignment, 
                                                   pos_encoding_type, pos_encoding_size):
    """A wrapper around positional_encoding that computes positional encodings for the combined reactant-product graphs on the fly
    TODO: Make this work for graph-batched data? Is going to be a bit tricky tbh. Putting this in the dataloader would be kind of optimal"""
    assert pos_encoding_type in ['laplacian_pos_enc', 'laplacian_pos_enc_gpu', 'random_walk', 'laplacian_and_smiles_pos_enc_with_product_id', 'gaussian_pos_enc', 'unaligned_laplacian_pos_enc']
    if pos_encoding_type == 'laplacian_and_smiles_pos_enc_with_product_id':
        # not implemented yet here
        return None

    pos_encoding = torch.zeros(num_nodes_total, pos_encoding_size)
    product_start = get_first_product_index(mol_assignment)
    edges_with_product = edge_index[:, (edge_index[0] >= product_start) & (edge_index[1] >= product_start)]
    # Hmm need to pick out the edges in edge_index that are only in the products hmm
    # print(f'edges_with_product.shape {edges_with_product.shape}\n')
    # print(f'edges_with_product.min() {edges_with_product.min()}\n')
    
    if edges_with_product.shape[-1] > 0: # at least one edge in the product graph
        prod_min = edges_with_product.min() # handling the case of molecules without edges
        pos_encoding_prod = positional_encoding(num_nodes_product, edges_with_product - prod_min,
                                pos_encoding_type, pos_encoding_size, pos_encoding_size)
    else: # This is the same as the product only having one node
        pos_encoding_prod = torch.ones(num_nodes_product, pos_encoding_size)
    # Place in the products
    pos_encoding[product_start:product_start+num_nodes_product,:pos_encoding_prod.shape[1]] = pos_encoding_prod
    # Transfer to reactants (assuming atom-mapped aligned indices)
    pos_encoding[:pos_encoding_prod.shape[0],:pos_encoding_prod.shape[1]] = pos_encoding_prod # This should always work: Since we have the dummy nodes, there will always be more nodes in the reactants than in the products

    return pos_encoding

def get_rct_prod_from_data(data):
    """Input: a pytorch geometric data object
    Output: reactant and product graph information separately from the data object. Basically the inverse of 
    Data(x=torch.cat([nodes_rct, nodes_prod], dim=0), 
                     edge_index=torch.cat([edge_index_rct, edge_index_prod + len(nodes_rct)], dim=1),
                     edge_attr=torch.cat([bond_types_rct, bond_types_prod], dim=0), y=y, idx=data_idx,
                     mol_assignment=torch.cat([mol_assignment_reactants, mol_assignment_products], dim=0),
                     atom_map_numbers=torch.cat([atom_map_reactants, atom_map_products], dim=0),
                     smiles=smiles_to_save,
                     atom_charges=torch.cat([atom_charges_rct, atom_charges_prod], dim=0),
                     atom_chiral=torch.cat([atom_chiral_rct, atom_chiral_prod], dim=0),
                     bond_dirs=torch.cat([bond_dirs_rct, bond_dirs_prod], dim=0))
    """
    mol_assignment = data.mol_assignment
    first_prod_index = get_first_product_index(mol_assignment)
    atom_map_reactants, atom_map_products = data.atom_map_numbers[:first_prod_index], data.atom_map_numbers[first_prod_index:]
    nodes_rct, nodes_prod = data.x[:first_prod_index], data.x[first_prod_index:]
    edge_index_rct, edge_index_prod = data.edge_index[:, (data.edge_index[0] < first_prod_index)], data.edge_index[:, (data.edge_index[0] >= first_prod_index)]
    edge_index_prod -= first_prod_index
    bond_types_rct, bond_types_prod = data.edge_attr[(data.edge_index[0] < first_prod_index)], data.edge_attr[ (data.edge_index[0] >= first_prod_index)]
    atom_charges_rct, atom_charges_prod = data.atom_charges[:first_prod_index], data.atom_charges[first_prod_index:]
    atom_chiral_rct, atom_chiral_prod = data.atom_chiral[:first_prod_index], data.atom_chiral[first_prod_index:]
    bond_dirs_rct, bond_dirs_prod = data.bond_dirs[(data.edge_index[0] < first_prod_index)], data.bond_dirs[(data.edge_index[0] >= first_prod_index)]
    mol_assignment_rct, mol_assignment_prod = data.mol_assignment[:first_prod_index], data.mol_assignment[first_prod_index:]
    if 'pos_encoding' in data:
        pos_enc_rct = data.pos_encoding[:first_prod_index]
        pos_enc_prod = data.pos_encoding[first_prod_index:]
    else:
        pos_enc_rct = None
        pos_enc_prod = None
    
    return atom_map_reactants, atom_map_products, nodes_rct, nodes_prod, edge_index_rct, edge_index_prod, bond_types_rct, bond_types_prod, atom_charges_rct, atom_charges_prod, atom_chiral_rct, atom_chiral_prod, bond_dirs_rct, bond_dirs_prod, mol_assignment_rct, mol_assignment_prod, pos_enc_rct, pos_enc_prod

def reactant_initialization_based_only_on_product_data(cfg, z_t):
    device = z_t.X.device

    prod_nodes = (z_t.mol_assignment[0].max().item() == z_t.mol_assignment[0]).nonzero().flatten()
    start_prod_idx = prod_nodes[0].item()
    size_of_prod = len(prod_nodes) # prod_nodes[-1].item() - start_prod_idx + 1
    # number of reactant nodes (plus supernode if applicable)
    n_reactant_nodes_plus_optional_supernode = size_of_prod + cfg.dataset.nb_rct_dummy_nodes + int(cfg.dataset.add_supernodes)

    # create an entirely new graph from scratch (assume that the entire batch here is for the same sample)
    bs = z_t.X.shape[0]
    X = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, z_t.X.shape[-1], device=device)
    E = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, n_reactant_nodes_plus_optional_supernode+size_of_prod, z_t.E.shape[-1], device=device)
    y = torch.zeros(bs, 1, device=device)
    atom_charges = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, z_t.atom_charges.shape[-1], device=device)
    atom_chiral = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, z_t.atom_chiral.shape[-1], device=device)
    bond_dirs = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, n_reactant_nodes_plus_optional_supernode+size_of_prod, z_t.bond_dirs.shape[-1], device=device)
    node_mask = torch.ones(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, device=device, dtype=torch.bool)
    atom_map_numbers = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, device=device, dtype=torch.long)
    mol_assignment = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, device=device, dtype=torch.long)
    pos_encoding = torch.zeros(bs, n_reactant_nodes_plus_optional_supernode+size_of_prod, z_t.pos_encoding.shape[-1], device=device)
    mask_X_idx = cfg.dataset.atom_types.index('Au')
    mask_E_idx = cfg.dataset.bond_types.index('none')
    mask_charges_idx = cfg.dataset.atom_charges.index(0)
    mask_chiral_idx = cfg.dataset.atom_chiral_tags.index('none')
    mask_bond_dirs_idx = cfg.dataset.bond_dirs.index('none')
    for i in range(bs):
        X[i, :n_reactant_nodes_plus_optional_supernode] = F.one_hot(torch.tensor(mask_X_idx), num_classes=len(cfg.dataset.atom_types)).float()
        E[i, :n_reactant_nodes_plus_optional_supernode, :n_reactant_nodes_plus_optional_supernode] = F.one_hot(torch.tensor(mask_E_idx), num_classes=len(cfg.dataset.bond_types)).float()
        atom_charges[i, :n_reactant_nodes_plus_optional_supernode] = F.one_hot(torch.tensor(mask_charges_idx), num_classes=len(cfg.dataset.atom_charges)).float()
        atom_chiral[i, :n_reactant_nodes_plus_optional_supernode] = F.one_hot(torch.tensor(mask_chiral_idx), num_classes=len(cfg.dataset.atom_chiral_tags)).float()
        bond_dirs[i, :n_reactant_nodes_plus_optional_supernode, :n_reactant_nodes_plus_optional_supernode] = F.one_hot(torch.tensor(mask_bond_dirs_idx), num_classes=len(cfg.dataset.bond_dirs)).float()
        X[i,-size_of_prod:] = z_t.X[i,start_prod_idx:start_prod_idx+size_of_prod]
        E[i,-size_of_prod:,-size_of_prod:] = z_t.E[i,start_prod_idx:start_prod_idx+size_of_prod,start_prod_idx:start_prod_idx+size_of_prod]
        atom_charges[i,-size_of_prod:] = z_t.atom_charges[i,start_prod_idx:start_prod_idx+size_of_prod]
        atom_chiral[i,-size_of_prod:] = z_t.atom_chiral[i,start_prod_idx:start_prod_idx+size_of_prod]
        bond_dirs[i,-size_of_prod:,-size_of_prod:] = z_t.bond_dirs[i,start_prod_idx:start_prod_idx+size_of_prod,start_prod_idx:start_prod_idx+size_of_prod]
        if cfg.dataset.add_supernodes:
            supernode_idx = cfg.dataset.atom_types.index('SuNo')
            X[i, n_reactant_nodes_plus_optional_supernode] = F.one_hot(torch.tensor(supernode_idx), num_classes=len(cfg.dataset.atom_types)).float()
        mol_assignment[i, -size_of_prod:] = 1
        atom_map_numbers[i, :size_of_prod] = torch.arange(1, size_of_prod+1, device=device)
        atom_map_numbers[i, -size_of_prod:] = torch.arange(1, size_of_prod+1, device=device)
    
    # permute the order of the nodes on the product side just to be sure
    # perm = torch.randperm(size_of_prod, device=device)
    # X[:,-size_of_prod:] = X[:,-size_of_prod:][:,perm]
    # E[:,-size_of_prod:,-size_of_prod:] = E[:,-size_of_prod:,-size_of_prod:][:,perm][:,:,perm]
    # atom_charges[:,-size_of_prod:] = atom_charges[:,-size_of_prod:][:,perm]
    # atom_chiral[:,-size_of_prod:] = atom_chiral[:,-size_of_prod:][:,perm]
    # bond_dirs[:,-size_of_prod:,-size_of_prod:] = bond_dirs[:,-size_of_prod:,-size_of_prod:][:,perm][:,:,perm]
    # atom_map_numbers[:,-size_of_prod:] = atom_map_numbers[:,-size_of_prod:][:,perm]
    # mol_assignment[:,-size_of_prod:] = 1

    return z_t.get_new_object(X=X, E=E, y=y, atom_charges=atom_charges, atom_chiral=atom_chiral, bond_dirs=bond_dirs, node_mask=node_mask, atom_map_numbers=atom_map_numbers, mol_assignment=mol_assignment, pos_encoding=pos_encoding)

def drop_atom_maps(data, drop_atom_maps_rate):

    atom_map_reactants, atom_map_products, nodes_rct, nodes_prod, edge_index_rct, \
        edge_index_prod, bond_types_rct, bond_types_prod, atom_charges_rct, atom_charges_prod, \
        atom_chiral_rct, atom_chiral_prod, bond_dirs_rct, bond_dirs_prod, \
        mol_assignment_rct, mol_assignment_prod, pos_enc_rct, pos_enc_prod = get_rct_prod_from_data(data)    
        
    # Find values in product mapping that don't appear in reactant mapping
    # NOTE: this also gets rid of wrong atom mappings: anything not in the product mapping is set to 0
    # NOTE: start with this because if an AM number is in the product but not the reactant there is no way to recover it anyway, 
    # might as well set it to 0
    unique_prod_values = set(atom_map_products.tolist())
    mask = torch.tensor([x.item() not in unique_prod_values for x in atom_map_reactants])
    atom_map_reactants[mask] = 0
    # assign the product AMs that are not in the reactant to some of the 0ed reactant atoms
    in_product_not_reactant = [x for x in unique_prod_values if x not in atom_map_reactants.tolist()]
 
    for i, val in enumerate(in_product_not_reactant):
        mask = (atom_map_reactants == 0)
        indices = torch.nonzero(mask)[i] # Gets the i-th zero position
        atom_map_reactants[indices] = val
    
    assert atom_map_products.shape==atom_map_reactants[atom_map_reactants!=0].shape, \
        f"Removing the AMs that are not in the product mapping failed. Shape issue atom_map_reactants"+\
        f"{atom_map_reactants[atom_map_reactants!=0].sort()[0]} ({atom_map_reactants[atom_map_reactants!=0].shape})"+\
        f"atom_map_products {atom_map_products.sort()[0]} ({atom_map_products.shape}). "+\
        f"in_product_not_reactant {in_product_not_reactant.sort()[0]} ({in_product_not_reactant.shape})." +\
        f" atom_map_reactants[atom_map_reactants==0] {atom_map_reactants[atom_map_reactants==0]} ({atom_map_reactants[atom_map_reactants==0].shape})"
    assert (atom_map_products.sort()[0]==atom_map_reactants[atom_map_reactants!=0].sort()[0]).all(), \
        f"Removing the AMs that are not in the product mapping failed atom_map_reactants"+\
        f"{atom_map_reactants[atom_map_reactants!=0].sort()[0]} ({atom_map_reactants[atom_map_reactants!=0].shape})"+\
        f"atom_map_products {atom_map_products.sort()[0]} ({atom_map_products.shape})"
    
    # Get unique non-zero values from product atom mappings
    unique_prod_values = sorted(list(set([v.item() for v in atom_map_products if v.item() != 0])))
    n_unique = len(unique_prod_values)

    # Create mapping from old to new numbers (starting from 1)
    prod_values = {old: new for old, new in zip(unique_prod_values, torch.randperm(n_unique).add(1))}

    # Apply random numbering to both product and reactant
    atom_map_products = torch.tensor([prod_values.get(v.item(), 0) for v in atom_map_products])
    valid_mask = atom_map_reactants != 0
    atom_map_reactants[valid_mask] = torch.tensor([prod_values.get(v.item(), 0) for v in atom_map_reactants[valid_mask]])
    
    pos_enc = torch.cat([pos_enc_rct, pos_enc_prod], dim=0) if pos_enc_rct is not None else None

    return Data(x=torch.cat([nodes_rct, nodes_prod], dim=0), 
                     edge_index=torch.cat([edge_index_rct, edge_index_prod + len(nodes_rct)], dim=1),
                     edge_attr=torch.cat([bond_types_rct, bond_types_prod], dim=0), y=data.y, idx=data.idx,
                     mol_assignment=torch.cat([mol_assignment_rct, mol_assignment_prod], dim=0),
                     atom_map_numbers=torch.cat([atom_map_reactants, atom_map_products], dim=0),
                     smiles=data.smiles,
                     atom_charges=torch.cat([atom_charges_rct, atom_charges_prod], dim=0),
                     atom_chiral=torch.cat([atom_chiral_rct, atom_chiral_prod], dim=0),
                     bond_dirs=torch.cat([bond_dirs_rct, bond_dirs_prod], dim=0),
                     pos_encoding = pos_enc)
    
    # Get mapping values from product and create new random numbering
    n_prod = len(atom_map_products)
    # NOTE: + 1 because we want to start numbering from 1
    prod_values = {v.item():r.item() for v, r in zip(atom_map_products, (torch.randperm(n_prod)+1))}
    # Apply random numbering to both product and reactant
    atom_map_products = torch.tensor([prod_values[v.item()] for v in atom_map_products])
    valid_mask = atom_map_reactants != 0
    atom_map_reactants[valid_mask] = torch.tensor([prod_values[v.item()] for v in atom_map_reactants[valid_mask]])
    unique_prod_values = set(atom_map_products.tolist())
    mask = torch.tensor([x.item() not in unique_prod_values for x in atom_map_reactants])
    atom_map_reactants[mask] = 0
    assert atom_map_products.shape==atom_map_reactants[atom_map_reactants!=0].shape, \
        f"Assigning random AM numbers failed. Shape issue atom_map_reactants {atom_map_reactants[atom_map_reactants!=0].sort()[0]} ({atom_map_reactants[atom_map_reactants!=0].shape}) atom_map_products {atom_map_products.sort()[0]} ({atom_map_products.shape})"
    assert (atom_map_products.sort()[0]==atom_map_reactants[atom_map_reactants!=0].sort()[0]).all(), \
        f"Assigning random AM numbers failed atom_map_reactants {atom_map_reactants[atom_map_reactants!=0].sort()[0]} ({atom_map_reactants[atom_map_reactants!=0].shape}) atom_map_products {atom_map_products.sort()[0]} ({atom_map_products.shape})"
    
    # atom_map_reactants[torch.rand(*atom_map_reactants.shape) < drop_atom_maps_rate] = 0
    # atom_map_products[torch.rand(*atom_map_products.shape) < drop_atom_maps_rate] = 0
    # atom_map_reactants, atom_map_products = fix_atom_mappings(atom_map_reactants, atom_map_products)
    # Get number of product atoms

    pos_enc = torch.cat([pos_enc_rct, pos_enc_prod], dim=0) if pos_enc_rct is not None else None

    return Data(x=torch.cat([nodes_rct, nodes_prod], dim=0), 
                     edge_index=torch.cat([edge_index_rct, edge_index_prod + len(nodes_rct)], dim=1),
                     edge_attr=torch.cat([bond_types_rct, bond_types_prod], dim=0), y=data.y, idx=data.idx,
                     mol_assignment=torch.cat([mol_assignment_rct, mol_assignment_prod], dim=0),
                     atom_map_numbers=torch.cat([atom_map_reactants, atom_map_products], dim=0),
                     smiles=data.smiles,
                     atom_charges=torch.cat([atom_charges_rct, atom_charges_prod], dim=0),
                     atom_chiral=torch.cat([atom_chiral_rct, atom_chiral_prod], dim=0),
                     bond_dirs=torch.cat([bond_dirs_rct, bond_dirs_prod], dim=0),
                     pos_encoding = pos_enc)

    # Add noise through swaps
    if drop_atom_maps_rate > 0:
        n_swaps = int(n_prod * drop_atom_maps_rate / 2)
        if n_swaps > 0:
            available_indices = set(range(n_prod))
            
            for _ in range(n_swaps):
                if len(available_indices) < 2:
                    break
                    
                # Pick two random indices that haven't been swapped yet
                idx1, idx2 = torch.tensor(list(available_indices))[torch.randperm(len(available_indices))[:2]]
                idx1, idx2 = idx1.item(), idx2.item()
                
                # Remove these indices from available pool
                available_indices.remove(idx1)
                available_indices.remove(idx2)
                
                # Get values
                val1, val2 = atom_map_products[idx1].item(), atom_map_products[idx2].item()
                
                # Perform swap
                atom_map_products[idx1] = val2
                atom_map_products[idx2] = val1
                mask1 = atom_map_reactants == val1
                mask2 = atom_map_reactants == val2
                atom_map_reactants[mask1] = val2
                atom_map_reactants[mask2] = val1
    
    assert atom_map_products.shape==atom_map_reactants[atom_map_reactants!=0].shape, \
        f"Swapping AMs failed. Shape issue atom_map_reactants {atom_map_reactants[atom_map_reactants!=0].sort()[0]} ({atom_map_reactants[atom_map_reactants!=0].shape}) atom_map_products {atom_map_products.sort()[0]} ({atom_map_products.shape})"
    assert (atom_map_products.sort()[0]==atom_map_reactants[atom_map_reactants!=0].sort()[0]).all(), \
        f"Swapping AMs failed. Atom maps are not fixed correctly atom_map_reactants {atom_map_reactants[atom_map_reactants!=0].sort()[0]} ({atom_map_reactants[atom_map_reactants!=0].shape}) atom_map_products {atom_map_products.sort()[0]} ({atom_map_products.shape})"
    # Do the alignment here again
    
    # Align the graphs here according to the atom mapping, so that alignment is not necessary afterwards
    atom_map_reactants, indices = torch.sort(atom_map_reactants, dim=0, descending=True) # put the non-atom-mapped stuff as last
    inverse_indices = torch.argsort(indices) #torch.tensor([idx[0].item() for idx in sorted(zip(torch.arange(len(indices)), indices), key=lambda x: x[1])])
    nodes_rct = nodes_rct[indices]
    edge_index_rct = inverse_indices[edge_index_rct]
    mol_assignment_rct = mol_assignment_rct[indices]
    atom_chiral_rct = atom_chiral_rct[indices]
    atom_charges_rct = atom_charges_rct[indices]
    if pos_enc_rct is not None:
        pos_enc_rct = pos_enc_rct[indices]

    atom_map_products, indices = torch.sort(atom_map_products, dim=0, descending=True) # put the non-atom-mapped stuff as last
    inverse_indices = torch.argsort(indices)
    nodes_prod = nodes_prod[indices]
    edge_index_prod = inverse_indices[edge_index_prod]
    mol_assignment_prod = mol_assignment_prod[indices]
    atom_chiral_prod = atom_chiral_prod[indices]
    atom_charges_prod = atom_charges_prod[indices]
    if pos_enc_prod is not None:
        pos_enc_prod = pos_enc_prod[indices]

    pos_enc = torch.cat([pos_enc_rct, pos_enc_prod], dim=0) if pos_enc_rct is not None else None

    return Data(x=torch.cat([nodes_rct, nodes_prod], dim=0), 
                     edge_index=torch.cat([edge_index_rct, edge_index_prod + len(nodes_rct)], dim=1),
                     edge_attr=torch.cat([bond_types_rct, bond_types_prod], dim=0), y=data.y, idx=data.idx,
                     mol_assignment=torch.cat([mol_assignment_rct, mol_assignment_prod], dim=0),
                     atom_map_numbers=torch.cat([atom_map_reactants, atom_map_products], dim=0),
                     smiles=data.smiles,
                     atom_charges=torch.cat([atom_charges_rct, atom_charges_prod], dim=0),
                     atom_chiral=torch.cat([atom_chiral_rct, atom_chiral_prod], dim=0),
                     bond_dirs=torch.cat([bond_dirs_rct, bond_dirs_prod], dim=0),
                     pos_encoding = pos_enc)

# def recalculate_atom_maps_from_rxn_smiles(smiles_rxn):
#     r_mol = Chem.MolFromSmiles(smiles_rxn.split('>>')[0])
#     [r_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(r_mol.GetNumAtoms())]
#     p_mol = Chem.MolFromSmiles(smiles_rxn.split('>>')[1])
#     [p_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(p_mol.GetNumAtoms())]
#     smiles_rxn = Chem.MolToSmiles(r_mol, canonical=True) + ">>" + Chem.MolToSmiles(p_mol, canonical=True)
#     chython_smiles = smiles(smiles_rxn)
#     chython_smiles.reset_mapping() # assign atom mappings with the chython library
#     regular_smiles = format(chython_smiles, 'm')
#     # remove atom mappings from the reactant side for atoms that don't have atom maps
#     r_mol = Chem.MolFromSmiles(regular_smiles.split('>>')[0])
#     p_mol = Chem.MolFromSmiles(regular_smiles.split('>>')[1])
#     p_am_set = set([a.GetAtomMapNum() for a in p_mol.GetAtoms() if a.GetAtomMapNum() != 0])
#     for atom in r_mol.GetAtoms():
#         if atom.GetAtomMapNum() not in p_am_set:
#             atom.SetAtomMapNum(0)
#     return Chem.MolToSmiles(r_mol, canonical=True) + ">>" + Chem.MolToSmiles(p_mol, canonical=True)

def get_pos_enc_size(cfg):
    return cfg.neuralnet.num_lap_eig_vectors if cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc' or cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc_cpu' else cfg.neuralnet.random_walk_length

def positional_encoding_adding_transform(data, pos_encoding_type, pos_encoding_size):
    """data is a pyg Data() object, and we want to add the pos_enc field to it"""
    product_mol_index = data.mol_assignment.max()
    #print(f'data.mol_assignment {data.mol_assignment}\n')
    num_nodes_product = (data.mol_assignment == product_mol_index).sum().item()
    num_nodes_total = data.num_nodes
    data.pos_encoding = positional_encoding_for_reactant_product_graph(num_nodes_total, num_nodes_product, data.edge_index, 
                                                                       data.mol_assignment, pos_encoding_type, pos_encoding_size)
    return data

def add_product_id(data):
    """data is a pyg Data() object, and we want to add an identifier to the product atoms to the positional encoding"""
    product_mol_index = data.mol_assignment.max()
    num_product_nodes = (data.mol_assignment == product_mol_index).sum().item()
    num_reactant_nodes = len(data.mol_assignment) - num_product_nodes
    data.pos_encoding = torch.cat([data.pos_encoding, torch.cat([torch.zeros(num_reactant_nodes), torch.ones(num_product_nodes)])[:,None]], dim=-1)
    return data

def add_supernodes(cfg, data):
    """In case the data does not have a supernode as the first node of each molecule, we add it here. 
    Need to also update the mol_assignment, atom_map_numbers, and pos_encoding fields. 
    Also to atom_charges, atom_chiral, and edge_index. 
    Should reproduce the behavior of the way that the supernodes used to be added"""
    mol_assignment = data.mol_assignment.clone()
    nodes = data.x.clone()
    atom_map_numbers = data.atom_map_numbers.clone()
    edge_index = data.edge_index.clone()
    pos_encoding = data.pos_encoding.clone() if 'pos_encoding' in data else None
    atom_charges = data.atom_charges.clone()
    atom_chiral = data.atom_chiral.clone()
    # Create a supernode one-hot encoding
    suno = F.one_hot(torch.tensor([cfg.dataset.atom_types.index('SuNo')], dtype=torch.long), num_classes=len(cfg.dataset.atom_types))
    # Only add supernode to the beginning of the products.
    supernode_addition_indices = (mol_assignment == mol_assignment.max()).nonzero()[0]
    # add the supernodes
    for index in reversed(supernode_addition_indices):
        nodes = torch.cat([nodes[:index], suno.clone(), nodes[index:]], dim=0)
        mol_assignment = torch.cat([mol_assignment[:index], torch.tensor([0], dtype=torch.int), mol_assignment[index:]], dim=0) # supernode does not belong to any molecule
        # Update the atom_map_numbers
        atom_map_numbers = torch.cat([atom_map_numbers[:index], torch.tensor([0], dtype=torch.int), atom_map_numbers[index:]], dim=0)
        # Update the pos_encoding, if it exists
        if 'pos_encoding' in data:
            pos_encoding = torch.cat([pos_encoding[:index], torch.zeros(1, pos_encoding.shape[1]), pos_encoding[index:]], dim=0)
        atom_charges = torch.cat([atom_charges[:index], torch.zeros(1, atom_charges.shape[1]), atom_charges[index:]], dim=0)
        atom_chiral = torch.cat([atom_chiral[:index], torch.zeros(1, atom_chiral.shape[1]), atom_chiral[index:]], dim=0)
        # Update the edge_index
        edge_index = edge_index + (edge_index >= index).int()
    # Update the data object
    data.x = nodes
    data.mol_assignment = mol_assignment
    data.atom_map_numbers = atom_map_numbers
    data.edge_index = edge_index
    if 'pos_encoding' in data:
        data.pos_encoding = pos_encoding
    data.atom_charges = atom_charges
    data.atom_chiral = atom_chiral
    return data

def add_supernode_edges(cfg, data):
    """Add the connections to the supernodes, as they used to be done. 
    I think it is supposed to be so that the supernode is connected to all the atoms in the molecule with the supernode thing with the bond type 'mol', I think.
    Also need to change the bond dirs here"""
    suno_edge_attr = F.one_hot(torch.tensor([cfg.dataset.bond_types.index('mol')], dtype=torch.long), num_classes=len(cfg.dataset.bond_types))
    suno_idx = (data.x == F.one_hot(torch.tensor([cfg.dataset.atom_types.index('SuNo')], dtype=torch.long), num_classes=len(cfg.dataset.atom_types))).all(dim=-1).nonzero()
    product_indices = (data.mol_assignment == data.mol_assignment.max()).nonzero()
    # print(f'product_indices: {product_indices}')
    # print(f'suno_idx: {suno_idx}')
    for index in product_indices:
        data.edge_index = torch.cat([data.edge_index, torch.tensor([suno_idx, index], dtype=torch.long)[:,None]], dim=1)
        data.edge_index = torch.cat([data.edge_index, torch.tensor([index, suno_idx], dtype=torch.long)[:,None]], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, suno_edge_attr.clone(), suno_edge_attr.clone()], dim=0)
        if 'bond_dirs' in data:
            data.bond_dirs = torch.cat([data.bond_dirs, F.one_hot(torch.tensor([0], dtype=torch.long), data.bond_dirs.shape[-1]), F.one_hot(torch.tensor([0], dtype=torch.long), data.bond_dirs.shape[-1])], dim=0)
    return data

def fix_atom_mappings(atom_map_reactants, atom_map_products):
    # Find out which atom mappings are missing on the product side first: How to do this?
    # - I guess we make a set of all possible atom mappings and delete the actual set that we see. The remaining ones are missing, and we place them in an arbitrary order on the zero-atom map nodes
    
    # After that, we see which atom mappings are missing from the reactant side (again a set difference operation), and we place them on the dummy nodes (last num_dummy_nodes nodes)
    
    # After that, remove these missing atom mappings from the real atoms on the reactant side (since we have no info of them), and place them on the remaining dummy nodes (last num_dummy_nodes nodes)
    # - I guess we just loop over the atom mappings on the reactant side and check if they are in the set of missing atom mappings, and if so, we zero them out and place them on the dummy nodes
    # - And if we run out of dummy nodes, then we just zero out and don't place them anywhere on the reactant side (not optimal, but this shouldn't happen often)
    
    # UPDATE: I guess there would be other strategies as well:
    # 1) Just don't remove any atom mappings, and place them at some random places on the reactant side, instead of just the dummy nodes
    # 2) Remove bad atom mappings, and hope that the model will generalize such that it will work without them
    # -> For this, should look up some of those fancy papers. 

    # Alright so this now simply places the missing atom mappings on random places on the reactant side. Including the ones that were not present in the product side. 
    # So all clearly faulty atom mappings are just randomized on the reactant side. 

    # Wait can this fail in the case where size(reactant) < size(product)?
    # In that case, I guess what we want to fill out as many as we can?

    atom_map_reactants = atom_map_reactants.clone()
    atom_map_products = atom_map_products.clone()

    # Let's follow the strategy where we just put the faulty atom mappings to random locations? 
    all_possible_atom_maps = set(range(1,len(atom_map_products)+1)) 

    # NOTE the .max() originally here because it can be that the max atom mapping is quite a bit larger than the number of atoms
    # -> in that case should standardize the atom mappings to be in the range 1 to num_atoms
    # How to do this? 
    # first see which >0 atom mappings are not in the all_possible_atom_maps

    # remove duplicate atom mappings from both sides
    nums, counts = atom_map_reactants.unique(return_counts=True)
    for num, count in zip(nums, counts):
        if count > 1:
            atom_map_reactants[atom_map_reactants == num][1:] = 0
    nums, counts = atom_map_products.unique(return_counts=True)
    for num, count in zip(nums, counts):
        if count > 1:
            atom_map_products[atom_map_products == num][1:] = 0

    atom_maps_missing_in_product = torch.tensor(list(all_possible_atom_maps - set(atom_map_products.tolist())), dtype=torch.long)
    # atom_maps_missing_in_reactant = torch.tensor(list(all_possible_atom_maps - set(atom_map_reactants.tolist())), dtype=torch.long)
    # Reset the atom mappings to be in the range (1,...,num_product) <- THIS DOESN'T WORK ENTIRELY
    atom_maps_over_len = atom_map_products[atom_map_products > len(atom_map_products)]
    dict_map_to_missing_atom_maps = dict(zip(atom_maps_over_len.tolist(), atom_maps_missing_in_product.tolist()))
    for i in range(len(atom_map_products)):
        if atom_map_products[i].item() in dict_map_to_missing_atom_maps:
            atom_map_products[i] = dict_map_to_missing_atom_maps[atom_map_products[i].item()]
    for i in range(len(atom_map_reactants)):
        if atom_map_reactants[i].item() in dict_map_to_missing_atom_maps:
            atom_map_reactants[i] = dict_map_to_missing_atom_maps[atom_map_reactants[i].item()]

    atom_maps_missing_in_product = torch.tensor(list(all_possible_atom_maps - set(atom_map_products.tolist())), dtype=torch.long)
    atom_maps_missing_in_reactant = torch.tensor(list(all_possible_atom_maps - set(atom_map_reactants.tolist())), dtype=torch.long)

    # The product should be filled with atom mappings. Choose the first ones here that are zeros. Other missing atom mappings are left out
    atom_map_products[atom_map_products == 0] = atom_maps_missing_in_product[:(atom_map_products == 0).sum()]
    atom_maps_missing_in_product = atom_maps_missing_in_product[(atom_map_products == 0).sum():] # These ones should not be on the reactant side either, see below
    reactant_mask_without_atom_map = atom_map_reactants == 0
    # Get the indices
    reactant_indices_without_atom_map = torch.arange(len(atom_map_reactants))[reactant_mask_without_atom_map]
    # Choose a random subset of the indices
    reactant_indices_to_add_atom_map = reactant_indices_without_atom_map[torch.randperm(len(reactant_indices_without_atom_map))[:len(atom_maps_missing_in_reactant)]]

    if len(reactant_indices_to_add_atom_map) < len(atom_maps_missing_in_reactant):
        # If we run out of nodes, then we just don't use some of the atom maps
        reactant_indices_to_add_atom_map = reactant_indices_without_atom_map[:len(atom_maps_missing_in_reactant)]
    atom_map_reactants[reactant_indices_to_add_atom_map] = atom_maps_missing_in_reactant[:len(reactant_indices_to_add_atom_map)]
    
    if atom_map_reactants.ndim == 1:
        atom_map_reactants = atom_map_reactants[None,:]
    if atom_map_products.ndim == 1:
        atom_map_products = atom_map_products[None,:]
        
    for i in range(len(atom_map_reactants)):
        # Get valid numbers from products
        valid_numbers = atom_map_products[i][atom_map_products[i] != 0]
        
        # Get current row
        current_row = atom_map_reactants[i]
        
        # Create mask for valid numbers
        validity_mask = torch.isin(current_row, valid_numbers)
        
        # Create mask for first occurrences (False for duplicates)
        unique_vals, inverse_indices, counts = torch.unique(
            current_row, 
            return_inverse=True, 
            return_counts=True
        )
        
        # Create mask for first occurrences
        duplicate_mask = torch.zeros_like(current_row, dtype=torch.bool)
        for val_idx in range(len(unique_vals)):
            # Find positions of this value
            positions = (inverse_indices == val_idx).nonzero()
            if len(positions) > 0:
                # Keep only the first occurrence
                duplicate_mask[positions[0]] = True
    
        # Apply both masks
        atom_map_reactants[i][~(validity_mask & duplicate_mask)] = 0
        
    # TODO: this is not used anymore, but we should maybe remove it?
    # for i in range(len(atom_map_reactants)):
    #     if atom_map_reactants[i] in atom_maps_missing_in_product:
    #         atom_map_reactants[i] = 0 # zero out the atom mappings on the reactant side that are missing on the product side

    
    return atom_map_reactants.squeeze(), atom_map_products.squeeze()

def remove_atom_mapping_from_reaction(reaction_smiles):
    # Removes the atom mapping from a reaction SMILES string that contains a reaction by turning it into RDKit molecules and then back into a SMILES string
    # uses remove_atom_mapping_from_smiles
    reactants, products = reaction_smiles.split('>>')
    reactants = canonicalize_smiles(reactants)
    # reactants = [remove_atom_mapping_from_smiles(reactant, kekulize_molecule) for reactant in reactants.split('.')]
    # reactants = Chem.MolFromSmiles('.'.join('reactants'))
    products = canonicalize_smiles(products)
    return reactants + '>>' + products

def remove_atom_mapping_from_smiles(smiles, kekulize_molecule):
    # DEPRECATED
    # Removes the atom mapping from a SMILES string that contains a molecule by turning it into RDKit molecules and then back into a SMILES string
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # TODO: REPLACE THIS WITH SOME CENTRAL CODE THAT HANDLES THE CANONICALIZATION OF SMILES EVERYWHERE
    if kekulize_molecule: smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
    else: smiles = Chem.MolToSmiles(mol, canonical=True)
    return smiles

def canonicalize_smiles(smi):
    # Creates a canonical SMILES string from a SMILES string (for evaluation purposes)
    # Removes the atom mapping from the molecule because there can be no 'canonical' SMILES with atom mappings
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)

def create_canonical_smiles_from_mol(mol):
    # Creates a canonical SMILES string from an RDKit molecule object (for evaluation purposes)
    # Removes the atom mapping from the molecule because there can be no 'canonical' SMILES with atom mappings
    mol = copy.deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)# adds aromatic flags 
    except:
        pass

    return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)   

def create_smiles_like_raw_from_mol(mol):
    # Creates a canonical SMILES string from an RDKit molecule object (for evaluation purposes)
    # uses atom-mapping if available (i.e. does not remove them from the mol object)
    # because it's meant to be used to check processed data (from graphs) against raw smiles (i.e. assumes AM is the same)
    # example usage: src/test_processed_data.py -> mol.get_cano_smiles_from_dense_with_stereoche
    mol = copy.deepcopy(mol)
    try:
        Chem.SanitizeMol(mol) # adds aromatic flags 
    except:
        pass
    return Chem.MolToSmiles(mol, kekuleSmiles=False) # kekuleSmiles=False needed to get aromatic bonds in the string

from torch.utils.data import Sampler   
class VariableBatchSampler(Sampler):
    def __init__(self, data_source, size_bins, batchsize_bins, node_slices):
        self.data_source = data_source
        self.size_bins = size_bins
        self.batchsize_bins = batchsize_bins
        self.data_sizes = node_slices[1:] - node_slices[:-1]
        assert len(self.data_sizes) == len(data_source)
        self.batches = self._make_batches()

    def group_fn(self, size):
        upper_size = self.size_bins[-1] if size > self.size_bins[-1] else next(s for s in self.size_bins if size <= s)
        return upper_size

    def batch_size_fn(self, key):
        # Batch size for the given key (key is the upper bound of the size bin)
        # The first element of size_bins is non-zero, so we can use it to index directly
        return self.batchsize_bins[self.size_bins.index(key)]

    def _make_batches(self):
        # Group data by the provided function
        groups = {} # previously data_by_size
        for idx in range(len(self.data_source)):
            key = self.group_fn(self.data_sizes[idx])
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

        # Create batches within each group with varying sizes
        batches = []
        for key, indices in groups.items():
            batch_size = self.batch_size_fn(key)
            for i in range(0, len(indices), batch_size):
                batches.append(indices[i:i + batch_size])
        
        np.random.shuffle(batches)  # Shuffle the batches
        return batches

    def __iter__(self):
        # Yield indices from shuffled batches
        batch_order = np.random.permutation(len(self.batches))
        for i in batch_order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


if __name__ == '__main__':
    a = torch.tensor([12, 13, 14, 15, 16, 17, 18,  0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  0,  0,  0,  0,
         0,  0,  0, 32,  0,  0])
    b = torch.tensor([32, 33, 12, 13, 14, 15, 16, 17, 18,  9,  8,  7,  6,  5,  4,  2,  1,  3,
        11, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    a_, b_ = fix_atom_mappings(a, b)
    print(a_, b_)