import sys
sys.path.append('..')

import math
import logging
log = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
import numpy as np
import random
from src.utils import math_utils
from torch_sparse import spmm, SparseTensor  # Requires torch_sparse package

from src.neuralnet.layers import XEyTransformerLayer

from src.utils import graph, setup
from src.utils.diffusion import helpers
from src.neuralnet.layers import Xtoy, Etoy

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from torch.cuda.amp import autocast

# from torch_geometric.utils import get_laplacian

from scipy.sparse.linalg import eigsh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEmbeddingTorch(nn.Module):

    def __init__(self, dim, pos_emb_permutations):
        super().__init__()
        self.dim = dim
        self.pos_emb = SinusoidalPosEmb(dim=dim) # hidden dim 'x' in practice
        self.pos_emb_permutations = pos_emb_permutations

    def matched_positional_encodings_laplacian_scipy(self, E, atom_map_numbers, molecule_assignments, max_eigenvectors, direction = 'retro'):
        assert direction in ['retro']
        assert molecule_assignments != None, 'Need to provide molecule assigments'
        assert len(E.shape) == 3 and E.shape[1] == E.shape[2], 'E should not be in one-hot format'
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        E = E.clone() # TODO: Could also just change the different bond types all to 1
        molecule_assignments = molecule_assignments.cpu()
        atom_map_numbers = atom_map_numbers.cpu()
        
        # TODO: Make sure that these calculations are done in float32!!

        # import time
        # log.info("Calculating Laplacian pos enc!")
        # t0 = time.time()

        if direction == 'retro':
            product_indices = molecule_assignments.max(-1).values
            # Could just zero out the parts of E not related to the product & sparsify & calculate Laplacian eigenvecs
            
            # molecule_assignments is shape (bs, n) & product_indices of shape (bs,) and E of shape (bs, n, n)
            pos_embeddings = torch.zeros((bs, n, self.dim), dtype=torch.float32, device=E.device)
            
            for i in range(bs):
                E[i, product_indices[i] != molecule_assignments[i], :] = 0
                E[i, :, product_indices[i] != molecule_assignments[i]] = 0

                # TODO: Can't use pos_emb_dim here directly
                # ... what we could do is to just get whatever we get from this and then pad it to the correct dimensionality (with zeros or something)
                # ... or should we have a fixed dimensionality? -> this is tricky since some are really small, let's try padding
                k = min(max_eigenvectors, (product_indices[i] == molecule_assignments[i]).sum().item())
                pe = laplacian_eigenvectors_scipy(E[i], k) # Shape (bs, n, k)
                pe = torch.cat([pe, torch.zeros((n, self.dim - k), dtype=torch.float32, device=pe.device)], dim=-1)
                
                # Create a mapping from atom map number to the interesting positional encodings (zero pos enc for non-atom mapped atoms)
                pos_embs_prod = torch.cat([torch.zeros(1, self.dim), pe[molecule_assignments[i] == product_indices[i]]], 0)
                
                indices_of_product = (molecule_assignments[i] == product_indices[i]).nonzero().squeeze(1)
                max_atom_map = atom_map_numbers[i, :].max() # Need this because sometimes some of the atom maps are missing in the data :(
                # The atom map numbers of the product
                atom_map_nums_product = torch.cat([torch.tensor([0]), atom_map_numbers[i, indices_of_product]]) # The zero is for the non-atom mapped atoms
                # atom_map_to_idx = torch.zeros_like(atom_map_nums_product)
                atom_map_to_idx = torch.zeros(max_atom_map + 1, dtype=torch.long)
                atom_map_to_idx.scatter_(dim=0, index=atom_map_nums_product, src=torch.arange(len(atom_map_nums_product)))

                pos_embeddings[i] = pos_embs_prod[atom_map_to_idx[atom_map_numbers[i]]]
                #atom_map_to_pos_emb[atom_map_numbers[i, :]]
            return pos_embeddings
        else:
            log.info("direction must be retro")
            assert False

    def matched_positional_encodings_laplacian(self, E, atom_map_numbers, molecule_assignments, max_eigenvectors, direction = 'retro'):
        assert direction in ['retro', 'forward']
        assert molecule_assignments != None, 'Need to provide molecule assigments'
        assert len(E.shape) == 3 and E.shape[1] == E.shape[2], 'E should not be in one-hot format'
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        E = E.clone() # TODO: Could also just change the different bond types all to 1
        molecule_assignments = molecule_assignments.cpu()
        atom_map_numbers = atom_map_numbers.cpu()
        
        perm = torch.arange(atom_map_numbers.max().item()+1)[1:]
        perm = perm[torch.randperm(len(perm))]
        perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])
        atom_map_numbers = perm[atom_map_numbers]

        # TODO: Make sure that these calculations are done in float32!!
        # Soo this should be done somehow more efficiently, but preferably not actually moved all the way to the data
        # ... one way: Just remake the ELBO calculation and the sampling to calculate this just once and then use that!
        # ... yeah I guess that's the easiest hack for now

        # import time
        # log.info("Calculating Laplacian pos enc!")
        # t0 = time.time()

        if direction == 'retro':
            product_indices = molecule_assignments.max(-1).values
            # Could just zero out the parts of E not related to the product & sparsify & calculate Laplacian eigenvecs
            
            # molecule_assignments is shape (bs, n) & product_indices of shape (bs,) and E of shape (bs, n, n)
            pos_embeddings = torch.zeros((bs, n, self.dim), dtype=torch.float32, device=E.device)
            
            for i in range(bs):
                E[i, product_indices[i] != molecule_assignments[i], :] = 0
                E[i, :, product_indices[i] != molecule_assignments[i]] = 0

                # TODO: Can't use pos_emb_dim here directly
                # ... what we could do is to just get whatever we get from this and then pad it to the correct dimensionality (with zeros or something)
                # ... or should we have a fixed dimensionality? -> this is tricky since some are really small, let's try padding
                k = min(max_eigenvectors, (product_indices[i] == molecule_assignments[i]).sum().item())
                pe = laplacian_eigenvectors(E[i], k) # Shape (bs, n, k) <- can this be made faster by only calculating for the product?
                pe = torch.cat([pe, torch.zeros((n, self.dim - k), dtype=torch.float32, device=pe.device)], dim=-1)
                
                # Create a mapping from atom map number to the interesting positional encodings (zero pos enc for non-atom mapped atoms)
                # ... okay so this creates a data leak for sure... -> the pe should be assigned in order of atoms in the molecule, not in order of atom map numbers
                # 
                pos_embs_prod = torch.cat([torch.zeros(1, self.dim, device=device), pe[molecule_assignments[i] == product_indices[i]]], 0)
                

                indices_of_product = (molecule_assignments[i] == product_indices[i]).nonzero().squeeze(1)
                max_atom_map = atom_map_numbers[i, :].max() # Need this because sometimes some of the atom maps are missing in the data :(
                # The atom map numbers of the product
                atom_map_nums_product = torch.cat([torch.tensor([0]), atom_map_numbers[i, indices_of_product]]) # The zero is for the non-atom mapped atoms
                # atom_map_to_idx = torch.zeros_like(atom_map_nums_product)
                atom_map_to_idx = torch.zeros(max_atom_map + 1, dtype=torch.long)
                atom_map_to_idx.scatter_(dim=0, index=atom_map_nums_product, src=torch.arange(len(atom_map_nums_product)))
                
                # atom_map_nums_prod = torch.cat([torch.zeros(1, dtype=torch.long), atom_map_numbers[i, molecule_assignments[i] == product_indices[i]]], 0)
                # atom_map_to_product_index = torch.zeros(atom_map_nums_prod.shape[0], dtype=torch.long, device=pe.device)
                # atom_map_to_product_index.scatter_(dim=0, index=atom_map_nums_prod, src=torch.arange(len(atom_map_to_product_index)))

                #torch.zeros_like(atom_map_nums_prod, dtype=torch.float32, device=pe.device)
                #atom_map_to_pos_emb.scatter_(dim=0, index=atom_map_nums_prod, src=pos_embs_prod)
                pos_embeddings[i] = pos_embs_prod[atom_map_to_idx[atom_map_numbers[i]]]
                #atom_map_to_pos_emb[atom_map_numbers[i, :]]
            
            # for i in range(bs):
            #     # Create a mapping from atom map number to the interesting positional encodings
            #     pos_embs_prod = pe[i, molecule_assignments[i] == product_indices[i]]
            #     atom_map_nums_prod = atom_map_numbers[i, molecule_assignments[i] == product_indices[i]]
            #     atom_map_to_pos_emb = torch.scatter(dim=0, indx=atom_map_nums_prod, src=pos_embs_prod)
            #     pos_embeddings[i] = atom_map_to_pos_emb[atom_map_numbers[i, :]]
            
            # log.info(time.time() - t0)

            return pos_embeddings

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Input: (bs, )
        # Output: (bs, dim)
        # Can use the batch size dimension to store other stuff as well
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.type_as(x)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def laplacian_eigenvectors_scipy(E, k):
    """
    Computes the eigenvectors of the Laplacian matrix of a graph.

    Parameters:
    E (np.ndarray): A dense adjacency matrix of shape (n, n) where n is the number of nodes in the graph.
    k (int): The number of eigenvectors to compute.

    Returns:
    torch.Tensor: A tensor of shape (n, k) containing the real parts of the k eigenvectors of the Laplacian matrix, excluding the first eigenvector.
    """
    # TODO: Change this to calculate the eigenvectors from the dense format
    # At the moment sometimes the eigenvector dimensionality is smaller than the 
    
    edge_index, edge_attr = dense_to_sparse(E)
    num_nodes = E.shape[-1]

    L_edge_index, L_edge_weight = get_laplacian(
        edge_index,
        normalization='sym',
        num_nodes=num_nodes
    )
    
    L = to_scipy_sparse_matrix(L_edge_index, L_edge_weight, num_nodes)

    eig_vals, eig_vecs = eigsh(
        L,
        k=k+1,
        which='SA',
        return_eigenvectors=True,
        ncv=min(E.shape[0], max(20*k + 1, 40))
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
    # pe = torch.from_numpy(eig_vecs)
    sign = -1 + 2 * torch.randint(0, 2, (k, ))
    pe *= sign
    return pe

def laplacian_eigenvectors(E, k):
    """
    Computes the eigenvectors of the Laplacian matrix of a graph using PyTorch and CUDA.

    Parameters:
    E (torch.Tensor): A dense adjacency matrix of shape (n, n) where n is the number of nodes in the graph.
    k (int): The number of eigenvectors to compute.

    Returns:
    torch.Tensor: A tensor of shape (n, k) containing the real parts of the k eigenvectors of the Laplacian matrix, excluding the first eigenvector.
    """

    num_nodes = E.shape[-1]

    # # Convert dense matrix to sparse format
    # E_sparse = SparseTensor.from_dense(E)

    # Compute the Laplacian (this part needs to be adapted to your specific implementation)
    # Assuming get_laplacian function returns a SparseTensor
    edge_indices = torch.nonzero(E, as_tuple=False).t()
    L_edge_index, L_edge_weight = get_laplacian(edge_indices, normalization='sym', num_nodes=num_nodes)

    # Convert to dense format for eigen decomposition
    # L_dense = L_sparse.to_dense()
    L_dense = torch.zeros((num_nodes, num_nodes), device=E.device)
    L_dense[L_edge_index[0], L_edge_index[1]] = L_edge_weight

    # Compute eigenvalues and eigenvectors using PyTorch
    eig_vals, eig_vecs = torch.linalg.eigh(L_dense)

    # Sort eigenvectors based on eigenvalues and exclude the first eigenvector
    sorted_indices = torch.argsort(eig_vals)
    pe = eig_vecs[:, sorted_indices[1:k + 1]]

    # Apply random sign flipping
    sign = -1 + 2 * torch.randint(0, 2, (k, ), device=E.device)
    pe *= sign

    return pe

# def laplacian_eigenvectors(E, k):
#     """
#     Computes the eigenvectors of the Laplacian matrix of a graph.

#     Parameters:
#     E (np.ndarray): A dense adjacency matrix of shape (n, n) where n is the number of nodes in the graph.
#     k (int): The number of eigenvectors to compute.

#     Returns:
#     torch.Tensor: A tensor of shape (n, k) containing the real parts of the k eigenvectors of the Laplacian matrix, excluding the first eigenvector.
#     """
#     # TODO: Change this to calculate the eigenvectors from the dense format
#     # At the moment sometimes the eigenvector dimensionality is smaller than the 
    
#     edge_index, edge_attr = dense_to_sparse(E)
#     num_nodes = E.shape[-1]

#     L_edge_index, L_edge_weight = get_laplacian(
#         edge_index,
#         normalization='sym',
#         num_nodes=num_nodes
#     )
    
#     L = to_scipy_sparse_matrix(L_edge_index, L_edge_weight, num_nodes)

#     eig_vals, eig_vecs = eigsh(
#         L,
#         k=k+1,
#         which='SA',
#         return_eigenvectors=True,
#         ncv=min(E.shape[0], max(20*k + 1, 40))
#     )

#     eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
#     pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
#     # pe = torch.from_numpy(eig_vecs)
#     sign = -1 + 2 * torch.randint(0, 2, (k, ))
#     pe *= sign
#     return pe

class GraphTransformerWithYStacked(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), pos_emb_permutations: int = 0,
                 improved=False, dropout=0.1, p_to_r_skip_connection=False, p_to_r_init=10.):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.input_dim_X = input_dims['X']
        self.pos_emb_permutations = pos_emb_permutations
        self.p_to_r_skip_connection = p_to_r_skip_connection

        self.pos_emb_module = PositionalEmbeddingTorch(dim=input_dims['X'], pos_emb_permutations=-1)

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X']*3, hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in) # Reactants, products, and positional encodings

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E']*2, hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            improved=improved, dropout=dropout)
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))
        
        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))
        
        if self.p_to_r_skip_connection:
            # for p in self.mlp_out_X[-1].parameters(): # zero out 
            #     p.detach().zero_()
            # for p in self.mlp_out_E[-1].parameters(): # zero out 
            #     p.detach().zero_()
            self.skip_scaling = nn.Parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_2 = nn.Parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_3 = nn.Parameter(torch.tensor([1.], dtype=torch.float))

    def cut_reaction_reactant_part_X_only(self, X, reaction_side_separation_index):
        # TODO: REDO EVERYTHING HERE
        device = X.device
        rct_side = torch.arange(X.shape[1], device=device)[None,:].repeat(X.shape[0], 1) < reaction_side_separation_index[:,None]
        prod_side = torch.arange(X.shape[1], device=device)[None,:].repeat(X.shape[0], 1) > reaction_side_separation_index[:,None]

        # prod_assignment = mol_assignments.max(-1).values
        bs, n, dx = X.shape[0], X.shape[1], X.shape[2]
        # rct_mask = ((mol_assignments > 0) & (mol_assignments < prod_assignment[:,None]))
        # first_prod_index = torch.argmax((prod_assignment[:,None] == mol_assignments).int(), dim=1) # argmax picks the first nonzero value, since others are ones
        # biggest_reactant_size = first_prod_index.max()
        biggest_reactant_set_size = reaction_side_separation_index.max()
        node_mask_cut = torch.zeros(bs, biggest_reactant_set_size, device=device, dtype=torch.bool)
        X_cut = torch.zeros(bs, biggest_reactant_set_size, dx, device=device)
        for i in range(bs):
            X_cut[i, rct_side[i][:biggest_reactant_set_size]] = X[i, rct_side[i]]
            node_mask_cut[i, rct_side[i][:biggest_reactant_set_size]] = True
        return X_cut, node_mask_cut

    def cut_reaction_reactant_part(self, X, E, reaction_side_separation_index):
        # reaction_side_separation_index (n,) are the indices of the supernode, an additional node not belonging to either reaction
        # Takes the graph specified by X, E and returns the subset that only has the reactants. Padding 
        device = X.device
        rct_side = torch.arange(X.shape[1], device=device)[None,:].repeat(X.shape[0], 1) < reaction_side_separation_index[:,None]
        prod_side = torch.arange(X.shape[1], device=device)[None,:].repeat(X.shape[0], 1) > reaction_side_separation_index[:,None]
        # prod_assignment = mol_assignments.max(-1).values
        bs, n, dx = X.shape[0], X.shape[1], X.shape[2]
        de = E.shape[3]

        biggest_reactant_set_size = reaction_side_separation_index.max()
        # need to create new padding, and also node_ma
        # or could just zero out everything starting from the first prod index? X[first_prod_index]
        # Is this way too complicated? Nah a different amount of zeroing needs to be done for each element in the batch
        node_mask_cut = torch.zeros(bs, biggest_reactant_set_size, device=device, dtype=torch.bool)
        X_cut = torch.zeros(bs, biggest_reactant_set_size, dx, device=device)
        # old method:
        for i in range(bs):
            X_cut[i, rct_side[i][:biggest_reactant_set_size]] = X[i, rct_side[i]]
            node_mask_cut[i, rct_side[i][:biggest_reactant_set_size]] = True
        # Vectorized method:
        # Create batch index tensor
        # batch_indices = torch.arange(bs).view(-1, 1)
        # # Use torch.nonzero to find indices where rct_mask is True
        # true_indices = torch.nonzero(rct_mask)
        # # Filter for the first 'new_max_nodes' per batch
        # filtered_indices = true_indices[true_indices[:, 1] < biggest_reactant_size]
        # # Use advanced indexing to copy the values
        # X_cut[batch_indices, filtered_indices[:, 1], :] = X[batch_indices, filtered_indices[:, 0], :]
        # node_mask_cut[batch_indices, filtered_indices[:, 1]] = True

        E_cut = torch.zeros(bs, biggest_reactant_set_size, biggest_reactant_set_size, de, device=device)
        # old method: 
        for i in range(bs):
            rct_mask_E_cut = rct_side[i][:biggest_reactant_set_size][:,None] * rct_side[i][:biggest_reactant_set_size][None,:]
            E_cut[i][rct_mask_E_cut] = E[i][rct_side[i][:,None] * rct_side[i][None,:]]

        # Vectorized method:
        # Correctly expand rct_mask to apply it to both dimensions of E
        # rct_mask_expanded = rct_mask[:, :biggest_reactant_size, None, None]
        # rct_mask_expanded = rct_mask_expanded * rct_mask_expanded.transpose(1, 2)
        # # Apply the expanded mask to E
        # selected_E = E[:, :biggest_reactant_size, :biggest_reactant_size, :] * rct_mask_expanded
        # # Flatten the tensors for advanced indexing
        # E_cut_flat = E_cut.view(bs, -1, de)
        # selected_E_flat = selected_E.view(bs, -1, de)
        # rct_mask_expanded_flat = rct_mask_expanded.view(bs, -1)
        # # Assign the flattened selected elements to E_cut
        # E_cut_flat[rct_mask_expanded_flat] = selected_E_flat[rct_mask_expanded_flat]
        # # Reshape E_cut back to original shape
        # E_cut = E_cut_flat.view(bs, biggest_reactant_size, biggest_reactant_size, de)

        return X_cut, E_cut, node_mask_cut

    def get_X_E_product_aligned_with_reactants(self, X, E, atom_map_numbers, reaction_side_separation_index):
        orig_E = E.clone()
        orig_X = X.clone()
        bs, n, dx = X.shape[0], X.shape[1], X.shape[2]
        device = X.device
        # First we need to split X and E up, find the product indices, then align the product with the reactants.
        # but need to find the first index that belongs to products, then cut X based on that
        # ... in the output, expand X out again to the correct dimensions
        
        # suno_number = atom_types.index("SuNo")
        # reaction_side_separation_index = (X.argmax(-1) == suno_number).nonzero(as_tuple=True)[1]
        rct_side = torch.arange(X.shape[1], device=device)[None,:].repeat(X.shape[0], 1) < reaction_side_separation_index[:,None]
        prod_side = torch.arange(X.shape[1], device=device)[None,:].repeat(X.shape[0], 1) > reaction_side_separation_index[:,None]
        # prod_assignment = mol_assignments.max(-1).values

        atom_map_numbers_prod, atom_map_numbers_rct = atom_map_numbers.clone(), atom_map_numbers.clone()
        atom_map_numbers_prod[rct_side] = 0
        atom_map_numbers_rct[prod_side] = 0
        # The next picks out the relevant indices, they are of different lengths for different elements in the batch
        # -> need to change into lists, I guess

        # Okay so here we need to pad to the correct dimension as well: The biggest along the.. Hmm the node mask has to be redone also, I guess. 
        # soo this outputs the node mask as well?

        #suno_number = atom_types.index("SuNo")
        # reaction_side_separation_index = (X.argmax(-1) == suno_number).nonzero(as_tuple=True)[1]

        X_cut, E_cut, node_mask_cut = self.cut_reaction_reactant_part(X, E, reaction_side_separation_index)
        
        # Okay so now we have the correct input to the NN. Now need to concatenate Y input as well, using the atom mapping. 
        # This can be done still with the following piece of code: 
        X_prod, E_prod = torch.zeros_like(X), torch.zeros_like(E)

        atom_map_numbers_prod_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_prod[i]>0] for i in range(bs)]
        atom_map_numbers_rct_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_rct[i]>0] for i in range(bs)]
        # atom_map_numbers_prod_idx = torch.arange(atom_map_numbers.shape[-1])[atom_map_numbers_prod > 0]
        # atom_map_numbers_rct_idx = torch.arange(atom_map_numbers.shape[-1])[atom_map_numbers_rct.squeeze(0) > 0]
        # The selection chooses the correct atom map numbers
        E_prods_atom_mapped = [
            orig_E[i,atom_map_numbers_prod_idxs[i]][:, atom_map_numbers_prod_idxs[i]].unsqueeze(0)
            for i in range(bs)]
        assert all(len(atom_map_numbers_prod_idxs[i]) == len(atom_map_numbers_rct_idxs[i]) for i in range(bs))
        # Create the permutation matrix required to place the atom-mapped atoms on the product side to the reactant side
        # TODO: These may throw errors right now, in case there's some atom mappings that don't match on both sides
        # TODO: And it may be defined the wrong way around here
        Ps = [math_utils.create_permutation_matrix_torch(atom_map_numbers_prod[i][atom_map_numbers_prod_idxs[i]],
                                                    atom_map_numbers_rct[i][atom_map_numbers_rct_idxs[i]]).float().to(device)
                                                    for i in range(bs)]
        P_expanded = [P.unsqueeze(0) for P in Ps] # The unsqueeze will be unnecessary with proper batching here
        # Permute the edges obtained from the product: P @ E @ P^T
        E_prods_am_permuted = [torch.movedim(P_expanded[i].transpose(dim0=1,dim1=2) @ torch.movedim(E_prods_atom_mapped[i].float(), -1, 1) @ P_expanded[i], 1, -1) for i in range(bs)]
        # E_prods_am_permuted = [F.one_hot(E_prods_am_permuted[i], self.out_dim_E) for i in range(bs)]
        for i in range(bs):
            # The following is used for choosing which parts to change in the output
            am_rct_selection = (atom_map_numbers_rct[i] > 0)
            E_prod[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                E_prods_am_permuted[i].shape[3]).float()
            E_prod[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
        
        # ... do the same for X
        # The selection chooses the correct atom map numbers
        X_prods_atom_mapped = [orig_X[i,:,:][atom_map_numbers_prod_idxs[i]].unsqueeze(0) for i in range(bs)]
        # need to unsqueeze to do batched matrix multiplication correctly: (bs,N,N) @ (bs,N,1) -> (bs,N,1). (N is the count of atom mapped nodes)
        X_prods_am_permuted = [P_expanded[i].transpose(dim0=1,dim1=2) @ X_prods_atom_mapped[i] for i in range(bs)]
        # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
        for i in range(bs):
            am_rct_selection = (atom_map_numbers_rct[i] > 0)
            X_prod[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float()

        X_prod_cut, E_prod_cut, node_mask_cut_2 = self.cut_reaction_reactant_part(X_prod, E_prod, reaction_side_separation_index)
        assert torch.equal(node_mask_cut, node_mask_cut_2)
        return X_cut, X_prod_cut, E_cut, E_prod_cut, node_mask_cut
        # return torch.cat([X_cut, X_prod_cut], dim=-1), torch.cat([E_cut, E_prod_cut], dim=-1)
        
    def expand_to_full_size(self, X, E, n_nodes):
        # Fills X and E with zeros up to n_nodes dim
        bs, n, dx = X.shape[0], X.shape[1], X.shape[2]
        de = E.shape[3]
        X_ = torch.zeros(bs, n_nodes, dx, device=X.device)
        E_ = torch.zeros(bs, n_nodes, n_nodes, de, device=X.device)
        X_[:, :n] = X
        E_[:, :n, :n] = E
        return X_, E_

    def choose_pos_enc(self, X, E, reaction_side_separation_index, mol_assignments, atom_map_numbers, pos_encoding_type, num_lap_eig_vectors):
        
        if pos_encoding_type == 'laplacian_pos_enc_gpu':
            # This needs to be designed as:
            # 1. Cut the correct part of E
            # 2. Calculate the eigendecomposition
            # 3. Place it in the correct part with the code that we already have elsewhere, maybe just reuse the code that we already have? And then just cut it. Let's see...
            pos_encodings = self.pos_emb_module.matched_positional_encodings_laplacian(E.argmax(-1), atom_map_numbers, mol_assignments, num_lap_eig_vectors, direction = 'retro')
            pos_encodings, _ = self.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index) # ... this could be done in forward as well, more efficient with multiple GPUs. Actually both parts hmm
        elif pos_encoding_type == 'laplacian_pos_enc':
            # 3. Place it in the correct part with the code that we already have elsewhere, maybe just reuse the code that we already have? And then just cut it. Let's see...
            pos_encodings = self.pos_emb_module.matched_positional_encodings_laplacian_scipy(E.argmax(-1), atom_map_numbers, mol_assignments, num_lap_eig_vectors, direction = 'retro')
            pos_encodings, _ = self.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index)
        else:
            pos_encodings = torch.zeros(X.shape[0], X.shape[1], self.input_dim_X, device=X.device)
            pos_encodings, _ = self.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index)
        return pos_encodings

    def forward(self, X, E, y, node_mask, atom_map_numbers, pos_encodings, mol_assignments, use_pos_encoding_if_applicable, pos_encoding_type, num_lap_eig_vectors, atom_types):
        # TODO: Change this so that Y is included with atom mapping in X and E
        # -> need to figure out the relative permutation, I guess
        # suno_number = atom_types.index("SuNo")
        # Use the first mol assignment here for this
        prod_assignment = mol_assignments.max(-1).values
        reaction_side_separation_index = (mol_assignments == prod_assignment[:,None]).to(torch.int).argmax(-1) - 1 # -1 because the supernode doesn't belong to product according to mol_assignments
        #(X == suno_number).nonzero(as_tuple=True)[1]

        if pos_encodings == None: 
            with autocast(enabled=False):
                pos_encodings = self.choose_pos_enc(X, E, reaction_side_separation_index, mol_assignments, atom_map_numbers, pos_encoding_type, num_lap_eig_vectors)
        pos_encodings *= use_pos_encoding_if_applicable[:,None,None].to(pos_encodings.device).float()

        n_nodes_original = X.shape[1]
        orig_node_mask = node_mask        

        X_cut, X_prod_aligned, E_cut, E_prod_aligned, node_mask_cut = self.get_X_E_product_aligned_with_reactants(X, E, atom_map_numbers, reaction_side_separation_index)

        X = torch.cat([X_cut, X_prod_aligned, pos_encodings], dim=-1)
        E = torch.cat([E_cut, E_prod_aligned], dim=-1)
        node_mask = node_mask_cut
        # node_mask = node_mask[]

        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        # orig_E = E.clone()
        # orig_X = X.clone()

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = graph.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Add the positional encoding to X. X shape is now (bs, n, dx)
        # TODO: Maybe concatenate instead so that this works with the Laplacian eigenvectors as well?
        # X = X + pos_encodings

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = y[..., :self.out_dim_y] # TODO: Changed
        #y = self.mlp_out_y(y) # TODO: Changed

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        # E *= self.skip_scaling_3
        # X *= self.skip_scaling_3

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        # Potential edge-skip connection from product side to reactant side
        if self.p_to_r_skip_connection:
            E += E_prod_aligned[...,:self.out_dim_E] * self.skip_scaling_2
            X += X_prod_aligned[...,:self.out_dim_X] * self.skip_scaling

        X, E = self.expand_to_full_size(X, E, n_nodes_original) # for calculating the loss etc. 

        return X, E, y, orig_node_mask
        # return graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask, atom_map_numbers=atom_map_numbers).mask(node_mask)
        
class GraphTransformerWithYAndPosEmb(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), improved=False,
                 dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.pos_emb = SinusoidalPosEmb(dim=hidden_dims['dx'])

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            improved=improved, dropout=dropout)
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))
        
        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = graph.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y
        # Add the positional encoding to X. X shape is now (bs, n, dx)
        X = X + self.pos_emb(torch.arange(n).to(X.device).to(torch.float32).repeat(bs)).reshape(bs, n, self.pos_emb.dim)
        
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)
    
        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask).mask(node_mask)