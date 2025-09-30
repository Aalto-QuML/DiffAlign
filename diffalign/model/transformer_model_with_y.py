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
from diffalign.model import math_utils
import copy
from diffalign.model.layers import XEyTransformerLayer

from diffalign.model import data_utils
from diffalign.data import graph

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from scipy.sparse.linalg import eigsh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphTransformerWithY(nn.Module):
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

    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask):

        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        # log.info(E.shape)
        # log.info(self.mlp_in_E)
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = graph.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        # log.info(y.shape)
        # log.info(self.mlp_out_y)
        y = y[..., :self.out_dim_y] # TODO: Changed
        #y = self.mlp_out_y(y) # TODO: Changed
    
        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask

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

def laplacian_eigenvectors_gpu(E, k):
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

class PositionalEmbedding(nn.Module):

    def __init__(self, dim, pos_emb_permutations):
        super().__init__()
        self.dim = dim
        self.pos_emb = SinusoidalPosEmb(dim=dim) # hidden dim 'x' in practice
        self.pos_emb_permutations = pos_emb_permutations

    def matched_positional_encodings_laplacian(self, E, atom_map_numbers, molecule_assignments, max_eigenvectors, direction = 'retro'):
        assert direction in ['retro', 'forward']
        assert molecule_assignments != None, 'Need to provide molecule assigments'
        assert len(E.shape) == 3 and E.shape[1] == E.shape[2], 'E should not be in one-hot format'
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        E = E.clone() # TODO: Could also just change the different bond types all to 1
        molecule_assignments = molecule_assignments.cpu()
        atom_map_numbers = atom_map_numbers.cpu()

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

            return pos_embeddings
        
        else:
            # TODO: Can we just calculate the Laplacian eigenvecs for all the graphs in batch together?
            # -> does it automatically create the separte Laplacians? I guess it should if the 'retro' case works
            # ... but we need to zero out the parts of E that are not related to the reactants

            product_indices = molecule_assignments.max(-1).values
            for i in range(bs):
                E[i, product_indices[i] == molecule_assignments[i], :] = 0
                E[i, :, 0 == molecule_assignments[i]] = 0
                E[i, product_indices[i] == molecule_assignments[i], :] = 0
                E[i, :, 0 == molecule_assignments[i]] = 0
            
            pe = laplacian_eigenvectors_scipy(E, self.dim) # Shape (bs, n, k)

            pos_embeddings = torch.zeros((bs, n, self.dim), dtype=torch.float32, device=E.device)
            for i in range(bs):
                # Create a mapping from atom map number to the interesting positional encoding
                reactant_indices = (molecule_assignments[i] != 0) & (molecule_assignments[i] != product_indices[i])
                pos_embs_reactants = pe[i, reactant_indices]
                atom_map_nums_reactants = atom_map_numbers[i, reactant_indices]
                atom_map_to_pos_emb = torch.scatter(dim=0, indx=atom_map_nums_reactants, src=pos_embs_reactants)
                pos_embeddings[i] = atom_map_to_pos_emb[atom_map_numbers[i, :]]

            return pos_embeddings

    def unaligned_positional_encodings_laplacian(self, E, atom_map_numbers, molecule_assignments, max_eigenvectors, direction = 'retro'):
        assert direction in ['retro', 'forward']
        assert molecule_assignments != None, 'Need to provide molecule assigments'
        assert len(E.shape) == 3 and E.shape[1] == E.shape[2], 'E should not be in one-hot format'
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        E = E.clone() # TODO: Could also just change the different bond types all to 1
        molecule_assignments = molecule_assignments.cpu()
        atom_map_numbers = atom_map_numbers.cpu()
        
        # molecule_assignments is shape (bs, n) & product_indices of shape (bs,) and E of shape (bs, n, n)
        pos_embeddings = torch.zeros((bs, n, self.dim), dtype=torch.float32, device=E.device)
        
        for i in range(bs):
            k = min(max_eigenvectors, len(molecule_assignments[i]))
            pe = laplacian_eigenvectors_scipy(E[i], k) # Shape (bs, n, k)
            pe = torch.cat([pe, torch.zeros((n, self.dim - k), dtype=torch.float32, device=pe.device)], dim=-1)
            pos_embeddings[i] = pe
        
        return pos_embeddings

    def matched_positional_encodings_laplacian_gpu(self, E, atom_map_numbers, molecule_assignments, max_eigenvectors, direction = 'retro'):
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
                pe = laplacian_eigenvectors_gpu(E[i], k) # Shape (bs, n, k) <- can this be made faster by only calculating for the product?
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

    def matched_positional_encodings_sinusoidal(self, atom_map_numbers, molecule_assignments, direction = 'retro'):
        assert direction in ['retro', 'forward']
        assert molecule_assignments != None, 'Need to provide molecule assigments'
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        atom_map_numbers, molecule_assignments = atom_map_numbers.cpu(), molecule_assignments.cpu()
        # TODO: Make this work with the GPU
        # Should match the positional encodings on both sides of the reaction: Give as input whether it's forward or retrosynthesis
        # TODO: Change the input dimensionality of the neural net input, but don't screw up the model definition so that we can still load weights from before
        DATA_IDX = 0
        if direction == 'retro':
            # THIS PART DEBUGGED
            product_indices = molecule_assignments.max(-1).values # (of shape (bs,), contains which index is the product in each batch element)            
            # maps from atom map number to the product index (and thus positional encoding)
            positions = torch.zeros_like(atom_map_numbers)
            for i in range(bs):
                indices_of_product = (molecule_assignments[i]==product_indices[i]).nonzero().squeeze(1)
                max_atom_map = atom_map_numbers[i,:].max().item() # Need this because sometimes some of the atom maps are missing in the data :(
                # The atom map numbers of the product
                atom_map_nums_product = torch.cat([torch.tensor([0]), atom_map_numbers[i, indices_of_product]]) # The zero is for the non-atom mapped atoms
                # atom_map_to_idx = torch.zeros_like(atom_map_nums_product)
                atom_map_to_idx = torch.zeros(max_atom_map + 1, dtype=torch.long)
                atom_map_to_idx.scatter_(dim=0, index=atom_map_nums_product, src=torch.arange(len(atom_map_nums_product)))
                # Then map the atom_map_numbers to the positional encodings
                positions[i] = atom_map_to_idx[atom_map_numbers[i, :]]

                DATA_IDX += 1
            
            positional_encodings = self.pos_emb(positions.to(device).to(torch.float32).reshape(-1)).reshape(bs, n, self.dim)
            return positional_encodings
        else:
            # Here we assume that the data is equal sizes on both sizes, so that atom mappings exist for both sides
            product_indices = molecule_assignments.max(-1).values # We can deduce the reactant indices from the product indices
            num_reactants = product_indices - 1
            # maps from atom map number to the reactant indices that have atom map numbers != 0
            # TODO: Whether to start indexing of each reactant from zero or to continue from one to the next and randomly permute them?
            # ... maybe just each separately for now, can add the fancier pos encodings later
            positions = torch.zeros_like(atom_map_numbers)
            for i in range(bs):
                # Build a mapping from atom map number to the reactant index. For atom map 0, we just map to zero.
                atom_map_nums_reactants = [torch.tensor([0])]
                corresponding_reactant_indices = [torch.tensor([0], dtype=torch.long)]

                # We first shuffle the reactants to make sure that we don't use the order of the reactants as a signal
                reactants = list(range(1, num_reactants[i] + 1))
                random.shuffle(reactants)

                offset = 1
                for reactant_index in reactants:
                    indices_of_reactant = (molecule_assignments[i] == reactant_index)
                    atom_map_nums_reactant = atom_map_numbers[i, indices_of_reactant]
                    corresponding_reactant_idx = torch.arange(offset, offset + len(atom_map_nums_reactant))
                    atom_map_nums_reactants.append(atom_map_nums_reactant)
                    corresponding_reactant_indices.append(corresponding_reactant_idx)
                    offset = offset + len(atom_map_nums_reactant) # Number of atoms in the reactant
                atom_map_nums_reactants = torch.cat(atom_map_nums_reactants)
                corresponding_reactant_indices = torch.cat(corresponding_reactant_indices)
                atom_map_to_idx = torch.scatter(dim=0, indx=atom_map_nums_reactants, src=corresponding_reactant_indices)
                positions[i] = atom_map_to_idx[atom_map_numbers[i, :]]

            positional_encodings = self.pos_emb(positions.to(atom_map_numbers.device).to(torch.float32).reshape(-1)).reshape(bs, n, self.dim)
            return positional_encodings

    def matched_positional_encodings_infoleak(self, atom_map_numbers, molecule_assignments, direction = 'retro'):
        assert direction in ['retro', 'forward']
        assert molecule_assignments != None, 'Need to provide molecule assigments'
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        original_device = atom_map_numbers.device
        atom_map_numbers = atom_map_numbers.cpu()
        molecule_assignments = molecule_assignments.cpu()
        if self.pos_emb_permutations > 0:
            # TODO: Maybe do this on the GPU 
            # We want to permute the atom_map_numbers so that each number above 1 turns to another
            # This is different for each graph in the batch
            max_atom_map = atom_map_numbers.max(-1).values.cpu().int()
            # Choose the permutation numbers here
            perm_numbers = np.random.randint(0,self.pos_emb_permutations, bs)
            # atom_map_numbers is a (bs, n) tensor with values in [0, max_atom_map]
            # The zeros we want to keep as zeros, the others we permute
            permutations = [list(range(1, max_map+1)) for max_map in max_atom_map]
            _ = [random.Random(perm_number).shuffle(permutations[idx]) for idx, perm_number in enumerate(perm_numbers)]
            max_max_atom_map = max(max_atom_map)
            permutations = torch.tensor([[0] + perm + (max_max_atom_map - len(perm))*[0] for perm in permutations]).to(atom_map_numbers.device) # Don't change the zeros, non-atom mapped
            # permutations is of shape (bs, max_max_atom_map) -> we want the numbers from there that are indexed
            # with the atom map numbers for each row separately
            atom_map_numbers = atom_map_numbers.long()
            for i in range(bs):
                atom_map_numbers[i] = permutations[i,atom_map_numbers[i]]
        elif self.pos_emb_permutations == -1:
            # Let's consider all permutations
            max_atom_map = atom_map_numbers.max(-1).values.cpu().int()
            permutations = [list(range(1, max_map+1)) for max_map in max_atom_map]
            _ = [random.shuffle(permutations[idx]) for idx in range(bs)]
            max_max_atom_map = max(max_atom_map)
            permutations = torch.tensor([[0] + perm + (max_max_atom_map - len(perm))*[0] for perm in permutations]).to(atom_map_numbers.device)
            atom_map_numbers = atom_map_numbers.long()
            for i in range(bs):
                atom_map_numbers[i] = permutations[i,atom_map_numbers[i]]
        else:
            # No permutations
            pass

        return self.pos_emb(atom_map_numbers.to(torch.float32).reshape(-1)).reshape(bs, n, self.dim).to(original_device)
    
    def additional_positional_encodings_smiles(self, atom_map_numbers, mol_assignment):
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        original_device = atom_map_numbers.device
        atom_map_numbers = atom_map_numbers.cpu()
        # Create a tensor to store the positional encodings
        pos_encodings = torch.zeros(bs, n, self.dim, device=original_device)

        for i in range(bs):
            # Find the last product atom index
            last_product_idx = (mol_assignment[i] == mol_assignment[i].max()).nonzero().max().item()

            # Find non-atom-mapped atoms (where atom_map_numbers == 0) up to the last product atom
            non_mapped_mask = (atom_map_numbers[i] == 0)
            non_mapped_mask[last_product_idx:] = False
            non_mapped_count = non_mapped_mask.sum().item()

            # Generate positional encodings for non-mapped atoms
            if non_mapped_count > 0:
                non_mapped_positions = torch.arange(1, non_mapped_count + 1, device=original_device).float()
                non_mapped_encodings = self.pos_emb(non_mapped_positions)

                # Assign the encodings to the non-mapped atoms
                pos_encodings[i, non_mapped_mask] = non_mapped_encodings

        return pos_encodings
    
    def additional_positional_encodings_product_id(self, molecule_assignments):
        bs, n = molecule_assignments.shape[0], molecule_assignments.shape[1]
        original_device = molecule_assignments.device
        molecule_assignments = molecule_assignments.cpu()
        # Create a tensor to store the positional encodings
        pos_encodings = torch.zeros(bs, n, self.dim, device=original_device)
        
        # Create a constant vector for product atoms
        product_identifier = torch.ones(self.dim, device=original_device)
        
        # Add the constant vector to product atoms
        max_assignment = molecule_assignments.max(dim=1, keepdim=True).values
        is_product = (molecule_assignments == max_assignment)
        pos_encodings[is_product] = product_identifier
        return pos_encodings
    
    def matched_positional_encodings_gaussian(self, atom_map_numbers, molecule_assignments):
        bs, n = atom_map_numbers.shape[0], atom_map_numbers.shape[1]
        original_device = atom_map_numbers.device
        atom_map_numbers = atom_map_numbers.cpu()
        gaussian_noise = torch.randn(bs, n, self.dim)
        gaussian_noise[:,0,:] = 0

        pos_embeddings = torch.zeros((bs, n, self.dim), dtype=torch.float32)
        for i in range(bs):
            pos_embs_prod = gaussian_noise[i, atom_map_numbers[i]]
            pos_embeddings[i] = pos_embs_prod
        return pos_embeddings.to(original_device)

def rowwise_atom_mapping_mask(atom_map_numbers_rct, atom_map_numbers_prod):
    """
    For each row in atom_map_numbers_rct, zero out values that don't appear 
    in the corresponding row of atom_map_numbers_prod (excluding zeros)
    
    Parameters:
    atom_map_numbers_rct: torch.Tensor of shape (batch_size, seq_len_rct)
    atom_map_numbers_prod: torch.Tensor of shape (batch_size, seq_len_prod)
    
    Returns:
    torch.Tensor of same shape as atom_map_numbers_rct with masked values
    """
    result = atom_map_numbers_rct.clone()
    
    for i in range(len(atom_map_numbers_rct)):
        # Get non-zero values in this row of prod tensor
        valid_numbers = atom_map_numbers_prod[i][atom_map_numbers_prod[i] != 0]
        
        # Create mask for this row of rct tensor
        mask = torch.isin(atom_map_numbers_rct[i], valid_numbers)
        
        # Apply mask
        result[i][~mask] = 0
        
    return result

def get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, alignment_type, out_dim_E, out_dim_X, device):
    bs = orig_X.shape[0]
    prod_assignment = mol_assignment.max(-1).values
    atom_map_numbers_prod, atom_map_numbers_rct = atom_map_numbers.clone(), atom_map_numbers.clone()
    # NOTE: only keep the atom_map_numbers_rct
    # get the matching am by counting the occurences of each number
    atom_map_numbers_prod[mol_assignment < prod_assignment[:,None]] = 0
    atom_map_numbers_rct[mol_assignment == prod_assignment[:,None]] = 0
    atom_map_numbers_rct = rowwise_atom_mapping_mask(atom_map_numbers_rct, atom_map_numbers_prod)
    if not (atom_map_numbers_prod.sort()[0]==atom_map_numbers_rct.sort()[0]).all():
        # print(f'atom_map_numbers_prod.shape: {atom_map_numbers_prod.shape}')
        # for i in range(atom_map_numbers_prod.shape[0]):
        #     if not (atom_map_numbers_prod[i].sort()[0] == atom_map_numbers_rct[i].sort()[0]).all():
        #         print(f'i {i}')
        #         print(f'atom_map_numbers_prod: {atom_map_numbers_prod[i].sort()[0]}')
        #         print(f'atom_map_numbers_rct: {atom_map_numbers_rct[i].sort()[0]}')
        #         print(f'\n')
        
        assert False, f"The atom map numbers should be the same for products and reactants over the batch."
        #assert (atom_map_numbers_prod.sort()[0]==atom_map_numbers_rct.sort()[0]).all(), f"The atom map numbers should be the same for products and reactants over the batch."
    
    # get indices in rct matching the am of product
    # select am which are not zero and which match with the produc
    
    # The next picks out the relevant indices, they are of different lengths for different elements in the batch
    # -> need to change into lists, I guess
    atom_map_numbers_prod_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_prod[i]>0] for i in range(bs)]
    atom_map_numbers_rct_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_rct[i]>0] for i in range(bs)]

    # The first selection drops out potential extra features calculated in the preprocessing (leaving one-hot encodings, 
    # although we don't have extra features in practice for E), the second neuralnet.architecture chooses the correct atom map numbers
    E_prods_atom_mapped = [
        orig_E[:,:,:,:out_dim_E][i,atom_map_numbers_prod_idxs[i]][:, atom_map_numbers_prod_idxs[i]].unsqueeze(0)
        for i in range(bs)]
    
    # NOTE: what does this do???
    # for i in range(bs):
    #     if not len(atom_map_numbers_prod_idxs[i]) == len(atom_map_numbers_rct_idxs[i]):
    #         # print(f'{i}: len {len(atom_map_numbers_prod_idxs[i])}, \n atom_map_numbers_prod_idxs: {atom_map_numbers_prod_idxs[i]} \n atom_map_numbers_prod: {atom_map_numbers_prod[i]}\n')
    #         #print(f'{i}: len {len(atom_map_numbers_rct_idxs[i])}, atom_map_numbers_rct_idxs: {atom_map_numbers_rct_idxs[i]} \n atom_map_numbers_rct: {atom_map_numbers_rct[i]}\n')
    #         try:
    #             from mpi4py import MPI
    #         except ImportError: # mpi4py is not installed, for local experimentation
    #             MPI = None
    #             log.warning("mpi4py not found. MPI will not be used.")
    #         if MPI:
    #             comm = MPI.COMM_WORLD
    #             mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
    #             mpi_rank = comm.Get_rank()
    #             log.info(f"MPI size: {mpi_size}, MPI rank: {mpi_rank}")

    #assert all(len(atom_map_numbers_prod_idxs[i]) == len(atom_map_numbers_rct_idxs[i]) for i in range(bs)), 'The atom map numbers should be the same length for each element in the batch'
    #print(f'past assert\n')
    # Create the permutation matrix required to place the atom-mapped atoms on the product side to the reactant side
    
    if alignment_type == 'old':
        Ps = [math_utils.create_permutation_matrix_torch(atom_map_numbers_prod_idxs[i] - atom_map_numbers_prod_idxs[i].min(),
                                                    atom_map_numbers_rct_idxs[i] - atom_map_numbers_rct_idxs[i].min()).float().to(device)
                                                    for i in range(bs)]
        P_expanded = [P.unsqueeze(0) for P in Ps] # The unsqueeze will be unnecessary with proper batching here
        # Permute the edges obtained from the product: P @ E @ P^T
        E_prods_am_permuted = [torch.movedim(P_expanded[i] @ torch.movedim(E_prods_atom_mapped[i].float(), -1, 1) @ P_expanded[i].transpose(dim0=1,dim1=2), 1, -1) for i in range(bs)]
    
        # ... do the same for X
        # The first selection drops out the extra features calculated in the preprocessing (leaving one-hot encodings), the second selection chooses the correct atom map numbers
        X_prods_atom_mapped = [orig_X[i,:,:out_dim_X][atom_map_numbers_prod_idxs[i]].unsqueeze(0) for i in range(bs)]
        # need to unsqueeze to do batched matrix multiplication correctly: (bs,N,N) @ (bs,N,1) -> (bs,N,1). (N is the count of atom mapped nodes)
        X_prods_am_permuted = [P_expanded[i] @ X_prods_atom_mapped[i] for i in range(bs)]
    
    elif alignment_type == 'correct':
        Ps = [math_utils.create_permutation_matrix_torch(atom_map_numbers_prod[i][atom_map_numbers_prod_idxs[i]],
                                            atom_map_numbers_rct[i][atom_map_numbers_rct_idxs[i]]).float().to(device)
                                            for i in range(bs)]
        P_expanded = [P.unsqueeze(0) for P in Ps] # The unsqueeze will be unnecessary with proper batching here
        # Permute the edges obtained from the product: P @ E @ P^T
        E_prods_am_permuted = [torch.movedim(P_expanded[i].transpose(dim0=1,dim1=2) @ torch.movedim(E_prods_atom_mapped[i].float(), -1, 1) @ P_expanded[i], 1, -1) for i in range(bs)]
    
        # ... do the same for X
        # The selection chooses the correct atom map numbers
        X_prods_atom_mapped = [orig_X[i,:,:out_dim_X][atom_map_numbers_prod_idxs[i]].unsqueeze(0) for i in range(bs)]
        # need to unsqueeze to do batched matrix multiplication correctly: (bs,N,N) @ (bs,N,1) -> (bs,N,1). (N is the count of atom mapped nodes)
        X_prods_am_permuted = [P_expanded[i].transpose(dim0=1,dim1=2) @ X_prods_atom_mapped[i] for i in range(bs)]
    else:
        assert False, f'Alignment type not set correctly ({alignment_type}). Should be old or correct'
    return X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct

class GraphTransformerWithYAtomMapPosEmb(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), cfg, pos_emb_permutations: int = 0,
                 improved=False, dropout=0.1, p_to_r_skip_connection=False, p_to_r_init=10.,
                 alignment_type='old', input_alignment=False):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.pos_emb_permutations = pos_emb_permutations
        self.p_to_r_skip_connection = p_to_r_skip_connection
        self.alignment_type = alignment_type
        self.input_alignment = input_alignment
        self.cfg = cfg
        if input_alignment:
            input_dims = copy.deepcopy(input_dims)
            original_data_feature_dim_X = output_dims['X']
            original_data_feature_dim_E = output_dims['E']
            input_dims['X'] += original_data_feature_dim_X # make the input feature dimensionality larger to include the aligned & concatenated product conditioning
            input_dims['E'] += original_data_feature_dim_E 

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
        
        if self.p_to_r_skip_connection:
            # for p in self.mlp_out_X[-1].parameters(): # zero out 
            #     p.detach().zero_()
            # for p in self.mlp_out_E[-1].parameters(): # zero out 
            #     p.detach().zero_()
            self.skip_scaling = nn.Parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_2 = nn.Parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_3 = nn.Parameter(torch.tensor([1.], dtype=torch.float))

    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask, atom_map_numbers, pos_encodings, mol_assignment):

        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_E = E.clone()
        orig_X = X.clone()

        # Potential edge-skip connection from product side to reactant side
        if self.input_alignment:
            # We assume that the atoms are ordered according to atom mapping number already
            fpi = data_utils.get_first_product_index(mol_assignment)
            npa = data_utils.get_num_product_atoms(mol_assignment)

            # Could do the following to vectorize, but would use more memory:
            # Create a mask where each row corresponds to a range defined by npa
            #mask = torch.zeros(bs, d).bool()
            #indices = torch.arange(d).expand(bs, -1)
            #starts = fpi.unsqueeze(1)
            #ends = (fpi + npa).unsqueeze(1)
            #mask = (indices >= starts) & (indices < ends)
            # Using the mask to select elements from X and place them in X_to_concatenate
            #X_to_concatenate[mask] = X[mask]

            X_to_concatenate = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
            E_to_concatenate_1 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            E_to_concatenate_2 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            for i in range(bs):
                X_to_concatenate[i, :npa[i]] = X[i, fpi[i]:fpi[i]+npa[i], :self.out_dim_X] # drop out the extra features
                E_to_concatenate_1[i, :npa[i], :npa[i]] = E[i, fpi[i]:fpi[i]+npa[i], fpi[i]:fpi[i]+npa[i]]
                E_to_concatenate_2[i, npa[i]:, :] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
                E_to_concatenate_2[i, :, npa[i]:] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()

            X = torch.cat([X, X_to_concatenate], dim=-1)
            E = torch.cat([E, E_to_concatenate_1 + E_to_concatenate_2], dim=-1)

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
        if pos_encodings.shape != X.shape:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(bs, n, X.shape[-1] - pos_encodings.shape[-1], device=device)], dim=-1)
        X = X + pos_encodings

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = y[..., :self.out_dim_y] # TODO: Changed
        #y = self.mlp_out_y(y) # TODO: Changed

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        # Potential edge-skip connection from product side to reactant side
        if self.p_to_r_skip_connection:            
            if not self.input_alignment: # if this stuff wasn't done already
                fpi = data_utils.get_first_product_index(mol_assignment)
                npa = data_utils.get_num_product_atoms(mol_assignment)
                X_to_concatenate = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
                E_to_concatenate_1 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
                E_to_concatenate_2 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
                for i in range(bs):
                    X[i, :npa[i]] += orig_X[i, fpi[i]:fpi[i]+npa[i]] * self.skip_scaling
                    E[i, :npa[i], :npa[i]] += orig_E[i, fpi[i]:fpi[i]+npa[i], fpi[i]:fpi[i]+npa[i]] * self.skip_scaling
                    if self.cfg.neuralnet.skip_connection_on_no_mapped_atoms:
                        E[i, npa[i]:, :] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()* self.skip_scaling_2
                        E[i, :, npa[i]:] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()* self.skip_scaling_2
            else: # reuse the previous calculations
                X += X_to_concatenate * self.skip_scaling
                E += E_to_concatenate_1 * self.skip_scaling
                E += E_to_concatenate_2 * self.skip_scaling_2

        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask
        # return graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask, atom_map_numbers=atom_map_numbers).mask(node_mask)

class GraphTransformerWithYAtomMapPosEmbInefficient(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), cfg, 
                 pos_emb_permutations: int = 0,
                 improved=False, dropout=0.1, p_to_r_skip_connection=False, p_to_r_init=10.,
                 alignment_type='old', input_alignment=False):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.pos_emb_permutations = pos_emb_permutations
        self.p_to_r_skip_connection = p_to_r_skip_connection
        self.alignment_type = alignment_type
        self.input_alignment = input_alignment
        self.cfg = cfg
        if input_alignment:
            input_dims = copy.deepcopy(input_dims)
            original_data_feature_dim_X = output_dims['X']
            original_data_feature_dim_E = output_dims['E']
            input_dims['X'] += original_data_feature_dim_X # make the input feature dimensionality larger to include the aligned & concatenated product conditioning
            input_dims['E'] += original_data_feature_dim_E 

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
        
        if self.p_to_r_skip_connection:
            # for p in self.mlp_out_X[-1].parameters(): # zero out 
            #     p.detach().zero_()
            # for p in self.mlp_out_E[-1].parameters(): # zero out 
            #     p.detach().zero_()
            self.skip_scaling = nn.Parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_2 = nn.Parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_3 = nn.Parameter(torch.tensor([1.], dtype=torch.float))

    def zero_initialize_specific_layers(self):
        # Define the layers to zero-initialize
        zero_init_layers = [
            'mlp_in_X.0', 'mlp_in_E.0', 'mlp_in_y.0',
            f'mlp_out_X.{len(self.mlp_out_X)-1}',
            f'mlp_out_E.{len(self.mlp_out_E)-1}',
            f'mlp_out_y.{len(self.mlp_out_y)-1}'
        ]
        
        # Iterate through named modules
        for name, module in self.named_modules():
            if name in zero_init_layers:
                if isinstance(module, nn.Linear):
                    # Zero-initialize weights and biases
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward_old(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask, atom_map_numbers, pos_encodings, mol_assignment):
        
        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_E = E.clone()
        orig_X = X.clone()

        # Potential edge-skip connection from product side to reactant side
        if self.input_alignment:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            X_prods_am_permuted, E_prods_am_permuted
            X_to_concatenate = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
            E_to_concatenate = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            for i in range(bs):
                # The following is used for choosing which parts to change in the output
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                E_to_concatenate[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                    E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                    E_prods_am_permuted[i].shape[3]).float()
                E_to_concatenate[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
            # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
            for i in range(bs):
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                X_to_concatenate[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float()
            X = torch.cat([X, X_to_concatenate], dim=-1)
            E = torch.cat([E, E_to_concatenate], dim=-1)

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
        X = X + pos_encodings

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
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            
            # E_prods_am_permuted = [F.one_hot(E_prods_am_permuted[i], self.out_dim_E) for i in range(bs)]
            if not self.input_alignment: # if this stuff wasn't done already
                for i in range(bs):
                    # The following is used for choosing which parts to change in the output
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    E[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                        E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                        E_prods_am_permuted[i].shape[3]).float() * self.skip_scaling
                    E[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float() * self.skip_scaling_2
                # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
                for i in range(bs):
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    X[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float() * self.skip_scaling
            else: # reuse the previous calculations
                X += X_to_concatenate * self.skip_scaling
                E += E_to_concatenate * self.skip_scaling_2
            # TODO: Could also add the dummy node output here to make things consistent, where do I get the dummy node index?

            # PROBLEM (solved): Input X and E dx and de are different than the output dimensions! Need to explicitly take this into account
            # ... can I use the orig_X and orig_E at all directly here if they contain the other features as well? Or is this only the case for X?
            # -> seems to be the case, good, but also a bit dangerous since this could just pass without errors if all values happen to be within the correct output dimensions

            # atom_mapped_prod_indices = atom_map_numbers

        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask

    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask, atom_map_numbers, pos_encodings, mol_assignment):
        #print(f'\tX.shape: {X.shape}, E.shape: {E.shape}, y.shape: {y.shape}')
        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_E = E.clone()
        orig_X = X.clone()

        # Potential edge-skip connection from product side to reactant side
        if self.input_alignment:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            X_prods_am_permuted, E_prods_am_permuted
            X_to_concatenate = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
            E_to_concatenate_1 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            E_to_concatenate_2 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            for i in range(bs):
                # The following is used for choosing which parts to change in the output
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                E_to_concatenate_1[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                    E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                    E_prods_am_permuted[i].shape[3]).float()
                if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms:
                    E_to_concatenate_2[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
            # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
            for i in range(bs):
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                X_to_concatenate[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float()
            X = torch.cat([X, X_to_concatenate], dim=-1)
            E = torch.cat([E, E_to_concatenate_1 + E_to_concatenate_2], dim=-1)

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
        if pos_encodings.shape != X.shape:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(bs, n, X.shape[-1] - pos_encodings.shape[-1], device=device)], dim=-1)
        X = X + pos_encodings
        
        # def log_memory_usage(step, location):
        #     memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        #     memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        #     print(f"========= Step {step}, {location}:")
        #     print(f"  Allocated: {memory_allocated:.2f} MB")
        #     print(f"  Reserved:  {memory_reserved:.2f} MB")
    
        for i, layer in enumerate(self.tf_layers):
            #log_memory_usage(f"in forward", f"before layer {i}")

            X, E, y = layer(X, E, y, node_mask)
            #log_memory_usage(f"in forward", f"after layer {i}")
            #torch.cuda.empty_cache()

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
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, 
                                                                                                             self.alignment_type, self.out_dim_E, self.out_dim_X, 
                                                                                                             device)
            
            # E_prods_am_permuted = [F.one_hot(E_prods_am_permuted[i], self.out_dim_E) for i in range(bs)]
            if not self.input_alignment: # if this stuff wasn't done already
                for i in range(bs):
                    # The following is used for choosing which parts to change in the output
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    E[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                        E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                        E_prods_am_permuted[i].shape[3]).float() * self.skip_scaling
                    if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms: # This puts zeros also on product side output, but we discard that anyways so it's fine
                        E[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float() * self.skip_scaling_2
                # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
                for i in range(bs):
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    X[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float() * self.skip_scaling
            else: # reuse the previous calculations
                X += X_to_concatenate * self.skip_scaling
                E += E_to_concatenate_1 * self.skip_scaling
                E += E_to_concatenate_2 * self.skip_scaling_2
            # TODO: Could also add the dummy node output here to make things consistent, where do I get the dummy node index?

            # PROBLEM (solved): Input X and E dx and de are different than the output dimensions! Need to explicitly take this into account
            # ... can I use the orig_X and orig_E at all directly here if they contain the other features as well? Or is this only the case for X?
            # -> seems to be the case, good, but also a bit dangerous since this could just pass without errors if all values happen to be within the correct output dimensions

            # atom_mapped_prod_indices = atom_map_numbers

        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask
        # return graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask, atom_map_numbers=atom_map_numbers).mask(node_mask)

class GraphTransformerWithYAtomMapPosEmbInefficientStereo(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), cfg, pos_emb_permutations: int = 0,
                 improved=False, dropout=0.1, p_to_r_skip_connection=False, p_to_r_init=10.,
                 alignment_type='old', input_alignment=False):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.out_dim_chiral = output_dims['atom_chiral']
        self.out_dim_bonddir = output_dims['bond_dirs']
        self.pos_emb_permutations = pos_emb_permutations
        self.p_to_r_skip_connection = p_to_r_skip_connection
        self.alignment_type = alignment_type
        self.input_alignment = input_alignment
        self.cfg = cfg
        if input_alignment:
            input_dims = copy.deepcopy(input_dims)
            original_data_feature_dim_X = output_dims['X']
            original_data_feature_dim_E = output_dims['E']
            input_dims['X'] += original_data_feature_dim_X # make the input feature dimensionality larger to include the aligned & concatenated product conditioning
            input_dims['E'] += original_data_feature_dim_E 
            input_dims['atom_chiral'] += output_dims['atom_chiral']
            input_dims['bond_dirs'] += output_dims['bond_dirs']

        self.pos_emb = SinusoidalPosEmb(dim=hidden_dims['dx'])

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'] + input_dims['atom_chiral'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'] + input_dims['bond_dirs'], hidden_mlp_dims['E']), act_fn_in,
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
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X'] + output_dims['atom_chiral']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E'] + output_dims['bond_dirs']))
        
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

    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask, atom_map_numbers, pos_encodings, mol_assignment):

        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_E = E.clone()
        orig_X = X.clone()
        orig_atom_chiral = atom_chiral.clone()
        orig_bond_dirs = bond_dirs.clone()

        # Potential edge-skip connection from product side to reactant side
        if self.input_alignment:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            atom_chiral_am_permuted, bond_dirs_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_atom_chiral, orig_bond_dirs, self.alignment_type, self.out_dim_bonddir, self.out_dim_chiral, device)
            # X_prods_am_permuted, E_prods_am_permuted
            X_to_concatenate = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
            E_to_concatenate_1 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            E_to_concatenate_2 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            atom_chiral_to_concatenate = torch.zeros(atom_chiral.shape[0], atom_chiral.shape[1], self.out_dim_chiral, device=device)
            bond_dirs_to_concatenate_1 = torch.zeros(bond_dirs.shape[0], bond_dirs.shape[1], self.out_dim_bonddir, device=device)
            bond_dirs_to_concatenate_2 = torch.zeros(bond_dirs.shape[0], bond_dirs.shape[1], self.out_dim_bonddir, device=device)
            for i in range(bs):
                # The following is used for choosing which parts to change in the output
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                E_to_concatenate_1[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                    E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                    E_prods_am_permuted[i].shape[3]).float()
                bond_dirs_to_concatenate_1[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += bond_dirs_am_permuted[i].reshape(
                                                                                                    bond_dirs_am_permuted[i].shape[1]*bond_dirs_am_permuted[i].shape[2],
                                                                                                    bond_dirs_am_permuted[i].shape[3]).float()
                if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms:
                    E_to_concatenate_2[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
                    bond_dirs_to_concatenate_2[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), bond_dirs.shape[-1]).float()
            # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
            for i in range(bs):
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                X_to_concatenate[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float()
                atom_chiral_to_concatenate[i, am_rct_selection] += atom_chiral_am_permuted[i].squeeze(0).float()
            X = torch.cat([X, X_to_concatenate], dim=-1)
            atom_chiral = torch.cat([atom_chiral, atom_chiral_to_concatenate], dim=-1)
            E = torch.cat([E, E_to_concatenate_1 + E_to_concatenate_2], dim=-1)
            bond_dirs = torch.cat([bond_dirs, bond_dirs_to_concatenate_1 + bond_dirs_to_concatenate_2], dim=-1)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]
        atom_chiral_to_out = atom_chiral[..., :self.out_dim_chiral]
        bond_dirs_to_out = bond_dirs[..., :self.out_dim_bonddir]

        X = torch.cat([X, atom_chiral], dim=-1)
        E = torch.cat([E, bond_dirs], dim=-1)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = graph.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Add the positional encoding to X. X shape is now (bs, n, dx)
        # TODO: Maybe concatenate instead so that this works with the Laplacian eigenvectors as well?
        if pos_encodings.shape != X.shape:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(bs, n, X.shape[-1] - pos_encodings.shape[-1], device=device)], dim=-1)
        X = X + pos_encodings

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = y[..., :self.out_dim_y] # TODO: Changed
        #y = self.mlp_out_y(y) # TODO: Changed

        # unpack
        X, atom_chiral = X[..., :self.out_dim_X], X[..., self.out_dim_X:]
        E, bond_dirs = E[..., :self.out_dim_E], E[..., self.out_dim_E:]

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        atom_chiral = atom_chiral + atom_chiral_to_out
        bond_dirs = (bond_dirs + bond_dirs_to_out) * diag_mask

        # E *= self.skip_scaling_3
        # X *= self.skip_scaling_3

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        # Potential edge-skip connection from product side to reactant side
        if self.p_to_r_skip_connection:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            atom_chiral_am_permuted, bond_dirs_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_atom_chiral, orig_bond_dirs, self.alignment_type, self.out_dim_bonddir, self.out_dim_chiral, device)

            # E_prods_am_permuted = [F.one_hot(E_prods_am_permuted[i], self.out_dim_E) for i in range(bs)]
            if not self.input_alignment: # if this stuff wasn't done already
                for i in range(bs):
                    # The following is used for choosing which parts to change in the output
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    E[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                        E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                        E_prods_am_permuted[i].shape[3]).float() * self.skip_scaling
                    bond_dirs[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += bond_dirs_am_permuted[i].reshape(
                                                                                                    bond_dirs_am_permuted[i].shape[1]*bond_dirs_am_permuted[i].shape[2],
                                                                                                    bond_dirs_am_permuted[i].shape[3]).float() * self.skip_scaling
                    if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms: # This puts zeros also on product side output, but we discard that anyways so it's fine
                        E[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float() * self.skip_scaling_2
                        bond_dirs[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), bond_dirs.shape[-1]).float() * self.skip_scaling_2
                # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
                for i in range(bs):
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    X[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float() * self.skip_scaling
                    atom_chiral[i, am_rct_selection] += atom_chiral_am_permuted[i].squeeze(0).float() * self.skip_scaling
            else: # reuse the previous calculations
                X += X_to_concatenate * self.skip_scaling
                E += E_to_concatenate_1 * self.skip_scaling
                E += E_to_concatenate_2 * self.skip_scaling_2
                bond_dirs += bond_dirs_to_concatenate_1 * self.skip_scaling
                bond_dirs += bond_dirs_to_concatenate_2 * self.skip_scaling_2
                atom_chiral += atom_chiral_to_concatenate * self.skip_scaling
            # TODO: Could also add the dummy node output here to make things consistent, where do I get the dummy node index?

            # PROBLEM (solved): Input X and E dx and de are different than the output dimensions! Need to explicitly take this into account
            # ... can I use the orig_X and orig_E at all directly here if they contain the other features as well? Or is this only the case for X?
            # -> seems to be the case, good, but also a bit dangerous since this could just pass without errors if all values happen to be within the correct output dimensions

            # atom_mapped_prod_indices = atom_map_numbers

        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask
        # return graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask, atom_map_numbers=atom_map_numbers).mask(node_mask)

class GraphTransformerWithYAtomMapPosEmbInefficientChargesSeparate(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), cfg, pos_emb_permutations: int = 0,
                 improved=False, dropout=0.1, p_to_r_skip_connection=False, p_to_r_init=10.,
                 alignment_type='old', input_alignment=False):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.out_dim_charges = output_dims['atom_charges']
        self.out_dim_bonddir = output_dims['bond_dirs']
        self.pos_emb_permutations = pos_emb_permutations
        self.p_to_r_skip_connection = p_to_r_skip_connection
        self.alignment_type = alignment_type
        self.input_alignment = input_alignment
        self.cfg = cfg
        if input_alignment:
            input_dims = copy.deepcopy(input_dims)
            original_data_feature_dim_X = output_dims['X']
            original_data_feature_dim_E = output_dims['E']
            input_dims['X'] += original_data_feature_dim_X # make the input feature dimensionality larger to include the aligned & concatenated product conditioning
            input_dims['E'] += original_data_feature_dim_E 
            input_dims['atom_charges'] += output_dims['atom_charges']

        self.pos_emb = SinusoidalPosEmb(dim=hidden_dims['dx'])

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'] + input_dims['atom_charges'], hidden_mlp_dims['X']), act_fn_in,
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
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X'] + output_dims['atom_charges']))

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

    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask, atom_map_numbers, pos_encodings, mol_assignment):

        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_E = E.clone()
        orig_X = X.clone()
        orig_atom_charges = atom_charges.clone()
        orig_bond_dirs = bond_dirs.clone()

        # Potential edge-skip connection from product side to reactant side
        if self.input_alignment:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            atom_charges_am_permuted, bond_dirs_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_atom_charges, orig_bond_dirs, self.alignment_type, self.out_dim_bonddir, self.out_dim_charges, device)
            # X_prods_am_permuted, E_prods_am_permuted
            X_to_concatenate = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
            E_to_concatenate_1 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            E_to_concatenate_2 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
            atom_charges_to_concatenate = torch.zeros(atom_charges.shape[0], atom_charges.shape[1], self.out_dim_charges, device=device)
            for i in range(bs):
                # The following is used for choosing which parts to change in the output
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                E_to_concatenate_1[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                    E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                    E_prods_am_permuted[i].shape[3]).float()
                if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms:
                    E_to_concatenate_2[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
            # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
            for i in range(bs):
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                X_to_concatenate[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float()
                atom_charges_to_concatenate[i, am_rct_selection] += atom_charges_am_permuted[i].squeeze(0).float()
            X = torch.cat([X, X_to_concatenate], dim=-1)
            atom_charges = torch.cat([atom_charges, atom_charges_to_concatenate], dim=-1)
            E = torch.cat([E, E_to_concatenate_1 + E_to_concatenate_2], dim=-1)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]
        atom_charges_to_out = atom_charges[..., :self.out_dim_charges]

        X = torch.cat([X, atom_charges], dim=-1)
        # E = torch.cat([E], dim=-1)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = graph.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Add the positional encoding to X. X shape is now (bs, n, dx)
        # TODO: Maybe concatenate instead so that this works with the Laplacian eigenvectors as well?
        if pos_encodings.shape != X.shape:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(bs, n, X.shape[-1] - pos_encodings.shape[-1], device=device)], dim=-1)
        X = X + pos_encodings

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = y[..., :self.out_dim_y] # TODO: Changed
        #y = self.mlp_out_y(y) # TODO: Changed

        # unpack
        X, atom_charges = X[..., :self.out_dim_X], X[..., self.out_dim_X:]
        # E, bond_dirs = E[..., :self.out_dim_E]

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        atom_charges = atom_charges + atom_charges_to_out

        # E *= self.skip_scaling_3
        # X *= self.skip_scaling_3

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        # Potential edge-skip connection from product side to reactant side
        if self.p_to_r_skip_connection:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            atom_charges_am_permuted, bond_dirs_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, atom_map_numbers, orig_atom_charges, orig_bond_dirs, self.alignment_type, self.out_dim_bonddir, self.out_dim_charges, device)

            # E_prods_am_permuted = [F.one_hot(E_prods_am_permuted[i], self.out_dim_E) for i in range(bs)]
            if not self.input_alignment: # if this stuff wasn't done already
                for i in range(bs):
                    # The following is used for choosing which parts to change in the output
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    E[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                        E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                        E_prods_am_permuted[i].shape[3]).float() * self.skip_scaling
                    # NOTE: code below is failing because of inplace operation in different grad contexts
                    # bond_dirs[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += bond_dirs_am_permuted[i].reshape(
                    #                                                                                 bond_dirs_am_permuted[i].shape[1]*bond_dirs_am_permuted[i].shape[2],
                    #                                                                                 bond_dirs_am_permuted[i].shape[3]).float() * self.skip_scaling
                    
                    # print(f"bond_dirs shape: {bond_dirs.shape}, dtype: {bond_dirs.dtype}, device: {bond_dirs.device}, requires_grad: {bond_dirs.requires_grad}")
                    # print(f"am_rct_selection shape: {am_rct_selection.shape}, dtype: {am_rct_selection.dtype}, device: {am_rct_selection.device}, requires_grad: {am_rct_selection.requires_grad}")
                    # print(f"torch.is_grad_enabled(): {torch.is_grad_enabled()}")
                    # print(f"bond_dirs has a base tensor: {hasattr(bond_dirs, '_base')}")
                    # print(f"===== bond_dirs {bond_dirs}")
                    # print(f"bond_dirs_am_permuted {bond_dirs_am_permuted}")
                    # print(f"am_rct_selection {am_rct_selection}")
                    bond_dirs = bond_dirs.clone()
                    mask = am_rct_selection[:,None] * am_rct_selection[None,:]
                    selected_bonds = bond_dirs[i, mask].clone()

                    reshaped_permuted = bond_dirs_am_permuted[i].reshape(
                        bond_dirs_am_permuted[i].shape[1] * bond_dirs_am_permuted[i].shape[2],
                        bond_dirs_am_permuted[i].shape[3]
                    ).float()

                    bond_dirs[i, mask] = selected_bonds + (reshaped_permuted * self.skip_scaling)

                    # bond_dirs[i, am_rct_selection[:,None] * am_rct_selection[None,:]] = (bond_dirs[i, am_rct_selection[:,None] * am_rct_selection[None,:]]+\
                    #                                                                     bond_dirs_am_permuted[i].reshape(
                    #                                                                                 bond_dirs_am_permuted[i].shape[1]*bond_dirs_am_permuted[i].shape[2],
                    #                                                                                 bond_dirs_am_permuted[i].shape[3]).float() * self.skip_scaling)
                                        
                    if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms: # This puts zeros also on product side output, but we discard that anyways so it's fine
                        E[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float() * self.skip_scaling_2
                        #bond_dirs[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), bond_dirs.shape[-1]).float() * self.skip_scaling_2
                        bond_dirs[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] = (bond_dirs[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])]+\
                                                                                             F.one_hot(torch.tensor([0], dtype=torch.long, device=device), bond_dirs.shape[-1]).float() * self.skip_scaling_2)
                # X_prods_am_permuted = [F.one_hot(X_prods_am_permuted[i], self.out_dim_X) for i in range(bs)] # Shape (bs, N, dx)
                for i in range(bs):
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    X[i, am_rct_selection] += X_prods_am_permuted[i].squeeze(0).float() * self.skip_scaling
                    atom_charges[i, am_rct_selection] += atom_charges_am_permuted[i].squeeze(0).float() * self.skip_scaling
            else: # reuse the previous calculations
                X += X_to_concatenate * self.skip_scaling
                E += E_to_concatenate_1 * self.skip_scaling
                E += E_to_concatenate_2 * self.skip_scaling_2
                atom_charges += atom_charges_to_concatenate * self.skip_scaling
            # TODO: Could also add the dummy node output here to make things consistent, where do I get the dummy node index?

            # PROBLEM (solved): Input X and E dx and de are different than the output dimensions! Need to explicitly take this into account
            # ... can I use the orig_X and orig_E at all directly here if they contain the other features as well? Or is this only the case for X?
            # -> seems to be the case, good, but also a bit dangerous since this could just pass without errors if all values happen to be within the correct output dimensions

            # atom_mapped_prod_indices = atom_map_numbers

        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask
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