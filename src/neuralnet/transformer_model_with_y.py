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
import copy
from src.neuralnet.layers import XEyTransformerLayer

from src.utils import graph, setup
from src.utils.diffusion import helpers
from src.neuralnet.layers import Xtoy, Etoy

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

    def forward(self, X, E, y, node_mask):

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

        return X, E, y, node_mask #TODO: Changed, and also the usage in model.forward()
        # return graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask).mask(node_mask)

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


def get_X_prods_and_E_prods_aligned(mol_assignments, atom_map_numbers, orig_X, orig_E, alignment_type, out_dim_E, out_dim_X, device):
    bs = orig_X.shape[0]
    prod_assignment = mol_assignments.max(-1).values
    atom_map_numbers_prod, atom_map_numbers_rct = atom_map_numbers.clone(), atom_map_numbers.clone()
    atom_map_numbers_prod[mol_assignments < prod_assignment[:,None]] = 0
    atom_map_numbers_rct[mol_assignments == prod_assignment[:,None]] = 0
    # The next picks out the relevant indices, they are of different lengths for different elements in the batch
    # -> need to change into lists, I guess
    atom_map_numbers_prod_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_prod[i]>0] for i in range(bs)]
    atom_map_numbers_rct_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_rct[i]>0] for i in range(bs)]
    # atom_map_numbers_prod_idx = torch.arange(atom_map_numbers.shape[-1])[atom_map_numbers_prod > 0]
    # atom_map_numbers_rct_idx = torch.arange(atom_map_numbers.shape[-1])[atom_map_numbers_rct.squeeze(0) > 0]
    # The first selection drops out potential extra features calculated in the preprocessing (leaving one-hot encodings, 
    # although we don't have extra features in practice for E), the second selneuralnet.archiection chooses the correct atom map numbers
    E_prods_atom_mapped = [
        orig_E[:,:,:,:out_dim_E][i,atom_map_numbers_prod_idxs[i]][:, atom_map_numbers_prod_idxs[i]].unsqueeze(0)
        for i in range(bs)]
    assert all(len(atom_map_numbers_prod_idxs[i]) == len(atom_map_numbers_rct_idxs[i]) for i in range(bs))
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
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), pos_emb_permutations: int = 0,
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

    def forward(self, X, E, y, node_mask, atom_map_numbers, pos_encodings, mol_assignments):

        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_E = E.clone()
        orig_X = X.clone()

        # Potential edge-skip connection from product side to reactant side
        if self.input_alignment:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignments, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
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
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignments, atom_map_numbers, orig_X, orig_E, self.alignment_type, self.out_dim_E, self.out_dim_X, device)
            
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

        return X, E, y, node_mask
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