import sys
sys.path.append('..')

import math
import logging
log = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from diffalign.neuralnet.layers import XEyTransformerLayer

from diffalign.utils import graph, data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphTransformerForAllFeatures(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), cfg,
                 improved=False, dropout=0.1, p_to_r_skip_connection=False, p_to_r_init=10., input_alignment=False):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.out_dim_charge = output_dims['atom_charges']
        self.out_dim_chiral = output_dims['atom_chiral']
        self.out_dim_bonddir = output_dims['bond_dirs']
        self.p_to_r_skip_connection = p_to_r_skip_connection
        self.input_alignment = input_alignment
        self.cfg = cfg
        if input_alignment:
            input_dims = copy.deepcopy(input_dims)
            input_dims['X'] += output_dims['X'] # make the input feature dimensionality larger to include the aligned & concatenated product conditioning
            input_dims['E'] += output_dims['E']
            input_dims['atom_charges'] += output_dims['atom_charges']
            input_dims['atom_chiral'] += output_dims['atom_chiral']
            input_dims['bond_dirs'] += output_dims['bond_dirs']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'] + input_dims['atom_charges'] + input_dims['atom_chiral'], hidden_mlp_dims['X']), act_fn_in,
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
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X'] + output_dims['atom_charges'] + output_dims['atom_chiral']))

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

    def align_from_product_to_reactant(self, X, E, atom_charges, atom_chiral, bond_dirs, mol_assignment):
        bs = X.shape[0]
        fpi = data_utils.get_first_product_index(mol_assignment)
        npa = data_utils.get_num_product_atoms(mol_assignment)
        X_aligned_to_reactant = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
        X_aligned_to_reactant_2 = torch.zeros(X.shape[0], X.shape[1], self.out_dim_X, device=device)
        E_aligned_to_reactant_1 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
        E_aligned_to_reactant_2 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.out_dim_E, device=device)
        atom_charges_aligned_to_reactant = torch.zeros(atom_charges.shape[0], atom_charges.shape[1], self.out_dim_charge, device=device)
        atom_chiral_aligned_to_reactant = torch.zeros(atom_chiral.shape[0], atom_chiral.shape[1], self.out_dim_chiral, device=device)
        bond_dirs_aligned_to_reactant_1 = torch.zeros(bond_dirs.shape[0], bond_dirs.shape[1], bond_dirs.shape[2], self.out_dim_bonddir, device=device)
        bond_dirs_aligned_to_reactant_2 = torch.zeros(bond_dirs.shape[0], bond_dirs.shape[1], bond_dirs.shape[2], self.out_dim_bonddir, device=device)
        dummy_node_idx = self.cfg.dataset.atom_types.index('U')
        for i in range(bs):
            X_aligned_to_reactant[i, :npa[i]] = X[i, fpi[i]:fpi[i]+npa[i], :self.out_dim_X]
            X_aligned_to_reactant_2[i, npa[i]:] = F.one_hot(torch.tensor([dummy_node_idx], dtype=torch.long, device=device), self.out_dim_X).float()
            E_aligned_to_reactant_1[i, :npa[i], :npa[i]] = E[i, fpi[i]:fpi[i]+npa[i], fpi[i]:fpi[i]+npa[i]]
            E_aligned_to_reactant_2[i, npa[i]:, :] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
            E_aligned_to_reactant_2[i, :, npa[i]:] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
            atom_charges_aligned_to_reactant[i, :npa[i]] = atom_charges[i, fpi[i]:fpi[i]+npa[i]]
            atom_chiral_aligned_to_reactant[i, :npa[i]] = atom_chiral[i, fpi[i]:fpi[i]+npa[i]]
            bond_dirs_aligned_to_reactant_1[i, :npa[i], :npa[i]] = bond_dirs[i, fpi[i]:fpi[i]+npa[i], fpi[i]:fpi[i]+npa[i]]
            bond_dirs_aligned_to_reactant_2[i, npa[i]:, :] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), bond_dirs.shape[-1]).float()
            bond_dirs_aligned_to_reactant_2[i, :, npa[i]:] = F.one_hot(torch.tensor([0], dtype=torch.long, device=device), bond_dirs.shape[-1]).float()
        return X_aligned_to_reactant, X_aligned_to_reactant_2, E_aligned_to_reactant_1, E_aligned_to_reactant_2, atom_charges_aligned_to_reactant, \
                atom_chiral_aligned_to_reactant, bond_dirs_aligned_to_reactant_1, bond_dirs_aligned_to_reactant_2

    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask, atom_map_numbers, pos_encodings, mol_assignment):

        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_X, orig_E, orig_atom_charges, orig_atom_chiral, orig_bond_dirs = X.clone(), E.clone(), atom_charges.clone(), atom_chiral.clone(), bond_dirs.clone()

        # Potential input alignment from product to reactant
        if self.input_alignment:
            X_align_1, X_align_2, E_align_1, E_align_2, atom_charges_align, atom_chiral_align, bond_dir_align_1, bond_dir_align_2 = self.align_from_product_to_reactant(X, E, atom_charges, atom_chiral, bond_dirs, mol_assignment)
            X = torch.cat([X, X_align_1 + X_align_2 * self.cfg.neuralnet.skip_connection_on_non_mapped_atoms_nodes], dim=-1)
            E = torch.cat([E, E_align_1 + E_align_2], dim=-1)
            atom_charges = torch.cat([atom_charges, atom_charges_align], dim=-1)
            atom_chiral = torch.cat([atom_chiral, atom_chiral_align], dim=-1)
            bond_dirs = torch.cat([bond_dirs, bond_dir_align_1 + bond_dir_align_2], dim=-1)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]
        atom_charges_to_out = atom_charges[..., :self.out_dim_charge]
        atom_chiral_to_out = atom_chiral[..., :self.out_dim_chiral]
        bond_dirs_to_out = bond_dirs[..., :self.out_dim_bonddir]

        new_E = self.mlp_in_E(torch.cat([E, bond_dirs], dim=-1))
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = graph.PlaceHolder(X=self.mlp_in_X(torch.cat([X, atom_charges, atom_chiral],-1)), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Add the positional encoding to X. X shape is now (bs, n, dx)
        # TODO: Maybe concatenate instead so that this works with the Laplacian eigenvectors as well?
        # NOTE: this does not work with all sizes of NN layers (e.g. X.shape[-1] < pos_encodings.shape[-1] )
        if pos_encodings.shape != X.shape:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(bs, n, X.shape[-1] - pos_encodings.shape[-1], device=device)], dim=-1)
        X = X + pos_encodings

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        X, atom_charges, atom_chiral = X[..., :self.out_dim_X], X[..., self.out_dim_X:self.out_dim_X+self.out_dim_charge], X[..., self.out_dim_X+self.out_dim_charge:]
        E, bond_dirs = E[..., :self.out_dim_E], E[..., self.out_dim_E:]
        y = y[..., :self.out_dim_y] # TODO: Changed
        #y = self.mlp_out_y(y) # TODO: Changed

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        atom_charges = atom_charges + atom_charges_to_out
        atom_chiral = atom_chiral + atom_chiral_to_out
        bond_dirs = (bond_dirs + bond_dirs_to_out) * diag_mask

        E = 1/2 * (E + torch.transpose(E, 1, 2))
        bond_dirs = 1/2 * (bond_dirs + torch.transpose(bond_dirs, 1, 2))

        # Potential edge-skip connection from product side to reactant side
        if self.p_to_r_skip_connection:            
            if not self.input_alignment: # if this stuff wasn't done already                
                X_align_1, X_align_2, E_align_1, E_align_2, atom_charges_align, atom_chiral_align, bond_dir_align_1, bond_dir_align_2 = self.align_from_product_to_reactant(orig_X, orig_E, orig_atom_charges, orig_atom_chiral, orig_bond_dirs, mol_assignment)
            X += X_align_1 * self.skip_scaling
            X += X_align_2 * self.skip_scaling_2 * self.cfg.neuralnet.skip_connection_on_non_mapped_atoms_nodes
            E += E_align_1 * self.skip_scaling
            E += E_align_2 * self.skip_scaling_2
            atom_charges += atom_charges_align * self.skip_scaling
            atom_chiral += atom_chiral_align * self.skip_scaling
            bond_dirs += bond_dir_align_1 * self.skip_scaling
            bond_dirs += bond_dir_align_2 * self.skip_scaling_2

        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask