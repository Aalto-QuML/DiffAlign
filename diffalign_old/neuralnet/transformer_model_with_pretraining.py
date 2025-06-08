import torch
import torch.nn as nn
import torch.nn.functional as F
from diffalign_old.utils.graph import PlaceHolder
from diffalign_old.utils import graph
from diffalign_old.neuralnet.transformer_model_with_y import get_X_prods_and_E_prods_aligned

class AtomEncoder(nn.Module):
    def __init__(self, input_dims, hidden_mlp_dims, out_dim):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )
        self.net = nn.Sequential(nn.Linear(input_dims, hidden_mlp_dims),
                                 nn.ReLU(),
                                 nn.Linear(hidden_mlp_dims, out_dim),
                                 nn.ReLU())
        
    def forward(self, x):
        '''
        x: (batch_size, num_atoms, input_dim)
        '''
        return self.net(x)
    
class GraphTransformerWithPretraining(nn.Module):
    '''
    A pretrained diffusion model that uses a new encoder for the nodes.
    This is meant to bypass the MLP_in module of the pretrained model, 
    replacing it with a new encoder of potentially different input dimension.
    '''
    def __init__(self, 
                 pretrained_model, 
                 new_encoder, 
                 pretrained_model_out_dim_X,
                 output_hidden_dim_X,
                 output_dim_X):
        super().__init__()
        self.encoder = new_encoder
        # Skip MLP, get intermediate layers
        self.pretrained_model = pretrained_model
        self.output_layers = nn.Sequential(
            nn.Linear(pretrained_model_out_dim_X, output_hidden_dim_X),
            nn.ReLU(),
            nn.Linear(output_hidden_dim_X, output_dim_X)
        )   
    
    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask,
                                atom_map_numbers, pos_encodings, mol_assignment):
        '''
            If this works, one better way to do it is to call the forward of the pretrained model with the new encoder.
            requires changing the pretraine model's forward
        '''
        embeddings = self.encoder(X)
        X, E, y, atom_charges, atom_chiral, \
            bond_dirs, node_mask = self.pretrained_model(embeddings, E, y, 
                                                            atom_charges, 
                                                            atom_chiral, 
                                                            bond_dirs, 
                                                            node_mask,
                                                            atom_map_numbers, 
                                                            pos_encodings, 
                                                            mol_assignment)
        X = self.output_layers(X)
        return X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask
    
    def forward_pretrained_model(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask,
                                atom_map_numbers, pos_encodings, mol_assignment):
        assert atom_map_numbers is not None
        bs, n = X.shape[0], X.shape[1]
        device = X.device

        orig_E = E.clone()
        orig_X = X.clone()

        # Potential edge-skip connection from product side to reactant side
        if self.pretrained_model.input_alignment:
            X_prods_am_permuted, E_prods_am_permuted, atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment, 
                                                                                                             atom_map_numbers, 
                                                                                                             orig_X, orig_E, 
                                                                                                             self.pretrained_model.alignment_type, 
                                                                                                             self.pretrained_model.out_dim_E, 
                                                                                                             self.pretrained_model.out_dim_X, 
                                                                                                             device)
            X_to_concatenate = torch.zeros(X.shape[0], X.shape[1], self.pretrained_model.out_dim_X, device=device)
            E_to_concatenate_1 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.pretrained_model.out_dim_E, device=device)
            E_to_concatenate_2 = torch.zeros(E.shape[0], E.shape[1], E.shape[2], self.pretrained_model.out_dim_E, device=device)
            for i in range(bs):
                # The following is used for choosing which parts to change in the output
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                E_to_concatenate_1[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                    E_prods_am_permuted[i].shape[1]*E_prods_am_permuted[i].shape[2],
                                                                                                    E_prods_am_permuted[i].shape[3]).float()
                if self.pretrained_model.cfg.neuralnet.skip_connection_on_non_mapped_atoms:
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

        X_to_out = X[..., :self.pretrained_model.out_dim_X]
        E_to_out = E[..., :self.pretrained_model.out_dim_E]
        y_to_out = y[..., :self.pretrained_model.out_dim_y]

        new_E = self.pretrained_model.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = graph.PlaceHolder(X=self.encoder(X), E=new_E, y=self.pretrained_model.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Add the positional encoding to X. X shape is now (bs, n, dx)
        # TODO: Maybe concatenate instead so that this works with the Laplacian eigenvectors as well?
        if pos_encodings.shape != X.shape:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(bs, n, X.shape[-1] - pos_encodings.shape[-1], device=device)], dim=-1)
        X = X + pos_encodings
    
        for i, layer in enumerate(self.tf_layers):
            X, E, y = layer(X, E, y, node_mask)

        X = self.pretrained_model.mlp_out_X(X)
        E = self.pretrained_model.mlp_out_E(E)
        y = y[..., :self.pretrained_model.out_dim_y]

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        # Potential edge-skip connection from product side to reactant side
        if self.p_to_r_skip_connection:
            X_prods_am_permuted, E_prods_am_permuted, \
                atom_map_numbers_rct = get_X_prods_and_E_prods_aligned(mol_assignment,
                                                                       atom_map_numbers, 
                                                                         orig_X, 
                                                                         orig_E, 
                                                                         self.pretrained_model.alignment_type, 
                                                                         self.pretrained_model.out_dim_E, 
                                                                         self.pretrained_model.out_dim_X, 
                                                                         device)
            if not self.pretrained_model.input_alignment: # if this stuff wasn't done already
                for i in range(bs):
                    # The following is used for choosing which parts to change in the output
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    E[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += E_prods_am_permuted[i].reshape(
                                                                                                        E_prods_am_permuted[i].shape[1]*\
                                                                                                            E_prods_am_permuted[i].shape[2],
                                                                                                        E_prods_am_permuted[i].shape[3])\
                                                                                                            .float()*\
                                                                                                        self.skip_scaling
                    if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms: # This puts zeros also on product side output, but we discard that anyways so it's fine
                        E[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], 
                                                                                                             dtype=torch.long, 
                                                                                                             device=device), E.shape[-1]).float() * self.skip_scaling_2
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

def freeze_except_mlp(model):
    # Option 1: Freeze by parameter name
    for name, param in model.named_parameters():
        if not name.startswith('mlp'):  # Adjust pattern as needed
            param.requires_grad = False