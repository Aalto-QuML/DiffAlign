import hydra
import torch.nn as nn
import torch
from diffalign.graph_transformer_old import GraphTransformer

@hydra.main(config_path='../configs', config_name='default.yaml')
def train(cfg):
    denoiser = GraphTransformer(
            n_layers=cfg.model.n_layers, 
            input_dims=cfg.model.input_dims, 
            hidden_mlp_dims=cfg.model.hidden_mlp_dims, 
            hidden_dims=cfg.model.hidden_dims,
            output_dims=cfg.model.output_dims, 
            act_fn_in=nn.ReLU(), 
            act_fn_out=nn.ReLU(), 
            cfg=cfg, 
            pos_emb_permutations=0,
            improved=False, 
            dropout=0.1, 
            p_to_r_skip_connection=False, 
            p_to_r_init=10.,
            alignment_type='old', 
            input_alignment=False
    )
    X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask = denoiser(
        X=torch.randn(10, 10).unsqueeze(0), 
        E=torch.randn(10, 10, 10).unsqueeze(0), 
        y=torch.randn(10).unsqueeze(0), 
        atom_charges=torch.randn(10, 10).unsqueeze(0), 
        atom_chiral=torch.randn(10, 10).unsqueeze(0), 
        bond_dirs=torch.randn(10, 10, 10).unsqueeze(0), 
        node_mask=torch.randint(0, 2, (1, 10)).bool(),
        atom_map_numbers=torch.randn(10).unsqueeze(0), 
        pos_encodings=torch.randn(10, 10).unsqueeze(0), 
        mol_assignment=torch.randn(10).unsqueeze(0)
    )
    print(X.shape, E.shape, y.shape, atom_charges.shape, atom_chiral.shape, bond_dirs.shape, node_mask.shape)

if __name__ == '__main__':
    train()