import copy
import math
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from diffalign.utils import assert_correctly_masked, device, DotDict    

class NodeEdgeBlockImproved(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(de, de)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        assert_correctly_masked(Q, x_mask)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)
        newE = Y.flatten(start_dim=3)
        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        assert_correctly_masked(newE, e_mask1 * e_mask2)

        Y = Y.sum(dim=-1)   # (bs, n, n, n_head) # calculate the edge-altered 'dot product'
        # Compute attentions. attn is now (bs, n, n, n_head)
        attn = F.softmax(Y, dim=2)

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values as A.dot(V)
        weighted_V = attn.unsqueeze(-1) * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx
        # newX = weighted_V
        # # Output X
        # newX = self.x_out(newX) * x_mask
        # helpers.assert_correctly_masked(newX, x_mask)

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        assert_correctly_masked(newX, x_mask)

        ## Process y based on X and E
        new_y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy
        # new_y = y 

        return newX, newE, new_y

class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1) if X.shape[1] > 1 else torch.zeros(X.shape[0], X.shape[2]).to(device)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2)) if E.shape[1] > 1 else torch.zeros(E.shape[0], E.shape[3]).to(device)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(de, de)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        assert_correctly_masked(Q, x_mask)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                   # (bs, n, 1, n head, df)                              
        K = K.unsqueeze(1)                  # (bs, 1, n, n_head, df)                            

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E

        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
       # print(f'ye1.shape: {ye1.shape}, ye2.shape: {ye2.shape}, newE.shape: {newE.shape}')
        newE = ye1 + (ye2 + 1) * newE
        #newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        attn = F.softmax(Y, dim=2)

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        assert_correctly_masked(newX, x_mask)

        ## Process y based on X and E
        new_y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy
        # new_y = y 

        return newX, newE, new_y
    
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
    
class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, improved=False) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw) if not improved else NodeEdgeBlockImproved(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)
    
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)
        
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y

class GraphTransformer(nn.Module):
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
        #self.pos_emb_permutations = pos_emb_permutations
        self.p_to_r_skip_connection = p_to_r_skip_connection
        self.alignment_type = alignment_type
        self.input_alignment = input_alignment
        # self.cfg = cfg
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

    def forward(self, X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask, atom_map_numbers, pos_encodings, mol_assignment):
        #print(f'\tX.shape: {X.shape}, E.shape: {E.shape}, y.shape: {y.shape}')
        #assert atom_map_numbers is not None
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
                # if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms:
                #     E_to_concatenate_2[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float()
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
        after_in = DotDict({'X': self.mlp_in_X(X), 'E': new_E, 'y': self.mlp_in_y(y)})
        #after_in = graph.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Add the positional encoding to X. X shape is now (bs, n, dx)
        # TODO: Maybe concatenate instead so that this works with the Laplacian eigenvectors as well?
        if pos_encodings is not None and pos_encodings.shape != X.shape:
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
                    # if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms: # This puts zeros also on product side output, but we discard that anyways so it's fine
                    #     E[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), E.shape[-1]).float() * self.skip_scaling_2
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