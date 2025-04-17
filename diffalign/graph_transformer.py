'''
    Graph Transformer model.
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffalignX.layers.film import FiLM
from diffalignX.sinusoidal_positional_embedding import SinusoidalPosEmb
from diffalignX.graph_data_structure import Graph

def create_permutation_matrix_torch(original, permuted):
    """
    Create a permutation matrix in pyTorch that represents the permutation required
    to transform the 'original' list into the 'permuted' list.

    The function assumes that both input lists contain distinct elements and have
    the same length. It uses pyTorch operations to efficiently create the matrix
    without explicit python loops.

    parameters:
    original (list of int): The original list of distinct integers.
    permuted (list of int): The permuted list of the same distinct integers found in 'original'.

    Returns:
    torch.Tensor: A 2D tensor (matrix) of shape (N, N) where N is the length of the input lists.
                  The matrix is a binary (0s and 1s) permutation matrix, where each row and
                  column has exactly one entry of 1, indicating the mapping from 'original' to
                  'permuted' list.

    edgesxample:
    >>> original_list = [3, 1, 4, 2]
    >>> permuted_list = [1, 4, 3, 2]
    >>> matrix = create_permutation_matrix_torch(original_list, permuted_list)
    >>> print(matrix)
    tensor([[0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]], dtype=torch.int32)
    """
    size = len(original)
    # Convert lists to tensors, in case not already
    original_tensor = torch.tensor(original)
    permuted_tensor = torch.tensor(permuted)
    
    # Create index tensors
    original_indices = torch.argsort(original_tensor)
    permuted_indices = torch.argsort(permuted_tensor)
    
    # Create permutation matrix
    permutation_matrix = torch.zeros((size, size), dtype=torch.int32)
    permutation_matrix[original_indices, permuted_indices] = 1
    
    return permutation_matrix

def rowwise_atom_mapping_mask(atom_map_numbers_rct, atom_map_numbers_prod):
    """
    For each row in atom_map_numbers_rct, zero out values that don't appear 
    in the corresponding row of atom_map_numbers_prod (excluding zeros)
    
    parameters:
    atom_map_numbers_rct: torch.Tensor of shape (batch_size, seq_len_rct)
    atom_map_numbers_prod: torch.Tensor of shape (batch_size, seq_len_prod)
    
    Returns:
    torch.Tensor of same shape as atom_map_numbers_rct with masked values
    """
    result = atom_map_numbers_rct.clone()
    for atom_map_numbers_rct_i, atom_map_numbers_prod_i \
        in zip(atom_map_numbers_rct, atom_map_numbers_prod):
        valid_numbers = atom_map_numbers_prod_i[atom_map_numbers_prod_i != 0]
        mask = torch.isin(atom_map_numbers_rct_i, valid_numbers)
        result[atom_map_numbers_rct_i][~mask] = 0
    return result

def get_nodes_prods_and_edges_prods_aligned(mol_assignment, atom_map_numbers, orig_nodes, orig_edges, 
                                            alignment_type, out_dim_edges, out_dim_nodes, device):
    bs = orig_nodes.shape[0]
    prod_assignment = mol_assignment.max(-1).values
    atom_map_numbers_prod, atom_map_numbers_rct = atom_map_numbers.clone(), atom_map_numbers.clone()
    # NOTedges: only keep the atom_map_numbers_rct
    # get the matching am by counting the occurences of each number
    atom_map_numbers_prod[mol_assignment < prod_assignment[:,None]] = 0
    atom_map_numbers_rct[mol_assignment == prod_assignment[:,None]] = 0
    atom_map_numbers_rct = rowwise_atom_mapping_mask(atom_map_numbers_rct, atom_map_numbers_prod)
    if not (atom_map_numbers_prod.sort()[0]==atom_map_numbers_rct.sort()[0]).all():
        print(f'atom_map_numbers_prod.shape: {atom_map_numbers_prod.shape}')
        for i in range(atom_map_numbers_prod.shape[0]):
            print(f'atom_map_numbers_prod: {atom_map_numbers_prod[i].sort()[0]}')
            print(f'atom_map_numbers_rct: {atom_map_numbers_rct[i].sort()[0]}\n')
        
        assert False, f"The atom map numbers should be the same for products and reactants over the batch."

    atom_map_numbers_prod_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_prod[i]>0] for i in range(bs)]
    atom_map_numbers_rct_idxs = [torch.arange(atom_map_numbers.shape[-1], device=device)[atom_map_numbers_rct[i]>0] for i in range(bs)]
    
    edges_prods_atom_mapped = [
        orig_edges[:,:,:,:out_dim_edges][i,atom_map_numbers_prod_idxs[i]][:, atom_map_numbers_prod_idxs[i]].unsqueeze(0)
        for i in range(bs)]
    
    if alignment_type == 'old':
        ps = [create_permutation_matrix_torch(atom_map_numbers_prod_idxs[i] - atom_map_numbers_prod_idxs[i].min(),
                                                    atom_map_numbers_rct_idxs[i] - atom_map_numbers_rct_idxs[i].min()).float().to(device)
                                                    for i in range(bs)]
        p_expanded = [p.unsqueeze(0) for p in ps] # The unsqueeze will be unnecessary with proper batching here
        # permute the edges obtained from the product: p @ edges @ p^T
        edges_prods_am_permuted = [torch.movedim(p_expanded[i] @ torch.movedim(edges_prods_atom_mapped[i].float(), -1, 1) @ p_expanded[i].transpose(dim0=1,dim1=2), 1, -1) for i in range(bs)]
    
        # ... do the same for nodes
        # The first selection drops out the y features calculated in the preprocessing (leaving one-hot encodings), the second selection chooses the correct atom map numbers
        nodes_prods_atom_mapped = [orig_nodes[i,:,:out_dim_nodes][atom_map_numbers_prod_idxs[i]].unsqueeze(0) for i in range(bs)]
        # need to unsqueeze to do batched matrix multiplication correctly: (bs,N,N) @ (bs,N,1) -> (bs,N,1). (N is the count of atom mapped nodes)
        nodes_prods_am_permuted = [p_expanded[i] @ nodes_prods_atom_mapped[i] for i in range(bs)]
    
    elif alignment_type == 'correct':
        ps = [create_permutation_matrix_torch(atom_map_numbers_prod[i][atom_map_numbers_prod_idxs[i]],
                                            atom_map_numbers_rct[i][atom_map_numbers_rct_idxs[i]]).float().to(device)
                                            for i in range(bs)]
        p_expanded = [p.unsqueeze(0) for p in ps] # The unsqueeze will be unnecessary with proper batching here
        # permute the edges obtained from the product: p @ edges @ p^T
        edges_prods_am_permuted = [torch.movedim(p_expanded[i].transpose(dim0=1,dim1=2) @ torch.movedim(edges_prods_atom_mapped[i].float(), -1, 1) @ p_expanded[i], 1, -1) for i in range(bs)]
    
        # ... do the same for nodes
        # The selection chooses the correct atom map numbers
        nodes_prods_atom_mapped = [orig_nodes[i,:,:out_dim_nodes][atom_map_numbers_prod_idxs[i]].unsqueeze(0) for i in range(bs)]
        # need to unsqueeze to do batched matrix multiplication correctly: (bs,N,N) @ (bs,N,1) -> (bs,N,1). (N is the count of atom mapped nodes)
        nodes_prods_am_permuted = [p_expanded[i].transpose(dim0=1,dim1=2) @ nodes_prods_atom_mapped[i] for i in range(bs)]
    else:
        assert False, f'Alignment type not set correctly ({alignment_type}). Should be old or correct'
    return nodes_prods_am_permuted, edges_prods_am_permuted, atom_map_numbers_rct

class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU, act_fn_out: nn.ReLU, cfg, 
                 pos_emb_permutations: int = 0,
                 improved=False, dropout=0.1, p_to_r_skip_connection=False, p_to_r_init=10.,
                 alignment_type='old', input_alignment=False):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_nodes = output_dims['nodes']
        self.out_dim_edges = output_dims['edges']
        self.out_dim_y = output_dims['y']
        self.pos_emb_permutations = pos_emb_permutations
        self.p_to_r_skip_connection = p_to_r_skip_connection
        self.alignment_type = alignment_type
        self.input_alignment = input_alignment
        self.cfg = cfg
        if input_alignment:
            input_dims = copy.deepcopy(input_dims)
            original_data_feature_dim_nodes = output_dims['nodes']
            original_data_feature_dim_edges = output_dims['edges']
            input_dims['nodes'] += original_data_feature_dim_nodes # make the input feature dimensionality larger to include the aligned & concatenated product conditioning
            input_dims['edges'] += original_data_feature_dim_edges 

        self.pos_emb = SinusoidalPosEmb(dim=hidden_dims['num_nodes_features'])

        self.mlp_in_nodes = nn.Sequential(nn.Linear(input_dims['nodes'], hidden_mlp_dims['nodes']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['nodes'], hidden_dims['num_nodes_features']), act_fn_in)

        self.mlp_in_edges = nn.Sequential(nn.Linear(input_dims['edges'], hidden_mlp_dims['edges']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['edges'], hidden_dims['num_edge_features']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['num_y_features']), act_fn_in)

        self.tf_layers = nn.ModuleList([FiLM(dx=hidden_dims['num_nodes_features'],
                                                            de=hidden_dims['num_edge_features'],
                                                            dy=hidden_dims['num_y_features'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffnodes=hidden_dims['dim_ffnodes'],
                                                            dim_ffedges=hidden_dims['dim_ffedges'],
                                                            improved=improved, dropout=dropout)
                                        for i in range(n_layers)])

        self.mlp_out_nodes = nn.Sequential(nn.Linear(hidden_dims['num_nodes_features'], hidden_mlp_dims['nodes']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['nodes'], output_dims['nodes']))

        self.mlp_out_edges = nn.Sequential(nn.Linear(hidden_dims['num_edge_features'], hidden_mlp_dims['edges']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['edges'], output_dims['edges']))
        
        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['num_y_features'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))
        
        if self.p_to_r_skip_connection:
            self.skip_scaling = nn.parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_2 = nn.parameter(torch.tensor([p_to_r_init], dtype=torch.float))
            self.skip_scaling_3 = nn.parameter(torch.tensor([1.], dtype=torch.float))

    def forward(self, g_dense: Graph, t: int):
        
        nodes = g_dense.nodes_dense
        edges = g_dense.edges_dense
        y = g_dense.y
        node_mask = g_dense.node_mask
        # atom_charges = g_dense.atom_charges
        # atom_chiral = g_dense.atom_chiral
        # bond_dirs = g_dense.bond_dirs
        # atom_map_numbers = g_dense.atom_map_numbers
        # pos_encodings = g_dense.pos_encodings
        pos_encodings = torch.zeros(nodes.shape[0], nodes.shape[1], nodes.shape[-1])
        # mol_assignment = g_dense.mol_assignment
                
        #assert atom_map_numbers is not None
        bs, n = nodes.shape[0], nodes.shape[1]
        device = nodes.device

        orig_edges = edges.clone()
        orig_nodes = nodes.clone()

        # potential edge-skip connection from product side to reactant side
        if self.input_alignment:
            nodes_prods_am_permuted, edges_prods_am_permuted, atom_map_numbers_rct = get_nodes_prods_and_edges_prods_aligned(mol_assignment, atom_map_numbers, orig_nodes, orig_edges, self.alignment_type, self.out_dim_edges, self.out_dim_nodes, device)
            nodes_prods_am_permuted, edges_prods_am_permuted
            nodes_to_concatenate = torch.zeros(nodes.shape[0], nodes.shape[1], self.out_dim_nodes, device=device)
            edges_to_concatenate_1 = torch.zeros(edges.shape[0], edges.shape[1], edges.shape[2], self.out_dim_edges, device=device)
            edges_to_concatenate_2 = torch.zeros(edges.shape[0], edges.shape[1], edges.shape[2], self.out_dim_edges, device=device)
            for i in range(bs):
                # The following is used for choosing which parts to change in the output
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                edges_to_concatenate_1[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += edges_prods_am_permuted[i].reshape(
                                                                                                    edges_prods_am_permuted[i].shape[1]*edges_prods_am_permuted[i].shape[2],
                                                                                                    edges_prods_am_permuted[i].shape[3]).float()
                if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms:
                    edges_to_concatenate_2[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), edges.shape[-1]).float()
            # nodes_prods_am_permuted = [F.one_hot(nodes_prods_am_permuted[i], self.out_dim_nodes) for i in range(bs)] # Shape (bs, N, dx)
            for i in range(bs):
                am_rct_selection = (atom_map_numbers_rct[i] > 0)
                nodes_to_concatenate[i, am_rct_selection] += nodes_prods_am_permuted[i].squeeze(0).float()
            nodes = torch.cat([nodes, nodes_to_concatenate], dim=-1)
            edges = torch.cat([edges, edges_to_concatenate_1 + edges_to_concatenate_2], dim=-1)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(edges).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        nodes_to_out = nodes[..., :self.out_dim_nodes]
        edges_to_out = edges[..., :self.out_dim_edges]
        y_to_out = y[..., :self.out_dim_y]

        new_edges = self.mlp_in_edges(edges)
        new_edges = (new_edges + new_edges.transpose(1, 2)) / 2

        # all mask padding nodes (with node_mask)
        after_in = Graph(nodes_dense=self.mlp_in_nodes(nodes), edges_dense=new_edges, y=self.mlp_in_y(y)).mask(node_mask)
        nodes, edges, y = after_in.nodes_dense, after_in.edges_dense, after_in.y

        # Add the positional encoding to nodes. nodes shape is now (bs, n, dx)
        # TODO: Maybe concatenate instead so that this works with the Laplacian eigenvectors as well?
        if pos_encodings.shape != nodes.shape:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(bs, n, nodes.shape[-1] - pos_encodings.shape[-1], device=device)], dim=-1)
        nodes = nodes + pos_encodings
        
        for i, layer in enumerate(self.tf_layers):
            nodes, edges, y = layer(nodes, edges, y, node_mask)

        nodes = self.mlp_out_nodes(nodes)
        edges = self.mlp_out_edges(edges)
        y = y[..., :self.out_dim_y]

        nodes = (nodes + nodes_to_out)
        edges = (edges + edges_to_out) * diag_mask
        y = y + y_to_out

        edges = 1/2 * (edges + torch.transpose(edges, 1, 2))

        # potential edge-skip connection from product side to reactant side
        if self.p_to_r_skip_connection:
            nodes_prods_am_permuted, edges_prods_am_permuted, atom_map_numbers_rct = get_nodes_prods_and_edges_prods_aligned(mol_assignment, atom_map_numbers, orig_nodes, orig_edges, 
                                                                                                             self.alignment_type, self.out_dim_edges, self.out_dim_nodes, 
                                                                                                             device)
            if not self.input_alignment: # if this stuff wasn't done already
                for i in range(bs):
                    # The following is used for choosing which parts to change in the output
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    edges[i, am_rct_selection[:,None] * am_rct_selection[None,:]] += edges_prods_am_permuted[i].reshape(
                                                                                                        edges_prods_am_permuted[i].shape[1]*edges_prods_am_permuted[i].shape[2],
                                                                                                        edges_prods_am_permuted[i].shape[3]).float() * self.skip_scaling
                    if self.cfg.neuralnet.skip_connection_on_non_mapped_atoms: # This puts zeros also on product side output, but we discard that anyways so it's fine
                        edges[i, ~(am_rct_selection[:,None]*am_rct_selection[None,:])] += F.one_hot(torch.tensor([0], dtype=torch.long, device=device), edges.shape[-1]).float() * self.skip_scaling_2
                # nodes_prods_am_permuted = [F.one_hot(nodes_prods_am_permuted[i], self.out_dim_nodes) for i in range(bs)] # Shape (bs, N, dx)
                for i in range(bs):
                    am_rct_selection = (atom_map_numbers_rct[i] > 0)
                    nodes[i, am_rct_selection] += nodes_prods_am_permuted[i].squeeze(0).float() * self.skip_scaling
            else: # reuse the previous calculations
                nodes += nodes_to_concatenate * self.skip_scaling
                edges += edges_to_concatenate_1 * self.skip_scaling
                edges += edges_to_concatenate_2 * self.skip_scaling_2

        g_0_pred = Graph(nodes_dense=nodes, edges_dense=edges, node_mask=node_mask)
        
        return g_0_pred
