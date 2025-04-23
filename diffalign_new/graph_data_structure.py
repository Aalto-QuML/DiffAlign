'''
    Graph object used to represent reactions and other related objects 
    (e.g. potentially transition matrices).
'''
import torch
            
def encode_no_element(a: torch.Tensor) -> torch.Tensor:
    '''
        Turns no elements (e.g. from dense padding) to one-hot encoded vectors.
        Works on X and E.
    '''
    assert len(a.shape) >= 3
    if a.shape[-1]==0:
        return a
    no_elt = torch.sum(a, dim=-1) == 0
    first_elt = a[..., 0]
    first_elt[no_elt] = 1
    a[..., 0] = first_elt
    return a

class Graph:
    def __init__(self,
                 nodes_dense: torch.Tensor,
                 edges_dense: torch.Tensor,
                 y :torch.Tensor = None,
                 node_mask: torch.Tensor = None):
        assert nodes_dense.ndim == 3, f'Nodes must be in dense representation with shape (bs, n, d_x), got {nodes_dense.shape}'
        #assert edges_dense.ndim == 4, f'Edges must be in dense representation with shape (bs, n, n, d_e), got {edges_dense.shape}'
        self.nodes_dense = nodes_dense
        self.edges_dense = edges_dense
        self.y = y
        self.node_mask = node_mask
    
    def dot(self, other: 'Graph') -> 'Graph':
        '''
            Performs a dot product between the relevant components of the two graphs.
            
            other: the other graph to dot with.
            returns: the resulting graph.
        '''
        self.nodes_dense = self.nodes_dense @ other.nodes_dense
        self.edges_dense = self.edges_dense @ other.edges_dense
        return self
    
    def mask(self, node_mask=None, collapse=False):
        '''
            Masks the nodes and edges of the graph.
        '''
        if node_mask==None: node_mask = self.node_mask
            
        assert node_mask is not None, 'node_mask is None.'
            
        x_node_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_node_mask1 = x_node_mask.unsqueeze(2)            # bs, n, 1, 1
        e_node_mask2 = x_node_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.nodes_dense = torch.argmax(self.nodes_dense, dim=-1) # (bs, n)
            self.edges_dense = torch.argmax(self.edges_dense, dim=-1) # (bs, n, n)
            # self.atom_charges = torch.argmax(self.atom_charges, dim=-1) if self.atom_charges!=None else None
            # self.atom_chiral = torch.argmax(self.atom_chiral, dim=-1) if self.atom_chiral!=None else None
            # self.bond_dirs = torch.argmax(self.bond_dirs, dim=-1) if self.bond_dirs!=None else None

            self.nodes_dense[node_mask == 0] = 0
            self.edges_dense[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0
            # self.atom_charges[node_mask == 0] = 0 if self.atom_charges!=None else None
            # self.atom_chiral[node_mask == 0] = 0 if self.atom_chiral!=None else None
            # self.bond_dirs[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0 if self.bond_dirs!=None else None
        else:
            # always mask by node, masking by subgraph is a subset of that
    
            # X_0 = NN(noisy.X) => (bs, n_max, v)
            # => e.g.: true nodes: (0, n<n_max, v) = [0.9, 0.8, .....]
            #          fake nodes: (0, n>n_max, v) = [0.9, 0.8, .....] => [1, 0, 0, 0....]
                    
            # => how to get correct fake nodes?
            # => node_mask: (bs, n_max, 1)
            # X_0 * node_mask => X_0' = (bs, n<n_max, v) = [0.9, 0.8, .....]
            #                    X_0' = (bs, n>n_max, v) = [0, 0, .....] (doesn't exist)
                                
            # => last step: fix the [0, ...] to [1, 0, ...]
            # => last step for other masks: X_0' = X_0 * node_mask + X_orig * (~node_mask)
            #     => X_0': (bs, n<n_max, v) = [0.9, 0.8, .....] (e.g. output of NN)
            #  => X_0': (bs, n>n_max, v) = [1, 0, .....] (e.g. orig one_hot) => perks: 
            # already proba dist, already one-hot...
        
            self.nodes_dense = self.nodes_dense * x_node_mask
            # self.atom_charges = self.atom_charges * x_node_mask if self.atom_charges!=None else None
            # self.atom_chiral = self.atom_chiral * x_node_mask if self.atom_charges!=None else None
            self.edges_dense = self.edges_dense * e_node_mask1 * e_node_mask2
            # self.bond_dirs = self.bond_dirs * e_node_mask1 * e_node_mask2 if self.bond_dirs!=None else None
            diag = torch.eye(self.edges_dense.shape[1], dtype=torch.bool).unsqueeze(0).expand(self.edges_dense.shape[0], -1, -1)
            self.edges_dense[diag] = 0
            self.nodes_dense = encode_no_element(self.nodes_dense)
            self.edges_dense = encode_no_element(self.edges_dense)
            # self.atom_charges = encode_no_element(self.atom_charges) if self.atom_charges!=None else None
            # self.atom_chiral = encode_no_element(self.atom_chiral) if self.atom_chiral!=None else None
            # self.bond_dirs = encode_no_element(self.bond_dirs) if self.bond_dirs!=None else None

            # adjacency matrix of undirected graph => mirrored over the diagonal
            # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self
    