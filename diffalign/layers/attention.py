'''
    Implements the attention layer as described in DiGress/GraphTransformer model.
'''
import math
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from diffalignX.layers.map_node_to_global_features import Nodestoy
from diffalignX.layers.map_edges_to_global_features import Edgestoy

def assert_correctly_masked(variable, node_mask):
    '''
    Assert that the variable is correctly masked.
    '''
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

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

        # FiLM edges to nodes
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to edges
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to nodes
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(de, de)
        self.x_y = Nodestoy(dx, dy)
        self.e_y = Edgestoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, nodes, edges, y, node_mask):
        """
        :param nodes: bs, n, d        node features
        :param edges: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newnodes, newedges, new_y with the same shape.
        """
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map nodes to keys and queries
        Q = self.q(nodes) * x_mask           # (bs, n, dx)
        K = self.k(nodes) * x_mask           # (bs, n, dx)
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

        edges1 = self.e_mul(edges) * e_mask1 * e_mask2                        # bs, n, n, dx
        edges1 = edges1.reshape((edges.size(0), edges.size(1), edges.size(2), self.n_head, self.df))

        edges2 = self.e_add(edges) * e_mask1 * e_mask2                        # bs, n, n, dx
        edges2 = edges2.reshape((edges.size(0), edges.size(1), edges.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (edges1 + 1) + edges2                  # (bs, n, n, n_head, df)

        # Incorporate y to edges
        newedges = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newedges = ye1 + (ye2 + 1) * newedges

        # Output edges
        newedges = self.e_out(newedges) * e_mask1 * e_mask2      # bs, n, n, de
        assert_correctly_masked(newedges, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        attn = F.softmax(Y, dim=2)

        V = self.v(nodes) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to nodes
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newnodes = yx1 + (yx2 + 1) * weighted_V

        # Output nodes
        newnodes = self.x_out(newnodes) * x_mask
        assert_correctly_masked(newnodes, x_mask)

        ## Process y based on nodes and edges
        new_y = self.y_y(y)
        e_y = self.e_y(edges)
        x_y = self.x_y(nodes)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy
        # new_y = y 

        return newnodes, newedges, new_y