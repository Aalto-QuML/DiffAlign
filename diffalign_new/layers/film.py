'''
    Implements the FiLM layer as described in DiGress/GraphTransformer model.
'''
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, Dropout
import torch.nn.functional as F
from torch import Tensor
from diffalignX.layers.attention import NodeEdgeBlock
from diffalignX.layers.attention_improved import NodeEdgeBlockImproved

class FiLM(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffnodes: int = 2048,
                 dim_ffedges: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, improved=False) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw) if not improved else NodeEdgeBlockImproved(dx, de, dy, n_head, **kw)

        self.linnodes1 = Linear(dx, dim_ffnodes, **kw)
        self.linnodes2 = Linear(dim_ffnodes, dx, **kw)
        self.normnodes1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normnodes2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutnodes1 = Dropout(dropout)
        self.dropoutnodes2 = Dropout(dropout)
        self.dropoutnodes3 = Dropout(dropout)

        self.linedges1 = Linear(de, dim_ffedges, **kw)
        self.linedges2 = Linear(dim_ffedges, de, **kw)
        self.normedges1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normedges2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutedges1 = Dropout(dropout)
        self.dropoutedges2 = Dropout(dropout)
        self.dropoutedges3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, nodes: Tensor, edges: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            nodes: (bs, n, d)
            edges: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newnodes, newedges, new_y with the same shape.
        """
        newnodes, newedges, new_y = self.self_attn(nodes, edges, y, node_mask=node_mask)
    
        newnodes_d = self.dropoutnodes1(newnodes)
        nodes = self.normnodes1(nodes + newnodes_d)

        newedges_d = self.dropoutedges1(newedges)
        edges = self.normedges1(edges + newedges_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)
        
        ff_outputnodes = self.linnodes2(self.dropoutnodes2(self.activation(self.linnodes1(nodes))))
        ff_outputnodes = self.dropoutnodes3(ff_outputnodes)
        nodes = self.normnodes2(nodes + ff_outputnodes)

        ff_outputedges = self.linedges2(self.dropoutedges2(self.activation(self.linedges1(edges))))
        ff_outputedges = self.dropoutedges3(ff_outputedges)
        edges = self.normedges2(edges + ff_outputedges)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return nodes, edges, y
