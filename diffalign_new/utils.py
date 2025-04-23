import torch
import torch.nn.functional as F
from torch_geometric import data
from torch_geometric.utils import to_dense_adj, to_dense_batch
from diffalignX.graph_data_structure import Graph

def cross_entropy(g_0_pred: Graph, g_dense: Graph):
    '''
        Computes the cross entropy loss between the predicted and dense graphs.
    '''
    nodes = F.cross_entropy(g_0_pred.nodes_dense, g_dense.nodes_dense)
    edges = F.cross_entropy(g_0_pred.edges_dense, g_dense.edges_dense)
    return (nodes + edges).mean()

def turn_pyg_to_dense_graph(g: data, t: int):
    '''
        Converts a PyG data object to a graph object with dense node and edge representations.
        g: PyG data object
        returns: Graph object with dense node and edge representations
    '''
    nodes_dense, node_mask = to_dense_batch(g.x)
    edges_dense = to_dense_adj(g.edge_index, batch=g.batch, edge_attr=g.edge_attr)
    # TODO: could add other features here
    graph = Graph(nodes_dense, edges_dense, y=torch.ones((1, 1), dtype=torch.float)*t, node_mask=node_mask)
    return graph

def turn_dense_graph_to_pyg(g: Graph):
    '''
        Converts a dense graph object to a PyG data object.
        g: dense graph object
        returns: PyG data object
    '''
    pass
    