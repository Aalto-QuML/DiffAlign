'''
    Uniform transition model.
'''
import torch
from diffalignX.diffusion.noise_schedule import CosineSchedule
from diffalignX.graph_data_structure import Graph

class UniformTransition(torch.nn.Module):
    def __init__(self, dim: int, num_timesteps: int):
        super().__init__()
        self.dim = dim
        self.noise_schedule = CosineSchedule(num_timesteps=num_timesteps)

    def get_q_t_bar(self, t: int) -> torch.Tensor:
        '''
            Get the transition distribution q(x_t|x_0).
        '''
        alpha_t = self.noise_schedule.get_alpha_t_bar(t)
        
        # for nodes
        uniform_transitions_nodes = torch.ones((1, self.dim['nodes'], self.dim['nodes']))/self.dim['nodes']
        q_t_bar_nodes = alpha_t * torch.eye(self.dim['nodes']) + (1 - alpha_t) * uniform_transitions_nodes
        
        # for edges
        uniform_transitions_edges = torch.ones((1, self.dim['edges'], self.dim['edges']))/self.dim['edges']
        q_t_bar_edges = alpha_t * torch.eye(self.dim['edges']) + (1 - alpha_t) * uniform_transitions_edges
        
        return Graph(nodes_dense=q_t_bar_nodes, edges_dense=q_t_bar_edges)
    
    def get_q_t(self, t: int) -> torch.Tensor:
        '''
            Get the cumulative transition distribution q(x_t|x_0).
        '''
        beta_t = self.noise_schedule.get_beta_t(t)
        uniform_transitions_nodes = torch.ones((1, self.dim['nodes'], self.dim['nodes']))/self.dim['nodes']
        q_t_nodes = beta_t * uniform_transitions_nodes + (1 - beta_t) * uniform_transitions_nodes
        uniform_transitions_edges = torch.ones((1, self.dim['edges'], self.dim['edges']))/self.dim['edges']
        q_t_edges = beta_t * uniform_transitions_edges + (1 - beta_t) * uniform_transitions_edges
        
        return Graph(nodes_dense=q_t_nodes, edges_dense=q_t_edges)
    
    def get_q_s_given_t_and_0(self, x_t: torch.Tensor, x_0: torch.Tensor, t: int, s: int) -> torch.Tensor:
        '''
            Get the transition distribution q(x_s|x_t, x_0).
        '''
        q_s_given_t_and_0 = (x_t @ self.get_q_t(t).T) * (x_0 @ self.get_q_t_bar(s))
        
        return q_s_given_t_and_0
    