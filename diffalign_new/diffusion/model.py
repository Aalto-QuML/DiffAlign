import torch 
from torch.nn import CrossEntropyLoss
from torch_geometric import data
from diffalignX.utils import turn_pyg_to_dense_graph
from diffalignX.diffusion.uniform_transition import UniformTransition
from diffalignX.graph_transformer import GraphTransformer
from diffalignX.graph_data_structure import Graph
from diffalignX.utils import cross_entropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DiffAlign(torch.nn.Module):
    def __init__(self, num_timesteps: int = 1000, dim: int = 10, nn_params: dict = None):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.transition = UniformTransition(dim=dim, num_timesteps=num_timesteps)
        self.denoiser  = GraphTransformer(**nn_params)
        self.loss_fn = None

    def noise_one_step(self, g_dense: Graph, t: int) -> Graph:
        '''
            Performs a single forward process step, i.e. samples from the conditional distribution q(x_t|x_0).
        '''
        q_t = self.transition.get_q_t(t)
        g_t = g_dense.dot(q_t)
        return g_t

    def denoise_one_step(self, g_dense: Graph, t: int) -> Graph:
        '''
            Performs a single denoising step, i.e. samples from the conditional distribution q(x_s|x_t),
            s.t. q(x_s|x_t) =  q(x_s|x_t, x_0) * p_{\\theta}(x_0|x_t), 
            where p_{\\theta}(x_0|x_t) is the denoiser NN's output, and s = t-1.
        '''
        # NOTE: s = t-1
        # sample from p_{\\theta}(x_0|x_t)
        g_0_pred = self.denoiser.forward(g_dense, t)
        # sample from q(x_{t-1}|x_t, x_0)
        q_s_given_t_and_0 = self.transition.get_q_s_given_t_and_0(g_t=g_dense, g_0=g_0_pred, t=t, s=t-1)
        # sample from q(x_{t-1}|x_t)
        q_s_given_t = g_0_pred.dot(q_s_given_t_and_0)
        # sample from q(x_{t-1}|x_t)
        g_s = q_s_given_t.sample()
        return g_s

    def sample(self) -> Graph:
        '''
            Sample from the diffusion model, starting from x_T and going to x_0.
        '''
        # sample x_T
        q_prior = self.transition.get_q_t(t=self.num_timesteps)
        g_t = q_prior.sample()
        # denoise
        for t in reversed(range(self.num_timesteps)):
            g_t = self.denoise_one_step(g_t, t)
        return g_t
    
    def forward(self, g: Graph, t: int):
        '''
            Forward pass of the model.
        '''
        return self.denoiser.forward(g, t)
    
    def training_step(self, g: data):
        '''
            Performs a single training step.
        '''
        t_int = torch.randint(1, self.num_timesteps+1, size=(len(g),1), device=device)
        g_dense = turn_pyg_to_dense_graph(g, t_int)
        # noise the sample
        g_t = self.noise_one_step(g_dense, t_int)
        # denoise the sample
        g_0_pred = self.forward(g_t)
        # compute loss
        loss = cross_entropy(g_0_pred, g_dense)
        
        return loss
    
    