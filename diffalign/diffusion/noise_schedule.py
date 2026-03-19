import numpy as np
import torch
import torch.nn.functional as F
from diffalign.utils import graph, setup
from diffalign.utils.diffusion import helpers
from diffalign.utils.diffusion import maskdiffusion
import logging
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = helpers.cosine_beta_schedule(timesteps)
        elif noise_schedule == 'custom':
            raise NotImplementedError()
        else:
            raise ValueError(noise_schedule)
        
        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2     # (timesteps + 1, )

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = helpers.cosine_beta_schedule_discrete(timesteps, s=0)
        elif noise_schedule == 'cosine_alternate':
            betas = helpers.cosine_beta_schedule_discrete_alternate(timesteps)
        elif noise_schedule == 'cosine_alternate_2':
            betas = helpers.cosine_beta_schedule_discrete_alternate_2(timesteps)
        elif noise_schedule == 'custom':
            betas = helpers.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float().to(device))

        # Hmm the alpha clamping over here is potentially dubious for discrete diffusion
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
            
        t_int = t_int.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        
        return self.alphas_bar[t_int.long()]

class DiffusionTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int, diffuse_edges: bool=True, 
                 node_idx_to_mask: torch.tensor=torch.ones((1,), dtype=torch.long), edge_idx_to_mask=None):
        self.x_classes = x_classes
        self.e_classes = e_classes
        self.y_classes = y_classes
        self.diffuse_edges = diffuse_edges

        # The following lines make sure that masked node types (node_idx_to_mask) don't mess with the diffusion
        # i.e. that we cannot transition from or to them
        self.Qts_x[...,:,node_idx_to_mask] = 0.
        self.Qts_x[...,node_idx_to_mask,:] = 0.
        self.Qts_x[...,node_idx_to_mask,node_idx_to_mask] = 1.
        if edge_idx_to_mask is not None:
            self.Qts_e[...,edge_idx_to_mask] = 0.
            self.Qts_e[...,edge_idx_to_mask,:] = 0.
            self.Qts_e[...,edge_idx_to_mask,edge_idx_to_mask] = 1.

        # reweighting
        self.Qts_x /= self.Qts_x.sum(dim=-1).unsqueeze(dim=-1)
        self.Qts_e /= self.Qts_e.sum(dim=-1).unsqueeze(dim=-1)

        self.Qt_bars_x[...,:,node_idx_to_mask] = 0.
        self.Qt_bars_x[...,node_idx_to_mask,:] = 0.
        self.Qt_bars_x[...,node_idx_to_mask,node_idx_to_mask] = 1.
        
        if edge_idx_to_mask is not None:
            self.Qt_bars_e[...,edge_idx_to_mask] = 0.
            self.Qt_bars_e[...,edge_idx_to_mask,:] = 0.
            self.Qt_bars_e[...,edge_idx_to_mask,edge_idx_to_mask] = 1.
        
        # reweighting
        self.Qt_bars_x /= self.Qt_bars_x.sum(dim=-1).unsqueeze(dim=-1)
        self.Qt_bars_e /= self.Qt_bars_e.sum(dim=-1).unsqueeze(dim=-1)
        
        # Make limiting distribution not have any probability mass on the masked nodes 
        # (was not an issue with the absorbing state transition, but does matter with the others)
        self.x_limit[node_idx_to_mask] = 0.
        self.e_limit[edge_idx_to_mask] = 0.
        self.x_limit = self.x_limit / self.x_limit.sum()
        self.e_limit = self.e_limit / self.e_limit.sum()

        # Move everything to GPU if we have one
        # TODO: This is probably not the best way to do this, should give the device as an argument
        # But since this is how it's done in a lot of places in the code right now, let's
        # keep it for now
        self.to_device(device)

    def to_device(self, device):
        self.Qts_x = self.Qts_x.to(device)
        self.Qts_e = self.Qts_e.to(device)
        self.Qts_y = self.Qts_y.to(device)
        self.Qt_bars_x = self.Qt_bars_x.to(device)
        self.Qt_bars_e = self.Qt_bars_e.to(device)
        self.Qt_bars_y = self.Qt_bars_y.to(device)
        self.x_limit = self.x_limit.to(device)
        self.e_limit = self.e_limit.to(device)
        self.y_limit = self.y_limit.to(device)

    def get_Qt(self, t, device, diffuse_edges=True):
        # TODO: Use the device argument
        idx = t[:,0].long()
        if self.diffuse_edges:
            return graph.PlaceHolder(X=self.Qts_x[idx], E=self.Qts_e[idx], y=self.Qts_y[idx])
        else:
            # Identity transform for edges
            edge_transition = torch.eye(self.e_classes, device=device)[None].repeat(t.shape[0], 1, 1)
            return graph.PlaceHolder(X=self.Qts_x[idx], E=edge_transition, y=self.Qts_y[idx])

    def get_Qt_bar(self, t, device, diffuse_edges=True):
        # TODO: Use the device argument
        idx = t[:,0].long()
        if self.diffuse_edges:
            return graph.PlaceHolder(X=self.Qt_bars_x[idx], E=self.Qt_bars_e[idx], y=self.Qt_bars_y[idx])
        else:
            # Identity transform for edges
            edge_transition = torch.eye(self.e_classes, device=device)[None].repeat(t.shape[0], 1, 1)
            return graph.PlaceHolder(X=self.Qt_bars_x[idx], E=edge_transition, y=self.Qt_bars_y[idx])
            
    def get_limit_dist(self):
        limit_dist = graph.PlaceHolder(X=self.x_limit, E=self.e_limit, y=self.y_limit)
        return limit_dist

class DiscreteUniformTransition(DiffusionTransition):
    def __init__(self, noise_schedule, timesteps, x_classes: int, e_classes: int, y_classes: int, 
                 diffuse_edges=True, node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long),
                 edge_idx_to_mask=None):
        self.diffuse_edges = diffuse_edges
        self.predefined_noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule, timesteps)
        self.noise_schedule = noise_schedule
        self.timesteps = timesteps

        self.x_classes = x_classes
        self.e_classes = e_classes
        self.y_classes = y_classes

        # TODO: The limiting distributions now include the padding node and SUNo nodes etc.
        self.u_x = torch.ones(1, self.x_classes, self.x_classes)
        self.x_limit = torch.ones(self.x_classes)
        if self.x_classes > 0:
            self.u_x = self.u_x / self.x_classes
            self.x_limit = self.x_limit / self.x_classes

        self.u_e = torch.ones(1, self.e_classes, self.e_classes)
        self.e_limit = torch.ones(self.e_classes)
        if self.e_classes > 0:
            self.u_e = self.u_e / self.e_classes
            self.e_limit = self.e_limit / self.e_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        self.y_limit = torch.ones(self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes
            self.y_limit = self.y_limit / self.y_classes

        self.Qts_x, self.Qt_bars_x = torch.zeros(self.timesteps+1, self.x_classes, self.x_classes), torch.zeros(self.timesteps+1, self.x_classes, self.x_classes)
        self.Qts_e, self.Qt_bars_e = torch.zeros(self.timesteps+1, self.e_classes, self.e_classes), torch.zeros(self.timesteps+1, self.e_classes, self.e_classes)
        self.Qts_y, self.Qt_bars_y = torch.zeros(self.timesteps+1, self.y_classes, self.y_classes), torch.zeros(self.timesteps+1, self.y_classes, self.y_classes)

        for t in range(self.timesteps+1):
            t_int = torch.tensor([t], dtype=torch.long, device=device)
            beta_t = self.predefined_noise_schedule(t_int=t_int)
            beta_t = beta_t.unsqueeze(1)
            beta_t = beta_t.to(device)
            alpha_bar_t = self.predefined_noise_schedule.get_alpha_bar(t_int=t_int) # (bs, 1)
            alpha_bar_t = alpha_bar_t.unsqueeze(1)
            alpha_bar_t = alpha_bar_t.to(device)

            self.u_x = self.u_x.to(device)
            self.u_e = self.u_e.to(device)
            self.u_y = self.u_y.to(device)

            q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.x_classes, device=device).unsqueeze(0)
            q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.e_classes, device=device).unsqueeze(0)
            q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

            q_x_bar = alpha_bar_t * torch.eye(self.x_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
            q_e_bar = alpha_bar_t * torch.eye(self.e_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
            q_y_bar = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

            self.Qts_x[t], self.Qt_bars_x[t] = q_x, q_x_bar
            self.Qts_e[t], self.Qt_bars_e[t] = q_e, q_e_bar
            self.Qts_y[t], self.Qt_bars_y[t] = q_y, q_y_bar

        super().__init__(self.x_classes, self.e_classes, self.y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class MarginalUniformTransition(DiffusionTransition):
    def __init__(self, x_marginals, e_marginals, y_classes, noise_schedule, timesteps, 
                 diffuse_edges=True, node_idx_to_mask=None, edge_idx_to_mask=None):
        
        self.diffuse_edges = diffuse_edges

        self.predefined_noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule, timesteps)
        self.noise_schedule = noise_schedule
        self.timesteps = timesteps

        self.x_classes = len(x_marginals)
        self.e_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        self.u_x = x_marginals.unsqueeze(0).expand(self.x_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.e_classes, -1).unsqueeze(0)

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

        self.Qts_x, self.Qt_bars_x = torch.zeros(self.timesteps+1, self.x_classes, self.x_classes), torch.zeros(self.timesteps + 1, self.x_classes, self.x_classes)
        self.Qts_e, self.Qt_bars_e = torch.zeros(self.timesteps+1, self.e_classes, self.e_classes), torch.zeros(self.timesteps + 1, self.e_classes, self.e_classes)
        self.Qts_y, self.Qt_bars_y = torch.zeros(self.timesteps+1, self.y_classes, self.y_classes), torch.zeros(self.timesteps + 1, self.y_classes, self.y_classes)
        
        for t in range(self.timesteps+1):
            t_int = torch.tensor([t], dtype=torch.long, device=device)
            beta_t = self.predefined_noise_schedule(t_int=t_int)
            beta_t = beta_t.unsqueeze(1)
            beta_t = beta_t.to(device)
            alpha_bar_t = self.predefined_noise_schedule.get_alpha_bar(t_int=t_int) # (bs, 1)
            alpha_bar_t = alpha_bar_t.unsqueeze(1)
            alpha_bar_t = alpha_bar_t.to(device)

            self.u_x = self.u_x.to(device)
            self.u_e = self.u_e.to(device)
            self.u_y = self.u_y.to(device)

            q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.x_classes, device=device).unsqueeze(0)
            q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.e_classes, device=device).unsqueeze(0)
            q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

            q_x_bar = alpha_bar_t * torch.eye(self.x_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
            q_e_bar = alpha_bar_t * torch.eye(self.e_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
            q_y_bar = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

            self.Qts_x[t], self.Qt_bars_x[t] = q_x, q_x_bar
            self.Qts_e[t], self.Qt_bars_e[t] = q_e, q_e_bar
            self.Qts_y[t], self.Qt_bars_y[t] = q_y, q_y_bar

        # TODO: Check that there is no probability for the supernodes etc. 
        self.x_limit = self.x_marginals
        self.e_limit = self.e_marginals

        self.y_limit = torch.ones(self.y_classes) / self.y_classes

        super().__init__(self.x_classes, self.e_classes, self.y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class AbsorbingStateTransitionMaskNoEdge(DiffusionTransition):
    def __init__(self, timesteps: int, x_classes: int, e_classes: int, y_classes: int, diffuse_edges=True, 
                 abs_state_position_x=-1, abs_state_position_e=0, node_idx_to_mask: torch.tensor=3*torch.ones((1,), 
                 dtype=torch.long), edge_idx_to_mask=None):
        T = timesteps
        xdiffusion = maskdiffusion.MaskDiffusion(T=T, K=x_classes, aux_lambda=0)
        ediffusion = maskdiffusion.MaskDiffusion(T=T, K=e_classes, aux_lambda=0)
        ydiffusion = maskdiffusion.MaskDiffusion(T=T, K=y_classes+1, aux_lambda=0)
        
        self.Qts_x = [torch.eye(x_classes, device=device)] + [1/(T - t + 1) * F.one_hot(torch.tensor([abs_state_position_x], dtype=torch.long, device=device),x_classes).repeat(x_classes, 1) 
                                               + (1-1/(T - t + 1)) * torch.eye(x_classes, device=device, dtype=torch.float) for t in range(1,timesteps+1)]
        self.Qt_bars_x = [torch.eye(x_classes, device=device)] + [t/(T) * F.one_hot(torch.tensor([abs_state_position_x], dtype=torch.long, device=device),x_classes).repeat(x_classes, 1) 
                                               + (1 - t/(T)) * torch.eye(x_classes, device=device, dtype=torch.float) for t in range(1,timesteps+1)]
        self.Qts_e = [torch.eye(e_classes, device=device)] + [1/(T - t + 1) * F.one_hot(torch.tensor([abs_state_position_e], dtype=torch.long, device=device),e_classes).repeat(e_classes, 1)
                                                  + (1-1/(T - t + 1)) * torch.eye(e_classes, device=device, dtype=torch.float) for t in range(1,timesteps+1)]
        self.Qt_bars_e = [torch.eye(e_classes, device=device)] + [t/(T) * F.one_hot(torch.tensor([abs_state_position_e], dtype=torch.long, device=device),e_classes).repeat(e_classes, 1)
                                                    + (1 - t/(T)) * torch.eye(e_classes, device=device, dtype=torch.float) for t in range(1,timesteps+1)]
        self.Qts_y = [torch.eye(y_classes+1, device=device)] + [1/(T - t + 1) * F.one_hot(torch.tensor([y_classes], dtype=torch.long, device=device),y_classes+1).repeat(y_classes+1, 1)
                                                    + (1-1/(T - t + 1)) * torch.eye(y_classes+1, device=device, dtype=torch.float) for t in range(1,timesteps+1)]
        self.Qt_bars_y = [torch.eye(y_classes+1, device=device)] + [t/(T) * F.one_hot(torch.tensor([y_classes], dtype=torch.long, device=device),y_classes+1).repeat(y_classes+1, 1)
                                                        + (1 - t/(T)) * torch.eye(y_classes+1, device=device, dtype=torch.float) for t in range(1,timesteps+1)]
        self.Qts_x = torch.stack(self.Qts_x)
        self.Qt_bars_x = torch.stack(self.Qt_bars_x)
        self.Qts_e = torch.stack(self.Qts_e)
        self.Qt_bars_e = torch.stack(self.Qt_bars_e)
        self.Qts_y = torch.stack(self.Qts_y)
        self.Qt_bars_y = torch.stack(self.Qt_bars_y)

        self.x_limit = F.one_hot(torch.tensor([abs_state_position_x], dtype=torch.long), x_classes)[0].float() # remove batch dimension
        self.e_limit = F.one_hot(torch.tensor([abs_state_position_e], dtype=torch.long), e_classes)[0].float()
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

