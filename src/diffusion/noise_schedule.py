import numpy as np
import torch
import torch.nn.functional as F
from src.utils import graph, setup
from src.utils.diffusion import helpers
from src.utils.diffusion import tokenwisediffusion
from src.utils.diffusion import blocktokenwisediffusion, maskdiffusion
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

class MarginalEdgesMaskNodesTransition(DiffusionTransition):
    def __init__(self, x_classes, e_marginals, y_classes, noise_schedule, timesteps, 
                 diffuse_edges=True, node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long),
                 edge_idx_to_mask=None):
        self.diffuse_edges = diffuse_edges

        self.predefined_noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule, timesteps)
        self.noise_schedule = noise_schedule
        self.timesteps = timesteps
        self.x_classes = x_classes
        self.e_classes = len(e_marginals)
        self.y_classes = y_classes
        self.e_marginals = e_marginals

        self.u_e = e_marginals.unsqueeze(0).expand(self.e_classes, -1).unsqueeze(0)

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

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

            self.u_e = self.u_e.to(device)
            self.u_y = self.u_y.to(device)

            q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.e_classes, device=device).unsqueeze(0)
            q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

            q_e_bar = alpha_bar_t * torch.eye(self.e_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
            q_y_bar = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

            self.Qts_e[t], self.Qt_bars_e[t] = q_e, q_e_bar
            self.Qts_y[t], self.Qt_bars_y[t] = q_y, q_y_bar

        self.e_limit = self.e_marginals

        xdiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=x_classes, aux_lambda=0)
        self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices()
        self.x_limit = F.one_hot(torch.tensor([x_classes-1], dtype=torch.long), x_classes)[0].float() # remove batch dimension

        self.y_limit = torch.ones(self.y_classes) / self.y_classes

        super().__init__(self.x_classes, self.e_classes, self.y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class AbsorbingStateTransitionMaskNoEdgeOld(DiffusionTransition):
    # Old version that uses the MaskDiffusion class
    def __init__(self, timesteps: int, x_classes: int, e_classes: int, y_classes: int, diffuse_edges=True, 
                 abs_state_position_x=-1, abs_state_position_e=0, node_idx_to_mask: torch.tensor=3*torch.ones((1,), 
                 dtype=torch.long), edge_idx_to_mask=None):
        xdiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=x_classes, aux_lambda=0)
        ediffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=e_classes, aux_lambda=0)
        ydiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=y_classes+1, aux_lambda=0)
        self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices(abs_state_position=abs_state_position_x)
        self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=abs_state_position_e)
        self.Qts_y, self.Qt_bars_y = ydiffusion.return_all_transition_matrices()

        self.x_limit = F.one_hot(torch.tensor([abs_state_position_x], dtype=torch.long), x_classes)[0].float() # remove batch dimension
        self.e_limit = F.one_hot(torch.tensor([abs_state_position_e], dtype=torch.long), e_classes)[0].float()# TODO Make sure that this is correct
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

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
        self.e_limit = F.one_hot(torch.tensor([abs_state_position_e], dtype=torch.long), e_classes)[0].float()# TODO Make sure that this is correct
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class TokenwiseAbsorbingStateTransition(DiffusionTransition):
    def __init__(self, x_classes: int, e_classes: int, y_classes: int, timesteps: int,
                 node_type_counts_unnormalized, edge_type_counts_unnormalized, abs_state_position_x=-1,
                 abs_state_position_e=0, diffuse_edges=True, node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long),
                 edge_idx_to_mask=None):
        xdiffusion = tokenwisediffusion.TokenWiseDiffusion(T=timesteps, K=x_classes, token_counts=node_type_counts_unnormalized[:-1], aux_lambda=0, 
                                                           discr_num=30000)
        ediffusion = tokenwisediffusion.TokenWiseDiffusion(T=timesteps, K=e_classes, token_counts=edge_type_counts_unnormalized[:-1], aux_lambda=0,
                                                           discr_num=80000)
        ydiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=y_classes+1, aux_lambda=0)
        self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices(abs_state_position=abs_state_position_x)
        self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=abs_state_position_e)
        self.Qts_y, self.Qt_bars_y = ydiffusion.return_all_transition_matrices()

        self.x_limit = F.one_hot(torch.tensor([x_classes-1], dtype=torch.long), x_classes)[0].float() # remove batch dimension
        self.e_limit = F.one_hot(torch.tensor([e_classes-1], dtype=torch.long), e_classes)[0].float()
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class TokenwiseAbsorbingStateTransitionMaskNoEdge(DiffusionTransition):
    def __init__(self, x_classes: int, e_classes: int, y_classes: int, timesteps: int,
                 node_type_counts_unnormalized, edge_type_counts_unnormalized, 
                 sort_by_count=True, diffuse_edges= True, 
                 abs_state_position_x=-1, abs_state_position_e=0,
                 node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long),
                 edge_idx_to_mask=None):
        if sort_by_count:
            # The padding token is position zero and the mask token is position -1, so we don't change those
            # Also the first state for the edges is the 'no edge' state, so we don't change that
            node_type_counts_unnormalized_sorted, node_permute_indices = torch.sort(node_type_counts_unnormalized[1:-1])
            edge_type_counts_unnormalized_sorted, edge_permute_indices = torch.sort(edge_type_counts_unnormalized[1:])
            self.node_type_counts_unnormalized_sorted = torch.cat([node_type_counts_unnormalized[:1], node_type_counts_unnormalized_sorted, node_type_counts_unnormalized[-1:]])
            self.edge_type_counts_unnormalized_sorted = torch.cat([edge_type_counts_unnormalized[:1], edge_type_counts_unnormalized_sorted])
            self.node_permute_indices = torch.cat([torch.tensor([0]), node_permute_indices, torch.tensor([len(node_type_counts_unnormalized)-1])])
            self.edge_permute_indices = torch.cat([torch.tensor([0]), edge_permute_indices])
            # Make the change in the array that is actually used
            node_type_counts_unnormalized = self.node_type_counts_unnormalized_sorted
            edge_type_counts_unnormalized = self.edge_type_counts_unnormalized_sorted

        """Logic here: Create MaskDiffusion classes"""
        # Assumption here is that the last node type is the absorbing state
        # ... and there is no extra 'noedge' edge type, instead we use the first actual 'no edge' state as mask
        xdiffusion = tokenwisediffusion.TokenWiseDiffusion(T=timesteps, K=x_classes, token_counts=node_type_counts_unnormalized[:-1], aux_lambda=0, 
                                                           discr_num=120*timesteps)
        # We don't probably want to count the 'no edge'/mask in the MI calculations
        # What is the last edge type? Where does it come from?
        # ... no yeah we definitely do want to count the 'no edge'/mask in the MI calculations, actually
        # ... soo I guess our code doesn't quite work as is in this setting
        # How to make it work?
        # Yeah will have to do the mi calculations again here -> try to do really quick
        # -> no, if we use the 'no edge' thing, then on the first step, those will get turned into the mask state anyways
        # so we just create these larger matrices and then erase the second row and column, corresponding to the 'no edge' state
        ediffusion = tokenwisediffusion.TokenWiseDiffusion(T=timesteps, K=e_classes+1, token_counts=edge_type_counts_unnormalized[:], aux_lambda=0,
                                                           discr_num=240*timesteps)
        # print(node_type_counts_unnormalized)
        # print(edge_type_counts_unnormalized)
        ydiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=y_classes+1, aux_lambda=0)
        self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices(abs_state_position=abs_state_position_x)
        self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=abs_state_position_e, drop_state=1)
        self.Qts_y, self.Qt_bars_y = ydiffusion.return_all_transition_matrices()
        if sort_by_count:
            # Undo the sorting
            self.Qts_x, self.Qt_bars_x = xdiffusion.permute_transition_matrices(self.Qts_x, self.Qt_bars_x, torch.argsort(self.node_permute_indices))
            self.Qts_e, self.Qt_bars_e = ediffusion.permute_transition_matrices(self.Qts_e, self.Qt_bars_e, torch.argsort(self.edge_permute_indices))

        self.x_limit = F.one_hot(torch.tensor([x_classes-1], dtype=torch.long), x_classes)[0].float()
        self.e_limit = F.one_hot(torch.tensor([e_classes-1], dtype=torch.long), e_classes)[0].float()
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class BlockTokenwiseAbsorbingStateTransitionMaskNoEdge(DiffusionTransition):

    # TODO: Could combine this with the regular tokenwise transition class

    def __init__(self, x_classes: int, e_classes: int, y_classes: int, timesteps: int,
                 node_type_counts_unnormalized, edge_type_counts_unnormalized, 
                 abs_state_position_x=-1, abs_state_position_e=0, sort_by_count=True, 
                 diffuse_edges= True, node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long),
                 edge_idx_to_mask=None):

        if sort_by_count:
            # The padding token is position zero and the mask token is position -1, so we don't change those
            # Also the first state for the edges is the 'no edge' state, so we don't change that
            node_type_counts_unnormalized_sorted, node_permute_indices = torch.sort(node_type_counts_unnormalized[1:-1])
            # edge_type_counts_unnormalized_sorted, edge_permute_indices = torch.sort(edge_type_counts_unnormalized[1:])
            self.node_type_counts_unnormalized_sorted = torch.cat([node_type_counts_unnormalized[:1], node_type_counts_unnormalized_sorted, node_type_counts_unnormalized[-1:]])
            # self.edge_type_counts_unnormalized_sorted = torch.cat([edge_type_counts_unnormalized[:1], edge_type_counts_unnormalized_sorted])
            self.node_permute_indices = torch.cat([torch.tensor([0]), node_permute_indices, torch.tensor([len(node_type_counts_unnormalized)-1])])
            # self.edge_permute_indices = torch.cat([torch.tensor([0]), edge_permute_indices])         
            # Make the change in the array that is actually used
            node_type_counts_unnormalized = self.node_type_counts_unnormalized_sorted
            # edge_type_counts_unnormalized = self.edge_type_counts_unnormalized_sorted

        """Logic here: Create MaskDiffusion classes"""
        # Assumption here is that the last node type is the absorbing state
        # ... and there is no extra 'noedge' edge type, instead we use the first actual 'no edge' state as mask
        groups = [list(range(1, x_classes-1)), [x_classes-1]] # This is using the convention within the blocktokenwisediffusion class, where mask state is not included in the groups
        # and the last state will be carbon (the most frequent node type). TODO make this more general
        xdiffusion = blocktokenwisediffusion.BlockTokenWiseDiffusion(T=timesteps, K=x_classes, token_counts=node_type_counts_unnormalized[:-1], aux_lambda=0, 
                                                           discr_num=120*timesteps, groups=groups, device=device)
        # We don't probably want to count the 'no edge'/mask in the MI calculations
        # What is the last edge type? Where does it come from?
        # ... no yeah we definitely do want to count the 'no edge'/mask in the MI calculations, actually
        # ... soo I guess our code doesn't quite work as is in this setting
        # How to make it work?
        # Yeah will have to do the mi calculations again here -> try to do really quick
        # -> no, if we use the 'no edge' thing, then on the first step, those will get turned into the mask state anyways
        # so we just create these larger matrices and then erase the second row and column, corresponding to the 'no edge' state
        # ediffusion = tokenwisediffusion.TokenWiseDiffusion(T=timesteps, K=e_classes+1, token_counts=edge_type_counts_unnormalized[:], aux_lambda=0,
        #                                                    discr_num=240*timesteps, device=device)

        # CHANGE: For now let's just use regular masking diffusion here
        ediffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=e_classes, aux_lambda=0)

        # print(node_type_counts_unnormalized)
        # print(edge_type_counts_unnormalized)
        ydiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=y_classes+1, aux_lambda=0)
        self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices()
        # The following line is with the more complex version where we use the tokenwise diffusion for edges
        #self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=0, drop_state=1)
        self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=abs_state_position_e)
        self.Qts_y, self.Qt_bars_y = ydiffusion.return_all_transition_matrices()
        if sort_by_count:
            # Undo the sorting
            self.Qts_x, self.Qt_bars_x = xdiffusion.permute_transition_matrices(self.Qts_x, self.Qt_bars_x, torch.argsort(self.node_permute_indices))
            # self.Qts_e, self.Qt_bars_e = ediffusion.permute_transition_matrices(self.Qts_e, self.Qt_bars_e, torch.argsort(self.edge_permute_indices))

        self.x_limit = F.one_hot(torch.tensor([x_classes-1], dtype=torch.long), x_classes)[0].float()
        self.e_limit = F.one_hot(torch.tensor([0], dtype=torch.long), e_classes)[0].float()
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class NodesBeforeEdges(DiffusionTransition):
    def __init__(self, x_classes: int, e_classes: int, y_classes: int, timesteps: int, diffuse_edges=True, 
                 abs_state_position_x=-1, abs_state_position_e=0, node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long), 
                 edge_idx_to_mask=None, num_node_steps=1):
    
        # Hmm something off, abs_state_position should be 27, clearly

        # Change so that timesteps-num_node_steps of generation is spent on edges and num_node_steps is spent on nodes
        # (num_node_steps+1 because we include the identity transition in the beginning)
        xdiffusion = maskdiffusion.MaskDiffusion(T=num_node_steps, K=x_classes, aux_lambda=0)
        ediffusion = maskdiffusion.MaskDiffusion(T=timesteps-num_node_steps, K=e_classes, aux_lambda=0)
        ydiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=y_classes+1, aux_lambda=0)

        # How can this function work if x_classes = 28 and abs_state_position_x = 28? Should be index error here
        self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices(abs_state_position=abs_state_position_x)
        self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=abs_state_position_e)
        # Drop out the first identity transform from the edge transitions
        # self.Qts_e, self.Qt_bars_e = self.Qts_e[1:], self.Qt_bars_e[1:]

        # Put identity matrices where nothing happens on edges first
        identity_matrices_e = torch.cat([torch.eye(e_classes)[None,:,:] for _ in range(num_node_steps)], 0).to(device)
        # And the absorbing state transition for the node transitions in the second part.
        # Note: The absorbing state was here defined to be the last state in return_all_transition_matrices
        # TODO: Change the absorbing state to the correct thing instead of the last state
        absorbing_matrices_x = torch.cat([torch.zeros(1, x_classes, x_classes) for _ in range(timesteps - num_node_steps)], 0).to(device)
        absorbing_matrices_x[:, :, abs_state_position_x] = 1
        
        self.Qts_x = torch.cat([self.Qts_x, absorbing_matrices_x], axis=0)
        self.Qt_bars_x = torch.cat([self.Qt_bars_x, absorbing_matrices_x], axis=0)
        self.Qts_e = torch.cat([identity_matrices_e, self.Qts_e], axis=0)
        self.Qt_bars_e = torch.cat([identity_matrices_e, self.Qt_bars_e], axis=0)
        self.Qts_y, self.Qt_bars_y = ydiffusion.return_all_transition_matrices()

        # TODO: Fix the absorbing states here as well
        # TODO: Check the correctness of the -1 thing here
        self.x_limit = F.one_hot(torch.tensor([abs_state_position_x], dtype=torch.long), x_classes)[0].float() # remove batch dimension
        self.e_limit = F.one_hot(torch.tensor([abs_state_position_e], dtype=torch.long), e_classes)[0].float()
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

class EdgesBeforeNodes(DiffusionTransition):
    def __init__(self, x_classes: int, e_classes: int, y_classes: int, timesteps: int, diffuse_edges=True, 
                 abs_state_position_x=-1, abs_state_position_e=0, node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long), 
                 edge_idx_to_mask=None, num_node_steps=1):
    
        # Change so that timesteps-num_node_steps of generation is spent on edges and num_node_steps is spent on nodes
        xdiffusion = maskdiffusion.MaskDiffusion(T=num_node_steps, K=x_classes, aux_lambda=0)
        ediffusion = maskdiffusion.MaskDiffusion(T=timesteps-num_node_steps, K=e_classes, aux_lambda=0)
        ydiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=y_classes+1, aux_lambda=0)

        # How can this function work if x_classes = 28 and abs_state_position_x = 28? Should be index error here
        self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices(abs_state_position=abs_state_position_x)
        self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=abs_state_position_e)
        # Drop out the first identity transform from the edge transitions
        # self.Qts_e, self.Qt_bars_e = self.Qts_e[1:], self.Qt_bars_e[1:]

        # Put identity matrices where nothing happens on nodes first
        identity_matrices_x = torch.cat([torch.eye(x_classes)[None,:,:] for _ in range(timesteps-num_node_steps)], 0).to(device)
        # And the absorbing state transition for the node transitions in the second part.
        # Note: The absorbing state was here defined to be the last state in return_all_transition_matrices
        # TODO: Change the absorbing state to the correct thing instead of the last state
        absorbing_matrices_e = torch.cat([torch.zeros(1, e_classes, e_classes) for _ in range(num_node_steps)], 0).to(device)
        absorbing_matrices_e[:, :, abs_state_position_e] = 1
        
        self.Qts_e = torch.cat([self.Qts_e, absorbing_matrices_e], axis=0)
        self.Qt_bars_e = torch.cat([self.Qt_bars_e, absorbing_matrices_e], axis=0)
        self.Qts_x = torch.cat([identity_matrices_x, self.Qts_x], axis=0)
        self.Qt_bars_x = torch.cat([identity_matrices_x, self.Qt_bars_x], axis=0)
        self.Qts_y, self.Qt_bars_y = ydiffusion.return_all_transition_matrices()

        # TODO: Fix the absorbing states here as well
        # TODO: Check the correctness of the -1 thing here
        self.x_limit = F.one_hot(torch.tensor([abs_state_position_x], dtype=torch.long), x_classes)[0].float() # remove batch dimension
        self.e_limit = F.one_hot(torch.tensor([abs_state_position_e], dtype=torch.long), e_classes)[0].float()
        self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

        super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)

# class EdgesBeforeNodes(DiffusionTransition):
#     def __init__(self, x_classes: int, e_classes: int, y_classes: int, timesteps: int,
#                  node_type_counts_unnormalized, edge_type_counts_unnormalized, 
#                  abs_state_position_x=-1, abs_state_position_e=0,
#                  tokenwise=False, diffuse_edges=True, node_idx_to_mask: torch.tensor=3*torch.ones((1,), dtype=torch.long), 
#                  edge_idx_to_mask=None):

#         assert timesteps % 2 == 0 # This is required for the logic to work well when we split node and edge generation to be separate

#         if tokenwise:
#             xdiffusion = tokenwisediffusion.TokenWiseDiffusion(T=timesteps, K=x_classes, token_counts=node_type_counts_unnormalized[:-1], aux_lambda=0, 
#                                                             discr_num=30000)
#             ediffusion = tokenwisediffusion.TokenWiseDiffusion(T=timesteps, K=e_classes+1, token_counts=edge_type_counts_unnormalized[:], aux_lambda=0,
#                                                             discr_num=120000)
#         else:
#             xdiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=x_classes, aux_lambda=0)
#             ediffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=e_classes, aux_lambda=0)
#         ydiffusion = maskdiffusion.MaskDiffusion(T=timesteps, K=y_classes+1, aux_lambda=0)
#         self.Qts_x, self.Qt_bars_x = xdiffusion.return_all_transition_matrices(abs_state_position=abs_state_position_x)
#         self.Qts_e, self.Qt_bars_e = ediffusion.return_all_transition_matrices(abs_state_position=abs_state_position_e)
#         # Change so that half of generation is spent on edges and half is spent on nodes
#         self.Qts_x, self.Qt_bars_x = self.Qts_x[::2], self.Qt_bars_x[::2]
#         self.Qts_e, self.Qt_bars_e = self.Qts_e[::2], self.Qt_bars_e[::2]
#         # Put identity matrices where nothing happens on nodes first
#         identity_matrices_x = torch.cat([torch.eye(x_classes)[None,:,:] for _ in range(timesteps//2)], 0).to(device)
#         # And the absorbing state transition for the edge transitions in the second part.
#         # Note: The absorbing state was here defined to be the first state, 'no-edge' one
#         absorbing_matrices_e = torch.cat([torch.ones(e_classes, e_classes)[None,:,:] for _ in range(timesteps//2)], 0).to(device)
#         absorbing_matrices_e[:, :, 1:] = 0
#         self.Qts_x = torch.cat([identity_matrices_x, self.Qts_x], axis=0)
#         self.Qt_bars_x = torch.cat([identity_matrices_x, self.Qt_bars_x], axis=0)
#         self.Qts_e = torch.cat([self.Qts_e, absorbing_matrices_e], axis=0)
#         self.Qt_bars_e = torch.cat([self.Qt_bars_e, absorbing_matrices_e], axis=0)
#         self.Qts_y, self.Qt_bars_y = ydiffusion.return_all_transition_matrices()

#         self.x_limit = F.one_hot(torch.tensor([x_classes-1], dtype=torch.long), x_classes)[0].float() # remove batch dimension
#         self.e_limit = F.one_hot(torch.tensor([0], dtype=torch.long), e_classes)[0].float()
#         self.y_limit = F.one_hot(torch.tensor([y_classes], dtype=torch.long), y_classes+1)[0].float()

#         super().__init__(x_classes, e_classes, y_classes, diffuse_edges, node_idx_to_mask, edge_idx_to_mask)
