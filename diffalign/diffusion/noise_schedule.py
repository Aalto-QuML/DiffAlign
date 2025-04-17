import torch
import math

class NoiseSchedule:
    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps
        self.betas = self._compute_betas()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
            
    def _compute_betas(self):
        """Implement in subclass for different schedules"""
        raise NotImplementedError
         
    def get_beta_t(self, t):
        """Get transition matrix for timestep t"""
        return self.betas[t]
     
    def get_alpha_t_bar(self, t):
        """Get cumulative transition matrix up to t"""
        return self.alpha_bars[t]

class LinearSchedule(NoiseSchedule):
    def _compute_betas(self):
        '''
        Linear schedule from 1e-4 to 0.02.
        '''
        return torch.linspace(1e-4, 0.02, self.num_timesteps)

class CosineSchedule(NoiseSchedule):
    '''
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    '''
    def _compute_betas(self, s: float = 0.008):
        '''
        Compute betas for cosine schedule. alphas and alpha_bars are computed in the parent class.
        '''
        steps = torch.arange(self.num_timesteps + 2, dtype=torch.float32) / self.num_timesteps
        alpha_bars = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return torch.clip(betas, 0, 0.999)