from abc import abstractmethod
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenericDiffusion():
    def __init__(self, T, K, aux_lambda):
        self.T = T
        self.K = K
        self.aux_lambda = aux_lambda
    
    @abstractmethod
    def forward_sample(self, x_0, ta):
        pass
    
    @abstractmethod
    def q_xt_to_mask_given_x0(self, x_0, t):
        pass
    
    @abstractmethod
    def q_xt_given_x0(self, x_0, t):
        pass
    
    @abstractmethod
    def q_xt_to_mask_given_xtm(self, x_tm, t):
        pass
    
    @abstractmethod
    def q_xt_given_xtm(self, x_t, x_tm, t):
        pass
    
    @abstractmethod
    def q_step_transpose_given_xt(self,x_t, t):
        pass
    
    def transition_matrix_xt_given_x0(self, t, abs_state_position=-1, drop_state=None):
        """Return the transition matrix from timestep 0 to timestep t."""
        transition_matrix = torch.zeros(self.K, self.K, dtype=torch.float, device=device)
        transition_matrix[0,0] = 1 # The absorbing state always stays in itself
        for k in range(1, self.K):
            prob_mask = self.q_xt_to_mask_given_x0(torch.tensor([k],device=device), torch.tensor([t], device=device))
            transition_matrix[k,0] = prob_mask
            transition_matrix[k,k] = 1 - prob_mask
        if abs_state_position == -1:
            perm = torch.tensor(list(range(1,self.K)) + [0], device=device)
        elif abs_state_position == 0:
            perm = torch.tensor(list(range(0,self.K)), device=device)
        else:
            perm = torch.tensor(list(range(1,abs_state_position+1)) + [0] + list(range(abs_state_position+1,self.K)), device=device)
        # How to join the absorbing state with some other state in the transition matrix?
        # ... I guess we can just first create a larger transition matrix with the additional absorbing state,
        # ... nah not so sure about that
        transition_matrix = transition_matrix[perm][:,perm]
        if drop_state is not None:
            transition_matrix = transition_matrix[torch.tensor(list(range(0,drop_state)) + list(range(drop_state+1,self.K)), device=device)][:,torch.tensor(list(range(0,drop_state)) + list(range(drop_state+1,self.K)), device=device)]
        return transition_matrix.to(device=device)
    
    def transition_matrix_xt_given_xtm(self, t, abs_state_position=-2, drop_state=None):
        """Return the transition matrix from timestep t-1 to timestep t."""
        transition_matrix = torch.zeros(self.K, self.K, dtype=torch.float,device=device)
        transition_matrix[0,0] = 1 # The abs->abs transition. Will be moved out of zero later on
        for k in range(1, self.K):
            prob_mask = self.q_xt_to_mask_given_xtm(torch.tensor([k], device=device), torch.tensor([t], device=device))
            # either move to the mask state
            transition_matrix[k,0] = prob_mask
            # or stay in the current state
            transition_matrix[k,k] = 1 - prob_mask
        if abs_state_position == -1:
            perm = torch.tensor(list(range(1,self.K)) + [0], device=device)
        elif abs_state_position == 0:
            perm = torch.tensor(list(range(0,self.K)), device=device)
        else:
            perm = torch.tensor(list(range(1,abs_state_position+1)) + [0] + list(range(abs_state_position+1,self.K)), device=device)
        transition_matrix = transition_matrix[perm][:,perm]
        if drop_state is not None:
            transition_matrix = transition_matrix[torch.tensor(list(range(0,drop_state)) + list(range(drop_state+1,self.K)),device=device)][:,torch.tensor(list(range(0,drop_state)) + list(range(drop_state+1,self.K)),device=device)]
        return transition_matrix.to(device=device)
    
    def return_all_transition_matrices(self, abs_state_position=-1, drop_state=None):
        Qts = []
        Qt_bars = []
        Qts.append(torch.eye(self.K-1 if drop_state else self.K, dtype=torch.float, device=device))
        Qt_bars.append(torch.eye(self.K-1 if drop_state else self.K, dtype=torch.float, device=device))
        for t in range(1, self.T+1):
            Qts.append(self.transition_matrix_xt_given_xtm(torch.tensor([t], device=device), abs_state_position, drop_state))
            Qt_bars.append(self.transition_matrix_xt_given_x0(torch.tensor([t], device=device), abs_state_position, drop_state))
        Qts = torch.cat([Qts[t][None,:,:] for t in range(self.T+1)], 0)
        Qt_bars = torch.cat([Qt_bars[t][None,:,:] for t in range(self.T+1)], 0)
        
        return Qts, Qt_bars
    
    def permute_transition_matrices(self, Qts, Qt_bars, perm):
        Qts = Qts[:,perm][:,:,perm]
        Qt_bars = Qt_bars[:,perm][:,:,perm]
        return Qts, Qt_bars

    @abstractmethod
    def q_posterior(self,x_t, x_0, t):
        pass
    
    @abstractmethod
    def log_reverse_param(self,x_t, x_0_logits, t):
        pass
    
    @abstractmethod
    def loss(self, x_t, x_0, x_0_logits, t):
        pass
    
    @abstractmethod
    def elbo(self,dataloader, model, device, num_batches = None):
        pass