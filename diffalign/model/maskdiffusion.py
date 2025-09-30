from diffalign.model.generic_diffusion import GenericDiffusion
import torch
import math
import torch.nn as nn

class MaskDiffusion(GenericDiffusion):
    
    def forward_sample(self, x_0, t):
        """Sample from q(x_t|x_0)."""
        if x_0.size() != t.size():
            t = t.expand(x_0.size())
        r = torch.rand_like(x_0, dtype=torch.float, device=x_0.device, requires_grad=False)
        p_to_absorbing = t/self.T
        return x_0 * (r > p_to_absorbing)
    
    def q_xt_to_mask_given_x0(self, x_0, t):
        """Give probability of moving to absorbing state."""
        assert torch.all(x_0) # x_0 can't be the mask state
        return t.expand(x_0.size())/self.T
    
    def q_xt_given_x0(self, x_t, x_0, t):
        """Calculate probability of x_t given x_0, i.e., q(x_t=x_t'|x_0)."""
        assert torch.all(x_0)
        te = t.expand(x_t.size())
        prob_to_xt = torch.zeros_like(x_0, device=x_t.device, requires_grad=False, dtype=torch.float)
        prob_to_xt[x_t == 0] = te[x_t == 0]/self.T
        prob_to_xt[x_t == x_0] = 1 - te[x_t == x_0]/self.T
        return prob_to_xt
    
    def q_xt_given_xtm(self, x_t, x_tm, t):
        """
        Calculate probability of moving to x_t from x_{t-1}, i.e., q(x_t=x_t'|x_{t-1}=x_{t-1}').
        
        Arguments:
        x_t -- The state at time t
        x_tm -- The state at time t-1
        t -- The time in the diffusion process
        """
        assert x_t.size() == x_tm.size()
        te = t.expand(x_t.size())
        prob_to_x_t = torch.zeros_like(x_tm, dtype=torch.float, device=x_t.device, requires_grad=False)
        prob_to_x_t[(x_t == 0) & (x_tm == 0)] = 1.
        prob_to_x_t[(x_t == 0) & (x_tm != 0)] = 1/(self.T - te[(x_t == 0) & (x_tm != 0)] + 1)
        prob_to_x_t[(x_t != 0) & (x_t == x_tm)] = 1 - 1/(self.T - te[(x_t != 0) & (x_t == x_tm)] + 1)
        return prob_to_x_t
    
    def q_xt_to_mask_given_xtm(self, x_tm, t):
        """Calculate probability of moving to the absorbing state from x_{t-1}, on step t."""
        te = t.expand(x_tm.size())
        prob_to_mask = torch.zeros_like(x_tm, device=x_tm.device, requires_grad=False, dtype=torch.float)
        prob_to_mask[x_tm == 0] = 1
        prob_to_mask[x_tm != 0] = 1 / (self.T - te[x_tm != 0] + 1)

        return prob_to_mask

    def q_step_transpose_given_xt(self, x_t, t):
        """
        Return the probability vector x_t Q_t^T for each element in x_t.
        
        Arguments:
        x_t -- (seq_len, *batch_sizes) tensor composed of token ids
        t -- The corresponding diffusion time, as a pytorch tensor
        
        Returns:
        A (seq_len, *batch_sizes, K) tensor with probability distributions along the last dimension.
        """
        device = x_t.device
        te = t.expand(x_t.size())
        prob_from_mask = torch.ones(x_t.size() + torch.Size([self.K]), device=device, requires_grad=False)  * 1/(self.T - t[...,None] + 1)
        prob_from_mask[...,0] = 1.
        prob_from_others = torch.nn.functional.one_hot(x_t, self.K) * (1 - 1/(self.T - t[...,None] + 1))
        return prob_from_mask * (x_t == 0)[...,None] + prob_from_others * (x_t != 0)[...,None]
    
    def q_posterior(self, x_t, x_0, t):
        """Return q(x_{t-1}|x_t, x_0)."""
        device = x_t.device
        assert torch.all(x_0) # x_0 doesn't have zeros
        # numerator
        p1 = self.q_step_transpose_given_xt(x_t, t)# (batch_dims, K)
        p2_to_mask = self.q_xt_to_mask_given_x0(x_0, t-1)[...,None]# (batch_dims, 1)
        p2 = torch.nn.functional.one_hot(x_0, self.K) * (1-p2_to_mask) \
            + torch.nn.functional.one_hot(torch.zeros_like(x_0, dtype=torch.long), self.K) * p2_to_mask
        # denumerator
        x_0_to_mask_prob = self.q_xt_to_mask_given_x0(x_0, t)# (batch_dims,)
        denumerator_probs = torch.zeros_like(x_0, dtype=torch.float, device=device)
        denumerator_probs[x_t == 0] = x_0_to_mask_prob[x_t == 0]
        denumerator_probs[(x_t != 0) & (x_t == x_0)] = 1 - x_0_to_mask_prob[(x_t != 0) & (x_t == x_0)]
        denumerator_probs = denumerator_probs[...,None]
        eps = 1e-20
        return p1 * p2 / (denumerator_probs + eps)
    
    def log_reverse_param(self, x_t, x_0_logits, t):
        """
        Return p(x_{t-1}|x_t) given the neural network output logits that predict x_0.
        
        TODO: Reimplement this using q_posterior, but redo the normalization at a different point?
        """
        device = x_t.device
    
        # First calculate q(x_t|x_0) for all x_0. Do this by adding extra batch dim
        all_x0 = torch.arange(1, self.K, device=device, requires_grad=False).expand(x_t.size() + torch.Size([self.K - 1], device=device))
        prob_mask_xtm_all_x0 = self.q_xt_to_mask_given_x0(all_x0, (t-1)[...,None].expand(all_x0.size()))# (batchdims, K - 1)
        
        # Then turn this into (batch, K-1, 2) tensor that includes the probabilities q(x_t,x_{t-1}|x_0)
        # for x_{t-1} = mask and x_{t-1} = x_0
        q_xt_xtm_given_x0 = torch.cat(
            [self.q_xt_given_xtm(x_t[...,None].expand(all_x0.size()), torch.zeros_like(all_x0, dtype=torch.long, device=device, requires_grad=False), t[...,None].expand(all_x0.size()))[...,None],
                self.q_xt_given_xtm(x_t[...,None].expand(all_x0.size()), all_x0, t[...,None].expand(all_x0.size()))[...,None]], -1) \
                    * torch.cat([prob_mask_xtm_all_x0[...,None], 1-prob_mask_xtm_all_x0[...,None]],-1)
        
        # Then weight with the logits. Use something similar to the log-sum-exp trick for numerical stability
        max_logit = torch.max(x_0_logits[...,1:], -1).values
        weighted_logits = q_xt_xtm_given_x0 * (torch.exp(x_0_logits[...,1:,None]-max_logit[...,None,None]))
        xtm_indices = torch.cat([torch.zeros_like(all_x0, dtype=torch.long, device=device, requires_grad=False)[...,None], all_x0[...,None]], -1)
        
        # Then sum up along the x_0 dimension using scatter_add_
        summed_along_x0 = torch.zeros_like(x_0_logits, device=device, requires_grad=False)
        summed_along_x0.scatter_add_(dim=-1,
                index=xtm_indices.reshape(x_t.size() + torch.Size([(self.K-1)*2])),
                src=weighted_logits.reshape(x_t.size() + torch.Size([(self.K-1)*2])))
        
        # This was previously 1e-10, was a bit too large, made this behave in an incorrect way
        # ... although not sure what happens after we train this again
        eps = 1e-20
        log_of_sum = torch.log(summed_along_x0 + eps)
        denominator = torch.log(summed_along_x0.sum(-1, keepdim=True) + eps)
        
        return log_of_sum - denominator
    
    def loss(self, x_t, x_0, x_0_logits, t):
        """Return the part of the ELBO that is relevant for optimizing the neural network."""
        log_p = self.log_reverse_param(x_t, x_0_logits, t)
        q = self.q_posterior(x_t, x_0, t)
        # Should here be a sum as well somewhere? Sum over the last dimension?
        # -> yeah, then should be comparable with the other part
        main_loss = (-q*log_p).sum(-1).mean()
        aux_loss = self.aux_lambda * self.aux_loss(x_0, x_0_logits)
        return main_loss, aux_loss
    
    def aux_loss(self, x_0, x_0_logits):
        # Additional cross-entropy loss
        loss = nn.CrossEntropyLoss(reduction = 'mean')
        return loss(x_0_logits.view(-1,x_0_logits.shape[-1]), x_0.view(-1))
        
    def elbo(self, dataloader, model, device, num_batches = None):
        """
        Estimates the ELBO on a given data set in bits-per-token.
        
        Bits-per token is obtained by dividing the ELBO with math.log(2), which 
        transforms the value into base 2. 
        Perplexity is obtained from the same thing. 
        Note: -log p_theta(x_0|x_1) should also work with the same KL divergence
        formulas, since q(x_0|x_1,x_0') = delta(x_0;x_0'), but now implemented separately.
        For each batch, all t=1...T KL divs are evaluated.
        
        Arguments:
        dataloder -- A pytorch dataloader for a given data set (train/test)
        model -- The neural network model
        device -- The device
        num_batches -- How many batches to evaluate on? 
        
        Returns:
        An estimate of the ELBO in bits-per-dim, and the corresponding perplexity.
        """
        elbo = 0
        if num_batches == None:
            num_batches = len(dataloader)
        #model.eval()
        with torch.no_grad():
            i = 0
            for x_0 in dataloader:
                if i >= num_batches:
                    break
                x_0 = x_0.to(device)
                for t in range(self.T, 1, -1):
                    t = (torch.ones_like(x_0) * t).to(device)
                    #t = torch.randint(1,T,[x_0.size(1)],device=device).expand(x_0.size())
                    x_t = self.forward_sample(x_0, t)
                    x_0_logits = model(x_t)
                    x_0_logits[...,0] = -torch.inf
                    log_p = self.log_reverse_param(x_t, x_0_logits, t)
                    q = self.q_posterior(x_t, x_0, t)
                    # averaged over the batch (data set) and the amount of tokens (per token NLL)
                    # p(x_0|x_1) term has to be handled separately
                    eps = 1e-20
                    elbo += (q * (torch.log(q + eps) - log_p)).sum(-1).mean()
                
                # Special case for t == 1 (output log-likelihood)
                # ... actually the code above should work for t=1, at least for mask diffusion
                t = torch.ones_like(x_0).to(device)
                x_t = self.forward_sample(x_0, t)
                x_0_logits = model(x_t)
                x_0_logits[...,0] = -torch.inf
                log_p = self.log_reverse_param(x_t, x_0_logits, t)
                eps = 1e-20
                # Cross-entropy between the observed distribution and x_0 log-likelihoods
                elbo += -(torch.nn.functional.one_hot(x_0, self.K) * log_p).sum(-1).mean()
                i += 1
        elbo /= num_batches
        elbo /= math.log(2)
        perplexity = torch.exp(elbo)
        return elbo, perplexity
    
    def sample_reverse(self, model, seq_len, device, save_times=[], batch_size=1):
        """Sample sentences from the model."""
        x = torch.zeros(seq_len, batch_size, dtype=torch.long, device=device)
        saved_x = []
        with torch.no_grad():
            for t in range(self.T, 0, -1):
                if t in save_times:
                    saved_x.append(x.clone().detach())
                x_0_logits = model(x)
                log_p = self.log_reverse_param(x, x_0_logits, torch.tensor([t], device=device))
                K = log_p.shape[-1]
                x = torch.multinomial(torch.exp(log_p).reshape(-1,K), 1).reshape(seq_len, batch_size)
        if 0 in save_times:
            saved_x.append(x.clone().detach())
        return x, saved_x