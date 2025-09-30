from diffalign.model.generic_diffusion import GenericDiffusion
import torch
import torch.nn as nn
import math
import logging

class BlockTokenWiseDiffusion(GenericDiffusion):
    
    def __init__(self, T, K, token_counts, discr_num, groups, aux_lambda=0, device='cpu'):
        self.T = T
        self.K = K
        self.groups = groups
        assert len(groups) > 0
        groups_flattened = [item for sublist in groups for item in sublist]
        assert all(i in groups_flattened for i in range(1,K))
        assert all(i in range(0,K) for i in groups_flattened)
        if not isinstance(self.groups[0], torch.Tensor):
            for i in range(len(self.groups)):
                self.groups[i] = torch.tensor(self.groups[i], dtype=torch.long)
        if not len(groups[0]) == 1 and not groups[0][0] == 0:
            self.groups.insert(0, torch.tensor([0], dtype=torch.long))# Add the absorbing state to the first group
        self.group_min = torch.tensor([min(g) for g in groups])
        self.group_max = torch.tensor([max(g) for g in groups])
        self.K_groups = len(groups) - 1
        self.aux_lambda = aux_lambda
        self.counts = torch.ones(K-1, dtype=torch.long)# Estimate probabilities using a standard count of 1. Mainly in case we have a <unk> token that is not in the data
        self.counts += token_counts
        self.tokens = torch.tensor(range(1,K))
        self.probs = self.counts.double() / sum(self.counts.double())
        self.P = discr_num
        mi_matrix = self.mi_matrix_()
        self.t_to_step, self.t_to_mi_reduction = self.schedule_efficient_(mi_matrix)
        assert all(self.t_to_mi_reduction > 0)# If some t steps effectively do nothing, the code breaks (multiple t points to the last step in t_to_step)
        self.t_to_step = self.t_to_step.to(device)
        self.probs = self.probs.to(device)
        self.group_min = self.group_min.to(device)
        self.group_max = self.group_max.to(device)
        # TODO could make this a Pytorch module so the params could be moved with diffusion.to(device)
    
    def mi_matrix_(self):
        """
        Calculate a matrix of sub-token (and sub-group) mutual information reductions, 
        used for defining the MI schedule.
        
        Each token is divided into P steps that correspond to linearly interpolated probabilities 
        (between 0 and 1) of having that token been turned into the absorbing state. The tokens themselves
        are divided into K_groups groups, and within these groups, we perform standard absorbing state
        diffusion, one after the other. 
        
        Returns:
        mi_matrix -- A matrix of size K_groupsxP
        """
        mi_matrix = torch.zeros(self.K_groups, self.P, dtype=torch.double)
        neg_entropy = (self.probs * torch.log(self.probs)).sum()
        tokens_masked_so_far = torch.tensor([], dtype=torch.long)
        for k in range(1, self.K_groups+1):
            prop = torch.linspace(1/self.P,1,self.P,dtype=torch.double)
            eps = 1e-30
            
            # Here p has the probabilities of the previous tokens, :k, and the probability of the new ones
            # ... think about best way to explain this, this feels convoluted
            # (p * log(p/p.sum())).sum()
            # = (p * log(p)).sum() - (p * log(p.sum())).sum()
            # = (p * log(p)).sum() - p.sum() * log(p.sum())
            #(p * log(p)).sum(0) - p.sum(0) * log(p_sum_so_far)
            
            # Calculates the mutual information reduction in blocks of size block_size (for balance between memory and speed)
            # Create a list of lists of indices from [1,...,self.P] for blocks of size block_size
            block_size = 10000
            n_blocks = math.ceil(self.P/block_size)
            indices = [list(range(i*block_size, min((i+1)*block_size, self.P))) for i in range(n_blocks)]
            for block in indices:
                new_token_probs = self.probs[self.groups[k]-1,None]*prop[block] #(group_size, block_size)
                p_sum_so_far = self.probs[tokens_masked_so_far-1].sum() + new_token_probs.sum(0)
                mi_term_1 = (self.probs[tokens_masked_so_far-1] * torch.log(self.probs[tokens_masked_so_far-1] + eps)).sum() \
                     + (new_token_probs * torch.log(new_token_probs)).sum(0) # shape P vector
                mi_term_2 = - p_sum_so_far * torch.log(p_sum_so_far + eps)
                mis_for_token_prop = (mi_term_1 + mi_term_2) / neg_entropy
                mi_matrix[k-1,block] = mis_for_token_prop
                
            # for p in range(self.P):
            #     #new_token_prob = self.probs[self.groups[k]-1]*prop[p] #(group_size,). Note that <mask> is not in self.groups
            #     p_sum_so_far = self.probs[tokens_masked_so_far-1].sum() + probs_sum*prop[p] # (1,)
            #     mi_term_1 = (self.probs[tokens_masked_so_far-1] * torch.log(self.probs[tokens_masked_so_far-1] + eps)).sum() \
            #         + (probs*prop[p] * torch.log(probs*prop[p])).sum(0) # shape P vector
            #     mi_term_2 = - p_sum_so_far * torch.log(p_sum_so_far + eps)
            #     mis_for_token_prop[p] = (mi_term_1 + mi_term_2) / neg_entropy
            
            # new_token_probs = self.probs[self.groups[k]-1,None]*prop #group_size x P matrix. Note that <mask> is not in self.groups
            # p_sum_so_far = self.probs[tokens_masked_so_far-1].sum() + new_token_probs.sum(0) #group_size x P matrix
            # mi_term_1 = (self.probs[tokens_masked_so_far-1] * torch.log(self.probs[tokens_masked_so_far-1] + eps)).sum() \
            #         + (new_token_probs * torch.log(new_token_probs)).sum(0) # shape P vector
            # mi_term_2 = - p_sum_so_far * torch.log(p_sum_so_far + eps)
            # mis_for_token_prop = (mi_term_1 + mi_term_2) / neg_entropy
            
            tokens_masked_so_far = torch.cat([tokens_masked_so_far, self.groups[k]])
            
            # Old way to calculate that worked for sure:s
            # p = torch.cat([self.probs[:k,None].repeat(1,self.P), self.probs[k]*prop[None,:]], 0)
            #mis_for_token_prop = (p * torch.log(p / (p.sum(0,keepdim=True) + eps) + eps)).sum(0) / neg_entropy
            #print(mis_for_token_prop)
            # mi_matrix[k-1,:] = mis_for_token_prop
        # Normalize, numerical errors can accumulate so that mi_matrix[-1,-1] is not exactly 1.
        mi_matrix /= mi_matrix[-1,-1].item()
        return mi_matrix
    
    def schedule_efficient_(self, mi_matrix):
        num_steps = (self.K_groups) * self.P
        t_to_step = []
        t_to_mi_reduction = []
        prev_step = 0
        next_step = 0
        mi_flattened = mi_matrix.reshape(-1)
        mi_flattened = torch.cat([torch.tensor([0.], dtype=torch.double), mi_flattened]) # len num_steps + 1
        mi_added_flat = mi_flattened[1:] - mi_flattened[:-1] # len num_steps
        #torch.cat([torch.tensor([mi_flattened[0]]), mi_flattened[1:] - mi_flattened[:-1]])
        for t in range(self.T):
            mi_reduction_per_t = torch.tensor([0.], dtype=torch.double)
            if next_step + self.P < num_steps and mi_reduction_per_t < 1/self.T:# Figure out if we can start taking big steps
                while next_step + self.P < num_steps and mi_reduction_per_t < 1/self.T:
                    next_step += self.P# Try out a big step
                    mi_reduction_per_t = mi_flattened[next_step] - mi_flattened[prev_step]
                next_step -= self.P # backtrack
            mi_reduction_per_t = mi_flattened[next_step] - mi_flattened[prev_step]
            while next_step < num_steps - 1 and mi_reduction_per_t < 1/self.T:# -1 is important for the final step to be properly indexed!
                mi_reduction_per_t += mi_added_flat[next_step]
                next_step += 1
            t_to_step.append(next_step)
            t_to_mi_reduction.append(mi_reduction_per_t)
            #print(t, next_step, mi_reduction_per_t, num_steps)
            prev_step = next_step
        
        # Add a zero for time t=0, since the diffusion starts at t=1.
        t_to_step = torch.tensor([0] + [i for i in t_to_step])
        #print(t_to_step)
        return t_to_step, torch.tensor(t_to_mi_reduction)

    def expand_t_to_shape_(self, shape, t):
        te = t.clone()
        # This is a bit of hacky code to make sure that t is expanded to the right shape in different use-cases.
        if te.size() != shape:
            if len(te) == 1: # if t is just a number (torch.Tensor([t]))
                te = te.expand(shape)
            elif len(te.shape) == 1: # if t is same length as batch size (torch.Tensor([t1, t1, t3, ...]))
                te = te[None,:].expand(shape)
            if len(shape) == 3 and len(te.shape) == 2: # if we want dimension -> (seq_len, batch_size, K)
                te = te[...,None].expand(shape)
        return te

    def q_xt_to_mask_given_x0(self, x_0, t):
        """Probability q(x_t=mask|x_0) according to the MI schedule."""
        te = self.expand_t_to_shape_(x_0.size(), t) # shape (seq_len, batch_size)
        step = self.t_to_step[te] # shape (seq_len, batch_size)
        latest_group_index = (step // self.P) + 1 # 0 group is just the mask token, we start at the next one
        latest_group_min = self.group_min[latest_group_index]
        latest_group_max = self.group_max[latest_group_index]
        additional_prop = (step % self.P + 1) * 1 / self.P * (te != 0)
        prob_to_mask = torch.zeros_like(x_0, dtype=torch.float, device=x_0.device)
        prob_to_mask[x_0 < latest_group_min[0]] = 1.
        prob_to_mask[(x_0 >= latest_group_min[0]) & (x_0 < latest_group_max[-1] + 1)] = additional_prop[(x_0 >= latest_group_min[0]) & (x_0 < latest_group_max[-1] + 1)]
        return prob_to_mask

    def forward_sample(self, x_0, t):
        """Sample from the forward process"""
        q_xt_to_mask = self.q_xt_to_mask_given_x0(x_0, t) # shape x_0
        r = torch.rand_like(x_0, dtype=torch.float, device=x_0.device, requires_grad=False)
        return x_0 * (r > q_xt_to_mask) # mapped to zero, corresponding to <mask>
    
    def q_xt_given_x0(self, x_t, x_0, t):
        """Probability q(x_t=x_t'|x_0) according to the MI schedule."""
        q_xt_to_mask = self.q_xt_to_mask_given_x0(x_0, t) # shape x_0
        probs = torch.zeros_like(x_0, dtype=torch.float, device=x_0.device)
        probs[x_t == 0] = q_xt_to_mask[x_t == 0]
        probs[x_t == x_0] = 1 - q_xt_to_mask[x_t == x_0]
        return probs
    
    def q_xt_to_mask_given_xtm(self, x_tm, t):
        """
        Probability q(x_t=mask|x_{t-1}) according to the MI schedule.
        
        Representing this transition is a bit tricky. First we get the token that is midway to 
        being erased to the mask state at step t. Then, there are three main cases:
        x_{t-1} < latest_token at time t -> prob 1
        x_{t-1} == latest token at time t but not at time t-1 
                -> Output the total probability to move to the mask state, same as q(x_t=mask|x_0=x_{t-1}).
        x_{t-1} == latest token at time t and also at time t-1 
                -> Output the additional probability of moving to the mask state, such that the total 
                probability is q(x_t=mask|x_0=x_{t-1}) for this token. In other words,
                sum_x_{t-1}[ q(x_t=mask|x_{t-1}) q(x_{t-1}|x_0=x_{t-1}') ] = q(x_t=mask|x_0=x_{t-1}')
        x_{t-1} > latest token at time t -> prob 0
        """
        assert torch.all(t) # t doesn't have any zeros
        
        t = self.expand_t_to_shape_(x_tm.size(), t)
        
        # Q_t definition
        step_t = self.t_to_step[t]
        step_tm = self.t_to_step[t-1]
        latest_group_t = (step_t // self.P) + 1
        latest_group_min_t = self.group_min[latest_group_t]
        latest_group_max_t = self.group_max[latest_group_t]
        additional_prop_t = (step_t % self.P + 1) * 1 / self.P
        latest_group_tm = (step_tm // self.P) + 1
        latest_group_min_tm = self.group_min[latest_group_tm]
        latest_group_max_tm = self.group_max[latest_group_tm]
        additional_prop_tm = (step_tm % self.P + 1) * 1 / self.P
        
        # The rules
        # x_tm < latest_token at time t -> prob 1
        # x_tm == latest token at time t & x_tm == latest token at time t-1 -> (new_prop-old_prop)/(1-old_prop)
        # x_tm == latest token at time t & x_tm != latest token at time t-1 -> new_prop
        prob_to_mask = torch.zeros_like(x_tm, dtype=torch.float, device=x_tm.device)
        # ... need to think what happens if t=1! Special case (-> seems to work)
        prob_to_mask[x_tm < latest_group_min_t] = 1 # This is a bit problematic, assumes that x_tm is got 
        # by sampling from the forward process (add extra clause?)
        same_group = (x_tm >= latest_group_min_t) & (x_tm <= latest_group_max_t) \
                        & (x_tm >= latest_group_min_tm) & (x_tm <= latest_group_max_tm)
        prob_to_mask[same_group] = (additional_prop_t[same_group] - additional_prop_tm[same_group]) / (1 - additional_prop_tm[same_group])
        different_group = (x_tm >= latest_group_min_t) & (x_tm <= latest_group_max_t) \
                            & ~((x_tm >= latest_group_min_tm) & (x_tm <= latest_group_max_tm))
        prob_to_mask[different_group] = additional_prop_t[different_group]
        
        return prob_to_mask
    
    def q_xt_given_xtm(self, x_t, x_tm, t):
        """Calculate probability of moving to x_t from x_{t-1}, i.e., q(x_t=x_t'|x_{t-1}=x_{t-1}')."""
        q_xt_to_mask = self.q_xt_to_mask_given_xtm(x_tm, t) # shape x_0
        probs = torch.zeros_like(x_tm, dtype=torch.float, device=x_t.device)
        # q_xt_to_mask should give the correct probability if x_t == mask, always
        probs[x_t == 0] = q_xt_to_mask[x_t == 0]
        # We don't want to change this anymore if x_t already was the mask. This is only in the case where x_t == x_tm != mask
        probs[(x_t == x_tm) & (x_t != 0)] = 1 - q_xt_to_mask[(x_t == x_tm) & (x_t != 0)]
        return probs
    
    def q_step_transpose_given_xt(self, x_t, t):
        device = x_t.device
        te = self.expand_t_to_shape_(x_t.size(), t)
        
        x_tm = torch.zeros_like(x_t, dtype=torch.long, device=device)[...,None].expand(x_t.size() + torch.Size([self.K]))
        x_tm = torch.arange(0, self.K, device=device).expand(x_t.size() + torch.Size([self.K]))
        
        return self.q_xt_given_xtm(x_t[...,None].expand(x_t.size() + torch.Size([self.K])), x_tm, t)
    
    def q_posterior(self, x_t, x_0, t):
        """Return q(x_{t-1}|x_t, x_0)."""
        device = x_t.device
        assert torch.all(x_0) # x_0 doesn't have zeros
        # numerator
        p1 = self.q_step_transpose_given_xt(x_t, t)# (batch_dims, K)
        p2_to_mask = self.q_xt_to_mask_given_x0(x_0, t-1)[...,None]# (batch_dims, 1)
        p2 = torch.nn.functional.one_hot(x_0, self.K) * (1-p2_to_mask) \
            + torch.nn.functional.one_hot(torch.zeros_like(x_0, dtype=torch.long, device=device), self.K) * p2_to_mask
        # denumerator
        eps = 1e-20
        return p1 * p2 / ((p1 * p2).sum(-1, keepdim=True) + eps)
    
    def log_reverse_param(self, x_t, x_0_logits, t):
        device = x_t.device
        #t = self.expand_t_to_shape_(x_t.size(), t)# possibly unnecessary
        p1 = self.q_step_transpose_given_xt(x_t, t)# (batch_dims, K)
        p2_to_mask = self.q_xt_to_mask_given_x0(torch.arange(0, self.K, device=device).expand(x_t.size()\
                    + torch.Size([self.K])), t-1) # (batch_dims, K)
        
        max_logit = torch.max(x_0_logits[...,1:], -1).values
        # The following is possible because states can't change to others than themselves or mask
        p2_weighted_to_not_mask = torch.exp(x_0_logits[...,1:] - max_logit[...,None]) * (1-p2_to_mask[...,1:])
        p2_weighted_to_mask = torch.exp(x_0_logits[...,1:] - max_logit[...,None]) * p2_to_mask[...,1:]# (batch_dims, K-1)
        q_xt_xtm_given_x0_summed = torch.zeros_like(p1)
        q_xt_xtm_given_x0_summed[...,1:] = p1[...,1:] * p2_weighted_to_not_mask * (x_t != 0)[...,None] # case x_t != mask
        q_xt_xtm_given_x0_summed[...,0] += (x_t == 0) * p2_weighted_to_mask.sum(-1) #case x_t = mask and x_{t-1} = mask. 
        q_xt_xtm_given_x0_summed[...,1:] += (x_t == 0)[...,None] * p2_weighted_to_not_mask * p1[...,1:] # case x_t = mask and x_{t-1} != mask
        eps = 1e-20
        log_of_sum = torch.log(q_xt_xtm_given_x0_summed + eps)
        denominator = torch.log(q_xt_xtm_given_x0_summed.sum(-1, keepdim=True) + eps)
        return log_of_sum - denominator
    
    def log_reverse_param2(self, x_t, x_0_logits, t):
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
        main_loss = (-q*log_p).sum(-1).mean()
        aux_loss = self.aux_lambda * self.aux_loss(x_0, x_0_logits)
        return main_loss, aux_loss
    
    def aux_loss(self, x_0, x_0_logits):
        # Additional cross-entropy loss
        loss = nn.CrossEntropyLoss(reduction = 'mean')
        return loss(x_0_logits.view(-1,x_0_logits.shape[-1]), x_0.view(-1))
    
    def elbo(self, dataloader, model, device, num_batches = None):
        """TODO: This probably needs to be looked at again?
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