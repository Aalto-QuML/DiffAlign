import logging

import torch
import torch.nn.functional as F

from diffalign.utils import graph
from diffalign.utils.diffusion import helpers

# A logger for this file
log = logging.getLogger(__name__)


class ELBOMixin:
    """Mixin class containing ELBO-related methods for DiscreteDenoisingDiffusion."""

    def compute_Lt_all(self, dense_true):
        '''
            Compute L_s terms: E_{q(x_t|x)} KL[q(x_s|x_t,x_0)||p(x_s|x_t)], with s = t-1
            But compute all of the terms, is this how we want the function to behave?
            To test this, would be nice to have a function for defining the transition matrices
            for different time steps
        '''

        device = dense_true.X.device
        true_X, true_E = dense_true.X, dense_true.E

        Lts = []

        assert self.T % self.cfg.diffusion.diffusion_steps_eval == 0, 'diffusion_steps_eval should be divisible by diffusion_steps'
        all_steps = list(range(self.cfg.diffusion.diffusion_steps_eval+1))
        eval_step_size = self.T // self.cfg.diffusion.diffusion_steps_eval
        steps_to_eval_here = all_steps[2:]

        pos_encodings = self.get_pos_encodings_if_relevant(dense_true)

        for idx, t in enumerate(steps_to_eval_here):
            t_int = torch.ones((true_X.shape[0], 1)).to(device)*t
            z_t, _ = self.apply_noise(dense_true, t_int=t_int, transition_model=self.transition_model_eval)
            z_t.y *= eval_step_size # Adjust the neural net input to the correct range

            z_t = graph.apply_mask(orig=dense_true, z_t=z_t,
                               atom_decoder=self.dataset_info.atom_decoder,
                               bond_decoder=self.dataset_info.bond_decoder,
                               mask_nodes=self.cfg.diffusion.mask_nodes,
                               mask_edges=self.cfg.diffusion.mask_edges,
                               return_masks=False)

            pred = self.forward(z_t=z_t, pos_encodings=pos_encodings)

            # compute q(x_{t-1}|x_t, x_0) for X and E
            Lt = self.compute_Lt(dense_true=dense_true, z_t=z_t, t=t_int, x_0_tilde_logit=pred,
                                 transition_model=self.transition_model_eval)
            Lt.to_device('cpu')
            Lts.append(Lt)

        return Lts

    def compute_Lt(self, dense_true, z_t, t, x_0_tilde_logit, transition_model, log=False):
        # t: a tensor of shape (bs, 1) with the time step
        assert t.shape[1]==1, 't should be a tensor of shape (bs, 1)'
        bs, n, v = dense_true.X.shape
        e = dense_true.E.shape[-1]
        device = z_t.X.device
        # t = t#z_t.y[...,0][...,None]
        s = t - 1

        Qt = transition_model.get_Qt(t, device=device)
        Qtb = transition_model.get_Qt_bar(t, device=device)
        Qsb = transition_model.get_Qt_bar(s, device=device)

        # compute q(x_{t-1}|x_t) = q(x_{t-1}|x_t, x_0)
        # This part should have the masking logic as well, to account for the corner case t=T & we don't have the masking state for conditioned signals
        q_s_given_t_0_X = helpers.compute_posterior_distribution(M=dense_true.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
        q_s_given_t_0_X = q_s_given_t_0_X.reshape(bs, n, v)
        q_s_given_t_0_E = helpers.compute_posterior_distribution(M=dense_true.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)
        q_s_given_t_0_E = q_s_given_t_0_E.reshape(bs, n, n, e)
        # TODO: check why q_s_given_t_0_E is not symmetric
        q_s_given_t_0 = z_t.get_new_object(X=q_s_given_t_0_X, E=q_s_given_t_0_E)
        #q_s_given_t_0 = graph.PlaceHolder(X=q_s_given_t_0_X, E=q_s_given_t_0_E, y=z_t.y, node_mask=z_t.node_mask)

        # compute p(x_{t-1}|x_t) = \sum_{\tilde{x_0}} p(x_{t-1}|x_t, \tilde{x_0}) * p(\tilde{x_0}|x_t)
        # comes down to replacing x_0 with the prediction x_0_tilde

        if log==False:
            x_0_tilde = z_t.get_new_object(X=F.softmax(x_0_tilde_logit.X, dim=-1), E=F.softmax(x_0_tilde_logit.E, dim=-1))
            #x_0_tilde = graph.PlaceHolder(X=F.softmax(x_0_tilde_logit.X, dim=-1), E=F.softmax(x_0_tilde_logit.E, dim=-1), y=z_t.y, node_mask=z_t.node_mask)
            p_s_given_t_X = helpers.compute_posterior_distribution(M=x_0_tilde.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
            p_s_given_t_X = p_s_given_t_X.reshape(bs, n, v)
            p_s_given_t_E = helpers.compute_posterior_distribution(M=x_0_tilde.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)
            p_s_given_t_E = p_s_given_t_E.reshape(bs, n, n, e)
            p_s_given_t = z_t.get_new_object(X=p_s_given_t_X, E=p_s_given_t_E)
            #p_s_given_t = graph.PlaceHolder(X=p_s_given_t_X, E=p_s_given_t_E, y=z_t.y, node_mask=z_t.node_mask)
            p_s_given_t = graph.apply_mask(orig=dense_true, z_t=p_s_given_t,
                                            atom_decoder=self.dataset_info.atom_decoder,
                                            bond_decoder=self.dataset_info.bond_decoder,
                                            mask_nodes=self.cfg.diffusion.mask_nodes,
                                            mask_edges=self.cfg.diffusion.mask_edges,
                                            node_states_to_mask=self.cfg.diffusion.node_states_to_mask,
                                            edge_states_to_mask=self.cfg.diffusion.edge_states_to_mask)
        else:
            log_p_s_given_t_X = helpers.compute_posterior_distribution(M=x_0_tilde_logit.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X, log=True)
            log_p_s_given_t_X = log_p_s_given_t_X.reshape(bs, n, v)
            log_p_s_given_t_E = helpers.compute_posterior_distribution(M=x_0_tilde_logit.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E, log=True)
            log_p_s_given_t_E = log_p_s_given_t_E.reshape(bs, n, n, e)
            log_p_s_given_t = z_t.get_new_object(X=log_p_s_given_t_X, E=log_p_s_given_t_E)
            #log_p_s_given_t = graph.PlaceHolder(X=log_p_s_given_t_X, E=log_p_s_given_t_E, y=z_t.y, node_mask=z_t.node_mask)
            log_p_s_given_t = graph.apply_mask(orig=dense_true, z_t=log_p_s_given_t,
                                                atom_decoder=self.dataset_info.atom_decoder,
                                                bond_decoder=self.dataset_info.bond_decoder,
                                                mask_nodes=self.cfg.diffusion.mask_nodes,
                                                mask_edges=self.cfg.diffusion.mask_edges,
                                                node_states_to_mask=self.cfg.diffusion.node_states_to_mask,
                                                edge_states_to_mask=self.cfg.diffusion.edge_states_to_mask,
                                                as_logits=True)

        q_s_given_t_0 = graph.apply_mask(orig=dense_true, z_t=q_s_given_t_0,
                                         atom_decoder=self.dataset_info.atom_decoder,
                                         bond_decoder=self.dataset_info.bond_decoder,
                                         mask_nodes=self.cfg.diffusion.mask_nodes,
                                         mask_edges=self.cfg.diffusion.mask_edges,
                                         node_states_to_mask=self.cfg.diffusion.node_states_to_mask,
                                         edge_states_to_mask=self.cfg.diffusion.edge_states_to_mask)

        # compute KL(true||pred) = KL(target||input)
        if log == False:
            kl_x = F.kl_div(input=(p_s_given_t.X+self.eps).log(), target=q_s_given_t_0.X, reduction='none').sum(-1)
            kl_e = F.kl_div(input=(p_s_given_t.E+self.eps).log(), target=q_s_given_t_0.E, reduction='none').sum(-1)
        else:
            kl_x = F.kl_div(input=torch.log_softmax(log_p_s_given_t.X, -1), target=q_s_given_t_0.X, reduction='none').sum(-1)
            kl_e = F.kl_div(input=torch.log_softmax(log_p_s_given_t.E, -1), target=q_s_given_t_0.E, reduction='none').sum(-1)

        Lt = z_t.get_new_object(X=kl_x, E=kl_e)
        #Lt = graph.PlaceHolder(X=kl_x, E=kl_e, node_mask=dense_true.node_mask, y=z_t.y)

        return Lt

    def compute_L1(self, dense_true, pos_encodings=None):
        device = dense_true.X.device
        t_int = torch.ones((dense_true.X.shape[0],1), device=device)

        z_1, _ = self.apply_noise(dense_true, t_int=t_int, transition_model=self.transition_model_eval)

        z_1 = graph.apply_mask(orig=dense_true, z_t=z_1,
                            atom_decoder=self.dataset_info.atom_decoder,
                            bond_decoder=self.dataset_info.bond_decoder,
                            mask_nodes=self.cfg.diffusion.mask_nodes,
                            mask_edges=self.cfg.diffusion.mask_edges,
                            return_masks=False)

        assert self.T % self.cfg.diffusion.diffusion_steps_eval == 0, 'diffusion_steps_eval should be divisible by diffusion_steps'
        eval_step_size = self.T // self.cfg.diffusion.diffusion_steps_eval
        z_1.y *= eval_step_size # Adjust the neural net input to the correct range

        pred0 = self.forward(z_t=z_1, pos_encodings=pos_encodings)
        pred0 = graph.apply_mask(orig=dense_true, z_t=pred0,
                                 atom_decoder=self.dataset_info.atom_decoder,
                                 bond_decoder=self.dataset_info.bond_decoder,
                                 mask_nodes=self.cfg.diffusion.mask_nodes,
                                 mask_edges=self.cfg.diffusion.mask_edges,
                                 node_states_to_mask=self.cfg.diffusion.node_states_to_mask,
                                 edge_states_to_mask=self.cfg.diffusion.edge_states_to_mask,
                                 as_logits=True)
        pred0.X = F.log_softmax(pred0.X,-1)
        pred0.E = F.log_softmax(pred0.E,-1)

        loss_term_0 = helpers.reconstruction_logp(orig=dense_true, pred_t0=pred0)

        return loss_term_0

    def kl_prior(self, dense_true):
        device = dense_true.X.device

        X, E = dense_true.X, dense_true.E
        bs, n, v, e = X.shape[0], X.shape[1], X.shape[-1], E.shape[-1]

        # compute p(x_T)
        Ts = self.T*torch.ones((bs,1), device=device)
        Qtb = self.transition_model.get_Qt_bar(Ts, device)

        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        prob = dense_true.get_new_object(X=probX, E=probE)
        #prob = graph.PlaceHolder(X=probX, E=probE, y=discrete_true.y, node_mask=dense_true.node_mask)

        # compute q(x_T)
        limitX = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(X)
        limitE = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(E)
        limit = dense_true.get_new_object(X=limitX, E=limitE)
        #limit = graph.PlaceHolder(X=limitX, E=limitE, y=discrete_true.y, node_mask=dense_true.node_mask)

        limit = graph.apply_mask(orig=dense_true, z_t=limit,
                                 atom_decoder=self.dataset_info.atom_decoder,
                                 bond_decoder=self.dataset_info.bond_decoder,
                                 mask_nodes=self.cfg.diffusion.mask_nodes,
                                 mask_edges=self.cfg.diffusion.mask_edges,
                                 node_states_to_mask=self.cfg.diffusion.node_states_to_mask,
                                 edge_states_to_mask=self.cfg.diffusion.edge_states_to_mask)

        prob = graph.apply_mask(orig=dense_true, z_t=prob,
                                 atom_decoder=self.dataset_info.atom_decoder,
                                 bond_decoder=self.dataset_info.bond_decoder,
                                 mask_nodes=self.cfg.diffusion.mask_nodes,
                                 mask_edges=self.cfg.diffusion.mask_edges,
                                 node_states_to_mask=self.cfg.diffusion.node_states_to_mask,
                                 edge_states_to_mask=self.cfg.diffusion.edge_states_to_mask)

        kl_prior_ = helpers.kl_prior(prior=prob, limit=limit, eps=self.eps)

        return kl_prior_

    @torch.no_grad()
    def get_elbo_of_data(self, dataloader, n_samples, device):
        """
        Computes the negative Evidence Lower Bound (ELBO) of the model on a given dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object that provides batches of data.
            n_samples (int): The number of samples to use for the ELBO estimation.

        Returns:
            float: The estimated ELBO of the model on the given dataset.
        """

        # For simplicity, let's only do full batches here, which may result in more samples than n_samples
        # because of the last batch
        batch_size = graph.get_batch_size_of_dataloader(dataloader)
        num_batches = max(n_samples//batch_size+int(n_samples%batch_size>0),1)
        assert num_batches<=len(dataloader), 'ELBO: testing more batches than is available in the dataset.'
        log.info(f"Num of batches needed for ELBO estimation: {num_batches}")
        total_elbo = 0
        dataiter = iter(dataloader)
        for _ in range(num_batches):
            data = next(dataiter)
            data = data.to(device)
            dense_true = graph.to_dense(data=data).to_device(device)
            elbo, _, _ = self.elbo(dense_true)
            total_elbo += elbo

        total_elbo /= num_batches

        # returning negative elbo as an upper bound on NLL
        return total_elbo

    def elbo_batch_quick(self, dense_true, pred, z_t, lambda_E=1.0, avg_over_batch=True):
        """
        Computes an estimator for the variational lower bound, but sampled such that we only
        get a single timestep t for each batch element. This makes it possible to train the model
        as well.

        input:
           discrete_true: a batch of data in discrete format (batch_size, n, total_features)
            z_t: sampled data at some timestep t (containts that in z_t.y)
            lambda_E: weight for the E term in the loss
            pred: the prediction of the model for the given z_t
            avg_over_batch: whether to average over the batch or not

        output:
            (float) the ELBO value of the given data batch.
        """
        t = z_t.y
        device = dense_true.X.device

        # Prior term
        # pred = self.forward(z_t=z_t)
        # If the transition matrix goes to identity as t->0, then this works for all steps, including t=1
        term_t = self.compute_Lt(dense_true=dense_true, z_t=z_t, t=t, x_0_tilde_logit=pred,
                               transition_model=self.transition_model, log=True) # Placeholder object, with, e.g., X of shape (batch_size, n, total_features)
        #loss *= self.cfg.diffusion.diffusion_steps # scale the estimator to the full ELBO
        kl_prior = self.kl_prior(dense_true) # Should be zero

        # TODO: This is not really quite right... the z_t is sampled from noise, the reconstruction loss is now just another CE loss
        pred.X, pred.E = F.log_softmax(pred.X, dim=-1), F.log_softmax(pred.E, dim=-1)
        term_1 = helpers.reconstruction_logp(orig=dense_true, pred_t0=pred)
        term_1.X, term_1.E = -term_1.X, -term_1.E

        # Manually weight the E term for training
        term_t.E, term_1.E, kl_prior.E = term_t.E * lambda_E, term_1.E * lambda_E, kl_prior.E * lambda_E

        # Combine the loss 1 term and the t terms
        terms_t_1 = term_t.get_new_object(X=torch.zeros_like(term_1.X), E=torch.zeros_like(term_1.E))
        #terms_t_1 = graph.PlaceHolder(X = torch.zeros_like(term_1.X), E = torch.zeros_like(term_1.E), y = term_t.y, node_mask = term_t.node_mask)
        expanded_t = t.repeat([1, dense_true.X.shape[1]])
        terms_t_1.X[expanded_t==1], terms_t_1.E[expanded_t==1] = term_1.X[expanded_t==1], term_1.E[expanded_t==1]
        terms_t_1.X[expanded_t!=1], terms_t_1.E[expanded_t!=1] = term_t.X[expanded_t!=1], term_t.E[expanded_t!=1]

        # Ignore padding & conditioning nodes in the averaging of the losses
        _, mask_X, mask_E = graph.apply_mask(orig=dense_true, z_t=z_t,
                                             atom_decoder=self.dataset_info.atom_decoder,
                                             bond_decoder=self.dataset_info.bond_decoder,
                                             mask_nodes=self.cfg.diffusion.mask_nodes,
                                             mask_edges=self.cfg.diffusion.mask_edges,
                                             return_masks=True)

        terms_t_1 = helpers.mean_without_masked(graph_obj=terms_t_1, mask_X=mask_X, mask_E=mask_E,
                                                diffuse_edges=self.cfg.diffusion.diffuse_edges,
                                                diffuse_nodes=self.cfg.diffusion.diffuse_nodes,
                                                avg_over_batch=avg_over_batch)
        kl_prior = helpers.mean_without_masked(graph_obj=kl_prior, mask_X=mask_X, mask_E=mask_E,
                                               diffuse_edges=self.cfg.diffusion.diffuse_edges,
                                               diffuse_nodes=self.cfg.diffusion.diffuse_nodes,
                                               avg_over_batch=avg_over_batch)

        elbo = terms_t_1 * self.cfg.diffusion.diffusion_steps + kl_prior

        return elbo

    def elbo(self, dense_true, avg_over_batch=True):
        """
        Computes an estimator for the variational lower bound.

        input:
           discrete_true: a batch of data in discrete format (batch_size, n, total_features)

        output:
            (float) the ELBO value of the given data batch.
       """
        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        device = dense_true.X.device
        kl_prior = self.kl_prior(dense_true)
        kl_prior.to_device('cpu') # move everything to CPU to avoid memory issues when computing all the Lt terms

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt_all(dense_true)
        for loss_t in loss_all_t:
            loss_t.E = loss_t.E * self.cfg.diffusion.lambda_test

        # TODO: Make sure that the extra conditioning terms are zeroed out! -> probably they are but should make sure
        _, mask_X, mask_E = graph.apply_mask(orig=dense_true, z_t=dense_true,
                                             atom_decoder=self.dataset_info.atom_decoder,
                                             bond_decoder=self.dataset_info.bond_decoder,
                                             mask_nodes=self.cfg.diffusion.mask_nodes,
                                             mask_edges=self.cfg.diffusion.mask_edges,
                                             return_masks=True)
        mask_X = mask_X.to('cpu')
        mask_E = mask_E.to('cpu')

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        # should be equivalent to CE?
        loss_0s = []

        pos_encodings = self.get_pos_encodings_if_relevant(dense_true)

        for i in range(self.cfg.test.loss_0_repeat):
            loss_term_0 = self.compute_L1(dense_true=dense_true, pos_encodings=pos_encodings)
            loss_term_0.to_device('cpu')
            loss_term_0.E *= self.cfg.diffusion.lambda_test
            loss_0s.append(helpers.mean_without_masked(graph_obj=loss_term_0,
                                                       mask_X=mask_X, mask_E=mask_E,
                                                       diffuse_edges=self.cfg.diffusion.diffuse_edges,
                                                       diffuse_nodes=self.cfg.diffusion.diffuse_nodes,
                                                       avg_over_batch=avg_over_batch))

        loss_0_per_dim = sum(loss_0s)/self.cfg.test.loss_0_repeat
        # normalize: combine X and E as elements of equal value, ignore padding nodes/edges, divide by number of all elements
        kl_prior_per_dim = helpers.mean_without_masked(graph_obj=kl_prior, mask_X=mask_X, mask_E=mask_E,
                                                       diffuse_edges=self.cfg.diffusion.diffuse_edges,
                                                       diffuse_nodes=self.cfg.diffusion.diffuse_nodes,
                                                       avg_over_batch=avg_over_batch)

        loss_t_per_dim = sum([helpers.mean_without_masked(graph_obj=loss_t, mask_X=mask_X, mask_E=mask_E,
                                                          diffuse_edges=self.cfg.diffusion.diffuse_edges,
                                                          diffuse_nodes=self.cfg.diffusion.diffuse_nodes,
                                                          avg_over_batch=avg_over_batch) for loss_t in loss_all_t])

        if len(loss_all_t)==0: loss_t_per_dim = torch.empty_like(kl_prior_per_dim, dtype=torch.float)
        # Combine terms
        vb =  kl_prior_per_dim + loss_t_per_dim - loss_0_per_dim

        return vb, loss_t_per_dim, loss_0_per_dim
