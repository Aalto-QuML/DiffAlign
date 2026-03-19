import copy

import torch
import torch.nn.functional as F

from diffalign.utils import graph
from diffalign.utils.diffusion import helpers


class SamplingMixin:
    @torch.no_grad()
    def sample_one_batch(self, device=None, n_samples=None, data=None, get_chains=False, get_true_rxns=False, inpaint_node_idx=None, inpaint_edge_idx=None):
        assert data!=None or n_samples!=None, 'You need to give either data or n_samples.'
        assert data!=None or self.cfg.diffusion.mask_nodes==None, 'You need to give data if the model is using a mask.'
        assert data!=None or get_true_rxns, 'You need to give data if you want to return true_rxns.'

        if data!=None:
            dense_data = data
            node_mask = dense_data.node_mask.to(device)
        else:
            n_nodes = self.node_dist.sample_n(n_samples, device)
            n_max = torch.max(n_nodes).item()
            arange = torch.arange(n_max, device=device).unsqueeze(0).expand(n_samples, -1)
            node_mask = arange < n_nodes.unsqueeze(1)
            dense_data = None

        # Absolutely make sure that the product ordering does not contain any information
        # ... but this may cause a distribution shift, since the pos encodings do use the data ordering, I think

        pos_encodings = self.get_pos_encodings_if_relevant(dense_data) # precalculate the pos encodings, since they are the same at each step

        z_t = helpers.sample_from_noise(limit_dist=self.limit_dist, node_mask=node_mask, T=self.T)
        if data is not None: z_t = dense_data.get_new_object(X=z_t.X, E=z_t.E, y=z_t.y)
        # The orig=z_t if data==None else dense_data covers the special case when we don't do any conditioning
        # TODO: Make sure that this works, e.g., when we condition on supernodes but not on edges
        z_t = graph.apply_mask(orig=z_t if data==None else dense_data, z_t=z_t,
                               atom_decoder=self.dataset_info.atom_decoder,
                               bond_decoder=self.dataset_info.bond_decoder,
                               mask_nodes=self.cfg.diffusion.mask_nodes,
                               mask_edges=self.cfg.diffusion.mask_edges)

        mask_X, mask_E = graph.fix_nodes_and_edges_by_idx(data=dense_data, node_idx=inpaint_node_idx,
                                                          edge_idx=inpaint_edge_idx)
        z_t.X[mask_X], z_t.E[mask_E] = dense_data.X[mask_X], dense_data.E[mask_E]
        z_t.X[~mask_X], z_t.E[~mask_E] = z_t.X[~mask_X], z_t.E[~mask_E]

        # if: # of rct nodes <= # of prod nodes + cfg.dataset.nb_dummy_nodes
        #... cut (# of rct nodes - # of prod nodes + cfg.dataset.nb_dummy_nodes) from beginning of vectors?
        # get: # of rct nodes, # of prod nodes from masked_z_t.split('SuNo')
        #... shld never have SuNo in rct side
        #... when conditioning on the number of rcts and rct-nodes, will have more than one SuNo, and no need to worry about this cut
        if not self.cfg.diffusion.diffuse_edges and data!=None: z_t.E = dense_data.E.clone()
        if not self.cfg.diffusion.diffuse_nodes and data!=None: z_t.X = dense_data.X.clone()

        if self.cfg.test.with_denoising:
            if get_chains: sample_chains, prob_s_chains, pred_0_chains = [], [], []

            # TODO: Actually self.T should be used here instead, after all.
            # Maybe could create two different transition matrix groups, one for eval and one for testing? -< and have the testing one still have to be adjusted manually with the neural network
            assert self.T % self.cfg.diffusion.diffusion_steps_eval == 0, 'diffusion_steps_eval should be divisible by diffusion_steps'
            all_steps = list(range(self.cfg.diffusion.diffusion_steps_eval+1)) #np.linspace(0, self.T, self.cfg.diffusion.diffusion_steps_eval+1).astype('int')
            eval_step_size = self.T // self.cfg.diffusion.diffusion_steps_eval
            t_steps = all_steps[1:]
            s_steps = all_steps[:-1]

            for i in reversed(range(len(t_steps))):
                t_int = t_steps[i]
                s_int = s_steps[i]

                s_array = s_int * torch.ones((z_t.X.shape[0], 1)).long().to(device)
                t_array = t_int * torch.ones((z_t.X.shape[0], 1)).long().to(device) #s_array + 1

                # z_t.y = s_array.clone().float()
                # NOTE 1: This was changed from s_array to t_array. This is how the model was trained, so should be t instead of s
                # NOTE 2: This is multiplied by eval_step_size so that the neural net output is consistent with the training data
                z_t.y = t_array.clone().float() * eval_step_size

                # compute p(x | z_t)
                pred = self.forward(z_t=z_t, pos_encodings=pos_encodings)

                # Temperature scaling
                pred = pred.get_new_object(X=torch.log_softmax(pred.X, -1) * self.cfg.diffusion.temperature_scaling_node, E=torch.log_softmax(pred.E, -1) * self.cfg.diffusion.temperature_scaling_edge)

                # compute p(z_s | z_t) (denoiser)
                # Note: no need to mask pred because computing denoiser proba is done independently for each node/edge
                prob_s = helpers.get_p_zs_given_zt(transition_model=self.transition_model_eval, t_array=t_array, pred=pred, z_t=z_t, return_prob=True)

                if self.cfg.diffusion.classifier_free_guidance:
                    assert data != None, 'Need conditioning data for classifier-free guidance'
                    _, mask_X, mask_E = graph.apply_mask(orig=dense_data, z_t=z_t,
                                                        atom_decoder=self.dataset_info.atom_decoder,
                                                        bond_decoder=self.dataset_info.bond_decoder,
                                                        mask_nodes=self.cfg.diffusion.mask_nodes,
                                                        mask_edges=self.cfg.diffusion.mask_edges, return_masks=True,
                                                        include_supernode=False)
                    mask_X_, mask_E_ = graph.fix_nodes_and_edges_by_idx(data=dense_data, node_idx=inpaint_node_idx,
                                                                      edge_idx=inpaint_edge_idx)
                    z_t.X[mask_X_], z_t.E[mask_E_] = dense_data.X[mask_X_], dense_data.E[mask_E_]
                    z_t.X[~mask_X_], z_t.E[~mask_E_] = z_t.X[~mask_X_], z_t.E[~mask_E_]
                    if self.cfg.diffusion.classifier_free_full_unconditioning and self.cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc':
                        z_t_no_cond = z_t.get_new_object(X=z_t.X*mask_X, E=z_t.E*mask_E, atom_map_numbers=torch.zeros_like(z_t.atom_map_numbers))
                    else:
                        z_t_no_cond = z_t.get_new_object(X=z_t.X*mask_X, E=z_t.E*mask_E)
                        #z_t_no_cond = graph.PlaceHolder(X=z_t.X * mask_X, E=z_t.E * mask_E, y=z_t.y, node_mask=z_t.node_mask, atom_map_numbers=z_t.atom_map_numbers)
                    pred_uncond = self.forward(z_t=z_t_no_cond, use_pos_encoding_if_applicable=torch.zeros(z_t.X.shape[0], dtype=torch.bool, device=device))
                    prob_s_uncond = helpers.get_p_zs_given_zt(transition_model=self.transition_model, t_array=t_array, pred=pred_uncond, z_t=z_t,
                                                              return_prob=True)
                    w = self.cfg.diffusion.classifier_free_guidance_weight
                    eps = 1e-30
                    prob_s.X = torch.softmax((1+w) * torch.log(prob_s.X + eps) - w * torch.log(prob_s_uncond.X + eps), -1)
                    prob_s.E = torch.softmax((1+w) * torch.log(prob_s.E + eps) - w * torch.log(prob_s_uncond.E + eps), -1)
                    prob_s.X[prob_s.X < 1e-28] = 0. # Keep sure that, e.g., supernode transitions are impossible. The numerics above doesn't guarantee that
                    prob_s.E[prob_s.E < 1e-28] = 0.

                if not self.cfg.diffusion.diffuse_edges and data!=None: prob_s.E = dense_data.E.clone()
                if not self.cfg.diffusion.diffuse_nodes and data!=None: prob_s.X = dense_data.X.clone()

                # save chains if relevant
                # need to mask here to keep consistency for plotting
                if get_chains and (s_int%self.cfg.train.log_every_t==0 or s_int==self.T-1):
                    # turn pred from logits to proba for plotting
                    pred.X = F.softmax(pred.X, dim=-1)
                    pred.E = F.softmax(pred.E, dim=-1)
                    # TODO: make this better (more generic): ignore SuNo predictions
                    pred.X[...,-1] = 0.
                    pred.X /= pred.X.sum(-1).unsqueeze(-1)

                    pred = graph.apply_mask(orig=z_t if data==None else dense_data, z_t=pred,
                                            atom_decoder=self.dataset_info.atom_decoder,
                                            bond_decoder=self.dataset_info.bond_decoder,
                                            mask_nodes=self.cfg.diffusion.mask_nodes,
                                            mask_edges=self.cfg.diffusion.mask_edges)
                    mask_X, mask_E = graph.fix_nodes_and_edges_by_idx(data=dense_data, node_idx=inpaint_node_idx, edge_idx=inpaint_edge_idx)
                    pred.X[mask_X], pred.E[mask_E] = dense_data.X[mask_X], dense_data.E[mask_E]
                    pred.X[~mask_X], pred.E[~mask_E] = pred.X[~mask_X], pred.E[~mask_E]
                    if not self.cfg.diffusion.diffuse_edges and data!=None: pred.E = dense_data.E.clone()
                    if not self.cfg.diffusion.diffuse_nodes and data!=None: pred.X = dense_data.X.clone()

                    # pred_0_chain = pred.slice_by_idx(idx=self.cfg.train.chains_to_save)

                    # save p(z_s | z_t)
                    prob_s = graph.apply_mask(orig=z_t if data==None else dense_data, z_t=prob_s,
                                              atom_decoder=self.dataset_info.atom_decoder,
                                              bond_decoder=self.dataset_info.bond_decoder,
                                              mask_nodes=self.cfg.diffusion.mask_nodes,
                                              mask_edges=self.cfg.diffusion.mask_edges)
                    mask_X, mask_E = graph.fix_nodes_and_edges_by_idx(data=dense_data, node_idx=inpaint_node_idx, edge_idx=inpaint_edge_idx)
                    prob_s.X[mask_X], prob_s.E[mask_E] = dense_data.X[mask_X], dense_data.E[mask_E]
                    prob_s.X[~mask_X], prob_s.E[~mask_E] = prob_s.X[~mask_X], prob_s.E[~mask_E]
                    if not self.cfg.diffusion.diffuse_edges and data!=None: prob_s.E = dense_data.E.clone()
                    if not self.cfg.diffusion.diffuse_nodes and data!=None: prob_s.X = dense_data.X.clone()

                    # prob_s_chain = prob_s.slice_by_idx(idx=self.cfg.train.chains_to_save)

                    # save the chain of the actual sample
                    # sample = z_t.slice_by_idx(idx=self.cfg.train.chains_to_save)

                    # The logic: We plot the samples starting from T, and the denoising/NN outputs starting at T-1, e.g., p(x_{T-1}|x_T)
                    sample_chains.append((s_int+1, z_t.mask(z_t.node_mask, collapse=True)))
                    prob_s_chains.append((s_int, prob_s))
                    pred_0_chains.append((s_int, pred))

                # sample from p(z_s | z_t)
                # Note: no need to mask pred because sampling is done independently for each node/edge
                z_t = helpers.sample_discrete_features(prob=prob_s)

                # sanity check
                assert (z_t.E==torch.transpose(z_t.E, 1, 2)).all(), 'E is not symmetric.'

                z_t = graph.apply_mask(orig=z_t if data==None else dense_data, z_t=z_t,
                                       atom_decoder=self.dataset_info.atom_decoder,
                                       bond_decoder=self.dataset_info.bond_decoder,
                                       mask_nodes=self.cfg.diffusion.mask_nodes,
                                       mask_edges=self.cfg.diffusion.mask_edges)
                mask_X, mask_E = graph.fix_nodes_and_edges_by_idx(data=dense_data, node_idx=inpaint_node_idx,
                                                                  edge_idx=inpaint_edge_idx)
                z_t.X[mask_X], z_t.E[mask_E] = dense_data.X[mask_X], dense_data.E[mask_E]
                z_t.X[~mask_X], z_t.E[~mask_E] = z_t.X[~mask_X], z_t.E[~mask_E]

                # TODO: Right now the code can only do edge-conditional generation if there is any conditioning at all
                if not self.cfg.diffusion.diffuse_edges and data!=None: z_t.E = dense_data.E.clone()
                if not self.cfg.diffusion.diffuse_nodes and data!=None: z_t.X = dense_data.X.clone()

        if get_chains:
            # Save also the final sample in sample_chains
            # sample = z_t.slice_by_idx(idx=self.cfg.train.chains_to_save)
            sample = copy.deepcopy(z_t)
            sample_chains.append((0, sample.mask(sample.node_mask, collapse=True)))

        if get_true_rxns:
            #z_t_disc = z_t.get_new_object()
            return (z_t.mask(sample.node_mask, collapse=True), sample_chains, prob_s_chains, pred_0_chains, dense_data)

        if get_chains:
            return (z_t.mask(sample.node_mask, collapse=True), sample_chains, prob_s_chains, pred_0_chains)

        return z_t
