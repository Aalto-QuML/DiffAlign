import os
import pathlib
import logging
import time
import wandb
import pickle
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch.cuda.amp import autocast

from datetime import date

from diffalign_old.neuralnet.transformer_model_with_y import PositionalEmbedding
from diffalign_old.diffusion.noise_schedule import *
from diffalign_old.utils import graph, model_utils, mol
from diffalign_old.neuralnet.ema_pytorch import EMA
from diffalign_old.utils.diffusion import helpers
from diffalign_old.utils import diffusion_utils
import traceback

# A logger for this file
log = logging.getLogger(__name__)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Disable rdkit warnings

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]

# MAX_NODES_FOR_TRAINING = 200
MAX_NODES = 100

class DiscreteDenoisingDiffusion(nn.Module):
    def __init__(self, cfg, dataset_infos, node_type_counts_unnormalized=None, 
                 edge_type_counts_unnormalized=None, use_data_parallel=None,
                 denoiser=None):
        super().__init__()
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist
        if cfg.neuralnet.extra_features:
            # Hardcoded for now, based on one sample of extra features
            input_dims['X'] = input_dims['X'] + 8 
            input_dims['E'] = input_dims['E']
            input_dims['y'] = input_dims['y'] + 12
        self.cfg = cfg
        self.T = cfg.diffusion.diffusion_steps
        # self.Xdim_input = input_dims['X']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.output_dims = output_dims
        if self.cfg.neuralnet.increase_y_dim_for_multigpu:
            output_dims['y'] += 1 # TODO: Make this a bit less hackier: it makes it possible for the model to work with multiple GPUs. Somehow output_dims['y'] is used somewhere directly, but not sure where and it needs to be changed here
        self.ydim_output = output_dims['y'] 
        self.node_dist = nodes_dist

        self.pos_emb_module = PositionalEmbedding(cfg.neuralnet.hidden_dims['dx'], cfg.neuralnet.pos_emb_permutations)

        self.dataset_info = dataset_infos
        self.log_to_wandb = cfg.train.log_to_wandb

        self.eps = 1e-6
        self.log_every_steps= cfg.general.log_every_steps
        
        if denoiser is None:
            self.model = model_utils.choose_denoiser(cfg, input_dims, output_dims)
        else:
            self.model = denoiser
        self.transition_model, self.transition_model_eval, self.limit_dist = model_utils.choose_transition(cfg, self.dataset_info, node_type_counts_unnormalized, edge_type_counts_unnormalized)
        if use_data_parallel:
            #if torch.cuda.device_count() > 1 and cfg.neuralnet.use_all_gpus:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        log.info(cfg.neuralnet.use_ema)
        self.ema = None
        if cfg.neuralnet.use_ema:
            log.info("???? Using EMA")
            self.ema = EMA(self.model, beta=cfg.neuralnet.ema_decay, power=1)

    def training_step(self, data, i, device):
        dense_data = graph.to_dense(data=data).to_device(device)
        #print(f'batch dimensions: X={dense_data.X.shape}, E={dense_data.E.shape}, y={dense_data.y.shape}')
        node_count = dense_data.X.shape[-2]
        if node_count > self.cfg.train.max_nodes_for_training:
            log.info(f"Too many nodes for training: {node_count}")
            return None, None, None, None, None, None
        t_int = torch.randint(1, self.T+1, size=(len(data),1), device=device)
        # TODO: apply_noise sometimes noises out the edges too much: with t=1 and T=2, goes all blank
        z_t, _ = self.apply_noise(dense_data, t_int = t_int, transition_model=self.transition_model)
        z_t, outside_reactant_mask_nodes, \
            outside_reactant_mask_edges = graph.fix_others_than_reactant_to_original(self.cfg, 
                                                                                     z_t, 
                                                                                     dense_data)
        
        if torch.cuda.is_available() and self.cfg.train.use_mixed_precision: #and self.model.training:
            if self.cfg.train.use_bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.forward(z_t=z_t)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = self.forward(z_t=z_t)
        else:
            pred = self.forward(z_t=z_t)

        if self.cfg.train.loss=='ce':
            loss_X, loss_E, loss_atom_charges, loss_atom_chiral, loss_bond_dirs, loss = helpers.ce(self.cfg, pred=pred, dense_true=dense_data, lambda_E=self.cfg.diffusion.lambda_train[0], 
                                                                                                   log=(i % self.log_every_steps == 0) and self.log_to_wandb, 
                                                                                                   outside_reactant_mask_nodes=outside_reactant_mask_nodes, 
                                                                                                   outside_reactant_mask_edges=outside_reactant_mask_edges)
        elif self.cfg.train.loss=='vb':
            loss = self.elbo_batch_quick(dense_data, pred=pred, z_t=z_t, lambda_E=self.cfg.diffusion.lambda_train[0])
            loss_X, loss_E = torch.zeros((1,)), torch.zeros((1,))
        elif self.cfg.train.loss=='vbce':
            loss_X_ce, loss_E_ce, ce_loss = helpers.ce(self.cfg, pred=pred, dense_true=dense_data,
                                                        lambda_E=self.cfg.diffusion.lambda_train[0], 
                                                        log=(i % self.log_every_steps == 0) and self.log_to_wandb,
                                                        outside_reactant_mask_nodes=outside_reactant_mask_nodes, outside_reactant_mask_edges=outside_reactant_mask_edges)
            elbo = self.elbo_batch_quick(dense_data, pred=pred, z_t=z_t, lambda_E=self.cfg.diffusion.lambda_train[0])
            loss = elbo + self.cfg.diffusion.ce_lambda * ce_loss
            loss_X, loss_E = torch.zeros((1,)), torch.zeros((1,))
            
        return loss_X, loss_E, loss_atom_charges, loss_atom_chiral, loss_bond_dirs, loss 

    def apply_noise(self, dense_data, t_int, transition_model):
        """ 
            Sample noise and apply it to the data. 
            
            input:
                discrete_data: batch graph object with nodes and edges in discrete form.
                t_int: time step for noise.
            return: 
                (PlaceHolder) z_t.
        """
        X, E, atom_charges, atom_chiral, bond_dirs, y = dense_data.X, dense_data.E, dense_data.atom_charges, dense_data.atom_chiral, dense_data.bond_dirs, dense_data.y
        device = dense_data.X.device

        assert X.dim()==3, 'Expected X in batch format.'+\
               f' Got X.dim={X.dim()}, If using one example, add batch dimension with: X.unsqueeze(dim=0).'
        
        Qtb = transition_model.get_Qt_bar(t_int.cpu(), device=device) # (bs, dx_in, dx_out), (bs, de_in, de_out)
        
        # Qtb.X and Qtb.E should have batch dimension
        assert Qtb.X.dim()==3 and Qtb.E.dim()==3, 'Expected Qtb.X and Qtb.E to have ndim=3 ((bs, dx/de, dx/de)) respectively. '+\
                                                  f'Got Qtb.X.dim={Qtb.X.dim()} and Qtb.E.dim={Qtb.E.dim()}.'
        # both Qtb.X and Qtb.E should be row normalized
        assert (abs(Qtb.X.sum(dim=-1)-1.) < 1e-4).all()
        assert (abs(Qtb.E.sum(dim=-1)-1.) < 1e-4).all()

        # compute transition probabilities
        probE = E @ Qtb.E.unsqueeze(1) # (bs, n, n, de_out)
        probX = X @ Qtb.X  # (bs, n, dx_out)
        prob_atom_charges = atom_charges @ Qtb.atom_charges # (bs, n, n_charges)
        prob_atom_chiral = atom_chiral @ Qtb.atom_chiral # (bs, n, n_chiral)
        prob_bond_dirs = bond_dirs @ Qtb.bond_dirs.unsqueeze(1) # (bs, n, n, n_bond_dirs)
        
        prob_t = dense_data.get_new_object(X=probX, E=probE, atom_charges=prob_atom_charges, atom_chiral=prob_atom_chiral,
                                            bond_dirs=prob_bond_dirs, y=t_int.float()).mask(dense_data.node_mask)
        
        z_t = helpers.sample_categoricals_simple(prob=prob_t)
        z_t.mol_assignment = dense_data.mol_assignment
        z_t.atom_map_numbers = dense_data.atom_map_numbers
        z_t.node_mask = dense_data.node_mask
        z_t.pos_encoding = dense_data.pos_encoding

        assert (X.shape==z_t.X.shape) and (E.shape==z_t.E.shape), 'Noisy and original data do not have the same shape.'

        return z_t, prob_t
    
    def get_pos_encodings_if_relevant(self, z_t):
        # DEPRECATED
        if self.cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc' or 'with_y_stacked': # Precalculate the pos encs, since they are the same for each step in the loop
            pos_encodings = model_utils.pos_encoding_choose(self.cfg, self.pos_emb_module, z_t)
        else:
            pos_encodings = None
        return pos_encodings

    def forward_old(self, z_t, use_pos_encoding_if_applicable=None, pos_encodings=None):
        device = z_t.X.device

        # randomly permute the atom mappings to make sure we don't use them wrongly
        perm = torch.arange(z_t.atom_map_numbers.max().item()+1, device=device)[1:]
        perm = perm[torch.randperm(len(perm))]
        perm = torch.cat([torch.zeros(1, dtype=torch.long, device=device), perm])
        z_t.atom_map_numbers = perm[z_t.atom_map_numbers]
        
        if use_pos_encoding_if_applicable is None: # handles the case where no input given
            use_pos_encoding_if_applicable = torch.ones(z_t.X.shape[0], dtype=torch.bool, device=device)
        if self.cfg.diffusion.denoiser=='carbon':
            carbon_idx = mol.atom_types.index('C') + mol.atom_type_offset
            baseline_X = F.one_hot(torch.tensor(carbon_idx), num_classes=len(mol.atom_types) + mol.atom_type_offset).float()
            baseline_X = baseline_X*100
            baseline_X[baseline_X==0] = -100
            pred = z_t.get_new_object(X=baseline_X)
            #pred = graph.PlaceHolder(X=baseline_X, E=z_t.E.clone(), y=z_t.y, node_mask=z_t.node_mask).type_as(z_t.X).mask(z_t.node_mask)
            return pred
        elif self.cfg.diffusion.denoiser=='random-uniform':
            # logits shld just be any number that is thetrain same for the whole vector
            baseline_X = torch.ones_like(z_t.X, device=device)
            pred = z_t.get_new_object(X=baseline_X)
            #pred = graph.PlaceHolder(X=baseline_X, E=z_t.E.clone(), y=z_t.y, node_mask=z_t.node_mask).type_as(z_t.X).mask(z_t.node_mask)
            return pred
        elif self.cfg.diffusion.denoiser=='neuralnet':
            if self.cfg.neuralnet.extra_features:
                # t0 = time.time()
                with autocast(enabled=False):
                    z_t = self.compute_extra_data(z_t)
                # log.info(f"Time for extra features: {time.time() - t0}")
            if self.cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc':
                # assert self.cfg.diffusion.mask_nodes == 'atom_mapping'
                # TODO: Need to create the molecule assignments here as well (and pass it around everywhere...)
                # That is done by including it in the data objects created 
                assert z_t.mol_assignment is not None, 'molecule_assigments is None in forward()'
                # TODO: Here use_pos_encoding_if_applicable is now not a Boolean, but a Tensor
                # ... so just calcualte the positional encodings here and zero out if not applicable
                # if use_pos_encoding_if_applicable:
                if pos_encodings == None: 
                    with autocast(enabled=False):
                    # if pos encs weren't precalculated, e.g., in the sampling loop
                        pos_encodings = self.get_pos_encodings(z_t)
                # zero out the positional encoding if not applicable. This related to CLFG or something like that
                pos_encodings *= use_pos_encoding_if_applicable[:,None,None].to(pos_encodings.device).float()
                # pos_encodings = torch.zeros(z_t.X.shape[0], z_t.X.shape[1], self.pos_emb_module.dim, device=z_t.X.device)

                # else:
                #     pos_encodings = torch.zeros(z_t.X.shape[0], z_t.X.shape[1], self.pos_emb_module.dim, device=z_t.X.device) # for now this is fine but will have to change if we do the concatenation instead
                if self.cfg.neuralnet.use_ema and not self.training:
                    res = self.ema(z_t.X, z_t.E, z_t.y, None, None, None, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignment)
                else:
                    res = self.model(z_t.X, z_t.E, z_t.y, None, None, None, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignment)
            elif self.cfg.neuralnet.architecture=='with_y_stacked':
                # pos encodings can't be calculated here, since the dimensions are different for different GPUs (different part of batch) -> could also put them with the extra features, but oh well, this is faster as it parallelizes
                # if pos_encodings == None: 
                #     with autocast(enabled=False):
                #         pos_encodings = self.get_pos_encodings(z_t)
                # pos_encodings *= use_pos_encoding_if_applicable[:,None,None].to(pos_encodings.device).float()
                if self.cfg.neuralnet.use_ema and not self.training:
                    res = self.ema(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignment, use_pos_encoding_if_applicable, self.cfg.neuralnet.pos_encoding_type, self.cfg.neuralnet.num_lap_eig_vectors, self.cfg.dataset.atom_types)
                else:
                    res = self.model(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignment, use_pos_encoding_if_applicable, self.cfg.neuralnet.pos_encoding_type, self.cfg.neuralnet.num_lap_eig_vectors, self.cfg.dataset.atom_types)
            else:
                if self.cfg.neuralnet.use_ema and not self.training:
                    res = self.ema(z_t.X, z_t.E, z_t.y, z_t.node_mask)
                else:
                    res = self.model(z_t.X, z_t.E, z_t.y, z_t.node_mask)
            if isinstance(res, tuple):
                X, E, y, _, _, _, node_mask = res
                res = z_t.get_new_object(X=X, E=E, y=y, node_mask=node_mask)
                #res = graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask, atom_map_numbers=z_t.atom_map_numbers).mask(node_mask)
            return res
        else:
            assert f'Denoiser model not recognized. Value given: {self.cfg.diffusion.denoiser}. You need to choose from: uniform, carbon and neuralnet.'

    def forward(self, z_t):
        # randomly permute the atom mappings
        device = z_t.X.device
        perm = torch.arange(z_t.atom_map_numbers.max().item()+1, device=device)[1:]
        perm = perm[torch.randperm(len(perm))]
        perm = torch.cat([torch.zeros(1, dtype=torch.long, device=device), perm])
        z_t.atom_map_numbers = perm[z_t.atom_map_numbers]

        if self.cfg.neuralnet.extra_features:
            with autocast(enabled=False):
                z_t = self.compute_extra_data(z_t)

        # breakpoint()
        # res = model_utils.forward_choose_careful(self.cfg, self.model, self.ema, z_t, self.training)
        res = model_utils.forward_choose(self.cfg, self.model, self.ema, z_t, self.training, self.pos_emb_module)
        
        if isinstance(res, tuple):
            X, E, y, atom_charges, atom_chiral, bond_dirs, node_mask = res
            res = z_t.get_new_object(X=X, E=E, y=y, atom_charges=atom_charges, atom_chiral=atom_chiral, bond_dirs=bond_dirs, node_mask=node_mask)
        return res
    
    @torch.no_grad()
    def sample_one_batch_old(self, device=None, n_samples=None, data=None, get_chains=False, 
                             get_true_rxns=False, inpaint_node_idx=None, inpaint_edge_idx=None):
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

        pos_encodings = self.get_pos_encodings_if_relevant(dense_data) # precalculate the pos encodings, since they are the same at each step

        z_t = helpers.sample_from_noise(limit_dist=self.limit_dist, node_mask=node_mask, T=self.T)
        if data is not None: z_t = dense_data.get_new_object(X=z_t.X, E=z_t.E, y=z_t.y)
        # The orig=z_t if data==None else dense_data covers the special case when we don't do any conditioning
        # TODO: Make sure that this works, e.g., when we condition on supernodes but not on edges
        z_t = graph.apply_mask(cfg=self.cfg, orig=z_t if data==None else dense_data, z_t=z_t,
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
                    _, mask_X, mask_E = graph.apply_mask(self.cfg, orig=dense_data, z_t=z_t,
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
                    # w = self.cfg.diffusion.classifier_free_guidance_weight
                    # pred.X = (1+w) * torch.log_softmax(pred.X, -1) - w * torch.log_softmax(pred_uncond.X, -1)
                    # pred.E = (1+w) * torch.log_softmax(pred.X, -1) - w * torch.log_softmax(pred_uncond.E, -1)
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
                
                    pred = graph.apply_mask(self.cfg, orig=z_t if data==None else dense_data, z_t=pred,
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
                    prob_s = graph.apply_mask(self.cfg, orig=z_t if data==None else dense_data, z_t=prob_s,
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
                
                z_t = graph.apply_mask(self.cfg, orig=z_t if data==None else dense_data, z_t=z_t,
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

    @torch.no_grad()
    def sample_one_batch(self, device=None, n_samples=None, data=None, get_chains=False, get_true_rxns=False, inpaint_node_idx=None, inpaint_edge_idx=None):
        assert data!=None or n_samples!=None, 'You need to give either data or n_samples.'
        assert data!=None or self.cfg.diffusion.mask_nodes==None, 'You need to give data if the model is using a mask.'
        assert data!=None or get_true_rxns, 'You need to give data if you want to return true_rxns.'
        assert data!=None

        dense_data = data
        node_mask = dense_data.node_mask.to(device)

        z_t = helpers.sample_from_noise(limit_dist=self.limit_dist, node_mask=node_mask, T=self.T)
        z_t = z_t.to_device(device)
        z_t = dense_data.get_new_object(X=z_t.X, E=z_t.E, y=z_t.y, atom_charges=z_t.atom_charges, atom_chiral=z_t.atom_chiral, bond_dirs=z_t.bond_dirs)
        z_t,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, z_t, dense_data, include_supernode=self.cfg.dataset.add_supernodes)
        z_t,_,_ = graph.fix_nodes_and_edges_by_idx(z_t, data=dense_data, node_idx=inpaint_node_idx,
                                                          edge_idx=inpaint_edge_idx)
        
        # import src.utils.data_utils as data_utils
        # breakpoint()
        # z_t = data_utils.reactant_initialization_based_only_on_product_data(self.cfg, z_t)

        # Recreate the z_t object with new atom_map_numbers, mol_assignments and pos_encodings to avoid any possible data leaks from them.
        # z_t_new = z_t.get_new_object()
        # z_t_new.atom_map_numbers[:] = 0
        # z_t_new.mol_assignment[:] = 0
        # if z_t_new.pos_encoding is not None:
        #     z_t_new.pos_encoding[:] = 0
        # for i in range(z_t.X.shape[0]):
        #     # start of the product nodes
        #     prod_nodes = (z_t.mol_assignment[i].max().item() == z_t.mol_assignment[i]).nonzero().flatten()
        #     start_prod_idx = prod_nodes[0].item()
        #     size_of_prod = len(prod_nodes) # prod_nodes[-1].item() - start_prod_idx + 1
        #     # number of reactant nodes (plus supernode if applicable)
        #     n_reactant_nodes_plus_optional_supernode = size_of_prod + self.cfg.dataset.nb_rct_dummy_nodes + int(self.cfg.dataset.add_supernodes)
        #     # check that this is actually how many reactant nodes there are
        #     if start_prod_idx != n_reactant_nodes_plus_optional_supernode:
        #         assert n_reactant_nodes_plus_optional_supernode < start_prod_idx, 'The following code only handles the case where the reactants cannot be generated due to not enough dummy nodes'
        #         # try:
        #         #     from mpi4py import MPI
        #         #     comm = MPI.COMM_WORLD
        #         #     mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        #         #     mpi_rank = comm.Get_rank()
        #         #     log.info(f"Error in mpi_rank {mpi_rank} and cfg.test.condition_index {self.cfg.test.condition_index}. \n start_prod_idx: {start_prod_idx}, size_of_prod: {size_of_prod}, n_reactant_nodes_plus_optional_supernode: {n_reactant_nodes_plus_optional_supernode}")
        #         # except ImportError: # mpi4py is not installed, for local experimentation
        #         #     MPI = None
        #         #     log.warning("mpi4py not found. MPI will not be used.")
                
        #         # redo the size of the entire graph
        #         extra_nodes_in_graph = start_prod_idx - n_reactant_nodes_plus_optional_supernode
        #         z_t_new.drop_n_first_nodes(extra_nodes_in_graph)
        #         z_t.drop_n_first_nodes(extra_nodes_in_graph)
        #         # dense_data = dense_data.get_new_object()
        #         # also drop the nodes from dense_data (let's hope this doesn't screw it up badly)
        #         dense_data.drop_n_first_nodes(extra_nodes_in_graph)
                
        #         prod_nodes = (z_t.mol_assignment[i].max().item() == z_t.mol_assignment[i]).nonzero().flatten()
        #         start_prod_idx = prod_nodes[0].item()
        #         # if get_true_rxns:
        #         #     return z_t.mask(z_t.node_mask, collapse=True), [(1,z_t)], [(1,z_t)], [(1,z_t)], dense_data
        #         # elif get_chains:
        #         #     return z_t.mask(z_t.node_mask, collapse=True), [(1,z_t)], [(1,z_t)], [(1,z_t)]
        #         # else:
        #         #     return z_t  # if this occurs for whatever reason in our code, don't even try
        #         # 'The start of the product nodes should be the number of reactant nodes.'
        #     # set the atom mappings on the product side and reactant side to 1...number of product nodes
        #     z_t_new.atom_map_numbers[i, start_prod_idx:start_prod_idx+size_of_prod] = torch.arange(1, size_of_prod+1, device=device)
        #     z_t_new.atom_map_numbers[i, :size_of_prod] = torch.arange(1, size_of_prod+1, device=device)
        #     # don't move the mol_assignment in the reactant side just to be sure that we can't use information about the reactant node sizes
        #     z_t_new.mol_assignment[i, start_prod_idx:] = z_t.mol_assignment[i, start_prod_idx:]

        # z_t = z_t_new

        if get_chains: sample_chains, prob_s_chains, pred_0_chains = [], [], []
        
        print(f'self.T {self.T}\n')
        print(f'self.cfg.diffusion.diffusion_steps_eval {self.cfg.diffusion.diffusion_steps_eval}\n')
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

            z_t.y = t_array.clone().float() * eval_step_size             

            # IF EXTRA CONDITIONING, THEN CALCULATE log_prob_s INSTEAD, and add the log p(y|x_t) GRADIENT TO IT
            if self.cfg.diffusion.count_conditioning_gamma != 0:                
                pred, grad_log_p_y_x_t = diffusion_utils.grad_log_p_y_x_t_approx(self, z_t, self.cfg.diffusion.count_conditioning_a, self.cfg.diffusion.count_conditioning_b, self.cfg.diffusion.count_conditioning_gamma, self.cfg.dataset.atom_types.index('U'))
                pred = pred.get_new_object(X=torch.log_softmax(pred.X, -1), E=torch.log_softmax(pred.E, -1),
                                           atom_charges=torch.log_softmax(pred.atom_charges, -1), atom_chiral=torch.log_softmax(pred.atom_chiral, -1), bond_dirs=torch.log_softmax(pred.bond_dirs, -1))
                prob_s_ = helpers.get_p_zs_given_zt(transition_model=self.transition_model_eval, t_array=t_array, pred=pred, z_t=z_t, log=False)
                log_prob_s_adjusted = prob_s_.log().add(grad_log_p_y_x_t)
                prob_s = log_prob_s_adjusted.softmax() # re-normalize
            else:
                # compute p(x | z_t)
                pred = self.forward(z_t=z_t)
                # Temperature scaling & normalize logits
                pred = pred.get_new_object(X=torch.log_softmax(pred.X, -1) * self.cfg.diffusion.temperature_scaling_node, E=torch.log_softmax(pred.E, -1) * self.cfg.diffusion.temperature_scaling_edge)
                # if i%10==0:
                #     print(f'pred.X.shape {pred.X.shape}\n')
                # compute p(z_s | z_t) (denoiser)
                # Note: no need to mask pred because computing denoiser proba is done independently for each node/edge
                z_t = z_t.to_device(device)
                pred = pred.to_device(device)
                print(f'device in sample_one_batch: {z_t.X.device}')
                prob_s = helpers.get_p_zs_given_zt(transition_model=self.transition_model_eval, t_array=t_array, pred=pred, z_t=z_t)
            # Tested that helpers.get_p_zs_given_zt_old is pretty much exactly equivalent to this (which is the new helpers.get_p_zs_given_zt)
            #Qt = self.transition_model_eval.get_Qt(t_array, device=device)
            #Qtb = self.transition_model_eval.get_Qt_bar(t_array, device=device) 
            #Qsb = self.transition_model_eval.get_Qt_bar(t_array-1, device=device)
            #q_s_given_t_0_X = helpers.compute_posterior_distribution(M=torch.softmax(pred.X, -1), M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
            #q_s_given_t_0_E = helpers.compute_posterior_distribution(M=torch.softmax(pred.E, -1), M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)

            # save chains if relevant
            # need to mask here to keep consistency for plotting
            if get_chains and (s_int%self.cfg.train.log_every_t==0 or s_int==self.T-1): 
                prob_s_, pred_ = helpers.format_intermediate_samples_for_plotting(self.cfg, prob_s, pred, dense_data, inpaint_node_idx, inpaint_edge_idx)
                # The logic: We plot the samples starting from T, and the denoising/NN outputs starting at T-1, e.g., p(x_{T-1}|x_T)
                sample_chains.append((s_int+1, z_t.mask(z_t.node_mask, collapse=True)))
                prob_s_chains.append((s_int, prob_s_))
                pred_0_chains.append((s_int, pred_))
                
            # sample from p(z_s | z_t)
            # Note: no need to mask pred because sampling is done independently for each node/edge
            z_t = helpers.sample_categoricals_simple(prob=prob_s)

            # sanity check
            assert (z_t.E==torch.transpose(z_t.E, 1, 2)).all(), 'E is not symmetric.'
            
            z_t,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, z_t, dense_data)
            z_t,_,_ = graph.fix_nodes_and_edges_by_idx(z_t, data=dense_data, node_idx=inpaint_node_idx,
                                                                edge_idx=inpaint_edge_idx)
            #print(f'z_t.E.shape {z_t.E.shape}\n')
        
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

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        
        from diffalign_old.neuralnet.extra_features import ExtraFeatures
        from diffalign_old.neuralnet.extra_features_molecular import ExtraMolecularFeatures
        self.extra_features_calculator = ExtraFeatures('all', self.dataset_info)
        self.extra_molecular_features_calculator = ExtraMolecularFeatures(self.dataset_info)

        # if torch.cuda.device_count() > 1 and self.cfg.neuralnet.use_all_gpus:
        #     self.extra_features_calculator = torch.nn.DataParallel(self.extra_features_calculator)
        #     self.extra_molecular_features_calculator = torch.nn.DataParallel(self.extra_molecular_features_calculator)
        device = noisy_data.X.device

        # input_dims['X'] = input_dims['X'] + 8 
        # input_dims['E'] = input_dims['E']
        # input_dims['y'] = input_dims['y'] + 12

        try:
            X_, E_, y_ = self.extra_features_calculator(noisy_data.E.to('cpu'), noisy_data.node_mask.to('cpu'))
            extra_features = noisy_data.get_new_object(X=X_.to(device), E=E_.to(device), y=y_.to(device))
            X_, E_, y_ = self.extra_molecular_features_calculator(noisy_data.X.to('cpu'), noisy_data.E.to('cpu'))
            extra_molecular_features = noisy_data.get_new_object(X=X_.to(device), E=E_.to(device), y=y_.to(device))

            extra_features.X = extra_features.X.detach() # don't allow backpropagation through these, it doesn't work
            extra_features.E = extra_features.E.detach()
            extra_features.y = extra_features.y.detach()
            extra_molecular_features.X = extra_molecular_features.X.detach()
            extra_molecular_features.E = extra_molecular_features.E.detach()
            extra_molecular_features.y = extra_molecular_features.y.detach()

            extra_X = torch.cat((noisy_data.X, extra_features.X, extra_molecular_features.X), dim=-1) 
            extra_E = torch.cat((noisy_data.E, extra_features.E, extra_molecular_features.E), dim=-1)
            extra_y = torch.cat((noisy_data.y, extra_features.y, extra_molecular_features.y), dim=-1)
        except:
            log.info("Couldn't calculate the extra features for some reason")
            bs = noisy_data.X.shape[0]
            n = noisy_data.X.shape[1]
            device = noisy_data.X.device
            extra_features = noisy_data.get_new_object(X=torch.zeros(bs,n,8,device=device), E=torch.zeros(bs,n,n,0, device=device), y=torch.zeros(n,12,device=device))
            # extra_molecular_features = noisy_data.get_new_object()
            extra_X = torch.cat((noisy_data.X, extra_features.X), dim=-1) 
            extra_E = torch.cat((noisy_data.E, extra_features.E), dim=-1)
            extra_y = torch.cat((noisy_data.y, extra_features.y), dim=-1)

        # extra_features.X = extra_features.X.detach() # don't allow backpropagation through these, it doesn't work
        # extra_features.E = extra_features.E.detach()
        # extra_features.y = extra_features.y.detach()
        # extra_molecular_features.X = extra_molecular_features.X.detach()
        # extra_molecular_features.E = extra_molecular_features.E.detach()
        # extra_molecular_features.y = extra_molecular_features.y.detach()

        # extra_X = torch.cat((noisy_data.X, extra_features.X, extra_molecular_features.X), dim=-1) 
        # extra_E = torch.cat((noisy_data.E, extra_features.E, extra_molecular_features.E), dim=-1)
        # extra_y = torch.cat((noisy_data.y, extra_features.y, extra_molecular_features.y), dim=-1)
        
        extra_z = noisy_data.get_new_object(X=extra_X, E=extra_E, y=extra_y)
        
        return extra_z
    
    @torch.no_grad()
    def evaluate(self, dataloader, data_class, save_samples_as_smiles, epoch=0):
        raise NotImplementedError
    
    @torch.no_grad()
    def evaluate_one_batch(self, true_rxns, sample_chains, epoch):
        raise NotImplementedError
    
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
        assert num_batches<=len(dataloader), f'ELBO: testing more batches ({num_batches}) than is available in the dataset ({len(dataloader)}).'
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

    @torch.no_grad()
    def elbo_simple(self, dense_true):
        """Computes an estimator for the variational lower bound, but more straightforwardly than the elbo function.
        Uses the fact that L_1 can also be expressed as a KL divergence between q(x_0|x_1,x_0) and p(x_0|x_1).
        """
        assert self.T % self.cfg.diffusion.diffusion_steps_eval == 0, 'diffusion_steps_eval should be divisible by diffusion_steps'
        all_steps = list(range(1, self.cfg.diffusion.diffusion_steps_eval+1)) #np.linspace(0, self.T, self.cfg.diffusion.diffusion_steps_eval+1).astype('int')
        eval_step_size = self.T // self.cfg.diffusion.diffusion_steps_eval
        steps_to_eval_here = all_steps
        
        total_kl = torch.zeros(dense_true.X.shape[0], device=dense_true.X.device)

        for t in steps_to_eval_here:
            t_int = torch.ones((dense_true.X.shape[0], 1)).to(device)*t
            z_t, _ = self.apply_noise(dense_true, t_int=t_int, transition_model=self.transition_model_eval)
            z_t.y *= eval_step_size # Adjust the neural net input to the correct range
            z_t,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, z_t, dense_true)
            pred = self.forward(z_t=z_t)
            # compute q(x_{t-1}|x_t, x_0) for X and E
            
            Qt = self.transition_model_eval.get_Qt(t, device=device)
            Qtb = self.transition_model_eval.get_Qt_bar(t, device=device) 
            Qsb = self.transition_model_eval.get_Qt_bar(t-1, device=device)
            q_s_given_t_0_X = helpers.compute_posterior_distribution(M=dense_true.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
            q_s_given_t_0_E = helpers.compute_posterior_distribution(M=dense_true.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)
            q_s_given_t_0 = z_t.get_new_object(X=q_s_given_t_0_X, E=q_s_given_t_0_E)
            _,outside_reactant_mask_nodes, outside_reactant_mask_edges = graph.fix_others_than_reactant_to_original(self.cfg, dense_true, dense_true)
            outside_reactant_mask_nodes = outside_reactant_mask_nodes[...,0] # drop the feature dim of the mask
            outside_reactant_mask_edges = outside_reactant_mask_edges[...,0]

            log_p_s_given_t_X = helpers.compute_posterior_distribution(M=pred.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X, log=True)
            log_p_s_given_t_E = helpers.compute_posterior_distribution(M=pred.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E, log=True)
            log_p_s_given_t = z_t.get_new_object(X=log_p_s_given_t_X, E=log_p_s_given_t_E)
            log_p_s_given_t,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, log_p_s_given_t, dense_true, as_logits=True)

            for i in range(len(dense_true.X)):
                kl_x = F.kl_div(input=torch.log_softmax(log_p_s_given_t.X[i], -1)[~outside_reactant_mask_nodes[i]], target=q_s_given_t_0.X[~outside_reactant_mask_nodes[i]], reduction='none').sum()
                kl_e = F.kl_div(input=torch.log_softmax(log_p_s_given_t.E[i], -1)[~outside_reactant_mask_edges[i]], target=q_s_given_t_0.E[~outside_reactant_mask_edges[i]], reduction='none').sum()
                kl = (kl_x + kl_e) / ((~outside_reactant_mask_nodes[i]).sum() + (~outside_reactant_mask_edges[i]).sum()) # normalize with respect to dimensions in data
                total_kl[i] += kl

        return total_kl

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
        
        _, mask_X, mask_E = graph.fix_others_than_reactant_to_original(self.cfg, dense_true, dense_true)
        mask_X = ~mask_X.to('cpu') # switch to mode where we include the reactants, but not the products
        mask_E = ~mask_E.to('cpu')

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        # should be equivalent to CE?
        loss_0s = []
        for i in range(self.cfg.test.loss_0_repeat):
            loss_term_0 = self.compute_L1(dense_true=dense_true)
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
        if len(loss_all_t)==0: 
            loss_t_per_dim = torch.zeros_like(kl_prior_per_dim, dtype=torch.float)
        # Combine terms
        vb =  kl_prior_per_dim + loss_t_per_dim - loss_0_per_dim
        return vb, loss_t_per_dim, loss_0_per_dim

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
        all_steps = list(range(self.cfg.diffusion.diffusion_steps_eval+1)) #np.linspace(0, self.T, self.cfg.diffusion.diffusion_steps_eval+1).astype('int')
        #print(f'==== all_steps: {all_steps}')
        eval_step_size = self.T // self.cfg.diffusion.diffusion_steps_eval
        #print(f'==== eval_step_size: {eval_step_size}')
        steps_to_eval_here = all_steps[2:]
        #print(f'==== steps_to_eval_here: {steps_to_eval_here}')
        
        # pos_encodings = self.get_pos_encodings_if_relevant(dense_true)
        
        for idx, t in enumerate(steps_to_eval_here):
            t_int = torch.ones((true_X.shape[0], 1)).to(device)*t
            z_t, _ = self.apply_noise(dense_true, t_int=t_int, transition_model=self.transition_model_eval)
            z_t.y *= eval_step_size # Adjust the neural net input to the correct range

            z_t,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, z_t, dense_true)
            
            pred = self.forward(z_t=z_t)
            
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
        # TODO Make sure this works
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
        q_s_given_t_0 = z_t.get_new_object(X=q_s_given_t_0_X, E=q_s_given_t_0_E)

        # compute p(x_{t-1}|x_t) = \sum_{\tilde{x_0}} p(x_{t-1}|x_t, \tilde{x_0}) * p(\tilde{x_0}|x_t)
        # comes down to replacing x_0 with the prediction x_0_tilde

        # TODO: Test this new functionality, make sure that still works fine

        if log==False:
            x_0_tilde = z_t.get_new_object(X=F.softmax(x_0_tilde_logit.X, dim=-1), E=F.softmax(x_0_tilde_logit.E, dim=-1))
            #x_0_tilde = graph.PlaceHolder(X=F.softmax(x_0_tilde_logit.X, dim=-1), E=F.softmax(x_0_tilde_logit.E, dim=-1), y=z_t.y, node_mask=z_t.node_mask)
            p_s_given_t_X = helpers.compute_posterior_distribution(M=x_0_tilde.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)        
            p_s_given_t_X = p_s_given_t_X.reshape(bs, n, v)
            p_s_given_t_E = helpers.compute_posterior_distribution(M=x_0_tilde.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)
            p_s_given_t_E = p_s_given_t_E.reshape(bs, n, n, e)
            p_s_given_t = z_t.get_new_object(X=p_s_given_t_X, E=p_s_given_t_E)
            #p_s_given_t = graph.PlaceHolder(X=p_s_given_t_X, E=p_s_given_t_E, y=z_t.y, node_mask=z_t.node_mask)
            p_s_given_t,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, p_s_given_t, dense_true)
            
        else:
            log_p_s_given_t_X = helpers.compute_posterior_distribution(M=x_0_tilde_logit.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X, log=True)
            log_p_s_given_t_X = log_p_s_given_t_X.reshape(bs, n, v)
            log_p_s_given_t_E = helpers.compute_posterior_distribution(M=x_0_tilde_logit.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E, log=True)
            log_p_s_given_t_E = log_p_s_given_t_E.reshape(bs, n, n, e)
            log_p_s_given_t = z_t.get_new_object(X=log_p_s_given_t_X, E=log_p_s_given_t_E)
            #log_p_s_given_t = graph.PlaceHolder(X=log_p_s_given_t_X, E=log_p_s_given_t_E, y=z_t.y, node_mask=z_t.node_mask)
            log_p_s_given_t,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, log_p_s_given_t, dense_true, as_logits=True)

        q_s_given_t_0,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, q_s_given_t_0, dense_true)

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

    def compute_L1(self, dense_true):
            
        device = dense_true.X.device
        t_int = torch.ones((dense_true.X.shape[0],1), device=device)

        z_1, _ = self.apply_noise(dense_true, t_int=t_int, transition_model=self.transition_model_eval)

        z_1,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, z_1, dense_true)

        assert self.T % self.cfg.diffusion.diffusion_steps_eval == 0, 'diffusion_steps_eval should be divisible by diffusion_steps'
        eval_step_size = self.T // self.cfg.diffusion.diffusion_steps_eval
        z_1.y *= eval_step_size # Adjust the neural net input to the correct range

        pred0 = self.forward(z_t=z_1)
        pred0,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, pred0, dense_true, as_logits=True)
        
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

        # compute q(x_T)
        limitX = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(X)
        limitE = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(E)
        limit = dense_true.get_new_object(X=limitX, E=limitE)
        
        limit,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, limit, dense_true)
        
        prob,_,_ = graph.fix_others_than_reactant_to_original(self.cfg, prob, dense_true)
        
        kl_prior_ = helpers.kl_prior(prior=prob, limit=limit, eps=self.eps)
        
        return kl_prior_
    
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
        _, mask_X, mask_E = graph.fix_others_than_reactant_to_original(self.cfg, z_t, dense_true)
        mask_X = ~mask_X.to('cpu') # switch to mode where we include the reactants, but not the products
        mask_E = ~mask_E.to('cpu')

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