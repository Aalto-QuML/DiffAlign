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

from src.neuralnet.transformer_model import GraphTransformer
from src.neuralnet.transformer_model_with_y import GraphTransformerWithY, GraphTransformerWithYAndPosEmb, GraphTransformerWithYAtomMapPosEmb, PositionalEmbedding
from src.neuralnet.transformer_model_with_y_improved import GraphTransformerWithYImproved
from src.neuralnet.transformer_model_stacked import GraphTransformerWithYStacked, PositionalEmbeddingTorch
from src.diffusion.noise_schedule import *
from src.utils import graph, mol
from src.neuralnet.ema_pytorch import EMA
from src.utils.diffusion import helpers
import traceback

# A logger for this file
log = logging.getLogger(__name__)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Disable rdkit warnings

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]

MAX_NODES = 100

class DiscreteDenoisingDiffusion(nn.Module):
    def __init__(self, cfg, dataset_infos, node_type_counts_unnormalized=None, edge_type_counts_unnormalized=None, use_data_parallel=None):
        super().__init__()
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist
        if cfg.neuralnet.extra_features:
            # Hardcoded for now, based on one sample of extra features
            input_dims = {'X': input_dims['X'] + 8, 'E': input_dims['E'], 'y': input_dims['y'] + 12}

        self.cfg = cfg
        self.T = cfg.diffusion.diffusion_steps

        # self.Xdim_input = input_dims['X']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        if self.cfg.neuralnet.increase_y_dim_for_multigpu:
            output_dims['y'] += 1 # TODO: Make this a bit less hackier: it makes it possible for the model to work with multiple GPUs. Somehow output_dims['y'] is used somewhere directly, but not sure where and it needs to be changed here
        self.ydim_output = output_dims['y'] 
        self.node_dist = nodes_dist

        self.pos_emb_module = PositionalEmbedding(cfg.neuralnet.hidden_dims['dx'], cfg.neuralnet.pos_emb_permutations)
        # if cfg.neuralnet.architecture == 'with_y_stacked': # move on to torch version of this, should be faster
        #     self.pos_emb_module = PositionalEmbeddingTorch(cfg.neuralnet.hidden_dims['dx'], cfg.neuralnet.pos_emb_permutations)

        self.dataset_info = dataset_infos
        self.log_to_wandb = cfg.train.log_to_wandb

        self.eps = 1e-6
        self.log_every_steps= cfg.general.log_every_steps
        
        node_idx_to_mask, edge_idx_to_mask = graph.get_index_from_states(atom_decoder=self.dataset_info.atom_decoder,
                                                                         bond_decoder=self.dataset_info.bond_decoder,
                                                                         node_states_to_mask=cfg.diffusion.node_states_to_mask,
                                                                         edge_states_to_mask=cfg.diffusion.edge_states_to_mask,
                                                                         device=device)

        abs_state_position_e = 0
        abs_state_position_x = self.dataset_info.atom_decoder.index('Au') 
        
        if self.cfg.neuralnet.architecture=='with_y':
            self.model = GraphTransformerWithY(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                               hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                               hidden_dims=cfg.neuralnet.hidden_dims,
                                               output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                               improved=cfg.neuralnet.improved, dropout=cfg.neuralnet.dropout)
        elif self.cfg.neuralnet.architecture=='without_y': # TODO: We don't really need this, right?
            self.model = GraphTransformer(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                          hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                          hidden_dims=cfg.neuralnet.hidden_dims,
                                          output_dims=output_dims,
                                          act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU())
        elif self.cfg.neuralnet.architecture=='with_y_pos_enc':
            self.model = GraphTransformerWithYAndPosEmb(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                                        hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                                        hidden_dims=cfg.neuralnet.hidden_dims,
                                                        output_dims=output_dims,
                                                        act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(), dropout=cfg.neuralnet.dropout)
        # elif self.cfg.neuralnet.architecture=='with_y_improved': # DEPRECATED, this is now a separate parameter
        #     self.model = GraphTransformerWithYImproved(n_layers=cfg.neuralnet.n_layers,
        #                                     input_dims=input_dims,
        #                                     hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
        #                                     hidden_dims=cfg.neuralnet.hidden_dims,
        #                                     output_dims=output_dims,
        #                                     act_fn_in=nn.ReLU(),
        #                                     act_fn_out=nn.ReLU())
        elif self.cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc':
            self.model = GraphTransformerWithYAtomMapPosEmb(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                               hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                               hidden_dims=cfg.neuralnet.hidden_dims,
                                               output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                               pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                               improved=cfg.neuralnet.improved, dropout=cfg.neuralnet.dropout,
                                               p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                               p_to_r_init=cfg.neuralnet.p_to_r_init, alignment_type=cfg.neuralnet.alignment_type,
                                               input_alignment=cfg.neuralnet.input_alignment)
        elif self.cfg.neuralnet.architecture=='with_y_stacked':
            self.model = GraphTransformerWithYStacked(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                               hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                               hidden_dims=cfg.neuralnet.hidden_dims,
                                               output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                               pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                               improved=cfg.neuralnet.improved, dropout=cfg.neuralnet.dropout,
                                               p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                               p_to_r_init=cfg.neuralnet.p_to_r_init)
        if use_data_parallel:
            #if torch.cuda.device_count() > 1 and cfg.neuralnet.use_all_gpus:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        log.info(cfg.neuralnet.use_ema)
        self.ema = None
        if cfg.neuralnet.use_ema:
            log.info("???? Using EMA")
            self.ema = EMA(self.model, beta=cfg.neuralnet.ema_decay, power=1)

        if cfg.diffusion.transition=='uniform':
            self.transition_model, self.transition_model_eval = (DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output, noise_schedule=cfg.diffusion.diffusion_noise_schedule,
                                                              timesteps=T_, diffuse_edges=cfg.diffusion.diffuse_edges,
                                                              node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
                                                              for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()

        elif cfg.diffusion.transition=='marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            self.transition_model, self.transition_model_eval = (MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output, 
                                                              noise_schedule=cfg.diffusion.diffusion_noise_schedule,
                                                              timesteps=T_, 
                                                              node_idx_to_mask=node_idx_to_mask,
                                                              edge_idx_to_mask=edge_idx_to_mask,
                                                              diffuse_edges=cfg.diffusion.diffuse_edges)
                                                              for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()

        elif cfg.diffusion.transition=='tokenwise_absorbing':
            self.transition_model, self.transition_model_eval = (TokenwiseAbsorbingStateTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                                      y_classes=self.ydim_output, timesteps=T_,
                                                                      node_type_counts_unnormalized=node_type_counts_unnormalized, 
                                                                      edge_type_counts_unnormalized=edge_type_counts_unnormalized,
                                                                      abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                      diffuse_edges=cfg.diffusion.diffuse_edges, node_idx_to_mask=node_idx_to_mask,
                                                                      edge_idx_to_mask=edge_idx_to_mask)
                                                                      for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()

        elif cfg.diffusion.transition=='absorbing_masknoedge':
            self.transition_model, self.transition_model_eval = (AbsorbingStateTransitionMaskNoEdge(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                                       y_classes=self.ydim_output, timesteps=T_,
                                                                       diffuse_edges=cfg.diffusion.diffuse_edges,
                                                                       abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                       node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
                                                                       for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()

        elif cfg.diffusion.transition=='tokenwise_absorbing_masknoedge':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            self.transition_model, self.transition_model_eval = (TokenwiseAbsorbingStateTransitionMaskNoEdge(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                                   y_classes=self.ydim_output, timesteps=T_,
                                                                   node_type_counts_unnormalized=node_type_counts_unnormalized, 
                                                                   edge_type_counts_unnormalized=edge_type_counts_unnormalized, 
                                                                   diffuse_edges=cfg.diffusion.diffuse_edges, abs_state_position_e=abs_state_position_e, 
                                                                   abs_state_position_x=abs_state_position_x,
                                                                   node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
                                                                   for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()

        elif cfg.diffusion.transition=='blocktokenwise_absorbing_masknoedge':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            self.transition_model, self.transition_model_eval = (BlockTokenwiseAbsorbingStateTransitionMaskNoEdge(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                                                     y_classes=self.ydim_output, timesteps=T_,
                                                                                     node_type_counts_unnormalized=node_type_counts_unnormalized, 
                                                                                     edge_type_counts_unnormalized=edge_type_counts_unnormalized, 
                                                                                     diffuse_edges=cfg.diffusion.diffuse_edges, abs_state_position_e=abs_state_position_e, 
                                                                                     abs_state_position_x=abs_state_position_x,
                                                                                     node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask) 
                                                                                     for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()
        
        elif cfg.diffusion.transition=='nodes_before_edges':
            self.transition_model, self.transition_model_eval = (NodesBeforeEdges(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                     y_classes=self.ydim_output, timesteps=T_,
                                                    abs_state_position_e=abs_state_position_e, 
                                                     abs_state_position_x=abs_state_position_x, 
                                                     diffuse_edges=cfg.diffusion.diffuse_edges, node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask,
                                                     num_node_steps=cfg.diffusion.num_node_steps)
                                                     for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()
            
        elif cfg.diffusion.transition=='edges_before_nodes':
            self.transition_model, self.transition_model_eval = (EdgesBeforeNodes(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                     y_classes=self.ydim_output, timesteps=T_,
                                                    abs_state_position_e=abs_state_position_e, 
                                                     abs_state_position_x=abs_state_position_x, 
                                                     diffuse_edges=cfg.diffusion.diffuse_edges, node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask,
                                                     num_node_steps=cfg.diffusion.num_node_steps)
                                                     for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()
        elif cfg.diffusion.transition=='marginaledge_absorbingnode':
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            self.transition_model, self.transition_model_eval = (MarginalEdgesMaskNodesTransition(x_classes=self.Xdim_output, e_marginals=e_marginals, 
                                                                     y_classes=self.ydim_output, 
                                                                     noise_schedule=cfg.diffusion.diffusion_noise_schedule,
                                                                     timesteps=T_,
                                                                     abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                     diffuse_edges=cfg.diffusion.diffuse_edges)
                                                                     for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()
        else: 
            assert f'Transition model undefined. Got {cfg.diffusion.transition}\n.'

    def training_step(self, data, i, device):
        dense_data = graph.to_dense(data=data).to_device(device)
        t_int = torch.randint(1, self.T+1, size=(len(data),1), device=device)
        z_t, _ = self.apply_noise(dense_data, t_int = t_int, transition_model=self.transition_model)
        # if 'mask_atom_mapping' in data.keys:
        #     # we don't care about the atom mapping numbers for now => turn to a bool mask
        #     # data.mask_atom_mapping = (data.mask_atom_mapping==0) <- this was a problem
        #     mask_atom_mapping = (data.mask_atom_mapping==0)
        #     mask_atom_mapping, _ = to_dense_batch(x=mask_atom_mapping, batch=data.batch)
        # else:
        #     mask_atom_mapping = None
        
        z_t = graph.apply_mask(orig=dense_data, z_t=z_t, atom_decoder=self.dataset_info.atom_decoder,
                               bond_decoder=self.dataset_info.bond_decoder, mask_nodes=self.cfg.diffusion.mask_nodes, 
                               mask_edges=self.cfg.diffusion.mask_edges, return_masks=False)
        
        # This will go to the vlb calculations (if we zero out later during classifier-free guidance, the zeroing out and vlb don't match well)
        z_t_ = z_t.get_new_object()
        # z_t_ = graph.PlaceHolder(X=z_t.X.clone(), E=z_t.E.clone(), y=z_t.y.clone(), node_mask=z_t.node_mask.clone(), atom_map_numbers=z_t.atom_map_numbers) 
        if not self.cfg.diffusion.diffuse_edges: z_t.E = dense_data.E.clone()
        if not self.cfg.diffusion.diffuse_nodes: z_t.X = dense_data.X.clone()

        # Okay so we want to change the following such that we drop out parts of the batch
        # at once, not the whole thing. And we want to add it as an option, I guess.
        # (the current method can be written as a special case of this)
        unconditional = torch.zeros(z_t.X.shape[0], dtype=torch.bool, device=z_t.X.device)
        if self.cfg.diffusion.classifier_free_guidance:
            # We want to use masks to delete parts of reactions, but not the supernodes. Get corresponding masks for that
            _, mask_X, mask_E = graph.apply_mask(orig=dense_data, z_t=z_t,
                                                atom_decoder=self.dataset_info.atom_decoder,
                                                bond_decoder=self.dataset_info.bond_decoder, 
                                                mask_nodes=self.cfg.diffusion.mask_nodes, 
                                                mask_edges=self.cfg.diffusion.mask_edges, return_masks=True,
                                                include_supernode=False)

            # We want to replace this with torch.rand(bs)
            r = torch.rand(z_t.X.shape[0])
            unconditional = r < self.cfg.diffusion.classifier_free_uncond_prob

            # This randomly assigns each batch element whether to drop or not.       
            if self.cfg.diffusion.classifier_free_drop_within_batch:
                mask_X[~unconditional] = 1
                mask_E[~unconditional] = 1
            else: # Here, we drop out the entire batch (how it was coded up until now)
                if not unconditional[0]:
                    mask_X = torch.ones_like(mask_X)
                    mask_E = torch.ones_like(mask_E)
            
            # TODO: Is this even used at all here, since later we zero out the positional encodings?
            if self.cfg.diffusion.classifier_free_full_unconditioning and self.cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc':
                # This needs to be updated also hmm
                z_t.atom_map_numbers[~unconditional] = torch.zeros_like(z_t.atom_map_numbers[~unconditional])
            z_t = helpers.zero_out_condition(z_t, mask_X, mask_E)

            # if unconditional:
                # z_t = helpers.drop_out_condition(z_t, mask_X, mask_E)
                # dense_data = helpers.drop_out_condition(dense_data, mask_X, mask_E)
                # alternative:
                # This in the case that we had the atom mapping info but don't want to condition on that during the unconditional step. Does this ever come up?
                # if self.cfg.diffusion.classifier_free_full_unconditioning and self.cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc':
                #     z_t.atom_map_numbers = torch.zeros_like(z_t.atom_map_numbers)
                # z_t = helpers.zero_out_condition(z_t, mask_X, mask_E)
                # dense_data = helpers.zero_out_condition(dense_data, mask_X, mask_E)
        
        if torch.cuda.is_available(): #and self.model.training:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # try:
                pred = self.forward(z_t=z_t, use_pos_encoding_if_applicable=~unconditional)
                # except:
                #     print(f'error in forward for batch {i}\n')
                #     pickle.dump(z_t, open('z_t.pickle', 'wb'))
                #     pickle.dump(dense_data, open('dense_data.pickle', 'wb'))
                #     pickle.dump(data, open('data.pickle', 'wb'))
                #     exit()
        else:
            pred = self.forward(z_t=z_t, use_pos_encoding_if_applicable=~unconditional)

        # TODO: Does this actually do anything here, given that we already have the conditioning info in the ce loss?
        # -> well at least we get the masks
        pred, mask_X, mask_E = graph.apply_mask(orig=dense_data, z_t=pred,
                                                atom_decoder=self.dataset_info.atom_decoder,
                                                bond_decoder=self.dataset_info.bond_decoder,
                                                mask_nodes=self.cfg.diffusion.mask_nodes,
                                                mask_edges=self.cfg.diffusion.mask_edges,
                                                as_logits=True, return_masks=True)
        if not self.cfg.diffusion.diffuse_edges: pred.E = dense_data.E.clone()
        if not self.cfg.diffusion.diffuse_nodes: pred.X = dense_data.X.clone()

        mask_nodes = mask_X.max(-1)[0].flatten(0,-1) # identify nodes to ignore in masking
        mask_edges = mask_E.max(-1)[0].flatten(0,-1)
            
        if self.cfg.train.loss=='ce':
            loss_X, loss_E, loss = helpers.ce(pred=pred, discrete_dense_true=dense_data,
                                                  diffuse_edges=self.cfg.diffusion.diffuse_edges,
                                                  diffuse_nodes=self.cfg.diffusion.diffuse_nodes, 
                                                  lambda_E=self.cfg.diffusion.lambda_train[0], 
                                                  log=(i % self.log_every_steps == 0) and self.log_to_wandb,
                                                  mask_nodes=mask_nodes, mask_edges=mask_edges)
        elif self.cfg.train.loss=='vb':
            # TODO: Should create a version of ELBO where we sample a subset of the timesteps for training
            # Now the predictions are not used here at all
            # TODO: Should we have the mask_nodes etc. here? or is it taken care of somewhere?
            loss = self.elbo_batch_quick(dense_data, pred=pred, z_t=z_t_, lambda_E=self.cfg.diffusion.lambda_train[0])
            loss_X, loss_E = torch.zeros((1,)), torch.zeros((1,))
        elif self.cfg.train.loss=='vbce':
            loss_X_ce, loss_E_ce, ce_loss = helpers.ce(pred=pred, discrete_dense_true=dense_data,
                                diffuse_edges=self.cfg.diffusion.diffuse_edges, 
                                diffuse_nodes=self.cfg.diffusion.diffuse_nodes, 
                                lambda_E=self.cfg.diffusion.lambda_train[0], 
                                log=(i % self.log_every_steps == 0) and self.log_to_wandb,
                                mask_nodes=mask_nodes, mask_edges=mask_edges)
            elbo = self.elbo_batch_quick(dense_data, pred=pred, z_t=z_t_, lambda_E=self.cfg.diffusion.lambda_train[0])
            loss = elbo + self.cfg.diffusion.ce_lambda * ce_loss
            loss_X, loss_E = torch.zeros((1,)), torch.zeros((1,))
            
        return loss_X, loss_E, loss 

    def apply_noise(self, dense_data, t_int, transition_model):
        """ 
            Sample noise and apply it to the data. 
            
            input:
                discrete_data: batch graph object with nodes and edges in discrete form.
                t_int: time step for noise.
            return: 
                (PlaceHolder) z_t.
        """
        # if type(discrete_data)!=graph.PlaceHolder:
        #     dense_data = graph.to_dense(data=discrete_data)
        # else:
        #     dense_data = discrete_data
        # # dense_data = graph.to_dense(data=discrete_data)
        X, E, y = dense_data.X, dense_data.E, dense_data.y
        device = dense_data.X.device
        # log.info(f"{X.shape}, {E.shape}, {y.shape}")

        assert X.dim()==3, 'Expected X in batch format.'+\
               f' Got X.dim={X.dim()}, If using one example, add batch dimension with: X.unsqueeze(dim=0).'

        # if t_int is None:
        #     # when training, can add 0 noise. In inference, cannot start with 0 noise?
        #     # lowest_t = 0 if self.training else 1 <- this doesn't make sense
        #     lowest_t = 1
        #     t_int = torch.randint(lowest_t, self.T+1, size=(X.size(0),1), device=device) # (bs,1)
        
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
        
        prob_t = dense_data.get_new_object(X=probX, E=probE, y=t_int.float()).mask(dense_data.node_mask)
        #prob_t = graph.PlaceHolder(X=probX, E=probE, y=t_int, node_mask=dense_data.node_mask, atom_map_numbers=dense_data.atom_map_numbers).type_as(X).mask(dense_data.node_mask)
        
        z_t = helpers.sample_discrete_features(prob=prob_t)

        assert (X.shape==z_t.X.shape) and (E.shape==z_t.E.shape), 'Noisy and original data do not have the same shape.'

        return z_t, prob_t
    
    def get_pos_encodings(self, z_t):
        if self.cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc':
            if self.cfg.neuralnet.pos_encoding_type == 'smiles_pos_enc':
                pos_encodings = self.pos_emb_module.matched_positional_encodings_sinusoidal(z_t.atom_map_numbers, z_t.mol_assignments, direction = 'retro')
            elif self.cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc':
                pos_encodings = self.pos_emb_module.matched_positional_encodings_laplacian(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignments, self.cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
            elif self.cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc_gpu':
                pos_encodings = self.pos_emb_module.matched_positional_encodings_laplacian_gpu(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignments, self.cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
            elif self.cfg.neuralnet.pos_encoding_type == 'infoleak_pos_enc':
                log.info("Using info leak positional encoding, for illustration purposes only")
                # TODO: Implement this
                pos_encodings = self.pos_emb_module.matched_positional_encodings_infoleak(z_t.atom_map_numbers, z_t.mol_assignments, direction = 'retro')
            elif self.cfg.neuralnet.pos_encoding_type == 'gaussian_pos_enc':
                pos_encodings = self.pos_emb_module.matched_positional_encodings_gaussian(z_t.atom_map_numbers, z_t.mol_assignments)
            else:
                raise ValueError(f'pos_encoding_type {self.cfg.neuralnet.pos_encoding_type} not recognized')
        elif self.cfg.neuralnet.architecture == 'with_y_stacked':
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            suno_number = self.cfg.dataset.atom_types.index("SuNo")
            reaction_side_separation_index = (z_t.X.argmax(-1) == suno_number).nonzero(as_tuple=True)[1]
            
            if self.cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc_gpu':
                # This needs to be designed as:
                # 1. Cut the correct part of E
                # 2. Calculate the eigendecomposition
                # 3. Place it in the correct part with the code that we already have elsewhere, maybe just reuse the code that we already have? And then just cut it. Let's see...
                pos_encodings = model.pos_emb_module.matched_positional_encodings_laplacian(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignments, self.cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
                pos_encodings, _ = model.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index) # ... this could be done in forward as well, more efficient with multiple GPUs. Actually both parts hmm
            elif self.cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc':
                # 3. Place it in the correct part with the code that we already have elsewhere, maybe just reuse the code that we already have? And then just cut it. Let's see...
                pos_encodings = model.pos_emb_module.matched_positional_encodings_laplacian_scipy(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignments, self.cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
                pos_encodings, _ = model.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index)
            else:
                pos_encodings = torch.zeros(z_t.X.shape[0], z_t.X.shape[1], model.input_dim_X, device=z_t.X.device)
                pos_encodings, _ = model.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index)
        else:
            pos_encodings = None
        return pos_encodings
    
    def get_pos_encodings_if_relevant(self, z_t):
        if self.cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc' or 'with_y_stacked': # Precalculate the pos encs, since they are the same for each step in the loop
            pos_encodings = self.get_pos_encodings(z_t)
        else:
            pos_encodings = None
        return pos_encodings

    def forward(self, z_t, use_pos_encoding_if_applicable=None, pos_encodings=None):
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
                assert z_t.mol_assignments is not None, 'molecule_assigments is None in forward()'
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
                    res = self.ema(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignments)
                else:
                    res = self.model(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignments)
            elif self.cfg.neuralnet.architecture=='with_y_stacked':
                # pos encodings can't be calculated here, since the dimensions are different for different GPUs (different part of batch) -> could also put them with the extra features, but oh well, this is faster as it parallelizes
                # if pos_encodings == None: 
                #     with autocast(enabled=False):
                #         pos_encodings = self.get_pos_encodings(z_t)
                # pos_encodings *= use_pos_encoding_if_applicable[:,None,None].to(pos_encodings.device).float()
                if self.cfg.neuralnet.use_ema and not self.training:
                    res = self.ema(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignments, use_pos_encoding_if_applicable, self.cfg.neuralnet.pos_encoding_type, self.cfg.neuralnet.num_lap_eig_vectors, self.cfg.dataset.atom_types)
                else:
                    res = self.model(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignments, use_pos_encoding_if_applicable, self.cfg.neuralnet.pos_encoding_type, self.cfg.neuralnet.num_lap_eig_vectors, self.cfg.dataset.atom_types)
            else:
                if self.cfg.neuralnet.use_ema and not self.training:
                    res = self.ema(z_t.X, z_t.E, z_t.y, z_t.node_mask)
                else:
                    res = self.model(z_t.X, z_t.E, z_t.y, z_t.node_mask)
            if isinstance(res, tuple):
                X, E, y, node_mask = res
                res = z_t.get_new_object(X=X, E=E, y=y, node_mask=node_mask)
                #res = graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask, atom_map_numbers=z_t.atom_map_numbers).mask(node_mask)
            return res
        else:
            assert f'Denoiser model not recognized. Value given: {self.cfg.diffusion.denoiser}. You need to choose from: uniform, carbon and neuralnet.'
    
    def compute_Lt_all(self, dense_true):
        '''
            Compute L_s terms: E_{q(x_t|x)} KL[q(x_s|x_t,x_0)||p(x_s|x_t)], with s = t-1
            But compute all of the terms, is this how we want the function to behave?
            To test this, would be nice to have a function for defining the transition matrices 
            for different time steps
        '''
                
        # if type(discrete_true)!=graph.PlaceHolder:
        #     dense_true = graph.to_dense(data=discrete_true)
        #     # if 'mask_atom_mapping' in discrete_true.keys and mask_atom_mapping==None:
        #     #     # we don't care about the atom mapping numbers for now => turn to a bool mask
        #     #     mask_atom_mapping = (discrete_true.mask_atom_mapping==0)
        #     #     mask_atom_mapping, _ = to_dense_batch(x=mask_atom_mapping, batch=discrete_true.batch)
        # else:
        #     dense_true = discrete_true
        
        device = dense_true.X.device
        true_X, true_E = dense_true.X, dense_true.E
        
        Lts = []
        
        # t0 = time.time()
        # # If we wanted to parallelize this, this is how it would happen:
        # import random
        # np.random.seed(1)
        # random.seed(1)
        # torch.manual_seed(1)
        # if isinstance(self.model, torch.nn.DataParallel):
        #     num_gpus = torch.cuda.device_count()
        # else:
        #     num_gpus = 1
        # t_groups = torch.split(torch.arange(2,self.T+1), num_gpus)
        # for idx, t_group in enumerate(t_groups):
        #     t_int = torch.cat([torch.ones((true_X.shape[0], 1)).to(device)*t for t in t_group], 0)
        #     X, E, y, node_mask = torch.repeat_interleave(dense_true.X, repeats=len(t_group), dim=0), \
        #         torch.repeat_interleave(dense_true.E, repeats=len(t_group), dim=0), \
        #         torch.repeat_interleave(dense_true.y, repeats=len(t_group), dim=0), \
        #         torch.repeat_interleave(dense_true.node_mask, repeats=len(t_group), dim=0)
        #     dense_repeated = graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask)
        #     mask_atom_mapping_repeated = torch.repeat_interleave(mask_atom_mapping, repeats=len(t_group), dim=0) if mask_atom_mapping is not None else None
        #     #log.info(f"{t_int.shape}, {X.shape}, {E.shape}")
        #     z_t, _ = self.apply_noise(discrete_data=dense_repeated, t_int=t_int)
        #     pred = self.forward(z_t=z_t)
        #     Lt = self.compute_Lt(dense_true=dense_repeated, z_t=z_t, x_0_tilde_logit=pred, 
        #                          transition_model=self.transition_model, mask_atom_mapping=mask_atom_mapping_repeated)
            # Lt is a PlaceHolder object, let's break it into multiple PlaceHolder Objects
            # Lt = [graph.PlaceHolder(X=X, E=E, y=y, node_mask=node_mask) for X, E, y, node_mask in zip(torch.split(Lt.X, len(t_group)), torch.split(Lt.E, len(t_group)), torch.split(Lt.y, len(t_group)), torch.split(Lt.node_mask, len(t_group)))]
            # Lts.extend(Lt)
        # log.info(sum([X.sum() for X in Lts.X]))
        # log.info(f"Time taken: {time.time() - t0}")

        assert self.T % self.cfg.diffusion.diffusion_steps_eval == 0, 'diffusion_steps_eval should be divisible by diffusion_steps'
        all_steps = list(range(self.cfg.diffusion.diffusion_steps_eval+1)) #np.linspace(0, self.T, self.cfg.diffusion.diffusion_steps_eval+1).astype('int')
        eval_step_size = self.T // self.cfg.diffusion.diffusion_steps_eval
        steps_to_eval_here = all_steps[2:]

        # t_steps = all_steps[1:]
        # s_steps = all_steps[:-1]
        
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
            # try:
            #     pred = self.forward(z_t=z_t)
            # except:
            #     torch.save(self.model.state_dict(), "failedmodel.pt") 
            #     pickle.dump(z_t, open('z_t.pkl', 'wb'))
            #     print(f'Error in forward process!!')
            #     # exit()
            # pred.X = F.softmax(pred.X, dim=-1) # bs, n, d0 compute_Lt now takes logits
            # pred.E = F.softmax(pred.E, dim=-1) # bs, n, n, d0
            
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
        # TODO: check why q_s_given_t_0_E is not symmetric 
        q_s_given_t_0 = z_t.get_new_object(X=q_s_given_t_0_X, E=q_s_given_t_0_E)
        #q_s_given_t_0 = graph.PlaceHolder(X=q_s_given_t_0_X, E=q_s_given_t_0_E, y=z_t.y, node_mask=z_t.node_mask)

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
        # if type(discrete_true)!=graph.PlaceHolder:
        #     dense_true = graph.to_dense(data=discrete_true)
        #     # if 'mask_atom_mapping' in discrete_true.keys and mask_atom_mapping==None:
        #     #     # we don't care about the atom mapping numbers for now => turn to a bool mask
        #     #     mask_atom_mapping = (discrete_true.mask_atom_mapping==0)
        #     #     mask_atom_mapping, _ = to_dense_batch(x=mask_atom_mapping, batch=discrete_true.batch)
        # else:
        #     dense_true = discrete_true
            
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
        # get dense representation of the data
        # if type(discrete_true)!=graph.PlaceHolder:
        #     dense_true = graph.to_dense(data=discrete_true)
        #     # if 'mask_atom_mapping' in discrete_true.keys and mask_atom_mapping==None:
        #     #     # we don't care about the atom mapping numbers for now => turn to a bool mask
        #     #     mask_atom_mapping = (discrete_true.mask_atom_mapping==0)
        #     #     mask_atom_mapping, _ = to_dense_batch(x=mask_atom_mapping, 
        #     #                                           batch=discrete_true.batch)
        # else:
        #     dense_true = discrete_true
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
            # if 'cannot_generate' in data.keys and data.cannot_generate:
            #     continue
            dense_true = graph.to_dense(data=data).to_device(device)
            elbo, _, _ = self.elbo(dense_true)
            # elbos = []
            # for i in range(self.cfg.test.repeat_elbo):
            #     elbos.append(elbo)
                
            # file.write(f'{np.mean(elbos)} +- {np.std(elbos)}\n')
            #total_elbo += sum(elbos)/self.cfg.test.repeat_elbo
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
        # if 'mask_atom_mapping' in discrete_true.keys:
        #     # we don't care about the atom mapping numbers for now => turn to a bool mask
        #     mask_atom_mapping = (discrete_true.mask_atom_mapping==0)
        #     mask_atom_mapping, _ = to_dense_batch(x=mask_atom_mapping, 
        #                                             batch=discrete_true.batch)
        # else:
        #     mask_atom_mapping = None
                
        #t = torch.randint(1, self.T+1, size=(discrete_true.X.shape[0],1), device=device)
        #z_t, _ = self.apply_noise(discrete_data=discrete_true, t_int=t)
        # dense_true = graph.to_dense(data=discrete_true)
        #t = z_t.y[...,0]
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
        # 1. Probability of choosing number of nodes
        # N = node_mask.sum(1).long()
        # log_pN = self.node_dist.log_prob(N)
        # print(f'log_pN {log_pN}\n')
        # mask = graph.get_mask(orig=discrete_true, atom_decoder=self.dataset_info.atom_decoder, 
        #                       bond_decoder=self.dataset_info.bond_decoder, 
        #                       mask_nodes=self.cfg.diffusion.mask_nodes, 
        #                       mask_edges=self.cfg.diffusion.mask_edges,
        #                       node_states_to_mask=self.cfg.diffusion.node_states_to_mask, 
        #                       edge_states_to_mask=self.cfg.diffusion.edge_states_to_mask,
        #                       include_supernode=True, return_mask_nodes=True)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        device = dense_true.X.device
        kl_prior = self.kl_prior(dense_true)
        kl_prior.to_device('cpu') # move everything to CPU to avoid memory issues when computing all the Lt terms

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt_all(dense_true)
        for loss_t in loss_all_t:
            loss_t.E = loss_t.E * self.cfg.diffusion.lambda_test

        # TODO: Make sure that the extra conditioning terms are zeroed out! -> probably they are but should make sure
        # if type(discrete_true)!=graph.PlaceHolder:
        #     dense_true = graph.to_dense(data=discrete_true)
        #     # if 'mask_atom_mapping' in discrete_true.keys:
        #     #     # we don't care about the atom mapping numbers for now => turn to a bool mask
        #     #     mask_atom_mapping = (discrete_true.mask_atom_mapping==0)
        #     #     mask_atom_mapping, _ = to_dense_batch(x=mask_atom_mapping, 
        #     #                                           batch=discrete_true.batch)
        # else:
        #     dense_true = discrete_true
        
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

    @torch.no_grad()
    def sample_one_batch(self, device=None, n_samples=None, data=None, get_chains=False, get_true_rxns=False, inpaint_node_idx=None, inpaint_edge_idx=None):
        assert data!=None or n_samples!=None, 'You need to give either data or n_samples.'
        assert data!=None or self.cfg.diffusion.mask_nodes==None, 'You need to give data if the model is using a mask.'
        assert data!=None or get_true_rxns, 'You need to give data if you want to return true_rxns.'
   
        if data!=None:
            # if type(data)!=graph.PlaceHolder:
            #     dense_data = graph.to_dense(data=data)
            #     node_mask = dense_data.node_mask.to(device)
            #     # if 'mask_atom_mapping' in data.keys and mask_atom_mapping==None:
            #     #     # we don't care about the atom mapping numbers for now => turn to a bool mask
            #     #     mask_atom_mapping = (data.mask_atom_mapping==0)
            #     #     mask_atom_mapping, _ = to_dense_batch(x=mask_atom_mapping, batch=data.batch)
            #     #     atom_map_numbers, _ = graph.to_dense(data.mask_atom_mapping)
            # else:
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
        
        # dense_data_perm, perm = graph.permute_placeholder(dense_data)

        # from rdkit.Chem import Draw
        # import rdkit.Chem as Chem
        # import matplotlib.pyplot as plt
        # product_mol_assignment = dense_data.mol_assignments[0].max().item()
        # product_selection = (dense_data.mol_assignments[0] == product_mol_assignment)
        # mol_ = mol.mol_from_graph(dense_data.X[0,product_selection].argmax(-1), dense_data.E[0,product_selection][:,product_selection].argmax(-1), self.cfg.dataset.atom_types, graph.bond_types)
        # mol_perm = mol.mol_from_graph(dense_data_perm.X[0, product_selection].argmax(-1), dense_data_perm.E[0,product_selection][:,product_selection].argmax(-1), self.cfg.dataset.atom_types, graph.bond_types)
        # matched_pos_emb = self.model.pos_emb_module.matched_positional_encodings_laplacian(dense_data.E.argmax(-1), dense_data.atom_map_numbers, dense_data.mol_assignments, self.cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
        # matched_pos_emb_perm = self.model.pos_emb_module.matched_positional_encodings_laplacian(dense_data_perm.E.argmax(-1), dense_data_perm.atom_map_numbers, dense_data_perm.mol_assignments, self.cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
        # plt.imshow(Draw.MolToImage(mol_))
        
        # pos_encodings = self.get_pos_encodings_if_relevant(dense_data_perm)
        # dense_data = dense_data_perm

        # self.model.get_X_E_product_aligned_with_reactants(dense_data.X, dense_data.E, dense_data.mol_assignments, dense_data.atom_map_numbers, atom_types)
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

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        
        from src.neuralnet.extra_features import ExtraFeatures
        from src.neuralnet.extra_features_molecular import ExtraMolecularFeatures
        self.extra_features_calculator = ExtraFeatures('all', self.dataset_info)
        self.extra_molecular_features_calculator = ExtraMolecularFeatures(self.dataset_info)

        # if torch.cuda.device_count() > 1 and self.cfg.neuralnet.use_all_gpus:
        #     self.extra_features_calculator = torch.nn.DataParallel(self.extra_features_calculator)
        #     self.extra_molecular_features_calculator = torch.nn.DataParallel(self.extra_molecular_features_calculator)
        device = noisy_data.X.device
        #print(f'device {device}\n')
        X_, E_, y_ = self.extra_features_calculator(noisy_data.E.to('cpu'), noisy_data.node_mask.to('cpu'))
        extra_features = noisy_data.get_new_object(X=X_.to(device), E=E_.to(device), y=y_.to(device))
        X_, E_, y_ = self.extra_molecular_features_calculator(noisy_data.X.to('cpu'), noisy_data.E.to('cpu'))
        extra_molecular_features = noisy_data.get_new_object(X=X_.to(device), E=E_.to(device), y=y_.to(device))

        extra_X = torch.cat((noisy_data.X, extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((noisy_data.E, extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((noisy_data.y, extra_features.y, extra_molecular_features.y), dim=-1)
        
        extra_z = noisy_data.get_new_object(X=extra_X, E=extra_E, y=extra_y)

        # extra_z = graph.PlaceHolder(X=extra_X, E=extra_E, y=extra_y, node_mask=noisy_data.node_mask, 
        #                             atom_map_numbers=noisy_data.atom_map_numbers)
        
        return extra_z
    
    @torch.no_grad()
    def evaluate(self, dataloader, data_class, save_samples_as_smiles, epoch=0):
        raise NotImplementedError
    
    @torch.no_grad()
    def evaluate_one_batch(self, true_rxns, sample_chains, epoch):
        raise NotImplementedError