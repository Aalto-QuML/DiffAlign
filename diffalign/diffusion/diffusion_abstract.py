import os
import pathlib
import logging
import time
import wandb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch.cuda.amp import autocast

from datetime import date

from diffalign.neuralnet.transformer_model_with_y import GraphTransformerWithY, GraphTransformerWithYAtomMapPosEmb, PositionalEmbedding
from diffalign.diffusion.noise_schedule import *
from diffalign.diffusion.elbo import ELBOMixin
from diffalign.diffusion.sampling import SamplingMixin
from diffalign.utils import graph, mol
from diffalign.neuralnet.ema_pytorch import EMA
from diffalign.utils.diffusion import helpers
from diffalign.constants import MAX_NODES
import traceback

# A logger for this file
log = logging.getLogger(__name__)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Disable rdkit warnings

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]

class DiscreteDenoisingDiffusion(ELBOMixin, SamplingMixin, nn.Module):
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
        else:
            raise ValueError(f'Unknown architecture: {self.cfg.neuralnet.architecture}')
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

        elif cfg.diffusion.transition=='absorbing_masknoedge':
            self.transition_model, self.transition_model_eval = (AbsorbingStateTransitionMaskNoEdge(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                                       y_classes=self.ydim_output, timesteps=T_,
                                                                       diffuse_edges=cfg.diffusion.diffuse_edges,
                                                                       abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                       node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
                                                                       for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
            self.limit_dist = self.transition_model.get_limit_dist()

        else:
            raise ValueError(f'Unknown transition: {cfg.diffusion.transition}')

    def training_step(self, data, i, device):
        dense_data = graph.to_dense(data=data).to_device(device)
        t_int = torch.randint(1, self.T+1, size=(len(data),1), device=device)
        z_t, _ = self.apply_noise(dense_data, t_int = t_int, transition_model=self.transition_model)

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

        if torch.cuda.is_available(): #and self.model.training:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = self.forward(z_t=z_t, use_pos_encoding_if_applicable=~unconditional)
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
        X, E, y = dense_data.X, dense_data.E, dense_data.y
        device = dense_data.X.device
        # log.info(f"{X.shape}, {E.shape}, {y.shape}")

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
        else:
            pos_encodings = None
        return pos_encodings
    
    def get_pos_encodings_if_relevant(self, z_t):
        if self.cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc':
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
                assert z_t.mol_assignments is not None, 'molecule_assigments is None in forward()'
                if pos_encodings == None: 
                    with autocast(enabled=False):
                    # if pos encs weren't precalculated, e.g., in the sampling loop
                        pos_encodings = self.get_pos_encodings(z_t)
                # zero out the positional encoding if not applicable. This related to CLFG or something like that
                pos_encodings *= use_pos_encoding_if_applicable[:,None,None].to(pos_encodings.device).float()
                if self.cfg.neuralnet.use_ema and not self.training:
                    res = self.ema(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignments)
                else:
                    res = self.model(z_t.X, z_t.E, z_t.y, z_t.node_mask, z_t.atom_map_numbers, pos_encodings, z_t.mol_assignments)
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
    
    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        
        from diffalign.neuralnet.extra_features import ExtraFeatures
        from diffalign.neuralnet.extra_features_molecular import ExtraMolecularFeatures
        self.extra_features_calculator = ExtraFeatures('all', self.dataset_info)
        self.extra_molecular_features_calculator = ExtraMolecularFeatures(self.dataset_info)

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

        return extra_z
    
    @torch.no_grad()
    def evaluate(self, dataloader, data_class, save_samples_as_smiles, epoch=0):
        raise NotImplementedError
    
    @torch.no_grad()
    def evaluate_one_batch(self, true_rxns, sample_chains, epoch):
        raise NotImplementedError