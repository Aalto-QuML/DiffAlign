import torch.nn as nn
from diffalign.neuralnet.transformer_model import GraphTransformer
from diffalign.neuralnet.transformer_model_with_y import GraphTransformerWithY, GraphTransformerWithYAndPosEmb, GraphTransformerWithYAtomMapPosEmb, GraphTransformerWithYAtomMapPosEmbInefficient, GraphTransformerWithYAtomMapPosEmbInefficientStereo, GraphTransformerWithYAtomMapPosEmbInefficientChargesSeparate
from diffalign.neuralnet.transformer_model_stacked import GraphTransformerWithYStacked
from diffalign.diffusion.noise_schedule import *
from diffalign.neuralnet.transformer_for_all_features import GraphTransformerForAllFeatures

"""This file gathers together some code for choosing the right model, transition, and positional encoding based on the configuration file"""

def choose_denoiser(cfg, input_dims, output_dims):

     if cfg.dataset.use_stereochemistry:
          assert cfg.neuralnet.architecture == 'transformer_for_all_features' or cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc_stereo', f'Got: cfg.neuralnet.architecture={cfg.neuralnet.architecture}'
     if cfg.dataset.use_charges_as_features:
          assert cfg.neuralnet.architecture == 'transformer_for_all_features' or cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc_charges_separate', f'Got: cfg.neuralnet.architecture={cfg.neuralnet.architecture}'

     if cfg.neuralnet.architecture=='transformer_for_all_features':
          model = GraphTransformerForAllFeatures(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                            hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                            hidden_dims=cfg.neuralnet.hidden_dims,
                                            output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                            improved=cfg.neuralnet.improved, cfg=cfg, dropout=cfg.neuralnet.dropout,
                                            p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                            p_to_r_init=cfg.neuralnet.p_to_r_init,
                                            input_alignment=cfg.neuralnet.input_alignment)
     elif cfg.neuralnet.architecture=='with_y':
          model = GraphTransformerWithY(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                            hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                            hidden_dims=cfg.neuralnet.hidden_dims,
                                            output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                            improved=cfg.neuralnet.improved, dropout=cfg.neuralnet.dropout)
     elif cfg.neuralnet.architecture=='without_y': # TODO: We don't really need this, right?
          model = GraphTransformer(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                        hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                        hidden_dims=cfg.neuralnet.hidden_dims,
                                        output_dims=output_dims,
                                        act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU())
     elif cfg.neuralnet.architecture=='with_y_pos_enc':
          model = GraphTransformerWithYAndPosEmb(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                                    hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                                    hidden_dims=cfg.neuralnet.hidden_dims,
                                                    output_dims=output_dims,
                                                    act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(), dropout=cfg.neuralnet.dropout)
     elif cfg.neuralnet.architecture=='all_alignments_efficient':
          model = GraphTransformerWithYAtomMapPosEmb(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                            hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                            hidden_dims=cfg.neuralnet.hidden_dims,
                                            output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                            pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                            improved=cfg.neuralnet.improved, cfg=cfg, dropout=cfg.neuralnet.dropout,
                                            p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                            p_to_r_init=cfg.neuralnet.p_to_r_init, alignment_type=cfg.neuralnet.alignment_type,
                                            input_alignment=cfg.neuralnet.input_alignment)
     elif cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc':
          model = GraphTransformerWithYAtomMapPosEmbInefficient(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                            hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                            hidden_dims=cfg.neuralnet.hidden_dims,
                                            output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                            pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                            improved=cfg.neuralnet.improved, cfg=cfg, dropout=cfg.neuralnet.dropout,
                                            p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                            p_to_r_init=cfg.neuralnet.p_to_r_init, alignment_type=cfg.neuralnet.alignment_type,
                                            input_alignment=cfg.neuralnet.input_alignment)
     elif cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc_stereo':
          model = GraphTransformerWithYAtomMapPosEmbInefficientStereo(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                            hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                            hidden_dims=cfg.neuralnet.hidden_dims,
                                            output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                            pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                            improved=cfg.neuralnet.improved, cfg=cfg, dropout=cfg.neuralnet.dropout,
                                            p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                            p_to_r_init=cfg.neuralnet.p_to_r_init, alignment_type=cfg.neuralnet.alignment_type,
                                            input_alignment=cfg.neuralnet.input_alignment)
     elif cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc_charges_separate':
          model = GraphTransformerWithYAtomMapPosEmbInefficientChargesSeparate(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                            hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                            hidden_dims=cfg.neuralnet.hidden_dims,
                                            output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                            pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                            improved=cfg.neuralnet.improved, cfg=cfg, dropout=cfg.neuralnet.dropout,
                                            p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                            p_to_r_init=cfg.neuralnet.p_to_r_init, alignment_type=cfg.neuralnet.alignment_type,
                                            input_alignment=cfg.neuralnet.input_alignment)
     elif cfg.neuralnet.architecture=='with_y_stacked':
          model = GraphTransformerWithYStacked(n_layers=cfg.neuralnet.n_layers, input_dims=input_dims,
                                            hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                            hidden_dims=cfg.neuralnet.hidden_dims,
                                            output_dims=output_dims, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                            pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                            improved=cfg.neuralnet.improved, dropout=cfg.neuralnet.dropout,
                                            p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                            p_to_r_init=cfg.neuralnet.p_to_r_init)
     return model

def choose_transition(cfg, dataset_info, node_type_counts_unnormalized, edge_type_counts_unnormalized):
     
     Xdim_output, Edim_output, ydim_output = dataset_info.output_dims['X'], dataset_info.output_dims['E'], dataset_info.output_dims['y']
     node_idx_to_mask, edge_idx_to_mask = graph.get_index_from_states(atom_decoder=dataset_info.atom_decoder,
                                                                         bond_decoder=dataset_info.bond_decoder,
                                                                         node_states_to_mask=cfg.diffusion.node_states_to_mask,
                                                                         edge_states_to_mask=cfg.diffusion.edge_states_to_mask,
                                                                         device=device)
     
     abs_state_position_e = 0
     abs_state_position_x = dataset_info.atom_decoder.index('Au') 
     abs_state_position_charge = 0 # assumption that the first element in cfg.dataset.atom_charges is the no-charge state

     if cfg.diffusion.transition=='uniform':
          transition_model, transition_model_eval = (DiscreteUniformTransition(x_classes=Xdim_output, e_classes=Edim_output, charge_classes=len(cfg.dataset.atom_charges),
                                                            y_classes=ydim_output, noise_schedule=cfg.diffusion.diffusion_noise_schedule,
                                                            timesteps=T_, diffuse_edges=cfg.diffusion.diffuse_edges,
                                                            node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
                                                            for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()

     elif cfg.diffusion.transition=='marginal':
          node_types = dataset_info.node_types.float()
          x_marginals = node_types / torch.sum(node_types)
          edge_types = dataset_info.edge_types.float()
          e_marginals = edge_types / torch.sum(edge_types)
          # TODO: change these to take the actual marginals
          # TODO: for now using uniform marginals
          chi_marginals = torch.ones(len(dataset_info.atom_chiral_tags)) / len(dataset_info.atom_chiral_tags)
          cha_marginals = torch.ones(len(cfg.dataset.atom_charges)) / len(cfg.dataset.atom_charges)
          bd_marginals = torch.ones(len(dataset_info.bond_dirs)) / len(dataset_info.bond_dirs)
          transition_model, transition_model_eval = (MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                            y_classes=ydim_output, 
                                                            noise_schedule=cfg.diffusion.diffusion_noise_schedule,
                                                            timesteps=T_, 
                                                            chi_marginals=chi_marginals, 
                                                            cha_marginals=cha_marginals, 
                                                            bd_marginals=bd_marginals,
                                                            node_idx_to_mask=node_idx_to_mask,
                                                            edge_idx_to_mask=edge_idx_to_mask,
                                                            diffuse_edges=cfg.diffusion.diffuse_edges)
                                                            for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()

     elif cfg.diffusion.transition=='tokenwise_absorbing':
          transition_model, transition_model_eval = (TokenwiseAbsorbingStateTransition(x_classes=Xdim_output, e_classes=Edim_output,
                                                                 y_classes=ydim_output, timesteps=T_,
                                                                 node_type_counts_unnormalized=node_type_counts_unnormalized, 
                                                                 edge_type_counts_unnormalized=edge_type_counts_unnormalized,
                                                                 abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                 diffuse_edges=cfg.diffusion.diffuse_edges, node_idx_to_mask=node_idx_to_mask,
                                                                 edge_idx_to_mask=edge_idx_to_mask)
                                                                 for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()

     elif cfg.diffusion.transition=='absorbing_masknoedge':
          AbsorbingStateTransitionMaskNoEdge(x_classes=Xdim_output, e_classes=Edim_output,
                                                                      y_classes=ydim_output, charge_classes=len(cfg.dataset.atom_charges), timesteps=cfg.diffusion.diffusion_steps,
                                                                      diffuse_edges=cfg.diffusion.diffuse_edges, abs_state_position_charge=abs_state_position_charge,
                                                                      abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                      node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
          transition_model, transition_model_eval = (AbsorbingStateTransitionMaskNoEdge(x_classes=Xdim_output, e_classes=Edim_output,
                                                                      y_classes=ydim_output, charge_classes=len(cfg.dataset.atom_charges), timesteps=T_,
                                                                      diffuse_edges=cfg.diffusion.diffuse_edges, abs_state_position_charge=abs_state_position_charge,
                                                                      abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                      node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
                                                                      for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()

     elif cfg.diffusion.transition=='tokenwise_absorbing_masknoedge':
          node_types = dataset_info.node_types.float()
          x_marginals = node_types / torch.sum(node_types)
          edge_types = dataset_info.edge_types.float()
          e_marginals = edge_types / torch.sum(edge_types)
          transition_model, transition_model_eval = (TokenwiseAbsorbingStateTransitionMaskNoEdge(x_classes=Xdim_output, e_classes=Edim_output,
                                                                 y_classes=ydim_output, timesteps=T_,
                                                                 node_type_counts_unnormalized=node_type_counts_unnormalized, 
                                                                 edge_type_counts_unnormalized=edge_type_counts_unnormalized, 
                                                                 diffuse_edges=cfg.diffusion.diffuse_edges, abs_state_position_e=abs_state_position_e, 
                                                                 abs_state_position_x=abs_state_position_x,
                                                                 node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask)
                                                                 for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()

     elif cfg.diffusion.transition=='blocktokenwise_absorbing_masknoedge':
          node_types = dataset_info.node_types.float()
          x_marginals = node_types / torch.sum(node_types)
          edge_types = dataset_info.edge_types.float()
          e_marginals = edge_types / torch.sum(edge_types)
          transition_model, transition_model_eval = (BlockTokenwiseAbsorbingStateTransitionMaskNoEdge(x_classes=Xdim_output, e_classes=Edim_output,
                                                                                y_classes=ydim_output, timesteps=T_,
                                                                                node_type_counts_unnormalized=node_type_counts_unnormalized, 
                                                                                edge_type_counts_unnormalized=edge_type_counts_unnormalized, 
                                                                                diffuse_edges=cfg.diffusion.diffuse_edges, abs_state_position_e=abs_state_position_e, 
                                                                                abs_state_position_x=abs_state_position_x,
                                                                                node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask) 
                                                                                for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()
     
     elif cfg.diffusion.transition=='nodes_before_edges':
          transition_model, transition_model_eval = (NodesBeforeEdges(x_classes=Xdim_output, e_classes=Edim_output,
                                                  y_classes=ydim_output, timesteps=T_,
                                                  abs_state_position_e=abs_state_position_e, 
                                                  abs_state_position_x=abs_state_position_x, 
                                                  diffuse_edges=cfg.diffusion.diffuse_edges, node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask,
                                                  num_node_steps=cfg.diffusion.num_node_steps)
                                                  for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()
          
     elif cfg.diffusion.transition=='edges_before_nodes':
          transition_model, transition_model_eval = (EdgesBeforeNodes(x_classes=Xdim_output, e_classes=Edim_output,
                                                  y_classes=ydim_output, timesteps=T_,
                                                  abs_state_position_e=abs_state_position_e, 
                                                  abs_state_position_x=abs_state_position_x, 
                                                  diffuse_edges=cfg.diffusion.diffuse_edges, node_idx_to_mask=node_idx_to_mask, edge_idx_to_mask=edge_idx_to_mask,
                                                  num_node_steps=cfg.diffusion.num_node_steps)
                                                  for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()
     elif cfg.diffusion.transition=='marginaledge_absorbingnode':
          edge_types = dataset_info.edge_types.float()
          e_marginals = edge_types / torch.sum(edge_types)
          transition_model, transition_model_eval = (MarginalEdgesMaskNodesTransition(x_classes=Xdim_output, e_marginals=e_marginals, 
                                                                 y_classes=ydim_output, 
                                                                 noise_schedule=cfg.diffusion.diffusion_noise_schedule,
                                                                 timesteps=T_,
                                                                 abs_state_position_e=abs_state_position_e, abs_state_position_x=abs_state_position_x,
                                                                 diffuse_edges=cfg.diffusion.diffuse_edges)
                                                                 for T_ in [cfg.diffusion.diffusion_steps, cfg.diffusion.diffusion_steps_eval])
          limit_dist = transition_model.get_limit_dist()
     else: 
          assert f'Transition model undefined. Got {cfg.diffusion.transition}\n.'

     return transition_model, transition_model_eval, limit_dist

def forward_choose(cfg, model, ema, z_t, training, pos_emb_module):
     if cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc' or \
          cfg.neuralnet.architecture=='all_alignments_efficient' or \
          cfg.neuralnet.architecture=='transformer_for_all_features' or \
          cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc_stereo' or \
          cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc_charges_separate':
            assert z_t.mol_assignment is not None, 'molecule_assigments is None in forward()'
            if cfg.neuralnet.architecture=='transformer_for_all_features':
                pos_encoding = z_t.pos_encoding #get_pos_encodings(z_t)
            else:
                 pos_encoding = pos_encoding_choose(cfg, pos_emb_module, z_t)
            if cfg.neuralnet.use_ema and not training:
                res = ema(z_t.X, z_t.E, z_t.y, z_t.atom_charges, z_t.atom_chiral, z_t.bond_dirs,
                    z_t.node_mask, z_t.atom_map_numbers, pos_encoding, z_t.mol_assignment)
            else:
                res = model(z_t.X, z_t.E, z_t.y, z_t.atom_charges, z_t.atom_chiral, z_t.bond_dirs,
                    z_t.node_mask, z_t.atom_map_numbers, pos_encoding, z_t.mol_assignment)
     elif cfg.neuralnet.architecture=='with_y_stacked':
          if cfg.neuralnet.use_ema and not training:
               res = ema(z_t.X, z_t.E, z_t.y, z_t.atom_charges, z_t.atom_chiral, z_t.bond_dirs, z_t.node_mask, 
                         z_t.atom_map_numbers, z_t.pos_encoding, z_t.mol_assignment, cfg.dataset.atom_types)
          else:
               res = model(z_t.X, z_t.E, z_t.y, z_t.atom_charges, z_t.atom_chiral, z_t.bond_dirs, 
                           z_t.node_mask, z_t.atom_map_numbers, z_t.pos_encoding, z_t.mol_assignment, cfg.dataset.atom_types)
     elif cfg.neuralnet.architecture=='with_y':
          if cfg.neuralnet.use_ema and not training:
               res = ema(z_t.X, z_t.E, z_t.y, z_t.node_mask)
          else:
               res = model(z_t.X, z_t.E, z_t.y, z_t.node_mask)
     return res

def forward_choose_careful(cfg, model, ema, z_t, training, pos_emb_module):
     try:
          forward_choose(cfg, model, ema, z_t, training, pos_emb_module)
     except:
          torch.save(model.state_dict(), "failedmodel.pt") 
          import pickle
          pickle.dump(z_t, open('z_t.pkl', 'wb'))
          print(f'Error in forward process!!')
          exit()

def pos_encoding_choose(cfg, pos_emb_module, z_t):
     if cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc' or \
          cfg.neuralnet.architecture == 'with_y_atommap_number_pos_enc_stereo' or \
          cfg.neuralnet.architecture=='with_y_atommap_number_pos_enc_charges_separate':
          if cfg.neuralnet.pos_encoding_type == 'smiles_pos_enc':
               pos_encodings = pos_emb_module.matched_positional_encodings_sinusoidal(z_t.atom_map_numbers, z_t.mol_assignment, direction = 'retro')
          elif cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc':
               pos_encodings = pos_emb_module.matched_positional_encodings_laplacian(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignment, cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
          elif cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc_gpu':
               pos_encodings = pos_emb_module.matched_positional_encodings_laplacian_gpu(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignment, cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
          elif cfg.neuralnet.pos_encoding_type == 'infoleak_pos_enc':
               log.info("Using info leak positional encoding, for illustration purposes only")
               # TODO: Implement this
               pos_encodings = pos_emb_module.matched_positional_encodings_infoleak(z_t.atom_map_numbers, z_t.mol_assignment, direction = 'retro')
          elif cfg.neuralnet.pos_encoding_type == 'gaussian_pos_enc':
               pos_encodings = pos_emb_module.matched_positional_encodings_gaussian(z_t.atom_map_numbers, z_t.mol_assignment)
          elif cfg.neuralnet.pos_encoding_type == 'laplacian_and_smiles_pos_enc_with_product_id':
               # positional encodings for the atom-mapped atoms
               pos_encodings_matched = pos_emb_module.matched_positional_encodings_laplacian(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignment, cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
               # positional encodings for the rest of the atoms on the reactant side
               pos_encodings_smiles = pos_emb_module.additional_positional_encodings_smiles(z_t.atom_map_numbers, z_t.mol_assignment)
               pos_encodings_product_id = pos_emb_module.additional_positional_encodings_product_id(z_t.mol_assignment)
               pos_encodings = pos_encodings_matched + pos_encodings_smiles + pos_encodings_product_id # these should be disjoint, so we can add them
          elif cfg.neuralnet.pos_encoding_type == 'no_pos_enc':
               pos_encodings = torch.zeros(z_t.X.shape[0], z_t.X.shape[1], 1, device=z_t.X.device) # This will get padded later
          elif cfg.neuralnet.pos_encoding_type == 'unaligned_laplacian_pos_enc':
               pos_encodings = pos_emb_module.unaligned_positional_encodings_laplacian(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignment, cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
          else:
               raise ValueError(f'pos_encoding_type {cfg.neuralnet.pos_encoding_type} not recognized')
     elif cfg.neuralnet.architecture == 'with_y_stacked':
          model = model.module if isinstance(model, nn.DataParallel) else model
          
          suno_number = cfg.dataset.atom_types.index("SuNo")
          reaction_side_separation_index = (z_t.X.argmax(-1) == suno_number).nonzero(as_tuple=True)[1]
          
          if cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc_gpu':
               # This needs to be designed as:
               # 1. Cut the correct part of E
               # 2. Calculate the eigendecomposition
               # 3. Place it in the correct part with the code that we already have elsewhere, maybe just reuse the code that we already have? And then just cut it. Let's see...
               pos_encodings = model.pos_emb_module.matched_positional_encodings_laplacian(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignment, cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
               pos_encodings, _ = model.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index) # ... this could be done in forward as well, more efficient with multiple GPUs. Actually both parts hmm
          elif cfg.neuralnet.pos_encoding_type == 'laplacian_pos_enc':
               # 3. Place it in the correct part with the code that we already have elsewhere, maybe just reuse the code that we already have? And then just cut it. Let's see...
               pos_encodings = model.pos_emb_module.matched_positional_encodings_laplacian_scipy(z_t.E.argmax(-1), z_t.atom_map_numbers, z_t.mol_assignment, cfg.neuralnet.num_lap_eig_vectors, direction = 'retro')
               pos_encodings, _ = model.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index)
          else:
               pos_encodings = torch.zeros(z_t.X.shape[0], z_t.X.shape[1], model.input_dim_X, device=z_t.X.device)
               pos_encodings, _ = model.cut_reaction_reactant_part_X_only(pos_encodings, reaction_side_separation_index)
     else:
          pos_encodings = None
     return pos_encodings