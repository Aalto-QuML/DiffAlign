import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions

import seaborn as sns
import numpy as np
import imageio
import os
import wandb
import logging

log = logging.getLogger(__name__)

from diffalign.data.graph import PlaceHolder
from diffalign.data import graph
from diffalign.data import mol
import copy

import torch
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-6

import torch
import torch.nn.functional as F
import copy
import numpy as np

# def grad_log_p_y_x_t_approx(model, z_t, a, b, gamma, idx):
#     """
#     Computes the gradient of the specified function with respect to z_t.

#     Parameters:
#     - model: The diffusion_abstract model that has the forward method implemented
#     - z_t: The noisy placeholder object
#     - a: Scalar value a
#     - b: Scalar value b
#     - gamma: Scalar value gamma, strength of guidance. Can be positive or negative. If positive, then we increase the amount of the nodes in idx. If negative, then the opposite.
#     - idx: Chosen index along the last dimension of z_0

#     Returns:
#     - Gradient of the function with respect to z_t
#     """
#     # Ensure that the input tensor requires gradient
#     with torch.enable_grad():
#         assert z_t.X.shape[0] == 1 # initially works only for batch size 1

#         z_t.X.requires_grad_(True)
#         z_t.E.requires_grad_(True)
#         z_t.atom_chiral.requires_grad_(True)
#         z_t.bond_dirs.requires_grad_(True)
#         z_t.atom_charges.requires_grad_(True)
#         # z_t.pos_encoding = z_t.pos_encoding.detach()

#         # Forward pass through the model
#         z_0 = model(z_t)

#         # Compute the softmax along the last dimension (only choose non-atom-mapped stuff and only the reactant side)
#         softmax_z_0 = torch.softmax(z_0.X, dim=-1) * ((z_t.atom_map_numbers == 0) * (z_t.mol_assignment != z_t.mol_assignment.max(-1, keepdim=True)[0]))[..., None]
        
#         # Sum along the last dimension at the specified index
#         X_0_sum_softmax = torch.sum(softmax_z_0[..., idx], dim=-1)# shape should be (batch_size,)

#         # Compute the desired function
#         output = F.logsigmoid((X_0_sum_softmax - a) / b).sum()
#         # output = X_0_sum_softmax.sum()

#         # Compute the gradient with respect to X for the batch
#         # How to get this separately for each batch element? Hmm okay I guess need to do a loop
#         # with torch.autograd.detect_anomaly():
#         output.backward()

#         # Compute gradients using autograd
#         # gradients = torch.autograd.grad(outputs=output, inputs=[z_t.X, z_t.E, z_t.atom_chiral, z_t.bond_dirs, z_t.atom_charges], create_graph=True)
        
#         # Extract the gradient
#         gradient = copy.deepcopy(z_t)
#         gradient.X = z_t.X.grad * gamma
#         gradient.E = z_t.E.grad * gamma
#         gradient.atom_chiral = z_t.atom_chiral.grad * gamma
#         gradient.bond_dirs = z_t.bond_dirs.grad * gamma
#         gradient.atom_charges = z_t.atom_charges.grad * gamma
    
#     return z_0, gradient

def grad_log_p_y_x_t_approx(model, z_t, a, b, gamma, idx):
    """
    Computes the gradient of the specified function with respect to z_t for batched input.

    Parameters:
    - model: The diffusion_abstract model that has the forward method implemented
    - z_t: The noisy placeholder object (batched)
    - a: Scalar value a or tensor of shape (batch_size,)
    - b: Scalar value b or tensor of shape (batch_size,)
    - gamma: Scalar value gamma or tensor of shape (batch_size,), strength of guidance.
    - idx: Chosen index along the last dimension of z_0

    Returns:
    - z_0: The model output
    - gradient: Gradient of the function with respect to z_t
    """
    batch_size = z_t.X.shape[0]

    # Ensure that the input tensors require gradient
    with torch.enable_grad():
        # z_t.X.requires_grad_(True)
        # z_t.E.requires_grad_(True)
        # z_t.atom_chiral.requires_grad_(True)
        # z_t.bond_dirs.requires_grad_(True)
        # z_t.atom_charges.requires_grad_(True)

        # Split into smaller chunks of 20
        chunk_size = 20
        # num_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
        chunk_indices = np.linspace(0,batch_size,batch_size//chunk_size+1).astype('int')
        
        z_0 = copy.deepcopy(z_t)
        gradient = copy.deepcopy(z_t)
        
        for i in range(len(chunk_indices[:-1])):
            z_t_ = z_t.subset_by_idx(chunk_indices[i], chunk_indices[i+1])
            chunk_size = chunk_indices[i+1] - chunk_indices[i]

            # Forward pass through the model

            z_t_.X.requires_grad_(True)
            z_t_.E.requires_grad_(True)
            z_t_.atom_chiral.requires_grad_(True)
            z_t_.bond_dirs.requires_grad_(True)
            z_t_.atom_charges.requires_grad_(True)

            z_0_ = model(z_t_)

            # Compute the softmax along the last dimension
            softmax_z_0_ = torch.softmax(z_0_.X, dim=-1) * ((z_t_.atom_map_numbers == 0) * (z_t_.mol_assignment != z_t_.mol_assignment.max(-1, keepdim=True)[0]))[..., None]
            
            # Sum along the last dimension at the specified index
            X_0_sum_softmax = torch.sum(softmax_z_0_[..., idx], dim=-1)  # shape: (batch_size,)

            # Compute the desired function for each batch element
            if not isinstance(a, torch.Tensor):
                a = torch.full((chunk_size,), a, device=z_t_.X.device)
            if not isinstance(b, torch.Tensor):
                b = torch.full((chunk_size,), b, device=z_t_.X.device)
            
            output = F.logsigmoid((X_0_sum_softmax - a) / b).sum()

            # Compute the gradient
            output.backward()

            # Extract and scale the gradients
            if not isinstance(gamma, torch.Tensor):
                gamma = torch.full((chunk_size,), gamma, device=z_t_.X.device)
            
            gradient.X[chunk_indices[i]:chunk_indices[i+1]] = z_t_.X.grad * gamma.view(-1, 1, 1)
            gradient.E[chunk_indices[i]:chunk_indices[i+1]] = z_t_.E.grad * gamma.view(-1, 1, 1, 1)
            if not z_t_.atom_chiral.grad == None:
                gradient.atom_chiral[chunk_indices[i]:chunk_indices[i+1]] = z_t_.atom_chiral.grad * gamma.view(-1, 1)
            else:
                gradient.atom_chiral[chunk_indices[i]:chunk_indices[i+1]] = 0
            if not z_t_.bond_dirs.grad == None:
                gradient.bond_dirs[chunk_indices[i]:chunk_indices[i+1]] = z_t_.bond_dirs.grad * gamma.view(-1, 1, 1)
            else:
                gradient.bond_dirs[chunk_indices[i]:chunk_indices[i+1]] = 0
            if not z_t_.atom_charges.grad == None:
                gradient.atom_charges[chunk_indices[i]:chunk_indices[i+1]] = z_t_.atom_charges.grad * gamma.view(-1, 1)
            else:
                gradient.atom_charges[chunk_indices[i]:chunk_indices[i+1]] = 0

            z_0.X[chunk_indices[i]:chunk_indices[i+1]] = z_0_.X.detach()
            z_0.E[chunk_indices[i]:chunk_indices[i+1]] = z_0_.E.detach()
            z_0.atom_chiral[chunk_indices[i]:chunk_indices[i+1]] = z_0_.atom_chiral.detach()
            z_0.bond_dirs[chunk_indices[i]:chunk_indices[i+1]] = z_0_.bond_dirs.detach()
            z_0.atom_charges[chunk_indices[i]:chunk_indices[i+1]] = z_0_.atom_charges.detach()
            
    return z_0, gradient

def get_batchsize_of_data(data):
    if type(data)!=graph.PlaceHolder:
        bs = data.batch.max()+1
    else:
        bs = data.X.shape[0]
        
    return bs

def average_rxn_scores(scores_list, counts_of_samples_in_list_elements):
    '''
        Averages together the scores in scores_list. 
        
        input:
            scores_list: list of dicts containing the scores
            counts_of_samples_in_list_elements: list of integers with the number of samples used to calculate the scores in scores_list
        output:
            avg: averaged scores
    '''
    total_samples = sum(counts_of_samples_in_list_elements)
    avg_scores = {}
    for i, scores in enumerate(scores_list):
        for metric in scores_list[0].keys():
            if metric not in avg_scores.keys():
                if type(scores[metric])==list:
                    avg_scores[metric] = [scores[metric]]
                else:
                    avg_scores[metric] = scores[metric] * counts_of_samples_in_list_elements[i] / total_samples
            else:
                if type(avg_scores[metric])==list:
                    avg_scores[metric].extend(scores[metric])
                else:
                    avg_scores[metric] += scores[metric]  * counts_of_samples_in_list_elements[i] / total_samples
    return avg_scores

def accumulate_rxn_scores(acc_scores, new_scores, total_iterations):
    '''
        Updates the acc_scores with new metric averages taking into account the new_scores.
        
        input:
            acc_scores: accumulated scores state
            new_scores: new_scores to add to the accumulation
            total_iterations: total number of batches considered. 
        output:
            acc_scores: accumulated scores state with the new_scores added.
    '''
    for metric in new_scores.keys():
        if type(new_scores[metric])==list: # accumulates the plots
            if acc_scores[metric]==0:
                acc_scores[metric] = new_scores[metric]
            else:
                acc_scores[metric].extend(new_scores[metric])
        else:
            acc_scores[metric] += new_scores[metric].mean()/total_iterations
        
    return acc_scores
        
def mean_without_masked(graph_obj, mask_X, mask_E, diffuse_nodes=True, diffuse_edges=True, avg_over_batch=False):
    '''
        Takes graph object (of type PlaceHolder) and returns the mean of the X and E values not considering masked elements
        
        input:
            graph: PlaceHolder object, with X=nodes and E=adjacency matrix
            mask_nodes: a boolean mask with True corresponding to the nodes we want to keep and False those we want to discard
        output:
            res: scalar (float), corresponding to the mean not taking into account the masked elements
    '''
    # get masks
    # mask_X, mask_E = graph.from_mask_to_maskX_maskE(mask_nodes=mask_nodes, mask_edges=mask_edges, 
    #                                                 shape_X=graph_obj.X.shape, shape_E=graph_obj.E.shape)

    # true in mask = keep node/edge
    # we use max to remove the feature dimension, and flatten to select nodes/edges across batches
    # mask_X = mask_X.max(-1)[0]
    # mask_E = mask_E.max(-1)[0].flatten(1,-1)
    # mask_E = mask_E.flatten(1, -1)
    
    # flatten data points
    # Note: we sum because this function handles CE and KL of categorical distributions
    # we first sum over the categories to get the quantity corresponding to the element (e.g. node in a graph)
    # then choose the elements we want to consider in the graph (e.g. non-padding nodes)
    # UPDATE NOTE: We don't sum over the last dimension anymore here, we do it earlier on already
    # graph_obj.X = #graph_obj.X.sum(-1)
    # graph_obj.E = graph_obj.E.flatten(1,-1)#graph_obj.E.sum(-1).flatten(1,-1)
    
    if len(mask_X.shape) == 3:
        mask_X = mask_X[...,0]
    if len(mask_E.shape) == 4:
        mask_E = mask_E[...,0]

    # take only elements we want to consider based on the mask
    graph_obj.X = graph_obj.X*mask_X
    graph_obj.E = graph_obj.E*mask_E
        
    # ignore padding nodes in when averaging
    # graph_obj.X = (bs, max_nodes) => term_per_dim
    if avg_over_batch:
        mean_X = graph_obj.X.sum()/mask_X.sum() 
        mean_E = graph_obj.E.sum()/mask_E.sum()
    else:
        mean_X = graph_obj.X.sum(1)/mask_X.sum(1) 
        mean_E = graph_obj.E.sum((1,2))/mask_E.sum((1,2))
    
    if diffuse_edges and diffuse_nodes:
        res = mean_X+mean_E
    elif diffuse_edges:
        res = mean_E
    elif diffuse_nodes:
        res = mean_X

    return res

def save_as_smiles(data, atom_types, bond_types, output_filename='default'):
    smiles = mol.rxn_from_graph_supernode(data=data, atom_types=atom_types, bond_types=bond_types)
    
    file_path = os.path.join(os.getcwd(), f'{output_filename}_output.gen')
    open(file_path, 'a').writelines(smiles+'\n')
    
def kl_prior(prior, limit, eps=EPS):
    """
        Computes the (point-wise) KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        
        input:
            prior: PlaceHolder graph object
            limit: PlaceHolder graph object
            eps: small offset value
        output:
            kl_prior_: PlaceHolder graph object
    """

    # log in kl because pytorch expects input (pred) in log-prob
    # can give true (target) in log-prob too but need to specify it with log_target=True
    # add small eps (e.g. 1e-20) to handle 0 in prob 
    # compute KL(true||pred) = KL(target||input)

    kl_x = F.kl_div(input=(prior.X+eps).log(), target=limit.X, reduction='none').sum(-1)
    kl_e = F.kl_div(input=(prior.E+eps).log(), target=limit.E, reduction='none').sum(-1)
    
    kl_prior_ = graph.PlaceHolder(X=kl_x, E=kl_e, node_mask=prior.node_mask, y=torch.zeros(1, dtype=torch.float))
    
    return kl_prior_

def reconstruction_logp(orig, pred_t0):
    # Normalize predictions

    # get prob pred for a given true rxn
    # E_{q(x_0)} E_{q(x_1|x_0)} log p(x_0|x_1)
    # x_0 ~ (q(x_0) = dataset) => x_1 ~ q(x_1|x_0) (by noising) => logits/probs p(x_0|x_1) (by denoising) 
    # => p(x_0|x_1)*x_0 to choose the probability of a specific category (x_0 is one-hot encoded)
    loss_term_0_x = (orig.X*pred_t0.X).sum(-1)
    loss_term_0_e = (orig.E*pred_t0.E).sum(-1)
    
    loss_term_0 = graph.PlaceHolder(X=loss_term_0_x, E=loss_term_0_e, node_mask=orig.node_mask, y=torch.ones(1, dtype=torch.float))

    return loss_term_0

def ce(cfg, pred, dense_true, lambda_E, outside_reactant_mask_nodes, outside_reactant_mask_edges, log=False):
        
    #dense_true = graph.to_dense(data=discrete_true)
    #true_X, true_E = discrete_dense_true.X, discrete_dense_true.E
    #pred_X, pred_E = pred.X, pred.E
        
    #true_X = true_X.reshape(-1,true_X.size(-1))  # (bs * n, dx)
    #true_E = true_E.reshape(-1,true_E.size(-1))  # (bs * n * n, de)
    #pred_X = pred_X.reshape(-1,pred_X.size(-1))  # (bs * n, dx)
    #pred_E = pred_E.reshape(-1,pred_E.size(-1))  # (bs * n * n, de)

    # Remove one-hot encoded rows that are all 0 
    # masked in batch sense: added for padding
    # padding_X = (true_X!=0.).any(dim=-1) # (bs*n,)
    # padding_E = (true_E!=0.).any(dim=-1) # (bs*n*n,)

    # flat_true_X = true_X[padding_X,:] # (bs*n,)
    # flat_pred_X = pred_X[padding_X,:] # (bs*n,)

    # flat_true_E = true_E[padding_E,:] # (bs*n*n,)
    # flat_pred_E = pred_E[padding_E,:] # (bs*n*n,)
    
    # Remove other masked nodes or edges
    #if mask_nodes is not None: 
    #    flat_true_X = true_X[mask_nodes,:]
    #    flat_pred_X = pred_X[mask_nodes,:]
    #if mask_edges is not None: 
    #    flat_true_E = true_E[mask_edges,:]
    #    flat_pred_E = pred_E[mask_edges,:]
    if len(outside_reactant_mask_nodes.shape) == 3:
        outside_reactant_mask_nodes = outside_reactant_mask_nodes[...,0]
    if len(outside_reactant_mask_edges.shape) == 4:
        outside_reactant_mask_edges = outside_reactant_mask_edges[...,0]

    diag_mask = torch.eye(dense_true.X.shape[1], dtype=torch.bool, device=dense_true.X.device)[None].repeat(dense_true.X.shape[0], 1, 1)

    if cfg.train.equal_weight_on_all_reactions:
        bs, n, dx = dense_true.X.shape
        de = dense_true.E.shape[-1]
        loss_X_noreduce = F.cross_entropy(pred.X.reshape(-1,dx), dense_true.X.argmax(-1).reshape(-1), reduction='none').reshape(bs,n) # shape (bs, n)
        loss_E_noreduce = F.cross_entropy(pred.E.reshape(-1,de), dense_true.E.argmax(-1).reshape(-1), reduction='none').reshape(bs,n,n) # shape (bs, n, n)
        loss_atom_charges_noreduce, loss_atom_chiral_noreduce, loss_bond_dirs_noreduce = torch.zeros(bs, n, device=pred.X.device), torch.zeros(bs, n, device=pred.X.device), torch.zeros(bs, n, n, device=pred.X.device)
        if cfg.dataset.use_charges_as_features:
            dc = dense_true.atom_charges.shape[-1]
            assert cfg.dataset.with_formal_charge_in_atom_symbols == False
            loss_atom_charges_noreduce = F.cross_entropy(pred.atom_charges.reshape(-1,dc), dense_true.atom_charges.argmax(-1).reshape(-1), reduction='none').reshape(bs,n)
        if cfg.dataset.use_stereochemistry:
            dchi, dbond = dense_true.atom_chiral.shape[-1], dense_true.bond_dirs.shape[-1]
            loss_atom_chiral_noreduce = F.cross_entropy(pred.atom_chiral.reshape(-1,dchi), dense_true.atom_chiral.argmax(-1).reshape(-1), reduction='none').reshape(bs,n)
            loss_bond_dirs_noreduce = F.cross_entropy(pred.bond_dirs.reshape(-1,dbond), dense_true.bond_dirs.argmax(-1).reshape(-1), reduction='none').reshape(bs,n,n)
        # Sum the losses over the last dimension & average taking into account the mask
        loss_X, loss_E, loss_atom_charges, loss_atom_chiral, loss_bond_dirs = torch.tensor(0., device=pred.X.device), torch.tensor(0., device=pred.X.device), torch.tensor(0., device=pred.X.device), torch.tensor(0., device=pred.X.device), torch.tensor(0., device=pred.X.device)
        for i in range(bs):
            loss_X += (loss_X_noreduce[i]*(~outside_reactant_mask_nodes[i])).sum() / (~outside_reactant_mask_nodes[i]).sum()
            loss_E += (loss_E_noreduce[i][(~outside_reactant_mask_edges[i]) & (~diag_mask[i])]).sum() / ((~outside_reactant_mask_edges[i]) & (~diag_mask[i])).sum()
            loss_atom_charges += (loss_atom_charges_noreduce[i]*(~outside_reactant_mask_nodes[i])).sum() / (~outside_reactant_mask_nodes[i]).sum()
            loss_atom_chiral += (loss_atom_chiral_noreduce[i]*(~outside_reactant_mask_nodes[i])).sum() / (~outside_reactant_mask_nodes[i]).sum()
            loss_bond_dirs += (loss_bond_dirs_noreduce[i][(~outside_reactant_mask_edges[i]) & (~diag_mask[i])]).sum() / ((~outside_reactant_mask_edges[i]) & (~diag_mask[i])).sum()
    else:
        flat_true_X_reactants = dense_true.X[~outside_reactant_mask_nodes].argmax(-1)
        flat_true_E_reactants = dense_true.E[(~outside_reactant_mask_edges) & (~diag_mask)].argmax(-1)
        flat_pred_X_reactants = pred.X[~outside_reactant_mask_nodes]
        flat_pred_E_reactants = pred.E[~outside_reactant_mask_edges & (~diag_mask)]

        #flat_true_X_discrete = flat_true_X.argmax(dim=-1) # (bs*n,)
        #flat_true_E_discrete = flat_true_E.argmax(dim=-1) # (bs*n*n,)

        loss_X = F.cross_entropy(flat_pred_X_reactants, flat_true_X_reactants, reduction='none')
        loss_E = F.cross_entropy(flat_pred_E_reactants, flat_true_E_reactants, reduction='none')
        # Note here we want to start looking into the config
        loss_atom_charges, loss_atom_chiral, loss_bond_dirs = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        if cfg.dataset.use_charges_as_features:
            assert cfg.dataset.with_formal_charge_in_atom_symbols == False
            flat_true_atom_charges_reactants = dense_true.atom_charges[~outside_reactant_mask_nodes].argmax(-1)
            flat_pred_atom_charge_reactants = pred.atom_charges[~outside_reactant_mask_nodes]
            loss_atom_charges = F.cross_entropy(flat_pred_atom_charge_reactants, flat_true_atom_charges_reactants, reduction='none')
        if cfg.dataset.use_stereochemistry:
            flat_true_atom_chiral_reactants = dense_true.atom_chiral[~outside_reactant_mask_nodes].argmax(-1)
            flat_pred_atom_chiral_reactants = pred.atom_chiral[~outside_reactant_mask_nodes]
            loss_atom_chiral = F.cross_entropy(flat_pred_atom_chiral_reactants, flat_true_atom_chiral_reactants, reduction='none')
            flat_true_bond_dirs_reactants = dense_true.bond_dirs[(~outside_reactant_mask_edges) & (~diag_mask)].argmax(-1)
            flat_pred_bond_dirs_reactants = pred.bond_dirs[(~outside_reactant_mask_edges) & (~diag_mask)]
            loss_bond_dirs = F.cross_entropy(flat_pred_bond_dirs_reactants, flat_true_bond_dirs_reactants, reduction='none')

    batch_ce = loss_X.mean() + lambda_E * loss_E.mean() + loss_atom_charges.mean() + loss_atom_chiral.mean() + loss_bond_dirs.mean()
    # NOTE: sum vs. mean makes a huge difference over here. 

    return loss_X, loss_E, loss_atom_charges, loss_atom_chiral, loss_bond_dirs, batch_ce

def get_p_zs_given_zt_old2(transition_model, t_array, pred, z_t, return_prob=False,
                      temperature_scaling=1.0):
    """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""

    # sample z_s
    bs, n, dxs = z_t.X.shape
    device = z_t.X.device

    # Retrieve transition matrices
    Qtb = transition_model.get_Qt_bar(t_array, device)
    Qsb = transition_model.get_Qt_bar(t_array - 1, device)
    Qt = transition_model.get_Qt(t_array, device)
            
    # Normalize predictions
    pred_X = F.softmax(pred.X, dim=-1) # bs, n, d0
    pred_E = F.softmax(pred.E, dim=-1) # bs, n, n, d0
    p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(X_t=z_t.X, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X) # shape (bs, n, [x_{0}], [x_{t-1}])
    p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(X_t=z_t.E, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)

    # Dim of these two tensors: bs, N, d0, d_t-1
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1 # p(x_{t-1}|x_t) = q(x_{t-1},x_t| x_0)p(x_0|x_t)
    unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
    unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5 # in case pred is 0?
    prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

    pred_E_ = pred_E.reshape((bs, -1, pred_E.shape[-1]))
    weighted_E = pred_E_.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
    unnormalized_prob_E = weighted_E.sum(dim=-2)
    unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
    prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
    prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
    
    prob_s = z_t.get_new_object(X=prob_X, E=prob_E).mask(z_t.node_mask)
    #prob_s = graph.PlaceHolder(X=prob_X.clone(), E=prob_E.clone(), y=z_t.y, node_mask=z_t.node_mask, atom_map_numbers=z_t.atom_map_numbers).mask(z_t.node_mask)
    
    if return_prob: return prob_s
    
    sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=z_t.node_mask)

    X_s = F.one_hot(sampled_s.X, num_classes=z_t.X.shape[-1]).float()
    E_s = F.one_hot(sampled_s.E, num_classes=z_t.E.shape[-1]).float()

    assert (E_s == torch.transpose(E_s, 1, 2)).all()
    assert (z_t.X.shape == X_s.shape) and (z_t.E.shape == E_s.shape)

    out_one_hot = z_t.get_new_object(X=X_s, E=E_s, y=torch.zeros(z_t.y.shape[0], 0).to(device))
    out_discrete = z_t.get_new_object(X=X_s, E=E_s, y=torch.zeros(z_t.y.shape[0], 0).to(device))

    # out_one_hot = graph.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(z_t.y.shape[0], 0).to(device), atom_map_numbers=z_t.atom_map_numbers)
    # out_discrete = graph.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(z_t.y.shape[0], 0).to(device), atom_map_numbers=z_t.atom_map_numbers)
    
    return out_one_hot.mask(z_t.node_mask).type_as(z_t.y), out_discrete.mask(z_t.node_mask, collapse=True).type_as(z_t.y)

def get_p_zs_given_zt(transition_model, t_array, pred, z_t, log=False):
    # sample z_s
    bs, n, dxs = z_t.X.shape
    device = z_t.X.device
    print(f'in get_p_zs_given_zt, device: {device}')

    # Retrieve transition matrices
    Qtb = transition_model.get_Qt_bar(t_array, device)
    Qsb = transition_model.get_Qt_bar(t_array - 1, device)
    Qt = transition_model.get_Qt(t_array, device)

    # Normalize predictions
    #pred_X = F.softmax(pred.X, dim=-1) # bs, n, d0
    #pred_E = F.softmax(pred.E, dim=-1) # bs, n, n, d0

    p_s_and_t_given_0_X = compute_posterior_distribution(M=F.softmax(pred.X, dim=-1), M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X, log=log)
    p_s_and_t_given_0_E = compute_posterior_distribution(M=F.softmax(pred.E, dim=-1), M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E, log=log)
    p_s_and_t_given_0_atom_charges = compute_posterior_distribution(M= F.softmax(pred.atom_charges, dim=-1), M_t=z_t.atom_charges, Qt_M=Qt.atom_charges, Qsb_M=Qsb.atom_charges, Qtb_M=Qtb.atom_charges, log=log)
    p_s_and_t_given_0_atom_chiral = compute_posterior_distribution(M= F.softmax(pred.atom_chiral, dim=-1), M_t=z_t.atom_chiral, Qt_M=Qt.atom_chiral, Qsb_M=Qsb.atom_chiral, Qtb_M=Qtb.atom_chiral, log=log)
    p_s_and_t_given_0_bond_dirs = compute_posterior_distribution(M= F.softmax(pred.bond_dirs, dim=-1), M_t=z_t.bond_dirs, Qt_M=Qt.bond_dirs, Qsb_M=Qsb.bond_dirs, Qtb_M=Qtb.bond_dirs, log=log)

    prob_s = z_t.get_new_object(X=p_s_and_t_given_0_X, E=p_s_and_t_given_0_E,
                                atom_charges=p_s_and_t_given_0_atom_charges,
                                atom_chiral=p_s_and_t_given_0_atom_chiral,
                                bond_dirs=p_s_and_t_given_0_bond_dirs).mask(z_t.node_mask)

    return prob_s

def get_p_zs_given_zt_old(transition_model, t_array, pred, z_t, return_prob=False,
                      temperature_scaling=1.0):
    """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""

    # sample z_s
    bs, n, dxs = z_t.X.shape
    device = z_t.X.device

    # Retrieve transition matrices
    Qtb = transition_model.get_Qt_bar(t_array, device)
    Qsb = transition_model.get_Qt_bar(t_array - 1, device)
    Qt = transition_model.get_Qt(t_array, device)
            
    # Normalize predictions
    pred_X = F.softmax(pred.X, dim=-1) # bs, n, d0
    pred_E = F.softmax(pred.E, dim=-1) # bs, n, n, d0
    p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(X_t=z_t.X, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X) # shape (bs, n, [x_{0}], [x_{t-1}])
    p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(X_t=z_t.E, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)

    # Dim of these two tensors: bs, N, d0, d_t-1
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1 # p(x_{t-1}|x_t) = q(x_{t-1},x_t| x_0)p(x_0|x_t)
    unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
    unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5 # in case pred is 0?
    prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

    pred_E_ = pred_E.reshape((bs, -1, pred_E.shape[-1]))
    weighted_E = pred_E_.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
    unnormalized_prob_E = weighted_E.sum(dim=-2)
    unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
    prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
    prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
    
    prob_s = z_t.get_new_object(X=prob_X, E=prob_E).mask(z_t.node_mask)
    
    if return_prob: return prob_s
    
    sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=z_t.node_mask)

    X_s = F.one_hot(sampled_s.X, num_classes=z_t.X.shape[-1]).float()
    E_s = F.one_hot(sampled_s.E, num_classes=z_t.E.shape[-1]).float()

    assert (E_s == torch.transpose(E_s, 1, 2)).all()
    assert (z_t.X.shape == X_s.shape) and (z_t.E.shape == E_s.shape)

    out_one_hot = z_t.get_new_object(X=X_s, E=E_s, y=torch.zeros(z_t.y.shape[0], 0).to(device))
    out_discrete = z_t.get_new_object(X=X_s, E=E_s, y=torch.zeros(z_t.y.shape[0], 0).to(device))
    
    return out_one_hot.mask(z_t.node_mask).type_as(z_t.y), out_discrete.mask(z_t.node_mask, collapse=True).type_as(z_t.y)

def zero_out_condition(z_t, mask_X, mask_E):
    # We don't maybe quite want literal zeroing out at this point
    z_t_ = z_t.get_new_object(X=z_t.X * mask_X, E=z_t.E * mask_E)
    return z_t_

# The following is deprecated:
# def drop_out_condition(z_t, mask_X, mask_E):
#     """Inputs:
#         z_t: PlaceHolder object
#         mask_X: (bs, n, dx) that chooses which elements of z_t.X to keep 
#         mask_E: (bs, n, n, de) that chooses which elements of z_t.E to keep
#         Outputs:
#         z_t_: PlaceHolder object with parts dropped out specified by mask_X and mask_E
#     """
#     # TODO: Could figure out a fast version of this but hopefully not necessary

#     # In case an edge that has endpoints in non-masked nodes is still masked out
#     E = z_t.E.clone()
#     E[~mask_E] = 0

#     bs = mask_X.shape[0]
#     Xs = []
#     Es = []
#     for idx in range(bs):
#         Xs.append(z_t.X[idx,mask_X[idx,:,0]])
#         Es.append(E[idx][mask_X[idx,:,0]][:,mask_X[idx,:,0]])
#     # Then just pad to same max size
#     max_size = max([X.shape[0] for X in Xs])
#     node_mask = torch.cat([F.pad(torch.ones_like(X[...,0], dtype=torch.bool)[None,:], (0,max_size-X.shape[0]), value=0) for X in Xs], dim=0)
#     X = torch.cat([F.pad(X[None], (0,0,0,max_size-X.shape[0]), value=0) for X in Xs], dim=0)
#     E = torch.cat([F.pad(E[None], (0,0, 0,max_size - E.shape[-3],
#                             0, max_size - E.shape[-2]),
#                             value=0) for E in Es], dim=0)
#     z_t_ = z_t.get_new_object(X=X, E=E)
#     return z_t_

def format_intermediate_samples_for_plotting(cfg, prob_s, pred, dense_data, inpaint_node_idx, inpaint_edge_idx):
    """A function that is mainly used in the sampling for doing some formatting of the intermediate samples when we want to save them.
    See diffusion_abstract.py for the usage"""
    # turn pred from logits to proba for plotting
    pred, prob_s = copy.deepcopy(pred), copy.deepcopy(prob_s)
    pred.X = F.softmax(pred.X, dim=-1)
    pred.E = F.softmax(pred.E, dim=-1)
    # TODO: make this better (more generic): ignore SuNo predictions
    pred.X[...,-1] = 0. # supernode, apparently
    pred.X /= pred.X.sum(-1).unsqueeze(-1)

    pred,_,_ = graph.fix_others_than_reactant_to_original(cfg, pred, dense_data)
    pred, mask_X, mask_E = graph.fix_nodes_and_edges_by_idx(pred, data=dense_data, node_idx=inpaint_node_idx, edge_idx=inpaint_edge_idx)
    
    # save p(z_s | z_t)
    prob_s,_,_ = graph.fix_others_than_reactant_to_original(cfg, prob_s, dense_data)
    prob_s, mask_X, mask_E = graph.fix_nodes_and_edges_by_idx(prob_s, data=dense_data, node_idx=inpaint_node_idx, edge_idx=inpaint_edge_idx)
    return pred, prob_s

def sample_batch_data(X, E, node_mask, get_chains):
    """Dummy method to repeat the sampled data, instead of generating anything
    Useful for, e.g., debugging evaluation metrics. """
    return_samples = graph.PlaceHolder(X=X, E=E, y=torch.ones(1), node_mask=node_mask).mask(node_mask, collapse=True)
    if get_chains:
        # TODO Check that the dimensions here are right
        dummy_to_save = graph.PlaceHolder(X=X, E=E, y=torch.ones(1), node_mask=node_mask)
        return return_samples, [(0, dummy_to_save)], [(0, dummy_to_save)], [(0, dummy_to_save)]
    else:
        return return_samples
           
def mol_diagnostic_chains(chains, atom_types, bond_types, chain_name='default'):
    '''
        Visualize chains of a process as an mp4 video.

        chains: list of PlaceHolder objects representing a batch of graphs at each time step.
        len(chains)==nb_time_steps.

        Returns:
            (str) list of paths of mp4 videos of chains.
    '''
    # repeat the last frame multiple times for a nicer video
    nb_of_chains = chains[0][1].X.shape[0] # number of graph chains to plot
    imgio_kargs = {'fps': 1, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
                   'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    # init a writer per chain
    writers = {}
    for i in range(nb_of_chains):
        writers[i] = imageio.get_writer(f'{chain_name}_chain{i}.mp4', **imgio_kargs)

    for t, samples_t in chains:
        for i in range(nb_of_chains):
            chain_pic_name = f'{chain_name}_sample_t{t}_chain{i}.png'
            one_sample = PlaceHolder(X=samples_t.X[i,...], E=samples_t.E[i,...], y=samples_t.y[i,...],
                                     node_mask=samples_t.node_mask[i,...])
            fig, mol = mol_diagnostic_plots(sample=one_sample, name=chain_pic_name, 
                                            atom_types=atom_types, bond_types=bond_types,
                                            show=False, return_mol=True)
            img = imageio.v2.imread(os.path.join(os.getcwd(), chain_pic_name))
            writers[i].append_data(img)
            # repeat the last frame 10 times for a nicer video
            if t==0:
                for _ in range(10):
                    writers[i].append_data(img)
            # Delete the png file in a hacky way to avoid clutter on the file system. 
            # TODO: Do this properly by not saving the pngs in the first place
            if os.path.exists(os.path.join(os.getcwd(), chain_pic_name)):
                os.remove(os.path.join(os.getcwd(), chain_pic_name))

    # close previous writers
    for i in range(nb_of_chains):
        writers[i].close()
        
    # plot the last molecule for each chain as a separate image
    sampled_smis = []
    for i in range(nb_of_chains):
    #     img = Draw.MolToImage(mol, size=(300, 300))
    #     plt.imshow(img)
        smi = Chem.MolToSmiles(mol)       
        sampled_smis.append(smi)
    #     plt.title(chain_name+'_mol')
    #     plt.axis('off')
    #     plt.savefig(chain_name+f'_mol{i}.png')
    #     sampled_smis.append(smi)

    return [(os.path.join(os.getcwd(), f'{chain_name}_chain{i}.mp4'), os.path.join(os.getcwd(), f'{chain_name}_mol{i}.png'), sampled_smis[i]) for i in range(nb_of_chains)]

def mol_diagnostic_plots(sample, atom_types, bond_types, name='mol.png', show=False, return_mol=False):  
    '''
        Plotting diagnostics of node and edge distributions.

        X: node matrix with distribution over node types. (n, dx)
        E: edge matrix with distribution over edge types. (n, n, de)
        name: the name of the file where to save the figure.
        show: Boolean to decide whether to show the matplot plot or not.

        return:
            plt fig object.
    ''' 
    # remove padding nodes
    X_no_padding = sample.X[sample.node_mask].cpu()
    E_no_padding = sample.E[sample.node_mask,...][:,sample.node_mask,...].cpu()
    # X_no_padding = X
    # E_no_padding = E
    # E_no_padding = E_no_padding.argmax(dim=-1) # because plotting the adjacency matrix in 2d

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 7), 
                             gridspec_kw={'width_ratios': [1.5, 1.5, 1]}) # x, y
    # unused axises
    axes[0,0].axis('off')
    axes[2,0].axis('off')
    axes[0,2].axis('off')
    axes[2,2].axis('off')

    ## plot distribution over atom types for each atom
    sns_kargs = {"vmin": 0, "vmax": 1.0, "cmap": sns.cm.rocket_r}

    ### if want to put back y labels:
    ### yticklabels=[f"{i}" for i in range(X_no_padding.shape[0])],
    sns.heatmap(X_no_padding, xticklabels=atom_types, 
                ax=axes[1,0], cbar_kws={'pad': 0.01}, **sns_kargs)
    axes[1,0].set_title('atom types')
    axes[1,0].tick_params(axis='x', rotation=90)
    axes[1,0].yaxis.set_tick_params(labelleft=False)
    axes[1,0].set_yticks([])
    
    ## plot adjacency matrix with bond types
    vmap = {i:b for i, b in enumerate(bond_types)}
    n = len(vmap)
    # colors from https://matplotlib.org/stable/gallery/color/named_colors.html
    myColors = (mcolors.to_rgb("bisque"),  mcolors.to_rgb("teal"), mcolors.to_rgb("forestgreen"),
                mcolors.to_rgb("midnightblue"))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    # single bond
    ax = sns.heatmap(E_no_padding[..., 1], 
                     xticklabels=[f"{i}" for i in range(X_no_padding.shape[0])], 
                     yticklabels=[f"{i}" for i in range(X_no_padding.shape[0])],
                     cbar_kws={'pad': 0.01}, ax=axes[0, 1], **sns_kargs)

    # colorbar = ax.collections[0].colorbar
    # r = colorbar.vmax - colorbar.vmin
    # colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(n)])
    # colorbar.set_ticklabels(list(vmap.values()))
    axes[0,1].set_title('single bonds')
    #axes[0, 1].tick_params(axis='x', rotation=90)
    axes[0,1].yaxis.set_tick_params(labelleft=False)
    axes[0,1].set_yticks([])
    axes[0,1].xaxis.set_tick_params(labelbottom=False)
    axes[0,1].set_xticks([])

    # double bond
    ax = sns.heatmap(E_no_padding[...,2], 
                     xticklabels=[f"{i}" for i in range(X_no_padding.shape[0])], 
                     yticklabels=[f"{i}" for i in range(X_no_padding.shape[0])],
                     cbar_kws={'pad': 0.01}, ax=axes[1, 1], **sns_kargs)
    axes[1,1].set_title('double bonds')
    axes[1,1].yaxis.set_tick_params(labelleft=False)
    axes[1,1].set_yticks([])
    axes[1,1].xaxis.set_tick_params(labelbottom=False)
    axes[1,1].set_xticks([])

    # triple bond
    ax = sns.heatmap(E_no_padding[...,3], 
                     xticklabels=[f"{i}" for i in range(X_no_padding.shape[0])], 
                     yticklabels=[f"{i}" for i in range(X_no_padding.shape[0])],
                     cbar_kws={'pad': 0.01}, ax=axes[2, 1], **sns_kargs)
    axes[2,1].set_title('triple bonds')
    axes[2,1].yaxis.set_tick_params(labelleft=False)
    axes[2,1].set_yticks([])
    axes[2,1].xaxis.set_tick_params(labelbottom=False)
    axes[2,1].set_xticks([])

    ## plot graphs as molecules
    sample.X = sample.X.unsqueeze(0)
    sample.E = sample.E.unsqueeze(0)
    sample.node_mask = sample.node_mask.unsqueeze(0)
    mol_g = sample_categoricals_simple(prob=sample)
    mol_g = mol_g.mask(mol_g.node_mask, collapse=True)
    mol_rdkit = mol.mol_from_graph(node_list=mol_g.X[0,...], adjacency_matrix=mol_g.E[0,...], 
                             atom_types=atom_types, bond_types=bond_types)
    img = Draw.MolToImage(mol_rdkit, size=(300, 300))
    axes[1,2].imshow(img)
    axes[1,2].set_title('sample molecule')
    axes[1,2].axis('off')
    axes[1,2].yaxis.set_tick_params(labelleft=False)
    axes[1,2].set_yticks([])
    axes[1,2].xaxis.set_tick_params(labelbottom=False)
    axes[1,2].set_xticks([])

    fig.suptitle(name.split('.png')[0])
    #plt.tight_layout(rect=[0, 0, 1, 0.97]) # hack to add space between suptitle and subplots. Yuck.
    plt.tight_layout()
    if show:
        plt.show()
   
    plt.savefig(name)
    plt.close()
    if return_mol:
        return fig, mol_rdkit
    else:
        return fig

def rxn_diagnostic_chains(chains, atom_types, bond_types, chain_name='default'):
    '''
        Visualize chains of a process as an mp4 video.

        chains: list of PlaceHolder objects representing a batch of graphs at each time step.
        len(chains)==nb_time_steps.

        Returns:
            (str) list of paths of mp4 videos of chains.
    '''
    nb_of_chains = chains[0][1].X.shape[0] # number of graph chains to plot
    imgio_kargs = {'fps': 1, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
                   'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    # init a writer per chain
    writers = {}  
    sampled_mols = {}
    for t, samples_t in chains:
        for c in range(nb_of_chains):
            suno_idx = atom_types.index('SuNo') # offset because index 0 is for no node   
    
            suno_indices = (samples_t.X[c,...].argmax(-1)==suno_idx).nonzero(as_tuple=True)[0].cpu() 
            mols_atoms = torch.tensor_split(samples_t.X[c,...], suno_indices, dim=0)[1:-1] # ignore first set (SuNo) and last set (product)
            mols_edges = torch.tensor_split(samples_t.E[c,...], suno_indices, dim=0)[1:-1]
            node_masks = torch.tensor_split(samples_t.node_mask[c,...], suno_indices, dim=-1)[1:-1]
            
            for m, mol_atoms in enumerate(mols_atoms): # for each mol in sample
                chain_pic_name = f'{chain_name}_sample_t{t}_chain{c}_mol{m}.png'

                if c not in writers.keys():
                    writer = imageio.get_writer(f'{chain_name}_chain{c}_mol{m}.mp4', **imgio_kargs)
                    writers[c] = {m: writer}
                else:
                    if m not in writers[c].keys():
                        writer = imageio.get_writer(f'{chain_name}_chain{c}_mol{m}.mp4', **imgio_kargs)
                        writers[c][m] = writer
                    else:
                        writer = writers[c][m]

                mol_edges_to_all = mols_edges[m] 
                mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=1)[1:] # ignore first because empty SuNo set
                mol_edges = mol_edges_t[m]
                mol_edges = mol_edges[1:,:][:,1:] # (n-1, n-1)
                mol_atoms = mol_atoms[1:] # (n-1)
                node_mask = node_masks[m][1:]
                
                one_sample = PlaceHolder(X=mol_atoms, E=mol_edges, node_mask=node_mask, y=torch.tensor([t], device=device).unsqueeze(-1))
                
                fig, mol = mol_diagnostic_plots(sample=one_sample, atom_types=atom_types, bond_types=bond_types, 
                                                name=chain_pic_name, show=False, return_mol=True)
                
                if c not in sampled_mols.keys():
                    sampled_mols[c] = {m: [mol]}
                else:
                    if m not in sampled_mols[c].keys():
                        sampled_mols[c][m] = [mol]
                    else:
                        sampled_mols[c][m].append(mol) 
                    
                img = imageio.v2.imread(os.path.join(os.getcwd(), chain_pic_name))
                writers[c][m].append_data(img)
            # repeat the last frame 10 times for a nicer video
            if t==0:
                for _ in range(10):
                    writers[c][m].append_data(img)

    # close previous writers
    for c in writers.keys():
        for m in writers[c].keys():
            writers[c][m].close()
    
    # for c in sampled_mols.keys():
    #     for m in sampled_mols[c].keys():
    #         img = Draw.MolToImage(sampled_mols[c][m][-1], size=(300, 300))
    #         plt.imshow(img)
    #         plt.title(f'chain{c}_mol{m}')
    #         plt.axis('off')
    #         plt.savefig(f'chain{c}_mol{m}.png')
        
    # plot the last rxn for each chain as a separate image
    # return [('', os.path.join(os.getcwd(), chain), smi) for chain, smi in zip(img_paths, sampled_smis)]
    return [(os.path.join(os.getcwd(), f'{chain_name}_chain{c}_mol{m}.mp4'), os.path.join(os.getcwd(), f'chain{c}_mol{m}.png'), Chem.MolToSmiles(sampled_mols[c][m][-1])) for c in writers.keys() for m in range(len(writers[c]))]

def rxn_vs_sample_plot(true_rxns, sampled_rxns, cfg, chain_name='default', rxn_offset_nb=0):
    '''
       Visualize the true rxn vs a rxn being sampled to compare the reactants more easily.
       
       rxn_offset_nb: where to start the count for naming the rxn plot (file).
    '''

    assert true_rxns.X.shape[0]==sampled_rxns[0][1].X.shape[0], 'You need to give as many true_rxns as there are chains.'+\
            f' Currently there are {true_rxns.X.shape[0]} true rxns and {sampled_rxns[0][1].X.shape[0]} chains.'
            
    # initialize the params of the video writer
    nb_of_chains = true_rxns.X.shape[0] # number of graph chains to plot
    imgio_kargs = {'fps': 1, 'quality': 10, 'macro_block_size': None, 'codec': 'h264', 'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    # create a frame for each time step t
    writers = []
    for t, samples_t in sampled_rxns:
        for c in range(nb_of_chains):
            chain_pic_name = f'{chain_name}_t{t}_rxn{c+rxn_offset_nb}.png'
            # get image of the true rxn, to be added to each plot at time t 
            true_rxn = true_rxns.subset_by_idx(start_idx=c, end_idx=c+1)
            # true_rxn = graph.PlaceHolder(X=true_rxns.X[c,...].unsqueeze(0), E=true_rxns.E[c,...].unsqueeze(0), node_mask=true_rxns.node_mask[c,...].unsqueeze(0), y=true_rxns.y,
            #                              mol_assignment=true_rxns.mol_assignment[c,...].unsqueeze(0))
            true_img = mol.rxn_plot(rxn=true_rxn, cfg=cfg)
            # true_img = true_img.resize((600, 300))

            # get image of the sample rxn at time t
            # one_sample_t = graph.PlaceHolder(X=samples_t.X[c,...].unsqueeze(0), E=samples_t.E[c,...].unsqueeze(0), y=samples_t.y, node_mask=samples_t.node_mask[c,...].unsqueeze(0),
            #                                  mol_assignment=samples_t.mol_assignment[c,...].unsqueeze(0))
            one_sample_t = samples_t.subset_by_idx(start_idx=c, end_idx=c+1)
            sampled_img = mol.rxn_plot(rxn=one_sample_t, cfg=cfg)
            # sampled_img = sampled_img.resize((600, 300))
            
            # plot sampled and true rxn in the same fig
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 7)) # x, y
            axes[0].axis('off')
            axes[1].axis('off')
            
            axes[0].set_title('sampled')
            axes[1].set_title('true')
            
            axes[0].imshow(sampled_img)
            axes[1].imshow(true_img)
            fig.suptitle(chain_pic_name.split('.png')[0])
            plt.savefig(chain_pic_name, dpi=199)
            plt.close()
            
            if c >= len(writers):
                writer = imageio.get_writer(f'{chain_name}_rxn{c+rxn_offset_nb}.mp4', **imgio_kargs)
                writers.append(writer)

            img = imageio.v2.imread(os.path.join(os.getcwd(), chain_pic_name), format='PNG')
            img = np.array(img) 
            writers[c].append_data(img)
            
            # repeat the last frame 10 times for a nicer video
            if t==0:
                for _ in range(10):
                    writers[c].append_data(img)
                
    # close previous writers
    for c in range(len(writers)):
        writers[c].close()
                
    return [os.path.join(os.getcwd(), f'{chain_name}_rxn{c+rxn_offset_nb}.mp4') for  c in range(nb_of_chains)]
    
def assert_correctly_masked(variable, node_mask):
    # print(f'(variable * (1 - node_mask.long())).abs().max().item() {(variable * (1 - node_mask.long())).abs().max().item()}\n')
    # exit()
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def cosine_beta_schedule_discrete_alternate(timesteps):
    """ Cosine schedule, modified so that it converges to zero transition probability when t->0
    This has an error and the overall shape of the schedule is not the same as the original cosine schedule."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    # Note that we don't have the s parameter here: It shouldn't be here in discrete diffusion!
    alphas_cumprod = np.cos(0.5 * np.pi * (x / steps)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod # This allows zero probability for the first step (is important!)
    betas = 1 - alphas
    return betas.squeeze()

def cosine_beta_schedule_discrete_alternate_2(timesteps):
    """ Cosine schedule, modified so that it converges to zero transition probability when t->0,
    and so that the shape otherwise matches the original cosine schedule."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    # Note that we don't have the s parameter here: It shouldn't be here in discrete diffusion!
    alphas_cumprod = np.cos(0.5 * np.pi * (x / steps)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.concatenate([np.array([1.0]), alphas])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)

def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)

def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)

def sample_categoricals_simple(prob):
    """
    Sample features from a multinomial distribution with probabilities (probX, probE, prob_atom_charges, prob_chiral, prob_bonddir)
    input:
    prob: Placeholder with the elements in one-hot format
    NOTE: Assumes that all the features in prob are in one-hot format, and the numbers are not zeros for any values (could happen, e.g., in node masking)
    """
    probX, probE, prob_atom_charges, prob_atom_chiral, prob_bond_dirs, y = prob.X.clone(), prob.E.clone(), prob.atom_charges.clone(), prob.atom_chiral.clone(), prob.bond_dirs.clone(), prob.y.clone()
    
    # Noise node features
    probX = probX.reshape(probX.size(0) * probX.size(1), -1) # (bs * n, dx_out)
    prob_atom_charges = prob_atom_charges.reshape(prob_atom_charges.size(0) * prob_atom_charges.size(1), -1) # (bs * n, n_atom_charges)
    prob_atom_chiral = prob_atom_chiral.reshape(prob_atom_chiral.size(0) * prob_atom_chiral.size(1), -1) # (bs * n, n_atom_chiral)
    
    for p in [probX, prob_atom_charges, prob_atom_chiral]:
        assert (abs(p.sum(dim=-1) - 1) < 1e-4).all()

    # Sample node features
    X_t = probX.multinomial(1).reshape(prob.node_mask.size(0), prob.node_mask.size(1)) * prob.node_mask # (bs, n)
    atom_charges_t = prob_atom_charges.multinomial(1).reshape(prob.node_mask.size(0), prob.node_mask.size(1)) * prob.node_mask # (bs, n)
    atom_chiral_t = prob_atom_chiral.multinomial(1).reshape(prob.node_mask.size(0), prob.node_mask.size(1)) * prob.node_mask # (bs, n)

    # Noise edge features
    probE = probE.reshape(probE.size(0) * probE.size(1) * probE.size(2), -1) # (bs * n * n, de_out)
    prob_bond_dirs = prob_bond_dirs.reshape(prob_bond_dirs.size(0) * prob_bond_dirs.size(1) * prob_bond_dirs.size(2), -1) # (bs * n * n, n_bond_dirs)
    
    # Sample E
    E_t = probE.multinomial(1).reshape(prob.node_mask.size(0), prob.node_mask.size(1), prob.node_mask.size(1))   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))
    E_t = E_t * prob.node_mask.unsqueeze(dim=1) * prob.node_mask.unsqueeze(dim=2)
    bond_dirs_t = prob_bond_dirs.multinomial(1).reshape(prob.node_mask.size(0), prob.node_mask.size(1), prob.node_mask.size(1))   # (bs, n, n)
    bond_dirs_t = torch.triu(bond_dirs_t, diagonal=1)
    bond_dirs_t = (bond_dirs_t + torch.transpose(bond_dirs_t, 1, 2))
    bond_dirs_t = bond_dirs_t * prob.node_mask.unsqueeze(dim=1) * prob.node_mask.unsqueeze(dim=2)

    X_t = F.one_hot(X_t, num_classes=probX.shape[-1]).float()
    E_t = F.one_hot(E_t, num_classes=probE.shape[-1]).float()
    atom_charges_t = F.one_hot(atom_charges_t, num_classes=prob_atom_charges.shape[-1]).float()
    atom_chiral_t = F.one_hot(atom_chiral_t, num_classes=prob_atom_chiral.shape[-1]).float()
    bond_dirs_t = F.one_hot(bond_dirs_t, num_classes=prob_bond_dirs.shape[-1]).float()
    
    z_t = prob.get_new_object(X=X_t, E=E_t, atom_charges=atom_charges_t, atom_chiral=atom_chiral_t, bond_dirs=bond_dirs_t, y=y)
    return z_t

def sample_discrete_features(prob):
    ''' 
        Sample features from multinomial distribution with given probabilities (probX, probE)
        input: 
            probX: node features. (bs, n, dx_out)
            probE: edge features. (bs, n, n, de_out)
            node_mask: mask used by PyG for batching. (bs, n)
            (optional) y_t: the y feature of noisy object (often time step).
            
    '''
    # Noise X
    # The masked rows should define probability distributions as well
    probX, probE, y = prob.X.clone(), prob.E.clone(), prob.y.clone()
    
    probX[~prob.node_mask] = 1 / probX.shape[-1] # masked is ignored
    probX = probX.reshape(probX.size(0) * probX.size(1), -1)       # (bs * n, dx_out)
        
    assert (abs(probX.sum(dim=-1) - 1) < 1e-4).all()

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(prob.node_mask.size(0), prob.node_mask.size(1))     # (bs, n)
    X_t = X_t * prob.node_mask

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(prob.node_mask.unsqueeze(1) * prob.node_mask.unsqueeze(2))
    diag_mask = torch.eye(probE.size(1), probE.size(2)).unsqueeze(0).expand(probE.size(0), -1, -1)
    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1] # why? => allows sampling self edges when what we want is valid dist
    probE = probE.reshape(probE.size(0) * probE.size(1) * probE.size(2), -1) # (bs * n * n, de_out)
    
    # Sample E
    E_t = probE.multinomial(1).reshape(prob.node_mask.size(0), prob.node_mask.size(1), prob.node_mask.size(1))   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))
    E_t = E_t * prob.node_mask.unsqueeze(dim=1) * prob.node_mask.unsqueeze(dim=2)

    X_t = F.one_hot(X_t, num_classes=probX.shape[-1]).float()
    E_t = F.one_hot(E_t, num_classes=probE.shape[-1]).float()
    
    z_t = prob.get_new_object(X=X_t, E=E_t, y=y)
    #z_t = PlaceHolder(X=X_t, E=E_t, y=y, node_mask=prob.node_mask, atom_map_numbers=prob.atom_map_numbers).type_as(X_t)
    return z_t

def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M, log=False):
    ''' M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    '''
    orig_shape = M.shape
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32) # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32) # same
    Qt_M_T = torch.transpose(Qt_M, -2, -1) # (bs, d, d)

    # How to do this in log-space for M?
    if log == False:
        eps = 1e-30 # The epsilon means that we avoid division by zero in the case of non-noised / conditioning nodes and absorbing state transitions
        numerator = (M_t @ Qt_M_T) * (M @ Qsb_M) + eps # (bs, N, d)
        # denom works because * and sum(dim=-1) generalizes dot product with xt.T to N xt
        #denom = ((M @ Qtb_M) * M_t).sum(dim=-1) # (bs, N, d) * (bs, N, d).T + sum = (bs, N, 1)
        # The former doesn't quite match with the following for some reason, maybe because we do the reweighting of the masked nodes in the diffusion schedule?
        # ... hmm can that then be an issue in practice?
        denom = numerator.sum(-1, keepdim=True)
        prob = numerator / denom #.unsqueeze(-1) # (bs, N, d)
    else:
        eps = 1e-30
        # log(x_t Q_t^T dot x_0 Qbar_{t-1}) = log(x_t Q_t^T) + log(exp(x_0_logit) Qbar_{t-1}) - log(sum(exp(x_0_logit)))
        # The clipping in the second term is to avoid inf values when we have masked out logits
        # ... maybe we can just log normalize before?
        # TODO some of the calculations here may be a bit redundant now, the logsumexp for instance
        M = torch.log_softmax(M, dim=-1)
        log_numerator = torch.log(M_t @ Qt_M_T + eps) + torch.log(torch.exp(M) @ Qsb_M + eps) #- torch.logsumexp(M, dim=-1, keepdim=True) # (bs, N, d)
        log_denom = torch.logsumexp(log_numerator,dim=-1,keepdim=True)
        prob = log_numerator - log_denom

    prob = prob.reshape(orig_shape)
    return prob

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ 
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0

        M: X or E
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt
    """
    # Flatten feature tensors
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32) # bs, N, dt with N=n or N=n*n
    Qt_T = Qt.transpose(-1, -2)                                  # bs, dt, d_t-1
    X_t_transposed = X_t.transpose(-1, -2)                       # bs, dt, N

    numerator = (X_t @ Qt_T).unsqueeze(dim=2) * Qsb.unsqueeze(1)             # bs, N, 1, d_t-1. Just use different rows of Qsb to represent the x_0 dimension. The last dimension should be x_{t-1} dimension
    denominator = (Qtb @ X_t_transposed).transpose(-1, -2).unsqueeze(-1)     # bs, d0, N, 1
    denominator[denominator==0] = 1e-6 

    out = numerator / denominator

    # Dimensions here: bs, N, d0, d_t-1
    return out

def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_X[~node_mask] = row_X
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    return true_X, true_E, pred_X, pred_E

def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)   # (bs, n, dx)
    prob_E = compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)   # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)

def sample_from_noise(limit_dist, node_mask, T):
    """ 
        Sample from the limit distribution of the diffusion process.
        
        input:
            limit_dist: stationary distribution of the diffusion process.
            node_mask: masking used by PyG for batching.
        output:
            z_T: sampled node and edge features.
    """
    bs, n_max = node_mask.shape
    # Node features
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    atom_charge_limit = limit_dist.atom_charges[None, None, :].expand(bs, n_max, -1)
    atom_chiral_limit = limit_dist.atom_chiral[None, None, :].expand(bs, n_max, -1)
    # Edge features
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    bond_dir_limit = limit_dist.bond_dirs[None, None, None, :].expand(bs, n_max, n_max, -1)

    # Sample from everything
    z_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).long()
    z_atom_charges = atom_charge_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).long()
    z_atom_chiral = atom_chiral_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).long()
    z_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).long()
    z_bond_dir = bond_dir_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).long()
    # One-hot encoding of the samples
    z_X = F.one_hot(z_X, num_classes=x_limit.shape[-1]).float().to(device)
    z_E = F.one_hot(z_E, num_classes=e_limit.shape[-1]).float().to(device)
    z_atom_charges = F.one_hot(z_atom_charges, num_classes=atom_charge_limit.shape[-1]).float().to(device)
    z_atom_chiral = F.one_hot(z_atom_chiral, num_classes=atom_chiral_limit.shape[-1]).float().to(device)
    z_bond_dir = F.one_hot(z_bond_dir, num_classes=bond_dir_limit.shape[-1]).float().to(device)
    z_y = T*torch.ones((bs,1)).to(device)

    # Get upper triangular part of edge features, without main diagonal
    
    def symmetrize(z):
        upper_triangular_mask = torch.zeros_like(z).to(device)
        indices = torch.triu_indices(row=z.size(1), col=z.size(2), offset=1)
        upper_triangular_mask[:,indices[0],indices[1],:] = 1
        # make sure adjacency matrix is symmetric over the diagonal
        z = z * upper_triangular_mask
        z = (z + torch.transpose(z, 1, 2))
        assert (z == torch.transpose(z, 1, 2)).all()
        return z

    z_E = symmetrize(z_E)
    z_bond_dir = symmetrize(z_bond_dir)

    return PlaceHolder(X=z_X.to(device), E=z_E.to(device), y=z_y.to(device), atom_charges=z_atom_charges, 
                       atom_chiral=z_atom_chiral, bond_dirs=z_bond_dir, node_mask=node_mask.to(device)).mask(node_mask.to(device))