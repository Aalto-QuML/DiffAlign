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