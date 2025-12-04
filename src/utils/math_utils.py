import torch

def create_permutation_matrix_torch(original, permuted):
    """
    Create a permutation matrix in PyTorch that represents the permutation required
    to transform the 'original' list into the 'permuted' list.

    The function assumes that both input lists contain distinct elements and have
    the same length. It uses PyTorch operations to efficiently create the matrix
    without explicit Python loops.

    Parameters:
    original (list of int): The original list of distinct integers.
    permuted (list of int): The permuted list of the same distinct integers found in 'original'.

    Returns:
    torch.Tensor: A 2D tensor (matrix) of shape (N, N) where N is the length of the input lists.
                  The matrix is a binary (0s and 1s) permutation matrix, where each row and
                  column has exactly one entry of 1, indicating the mapping from 'original' to
                  'permuted' list.

    Example:
    >>> original_list = [3, 1, 4, 2]
    >>> permuted_list = [1, 4, 3, 2]
    >>> matrix = create_permutation_matrix_torch(original_list, permuted_list)
    >>> print(matrix)
    tensor([[0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]], dtype=torch.int32)
    """
    size = len(original)
    # Convert lists to tensors, in case not already
    original_tensor = torch.tensor(original)
    permuted_tensor = torch.tensor(permuted)
    
    # Create index tensors
    original_indices = torch.argsort(original_tensor)
    permuted_indices = torch.argsort(permuted_tensor)
    
    # Create permutation matrix
    permutation_matrix = torch.zeros((size, size), dtype=torch.int32)
    permutation_matrix[original_indices, permuted_indices] = 1
    
    return permutation_matrix