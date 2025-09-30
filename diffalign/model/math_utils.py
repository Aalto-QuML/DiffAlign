import torch

def estimate_mrr(top_k_scores):
    """
    DON'T USE THIS, NOT RIGHT
    Estimates the Mean Reciprocal Rank (MRR) based on top-k accuracy scores.

    Args:
        top_k_scores (dict): A dictionary where keys are 'top-k' labels (e.g., 'top-1', 'top-3') and
                             values are the probabilities (as fractions) of the correct answer being within that top-k.

    Returns:
        float: An estimated MRR based on the provided top-k scores.
    """
    # Define average ranks for each top-k category, assuming uniform distribution within each range
    average_ranks = {
        'top-1': 1,
        'top-3': 2.5,  # Assumes ranks 2 and 3 are equally likely if not in top-1
        'top-5': 4.5,  # Assumes ranks 4 and 5 are equally likely if not in top-3
        'top-10': 7.5  # Assumes ranks 6 to 10 are equally likely if not in top-5
    }

    # Calculate weighted sum of reciprocal ranks
    weighted_sum_reciprocal_ranks = 0
    previous_top_k_prob = 0
    for top_k, prob in top_k_scores.items():
        # Adjust probability to reflect conditional probability, excluding higher ranks
        adjusted_prob = prob - previous_top_k_prob
        previous_top_k_prob = prob
        # Calculate and add the weighted reciprocal rank for this top-k category
        weighted_sum_reciprocal_ranks += adjusted_prob / average_ranks[top_k]

    # The estimated MRR is the weighted sum of reciprocal ranks
    estimated_mrr = weighted_sum_reciprocal_ranks
    
    return estimated_mrr

def estimate_mrr_discrete(top_k_scores):
    """
    Estimates the Mean Reciprocal Rank (MRR) from top-k scores assuming a uniform distribution of ranks within intervals.

    Args:
        top_k_scores (dict): A dictionary with keys as the top-k cutoffs (e.g., 1, 3, 5) and values as the corresponding
                             probabilities of the correct answer being within the top k.

    Returns:
        float: The estimated MRR based on the provided top-k scores.
    """
    sorted_keys = sorted(top_k_scores.keys())  # Ensure keys are sorted in ascending order
    estimated_mrr = 0
    prev_prob = 0  # To keep track of the previous cumulative probability

    for i, k in enumerate(sorted_keys):
        # Calculate the incremental probability for the current interval
        curr_prob = top_k_scores[k]
        delta_prob = curr_prob - prev_prob
        prev_prob = curr_prob

        # Calculate the expected reciprocal rank contribution for the current interval
        if i == 0:  # For the first interval, it starts from 1
            start_rank = 1
        else:
            start_rank = sorted_keys[i-1] + 1  # Start from the next rank after the previous cutoff
        
        sum_reciprocal_ranks = sum(1/r for r in range(start_rank, k+1))
        interval_length = k - start_rank + 1
        E_i = sum_reciprocal_ranks / interval_length

        # Update the estimated MRR with the weighted contribution of this interval
        estimated_mrr += delta_prob * E_i

    return estimated_mrr

def turn_topk_list_to_dict_1_3_10(topk_list):
    return {1: topk_list[0], 5: topk_list[1], 10: topk_list[2]}

def turn_topk_list_to_dict(topk_list):
    return {1: topk_list[0], 3: topk_list[1], 5: topk_list[2], 10: topk_list[3]}

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
    original_tensor = original.clone().detach()
    permuted_tensor = permuted.clone().detach()
    
    # Create index tensors
    original_indices = torch.argsort(original_tensor)
    permuted_indices = torch.argsort(permuted_tensor)
    
    # Create permutation matrix
    permutation_matrix = torch.zeros((size, size), dtype=torch.int32)
    permutation_matrix[original_indices, permuted_indices] = 1
    
    return permutation_matrix