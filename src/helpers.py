import numpy as np
import random
import torch

def set_seed(seed: int):
    '''
    Sets the seed for the random number generators.
    Args:
        seed: The seed to set for the random number generators.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # TODO: maybe add other generators here?