import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)
            
def assert_correctly_masked(variable, node_mask):
    # print(f'(variable * (1 - node_mask.long())).abs().max().item() {(variable * (1 - node_mask.long())).abs().max().item()}\n')
    # exit()
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'