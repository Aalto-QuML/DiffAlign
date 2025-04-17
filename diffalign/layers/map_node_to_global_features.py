'''
    Implements the mapping of node features to global features.
'''

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Nodestoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1) if X.shape[1] > 1 else torch.zeros(X.shape[0], X.shape[2]).to(device)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out