'''
This file contains helper functions for the multiguide package.
NOTE: best to keep this package as independent of third-party packages as possible. 
The goal is to make it callable by scripts from any conda environment.
'''

import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]
