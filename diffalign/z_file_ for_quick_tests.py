'''
    Evaluating a model without specifying wandb configs.
'''
# These imports are tricky because they use c++, do not move them
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import datetime
import pathlib
import warnings
import random
import numpy as np
import torch
import wandb
import hydra
import pickle
import logging
import itertools
# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign.utils import graph, setup
from diffalign.datasets import supernode_dataset, supernode_dataset_16atomlabels, uspto_molecule_dataset, supernode_dataset_old
from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign.diffusion.diffusion_mol import DiscreteDenoisingDiffusionMol

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    if cfg.train.log_to_wandb: cfg = setup.setup_wandb(cfg, parent_path)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    
    dense = pickle.load(open(os.path.join(parent_path, 'src', 'z_t_dense.pickle'), 'rb'))
    final = pickle.load(open(os.path.join(parent_path, 'src', 'z_t_final.pickle'), 'rb'))
    
    print(f'X {(dense.X==final.X).all()}\n')
    print(f'E {(dense.E==final.E).all()}\n')

    #print(f'E_final {[0,29:,:29]}\n')
    
    # pred_module1 = model.forward(z_t_module)
    # pred_prod1 = model.forward(z_t_prod)
    
    # exit()
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)