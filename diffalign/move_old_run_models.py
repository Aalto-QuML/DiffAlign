'''
    Sampling from a trained model.
'''
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import warnings
import random
import numpy as np
import torch
import hydra
import logging
import time
import wandb
import pickle 

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
    run = wandb.Api(overrides={"project": "retrodiffuser"}).run(f"retrodiffuser/{cfg.general.wandb.run_id}")
    all_artifacts = run.logged_artifacts()
    print(f'len(all_artifacts) {len(all_artifacts)}')
    
    
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)