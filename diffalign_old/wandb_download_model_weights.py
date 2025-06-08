'''
    Sampling from a trained model.
'''
from omegaconf import DictConfig
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
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign_old.utils import graph, setup
from diffalign_old.datasets import supernode_dataset, supernode_dataset_16atomlabels, uspto_molecule_dataset, supernode_dataset_old
from diffalign_old.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign_old.diffusion.diffusion_mol import DiscreteDenoisingDiffusionMol

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]

os.environ['WANDB__SERVICE_WAIT'] = '1000'

@hydra.main(version_base='1.1', config_path='../configs', config_name='default')
def main(cfg: DictConfig):
    assert cfg.general.wandb.run_id is not None, f'Expected run_id. Got cfg.general.wandb.run_id={ cfg.general.wandb.run_id}\n'
    assert cfg.general.wandb.checkpoint_epochs is not None, f'Expected checkpoint epochs. Got cfg.general.wandb.checkpoint_epochs={cfg.general.wandb.checkpoint_epochs}\n'
    
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    for e in cfg.general.wandb.checkpoint_epochs:
        if not os.path.exists(os.path.join(savedir, f'epoch{e}.pt')):
            checkpoint_file, artifact = setup.download_checkpoint_from_wandb(cfg, savedir, int(e))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)