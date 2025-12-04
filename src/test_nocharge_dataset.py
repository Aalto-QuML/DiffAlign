'''
    Training script: used to train new models or resume training runs from wandb.
'''
import time
import os
import sys
import datetime
import pathlib
import warnings
import random
import numpy as np
import torch
import wandb
import hydra
import logging
import copy

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src.utils import setup
from hydra.core.hydra_config import HydraConfig
from src.utils import setup
from datetime import date

warnings.filterwarnings("ignore", category=PossibleUserWarning)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    # creates a new wandb run or resumes a run given its id
    # when a run is resumed, cfg is set to the downloaded run's config from wandb
        
    # set artifact name based on cfg file (one artifact per experiment)
    # artifact contains model weights, optimizer states, and everything needed to resume training in a single object.
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    assert cfg.general.task in setup.task_to_class_and_model.keys(), f'Task {cfg.general.task} not in setup.task_to_class_and_model.'
    log.info('Getting dataset infos...')
    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, 
                                                  slices={'train': None, 'val': None, 'test': None})
    
    exit()
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
