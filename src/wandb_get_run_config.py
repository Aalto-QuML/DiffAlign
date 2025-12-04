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
import yaml

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
    run_config = setup.load_wandb_config(cfg)
    config_file_name = f'train_smilesposenc_fromrun_{cfg.general.wandb.run_id}_nocharge.yaml'
    config_file_path = os.path.join(parent_path, "configs", "experiment", config_file_name)
    log.info(f'Logging config to path: {config_file_path}\n')
    yaml.dump(OmegaConf.to_container(run_config, resolve=True, throw_on_missing=True), open(config_file_path, 'w'))
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
