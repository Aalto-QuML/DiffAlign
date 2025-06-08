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

from diffalign_old.utils import setup, mol
from hydra.core.hydra_config import HydraConfig
from diffalign_old.utils import setup
from datetime import date

from diffalign_old.datasets import supernode_dataset_temp

warnings.filterwarnings("ignore", category=PossibleUserWarning)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    assert cfg.general.task in setup.task_to_class_and_model.keys(), f'Task {cfg.general.task} not in setup.task_to_class_and_model.'
    log.info('Getting dataset infos...')
    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=supernode_dataset_temp,
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, 
                                                  slices={'train': None, 'val': None, 'test': None})
    
    exit()
    # subsets = {'train': datamodule.train_dataloader(), 
    #            'test': datamodule.test_dataloader(), 
    #            'val': datamodule.val_dataloader()}
    
    # for s, dataloader in subsets.items():
    #     dataset_folder = f'{cfg.dataset.name}-{cfg.dataset.dataset_nb}' if cfg.dataset.dataset_nb!='' else cfg.dataset.name
    #     out_file = open(os.path.join(parent_path, 'data', dataset_folder, 'raw', f'{s}-graph-parsed.txt'), 'w')
    #     for g_rxn in iter(dataloader):
            # rxn = mol.rxn_from_graph_supernode(g_rxn, atom_types=cfg.dataset.atom_types, 
            #                                    bond_types=setup.task_to_class_and_model[cfg.general.task]['data_class'].bond_types,
            #                                    plot_dummy_nodes=False)
    #         out_file.write(f'{rxn}\n')
    #     out_file.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
