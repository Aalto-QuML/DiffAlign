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
from utils import mol, graph

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign.utils import setup
from hydra.core.hydra_config import HydraConfig
from diffalign.utils import setup
from datetime import date

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
    datamodule, _ = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                      shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False)
    
    # make sure the dataset is loaded with batch size = 1
    # datamodule.batch_size = 1
    # # make sure datamodule is not shuffled
    # datamodule.shuffle = False
    # read raw smiles
    data_dir = cfg.dataset.datadir if cfg.dataset.dataset_nb=='' else cfg.dataset.datadir+'-'+str(cfg.dataset.dataset_nb)
    raw_smiles_path = os.path.join(parent_path, data_dir, 'processed', 'test_filtered_raw.csv')
    raw_smiles = open(raw_smiles_path, 'r').readlines()
    
    print(f'datamodule.train_dataloader() {len(datamodule.train_dataloader())}\n')
    print(f'datamodule.test_dataloader() {len(datamodule.test_dataloader())}\n')
    print(f'datamodule.val_dataloader() {len(datamodule.val_dataloader())}\n')
    print(f'raw_smiles {len(raw_smiles)}\n')

    # get the first batch of the dataset
    for i, (raw_smi, batch) in enumerate(zip(raw_smiles, datamodule.test_dataloader())):
        if i%1000==0: print(f'First {(i+1)*1000} clear.\n')
        # turn the graph into rxn smiles
        dense_data = graph.to_dense(data=batch).to_device(device)
        dense_data = dense_data.mask(collapse=True)
        rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=dense_data, cfg=cfg, with_atom_mapping=True, like_raw=True)
        # TODO: check that canonicalizing the reactant/product side (all molecules jointly) fixes the order too. For now don't think it does
        raw_smi_rcts, raw_smi_prods = set(raw_smi.strip().split('>>')[0].split('.')), set(raw_smi.strip().split('>>')[1].split('.'))
        rxn_smi_rcts, rxn_smi_prods = set(rxn_smiles[0].strip().split('>>')[0].split('.')), set(rxn_smiles[0].strip().split('>>')[1].split('.'))

        assert raw_smi_rcts==rxn_smi_rcts and raw_smi_prods==rxn_smi_prods, f'The rxn smiles generated from the dense data {rxn_smiles[0]} is not the same as the raw smiles {raw_smi}.\n'
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
