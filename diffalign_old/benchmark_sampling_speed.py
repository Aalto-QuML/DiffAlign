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
import torch_geometric
from diffalign.data import graph

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign_old.utils import setup
from hydra.core.hydra_config import HydraConfig
from diffalign_old.utils import setup
from datetime import date
import re
from rdkit import Chem

warnings.filterwarnings("ignore", category=PossibleUserWarning)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB__SERVICE_WAIT"] = "300"

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    print(f'cfg.dataset.atom_types {cfg.dataset.atom_types}\n')
    print(f'started\n')

    orig_cfg = copy.deepcopy(cfg)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    run = None

    assert cfg.general.task in setup.task_to_class_and_model.keys(), f'Task {cfg.general.task} not in setup.task_to_class_and_model.'
    log.info('Getting dataset infos...')
    cfg.train.batch_size = 10
    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, 
                                                  slices={'train': None, 'val': None, 'test': None})

    assert len(dataset_infos.valencies)==len(cfg.dataset.atom_types)

    log.info('Getting model...')
    savedir = os.path.join(parent_path, 'experiments', cfg.general.wandb.run_id) if cfg.general.wandb.resume else None
    # print(f'cfg.general.wandb.resume {cfg.general.wandb.resume}\n')
    # print(f'cfg.general.wandb.run_id {cfg.general.wandb.run_id}\n')
    model, optimizer, scheduler, scaler, last_epoch = setup.get_model_and_train_objects(cfg, run=run, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                        model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                      'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                      'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                      'use_data_parallel': torch.cuda.device_count() > 1 and cfg.neuralnet.use_all_gpus},
                                                                                        parent_path=parent_path, savedir=savedir, device=device,
                                                                                        device_count=torch.cuda.device_count())

    model.eval()
    # TODO: change this to something that measures actual time
    # scores = model.evaluate(epoch=epoch, datamodule=datamodule, device=device)
    batches = setup.get_batches_from_datamodule(cfg, parent_path, datamodule)
    random.shuffle(batches)

    n_samples_per_condition = 100

    t0 = time.time()
    times = []
    for i, samples in enumerate(datamodule.train_dataloader()):
        samples = samples
        data_list = samples.to_data_list()
        for j in range(len(data_list)):
            print(i,j)
            data_ = torch_geometric.data.Batch.from_data_list(data_list[(j):(j+1)])
            dense_data = graph.to_dense(data_).to_device(device)
            dense_data = graph.duplicate_data(dense_data, n_samples=n_samples_per_condition, get_discrete_data=False)
            # t0 = time.time()
            model.sample_one_batch(device=device, n_samples=None, data=dense_data, get_chains=False, get_true_rxns=False, inpaint_node_idx=None, inpaint_edge_idx=None)
            # times.append(t1-t0)

        if i > 2: # 100 examples
            break
    torch.cuda.synchronize()
    t1 = time.time()
    avg_time = np.array((t1 - t0) / 20)
    print(f"The average time to generate one sample: {avg_time}")
    # print(f"The mean time for sampling: {times.mean()}")
    # print(f"The median time for sampling: {np.median(times)}")
    # print(f"The standard deviation of sampling time: {times.std()}")
    # print(f"Times: ", "\n".join([str(t) for t in times]))

if __name__ == '__main__':
    main()