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

from diffalign_old.utils import graph, setup
from diffalign_old.datasets import supernode_dataset, supernode_dataset_16atomlabels, uspto_molecule_dataset, supernode_dataset_old
from diffalign_old.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign_old.diffusion.diffusion_mol import DiscreteDenoisingDiffusionMol

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    if cfg.general.wandb.mode=='online' or cfg.general.wandb.run_id!='': 
        run = setup.setup_wandb(cfg, job_type='sampling') # This creates a new wandb run or resumes a run given its id
        
    cfg = OmegaConf.create(dict(run.config))
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    
    # 1. get the data to evaluate on
    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False)
    log.info('Getting model...')
    # 2. create the model and load the weights to be used for sampling
    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, run=run, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False)
    t0 = time.time()
    log.info(f'About to sample')
    
    # 3. iterate over the epochs specified for sampling
    epochs = [int(e) for e in cfg.general.wandb.checkpoint_epochs.split(',')]
    for i in epochs:
        # 4. load the weights to the model
        savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_name + cfg.general.wandb.run_id)
        model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb(cfg, run, i, savedir, model, optimizer, scheduler, scaler)
        # 5. sample n_conditions and n_samples_per_condition
        # artifact name: 
        ## training run_id (prefer this because easier to compare different versions per model for fine tuning)? 
        ## runid_epoch# (prefer this because one checkpoint = one model)?
        # artifact alias: epoch#_cond#_sampercond#? cond#_sampercond#?
        all_final_samples, all_dense_data = model.sample_n_conditions(dataloader=datamodule.test_dataloader(), inpaint_node_idx=None, inpaint_edge_idx=None, epoch_num=i)
        pickle.dump(all_final_samples, open(f'all_final_samples_epoch{i}.pickle', 'wb'))
        pickle.dump(all_dense_data, open(f'all_dense_data_epoch{i}.pickle', 'wb'))
        # wandb.save(f'samples_epoch{i}.txt')
        artifact = wandb.Artifact(f'{cfg.general.wandb.run_id}_samples', type='samples')
        artifact.add_file(f'samples_epoch{i}.txt', name=f'samples_epoch{i}.txt')
        artifact.add_file(f'all_final_samples_epoch{i}.pickle', name=f'all_final_samples_epoch{i}.pickle')
        artifact.add_file(f'all_dense_data_epoch{i}.pickle', name=f'all_dense_data_epoch{i}.pickle')
        run.log_artifact(artifact, aliases=[f'epoch{i}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}'])
        log.info(f"Sampling time: {time.time()-t0}")
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)