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
from torch_geometric.data import Data, Batch

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign.utils import graph, setup, mol
from diffalign.datasets import supernode_dataset, supernode_dataset_16atomlabels, uspto_molecule_dataset, supernode_dataset_old
from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign.diffusion.diffusion_mol import DiscreteDenoisingDiffusionMol

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]

os.environ['WANDB__SERVICE_WAIT'] = '1000'

@hydra.main(version_base='1.1', config_path='../configs', config_name='default')
def main(cfg: DictConfig):
    # Extract only the command-line overrides
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')

    if cfg.general.wandb.mode=='online': 
        # run, cfg = setup.setup_wandb(cfg, cli_overrides=cli_overrides, job_type='ranking') # This creates a new wandb run or resumes a run given its id
        run, cfg = setup.setup_wandb(cfg, job_type='ranking')

    entity = cfg.general.wandb.entity
    project = cfg.general.wandb.project

    if cfg.general.wandb.load_run_config: 
        run_config = setup.load_wandb_config(cfg)
        cfg = setup.merge_configs(default_cfg=cfg, new_cfg=run_config, cli_overrides=cli_overrides)
    
    cfg.general.wandb.entity = entity
    cfg.general.wandb.project = project

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    log.info(f"Random seed: {cfg.train.seed}")
    log.info(f"Shuffling on: {cfg.dataset.shuffle}")
    
    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device_count: {device_count}, device: {device}\n')
    
    epoch_num = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], 
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False)
    
    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': device_count>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=device_count)

    # 4. load the weights to the model
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=device_count)
    
    
    product_pygs = [graph.get_graph_data_from_product_smi(p, cfg) for p in cfg.test.product_smis_list]
    product_batch = Batch.from_data_list(product_pygs)
    product_dense_data = graph.to_dense(product_batch).to_device(device)
    
    t0 = time.time()
    final_samples, actual_sample_chains, prob_s_chains, pred_0_chains, true_rxns = model.sample_for_condition(dense_data=product_dense_data, n_samples=cfg.test.n_samples_per_condition, 
                                                                                                              inpaint_node_idx=None, inpaint_edge_idx=None, device=None, return_chains=True)
    
    final_samples_smis = mol.get_cano_smiles_from_dense(X=final_samples.X, E=final_samples.E, mol_assignment=final_samples.mol_assignment, atom_types=dataset_infos.atom_decoder, 
                                               bond_types=dataset_infos.bond_decoder, return_dict=False)

    exit()
    
if __name__ == '__main__':
    # main()
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)