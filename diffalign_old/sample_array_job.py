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
import copy
# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign_old.utils import graph, setup
from diffalign_old.datasets import supernode_dataset, supernode_dataset_16atomlabels, uspto_molecule_dataset, supernode_dataset_old
from diffalign_old.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign_old.diffusion.diffusion_mol import DiscreteDenoisingDiffusionMol

# try:
#     from mpi4py import MPI
# except ImportError: # mpi4py is not installed, for local experimentation
#     MPI = None
#     log.warning("mpi4py not found. MPI will not be used.")

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]

os.environ['WANDB__SERVICE_WAIT'] = '1000'

@hydra.main(version_base='1.1', config_path='../configs', config_name='default')
def main(cfg: DictConfig):

    # breakpoint()
    orig_cfg = copy.deepcopy(cfg)

    # MPI related parameters (in case --ntasks>1)
    # if MPI:
    #     comm = MPI.COMM_WORLD
    #     mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
    #     mpi_rank = comm.Get_rank() # this will be 0
    # else:
    #     mpi_size = 1
    #     mpi_rank = 0
    mpi_size = 1
    mpi_rank = 0
    
    # whether we load new configs
    # load_from_wandb = cfg.general.wandb.resume or cfg.general.wandb.load_run_config

    # Extract only the command-line overrides
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')

    if cfg.general.wandb.mode=='online': 
        # run, cfg = setup.setup_wandb(cfg, cli_overrides=cli_overrides, job_type='ranking') # This creates a new wandb run or resumes a run given its id
        run, cfg = setup.setup_wandb(cfg, job_type='ranking')

    entity = cfg.general.wandb.entity
    project = cfg.general.wandb.project

    if cfg.general.wandb.load_run_config: 
        run_config = setup.load_wandb_config(orig_cfg)
        cfg = setup.merge_configs(default_cfg=orig_cfg, new_cfg=run_config, cli_overrides=cli_overrides)
    
    cfg.general.wandb.entity = entity
    cfg.general.wandb.project = project

    # log.info("1!------------------------------------------------")
    # log.info(f": {cfg}")
    # log.info(f": {cfg.general}")
    # log.info(f": {cfg.general.wandb}")

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    log.info(f"Random seed: {cfg.train.seed}")
    log.info(f"Shuffling on: {cfg.dataset.shuffle}")
    
    log.info(f"cfg.general.wandb.initialization_run_id: {cfg.general.wandb.initialization_run_id}")

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device_count: {device_count}, device: {device}\n')
    
    epoch_num = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval
    
    total_index = cfg.test.condition_index*mpi_size + mpi_rank
    log.info(f'cfg.test.condition_first & slurm array index & total condition index {cfg.test.condition_first}, {cfg.test.condition_index}, {total_index}\n')
    # condition_first: the first condition to be sampled overall
    # condition_index: defines the range of conditions to be sampled in this particular run (across multiple parallel ones)
    # So overall, we sample ranges [condition_first, condition_first+n_conditions], [condition_first+n_conditions, condition_first+2*n_conditions], etc.
    condition_start_for_job = int(cfg.test.condition_first) + int(total_index)*int(cfg.test.n_conditions)
    if condition_start_for_job is not None: # take only a slice of the 'true' edge conditional set
        log.info(f"Condition start: {int(cfg.test.condition_first)}+{int(total_index)*int(cfg.test.n_conditions)} = {condition_start_for_job}")
        data_slices = {'train': None, 'val': None, 'test': None}
        data_slices[cfg.diffusion.edge_conditional_set] = [int(condition_start_for_job), int(condition_start_for_job)+int(cfg.test.n_conditions)]
    # print(f'data_slices {data_slices}\n')

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], 
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, slices=data_slices)
    
    # print(f'dataset_infos {dataset_infos.input_dims}\n')
    # print(f'dataset_infos {dataset_infos.output_dims}\n')
    # exit()
    
    log.info("Getting the model and train objects...")
    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': device_count>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=device_count)
    
    log.info("2!------------------------------------------------")
    log.info(f": {cfg}")
    log.info(f": {cfg.general}")
    log.info(f": {cfg.general.wandb}")

    # 4. load the weights to the model
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=device_count)
    
    # 5. sample n_conditions and n_samples_per_condition
    output_file_smiles = f'samples_epoch{epoch_num}_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.txt'
    output_file_pyg = f'samples_epoch{epoch_num}_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.gz'

    if cfg.diffusion.edge_conditional_set=='test':
        dataloader = datamodule.test_dataloader()
    elif cfg.diffusion.edge_conditional_set=='val':
        dataloader = datamodule.val_dataloader()
    elif cfg.diffusion.edge_conditional_set=='train':
        dataloader = datamodule.train_dataloader()
        
    t0 = time.time()
    log.info(f'About to sample n_conditions={cfg.test.n_conditions}\n')
    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = model.sample_n_conditions(dataloader=dataloader, epoch_num=epoch_num, 
                                                                                                           device_to_use=None,  inpaint_node_idx=None, inpaint_edge_idx=None)
    
    #print(f'all_gen_rxn_smiles {all_gen_rxn_smiles}\n')
    # Save the results to a file
    for i in range(len(all_gen_rxn_smiles)):
        true_rxn_smiles = all_true_rxn_smiles[i]
        gen_rxn_smiles = all_gen_rxn_smiles[i]
        true_rcts_smiles = [rxn.split('>>')[0].split('.') for rxn in true_rxn_smiles]
        true_prods_smiles = [rxn.split('>>')[1].split('.') for rxn in true_rxn_smiles]
        graph.save_gen_rxn_smiles_to_file(output_file_smiles, condition_idx=condition_start_for_job+i, 
                                        gen_rxns=gen_rxn_smiles, true_rcts=true_rcts_smiles[0], true_prods=true_prods_smiles[0])
    # Save the sparse format generated graphs to a file (includes atom-mapping information) all_true_rxn_pyg
    graph.save_gen_rxn_pyg_to_file(filename=output_file_pyg, gen_rxns_pyg=all_gen_rxn_pyg, true_rxns_pyg=all_true_rxn_pyg)

    log.info(f'===== Total sampling time: {time.time()-t0}\n')
    
if __name__ == '__main__':
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        try:
            main()
        except Exception as e:
            log.exception("main crashed. Error: %s", e)
    else:
        main()