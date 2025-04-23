import wandb
import hydra
from omegaconf import OmegaConf
import numpy as np
import random
import pathlib
import os
from collections import OrderedDict
import torch # Must be done after setting CUDA_VISIBLE_DEVICES
from diffalign.utils import setup
from diffalign.datasets import supernode_dataset, supernode_dataset_16atomlabels, uspto_molecule_dataset, supernode_dataset_old
from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign.diffusion.diffusion_mol import DiscreteDenoisingDiffusionMol
parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# log.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# log.addHandler(handler)
# Add your custom handler
handler = logging.StreamHandler()
log.addHandler(handler)
# Optionally, set a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Access the run using the run path
# def get_run(run_path):
#     api = wandb.Api()
#     run = api.run(run_path)
#     return run

# 2. Download the specified checkpoint artifact and Hydra config
def download_files(savedir, run, epoch_num):
    # Download the checkpoint
    artifact_name_prefix = f"eval_epoch{epoch_num}"

    all_artifacts = run.logged_artifacts()
    artifact_name = None
    for a in all_artifacts:
        if a.name.startswith(artifact_name_prefix + ":"):
            artifact_name = a.name
            a.download(root=savedir)
    assert artifact_name is not None, f"Artifact with prefix {artifact_name_prefix} not found for the specified run."

# def download_shared_files(run, run_id):
#     # Downloads the run config (shared between processes) and creates the directory structure
#     savedir = os.path.join("experiments", "trained_models", run.name + run_id)
#     if not os.path.exists(savedir):
#         os.makedirs(savedir)
#     config_path = os.path.join(savedir, 'config.yaml')
#     with open(config_path, 'w') as f:
#         yaml.dump(run.config, f)
#     return savedir

# 3. Read the Hydra config and create the model
def create_model_from_config(cfg, device):
    # Note: The exact way you create a model from your config will depend on your config's structure.
    # This is just a generic placeholder.

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    if cfg.general.task=='mol': 
        data_class = uspto_molecule_dataset
        model_class = DiscreteDenoisingDiffusionMol
    elif cfg.general.task=='rxn': 
        data_class = supernode_dataset
        model_class = DiscreteDenoisingDiffusionRxn
    elif cfg.general.task=='rxn-uncharged':
        data_class = supernode_dataset_16atomlabels
        model_class = DiscreteDenoisingDiffusionRxn
    elif cfg.general.task=='rxn-old':
        data_class = supernode_dataset_old
        model_class = DiscreteDenoisingDiffusionRxn
    else:
        assert 'unknown task.'
    datamodule, dataset_infos = setup.get_dataset(cfg, data_class, shuffle=True, 
                                                  return_datamodule=True, recompute_info=False)
    
    model_kwargs = {'dataset_infos': dataset_infos, 
                    'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                    'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                    'use_data_parallel': False}
    model = model_class(cfg=cfg, **model_kwargs)
    model = model.to(device)
    
    return model, datamodule, data_class, dataset_infos

def check_if_wider_saved_state(state_dict):
    if 'model_state_dict' in list(state_dict.keys()):
        return True
    else:
        return False

def check_if_dataparallel_dict(state_dict):
    if 'module' in list(state_dict.keys())[0]:
        return True
    else:
        return False

def dataparallel_dict_to_regular_state_dict(state_dict):
    new_dict = OrderedDict()
    for key in state_dict.keys():
        new_dict[key.replace('module.', '')] = state_dict[key]
    return new_dict

def regular_state_dict_to_dataparallel_dict(state_dict):
    new_dict = OrderedDict()
    for key in state_dict.keys():
        new_dict['module.'+key] = state_dict[key]
    return new_dict

# 4. Load the weights into the model
def load_weights(model, checkpoint_path, device):
    # TODO: Multi-GPU support
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    if check_if_wider_saved_state(state_dict):
        state_dict = state_dict['model_state_dict']
    if check_if_dataparallel_dict(state_dict):
        state_dict = dataparallel_dict_to_regular_state_dict(state_dict)
    # elif not check_if_dataparallel_dict(state_dict) and torch.cuda.device_count() > 1:
    #     state_dict = regular_state_dict_to_dataparallel_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 5. Evaluate the model and log results back to the original run
def evaluate(cfg, data_class, datamodule, epoch, model, device, condition_range=None):
    # Initiate a new run that resumes the specified run
    
    if condition_range: # take only a slice of the 'true' edge conditional set
        # datamodule.datasets[cfg.diffusion.edge_conditional_set] = datamodule.datasets[cfg.diffusion.edge_conditional_set][condition_range[0]:condition_range[1]]
        data_slices = {'train': None, 'val': None, 'test': None}
        data_slices[cfg.diffusion.edge_conditional_set] = condition_range
        datamodule.prepare_data(datamodule.datasets, slices=data_slices)

    # Don't do this, just get the top-k scores separately
    model.eval()
    if cfg.diffusion.edge_conditional_set=='test':
        additional_dataloader = datamodule.test_dataloader()
    elif cfg.diffusion.edge_conditional_set=='val':
        additional_dataloader = datamodule.val_dataloader()
    elif cfg.diffusion.edge_conditional_set=='train':    
        additional_dataloader = datamodule.train_dataloader()

    scores = model.eval_n_conditions(dataloader=additional_dataloader, epoch=epoch, device_to_use=device)
    scores = dict(scores) # convert from defaultdict to dict
    scores["epoch"] = float(epoch)
    scores.pop("rxn_plots", None) # remove the plots

    return scores

    # with wandb.init(id=run_id, project=project, entity=entity, resume="allow") as run:
    #     # Move back to the original directory so that wandb.save works properly
    #     os.chdir(orig_dir)
    #     wandb.log({key: scores})
    #     wandb.save(os.path.join(dir_for_wandb, "modified_config.yaml"))
    #     if os.path.exists(os.path.join(dir_for_wandb, f'samples_epoch{epoch}.txt')): 
    #         wandb.save(os.path.join(dir_for_wandb, f'samples_epoch{epoch}.txt'))
        # metrics = {"epoch": epoch, "": 0.9}  # Placeholder
        # wandb.log(metrics)

# def main_subprocess(project, entity, run_id, savedir, epoch, cli_overrides, gpu_id):
    
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     log.info(f"Device: {device}")

#     # The default Hydra config
#     # This context manager ensures that we're working with a clean slate (Hydra's global state is reset upon exit)
#     with initialize(config_path="../configs"):
#         # Compose the configuration using the default config name
#         default_cfg = compose(config_name="default")
#         OmegaConf.set_struct(default_cfg, False) # Allow adding new fields (the experiment files sometimes have incorrectly added new stuff and not updated the default)
#     # The context is closed, and GlobalHydra is cleared, ensuring there are no lingering Hydra states
#     GlobalHydra.instance().clear()

#     # Log the number of GPUs
#     log.info(f"Number of GPUs: {torch.cuda.device_count()}")

#     orig_dir = os.getcwd()

#     log.info("Evaluating epoch " + str(epoch)  + " on GPU " + str(gpu_id))
#     try:
#         run_path = f"{entity}/{project}/{run_id}"
#         run = get_run(run_path)
#         download_files(savedir, run, epoch)
#         log.info(f"Shared directory: {savedir}")
        
#         config_path = os.path.join(os.getcwd(), savedir, 'config.yaml')
#         # Override some of the fields in the config
#         base_config = OmegaConf.load(config_path)
#         cfg = OmegaConf.merge(default_cfg, base_config)
#         cfg = OmegaConf.merge(cfg, cli_overrides)
#         # save the modified config to the artifact dir
#         key = f'eval_ncond_{cfg.diffusion.edge_conditional_set}_{cfg.test.n_conditions}/'
#         modified_cfg_dir = os.path.join(os.getcwd(), savedir, key)
#         if not os.path.exists(modified_cfg_dir):
#             os.makedirs(modified_cfg_dir)
#         dir_for_wandb = os.path.join(savedir, key)
#         log.info(f"Dir for wandb: {dir_for_wandb}")
#         modified_cfg_file = os.path.join(modified_cfg_dir, 'modified_config.yaml')
#         OmegaConf.save(cfg, modified_cfg_file)

#         model, datamodule, data_class, dataset_infos = create_model_from_config(cfg)
#         checkpoint_path = savedir + f'/eval_epoch{epoch}.pt'
#         log.info(f"Checkpoint path: {checkpoint_path}")
#         model = load_weights(model, checkpoint_path)

#         # Change the working directory so that samples get saved to the correct place, as they would be with Hydra
#         os.chdir(modified_cfg_dir)

#         evaluate_and_log(orig_dir, dir_for_wandb, key, cfg, data_class, datamodule, epoch, model, run_id, entity, project)
#     except Exception as e:
#         log.info(f"Exception occurred for epoch {epoch} on GPU {gpu_id}: {e}")
#         traceback.print_exc() 


# def main():
#     # Parse arguments
#     run_id = sys.argv[1] # c64gs0eh
#     epochs = sys.argv[2] # "19,39,59,..."
#     # Get the number of gpus & their ids
#     num_gpus = int(sys.argv[3])
#     gpu_ids = list(range(num_gpus))
#     epochs = [int(epoch) for epoch in epochs.split(",")]
#     assert len(epochs) <= num_gpus, "Number of epochs must be less than the number of GPUs."
#     cli_overrides = OmegaConf.from_cli(sys.argv[4:])

#     # Download the files shared between processes and set up directory structure
#     project = "retrodiffuser"
#     entity = "najwalb"
#     run_path = f"{entity}/{project}/{run_id}"
#     run = get_run(run_path)
#     savedir = download_shared_files(run, run_id)

#     # Create processes to run the training on each GPU
#     processes = []
#     for i in range(len(epochs)):
#         p = multiprocessing.Process(target=main_subprocess, args=(project, entity, run_id, savedir, epochs[i], cli_overrides, gpu_ids[i]))
#         p.start()
#         processes.append(p)

#     # Wait for all processes to complete
#     for p in processes:
#         p.join()

# # TO RUN:
# # python src/eval_from_wandb_2.py [RUN_ID] [CHECKPOINT_EPOCHS_COMMA_SEPARATED] [HYDRA_OVERRIDES]
# # NOTE: Only have a single script running for a single run at a time, otherwise wandb will drop some of the results
# # e.g., python src/eval_from_wandb_2.py c64gs0eh 19,39,59 test.n_conditions=128

# if __name__ == "__main__":
#     main()
