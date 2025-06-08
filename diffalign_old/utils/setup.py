import os
from copy import deepcopy
from typing import Optional, Union, Dict
from omegaconf import OmegaConf, open_dict
from overrides import overrides
import omegaconf
import wandb
import pathlib
import pickle
import torch
import yaml 
import sys
from torch_geometric.loader import DataLoader
import pickle
import io
from fcd_torch import FCD
from diffalign_old.utils import graph
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import logging
import datetime
from collections import OrderedDict
import yaml
import re
import copy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import socket
        
from diffalign_old.datasets import supernode_dataset, supernode_dataset_old,\
                         supernode_dataset_16atomlabels, uspto_molecule_dataset,\
                         supernode_dataset_forward, uspto_rxn_dataset, retrobridge_dataset
from diffalign_old.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign_old.diffusion.diffusion_mol import DiscreteDenoisingDiffusionMol

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ATOMS_RXN = 300

task_to_class_and_model = {                     
    'mol': {'data_class': uspto_molecule_dataset,
            'model_class': DiscreteDenoisingDiffusionMol},
    'rxn': {'data_class': supernode_dataset,
            'model_class': DiscreteDenoisingDiffusionRxn},
    'rxn-nosn': {'data_class': uspto_rxn_dataset,
                 'model_class': DiscreteDenoisingDiffusionRxn},
    'rxn-uncharged': {'data_class': supernode_dataset_16atomlabels,
                      'model_class': DiscreteDenoisingDiffusionRxn},
    'rxn-old': {'data_class': supernode_dataset_old,
                'model_class': DiscreteDenoisingDiffusionRxn},
    'rxn-forward': {'data_class': supernode_dataset_forward,
                    'model_class': DiscreteDenoisingDiffusionRxn},
    'rxn-retrobridge': {'data_class': retrobridge_dataset,
                        'model_class': DiscreteDenoisingDiffusionRxn},
}


def setup_multiprocessing(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=0.5))
    #setup_logging(rank, world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_logging(rank, world_size):
    """
    Set up logging for each process
    """
    # Define log format
    log_format = f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create a file handler for each process
    log_file = f'process_{rank}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # or logging.DEBUG for more verbose logging
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
            logging.StreamHandler()  # Also log to console
        ]
    )

def parse_value(value):
    
    # Check for boolean format
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    
    # Check for list format
    if value.startswith('[') and value.endswith(']'):
        return [convert_to_number(v.strip()) for v in value[1:-1].split(',')]
    else:
        # Attempt to convert to int, or leave as string if it fails
        return convert_to_number(value)

def convert_to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s  # Return as string if it's not a number
        
def capture_cli_overrides():
    """
    Capture the command-line arguments that represent configuration overrides.
    This function assumes that command-line overrides are in the format 'key=value'.
    """
    cli_args = sys.argv[1:]  # Exclude the script name
    overrides = {}
    for arg in cli_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Check if the value is a list
            value = parse_value(value)
            # Convert the key to Hydra's nested configuration format
            nested_keys = key.split('.')
            nested_dict = overrides
            for nested_key in nested_keys[:-1]:
                nested_dict = nested_dict.setdefault(nested_key, {})
            nested_dict[nested_keys[-1]] = value
                
    return OmegaConf.create(overrides)

def get_batches_from_datamodule(cfg, parent_path, datamodule):
    if cfg.train.batch_by_size_old:
        data_list_path = os.path.join(parent_path, datamodule.datadir, 'processed', 'train.pickle')
        train_data = pickle.load(open(data_list_path, 'rb'))
        # train_data = torch.load(data_list_path)
        batches, sizes_found = graph.batch_graph_by_size(input_data_list=train_data, 
                                                         size_bins=cfg.dataset.size_bins['train'], 
                                                         batchsize_bins=cfg.dataset.batchsize_bins['train'],
                                                         get_batches=True)
    else:
        batches = [b for b in datamodule.train_dataloader()]
        
    assert len(batches)>0, 'No batches.'
    
    return batches
        
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def update_config_with_run_id(parent_path, run_id):
    # TODO: if function still relevant, take out sys.argv from here
    experiment_arg = [arg for arg in sys.argv if arg.split('=')[0]=='+experiment' or arg.split('=')[0]=='+trained_experiment']
    assert len(experiment_arg)<=1, f'Expected at most 1 +experiment= argument. Got {len(experiment_arg)}.'
    
    config_file = f'{experiment_arg[0].split("=")[0].split("+")[1]}/{experiment_arg[0].split("=")[1]}' if len(experiment_arg)==1 else 'default.yaml'
    config_path = os.path.join(parent_path, 'configs', config_file)
    assert os.path.exists(config_path), f'config file {config_path} does not exist.'
    
    config_data = yaml.safe_load(open(config_path, "r"))  
    config_data['general']['wandb_id'] = run_id
    
    open(config_path, "w").write('# @package _global_\n')  
    yaml.dump(config_data, open(config_path, "a"))
    
    return config_data

def save_file_as_artifact_to_wandb(run, artifactname='default', alias='epoch00', type_='model', filepath=None, filename='model.pt'):
    '''
        Uploads model weights as artifact to wandb and returns the run id.
    '''
    filepath = filename if filepath is None else filepath
    artifact = wandb.Artifact(artifactname, type=type_)
    artifact.add_file(filepath, name=filename)
    run.log_artifact(artifact, aliases=[alias])
        
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass

def merge_configs(default_cfg, new_cfg, cli_overrides):
    default_cfg_ = copy.deepcopy(default_cfg)
    # allow adding new fields: 
    # no, everything the code is using currently should be in defaults
    # only override existing fields from run.config
    default_cfg_.neuralnet.p_to_r_skip_connection
    OmegaConf.set_struct(default_cfg_, False)
    # 1. merge run.config with default cfg to get any new missing fields
    merged_cfg = OmegaConf.merge(default_cfg_, new_cfg)
    merged_cfg.neuralnet.p_to_r_skip_connection
    # 2. override result with cli_overrides because they might be overriden by run.config
    merged_cfg = OmegaConf.merge(merged_cfg, cli_overrides)
    merged_cfg.neuralnet.p_to_r_skip_connection
    # e.g. scenario: 
    # new_cfg.neuralnet.n_layers=5, default_cfg.neuralnet.n_layers=None, 
    # upload_artifact does not exist in new_cfg, default_cfg.wandb.upload_artifact=True, 
    # new_cfg.test.n_conditions=8, default_cfg.test.n_conditions=3, (cli) default_cfg.test.n_conditions=10
    # we want the following: cfg.neuralnet.n_layers=5, cfg.wandb.upload_artifact=True, cfg.test.n_conditions=10
    
    return merged_cfg

def load_wandb_config(cfg):
    assert cfg.general.wandb.run_id is not None, f'Need to give run_id here. Got cfg.general.wandb.run_id={cfg.general.wandb.run_id}.'
    
    kwargs = {'entity': cfg.general.wandb.entity, 'project': cfg.general.wandb.project}
    # config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # kwargs['config'] = config_dict

    api = wandb.Api(overrides=kwargs)
    run = api.run(f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{cfg.general.wandb.run_id}") 
    run_config = OmegaConf.create(dict(run.config))
    if 'general' in run_config.keys() and 'wandb' in run_config.general.keys():
        run_config.general.wandb.initialization_run_id = None # Make sure that the model doesn't re-initialize when resuming or evaluating
    # run.finish() <- Can't have this, the run could actually be running if this is used during evaluation
    return run_config

def setup_wandb(cfg, job_type):
    assert (cfg.general.wandb.resume==False) or (cfg.general.wandb.resume and cfg.general.wandb.run_id!=''), "If wandb_resume is True, wandb.run_id must be set"
    # tags and groups
    kwargs = {'entity': cfg.general.wandb.entity, 'project': cfg.general.wandb.project, 'job_type': job_type,
              'group': cfg.general.wandb.group, 'tags': cfg.general.wandb.tags, 'mode': cfg.general.wandb.mode}
    
    log.info(kwargs)
    if cfg.general.wandb.resume:
        # NOTE: Currently, there is an issue with resuming runs from wandb where the run.config is deleted when using resume='allow'
        # resume='must' just fails, so we need to use 'allow' and then manually set the config
        # First verify the run exists
        try:
            api = wandb.Api(overrides=kwargs)
            run_path = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{cfg.general.wandb.run_id}"
            log.info(f"Attempting to access run: {run_path}")
            original_run = api.run(run_path)
            log.info(f"Found run with name: {original_run.name}")
            log.info(f"Run state: {original_run.state}")
            log.info(f"Run config exists: {original_run.config is not None}")
        except Exception as e:
            log.error(f"Error accessing run: {e}")
            raise

        kwargs['id'] = cfg.general.wandb.run_id
        kwargs['resume'] = 'allow'
        kwargs['config'] = dict(original_run.config)  # Explicitly pass the original config
        
        run = wandb.init(**kwargs)
        
        if 'train' in run.config.keys():
            run.config.update({'train': {'epochs': cfg.train.epochs}}, allow_val_change=True)
        if 'general' in run.config.keys():
            run.config.update({'general': {'wandb': {'resume': True, 
                                                   'run_id': run.id, 
                                                   'entity': 'najwalb',
                                                   'project': 'retrodiffuser', 
                                                   'mode': 'online'}}}, 
                            allow_val_change=True)
        
        cfg = OmegaConf.create(dict(run.config))
        cfg.general.wandb.initialization_run_id = None
    else:
        # if we're not resuming, use the cfg dictionary to create a run
        config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        kwargs['config'] = config_dict
        kwargs['name'] = cfg.general.wandb.run_name 
        run = wandb.init(**kwargs)

    return run, cfg

def get_wandb_run(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    return run

def resume_wandb_run(cfg):
    assert cfg.general.wandb_id != "" and cfg.general.wandb_id != None, "wandb_id must be set if wandb_resume is True"
    return wandb.init(id=cfg.general.wandb_id, project=cfg.general.project, entity=cfg.general.wandb_team, resume="allow")
    # return wandb.init(project=cfg.general.project, entity=cfg.general.wandb_team)

# def download_checkpoint_from_wandb_old(savedir, run, epoch_num):
#     # Download the checkpoint
#     artifact_name_prefix = f"eval_epoch{epoch_num}"
#     all_artifacts = run.logged_artifacts()
#     artifact_name = None
#     for a in all_artifacts:
#         if a.name.startswith(f"eval_epoch{epoch_num}:"):
#             artifact_name = a.name
#             a.download(root=savedir)
#     assert artifact_name is not None, f"Artifact with prefix {artifact_name_prefix} not found for the specified run."

#     # Get the name of the downloaded file
#     downloaded_file = os.path.join(savedir, "artifacts", artifact_name, artifact_name.split(":")[0] + ".pt")

#     return downloaded_file

def download_checkpoint_from_wandb(cfg, savedir, epoch_num, run=None, savename=None):
    # Download the checkpoint
    if run is None:
        run = wandb.Api(overrides={"entity": cfg.general.wandb.entity, "project": cfg.general.wandb.project}).run(f"{cfg.general.wandb.project}/{cfg.general.wandb.run_id}")
    try: # a hack to make this function work with different types of run inputs. (run.logged_artifacts() seems only accesible by wand.Api)
        all_artifacts = run.logged_artifacts()
    except:
        run = wandb.Api(overrides={"entity": cfg.general.wandb.entity, "project": cfg.general.wandb.project}).run(f"{cfg.general.wandb.project}/{cfg.general.wandb.run_id}")
        all_artifacts = run.logged_artifacts()
    downloaded_dir = None
    for a in all_artifacts:
        if a.name.startswith(f"eval_epoch{epoch_num}:"): # For resuming in the old format
            downloaded_dir = a.download(root=os.path.join(savedir))
            os.rename(os.path.join(downloaded_dir, f"eval_epoch{epoch_num}.pt"), 
                      os.path.join(downloaded_dir, f"epoch{epoch_num}.pt"))
            break
        if a.type=='model':
            if len([s for s in a.aliases if s.split('epoch')[-1]==str(epoch_num)])==1:
                downloaded_dir = a.download(root=os.path.join(savedir))
                if os.path.exists(os.path.join(downloaded_dir, f"eval_epoch{epoch_num}.pt")):
                    os.rename(os.path.join(downloaded_dir, f"eval_epoch{epoch_num}.pt"), 
                        os.path.join(downloaded_dir, f"epoch{epoch_num}.pt"))
                break
            else:
                assert f'Found more than one model checkpoint file with alias epoch{epoch_num}\n'

    # Get the name of the downloaded file
    assert downloaded_dir is not None, f"No checkpoint found for epoch={epoch_num}."
    
    if savename:
        downloaded_file = os.path.join(downloaded_dir, f"{savename}.pt")
    else:
        downloaded_file = os.path.join(downloaded_dir, f"epoch{epoch_num}.pt")

    return downloaded_file, a

# def get_latest_epoch_from_wandb_old(run):
#     """
#     Gets the number of the latest checkpoint epoch from the wandb run
#     """
#     all_artifacts = run.logged_artifacts()
#     epoch_nums = []
#     for a in all_artifacts:
#         if a.name.startswith("eval_epoch"):
#             epoch_nums.append(int(a.name.split("eval_epoch")[1].split(":")[0]))
#     assert len(epoch_nums) > 0, "No checkpoints found for the specified run."
#     return max(epoch_nums)

def get_latest_epoch_from_wandb(cfg):
    """
    Gets the number of the latest checkpoint epoch from the wandb run
    """
    run = wandb.Api(overrides={"entity": cfg.general.wandb.entity, "project": cfg.general.wandb.project}).run(f"{cfg.general.wandb.project}/{cfg.general.wandb.run_id}")
    all_artifacts = run.logged_artifacts()
    epoch_nums = []
    for a in all_artifacts:
        if a.name.startswith("eval_epoch"): # Old format for artifacts
            epoch_nums.append(int(a.name.split("eval_epoch")[1].split(":")[0]))
        elif a.type=='model':
            # assume only a single epoch# alias exists for each model artifact version
            epoch_nb = [re.findall(r'\d+', alias)[0] for alias in a.aliases if 'epoch' in alias][0]
            epoch_nums.append(int(epoch_nb))
    assert len(epoch_nums) > 0, "No checkpoints found for the specified run."
    return max(epoch_nums)

def get_wandb_run_path(cfg):
    return f"{cfg.general.wandb_team}/{cfg.general.project}/{cfg.general.wandb_id}"

def load_config_from_wandb(cfg, run, overrides=[]):
    # Download the config file
    # config_path = "loaded_config.yaml"
    # with open(config_path, 'w') as f:
    #     yaml.dump(run.config, f)
    loaded_config = OmegaConf.create(dict(run.config))

    # The default Hydra config
    # This context manager ensures that we're working with a clean slate (Hydra's global state is reset upon exit)
    # log.info("Current working dir: " + os.getcwd())

    # # TODO: Think of a way to add CLI overrides here, in case they are needed
    # OmegaConf.set_struct(cfg, False)
    # cfg = OmegaConf.merge(cfg, loaded_config)

    # TODO: Replace this with proper Hydra compose() etc. stuff, this can screw up the data types
    # for override in overrides: # This is mainly so that general.wandb_id and general.wandb_resume are set correctly
    #     key, value = override.split('=')
    #     old_val = OmegaConf.select(cfg, key, throw_on_resolution_failure=False)
    #     assert old_val != None, 'key not in the original config file'
    #     OmegaConf.update(cfg, key, type(old_val)(value), merge=True)

    return cfg

def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_diffusion = saved_cfg.diffusion

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.diffusion, True)
    with open_dict(cfg.diffusion):
        for key, val in saved_diffusion.items():
            if key not in cfg.diffusion.keys():
                setattr(cfg.diffusion, key, val)
    return cfg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataset(cfg, dataset_class, shuffle=True, recompute_info=False, return_datamodule=False, slices={'train':None, 'val':None, 'test':None}):
    datamodule = dataset_class.DataModule(cfg)
    datamodule.prepare_data(shuffle=shuffle, slices=slices)

    dataset_infos = dataset_class.DatasetInfos(datamodule=datamodule, cfg=cfg, recompute_info=recompute_info)
    print("Computing input/output dims")
    dataset_infos.compute_input_output_dims(datamodule=datamodule)
    print("Done computing input/output dims")

    return (datamodule, dataset_infos) if return_datamodule else dataset_infos

def check_if_dataparallel_dict(state_dict):
    if 'module' in list(state_dict.keys())[-1]: # don't use [0], but instead the last key, because for EMA, the first key is 'initted'
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
        # new_dict['module.'+key] = state_dict[key]
        new_dict['model.module.'+'.'.join(key.split('.')[1:])] = state_dict[key]
    return new_dict

def switch_between_dataparallel_and_parallel_dict(state_dict, device_count):
    print(f'device_count: {device_count}')
    print(f'check_if_dataparallel_dict(state_dict): {check_if_dataparallel_dict(state_dict)}')
    if check_if_dataparallel_dict(state_dict) and device_count <= 1:
        state_dict = dataparallel_dict_to_regular_state_dict(state_dict)
    elif not check_if_dataparallel_dict(state_dict) and device_count > 1:
        state_dict = regular_state_dict_to_dataparallel_dict(state_dict)
    return state_dict

def load_weights(model, model_state_dict, device_count=None):
    assert device_count is not None, f'Expected device_count to not be None. Found device_count={device_count}'

    print(check_if_dataparallel_dict(model_state_dict))
    model_state_dict = switch_between_dataparallel_and_parallel_dict(model_state_dict, device_count)
        
    model.load_state_dict(model_state_dict)
    
    return model

# def get_model_v2(cfg, model_class, model_kwargs, checkpoint_file=None):
#     model = model_class(cfg=cfg, **model_kwargs)
#     if checkpoint_file is not None:
#         model = load_weights(model, checkpoint_file)
#     return model

# # TODO: Get rid of this and replace it with the other functions
# def load_wandb_last_checkpoint(run_id, resume_from_last_n=1):
#     api = wandb.Api(overrides={"project": "retrodiffuser"})
#     run = api.run(f"retrodiffuser/{run_id}")
    
#     cnt = resume_from_last_n
#     for i, artifact in enumerate(reversed(run.logged_artifacts())):
#         if artifact.type=='model':
#             artifact_dir = artifact.download(os.getcwd())
#             checkpoint = torch.load(os.path.join(os.getcwd(), f'{artifact_dir}/{artifact.name.split(":")[0]+".pt"}'))
#             if cnt>0:
#                 cnt -= 1
#             if cnt==0:
#                 break
        
#     return checkpoint

def initialize_model_with_loaded_weights_reset_input_output(model, loaded_state_dict, device_count, exclude_some_layers):
    # TODO: THIS PROBABLY DOESN*T WORK FOR DATAPARALLEL YET

    # Get the current state dict of the new model
    new_state_dict = model.state_dict()
    
    loaded_state_dict = switch_between_dataparallel_and_parallel_dict(loaded_state_dict, device_count)
    
    # Define the layers to exclude
    
    if exclude_some_layers:
        exclude_layers = [
            'model.mlp_in_X.0', 'model.mlp_in_E.0', 'model.mlp_in_y.0',
            'model.mlp_out_X.', 'model.mlp_out_E.', 'model.mlp_out_y.'
        ]
    else:
        exclude_layers = []

    # Create an updated state dict
    updated_state_dict = {}
    
    for key in new_state_dict.keys():
        # Check if the key should be excluded
        if any(layer in key for layer in exclude_layers):
            # For output layers, only exclude the last layer
            if any(key.startswith(layer) for layer in ['mlp_out_X.', 'mlp_out_E.', 'mlp_out_y.']):
                if key.split('.')[-2] == str(len(getattr(model, key.split('.')[0])) - 1):
                    updated_state_dict[key] = new_state_dict[key]
                else:
                    updated_state_dict[key] = loaded_state_dict[key]
            else:
                # Keep the original initialization for excluded layers
                updated_state_dict[key] = new_state_dict[key]
        elif key in loaded_state_dict and new_state_dict[key].shape == loaded_state_dict[key].shape:
            # Load weights from the loaded state dict for non-excluded layers
            updated_state_dict[key] = loaded_state_dict[key]
        else:
            # Keep the original initialization if shapes don't match or key not in loaded dict
            updated_state_dict[key] = new_state_dict[key]
    
    # Load the updated state dict into the new model
    model.load_state_dict(updated_state_dict)

def load_all_state_dicts(cfg, model, optimizer, lr_scheduler, scaler, checkpoint_file, device_count=None):
    # TODO: Does this work with multi-GPU, or switching between GPU counts?
    checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
    if 'model_state_dict' in checkpoint.keys():
        load_weights(model, checkpoint['model_state_dict'], device_count=device_count)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint['scaler_state_dict']!={}: # need this because scalar only available in gpu (?)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'ema_state_dict' in checkpoint.keys(): # TODO: Does this need to be here?
            ema_state_dict = checkpoint['ema_state_dict']
            if check_if_dataparallel_dict(ema_state_dict) and device_count <= 1:
                ema_state_dict = dataparallel_dict_to_regular_state_dict(ema_state_dict)
            elif not check_if_dataparallel_dict(ema_state_dict) and device_count > 1:
                ema_state_dict = regular_state_dict_to_dataparallel_dict(ema_state_dict)
            model.ema.load_state_dict(ema_state_dict) # Ahh dataparallel!
    else: # Legacy, for continuing old runs where only the model weights were saved
        load_weights(model, checkpoint, device_count=device_count)

def get_model_and_train_objects(cfg, model_class, model_kwargs, parent_path, savedir,
                                run=None, epoch_num=None, load_weights_bool=True,
                                device=None, device_count=None):
    assert device is not None and device_count is not None, f'Expected device and device_count not to be None. Found device={device} and device_count={device_count}'
    
    log.info("initializing model...")
    model = model_class(cfg=cfg, **model_kwargs)
    log.info("initialized model!")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, amsgrad=True,
                                  weight_decay=cfg.train.weight_decay)
    lr_scheduler = get_lr_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler()
    last_epoch = 0
    # TODO: deprecated code, remove (loading weights is a separate function now)

    if load_weights_bool and (cfg.general.wandb.resume or cfg.general.wandb.run_id):
        assert cfg.general.wandb.resume==False or (cfg.general.wandb.resume and run is not None),\
           f'cfg.general.wandb.resume=={cfg.general.wandb.resume}, expected run != None.'
        last_epoch = epoch_num or get_latest_epoch_from_wandb(cfg)
        checkpoint_file, artifact = download_checkpoint_from_wandb(cfg, savedir, last_epoch, run=run)
        load_all_state_dicts(cfg, model, optimizer, lr_scheduler, scaler, checkpoint_file, device_count=device_count)
        artifact_name_in_wandb = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{artifact.name}"
        if run != None:
            run.use_artifact(artifact_name_in_wandb)

    if cfg.general.wandb.initialization_run_id:
        # load the weights from the initialization_run_id
        cfg.general.wandb.initialization_run_epoch # the epoch of the initialization run
        # load the weights 
        run = wandb.Api(overrides={"entity": cfg.general.wandb.entity, "project": cfg.general.wandb.project}).run(f"{cfg.general.wandb.project}/{cfg.general.wandb.initialization_run_id}")
        savename = f"epoch{cfg.general.wandb.initialization_run_epoch}"
        checkpoint_file, artifact = download_checkpoint_from_wandb(cfg, savedir, cfg.general.wandb.initialization_run_epoch, run=run, savename=savename) 
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
        initialize_model_with_loaded_weights_reset_input_output(model, checkpoint['model_state_dict'], device_count, cfg.general.wandb.exclude_some_layers_from_init)
        if cfg.general.wandb.initialization_zero_input_output:
            model.model.zero_initialize_specific_layers()

    return model, optimizer, lr_scheduler, scaler, last_epoch

def load_weights_from_wandb(cfg, epoch_num, savedir, model, optimizer, lr_scheduler, scaler, run=None, device_count=None):
    last_epoch = epoch_num or get_latest_epoch_from_wandb(cfg)
    checkpoint_file, artifact = download_checkpoint_from_wandb(cfg, savedir, last_epoch)
    load_all_state_dicts(cfg, model, optimizer, lr_scheduler, scaler, checkpoint_file, device_count)
    artifact_name_in_wandb = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{checkpoint_file.split('/')[-2]}"
    if run!=None: run.use_artifact(artifact_name_in_wandb)
    
    return model, optimizer, lr_scheduler, scaler, artifact_name_in_wandb

def load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, lr_scheduler, scaler, run=None, device_count=None):
    if not os.path.exists(os.path.join(savedir, f'epoch{epoch_num}.pt')) and \
       not os.path.exists(os.path.join(savedir, f'eval_epoch{epoch_num}.pt')):
        return load_weights_from_wandb(cfg, epoch_num, savedir, model, optimizer, lr_scheduler, 
                                       scaler, run=run, device_count=device_count)
           
    checkpoint_file = os.path.join(savedir, f'epoch{epoch_num}.pt')\
                      if os.path.exists(os.path.join(savedir, f'epoch{epoch_num}.pt'))\
                      else os.path.join(savedir, f'eval_epoch{epoch_num}.pt')
    
    print(f'checkpoint_file exists! {checkpoint_file}\n')
    load_all_state_dicts(cfg, model, optimizer, lr_scheduler, scaler, checkpoint_file, device_count)
    artifact_name_in_wandb = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{checkpoint_file.split('/')[-2]}"
    if run!=None: run.use_artifact(artifact_name_in_wandb)
    
    return model, optimizer, lr_scheduler, scaler, artifact_name_in_wandb

def get_lr_scheduler(cfg, optimizer):
    if cfg.train.lr_scheduler == 'none':
        lr_scale = lambda epoch: 1.0
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_scale)
        return lr_scheduler
    if cfg.train.lr_scheduler == 'linear':
        num_warmup_epochs = cfg.train.num_warmup_epochs
        num_annealing_epochs = cfg.train.num_annealing_epochs - num_warmup_epochs
        initial_lr = cfg.train.initial_lr
        warmup_lr = cfg.train.lr # This is what the lr should be
        final_lr = cfg.train.final_lr
        def lr_scale(epoch):
            if epoch < num_warmup_epochs:
                return ((epoch + 1) / num_warmup_epochs * (warmup_lr - initial_lr) + initial_lr) / warmup_lr
            elif epoch < num_warmup_epochs + num_annealing_epochs:
                t = (epoch - num_warmup_epochs) / num_annealing_epochs
                return ((1 - t) * warmup_lr + t * final_lr) / warmup_lr
            else:
                return final_lr / warmup_lr
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_scale)
        return lr_scheduler
    elif cfg.train.lr_scheduler == 'cosine':
        num_warmup_epochs = cfg.train.num_warmup_epochs
        num_annealing_epochs = cfg.train.num_annealing_epochs - num_warmup_epochs
        initial_lr = cfg.train.initial_lr
        warmup_lr = cfg.train.lr # This is what the lr should be
        final_lr = cfg.train.final_lr
        def lr_scale(epoch):
            if epoch < num_warmup_epochs:
                return ((epoch + 1) / num_warmup_epochs * (warmup_lr - initial_lr) + initial_lr) / warmup_lr
            elif epoch < num_warmup_epochs + num_annealing_epochs:
                t = (epoch - num_warmup_epochs) / num_annealing_epochs
                return ((np.cos(t*np.pi)+1)/2* warmup_lr + (1 - (np.cos(t*np.pi)+1)/2) * final_lr) / warmup_lr
            else:
                return final_lr / warmup_lr
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_scale)
        return lr_scheduler
    else:
        raise NotImplementedError
    
def get_standard_fcd_statistic_precalculated(cfg):
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    file = open(os.path.join(parent_path, 'data', 'fcd_stats', 'ref_combined_stat.p'), 'rb')
    fcd_stats = pickle.load(file)
    file.close()
    # fcd_torch uses 'sigma' instead of 'cov
    return {'mu': fcd_stats['mu'], 'sigma': fcd_stats['cov']}

def get_custom_fcd_statistic_precalculated(cfg, data_class, force_recompute=False):
    fcd = FCD(device=device, n_jobs=8, batch_size=512)
    
    # TODO: Refactor all code that refers to the data set dir in case the logic changes at some point
    if str(cfg.dataset.dataset_nb)!='': 
        res = '-' + str(cfg.dataset.dataset_nb)
    else:
        res = cfg.dataset.dataset_nb
    # TODO: Should use the data set dir defined directly in train.py and dataset class
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    root_path = os.path.join(base_path, cfg.dataset.datadir+res)
    # I think that the custom is to use the train set for the FCD reference set
    # TODO: There's some confusion with src/data and data/
    dataset = data_class.Dataset(stage=f'train', root=root_path)
    # This copy-pasted from the Dataset class
    # TODO: Should refactor so that we have easily access to all the data through datamodule or dataset
    # And those are then moved around so no need to define new dataset objects etc.  
    mol_path = dataset.raw_paths[dataset.file_idx].split('.csv')[0]+'mols.csv'
    all_smiles = []
    for smiles in open(mol_path, 'r'):
        all_smiles.append(smiles.strip())

    # Then precalculate statistics, could use the same root_path I guess?
    fcd_pregen_path = os.path.join(root_path, 'processed', 'fcd_custom_ref.pt')
    if not os.path.exists(fcd_pregen_path) or force_recompute:
        fcd_pregen = fcd.precalc(all_smiles)
        torch.save(fcd_pregen, fcd_pregen_path)
    else:
        fcd_pregen = torch.load(fcd_pregen_path)
    return fcd_pregen

def load_testfile(cfg, data_class):
    if str(cfg.dataset.dataset_nb)!='': 
        res = '-' + str(cfg.dataset.dataset_nb)
    else:
        res = cfg.dataset.dataset_nb
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    root_path = os.path.join(base_path, cfg.dataset.datadir+res)
    path = os.path.join(root_path, 'processed', f'test_{int(cfg.test.testfile)}.pt')
    assert os.path.exists(path), f'Path {path} does not exist.'
    dataset = data_class.Dataset(stage=f'test_{int(cfg.test.testfile)}', root=root_path)
    # path = os.path.join(parent_path, f'data/uspto-50k{res}/processed/test_{int(cfg.test.testfile)-1}.pt')
    # test_dataset = torch.load(path)
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)

    test_dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size,
                                 num_workers=cfg.dataset.num_workers, generator=g,
                                 shuffle=False)
    return test_dataloader


# import pytorch_lightning as pl
# from pytorch_lightning.utilities import rank_zero_only
# class EMA(pl.Callback):
#     """Implements EMA (exponential moving average) to any kind of model.
#     EMA weights will be used during validation and stored separately from original model weights.

#     How to use EMA:
#         - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
#           https://github.com/rwightman/pytorch-image-models/issues/102
#         - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
#           discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
#           https://github.com/rwightman/pytorch-image-models/issues/224
#         - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

#     Implementation detail:
#         - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
#         - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
#           This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
#           resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
#           performance.
#     """

#     def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
#         super().__init__()
#         self.decay = decay
#         self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
#         self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
#         self.ema_state_dict: Dict[str, torch.Tensor] = {}
#         self.original_state_dict = {}
#         self._ema_state_dict_ready = False

#     @staticmethod
#     def get_state_dict(pl_module: pl.LightningModule):
#         """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
#         For example, in pl_module has metrics, you don't want to return their parameters.

#         code:
#             # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
#             # like losses, metrics, etc.
#             patterns_to_ignore = ("metrics1", "metrics2")
#             return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
#         """
#         return pl_module.state_dict()

#     @overrides
#     def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
#         # Only keep track of EMA weights in rank zero.
#         if not self._ema_state_dict_ready and pl_module.global_rank == 0:
#             self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
#             if self.ema_device:
#                 self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in
#                                        self.ema_state_dict.items()}

#             if self.ema_device == "cpu" and self.ema_pin_memory:
#                 self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

#         self._ema_state_dict_ready = True

#     @overrides
#     def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, batch, batch_idx, *args,
#                              **kwargs) -> None:
#         if self.original_state_dict != {}:
#             # Replace EMA weights with training weights
#             pl_module.load_state_dict(self.original_state_dict, strict=False)

#     @rank_zero_only
#     def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
#         # Update EMA weights
#         with torch.no_grad():
#             for key, value in self.get_state_dict(pl_module).items():
#                 ema_value = self.ema_state_dict[key]
#                 ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

#         # Setup EMA for sampling in on_train_batch_end
#         self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
#         ema_state_dict = pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
#         self.ema_state_dict = ema_state_dict
#         assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
#             f"There are some keys missing in the ema static dictionary broadcasted. " \
#             f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
#         pl_module.load_state_dict(self.ema_state_dict, strict=False)

#         if pl_module.global_rank > 0:
#             # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
#             self.ema_state_dict = {}

#     @overrides
#     def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
#         if not self._ema_state_dict_ready:
#             return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

#     @overrides
#     def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         if not self._ema_state_dict_ready:
#             return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

#     # @overrides
#     # def on_save_checkpoint(
#     #         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict
#     # ) -> dict:
#     #     return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

#     # @overrides
#     # def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict):
#     #     self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
#     #     self.ema_state_dict = callback_state["ema_state_dict"]

