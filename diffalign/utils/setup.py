import os
from copy import deepcopy
from typing import Optional, Union, Dict
from omegaconf import OmegaConf, open_dict
import omegaconf
import wandb
import pathlib
import pickle
import torch
import yaml
import sys
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.loader import DataLoader
import io
from fcd_torch import FCD
from diffalign.utils import graph
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import logging
from collections import OrderedDict
import re
import copy

from diffalign.constants import MAX_ATOMS_RXN

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_task_map():
    from diffalign.datasets import supernode_dataset
    from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
    return {
        'rxn': {'data_class': supernode_dataset,
                'model_class': DiscreteDenoisingDiffusionRxn},
    }


class _LazyTaskMap:
    """Lazily builds task_to_class_and_model on first access to break circular imports."""
    _map = None

    def _ensure(self):
        if self._map is None:
            self._map = _build_task_map()

    def __getitem__(self, key):
        self._ensure()
        return self._map[key]

    def __contains__(self, key):
        self._ensure()
        return key in self._map

    def keys(self):
        self._ensure()
        return self._map.keys()

    def items(self):
        self._ensure()
        return self._map.items()

    def values(self):
        self._ensure()
        return self._map.values()


task_to_class_and_model = _LazyTaskMap()

def parse_value(value):
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

def get_batches_from_datamodule(cfg, parent_path, dataloaders):
    if cfg.train.batch_by_size:
        datadir = cfg.dataset.datadir
        if cfg.dataset.dataset_nb != '':
            datadir += '-' + str(cfg.dataset.dataset_nb)
        data_list_path = os.path.join(parent_path, datadir, 'processed', 'train.pickle')
        train_data = pickle.load(open(data_list_path, 'rb'))
        batches, sizes_found = graph.batch_graph_by_size(input_data_list=train_data,
                                                         size_bins=cfg.dataset.size_bins['train'],
                                                         batchsize_bins=cfg.dataset.batchsize_bins['train'],
                                                         get_batches=True)
    else:
        batches = [b for b in dataloaders['train']]

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
    
    api = wandb.Api() 
    run = api.run(f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{cfg.general.wandb.run_id}") 
    run_config = OmegaConf.create(dict(run.config))
    # run.finish() <- Can't have this, the run could actually be running if this is used during evaluation

    return run_config

def setup_wandb(cfg, job_type):
    assert (cfg.general.wandb.resume==False) or (cfg.general.wandb.resume and cfg.general.wandb.run_id!=''), "If wandb_resume is True, wandb.run_id must be set"
    # tags and groups
    kwargs = {'entity': cfg.general.wandb.entity, 'project': cfg.general.wandb.project, 'job_type': job_type,
              'group': cfg.general.wandb.group, 'tags': cfg.general.wandb.tags, 'mode': cfg.general.wandb.mode}
    
    log.info(kwargs)
    if cfg.general.wandb.resume:
        kwargs['id'] = cfg.general.wandb.run_id
        kwargs['resume'] = 'allow'
        run = wandb.init(**kwargs)
        run.config['train']['epochs'] = cfg.train.epochs # need this when resuming or otherwise overriding the epochs defined in the yaml file
        run.config['general']['wandb'] = {'resume': True, 'run_id': run.id, 'entity': 'najwalb', 
                                        'project': 'retrodiffuser', 'mode': 'online'}
        cfg = OmegaConf.create(dict(run.config))
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

def download_checkpoint_from_wandb(cfg, savedir, epoch_num, run=None):
    # Download the checkpoint
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
    
    downloaded_file = os.path.join(downloaded_dir, f"epoch{epoch_num}.pt")

    return downloaded_file, a

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



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataset(cfg, dataset_class, shuffle=True, recompute_info=False, return_datamodule=False, slices={'train':None, 'val':None, 'test':None}):
    dataset_infos = dataset_class.build_dataset_info(cfg)

    if return_datamodule:
        dataloaders = dataset_class.create_dataloaders(cfg, slices=slices)
        if recompute_info:
            datadist_dir = cfg.dataset.datadist_dir
            if cfg.dataset.dataset_nb != '':
                datadist_dir += '-' + str(cfg.dataset.dataset_nb)
            dataset_class.compute_dataset_statistics(dataloaders, cfg.dataset.atom_types, datadist_dir)
            dataset_infos = dataset_class.build_dataset_info(cfg)
        return dataloaders, dataset_infos

    return dataset_infos

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

def load_weights(model, model_state_dict, device_count=None):
    assert device_count is not None, f'Expected device_count to not be None. Found device_count={device_count}'

    if check_if_dataparallel_dict(model_state_dict) and device_count <= 1:
        model_state_dict = dataparallel_dict_to_regular_state_dict(model_state_dict)
    elif not check_if_dataparallel_dict(model_state_dict) and device_count > 1:
        model_state_dict = regular_state_dict_to_dataparallel_dict(model_state_dict)
        
    model.load_state_dict(model_state_dict)
    
    return model

def load_all_state_dicts(cfg, model, optimizer, lr_scheduler, scaler, checkpoint_file, device_count=None):
    # TODO: Does this work with multi-GPU, or switching between GPU counts?
    checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
    if 'model_state_dict' in checkpoint.keys():
        load_weights(model, checkpoint['model_state_dict'], device_count=device_count)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint['scaler_state_dict']!={}: # need this because scaler only available in gpu (?)
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

def get_model_and_train_objects(cfg, model_class, model_kwargs, parent_path, savedir, run=None, epoch_num=None, 
                                load_weights_bool=True, device=None, device_count=None):
    assert device is not None and device_count is not None, f'Expected device and device_count not to be None. Found device={device} and device_count={device_count}'
    
    model = model_class(cfg=cfg, **model_kwargs)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, amsgrad=True, weight_decay=cfg.train.weight_decay) 
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
        run.use_artifact(artifact_name_in_wandb)

    return model, optimizer, lr_scheduler, scaler, last_epoch

def load_weights_from_wandb(cfg, epoch_num, savedir, model, optimizer, lr_scheduler, scaler, run=None, device_count=None):
    """Load model weights, checking local files first, downloading from wandb if needed."""
    last_epoch = epoch_num or get_latest_epoch_from_wandb(cfg)

    # Check local first
    local_path = os.path.join(savedir, f'epoch{last_epoch}.pt')
    local_path_alt = os.path.join(savedir, f'eval_epoch{last_epoch}.pt')
    if os.path.exists(local_path):
        checkpoint_file = local_path
    elif os.path.exists(local_path_alt):
        checkpoint_file = local_path_alt
    else:
        checkpoint_file, artifact = download_checkpoint_from_wandb(cfg, savedir, last_epoch)

    load_all_state_dicts(cfg, model, optimizer, lr_scheduler, scaler, checkpoint_file, device_count)
    artifact_name_in_wandb = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{checkpoint_file.split('/')[-2]}"
    if run is not None:
        run.use_artifact(artifact_name_in_wandb)

    return model, optimizer, lr_scheduler, scaler, artifact_name_in_wandb

# Backward-compatible alias
load_weights_from_wandb_no_download = load_weights_from_wandb

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

class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    #@overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in
                                       self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    #@overrides
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, batch, batch_idx, *args,
                             **kwargs) -> None:
        if self.original_state_dict != {}:
            # Replace EMA weights with training weights
            pl_module.load_state_dict(self.original_state_dict, strict=False)

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

        # Setup EMA for sampling in on_train_batch_end
        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        ema_state_dict = pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        self.ema_state_dict = ema_state_dict
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    #@overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

    #@overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.
