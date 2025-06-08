'''
    Sampling from a trained model.
'''
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import warnings
import torch
import hydra
import logging
import yaml

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign_old.utils import setup

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    cfg.general.wandb.run_name = 'download_config'
    assert cfg.general.wandb.run_id is not None, f'Need to provide run_id for the target run config. Got run_id={cfg.general.wandb.run_id}.'
    assert cfg.general.wandb.resume, f'Need wandb.resume to be True for run_id={cfg.general.wandb.run_id}.'
    
    # resume a run to download its config
    # cfg = run.config if wandb.resume==True
    run, cfg = setup.setup_wandb(cfg, job_type='training') 
    dir_ = os.path.join(parent_path, 'configs', 'old_runs')
    setup.mkdir_p(dir_)
    open(os.path.join(dir_, f'{cfg.general.wandb.run_id}.yaml'), 'w').write('# @package _global_\n')
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    yaml.dump(config_dict, open(os.path.join(dir_, f'{cfg.general.wandb.run_id}.yaml'), 'a'))
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)