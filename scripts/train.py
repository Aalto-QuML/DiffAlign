'''
    Training script: used to train new models or resume training runs from wandb.
'''
import datetime
import logging
import os
import pathlib
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign.training.helpers import (
    assemble_checkpoint_state,
    build_dataloaders_and_infos,
    build_model_and_train_objects,
    evaluate_and_log,
    maybe_eval_before_first_epoch,
    save_checkpoint_to_wandb,
    setup_experiment,
    train_one_epoch,
)
from diffalign.utils import setup

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=PossibleUserWarning)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base='1.1', config_path='../configs', config_name='default')
def main(cfg: DictConfig):
    run, cfg, _cli = setup_experiment(cfg, job_type='training')

    log.info('Getting dataset infos...')
    dataloaders, dataset_infos = build_dataloaders_and_infos(cfg)

    log.info('Getting model...')
    device_count = torch.cuda.device_count()
    savedir = None
    if cfg.general.wandb.resume:
        savedir = os.path.join(parent_path, 'experiments', cfg.general.wandb.run_id)
    model, optimizer, scheduler, scaler, last_epoch = build_model_and_train_objects(
        cfg, run, dataset_infos,
        parent_path=parent_path, savedir=savedir, device=device,
        device_count=device_count,
        use_data_parallel=device_count > 1 and cfg.neuralnet.use_all_gpus,
    )
    print(f'last_epoch: {last_epoch}')
    start_epoch = last_epoch + 1 if (last_epoch is not None and last_epoch != 0) else 0
    model = model.to(device)
    assert start_epoch < cfg.train.epochs, \
        f'start_epoch={start_epoch}<cfg.train.epochs={cfg.train.epochs}.'
    log.info(f'model {setup.count_parameters(model)}\n')
    log.info('Done loading the model...')

    batches = setup.get_batches_from_datamodule(cfg, parent_path, dataloaders)
    losses = []
    start = time.time()
    log.info(f'Training from epoch {start_epoch} to epoch {cfg.train.epochs}\n')
    for epoch in range(start_epoch, cfg.train.epochs):
        maybe_eval_before_first_epoch(model, dataloaders, cfg, device,
                                      epoch=epoch, start_epoch=start_epoch)
        train_one_epoch(model, batches, optimizer, scaler, scheduler, cfg, device,
                        epoch=epoch, losses=losses)
        states_to_save = assemble_checkpoint_state(model, optimizer, scheduler, scaler, cfg)

        if epoch % cfg.train.eval_every_epoch == 0 or epoch == cfg.train.epochs - 1:
            evaluate_and_log(model, dataloaders, epoch, cfg, device)
            if cfg.train.save_models_at_all:
                save_checkpoint_to_wandb(run, cfg, states_to_save, epoch)

        if cfg.train.save_every_epoch and cfg.train.save_models_at_all:
            save_checkpoint_to_wandb(run, cfg, states_to_save, epoch)

    log.info(f'total training time: {datetime.timedelta(seconds=time.time()-start)}\n')
    if cfg.general.wandb.mode == 'online':
        run.finish()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
