"""Helpers shared by scripts/train.py (and reused by sample/evaluate for setup)."""
import copy
import logging
import os
import random
import time

import torch
import wandb

from diffalign.helpers import set_seed
from diffalign.utils import setup

log = logging.getLogger(__name__)


def setup_experiment(cfg, job_type, *, preserve_entity_project: bool = False):
    """wandb init + cli overrides + optional config merge + seed.

    Handles three paths:
    - resume (training): uses orig_cfg + run.config overrides
    - load_run_config (sample/evaluate): merges downloaded run config into cfg
    - neither: just seeds and returns

    Returns (run_or_None, cfg, cli_overrides).
    """
    orig_cfg = copy.deepcopy(cfg)
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')

    run = None
    if cfg.general.wandb.mode == 'online':
        run, cfg = setup.setup_wandb(cfg, job_type=job_type)
        if job_type == 'training':
            wandb.config.update({'experiment_group': run.id}, allow_val_change=True)

    if cfg.general.wandb.resume and run is not None:
        run.config['train']['epochs'] = cfg.train.epochs
        run.config['general']['wandb'] = {'resume': True, 'run_id': run.id, 'entity': 'najwalb',
                                          'project': 'retrodiffuser', 'mode': 'online'}
        cfg = setup.merge_configs(default_cfg=orig_cfg, new_cfg=dict(run.config),
                                  cli_overrides=cli_overrides)
    elif cfg.general.wandb.load_run_config:
        saved = None
        if preserve_entity_project:
            saved = (cfg.general.wandb.entity, cfg.general.wandb.project)
        run_config = setup.load_wandb_config(cfg)
        cfg = setup.merge_configs(default_cfg=cfg, new_cfg=run_config, cli_overrides=cli_overrides)
        if saved is not None:
            cfg.general.wandb.entity, cfg.general.wandb.project = saved

    set_seed(cfg.train.seed)
    return run, cfg, cli_overrides


def build_dataloaders_and_infos(cfg, *, return_datamodule: bool = True, slices=None):
    """Wrap setup.get_dataset with the task->data_class lookup."""
    if slices is None:
        slices = {'train': None, 'val': None, 'test': None}
    assert cfg.general.task in setup.task_to_class_and_model.keys(), \
        f'Task {cfg.general.task} not in setup.task_to_class_and_model.'
    result = setup.get_dataset(
        cfg=cfg,
        dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
        shuffle=cfg.dataset.shuffle,
        return_datamodule=return_datamodule,
        recompute_info=False,
        slices=slices,
    )
    if return_datamodule:
        dataloaders, dataset_infos = result
        return dataloaders, dataset_infos
    return None, result


def build_model_and_train_objects(cfg, run, dataset_infos, *, parent_path, savedir,
                                  device, device_count, use_data_parallel: bool,
                                  load_weights_bool: bool = True):
    """Wrap setup.get_model_and_train_objects; build model_kwargs inline.

    `use_data_parallel` is passed by the caller so sample/evaluate (which ignore
    cfg.neuralnet.use_all_gpus) can set it purely from device_count, while train
    gates it on both device_count and cfg.neuralnet.use_all_gpus.
    """
    model_class = setup.task_to_class_and_model[cfg.general.task]['model_class']
    model_kwargs = {
        'dataset_infos': dataset_infos,
        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
        'use_data_parallel': use_data_parallel,
    }
    return setup.get_model_and_train_objects(
        cfg, run=run, model_class=model_class, model_kwargs=model_kwargs,
        parent_path=parent_path, savedir=savedir, device=device,
        device_count=device_count, load_weights_bool=load_weights_bool,
    )


def assemble_checkpoint_state(model, optimizer, scheduler, scaler, cfg) -> dict:
    """Bundle the state dicts into the shape expected by save_checkpoint_to_wandb."""
    states_to_save = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'rng_state': torch.get_rng_state(),
    }
    if cfg.neuralnet.use_ema:
        states_to_save['ema_state_dict'] = model.ema.state_dict()
    return states_to_save


def maybe_eval_before_first_epoch(model, dataloaders, cfg, device, *, epoch: int, start_epoch: int):
    """Run one eval pass at the very start if the flag is set."""
    if cfg.test.eval_before_first_epoch is True and epoch == start_epoch:
        model.eval()
        model.evaluate(epoch=epoch, dataloaders=dataloaders, device=device)
        model.train()


def train_one_epoch(model, batches, optimizer, scaler, scheduler, cfg, device,
                    *, epoch: int, losses: list) -> float:
    """Run one epoch of training; append the epoch's running avg to `losses`, return it."""
    log.info(f"Training epoch {epoch}... learning rate {scheduler.get_last_lr()[0]:.6f}")
    model.train()
    t0 = time.time()
    random.shuffle(batches)
    total_loss = 0
    n_train_batches = len(batches)
    for i, train_samples in enumerate(batches):
        train_samples = train_samples.to(device)
        loss_X, loss_E, loss = model.training_step(train_samples, i, device)
        if cfg.general.wandb.mode == 'online':
            wandb.log({"train_loss/loss_X": loss_X.mean().detach(),
                       "train_loss/loss_E": loss_E.mean().detach(),
                       "train_loss/loss": loss.mean().detach()}, commit=True)
        total_loss += loss.cpu().detach().numpy()
        if cfg.diffusion.denoiser == 'neuralnet':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if cfg.neuralnet.use_ema:
                model.ema.update()
        losses.append(total_loss / n_train_batches)
    scheduler.step()
    log.info(f'Epoch {epoch}: {losses[-1]:.4f}. Time for epoch: {time.time()-t0} '
             f'Loss: {total_loss}')
    if cfg.general.wandb.mode == 'online':
        wandb.log({"lr": scheduler.get_last_lr()[0], "epoch": epoch, "avg_loss": losses[-1]},
                  commit=True)
    return losses[-1]


def evaluate_and_log(model, dataloaders, epoch, cfg, device) -> dict:
    """Periodic evaluation + wandb logging of scores, sample file, chain videos."""
    model.eval()
    scores = model.evaluate(epoch=epoch, dataloaders=dataloaders, device=device)
    assert cfg.train.best_model_criterion in scores.keys(), \
        f'{cfg.train.best_model_criterion} not in scores.'

    if cfg.general.wandb.mode == 'online':
        wandb.log({'sample_eval/': {k: v for k, v in scores.items() if k != 'rxn_plots'}})
        # TODO: save samples as artifact with n_conditions/n_samples_per_condition info
        if os.path.exists(f'samples_epoch{epoch}.txt'):
            wandb.save(f'samples_epoch{epoch}.txt')
        for chain_vid_path in scores['rxn_plots']:
            vid_name = chain_vid_path.split("/")[-1].split(".")[0]
            wandb.log({f'sample_chains/{vid_name}':
                       wandb.Video(chain_vid_path, fps=1, format='mp4')})
    return scores


def save_checkpoint_to_wandb(run, cfg, states_to_save, epoch: int) -> None:
    """torch.save to epoch{N}.pt and upload as a wandb artifact when online."""
    log.info('Saving latest model...\n')
    filename = f'epoch{epoch}.pt'
    torch.save(states_to_save, filename)
    if cfg.general.wandb.mode == 'online':
        setup.save_file_as_artifact_to_wandb(
            run, artifactname=f'{run.id}_model', alias=f'epoch{epoch}', filename=filename,
        )
