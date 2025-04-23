'''
    Training script: used to train new models or resume training runs from wandb.
'''
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
from torch.profiler import profile, record_function, ProfilerActivity

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign.utils import setup
from hydra.core.hydra_config import HydraConfig
from diffalign.utils import setup
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

    if cfg.general.sharing_strategy_file_system:
        torch.multiprocessing.set_sharing_strategy('file_system')
    # creates a new wandb run or resumes a run given its id
    # when a run is resumed, cfg is set to the downloaded run's config from wandb
    # periodic_table = Chem.rdchem.GetPeriodicTable()
    # weights = [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in cfg.dataset.atom_types[1:-1]]
    # print(f'weights {weights}\n')
    # exit()
    
    orig_cfg = copy.deepcopy(cfg)

    run = None
    if cfg.general.wandb.mode=='online': 
        run, cfg = setup.setup_wandb(cfg, job_type='training') 
        wandb.config.update({'experiment_group': run.id}, allow_val_change=True)
    
    if cfg.general.wandb.resume: 
        cli_overrides = setup.capture_cli_overrides()
        log.info(f'cli_overrides {cli_overrides}\n')
        run.config['train']['epochs'] = cfg.train.epochs # need this when resuming or otherwise overriding the epochs defined in the yaml file
        run.config['general']['wandb'] = {'resume': True, 'run_id': run.id, 'entity': 'najwalb', 'project': 'retrodiffuser', 'mode': 'online'}
        cfg = setup.merge_configs(default_cfg=orig_cfg, new_cfg=dict(run.config), cli_overrides=cli_overrides)
        
    # set artifact name based on cfg file (one artifact per experiment)
    # artifact contains model weights, optimizer states, and everything needed to resume training in a single object.
    # cfg = OmegaConf.create(dict(run.config))
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    assert cfg.general.task in setup.task_to_class_and_model.keys(), f'Task {cfg.general.task} not in setup.task_to_class_and_model.'
    log.info('Getting dataset infos...')
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

    start_epoch = last_epoch+1 if last_epoch>0 else 0
    model = model.to(device)
    assert start_epoch<cfg.train.epochs, f'start_epoch={start_epoch}<cfg.train.epochs={cfg.train.epochs}.'
 
    log.info(f'model {setup.count_parameters(model)}\n')
    log.info('Done loading the model...')
    batches = setup.get_batches_from_datamodule(cfg, parent_path, datamodule)
    losses = [] 
    start = time.time()
    log.info(f'Training from epoch {start_epoch} to epoch {cfg.train.epochs}\n')

    def train_wrapper(prof):
        for epoch in range(start_epoch, cfg.train.epochs):
            # Give the option to run the evaluation script here for debugging purposes
            if cfg.test.eval_before_first_epoch==True and epoch==start_epoch: # in case eval_before_first_epoch somehow gets set to string value "False" or something...
                model.eval()
                scores = model.evaluate(epoch=epoch, datamodule=datamodule, device=device)
                model.train()
            log.info(f"Training epoch {epoch}... learning rate {scheduler.get_last_lr()[0]:.6f}")
            model.train()
            t0 = time.time()
            #random.shuffle(batches)
            total_loss = 0
            #n_train_batches = len(batches)
            data_loading_time = 0
            training_time = 0
            i = 0
            indices = []
            random.shuffle(batches)
            for i, train_samples in enumerate(batches):
                start = time.time()
                train_samples = train_samples.to(device)
                indices.extend(train_samples.idx)
                data_loading_time += time.time() - start
                loss_X, loss_E, loss_atom_charges, loss_atom_chiral, loss_bond_dirs, loss = model.training_step(train_samples, i, device) # loss for one batch
                if loss == None:
                    optimizer.zero_grad()
                    continue
                if cfg.general.wandb.mode=='online':
                    wandb.log({"train_loss/loss_X": loss_X.mean().detach(),
                                "train_loss/loss_E": loss_E.mean().detach(),
                                "train_loss/loss_atom_charges": loss_atom_charges.mean().detach(),
                                "train_loss/loss_atom_chiral": loss_atom_chiral.mean().detach(),
                                "train_loss/loss_bond_dirs": loss_bond_dirs.mean().detach(),
                                "train_loss/loss": loss.mean().detach()}, commit=True)
                total_loss += loss.cpu().detach().numpy()
                if cfg.diffusion.denoiser=='neuralnet':
                    if cfg.train.use_mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    optimizer.zero_grad()
                    if cfg.neuralnet.use_ema:
                        model.ema.update()
                # Update the profiler
                if prof and i % 2000 == 0:
                    trace_location = f"trace_epoch{epoch}_{i}.json"
                    prof.step()
                    # prof.export_chrome_trace(trace_location)
                losses.append(total_loss/len(datamodule.train_dataloader()))
                i += 1
                training_time += time.time() - start
                # log.info(f"Time for the training step: {time.time() - start}")

            log.info(f"Epoch {epoch}, Data Loading Time: {data_loading_time:.3f}, Training Time: {training_time:.3f}")

            scheduler.step()
            log.info(f'Epoch {epoch}: {losses[-1]:.4f}. Time for epoch: {time.time()-t0} Loss: {total_loss}')
            if cfg.general.wandb.mode=='online':
                wandb.log({"lr": scheduler.get_last_lr()[0], "epoch": epoch, "avg_loss": losses[-1]}, commit=True)

            states_to_save = {'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(),
                              'scaler_state_dict': scaler.state_dict(),
                              'rng_state': torch.get_rng_state()}
            
            if cfg.neuralnet.use_ema: states_to_save['ema_state_dict'] =  model.ema.state_dict()
            
            ## evaluate every x epochs + last one
            if (epoch%cfg.train.eval_every_epoch==0 or epoch==cfg.train.epochs-1) and epoch != 0: 
                if cfg.train.save_models_at_all:
                    log.info(f'Saving latest model...\n')
                    filename = f'epoch{epoch}.pt'
                    torch.save(states_to_save, filename)
                    if cfg.general.wandb.mode=='online': 
                        setup.save_file_as_artifact_to_wandb(run, artifactname=f'{run.id}_model', alias=f'epoch{epoch}', filename=filename)
                
                model.eval()
                scores = model.evaluate(epoch=epoch, datamodule=datamodule, device=device)
                assert cfg.train.best_model_criterion in scores.keys(), f'{cfg.train.best_model_criterion} not in scores.'
                
                # save to wandb
                if cfg.general.wandb.mode=='online':
                    wandb.log({'sample_eval/': {k:v for k, v in scores.items() if k!='rxn_plots'}})
                    # TODO: change this to save samples as artifact with n_conditions and n_samples_per_condition added as info
                    if os.path.exists(f'samples_epoch{epoch}.txt'): wandb.save(f'samples_epoch{epoch}.txt')
                    for chain_vid_path in scores['rxn_plots']:
                        vid_name = chain_vid_path.split("/")[-1].split(".")[0]
                        wandb.log({f'sample_chains/{vid_name}': wandb.Video(chain_vid_path, fps=1, format='mp4')})

            # save every epoch
            if cfg.train.save_every_epoch and cfg.train.save_models_at_all:
                log.info(f'Saving latest model...\n')
                filename = f'epoch{epoch}.pt'
                torch.save(states_to_save, filename) 
                if cfg.general.wandb.mode=='online': setup.save_file_as_artifact_to_wandb(run, artifactname=f'{run.id}_model', alias=f'epoch{epoch}', filename=filename)
                
        end = time.time()
        log.info(f'total training time: {datetime.timedelta(seconds=end-start)}\n')
        if cfg.general.wandb.mode=='online': run.finish()

    def export_profiling_data(profiler, base_dir="profiling_data", filename_prefix="trace"):
        """
        Export profiling data to a specified directory with a unique filename.
        """
        os.makedirs(base_dir, exist_ok=True)
        filename = f"{filename_prefix}_{int(time.time())}.json"
        filepath = os.path.join(base_dir, filename)
        profiler.export_chrome_trace(filepath)
        print(f"Exported profiling data to {filepath}")

    if cfg.general.use_profiler:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=lambda prof: export_profiling_data(prof),
            record_shapes=False,
            profile_memory=True,
            with_stack=False
        ) as prof:
            train_wrapper(prof)
    else:
        train_wrapper(prof=None)

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
