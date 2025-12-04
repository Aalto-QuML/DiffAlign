'''
This file contains helper functions for the multiguide package.
NOTE: best to keep this package as independent of third-party packages as possible. 
The goal is to make it callable by scripts from any conda environment.
'''

import os
from pathlib import Path
import torch
import logging
import numpy as np
import wandb

from torch.optim.lr_scheduler import LambdaLR

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]

def get_dataset(cfg, dataset_class, shuffle=True, recompute_info=False, return_datamodule=False, slices={'train':None, 'val':None, 'test':None}):
    datamodule = dataset_class.DataModule(cfg)
    datamodule.prepare_data(shuffle=shuffle, slices=slices)

    dataset_infos = dataset_class.DatasetInfos(datamodule=datamodule, cfg=cfg, recompute_info=recompute_info)
    print("Computing input/output dims")
    dataset_infos.compute_input_output_dims(datamodule=datamodule)
    print("Done computing input/output dims")

    return (datamodule, dataset_infos) if return_datamodule else dataset_infos

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

def accumulate_rxn_scores(acc_scores, new_scores, total_iterations):
    '''
        Updates the acc_scores with new metric averages taking into account the new_scores.
        
        input:
            acc_scores: accumulated scores state
            new_scores: new_scores to add to the accumulation
            total_iterations: total number of batches considered. 
        output:
            acc_scores: accumulated scores state with the new_scores added.
    '''
    for metric in new_scores.keys():
        if type(new_scores[metric])==list: # accumulates the plots
            if acc_scores[metric]==0:
                acc_scores[metric] = new_scores[metric]
            else:
                acc_scores[metric].extend(new_scores[metric])
        else:
            acc_scores[metric] += new_scores[metric].mean()/total_iterations
        
    return acc_scores
    

def average_rxn_scores(scores_list, counts_of_samples_in_list_elements):
    '''
        Averages together the scores in scores_list. 
        
        input:
            scores_list: list of dicts containing the scores
            counts_of_samples_in_list_elements: list of integers with the number of samples used to calculate the scores in scores_list
        output:
            avg: averaged scores
    '''
    total_samples = sum(counts_of_samples_in_list_elements)
    avg_scores = {}
    for i, scores in enumerate(scores_list):
        for metric in scores_list[0].keys():
            if metric not in avg_scores.keys():
                if type(scores[metric])==list:
                    avg_scores[metric] = [scores[metric]]
                else:
                    avg_scores[metric] = scores[metric] * counts_of_samples_in_list_elements[i] / total_samples
            else:
                if type(avg_scores[metric])==list:
                    avg_scores[metric].extend(scores[metric])
                else:
                    avg_scores[metric] += scores[metric]  * counts_of_samples_in_list_elements[i] / total_samples
    return avg_scores


