import os
import multiprocessing
import wandb
from omegaconf import DictConfig
import pathlib
import pickle
import time
import hydra
from diffalign_old.utils import setup, io_utils
from collections import defaultdict
from utils.diffusion import helpers

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

os.environ['WANDB__SERVICE_WAIT'] = '1000'

def main_subprocess(cfg, queue, file_path, epoch, gpu_id, condition_range):
    gpu_nb = gpu_id.split('GPU ')[-1]
    
    # The following are imported here to make sure that the GPU binding for torch is correct (after os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id))
    import torch # Must be done after setting CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available(): 
        torch.cuda.set_device(int(gpu_nb))
        log.info(f"Main subprocess running on device: {torch.cuda.current_device()}.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                      shuffle=cfg.dataset.shuffle, return_datamodule=False, recompute_info=False)
        
    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=1)
    
    true_graph_data, sample_graph_data = io_utils.get_samples_from_file(cfg, file_path)
    true_graph_data = true_graph_data.mask(collapse=True)
    sample_graph_data = sample_graph_data.mask(collapse=True)
    true_graph_data.reshape_bs_n_samples(bs=cfg.test.n_conditions, n_samples=cfg.test.n_samples_per_condition, n=true_graph_data.X.shape[1])
    sample_graph_data.reshape_bs_n_samples(bs=cfg.test.n_conditions, n_samples=cfg.test.n_samples_per_condition, n=sample_graph_data.X.shape[1])
    
    sample_graph_data = sample_graph_data.to_device(device)
    true_graph_data = true_graph_data.to_device(device)
    scores = model.evaluate_from_artifact(dense_data=true_graph_data, final_samples=sample_graph_data, device=device, condition_range=condition_range)
    
    for key, value in scores.items():
        if isinstance(value, torch.Tensor):
            scores[key] = value.detach().cpu().numpy()
    log.info(scores)
    queue.put({f's{condition_range[0]}e{condition_range[1]}': dict(scores)})
    
@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    assert cfg.general.wandb.run_id is not None, "Please provide a run id to evaluate from."
    assert len(cfg.general.wandb.checkpoint_epochs)==1, "Please provide a single epoch to evaluate in the form of a list"+\
                                                        " (e.g. cfg.general.wandb.checkpoint_epochs=[300])."+\
                                                        f" Got len(cfg.general.wandb.checkpoint_epochs)={len(cfg.general.wandb.checkpoint_epochs)}."
    
    # reading device_count from torch.cuda seems to be working like this
    import torch
    num_gpus = torch.cuda.device_count() # just a safeguard to be able to run this code on cpu as well
    
    # handle the case where no GPUs are available
    if num_gpus == 0:
        log.info("No GPUs available, running on CPU.")
        gpu_ids = ['CPU'] # You could use None or a specific flag to indicate CPU usage
        num_gpus = 1
    else:
        log.info(f"Total # of GPUs available = {num_gpus}.")
        gpu_ids = [f'GPU {i}' for i in range(num_gpus)]
    
    # 1. get samples artifact from run_id
    collection_name = f"{cfg.general.wandb.run_id}_samples"
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name='samples', project=cfg.general.wandb.project).collections()
        if coll.name==collection_name
    ]
    assert len(collections)==1, f'Found {len(collections)} collections with name {collection_name}, expected 1.'
    
    coll = collections[0]
    aliases = [alias for art in coll.versions() for alias in art.aliases if 'epoch' in alias and int(alias.split('_')[0].split('epoch')[-1]) in cfg.general.wandb.checkpoint_epochs]
    log.info(f'aliases {aliases}\n')
    
    conditions_per_gpu = cfg.test.n_conditions // num_gpus  # Give this many to the first cfg.test.n_conditions-1 gpus, and the rest to the last gpu
    condition_ranges = [(conditions_per_gpu*i, conditions_per_gpu*(i+1)) for i in range(num_gpus)]
    condition_ranges[-1] = (conditions_per_gpu*(max(num_gpus,0)-1), cfg.test.n_conditions) # the actual last index
    log.info(f"Condition ranges: {condition_ranges}")
    
    # get samples from wandb
    assert len(aliases)>0, f'No artifact found for epoch {cfg.general.wandb.checkpoint_epochs}.'
    epoch = int(aliases[0].split('_')[0].split('epoch')[-1])
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    artifact_name = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{collection_name}:{aliases[0]}"
    samples_art = wandb.Api().artifact(artifact_name)
    samples_art.download(root=savedir)
    file_path = os.path.join(savedir, f'samples_epoch{epoch}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}.txt')
    
    queue = multiprocessing.Queue() # To aggregate the results in the end
    processes = []
    for i in range(len(gpu_ids)):
        p = multiprocessing.Process(target=main_subprocess, args=(cfg, queue, file_path, epoch, gpu_ids[i], condition_ranges[i]))
        p.start()
        processes.append(p)
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Get the results from the queue
    results = {}
    while not queue.empty():
        results.update(queue.get())

    log.info(results)
    assert len(results)==len(gpu_ids), f'Some processes failed. Got len(results)={len(results)} and len(gpu_ids)={len(gpu_ids)}. Evaluation numbers are not accurate. Aborting.'
    
    condition_ranges_that_worked = list(results.keys())
    scores = defaultdict(lambda:0) 
    for i, k in enumerate(condition_ranges_that_worked):
        scores = helpers.accumulate_rxn_scores(acc_scores=scores, new_scores=results[k], total_iterations=len(condition_ranges_that_worked))
        
    # pickle the scores to be uploaded to wandb via a dependent job
    art_ts = aliases[0].split('_')[-1]
    pickle.dump(dict(scores), open(os.path.join(savedir, f'scores_epoch{epoch}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_{art_ts}.pickle'), 'wb'))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # without this, we get often into problems with CUDA and multiprocessing on Unix-systems. 
    main()