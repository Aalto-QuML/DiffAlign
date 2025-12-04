import os
import multiprocessing
import wandb
from omegaconf import DictConfig
import pathlib
import pickle
import time
import hydra
from src.utils import setup

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

os.environ['WANDB__SERVICE_WAIT'] = '1000'

def main_subprocess(cfg, queue, alias, collection_name, gpu_id):
    gpu_nb = gpu_id.split('GPU ')[-1]
    
    # The following are imported here to make sure that the GPU binding for torch is correct (after os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id))
    import torch # Must be done after setting CUDA_VISIBLE_DEVICES
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
        
    # get samples from wandb
    epoch = int(alias.split('_')[0].split('epoch')[-1])
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb(cfg, epoch, savedir, model, optimizer, scheduler, scaler, device_count=1)
    artifact_name = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{collection_name}:{alias}"
    samples_art = wandb.Api().artifact(artifact_name)
    samples_art.download(root=savedir)
    
    # load the corresponding files from the downloaded wandb artifact
    final_samples = pickle.load(open(os.path.join(savedir, f'all_final_samples_epoch{epoch}.pickle'), 'rb'))
    dense_data = pickle.load(open(os.path.join(savedir, f'all_dense_data_epoch{epoch}.pickle'), 'rb'))
    dense_data = dense_data.to_device(device)
    final_samples = final_samples.to_device(device)
    
    # temporary fix
    if dense_data.X.ndim==2:
        n = dense_data.X.shape[1]
        dense_data = dense_data.get_new_object(X=dense_data.X.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n),
                                               E=dense_data.E.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n, n),
                                               y=torch.empty((cfg.test.n_conditions, cfg.test.n_samples_per_condition)),
                                               node_mask=dense_data.node_mask.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n),
                                               atom_map_numbers=dense_data.atom_map_numbers.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n),
                                               mol_assignments=dense_data.mol_assignments.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n))
        final_samples = final_samples.get_new_object(X=final_samples.X.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n),
                                                     E=final_samples.E.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n, n),
                                                     y=torch.empty((cfg.test.n_conditions, cfg.test.n_samples_per_condition)),
                                                     node_mask=final_samples.node_mask.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n),
                                                     atom_map_numbers=final_samples.atom_map_numbers.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n),
                                                     mol_assignments=final_samples.mol_assignments.reshape(cfg.test.n_conditions, cfg.test.n_samples_per_condition, n))

    scores = model.evaluate_from_artifact(dense_data=dense_data, final_samples=final_samples, device=device)
    
    for key, value in scores.items():
        if isinstance(value, torch.Tensor):
            scores[key] = value.detach().cpu().numpy()
    log.info(scores)
    queue.put({epoch: [dict(scores), samples_art, artifact_name_in_wandb]})
    
@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    assert cfg.general.wandb.run_id is not None, "Please provide a run id to evaluate from."
    assert cfg.general.wandb.checkpoint_epochs is not None, "Please provide a list of checkpoint epochs to evaluate."
    
    # reading device_count from torch.cuda seems to be working like this
    import torch
    num_gpus = torch.cuda.device_count()
    
    # handle the case where no GPUs are available
    if num_gpus == 0:
        log.info("No GPUs available, running on CPU.")
        gpu_ids = ['CPU'] # You could use None or a specific flag to indicate CPU usage
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
    assert len(collections)==1, f'Found {len(collections)} collections with the same name {collection_name}.'
    
    coll = collections[0]
    aliases = [alias for art in coll.versions() for alias in art.aliases if 'epoch' in alias and int(alias.split('_')[0].split('epoch')[-1]) in cfg.general.wandb.checkpoint_epochs]
    log.info(f'aliases {aliases}\n')
    
    num_concurrent_runs = len(aliases) // max(1, num_gpus) + (1 if len(aliases) % max(1, num_gpus) != 0 else 0)
    queue = multiprocessing.Queue() # To aggregate the results in the end
    for cr in range(num_concurrent_runs):
        processes = []
        for i in range(len(gpu_ids)):
            if cr * len(gpu_ids) + i >= len(aliases):
                break
            alias = aliases[cr * len(gpu_ids) + i]
            p = multiprocessing.Process(target=main_subprocess, args=(cfg, queue, alias, collection_name, gpu_ids[i]))
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
    epochs_that_worked = list(results.keys())
    epochs_that_worked.sort() # This to make sure that epochs are logged in an ascending order (in case the epoch numbers are not usable due to wandb bug)
    
    with wandb.init(project=cfg.general.wandb.project, entity=cfg.general.wandb.entity, resume="allow", job_type='ranking') as run:
        for i, epoch in enumerate(epochs_that_worked):
            wandb.log({'new_pipeline/':results[epoch][0]})
            run.use_artifact(results[epoch][1])

# TO RUN:
# python3 src/sample.py cfg.wandb.general.run_id=[RUN_ID] cfg.wandb.general.checkpoint_epochs=[CHECKPOINT_EPOCHS_COMMA_SEPARATED] [OTHER_HYDRA_OVERRIDES]
# NOTE: Only have a single script running for a single run at a time, otherwise wandb will drop some of the results
# e.g., python3 src/sample.py cfg.wandb.general.run_id=i41qfky8 cfg.wandb.general.checkpoint_epochs=5,9 cfg.test.n_conditions=128

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # without this, we get often into problems with CUDA and multiprocessing on Unix-systems. 
    main()
