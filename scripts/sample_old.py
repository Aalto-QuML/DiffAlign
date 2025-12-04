import os
import multiprocessing
import wandb
from omegaconf import DictConfig
import pathlib
import pickle
import time
import hydra
from datetime import datetime
from diffalign.helpers import get_dataset
import numpy as np
import logging

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

os.environ['WANDB__SERVICE_WAIT'] = '1000'

def main_subprocess(cfg, queue, epoch, gpu_id):
    assert cfg.general.wandb.run_id is not None, f'Expected a valid run ID. Got run_id={cfg.general.wandb.run_id}'
    assert cfg.general.wandb.checkpoint_epochs is not None, f"Expected a valid list of epochs. Epochs={cfg.general.wandb.checkpoint_epochs}"
    
    gpu_nb = gpu_id.split('GPU ')[-1]
    # The following are imported here to make sure that the GPU binding for torch is correct (after os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id))
    import torch # Must be done after setting CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(int(gpu_nb))
    log.info(f"Main subprocess running on device: {torch.cuda.current_device()}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule, dataset_infos = get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False)
        
    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=1) # evaluation here assumes a single device per epoch

    # Log the number of GPUs
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb(cfg, epoch, savedir, model, optimizer, scheduler, scaler, device_count=1)
    
    t0 = time.time()
    # NOTE: right now sample_n_conditions outputs the samples to a text file: samples.txt
    # NOTE: keep it this way or move saving here? if here => no side effects, if in function => intermediate results, no need to loop more, (-) multiple file access
    all_final_samples, all_dense_data = model.sample_n_conditions(dataloader=datamodule.test_dataloader(), inpaint_node_idx=None, 
                                                                  inpaint_edge_idx=None, epoch_num=epoch, device_to_use=device)

    # all_final_samples.reshape_bs_n_samples(bs=cfg.test.n_conditions, n_samples=cfg.test.n_samples_per_condition, n=all_final_samples.X.shape[1])
    # all_dense_data.reshape_bs_n_samples(bs=cfg.test.n_conditions, n_samples=cfg.test.n_samples_per_condition, n=all_dense_data.X.shape[1])
    
    # all_final_samples = all_final_samples.to_cpu()
    # all_dense_data = all_dense_data.to_cpu()
    
    # pickle.dump(all_final_samples, open(f'all_final_samples_epoch{epoch}.pickle', 'wb'))
    # pickle.dump(all_dense_data, open(f'all_dense_data_epoch{epoch}.pickle', 'wb'))
    queue.put({epoch: artifact_name_in_wandb})
    log.info(f"Sampling time: {time.time()-t0}")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    assert cfg.general.wandb.run_id!='', "Please provide a run id to evaluate from."
    assert cfg.general.wandb.checkpoint_epochs!='', "Please provide a list of checkpoint epochs to evaluate."
    
    # reading device_count from torch.cuda seems to be working like this
    import torch
    num_gpus = torch.cuda.device_count()
    
    # handle the case where no GPUs are available
    if num_gpus == 0:
        log.info("No GPUs available, running on CPU.")
        gpu_ids = ['CPU']  # You could use None or a specific flag to indicate CPU usage
    else:
        log.info(f"Total # of GPUs available = {num_gpus}.")
        gpu_ids = [f'GPU {i}' for i in range(num_gpus)]

    # Create processes to run the training on each GPU
    num_concurrent_runs = len(cfg.general.wandb.checkpoint_epochs) // max(1, num_gpus) + (1 if len(cfg.general.wandb.checkpoint_epochs) % max(1, num_gpus) != 0 else 0)
    queue = multiprocessing.Queue() # To aggregate the results in the end
    for cr in range(num_concurrent_runs):
        processes = []
        for i in range(len(gpu_ids)):
            if cr * len(gpu_ids) + i >= len(cfg.general.wandb.checkpoint_epochs):
                break
            epoch = cfg.general.wandb.checkpoint_epochs[cr * len(gpu_ids) + i]
            print(f'gpu_ids[i] {gpu_ids[i]}\n')
            p = multiprocessing.Process(target=main_subprocess, args=(cfg, queue, epoch, gpu_ids[i]))
            p.start()
            processes.append(p)
            print(f'done with process {i}\n')
        # Wait for all processes to complete
        for p in processes:
            print(f'join')
            p.join()
            
    ## Get the results from the queue
    results = {}
    while not queue.empty():
        results.update(queue.get())

    log.info(results)
    epochs_that_worked = list(results.keys())
    epochs_that_worked.sort() # This to make sure that epochs are logged in an ascending order (in case the epoch numbers are not usable due to wandb bug)
    print(f'epochs_that_worked {epochs_that_worked}\n')
    # add timestamp to uniquely identify the artifact versions with an alias
    # useful for example when running the same script twice (so generating the same aliases)
    # but the samples are different. This makes wandb keep the aliases in the latest version of the artifact only.
    ts = int(round(datetime.now().timestamp()))
    with wandb.init(project=cfg.general.wandb.project, entity=cfg.general.wandb.entity, resume="allow", job_type='sampling') as run:
        for i, epoch in enumerate(epochs_that_worked):
            run.use_artifact(results[epoch])
            artifact = wandb.Artifact(f'{cfg.general.wandb.run_id}_samples', type='samples')
            artifact.add_file(f'samples_epoch{epoch}.txt', name=f'samples_epoch{epoch}.txt')
            # artifact.add_file(f'all_final_samples_epoch{epoch}.pickle', name=f'all_final_samples_epoch{epoch}.pickle')
            # artifact.add_file(f'all_dense_data_epoch{epoch}.pickle', name=f'all_dense_data_epoch{epoch}.pickle')
            run.log_artifact(artifact, aliases=[f'epoch{epoch}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_{ts}'])

# TO RUN:
# python3 src/sample.py cfg.wandb.general.run_id=[RUN_ID] cfg.wandb.general.checkpoint_epochs=[CHECKPOINT_EPOCHS_COMMA_SEPARATED] [OTHER_HYDRA_OVERRIDES]
# NOTE: Only have a single script running for a single run at a time, otherwise wandb will drop some of the results
# e.g., python3 src/sample.py cfg.wandb.general.run_id=i41qfky8 cfg.wandb.general.checkpoint_epochs=5,9 cfg.test.n_conditions=128

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # without this, we get often into problems with CUDA and multiprocessing on Unix-systems. 
    main()
