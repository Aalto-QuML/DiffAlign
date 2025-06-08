import os
import multiprocessing
import wandb
from omegaconf import DictConfig
import pathlib
import pickle
import time
import hydra
from datetime import datetime
from diffalign_old.utils import setup, io_utils
import numpy as np

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

os.environ['WANDB__SERVICE_WAIT'] = '1000'

def main_subprocess(cfg, queue, epoch, gpu_id, condition_range, output_file):
    assert cfg.general.wandb.run_id is not None, f"Expected a valid run ID. Got run_id={cfg.general.wandb.run_id}."
    assert cfg.general.wandb.checkpoint_epochs is not None, f"Expected a valid list of epochs. Epochs={cfg.general.wandb.checkpoint_epochs}."
    
    gpu_nb = gpu_id.split('GPU ')[-1]
    # update the output file to be different per process
    output_file = f"{output_file.split('.txt')[0]}_{gpu_nb}.txt"
    
    # The following are imported here to make sure that the GPU binding for torch is correct (after os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id))
    import torch # Must be done after setting CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available(): 
        torch.cuda.set_device(int(gpu_nb))
        log.info(f"Main subprocess running on device: {torch.cuda.current_device()}.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if condition_range: # take only a slice of the 'true' edge conditional set
        log.info(f"Device : {device}, conditions: {condition_range}")
        cfg.test.n_conditions = int(condition_range[1] - condition_range[0])
        data_slices = {'train': None, 'val': None, 'test': None}
        data_slices[cfg.diffusion.edge_conditional_set] = condition_range

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], shuffle=cfg.dataset.shuffle, 
                                                  return_datamodule=True, recompute_info=False, slices=data_slices)
    
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
    model.sample_n_conditions(dataloader=datamodule.test_dataloader(), inpaint_node_idx=None, inpaint_edge_idx=None, 
                              epoch_num=epoch, device_to_use=device, start_count=condition_range[0], filename=output_file)

    # used to check if the subprocess was successful
    queue.put({f's{condition_range[0]}e{condition_range[1]}': gpu_nb})
    log.info(f"Sampling time: {time.time()-t0}")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    assert cfg.general.wandb.run_id!='', "Please provide a run id to evaluate from."
    assert len(cfg.general.wandb.checkpoint_epochs)==1, "Please provide a single epoch to evaluate in the form of a list"+\
                                                        " (e.g. cfg.general.wandb.checkpoint_epochs=[300])."+\
                                                        f" Got len(cfg.general.wandb.checkpoint_epochs)={len(cfg.general.wandb.checkpoint_epochs)}."
    
    # save this value here because it will be changed by the processes later
    total_n_conditions = cfg.test.n_conditions
    
    # reading device_count from torch.cuda seems to be working like this
    import torch
    num_gpus = torch.cuda.device_count()
    
    # handle the case where no GPUs are available
    if num_gpus==0:
        log.info("No GPUs available, running on CPU.")
        gpu_ids = ['CPU']  # You could use None or a specific flag to indicate CPU usage
    else:
        log.info(f"Total # of GPUs available = {num_gpus}.")
        gpu_ids = [f'GPU {i}' for i in range(num_gpus)]
        
    epoch = cfg.general.wandb.checkpoint_epochs[0]
    conditions_per_gpu = cfg.test.n_conditions // max(num_gpus,0) # Give this many to the first cfg.test.n_conditions-1 gpus, and the rest to the last gpu
    condition_ranges = [(conditions_per_gpu*i, conditions_per_gpu*(i+1)) for i in range(num_gpus)]
    condition_ranges[-1] = (conditions_per_gpu*(num_gpus-1), cfg.test.n_conditions) # the actual last index
    log.info(f"Condition ranges: {condition_ranges}")

    output_file = f'samples_epoch{epoch}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}.txt'
    file = open(output_file, 'w') 
    file.close()
    
    total_cores = os.cpu_count()
    print(f"Total CPU cores: {total_cores}")
    print(f'num_cpus = os.cpu_count() {os.cpu_count()}\n')
    
    # def set_affinity(pid, cpu_cores):
    #     try:

    #         num_cpus = os.cpu_count()
    #         print(f"Available CPUs: {num_cpus}")
    #         print(f"Attempting to set affinity for PID {pid} to {cpu_cores}")

    #         # Check if the PID is valid
    #         if not psutil.pid_exists(pid):
    #             print(f"Invalid PID: {pid}")
    #             return

    #         # Check if the CPU cores are valid
    #         num_cpus = os.cpu_count()
    #         if any(core >= num_cpus or core < 0 for core in cpu_cores):
    #             print(f"Invalid CPU core in list: {cpu_cores}")
    #             return

    #         # Set CPU affinity
    #         os.sched_setaffinity(pid, cpu_cores)
    #         print(f"Set CPU affinity for PID {pid} to cores {cpu_cores}")
    #     except OSError as e:
    #         print(f"Error setting CPU affinity for PID {pid}: {e}")

    # Create processes to run the training on each GPU
    queue = multiprocessing.Queue() # To aggregate the results in the end
    processes = []
    for i in range(len(gpu_ids)):
        p = multiprocessing.Process(target=main_subprocess, args=(cfg, queue, epoch, gpu_ids[i], condition_ranges[i], output_file))
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
    condition_ranges_that_worked = [{'s': k.split('s')[-1].split('e')[0], 'key': k} for k in results.keys()]
    # This to make sure that the samples being concatenated are in the right order (in case some subprocess fails)
    condition_ranges_that_worked.sort(key=lambda x: x['s']) 
    print(f'condition_ranges_that_worked {condition_ranges_that_worked}\n')
    all_output_files = []
    for i, cond in enumerate(condition_ranges_that_worked):
        gpu_nb = results[cond['key']]
        all_output_files.append(f'{output_file.split(".txt")[0]}_{gpu_nb}.txt')
    
    # merge all the output files into one
    io_utils.merge_smiles_sample_output_files(files_to_merge=all_output_files, merged_output_file_name=output_file)
    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # without this, we get often into problems with CUDA and multiprocessing on Unix-systems. 
    main()