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
from subprocess import call

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

os.environ['WANDB__SERVICE_WAIT'] = '1000'

def main_subprocess(gpu_id):
    gpu_nb = gpu_id.split('GPU ')[-1]
    
    # The following are imported here to make sure that the GPU binding for torch is correct (after os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id))
    import torch # Must be done after setting CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available(): 
        torch.cuda.set_device(int(gpu_nb))
        log.info(f"Main subprocess running on device: {torch.cuda.current_device()}.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'torch.cuda.get_device_name(gpu_nb) {torch.cuda.get_device_name(int(gpu_nb))}\n')
    # print(f'torch.cuda._raw_device_uuid_nvml() {torch.cuda._raw_device_uuid_nvml()}\n')
    print(call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free"]))
    print(f'in process {gpu_nb}\n')
    for i in range(100000000):
        print(f'i {i}\n')

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
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
    
    queue = multiprocessing.Queue() # To aggregate the results in the end
    processes = []
    for i in range(len(gpu_ids)):
        p = multiprocessing.Process(target=main_subprocess, args=(gpu_ids[i],))
        p.start()
        processes.append(p)
        # Wait for all processes to complete
        for p in processes:
            p.join()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # without this, we get often into problems with CUDA and multiprocessing on Unix-systems. 
    main()