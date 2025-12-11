import os
from pathlib import Path
from datetime import datetime
from slurm_utils import create_and_submit_batch_job, get_platform_info

start_array_job = 0 # 9, 36, 50, 73
end_array_job = 0
seed = 42 # 42, 101, 90

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
SCRIPT_DIR = 'scripts'
slurm_args = get_platform_info(use_gpu=True)
slurm_args.update({
    'use_srun': True,
    'job_dir': 'jobs',
    'job_ids_file': 'job_ids.txt',
    'output_dir': 'output',
    'time': '06:00:00',
    'nodes': 1,
    'ntasks-per-node': 1,
    'cpus-per-task': 1,
    'gpus-per-node': 1,
    'mem': '100G', # 50G not enough for uspto_full
    'start_array_job': start_array_job, 
    'end_array_job': end_array_job
})

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_file = 'align_absorbing'
epochs = 1

script_args = {"script_dir": SCRIPT_DIR,
                "use_torchrun": 'false',
                "args": {
                    '+experiment': experiment_file,
                    'general.seed': seed,
                    'train.epochs': epochs,
                },
                "variables": {}}
script_args['script_name'] = 'train.py'
slurm_args['job_name'] = 'train_' + experiment_file
slurm_args['output_dir'] = os.path.join(slurm_args['output_dir'], f'train{experiment_file}')
output = create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args['interactive'])
