import os
from pathlib import Path
from datetime import datetime
from slurm_utils import create_and_submit_batch_job, get_platform_info

start_array_job = 0 # 9, 36, 50, 73
end_array_job = 0
targets_per_job = 1
offset = 0
seed = 42 # 42, 101, 90

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
SCRIPT_DIR = 'scripts'
slurm_args = get_platform_info(use_gpu=True)
slurm_args.update({
    'use_srun': True,
    'job_dir': 'jobs',
    'job_ids_file': 'job_ids.txt',
    'output_dir': 'output',
    'time': '01:00:00',
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
num_samples = 1
experiment_name = 'sample_' + experiment_file + '_' + time_stamp

script_args = {"script_dir": SCRIPT_DIR,
                "use_torchrun": 'false',
                "args": {
                    '+experiment': experiment_file,
                    'general.seed': seed,
                    'general.name': experiment_name,
                    'general.wandb.mode': 'offline',
                    'general.wandb.checkpoint_epochs': [760],
                    'test.n_samples_per_condition': num_samples,
                    'test.n_conditions': targets_per_job,
                    'test.condition_first': '$start_idx' if not slurm_args['interactive'] else 0,
                },
                "variables": {
                    'targets_per_job': targets_per_job, # 5 molecules per job in array
                    'offset': offset,
                    'start_idx': '$((offset+(SLURM_ARRAY_TASK_ID * targets_per_job)))',
                    'end_idx': '$((start_idx+targets_per_job))'
                }}
script_args['script_name'] = 'sample.py'
slurm_args['job_name'] = experiment_name
slurm_args['output_dir'] = os.path.join(slurm_args['output_dir'], f'sample_{experiment_file}')
output = create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args['interactive'])
