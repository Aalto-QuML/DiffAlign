import os
import argparse
from pathlib import Path
from datetime import datetime
from slurm_utils import create_and_submit_batch_job, get_platform_info

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
SCRIPT_DIR = 'scripts'

parser = argparse.ArgumentParser(description='Submit SLURM training job')
parser.add_argument('--experiment', type=str, default='align_absorbing')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--time', type=str, default='01:00:00')
parser.add_argument('--mem', type=str, default='100G')
args, _ = parser.parse_known_args()

slurm_args = get_platform_info(use_gpu=True)
slurm_args.update({
    'use_srun': True,
    'job_dir': 'jobs',
    'job_ids_file': 'job_ids.txt',
    'output_dir': 'output',
    'time': args.time,
    'nodes': 1,
    'ntasks-per-node': 1,
    'cpus-per-task': 1,
    'gpus-per-node': 1,
    'mem': args.mem,
    'start_array_job': 0,
    'end_array_job': 0,
})

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_args = {
    "script_dir": SCRIPT_DIR,
    "use_torchrun": 'false',
    "args": {
        '+experiment': args.experiment,
        'train.epochs': args.epochs,
    },
    "variables": {},
}
script_args['script_name'] = 'train.py'
slurm_args['job_name'] = f'train_{args.experiment}_{time_stamp}'
slurm_args['output_dir'] = os.path.join(slurm_args['output_dir'], f'train_{args.experiment}')
create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args['interactive'])
