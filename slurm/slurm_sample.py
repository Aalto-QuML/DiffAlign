import os
import argparse
from pathlib import Path
from datetime import datetime
from slurm_utils import create_and_submit_batch_job, get_platform_info

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
SCRIPT_DIR = 'scripts'

parser = argparse.ArgumentParser(description='Submit SLURM sampling job')
parser.add_argument('--experiment', type=str, default='align_absorbing')
parser.add_argument('--epoch', type=int, default=760)
parser.add_argument('--n-conditions', type=int, default=1)
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--start-array', type=int, default=0)
parser.add_argument('--end-array', type=int, default=0)
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
    'start_array_job': args.start_array,
    'end_array_job': args.end_array,
})

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f'sample_{args.experiment}_{time_stamp}'

script_args = {
    "script_dir": SCRIPT_DIR,
    "use_torchrun": 'false',
    "args": {
        '+experiment': args.experiment,
        'general.seed': args.seed,
        'general.name': experiment_name,
        'general.wandb.mode': 'offline',
        'general.wandb.checkpoint_epochs': [args.epoch],
        'test.n_samples_per_condition': args.n_samples,
        'test.n_conditions': args.n_conditions,
        'test.condition_first': '$start_idx' if not slurm_args['interactive'] else 0,
    },
    "variables": {
        'targets_per_job': args.n_conditions,
        'offset': args.offset,
        'start_idx': '$((offset+(SLURM_ARRAY_TASK_ID * targets_per_job)))',
        'end_idx': '$((start_idx+targets_per_job))',
    },
}
script_args['script_name'] = 'sample.py'
slurm_args['job_name'] = experiment_name
slurm_args['output_dir'] = os.path.join(slurm_args['output_dir'], f'sample_{args.experiment}')
create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args['interactive'])
