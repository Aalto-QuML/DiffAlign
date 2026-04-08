import os
from pathlib import Path
from datetime import datetime
from slurm_utils import create_and_submit_batch_job, build_platform_info

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
SCRIPT_DIR = 'scripts'

# ---------------------------------------------------------------------------
# Configuration — edit these values directly instead of passing CLI flags.
# ---------------------------------------------------------------------------
PLATFORM = 'puhti'           # one of: puhti, mahti, lumi
INTERACTIVE = False          # run inline instead of submitting via sbatch
USE_PDB = False              # launch script under pdb

EXPERIMENT = 'align_absorbing'
EPOCH = 760
N_CONDITIONS = 1             # conditions per array job
N_SAMPLES = 1                # samples per condition
SEED = 42
OFFSET = 0                   # added to the array index when computing start_idx
START_ARRAY = 0
END_ARRAY = 0                # inclusive — total array jobs = END_ARRAY - START_ARRAY + 1

TIME = '01:00:00'
# On puhti/mahti, total memory per task = MEM * CPUS_PER_TASK (mem-per-cpu).
CPUS_PER_TASK = 1
MEM = '100G'
# ---------------------------------------------------------------------------

slurm_args = build_platform_info(
    PLATFORM, use_gpu=True, interactive=INTERACTIVE, use_pdb=USE_PDB,
)
slurm_args.update({
    'use_srun': True,
    'job_dir': 'jobs',
    'job_ids_file': 'job_ids.txt',
    'output_dir': 'output',
    'time': TIME,
    'nodes': 1,
    'ntasks-per-node': 1,
    'cpus-per-task': CPUS_PER_TASK,
    'gpus-per-node': 1,
    'mem': MEM,
    'start_array_job': START_ARRAY,
    'end_array_job': END_ARRAY,
})

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f'sample_{EXPERIMENT}_{time_stamp}'
run_subdir = f'{EXPERIMENT}_{time_stamp}'
experiment_output_dir = PROJECT_ROOT / 'experiments' / run_subdir

script_args = {
    "script_dir": SCRIPT_DIR,
    "use_torchrun": 'false',
    "args": {
        '+experiment': EXPERIMENT,
        'general.seed': SEED,
        'general.name': experiment_name,
        'general.wandb.mode': 'offline',
        'general.wandb.checkpoint_epochs': [EPOCH],
        'test.n_samples_per_condition': N_SAMPLES,
        'test.n_conditions': N_CONDITIONS,
        'test.condition_first': '$start_idx' if not slurm_args['interactive'] else 0,
        'test.output_dir': str(experiment_output_dir),
    },
    "variables": {
        'targets_per_job': N_CONDITIONS,
        'offset': OFFSET,
        'start_idx': '$((offset+(SLURM_ARRAY_TASK_ID * targets_per_job)))',
        'end_idx': '$((start_idx+targets_per_job))',
    },
}
script_args['script_name'] = 'sample.py'
slurm_args['job_name'] = experiment_name
slurm_args['output_dir'] = os.path.join(slurm_args['output_dir'], run_subdir)
create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args['interactive'])
