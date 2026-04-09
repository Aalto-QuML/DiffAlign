import os
import math
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
N_CONDITIONS = 500         # must match what was used at sampling time
N_SAMPLES = 100              # must match what was used at sampling time
EDGE_CONDITIONAL_SET = 'test'  # train | val | test

# Directory containing the samples_*.gz files. Outputs are written here too.
# Relative paths are resolved against PROJECT_ROOT.
SAMPLES_DIR = 'experiments/align_absorbing_20260407_144212'

SEED = 42
OFFSET = 0
START_ARRAY = 9
# Inclusive last array index. For uspto-50k test set with N_CONDITIONS=1 use 4948
# (4949 conditions). With N_CONDITIONS=k use ceil(4949 / k) - 1.
END_ARRAY = 9

TIME = '24:00:00'
# On puhti/mahti, total memory per task = MEM * CPUS_PER_TASK (mem-per-cpu).
# A single eval condition padded to ~250 nodes with N_SAMPLES=100 and
# diffusion_steps_eval=100 has been observed to need ~120 GiB of host RAM,
# so the default below allocates 200 GiB total. Bump CPUS_PER_TASK further
# (or lower N_SAMPLES) if you still hit OOM, or pair it with the override
# DIFFUSION_STEPS_EVAL below to cut memory linearly.
CPUS_PER_TASK = 2
MEM = '100G'
# Optional override for cfg.diffusion.diffusion_steps_eval. Set to None to keep
# the value from the experiment yaml (100 for align_absorbing). Halving this
# roughly halves the peak memory of compute_Lt_all.
DIFFUSION_STEPS_EVAL = None
# Chunk size for the per-condition n_samples batch when computing ELBOs. Set
# this to a value < N_SAMPLES (e.g. 25 or 10) to lower GPU peak memory; the
# model forward will be called more times but each call sees a smaller batch.
# None or 0 disables chunking (single forward over the full N_SAMPLES batch).
ELBO_BATCH_SIZE = 25
# ---------------------------------------------------------------------------
# /Users/laabidn1/DiffAlign/slurm/slurm_evaluate.py
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
experiment_name = f'eval_{EXPERIMENT}_{time_stamp}'

samples_dir = SAMPLES_DIR
if not os.path.isabs(samples_dir):
    samples_dir = str(PROJECT_ROOT / samples_dir)

eval_args = {
    '+experiment': EXPERIMENT,
    'general.seed': SEED,
    'general.name': experiment_name,
    'general.wandb.mode': 'offline',
    'general.wandb.load_run_config': 'false',
    'general.wandb.checkpoint_epochs': [EPOCH],
    'test.n_samples_per_condition': N_SAMPLES,
    'test.n_conditions': N_CONDITIONS,
    'test.condition_first': '$start_idx' if not slurm_args['interactive'] else 0,
    'test.condition_index': 0,
    'diffusion.edge_conditional_set': EDGE_CONDITIONAL_SET,
    'hydra.run.dir': samples_dir,
}
if DIFFUSION_STEPS_EVAL is not None:
    eval_args['diffusion.diffusion_steps_eval'] = DIFFUSION_STEPS_EVAL
if ELBO_BATCH_SIZE is not None:
    eval_args['test.elbo_batch_size'] = ELBO_BATCH_SIZE

script_args = {
    "script_dir": SCRIPT_DIR,
    "use_torchrun": 'false',
    "args": eval_args,
    "variables": {
        'targets_per_job': N_CONDITIONS,
        'offset': OFFSET,
        'start_idx': '$((offset+(SLURM_ARRAY_TASK_ID * targets_per_job)))',
        'end_idx': '$((start_idx+targets_per_job))',
    },
}
script_args['script_name'] = 'evaluate.py'
slurm_args['job_name'] = experiment_name
slurm_args['output_dir'] = os.path.join(slurm_args['output_dir'], experiment_name)
create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args['interactive'])
