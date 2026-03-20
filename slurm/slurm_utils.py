import os
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

from platform_configs import PLATFORMS, PlatformConfig

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]


def get_platform_info(use_gpu=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--use_pdb', action='store_true')
    parser.add_argument('--platform', type=str, required=True)
    args = parser.parse_args()

    if args.platform not in PLATFORMS:
        raise ValueError(f'Platform {args.platform} not supported. Choose from: {list(PLATFORMS.keys())}')

    pcfg = PLATFORMS[args.platform]
    partition = pcfg.partition_gpu if use_gpu else pcfg.partition_cpu

    return {
        'platform': pcfg.platform,
        'project': pcfg.project,
        'partition': partition,
        'with_containers': False,
        'container': None,
        'venv_path': pcfg.venv_path,
        'puhti_module': pcfg.module,
        'interactive': args.interactive,
        'use_pdb': args.use_pdb,
    }


def generate_env_setup(fh, slurm_args):
    """Write environment setup commands (WANDB, SYNTHESEUS, modules) for any platform."""
    project = slurm_args['project']
    scratch = f"/scratch/{project}"

    if slurm_args['platform'] == 'lumi':
        if slurm_args.get('with_containers'):
            fh.writelines('module purge\n')
            fh.writelines('module use /appl/local/containers/ai-modules\n')
            fh.writelines('module load singularity-AI-bindings\n\n')
            fh.writelines(f"export OMP_NUM_THREADS={slurm_args['cpus-per-task']}\n")
            fh.writelines(f"export MASTER_ADDR=$(hostname)\n")
            fh.writelines(f"export MASTER_PORT=25678\n")
            fh.writelines(f"export WORLD_SIZE=$SLURM_NPROCS\n")
            if slurm_args.get('gpus-per-node', 0) > 0:
                fh.writelines(f"export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE\n")
                fh.writelines(f"export HSA_FORCE_FINE_GRAIN_PCIE=1\n")
                fh.writelines(f"export HSA_TOOLS_LIB=1\n")
                fh.writelines(f"export NCCL_DEBUG=INFO\n")
                fh.writelines(f"export NCCL_SOCKET_IFNAME=hsn0,hsn1\n")
                fh.writelines(f"rm -rf {scratch}/miopen_db/* {scratch}/miopen_cache/*\n")
                fh.writelines(f"mkdir -p {scratch}/miopen_db {scratch}/miopen_cache\n")
                fh.writelines(f"chmod 777 {scratch}/miopen_db {scratch}/miopen_cache\n")
        else:
            fh.writelines('module use /appl/local/csc/modulefiles/\n')
            fh.writelines(f'module load {slurm_args["puhti_module"]}\n')
            fh.writelines('export OMP_NUM_THREADS=7\n')
            fh.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n")
    else:
        # puhti / mahti
        fh.writelines("module purge\n")
        fh.writelines(f"module load {slurm_args['puhti_module']}\n")
        pcfg = PLATFORMS.get(slurm_args['platform'])
        if pcfg and pcfg.extra_modules:
            for mod in pcfg.extra_modules:
                fh.writelines(f"module load {mod}\n")
        if pcfg and pcfg.extra_env:
            for k, v in pcfg.extra_env.items():
                fh.writelines(f"export {k}={v}\n")
        fh.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n")

    # Common env vars for all platforms
    fh.writelines(f"export WANDB_CACHE_DIR={scratch}/wandb_cache\n")
    fh.writelines(f"export MPLCONFIGDIR={scratch}\n")
    fh.writelines(f"export SYNTHESEUS_CACHE_DIR={scratch}/cache_Syntheseus\n")
    if slurm_args['platform'] == 'lumi' and slurm_args.get('with_containers'):
        fh.writelines(f"export WANDB_DIR={scratch}/wandb_files\n")
        fh.writelines(f"export WANDB_CONFIG_DIR={scratch}/wandb_config\n")
        fh.writelines(f"export WANDB_TEMP={scratch}/wandb_temp\n")
        fh.writelines(f"export TMPDIR={scratch}/tmp\n")
        fh.writelines(f"mkdir -p $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR $WANDB_TEMP $TMPDIR\n")
        fh.writelines(f"chmod 700 $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR $WANDB_TEMP $TMPDIR\n\n")
    elif slurm_args['platform'] in ('puhti', 'mahti'):
        fh.writelines(f"export WANDB_DATA_DIR={scratch}/wandb\n")


def add_eval_experiment_args(slurm_args, script_name,
                             script_dir, experiment_yml, experiment_name,
                             dataset_name, subset_to_evaluate,
                             augmentation, eval_epoch, num_samples, size_of_subset,
                             num_batches_per_job, resume_run_id, train_run_id, start_array_job=None, end_array_job=None,
                             load_samples='true', upload_denoising_videos='true', interactive=False, offset=0):

    script_args = {"script_dir": script_dir,
                    "args": {"+experiment": experiment_yml},
                    "variables": {}}
    experiment_dir = os.path.join(PROJECT_ROOT, 'experiments', experiment_name)
    script_args['args']['evaluation.eval_checkpoint'] = f'checkpoint_{eval_epoch}.pth'
    script_args['args']['evaluation.eval_subdir'] = f'{subset_to_evaluate}_epoch{eval_epoch}_numsamples{num_samples}'
    script_args['args']['evaluation.subset_to_evaluate'] = subset_to_evaluate
    script_args['args']['training.val_max_batches'] = '$VAL_MAX_BATCHES'
    script_args['args']['training.val_start_batch'] = '$VAL_START'
    script_args['args']['evaluation.eval_epoch'] = eval_epoch
    script_args['args']['evaluation.experiment_dir'] = experiment_dir
    script_args['args']['evaluation.num_samples'] = num_samples
    script_args['args']['evaluation.load_samples'] = load_samples
    script_args['args']['evaluation.upload_denoising_videos'] = upload_denoising_videos
    script_args['args']['dataset.augmentation'] = augmentation
    script_args['args']['dataset.dataset_name'] = dataset_name
    script_args['args']['wandb.name'] = experiment_name
    script_args['args']['wandb.resume_run_id'] = resume_run_id
    script_args['args']['wandb.train_run_id'] = train_run_id
    script_args['args']['hydra.run.dir'] = experiment_dir
    script_args['script_name'] = 'evaluate.py'
    script_args['variables']['VAL_START'] = f'$((SLURM_ARRAY_TASK_ID*{num_batches_per_job}+{offset}))'
    script_args['variables']['VAL_MAX_BATCHES'] = f'$(((SLURM_ARRAY_TASK_ID+1)*{num_batches_per_job}+{offset}))'
    if interactive:
        script_args['variables']['VAL_START'] = 0
        script_args['variables']['VAL_MAX_BATCHES'] = 1
        script_args['args']['classifier_guidance.dataset.num_workers'] = 0
        script_args['args']['evaluation.plot_denoising_video'] = 'false'
    task = script_name.split('.py')[0]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_args['job_name'] = f'{task}_{experiment_name}_numsamples{num_samples}_{time_stamp}'
    slurm_args['output_dir'] = experiment_dir
    slurm_args['job_dir'] = experiment_dir
    slurm_args['start_array_job'] = start_array_job if start_array_job is not None else 0
    slurm_args['end_array_job'] = end_array_job

    return script_args, slurm_args


def add_platform_specific_slurm_commands(fh, slurm_args):
    """Write platform-specific SBATCH resource directives."""
    if slurm_args['platform'] == 'lumi':
        fh.writelines(f"#SBATCH --nodes={slurm_args['nodes']}\n")
        if slurm_args.get('gpus-per-node', 0) > 0:
            fh.writelines(f"#SBATCH --gpus-per-node={slurm_args['gpus-per-node']}\n")
        fh.writelines(f"#SBATCH --ntasks-per-node={slurm_args['ntasks-per-node']}\n")
        fh.writelines(f"#SBATCH --cpus-per-task={slurm_args['cpus-per-task']}\n")
        fh.writelines(f"#SBATCH --mem={slurm_args['mem']}\n")
    else:
        # puhti / mahti
        fh.writelines(f"#SBATCH --nodes={slurm_args['nodes']}\n")
        gpu_partitions = ('gpu', 'gputest', 'gpusmall', 'gpumedium')
        if slurm_args['partition'] in gpu_partitions:
            pcfg = PLATFORMS.get(slurm_args['platform'])
            gpu_type = pcfg.gpu_type if pcfg else 'v100'
            fh.writelines(f"#SBATCH --gres=gpu:{gpu_type}:{slurm_args['gpus-per-node']}\n")
        fh.writelines(f"#SBATCH --cpus-per-task={slurm_args['cpus-per-task']}\n")
        fh.writelines(f"#SBATCH --mem-per-cpu={slurm_args['mem']}\n")

    fh.writelines("HYDRA_FULL_ERROR=1\n\n")
    generate_env_setup(fh, slurm_args)


def _write_job_script(job_file, script_path, script_args, slurm_args):
    """Write the inner job script (.sh) that runs the Python command."""
    os.makedirs(os.path.dirname(job_file) or '.', exist_ok=True)
    with open(job_file, 'w') as fj:
        fj.writelines("#!/bin/bash\n")
        # Platform-specific preamble
        if slurm_args['platform'] in ('puhti', 'mahti'):
            fj.writelines(f"module purge\n")
            fj.writelines(f"module load {slurm_args['puhti_module']}\n")
            pcfg = PLATFORMS.get(slurm_args['platform'])
            if pcfg and pcfg.extra_modules:
                for mod in pcfg.extra_modules:
                    fj.writelines(f"module load {mod}\n")
            if pcfg and pcfg.extra_env:
                for k, v in pcfg.extra_env.items():
                    fj.writelines(f"export {k}={v}\n")
            fj.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n")
        # Variables
        if 'variables' in script_args:
            for variable in script_args['variables']:
                fj.writelines(f"{variable}={script_args['variables'][variable]}\n")
        # Python command
        use_pdb = slurm_args.get('use_pdb', False)
        if use_pdb:
            fj.writelines(f"python3 -m pdb {script_path} \\\n")
        else:
            fj.writelines(f"python3 {script_path} \\\n")
        for arg, value in script_args['args'].items():
            fj.writelines(f"\t\t {arg}={value}\\\n")


def add_script_commands(script_args, slurm_args, fh=None, with_python=True):
    """Generate the job script and optionally write srun/execution commands to the SLURM file."""
    os.makedirs(slurm_args['job_dir'], exist_ok=True)
    job_file = os.path.join(slurm_args['job_dir'], f"{slurm_args['job_name']}.sh")
    script_path = os.path.join(PROJECT_ROOT, script_args['script_dir'], script_args['script_name'])

    if with_python:
        _write_job_script(job_file, script_path, script_args, slurm_args)
    else:
        os.makedirs(os.path.dirname(job_file) or '.', exist_ok=True)
        with open(job_file, 'w') as fj:
            fj.writelines("#!/bin/bash\n")
            if 'variables' in script_args:
                for variable in script_args['variables']:
                    fj.writelines(f"{variable}={script_args['variables'][variable]}\n")
            fj.writelines(f"{script_args['script_name']} \\\n")

    if fh is not None:
        fh.writelines(f"chmod +x {job_file}\n")
        has_gpu = slurm_args.get('gpus-per-node', 0) > 0

        if slurm_args['platform'] == 'lumi' and slurm_args.get('with_containers'):
            container_path = os.path.join(PROJECT_ROOT, slurm_args['container'])
            fh.writelines(f"CONTAINER={container_path}\n")
            fh.writelines(f"N={slurm_args['nodes']};\n")
            if slurm_args.get('use_srun'):
                fh.writelines(f"srun --ntasks=$N \\\n")
                fh.writelines(f"\t\t --ntasks-per-node=1 \\\n")
                if has_gpu:
                    fh.writelines(f"\t\t --gpus-per-node=${{SLURM_GPUS_PER_NODE}} \\\n")
            fh.writelines(f"\t\t singularity exec \\\n")
            if has_gpu:
                fh.writelines(f"\t\t --nv $CONTAINER \\\n")
                fh.writelines(f"\t\t {job_file} $N ${{SLURM_GPUS_PER_NODE}} \n")
            else:
                fh.writelines(f"\t\t $CONTAINER \\\n")
                fh.writelines(f"\t\t {job_file} $N \n")
        elif slurm_args['platform'] == 'lumi':
            fh.writelines(f"N={slurm_args['nodes']};\n")
            if has_gpu:
                fh.writelines(f"./{job_file} $N ${{SLURM_GPUS_PER_NODE}} \n")
            else:
                fh.writelines(f"./{job_file} $N \n")
        else:
            # puhti / mahti
            fh.writelines(f"N={slurm_args['nodes']};\n")
            if slurm_args.get('use_srun'):
                fh.writelines(f"srun --ntasks=$N \\\n")
                fh.writelines(f"\t\t --ntasks-per-node=1 \\\n")
                if has_gpu:
                    fh.writelines(f"\t\t --gpus-per-node=${{SLURM_GPUS_PER_NODE}} \\\n")
            fh.writelines(f"\t\t bash {job_file} $N ${{SLURM_GPUS_PER_NODE}} \n")

    return job_file


def add_general_slurm_job_setup(fh, slurm_args):
    fh.writelines("#!/bin/bash\n")
    fh.writelines(f"#SBATCH --job-name={slurm_args['job_name']}_%a.job\n")
    fh.writelines(f"#SBATCH --account={slurm_args['project']}\n")
    fh.writelines(f"#SBATCH --partition={slurm_args['partition']}\n")
    fh.writelines(f"#SBATCH --output={slurm_args['output_dir']}/{slurm_args['job_name']}_%a.out\n")
    fh.writelines(f"#SBATCH --error={slurm_args['output_dir']}/{slurm_args['job_name']}_%a.err\n")
    fh.writelines(f"#SBATCH --time={slurm_args['time']}\n")
    fh.writelines(f"#SBATCH --array={slurm_args['start_array_job']}-{slurm_args['end_array_job']}\n")
    if 'dependency' in slurm_args:
        fh.writelines(f"#SBATCH --dependency=afterok:{slurm_args['dependency']}\n")


def create_and_submit_batch_job(slurm_args, script_args, interactive=False, with_python=True):
    if interactive:
        script_args['args']['dataset.num_workers'] = 0
        script_file = add_script_commands(script_args, slurm_args, fh=None, with_python=with_python)
        print(f"Running script: {script_file}")
        result = subprocess.Popen(["bash", "-c", f"source {script_file} 1 1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        print(stdout.decode())
        print(stderr.decode())
    else:
        print(f"Creating job file for {slurm_args['job_name']} in {slurm_args['job_dir']}")
        os.makedirs(slurm_args['job_dir'], exist_ok=True)
        job_file = os.path.join(slurm_args['job_dir'], f"{slurm_args['job_name']}.job")
        with open(job_file, 'w') as fh:
            add_general_slurm_job_setup(fh, slurm_args)
            add_platform_specific_slurm_commands(fh, slurm_args)
            add_script_commands(script_args, slurm_args, fh=fh, with_python=with_python)

        result = subprocess.Popen(["/usr/bin/sbatch", job_file],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        if 'job' not in stdout.decode("utf-8"):
            print(stderr)
        else:
            job_id = stdout.decode("utf-8").strip().split('job ')[1]
            job_ids_path = os.path.join(slurm_args['job_dir'], slurm_args['job_ids_file'])
            with open(job_ids_path, 'a') as f:
                f.write(f"{slurm_args['job_name']}.job: {job_id}\n")
            print(f"=== {slurm_args['job_name']}. Slurm ID ={job_id}.")
