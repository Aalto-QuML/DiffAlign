import os
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]  

def get_platform_info(use_gpu=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--use_pdb', action='store_true')
    parser.add_argument('--platform', type=str)
    args = parser.parse_args()
    platform = args.platform

    if platform == 'puhti':
        project = 'project_2015607'
        partition = 'gpu' if use_gpu else 'small'
        with_containers = False
        puhti_module = 'pytorch/2.4'
        venv_path = '/projappl/project_2015607/diffalign'
        container = None
    elif platform == 'mahti':
        project = 'project_2015607'
        partition = 'gpusmall' if use_gpu else 'small'
        with_containers = False
        puhti_module = 'pytorch/2.4'
        venv_path = '/projappl/project_2015607/diffalign'
        container = None
    elif platform == 'lumi':
        platform = 'lumi'
        project = 'project_462001028'
        partition = 'small-g' if use_gpu else 'small'
        with_containers = False
        container = None
        venv_path = None
        puhti_module = 'pytorch/2.4'
        venv_path = '/projappl/project_462001028/multiguide'
    else:
        raise ValueError(f'Platform {platform} not supported')
    return {
        'platform': platform,
        'project': project,
        'partition': partition,
        'with_containers': with_containers,
        'container': container,
        'venv_path': venv_path,
        'puhti_module': puhti_module,
        'interactive': args.interactive,
        'use_pdb': args.use_pdb
    }

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
    # Calculate the value directly - this is the preferred method
    script_args['args']['evaluation.eval_checkpoint'] = f'checkpoint_{eval_epoch}.pth'
    script_args['args']['evaluation.eval_subdir'] = f'{subset_to_evaluate}_epoch{eval_epoch}_numsamples{num_samples}'
    script_args['args']['evaluation.subset_to_evaluate'] = subset_to_evaluate
    script_args['args']['training.val_max_batches'] = '$VAL_MAX_BATCHES'
    # this should not be trusted because it's often higher than the actual number of batches
    # it is set correctly inside the validation function with min(len(val_loader), config.training.val_max_batches)
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
    # if script_name=='evaluate.py' and end_array_job is None:
    #     total_num_batches = ((size_of_subset*augmentation) + val_batch_size - 1) // val_batch_size # 17
    #     end_array_job = total_num_batches // num_batches_per_job 
    slurm_args['end_array_job'] = end_array_job 

    return script_args, slurm_args


def add_platform_specific_slurm_commands(fh, slurm_args):
    if slurm_args['platform'] == 'lumi':
        fh.writelines(f"#SBATCH --nodes={slurm_args['nodes']}\n")
        if 'gpus-per-node' in slurm_args and slurm_args['gpus-per-node'] > 0:
            fh.writelines(f"#SBATCH --gpus-per-node={slurm_args['gpus-per-node']}\n")
        fh.writelines(f"#SBATCH --ntasks-per-node={slurm_args['ntasks-per-node']}\n")
        fh.writelines(f"#SBATCH --cpus-per-task={slurm_args['cpus-per-task']}\n")
        fh.writelines(f"#SBATCH --mem={slurm_args['mem']}\n")
        fh.writelines("HYDRA_FULL_ERROR=1\n\n")
        if slurm_args['with_containers']:
            # Add MIOpen directory setup
            fh.writelines('module purge\n')
            fh.writelines('module use /appl/local/containers/ai-modules\n')
            fh.writelines('module load singularity-AI-bindings\n\n')
            fh.writelines(f"export OMP_NUM_THREADS={slurm_args['cpus-per-task']}\n")

            fh.writelines(f"HYDRA_FULL_ERROR=1\n")
            fh.writelines(f"export MASTER_ADDR=$(hostname)\n")
            fh.writelines(f"export MASTER_PORT=25678\n")
            fh.writelines(f"export WORLD_SIZE=$SLURM_NPROCS\n")
            # miopen db and cache

            fh.writelines(f"export MASTER_ADDR=$(hostname)\n")
            fh.writelines(f"export MASTER_PORT=25678\n")
            fh.writelines(f"export WORLD_SIZE=$SLURM_NPROCS\n")
            if 'gpus-per-node' in slurm_args and slurm_args['gpus-per-node'] > 0:
                fh.writelines(f"export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE\n\n")
                fh.writelines(f"export HSA_FORCE_FINE_GRAIN_PCIE=1\n")
                fh.writelines(f"export HSA_TOOLS_LIB=1\n")
                fh.writelines(f"export HSA_TOOLS_LIB=1\n")
                fh.writelines(f"export NCCL_DEBUG=INFO\n")
                fh.writelines(f"export NCCL_SOCKET_IFNAME=hsn0,hsn1\n")
                fh.writelines(f"rm -rf /scratch/{slurm_args['project']}/miopen_db/*\n")
                fh.writelines(f"rm -rf /scratch/{slurm_args['project']}/miopen_cache/*\n")
                # Add MIOpen directory setup
                fh.writelines(f"mkdir -p /scratch/{slurm_args['project']}/miopen_db\n")
                fh.writelines(f"mkdir -p /scratch/{slurm_args['project']}/miopen_cache\n")
                fh.writelines(f"chmod 777 /scratch/{slurm_args['project']}/miopen_db\n")
                fh.writelines(f"chmod 777 /scratch/{slurm_args['project']}/miopen_cache\n")
            fh.writelines(f"export WANDB_DIR=/scratch/{slurm_args['project']}/wandb_files\n")
            fh.writelines(f"export WANDB_CACHE_DIR=/scratch/{slurm_args['project']}/wandb_cache\n")
            fh.writelines(f"export MPLCONFIGDIR=/scratch/{slurm_args['project']}\n")
            fh.writelines(f"export WANDB_CONFIG_DIR=/scratch/{slurm_args['project']}/wandb_config\n")
            fh.writelines(f"export WANDB_TEMP=/scratch/{slurm_args['project']}/wandb_temp\n")
            fh.writelines(f"export TMPDIR=/scratch/{slurm_args['project']}/tmp\n")
            fh.writelines(f"export SYNTHESEUS_CACHE_DIR=/scratch/{slurm_args['project']}/cache_Syntheseus\n")
            fh.writelines(f"mkdir -p $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR $WANDB_TEMP $TMPDIR\n")
            fh.writelines(f"chmod 700 $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR $WANDB_TEMP $TMPDIR\n\n")
        else:
            fh.writelines('module use /appl/local/csc/modulefiles/\n')
            fh.writelines(f'module load {slurm_args["puhti_module"]}\n')
            fh.writelines('export OMP_NUM_THREADS=7\n')
            fh.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n")
            fh.writelines(f"export SYNTHESEUS_CACHE_DIR=/scratch/{slurm_args['project']}/cache_Syntheseus\n")
            fh.writelines(f"export WANDB_CACHE_DIR=/scratch/{slurm_args['project']}/wandb_cache\n")
            fh.writelines(f"export MPLCONFIGDIR=/scratch/{slurm_args['project']}\n")
    elif slurm_args['platform'] == 'puhti' or slurm_args['platform'] == 'mahti':
        # Rest of puhti code unchanged

        # #SBATCH --partition=gpusmall
        # #SBATCH --ntasks=1
        # #SBATCH --cpus-per-task=32
        # #SBATCH --gres=gpu:a100:1,nvme:950

        # # If multithreading is OpenMP implementation then define also OMP_NUM_THREADS environment variable
        # export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

        fh.writelines(f"#SBATCH --nodes={slurm_args['nodes']}\n")
        if slurm_args['partition'] == 'gpu' or slurm_args['partition'] == 'gputest' or slurm_args['partition'] == 'gpusmall' or slurm_args['partition'] == 'gpumedium':
            if slurm_args['platform'] == 'mahti':
                fh.writelines(f"#SBATCH --gres=gpu:a100:{slurm_args['gpus-per-node']}\n")
            else:
                fh.writelines(f"#SBATCH --gres=gpu:v100:{slurm_args['gpus-per-node']}\n")
        fh.writelines(f"#SBATCH --cpus-per-task={slurm_args['cpus-per-task']}\n")
        fh.writelines(f"#SBATCH --mem-per-cpu={slurm_args['mem']}\n")
        fh.writelines("HYDRA_FULL_ERROR=1\n\n")
        fh.writelines(f"export WANDB_CACHE_DIR=/scratch/{slurm_args['project']}/wandb\n")
        fh.writelines(f"export WANDB_DATA_DIR=/scratch/{slurm_args['project']}/wandb\n")
        fh.writelines(f"export MPLCONFIGDIR=/scratch/{slurm_args['project']}\n")
        #export SYNTHESEUS_CACHE_DIR=/pfs/lustrep2/scratch/project_462001028/cache_Syntheseus
        fh.writelines(f"export SYNTHESEUS_CACHE_DIR=/scratch/{slurm_args['project']}/cache_Syntheseus\n")
        fh.writelines("module purge\n")
        fh.writelines(f"module load {slurm_args['puhti_module']}\n")
        fh.writelines(f"module load gcc/13.2.0\n")
        fh.writelines(f"export LD_PRELOAD=/appl/spack/v022/install-tree/gcc-8.5.0/gcc-13.2.0-hgaeyz/lib64/libstdc++.so.6:$LD_PRELOAD\n")
        fh.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n")
    else:
        raise ValueError(f"Platform {slurm_args['platform']} not supported")

def add_script_commands(script_args, slurm_args, fh=None, with_python=True):
    print(f"slurm_args['platform']: {slurm_args['platform']}")
    if slurm_args['platform'] == 'lumi':
        return add_script_commands_lumi(script_args, slurm_args, fh, with_python)
    elif slurm_args['platform'] == 'puhti' or slurm_args['platform'] == 'mahti':
        return add_script_commands_puhti(script_args, slurm_args, fh, with_python)
    else:
        raise ValueError(f"Platform {slurm_args['platform']} not supported")

def add_script_commands_puhti(script_args, slurm_args, fh=None, with_python=True):
    os.makedirs(slurm_args['job_dir'], exist_ok=True)
    job_file = os.path.join(slurm_args['job_dir'], 
                            f"{slurm_args['job_name']}.sh")
    script_path = os.path.join(PROJECT_ROOT,
                                script_args['script_dir'], 
                                script_args['script_name'])
    venv_path = os.path.join(PROJECT_ROOT, slurm_args['venv_path'])
    # /scratch/project_2015643/multiguide/slurm/jobs
    # TODO: Could load the yaml file in question the experiment name and log with that locally to outputs/
    with open(job_file, 'w') as fj:
        print(f'PROJECT_ROOT: {PROJECT_ROOT}')
        fj.writelines(f"#!/bin/bash\n")
        fj.writelines(f"module purge\n")
        fj.writelines(f"module load {slurm_args['puhti_module']}\n")
        fj.writelines(f"module load gcc/13.2.0\n")
        fj.writelines(f"export LD_LIBRARY_PATH=/appl/spack/v022/install-tree/gcc-8.5.0/gcc-13.2.0-hgaeyz/lib64:$LD_LIBRARY_PATH\n")
        fj.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n")
        if 'variables' in script_args:
            for variable in script_args['variables']:
                fj.writelines(f"{variable}={script_args['variables'][variable]}\n")
        if with_python:
            fj.writelines(f"python3 {script_path} \\\n")
            # Add each argument to command
            for arg, value in script_args['args'].items():
                fj.writelines(f"\t\t {arg}={value}\\\n") # adapted to hydra
        else:
            # TODO: refactor this to make it as flexible as possible
            fj.writelines(f"{script_args['script_name']} \\\n")

    if fh is not None:
        fh.writelines(f"chmod +x {job_file}\n")
        fh.writelines(f"N={slurm_args['nodes']};\n")
        if slurm_args['use_srun']:
            fh.writelines(f"srun --ntasks=$N \\\n")
            fh.writelines(f"\t\t --ntasks-per-node=1 \\\n")
            if 'gpus-per-node' in slurm_args and slurm_args['gpus-per-node'] > 0:
                fh.writelines(f"\t\t --gpus-per-node=${{SLURM_GPUS_PER_NODE}} \\\n")
            #fh.writelines(f"\t\t singularity exec \\\n")
        fh.writelines(f"\t\t bash {job_file} $N ${{SLURM_GPUS_PER_NODE}} \n")

    return job_file

def add_script_commands_lumi(script_args, slurm_args, fh=None, with_python=True):
    job_file = os.path.join(slurm_args['job_dir'], 
                            f"{slurm_args['job_name']}.sh")
    script_path = os.path.join(PROJECT_ROOT,
                                script_args['script_dir'], 
                                script_args['script_name'])
    
    if slurm_args['with_containers']:
        venv_path = os.path.join(PROJECT_ROOT, slurm_args['venv_path'])
        container_path = os.path.join(PROJECT_ROOT, slurm_args['container'])
        os.makedirs(slurm_args['job_dir'], exist_ok=True)
        # TODO: Could load the yaml file in question the experiment name and log with that locally to outputs/
        with open(job_file, 'w') as fj:
            print(f'PROJECT_ROOT: {PROJECT_ROOT}')
            fj.writelines(f"#!/bin/bash\n")
            fj.writelines(f"source {venv_path}/bin/activate\n")
            if 'variables' in script_args:
                for variable in script_args['variables']:
                    fj.writelines(f"{variable}={script_args['variables'][variable]}\n")
            print(f"script_args['use_torchrun']: {script_args['use_torchrun']}")
            if with_python:
                if script_args['use_torchrun'] == 'true':
                    fj.writelines(f"torchrun --nnodes=$1 \\\n" +\
                                f"\t\t --nproc-per-node=$2 \\\n" +\
                                f"\t\t {script_path} \\\n")
                elif 'use_pdb' in slurm_args and slurm_args['use_pdb']:
                    fj.writelines(f"python3 -m pdb {script_path} \\\n")
                else:
                    fj.writelines(f"python3 {script_path} \\\n")
            else:
                # NOTE: not sure if can use torchrun without python
                fj.writelines(f"{script_path} \\\n")
            # Add each argument to command
            for arg, value in script_args['args'].items():
                fj.writelines(f"\t\t {arg}={value}\\\n") # adapted to hydra
        if fh is not None:
            fh.writelines(f"chmod +x {job_file}\n")
            fh.writelines(f"CONTAINER={container_path}\n")
            fh.writelines(f"N={slurm_args['nodes']};\n")
            if slurm_args['use_srun']:
                fh.writelines(f"srun --ntasks=$N \\\n")
                fh.writelines(f"\t\t --ntasks-per-node=1 \\\n")
                if 'gpus-per-node' in slurm_args and slurm_args['gpus-per-node'] > 0:
                    fh.writelines(f"\t\t --gpus-per-node=${{SLURM_GPUS_PER_NODE}} \\\n")
            fh.writelines(f"\t\t singularity exec \\\n")
            if 'gpus-per-node' in slurm_args and slurm_args['gpus-per-node'] > 0:
                fh.writelines(f"\t\t --nv $CONTAINER \\\n")
                fh.writelines(f"\t\t {job_file} $N ${{SLURM_GPUS_PER_NODE}} \n")
            else:
                fh.writelines(f"\t\t $CONTAINER \\\n")
                fh.writelines(f"\t\t {job_file} $N \n")
            # NOTE: $N and ${{SLURM_GPUS_PER_NODE}} will be ignored if use_torchrun is false
            
    else:
        print(f'without containers')
        job_file = os.path.join(slurm_args['job_dir'], 
                                f"{slurm_args['job_name']}.sh")
        with open(job_file, 'w') as fj:
            #print(f'PROJECT_ROOT: {PROJECT_ROOT}')
            venv_path = os.path.join(PROJECT_ROOT, slurm_args['venv_path'])
            fj.writelines(f"#!/bin/bash\n")
            #fj.writelines(f"source {venv_path}/bin/activate\n")
            if 'variables' in script_args:
                for variable in script_args['variables']:
                    fj.writelines(f"{variable}={script_args['variables'][variable]}\n")
            print(f'use_pdb: {slurm_args["use_pdb"]}')
            if 'use_pdb' in slurm_args and slurm_args['use_pdb']:
                fj.writelines(f"python3 -m pdb {script_path} \\\n")
            else:
                fj.writelines(f"python3 {script_path} \\\n")
            for arg, value in script_args['args'].items():
                fj.writelines(f"\t\t {arg}={value}\\\n") # adapted to hydra
        if fh is not None:
            fh.writelines(f"chmod +x {job_file}\n")
            fh.writelines(f"N={slurm_args['nodes']};\n")
            if 'gpus-per-node' in slurm_args and slurm_args['gpus-per-node'] > 0:
                fh.writelines(f"./{job_file} $N ${{SLURM_GPUS_PER_NODE}} \n")
            else:
                fh.writelines(f"./{job_file} $N \n")
    
    return job_file
    
def add_general_slurm_job_setup(fh, slurm_args):
    fh.writelines("#!/bin/bash\n")
    fh.writelines(f"#SBATCH --job-name={slurm_args['job_name']}_%a.job\n") # add time stamp?
    fh.writelines(f"#SBATCH --account={slurm_args['project']}\n")
    fh.writelines(f"#SBATCH --partition={slurm_args['partition']}\n")
    fh.writelines(f"#SBATCH --output={slurm_args['output_dir']}/{slurm_args['job_name']}_%a.out\n")
    fh.writelines(f"#SBATCH --error={slurm_args['output_dir']}/{slurm_args['job_name']}_%a.err\n")
    fh.writelines(f"#SBATCH --time={slurm_args['time']}\n")
    fh.writelines(f"#SBATCH --array={slurm_args['start_array_job']}-{slurm_args['end_array_job']}\n")
    if 'dependency' in slurm_args:
        fh.writelines(f"#SBATCH --dependency=afterok:{slurm_args['dependency']}\n")

def create_and_submit_batch_job(slurm_args, 
                                script_args,
                                interactive=False,
                                with_python=True):
    if interactive:
        # TODO: for now we don't add the bindings necessary = assumed to be run inside a container in an interactive session
        # might want to add some safeguards here
        script_args['args']['classifier_guidance.dataset.num_workers'] = 0
        script_file = add_script_commands(script_args, slurm_args, fh=None, with_python=with_python)
        print(f"Running script: {script_file}")
        result = subprocess.Popen(["bash", "-c", f"source {script_file} 1 1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        print(stdout.decode())
        print(stderr.decode())
    else:
        print(f"Creating job file for {slurm_args['job_name']} in {slurm_args['job_dir']}")
        os.makedirs(slurm_args['job_dir'], 
                exist_ok=True)
        job_file = os.path.join(slurm_args['job_dir'], 
                                f"{slurm_args['job_name']}.job")
        # TODO: Could load the yaml file in question the experiment name and log with that locally to outputs/
        with open(job_file, 'w') as fh:
            add_general_slurm_job_setup(fh, slurm_args)
            add_platform_specific_slurm_commands(fh, slurm_args)
            command = add_script_commands(script_args, slurm_args, fh=fh, with_python=with_python)

        result = subprocess.Popen(["/usr/bin/sbatch", job_file], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        if 'job' not in stdout.decode("utf-8"):
            print(stderr)
        else:
            job_id = stdout.decode("utf-8").strip().split('job ')[1]
            job_ids_path = os.path.join(slurm_args['job_dir'], 
                                        slurm_args['job_ids_file'])
            with open(job_ids_path, 'a') as f:
                f.write(f"{slurm_args['job_name']}.job: {job_id}\n")
            print(f"=== {slurm_args['job_name']}. Slurm ID ={job_id}.")

def run_data_processing_pipeline(slurm_args, pipeline_args):
    # 1. get single reactions from uspto
    script_args = {
        'script_dir': 'data_processing',
        'script_name': '1_get_single_reactions_from_uspto.py',
        'args': {
            'uspto_data_dir': 'uspto_full_subset',
        }
    }
    get_single_reactions_job_id = create_and_submit_batch_job(slurm_args, script_args)
    # 2. get valid and unique reactions
    slurm_args['dependency'] = get_single_reactions_job_id
    script_args = {
        'script_dir': 'data_processing',
        'script_name': '20_get_valid_and_unique_reactions.py',
        'args': {
            'data_dir': pipeline_args['data_dir'],
            'input_file': 'stage_single_reactions.csv',
        }
    }
    get_valid_and_unique_reactions_job_id = create_and_submit_batch_job(slurm_args, script_args)
    if pipeline_args['fix_atom_mapping_before_remap']:
        # 3. get valid and unique reactions with atom mapping
        slurm_args['dependency'] = get_valid_and_unique_reactions_job_id
        script_args = {
            'script_dir': 'data_processing',
            'script_name': '21_fix_and_validate_atom_mapping.py',
            'args': {
                'data_dir': pipeline_args['data_dir'],
                'input_file': 'stage_origmap_valid_unique.csv',
            }
        }
        fix_mapping_job_id = create_and_submit_batch_job(slurm_args, script_args)
    else:
        fix_mapping_job_id = None
  
    if pipeline_args['remap_with_rxnmapper']:
        # 4. optional: get new atom-mapping with rxnmapper
        slurm_args['dependency'] = get_valid_and_unique_reactions_job_id
        script_args = {
            'script_dir': 'data_processing',
            'script_name': '21_get_new_atom_mapping_with_rxnmapper.py',
            'args': {
                'data_dir': pipeline_args['data_dir'],
                'input_file': 'stage_cano_valid_unique.csv',
            }
        }
        get_map_rxnmapper_job_id = create_and_submit_batch_job(slurm_args, script_args)
        # 5. fix and validate new mapping
        slurm_args['dependency'] = get_map_rxnmapper_job_id
        script_args = {
            'script_dir': 'data_processing',
            'script_name': '21_fix_and_validate_atom_mapping.py',
            'args': {
                'data_dir': pipeline_args['data_dir'],
                'input_file': 'stage_origmap_valid_unique.csv',
            }
        }
        fix_rxnmapper_mapping_job_id = create_and_submit_batch_job(slurm_args, script_args)
    else:
        fix_rxnmapper_mapping_job_id = None
        
    if pipeline_args['extract_templates']:
        if fix_rxnmapper_mapping_job_id is None and fix_mapping_job_id is None:
            print(f'Skipping template extraction as no atom-mapping was validated.')
        else:
            # 6. get valid and unique reactions with atom mapping and reagents in reactants and reactants are canonicalized
            slurm_args['dependency'] = fix_rxnmapper_mapping_job_id
            script_args = {
                'script_dir': 'data_processing',
                'script_name': '22_get_valid_and_unique_reactions_with_atom_mapping_and_reagents_in_reactants.py',
                'args': {
                    'data_dir': pipeline_args['data_dir'],
                    'input_file': 'stage_map_fixed.csv',
                }
            }
            extract_templates_job_id = create_and_submit_batch_job(slurm_args, script_args)
            
    slurm_args['dependency'] = fix_rxnmapper_mapping_job_id
    script_args = {
        'script_dir': 'data_processing',
        'script_name': '22_transform_and_filter_reactions.py',
        'args': {
            'data_dir': pipeline_args['data_dir'],
            'input_file': 'stage_rxnmapper_map_fixed.csv',
        }
    }
    create_and_submit_batch_job(slurm_args, script_args)
    