import os
import re
import logging
import random
import pathlib
import shutil   
import copy
import time
import pickle
from rdkit import Chem
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import numpy as np
import torch
import wandb
from os import listdir
from os.path import isfile, join
from datetime import datetime
import torch.nn as nn

from diffalign_old.neuralnet.transformer_model_with_pretraining import GraphTransformerWithPretraining, \
    AtomEncoder
from diffalign_old.neuralnet.transformer_model_with_y import GraphTransformerWithYAtomMapPosEmbInefficient

from diffalign_old.utils import setup, io_utils, graph, mol, math_utils

log = logging.getLogger(__name__)

def sample_wandb_log(cfg, output_folder):
    '''
    Sample from W&B.
    '''
    print(f'====== logging samples to W&B ======\n')
    cfg.general.wandb.project = 'multidiff'
    cfg.general.wandb.entity = 'najwalb'
    sampling_steps = cfg.general.wandb.eval_sampling_steps[0]
    epochs = cfg.general.wandb.checkpoint_epochs[0]
    ts = int(round(datetime.now().timestamp()))
    max_dataset_size = cfg.dataset.dataset_size.test if cfg.diffusion.edge_conditional_set=='test' \
                                                     else cfg.dataset.dataset_size.val \
                                                     if cfg.diffusion.edge_conditional_set=='val' \
                                                     else cfg.dataset.dataset_size.train
    #parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    samples_folder = os.path.join(output_folder, f'samples_from_{cfg.dataset.datadir.split("/")[-1]}')
    assert os.path.exists(samples_folder), f'Samples folder {samples_folder} does not exist!'
    run_name = f"sample_data{cfg.dataset.datadir.split('/')[-1]}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}_{ts}"
    
    with wandb.init(name=run_name,
                    project=cfg.general.wandb.project,
                    entity=cfg.general.wandb.entity,
                    job_type='sampling') as run:
        # merge all SMILES-encoded output files belonging to this epoch
        regex = r"samples_steps\d+_cond\d+_sampercond\d+_s\d+\.txt"
        all_output_files_smiles = [os.path.join(samples_folder, f) for f in listdir(samples_folder) if isfile(join(samples_folder, f)) if re.match(regex, f)] #if 'samples' in f and re.search(f'epoch{e}', f) and re.search(f'.txt', f)]
        # cond = min(cfg.test.total_cond_eval, max_dataset_size)
        cond = min(len(all_output_files_smiles)*int(cfg.test.n_conditions), max_dataset_size)
        output_file_smiles = f'samples_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}.txt'
        io_utils.merge_smiles_sample_output_files(files_to_merge=all_output_files_smiles, merged_output_file_name=os.path.join(samples_folder, output_file_smiles))
        
        # merge all PyG-encoded output files belonging to this epoch
        regex = r"samples_steps\d+_cond\d+_sampercond\d+_s\d+\.gz"
        all_output_files_pyg = [os.path.join(samples_folder, f) for f in listdir(samples_folder) if isfile(join(samples_folder, f)) if re.match(regex, f)] #if 'samples' in f and re.search(f'epoch{e}', f) and re.search(f'.pickle', f)]
        # cond = min(len(all_output_files_smiles)*int(cfg.test.n_conditions), max_dataset_size)
        output_file_pyg = f'samples_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}.gz'
        io_utils.merge_pyg_sample_output_files(files_to_merge=all_output_files_pyg, merged_output_file_name=os.path.join(samples_folder, output_file_pyg))

        # get name of the artifact corresponding to the model weights to be added as input to the sampling run
        # artifact_name_in_wandb = f"single_step_model_weights:{cfg.general.wandb.run_id}_epoch{epochs}"
        # print(f'artifact_name_in_wandb {artifact_name_in_wandb}')
        # run.use_artifact(artifact_name_in_wandb)
        # define a whole artifact per epoch. Artifact versions would correspond to the number of samples in each file (or other variables)
        artifact = wandb.Artifact(f'samples_{cfg.general.wandb.run_id}_e{epochs}', type='single_step_samples')
        assert os.path.exists(os.path.join(samples_folder, output_file_smiles)), f'Could not find file {output_file_smiles}.'
        assert os.path.exists(os.path.join(samples_folder, output_file_pyg)), f'Could not find file {output_file_pyg}.'
        artifact.add_file(os.path.join(samples_folder, output_file_smiles), name=output_file_smiles)
        artifact.add_file(os.path.join(samples_folder, output_file_pyg), name=output_file_pyg)
        run.log_artifact(artifact, aliases=[f'{output_file_smiles.split("/")[-1].split(".txt")[0]}'])

def eval_wandb_log(cfg, output_folder):
    '''
    Evaluate from W&B.
    '''
    cfg.general.wandb.project = 'multidiff'
    cfg.general.wandb.entity = 'najwalb'
    sampling_steps = cfg.general.wandb.eval_sampling_steps[0]
    e = cfg.general.wandb.checkpoint_epochs[0]
    samples_folder = os.path.join(output_folder, f'samples_from_{cfg.dataset.datadir.split("/")[-1]}')
    assert os.path.exists(samples_folder), f'Samples folder {samples_folder} does not exist!'
    run_name = f"eval_data{cfg.dataset.datadir.split('/')[-1]}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}"
    with wandb.init(name=run_name,
                    project=cfg.general.wandb.project,
                    entity=cfg.general.wandb.entity,
                    job_type='eval') as run:
        
        regex = r"eval_steps" + str(sampling_steps) + r"_s\d+_resorted_0.9.txt" # NOTE: WE ASSUME THAT THE DEFAULT IS ALWAYS 0.9, only change this in this script
        files_to_merge = [os.path.join(samples_folder, f) for f in listdir(samples_folder) \
                                                          if isfile(join(samples_folder, f)) \
                                                          if re.match(regex, f)]
        log.info(f"Files to merge for sampling_steps {sampling_steps}: {files_to_merge}")
        merged_output_file_name = f'eval_steps{sampling_steps}_resorted_{cfg.test.sort_lambda_value}_'+\
                                f'cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_'+\
                                f'{cfg.diffusion.edge_conditional_set}.txt'
        
        scores = None
        # merge all precalculated score dicts belonging to this epoch
        all_score_files = [os.path.join(samples_folder, f) for f in listdir(samples_folder) \
                             if isfile(join(samples_folder, f)) \
                             if 'scores' in f]
        
        print(f'all_score_files {len(all_score_files)}\n')
        print(f'all_score_files {all_score_files}\n')
        scores = io_utils.merge_scores(file_scores_to_merge=all_score_files)
        io_utils.merge_smiles_sample_output_files(files_to_merge, os.path.join(samples_folder, merged_output_file_name))
        artifact = wandb.Artifact(f'eval_{cfg.general.wandb.run_id}_e{e}', type='eval')
        artifact.add_file(os.path.join(samples_folder, merged_output_file_name), name=merged_output_file_name)
        run.log_artifact(artifact, aliases=[f'{merged_output_file_name.split("/")[-1].split(".txt")[0]}'])

        print(f'scores {scores}\n')
        dict_to_save = {k:v for k, v in scores.items() if k!='rxn_plots'}
        dict_to_save['epoch'] = e
        dict_to_save['sampling_steps'] = sampling_steps
        topks = [scores['top-1'], scores['top-3'], scores['top-5'], scores['top-10']]
        topks_weighted = [scores[f'top-1_weighted_{cfg.test.sort_lambda_value}'],\
                          scores[f'top-3_weighted_{cfg.test.sort_lambda_value}'],\
                          scores[f'top-5_weighted_{cfg.test.sort_lambda_value}'],\
                          scores[f'top-10_weighted_{cfg.test.sort_lambda_value}']]
        dict_to_save['mrr'] = math_utils.estimate_mrr_discrete(math_utils.turn_topk_list_to_dict(topks))
        dict_to_save['mrr_weighted'] = math_utils.estimate_mrr_discrete(math_utils.turn_topk_list_to_dict(topks_weighted))
        log.info(dict_to_save)
        run.log({'sample_eval/': dict_to_save})

def check_reaction_molecules_are_valid(reaction_str):
    '''
    Check that molecules in reaction string are valid.
    '''
    reactants = reaction_str.split('>>')[0]
    products = reaction_str.split('>>')[1]
    
    return Chem.MolFromSmiles(reactants) and Chem.MolFromSmiles(products)

def check_reaction_string_has_atom_mapping(reaction_str):
    '''
    Check that reaction string has atom-mapping.
    '''
    reactants = reaction_str.split('>>')[0]
    products = reaction_str.split('>>')[1]
    
    # use regex to check if patterns like [C:1] exist
    return bool(re.search(r'\[[A-Za-z]:\d+\]', reactants)) and bool(re.search(r'\[[A-Za-z]:\d+\]', products))

def add_atom_mapping(smiles):
    '''
    Add atom mapping to a molecule.
    
    input: smiles string
    output: smiles string with atom mapping
    '''
    # Create molecule from SMILES
    m = Chem.MolFromSmiles(smiles)
    # Add atom mapping numbers starting from 1
    for idx, atom in enumerate(m.GetAtoms(), start=1):
        atom.SetAtomMapNum(idx)
    # Convert back to SMILES with atom mapping
    mapped_smiles = Chem.MolToSmiles(m, canonical=False)
    return mapped_smiles

def run_diffalign_for_one_product(product_smiles, experiment_yaml_file,
                                  experiment_folder='../../RetroDiffuser/configs/experiment/', 
                                  output_folder='samples/'):
    '''
    Run DiffAlign for one product.
    
    NOTE: this is a hack for now to be able to use the diffalign codebase as it is
    ultimately diffalign should be rewritten to allow for more flexibility in the input format 
    1. we add fake precursors to the product: e.g. the identity reaction 
    (i.e. we copy the product as is to the precursors' side)
        1.1. add atom-mapping to the product
        2. run diffalign
    '''
    product_smiles_mapped = add_atom_mapping(product_smiles)
    rct_str = f'{product_smiles_mapped}>>{product_smiles_mapped}'

    return run_diffalign_for_one_reaction(rct_str, experiment_yaml_file, experiment_folder, output_folder)

def run_diffalign_for_one_reaction(reaction_str, experiment_yaml_file,
                                   experiment_folder='../../RetroDiffuser/configs/experiment/',
                                   output_folder='samples/'):
    '''
    Run DiffAlign for one reaction.
    
    reaction_str: atom-mapped reaction string, 
                e.g. '[C:1][C:2][O:3]+[H:4][H:5]>>[C:6][O:7][O:8]+[C:9][H:10][H:11]'
    experiment_yaml_file: path to the experiment yaml file, 
                e.g. 'RetroDiffuser/configs/experiment/severi_default_pe_skip.yaml'
    '''
    # check that reaction has atom-mapping
    #assert check_reaction_molecules_are_valid(reaction_str),
    # f'Reaction string {reaction_str} has some invalid molecules!'
    #assert check_reaction_string_has_atom_mapping(reaction_str),
    # f'Reaction string {reaction_str} does not have atom-mapping!'
    # check that experiment file exists
    experiment_file_path = os.path.join(experiment_folder, experiment_yaml_file)
    print(f'Using experiment file {experiment_file_path}...')
    assert os.path.exists(experiment_file_path),\
            f'Experiment file {experiment_file_path} does not exist!'
    #log.info(f'Using experiment file {experiment_file_path}...')
    
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
    print(f'parent_path {parent_path}')
    raw_dir = f'{parent_path}/data/one_reaction/raw'
    processed_dir = f'{parent_path}/data/one_reaction/processed'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    log.info(f'Saving reaction string to {raw_dir}...')
    print(f'Saving reaction string to {raw_dir}...')
    open(f'{raw_dir}/train.csv', 'w').write(reaction_str)
    open(f'{raw_dir}/test.csv', 'w').write(reaction_str)
    open(f'{raw_dir}/val.csv', 'w').write(reaction_str)
    # Reset Hydra if it was already initialized
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    # Get the directory containing the yaml file
    GlobalHydra.instance().clear()
    # Initialize and load the config
    print(f'before the hydra config')
    # NOTE overrides/defaults does not work here,
    # the yaml file we use needs to contain all the parameters
    with initialize(version_base=None, job_name="test_app",
                    config_path='../configs/experiment'):
        cfg = compose(config_name=f'{experiment_yaml_file}')
        print(f'testing config: {cfg.dataset.dataset_nb}')
        cfg.general.wandb.checkpoint_epochs = [620]
        cfg.general.wandb.run_id = '7ckmnkvc'
        cfg.dataset.datadir = 'data/one_reaction'
        cfg.diffusion.edge_conditional_set = 'test'
        cfg.dataset.shuffle = False
        cfg.dataset.dataset_nb= ''
        cfg.dataset.datadist_dir = 'data/one_reaction'
        cfg.test.num_samples_per_condition_subrepetitions = [100,100]
        cfg.test.num_samples_per_condition_subrepetition_ranges = [10]
        cfg.test.reassign_atom_map_nums = False
        cfg.test.condition_first = 0
        cfg.test.total_cond_eval = 22000
        cfg.test.batch_size = 1
        cfg.test.elbo_samples = 1
        cfg.test.n_conditions = 1 # condition = reaction = product
        cfg.test.n_samples_per_condition = 1 # number of samples/predictions/reactants for each product/reaction
        cfg.diffusion.diffusion_steps = 10 # the higher the better and the slower (training)
        cfg.diffusion.diffusion_steps_eval = 1 # the higher the better and the slower (evaluation)
        cfg.test.total_cond_eval = 1 # same as n_conditions, normally used when evaluation is parallelized
        sample_and_save_from_diffalign(cfg, output_folder)
        sample_wandb_log(cfg, output_folder)
        # override the dataset info
        eval_from_diffalign(cfg, output_folder)
        eval_wandb_log(cfg, output_folder)

def run_diffalign_for_reactions_in_file(reaction_file_path,
                                        experiment_yaml_file,
                                        wandb_mode='offline',
                                        subset='test',
                                        experiment_folder='../../RetroDiffuser/configs/experiment/',
                                        output_folder='samples/'):
    '''
    Run DiffAlign for reactions in a file.
    
    reaction_file_path: path to the file containing the reaction strings, 
                e.g. 'RetroDiffuser/data/one_reaction/raw/train.csv'
    experiment_yaml_file: path to the experiment yaml file, 
                e.g. 'RetroDiffuser/configs/experiment/severi_default_pe_skip.yaml'
    '''
    experiment_file_path = os.path.join(experiment_folder, experiment_yaml_file)
    log.info(f'Using experiment file {experiment_file_path}...')
    assert os.path.exists(experiment_file_path),\
            f'Experiment file {experiment_file_path} does not exist!'
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    data_dir = os.path.join(parent_path, 'data', reaction_file_path, 'raw', f'{subset}.csv')
    assert os.path.exists(data_dir),\
            f'Data directory {data_dir} does not exist!'
    log.info(f'Using data directory {data_dir}...')

    # Reset Hydra if it was already initialized
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    # Get the directory containing the yaml file
    GlobalHydra.instance().clear()
    # Initialize and load the config
    log.info('before the hydra config')
    # NOTE overrides/defaults does not work here,
    # the yaml file we use needs to contain all the parameters
    with initialize(version_base=None, job_name="test_app",
                    config_path='../../configs/experiment'):
        cfg = compose(config_name=f'{experiment_yaml_file}')
        cfg.general.wandb.checkpoint_epochs = [620]
        cfg.general.wandb.run_id = '7ckmnkvc'
        cfg.dataset.datadir = f'data/{reaction_file_path}'
        cfg.dataset.dataset_nb= ''
        cfg.dataset.num_workers = 0
        # cfg.dataset.num_processes = 0
        cfg.dataset.datadist_dir = f'data/{reaction_file_path}'
        cfg.test.batch_size = 1
        cfg.test.elbo_samples = 1
        cfg.test.n_conditions = 1 # condition = reaction = product
        cfg.test.n_samples_per_condition = 1# number of samples/predictions/reactants for each product/reaction
        cfg.diffusion.diffusion_steps = 10 # the higher the better and the slower (training)
        cfg.diffusion.diffusion_steps_eval = 1 # the higher the better and the slower (evaluation)
        cfg.general.wandb.eval_sampling_steps = [cfg.diffusion.diffusion_steps_eval]
        cfg.test.total_cond_eval = 1 # same as n_conditions, normally used when evaluation is parallelized
        
        sample_and_save_from_diffalign(cfg, output_folder)
        if wandb_mode == 'online': sample_wandb_log(cfg, output_folder)
        eval_from_diffalign(cfg, output_folder)
        if wandb_mode == 'online': eval_wandb_log(cfg, output_folder)
        
def sample_and_eval_pretrained(config_file,
                                wandb_mode='offline',
                                subset='test',
                                config_folder='../../RetroDiffuser/configs/experiment/',
                                output_folder='samples/'):
    '''
    Run DiffAlign for reactions in a file.
    
    reaction_file_path: path to the file containing the reaction strings, 
                e.g. 'RetroDiffuser/data/one_reaction/raw/train.csv'
    experiment_yaml_file: path to the experiment yaml file, 
                e.g. 'RetroDiffuser/configs/experiment/severi_default_pe_skip.yaml'
    '''
    # Reset Hydra if it was already initialized
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    # Get the directory containing the yaml file
    GlobalHydra.instance().clear()
    # Initialize and load the config
    log.info('before the hydra config')
    # NOTE overrides/defaults does not work here,
    # the yaml file we use needs to contain all the parameters
    with initialize(version_base=None, job_name="test_app",
                    config_path=config_folder):
        cfg = compose(config_name=f'{config_file}')
        cfg.dataset.num_workers = 0
        # cfg.dataset.num_processes = 0
        cfg.general.wandb.run_id = '9pwkh9rr'
        cfg.general.wandb.project = 'multidiff'
        cfg.general.wandb.checkpoint_epochs = [100]
        cfg.diffusion.edge_conditional_set = subset
        cfg.test.batch_size = 1
        cfg.test.elbo_samples = 1
        cfg.test.n_conditions = 1 # condition = reaction = product
        cfg.test.n_samples_per_condition = 1# number of samples/predictions/reactants for each product/reaction
        cfg.diffusion.diffusion_steps = 1 # the higher the better and the slower (training)
        cfg.diffusion.diffusion_steps_eval = 1 # the higher the better and the slower (evaluation)
        cfg.general.wandb.eval_sampling_steps = [cfg.diffusion.diffusion_steps_eval]
        cfg.test.total_cond_eval = 1 # same as n_conditions, normally used when evaluation is parallelized
        
        sample_pretrained_(cfg, output_folder)
        if wandb_mode == 'online': sample_wandb_log(cfg, output_folder)
        eval_pretrained_(cfg, output_folder)
        if wandb_mode == 'online': eval_wandb_log(cfg, output_folder)
        
def sample_pretrained_(cfg, output_folder):
    ''' 
        Sample from pretrained model.
    '''
    print(f'======= sampling =======\n')
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    print(f'parent_path {parent_path}')
    
    # NOTE: this is used to parallelize the sampling when multiple GPUs are available
    try:
        from mpi4py import MPI
    except ImportError: # mpi4py is not installed, for local experimentation
        MPI = None
        log.warning("mpi4py not found. MPI will not be used.")
        
    if MPI:
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        mpi_rank = comm.Get_rank() # this will be 0
    else:
        mpi_size = 1
        mpi_rank = 0
        
    start_index_for_current_gpu_device = cfg.test.condition_index*mpi_size + mpi_rank
    log.info(f'cfg.test.condition_first = {cfg.test.condition_first}, slurm array index = {cfg.test.condition_index}, start_index_for_current_gpu_device = {start_index_for_current_gpu_device}\n')
        
    # Extract only the command-line overrides
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    log.info(f"Random seed: {cfg.train.seed}")
    log.info(f"Shuffling on: {cfg.dataset.shuffle}")
    log.info(f"cfg.general.wandb.initialization_run_id: {cfg.general.wandb.initialization_run_id}")

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device_count: {device_count}, device: {device}\n')
    # condition_first: the first condition to be sampled overall
    # condition_index: defines the range of conditions to be sampled in this particular 
    # run (across multiple parallel ones)
    # So overall, we sample ranges [condition_first, condition_first+n_conditions], 
    # [condition_first+n_conditions, condition_first+2*n_conditions], etc.
    condition_start_for_job = int(cfg.test.condition_first) + int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    if condition_start_for_job is not None: # take only a slice of the 'true' edge conditional set
        log.info(f"Condition start: {int(cfg.test.condition_first)}+{int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)} = {condition_start_for_job}")
        data_slices = {'train': None, 'val': None, 'test': None}
        data_slices[cfg.diffusion.edge_conditional_set] = [int(condition_start_for_job), int(condition_start_for_job)+int(cfg.test.n_conditions)]

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], 
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, slices=data_slices)
    
        # create the denoiser and load the pretraining weights
    # make copies of the input and output dims for the pretrained model
    pretrained_input_dims = {'X': 37, 'y': 13, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
    pretrained_output_dims = {'X': 29, 'y': 1, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
    pretrained_model = GraphTransformerWithYAtomMapPosEmbInefficient(n_layers=cfg.neuralnet.n_layers,
                                                                     input_dims=pretrained_input_dims,
                                                                     hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                                                     hidden_dims=cfg.neuralnet.hidden_dims,
                                                                     output_dims=pretrained_output_dims, 
                                                                     act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                                                     pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                                                     improved=cfg.neuralnet.improved, cfg=cfg, 
                                                                     dropout=cfg.neuralnet.dropout,
                                                                     p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                                                     p_to_r_init=cfg.neuralnet.p_to_r_init, 
                                                                     alignment_type=cfg.neuralnet.alignment_type,
                                                                     input_alignment=cfg.neuralnet.input_alignment)
    # TODO: make this dynamic
    atom_encoder_input_dims = 85 # the size of the atom labels + PEs
    # TODO: decide what is an appropriate encoder here
    new_encoder = AtomEncoder(input_dims=atom_encoder_input_dims,
                              hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims['X'],
                              out_dim=pretrained_input_dims['X'])
    # TODO: could also experiment with output layer sizes
    denoiser_with_pretraining = GraphTransformerWithPretraining(pretrained_model, 
                                                                new_encoder,
                                                                pretrained_model_out_dim_X=pretrained_output_dims['X'],
                                                                output_hidden_dim_X=32,
                                                                output_dim_X=dataset_infos.output_dims['X'])
    
    log.info("Getting the model and train objects...")
    model, optimizer, scheduler, scaler, _ = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'],
                                                                                         model_kwargs={'dataset_infos': dataset_infos,
                                                                                                       'denoiser': denoiser_with_pretraining,
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': device_count>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'),
                                                                                         load_weights_bool=False, device=device, device_count=device_count)

    # 4. load the weights to the model
    savedir = os.path.join(parent_path, 'checkpoints')
    epoch_num = 100
    weights_path = os.path.join(savedir, f'pretrained_epoch{epoch_num}.pt')
    # For your model
    state_dict = torch.load(weights_path, map_location=device)['model_state_dict']
    #state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # 5. sample n_conditions and n_samples_per_condition
    if cfg.diffusion.edge_conditional_set=='test':
        dataloader = datamodule.test_dataloader()
    elif cfg.diffusion.edge_conditional_set=='val':
        dataloader = datamodule.val_dataloader()
    elif cfg.diffusion.edge_conditional_set=='train':
        dataloader = datamodule.train_dataloader()
    else:
        raise ValueError(f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set}'+\
                         'is not a valid value.\n')
    t0 = time.time()
    sampling_steps = cfg.diffusion.diffusion_steps_eval
    start_index_for_current_gpu_device = cfg.test.condition_index*mpi_size + mpi_rank
    condition_start_for_job = int(cfg.test.condition_first) + int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    print(f'About to sample n_conditions={cfg.test.n_conditions}\n')
    output_file_smiles = f'samples_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.txt'
    output_file_smiles_path = os.path.join(output_folder, f'samples_from_{cfg.dataset.datadir.split("/")[-1]}')
    print(f'output_file_smiles_path {output_file_smiles_path}\n')
    os.makedirs(output_file_smiles_path, exist_ok=True)
    output_file_smiles_path = os.path.join(output_file_smiles_path, output_file_smiles)
    output_file_pyg = f'samples_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.gz'
    output_file_pyg_path = os.path.join(output_folder, f'samples_from_{cfg.dataset.datadir.split("/")[-1]}')
    os.makedirs(output_file_pyg_path, exist_ok=True)
    output_file_pyg_path = os.path.join(output_file_pyg_path, output_file_pyg)


    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = model.sample_n_conditions(dataloader=dataloader, 
                                                                                                           epoch_num=epoch_num,
                                                                                                           device_to_use=None,  
                                                                                                           inpaint_node_idx=None,
                                                                                                           inpaint_edge_idx=None)
    for i, true_rxn_smiles in enumerate(all_true_rxn_smiles):
        gen_rxn_smiles = all_gen_rxn_smiles[i]
        true_rcts_smiles = [rxn.split('>>')[0].split('.') for rxn in true_rxn_smiles]
        true_prods_smiles = [rxn.split('>>')[1].split('.') for rxn in true_rxn_smiles]
        graph.save_gen_rxn_smiles_to_file(output_file_smiles_path, condition_idx=condition_start_for_job+i,
                                          gen_rxns=gen_rxn_smiles, true_rcts=true_rcts_smiles[0], true_prods=true_prods_smiles[0])
    # Save the sparse format generated graphs to a file (includes atom-mapping information) all_true_rxn_pyg
    graph.save_gen_rxn_pyg_to_file(filename=output_file_pyg_path, gen_rxns_pyg=all_gen_rxn_pyg, true_rxns_pyg=all_true_rxn_pyg)
    log.info(f'===== Total sampling time: {time.time()-t0}\n')
  
def eval_pretrained_(cfg, output_folder):
    '''
    Evaluate one reaction.
    '''
    # NOTE: this is used to parallelize the sampling when multiple GPUs are available
    try:
        from mpi4py import MPI
    except ImportError: # mpi4py is not installed, for local experimentation
        MPI = None
        log.warning("mpi4py not found. MPI will not be used.")
        
    if MPI:
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        mpi_rank = comm.Get_rank() # this will be 0
    else:
        mpi_size = 1
        mpi_rank = 0
        
    start_index_for_current_gpu_device = cfg.test.condition_index*mpi_size + mpi_rank
    log.info('cfg.test.condition_first = %d, slurm array index = %d, start_index_for_current_gpu_device = %d\n', \
                cfg.test.condition_first, cfg.test.condition_index, start_index_for_current_gpu_device)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    epoch = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval
    # just a safeguard to be able to run this code on cpu as well
    num_gpus = torch.cuda.device_count() 

    dataset_infos = setup.get_dataset(cfg=cfg,
                                      dataset_class=setup.task_to_class_and_model[cfg.general.task]\
                                                    ['data_class'],
                                      shuffle=cfg.dataset.shuffle, 
                                      return_datamodule=False, 
                                      recompute_info=False)
    
    # create the denoiser and load the pretraining weights
    # make copies of the input and output dims for the pretrained model
    pretrained_input_dims = {'X': 37, 'y': 13, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
    pretrained_output_dims = {'X': 29, 'y': 1, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
    pretrained_model = GraphTransformerWithYAtomMapPosEmbInefficient(n_layers=cfg.neuralnet.n_layers,
                                                                     input_dims=pretrained_input_dims,
                                                                     hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                                                     hidden_dims=cfg.neuralnet.hidden_dims,
                                                                     output_dims=pretrained_output_dims, 
                                                                     act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                                                     pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                                                     improved=cfg.neuralnet.improved, cfg=cfg, 
                                                                     dropout=cfg.neuralnet.dropout,
                                                                     p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                                                     p_to_r_init=cfg.neuralnet.p_to_r_init, 
                                                                     alignment_type=cfg.neuralnet.alignment_type,
                                                                     input_alignment=cfg.neuralnet.input_alignment)
    # TODO: make this dynamic
    atom_encoder_input_dims = 85 # the size of the atom labels + PEs
    # TODO: decide what is an appropriate encoder here
    new_encoder = AtomEncoder(input_dims=atom_encoder_input_dims,
                              hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims['X'],
                              out_dim=pretrained_input_dims['X'])
    # TODO: could also experiment with output layer sizes
    denoiser_with_pretraining = GraphTransformerWithPretraining(pretrained_model, 
                                                                new_encoder,
                                                                pretrained_model_out_dim_X=pretrained_output_dims['X'],
                                                                output_hidden_dim_X=32,
                                                                output_dim_X=dataset_infos.output_dims['X'])
    
    log.info("Getting the model and train objects...")
    device_count = torch.cuda.device_count()
    model, optimizer, scheduler, scaler, _ = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'],
                                                                                         model_kwargs={'dataset_infos': dataset_infos,
                                                                                                       'denoiser': denoiser_with_pretraining,
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': device_count>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'),
                                                                                         load_weights_bool=False, device=device, device_count=device_count)

    # 4. load the weights to the model
    savedir = os.path.join(parent_path, 'checkpoints')
    epoch_num = 100
    weights_path = os.path.join(savedir, f'pretrained_epoch{epoch_num}.pt')
    # For your model
    state_dict = torch.load(weights_path, map_location=device)['model_state_dict']
    #state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    # Dataset & slice statistics
    assert cfg.diffusion.edge_conditional_set in ['test', 'val', 'train'], f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set} is not a valid value.\n'
    #TODO: Fix this, here the validation set size is hardcoded, which is not good
    max_dataset_size = cfg.dataset.dataset_size.test if cfg.diffusion.edge_conditional_set=='test' \
                                                     else 4951 \
                                                     if cfg.diffusion.edge_conditional_set=='val' \
                                                     else cfg.dataset.dataset_size.train
    total_conditions = min(max_dataset_size, cfg.test.total_cond_eval)
    condition_start_zero_indexed = int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions) # zero-indexed because no condition_first here
    condition_range = [condition_start_zero_indexed, min(int(condition_start_zero_indexed)+int(cfg.test.n_conditions), max_dataset_size)]
    log.info(f'condition_range: {condition_range}\n')
    actual_n_conditions = condition_range[1] - condition_range[0] # handles the case where max_dataset_size < start+n_conditions
   
    # Load the data
    # file_path = samples_from_wandb(cfg.general.wandb.entity, cfg.general.wandb.run_id, cfg.general.wandb.project,
    #                     sampling_steps, epoch, total_conditions, cfg.test.n_samples_per_condition)
    # Assumes that hydra.run.dir is set to the same location as the samples
    condition_start_for_job = int(cfg.test.condition_first) + int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    file_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                             f"samples_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.gz")
    # How to get to experiments/if3aizpe_sample_ts/samples_epoch??_steps??_cond??_samplercond??_s{condition_first thing from sample_array_job}.gz?
    # just need the if3aizpe_sample_ts part.
    # -> or preferably actually just the corresponding .gz file here, don't download everything.
    # ... but then the condition_range stuff will go funny? Or will it?
    # What is the format of file_path here? Just the one .gz file? -> Then we can replace it with another .gz file
    # TODO: Change this such that it uses the correct stuff
    reaction_data = io_utils.get_samples_from_file_pyg(cfg, file_path, condition_range=None) # None means: don't do additional slicing anymore    

    # TODO: Why is the condition_range input here as well? -> It's not, it's fine
    scores, all_elbo_sorted_reactions, all_weighted_prob_sorted_rxns, placeholders_for_print = model.evaluate_from_artifact(reaction_data=reaction_data, 
                                                                                                                            actual_n_conditions=actual_n_conditions, 
                                                                                                                            device=device,
                                                                                                                            condition_range=condition_range,
                                                                                                                            epoch=epoch)
    #for i in range(len(placeholders_for_print)):
    for i, original_data_placeholder in enumerate(placeholders_for_print):
        elbo_sorted_reactions = all_elbo_sorted_reactions[i]
        weighted_prob_sorted_rxns = all_weighted_prob_sorted_rxns[i]
        true_rxns = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=original_data_placeholder, cfg=cfg)
        # true_rcts, true_prods = mol.get_cano_list_smiles(X=original_data_placeholder.X, E=original_data_placeholder.E, atom_types=model.dataset_info.atom_decoder,
        #                                                  bond_types=model.dataset_info.bond_decoder, plot_dummy_nodes=cfg.test.plot_dummy_nodes)
        samples_without_weighted_prob_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                                                          f'eval_steps{sampling_steps}_s{condition_start_for_job}.txt')
        samples_with_weighted_prob_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                                                       f'eval_steps{sampling_steps}_s{condition_start_for_job}_resorted{cfg.test.sort_lambda_value}.txt')
        graph.save_samples_to_file_without_weighted_prob(samples_without_weighted_prob_path, i, elbo_sorted_reactions, true_rxns, overwrite=True)
        graph.save_samples_to_file(samples_with_weighted_prob_path, i, weighted_prob_sorted_rxns, true_rxns, overwrite=True)
    for score in scores:
        for key, value in score.items():
            if isinstance(value, torch.Tensor):
                score[key] = value.detach().cpu().numpy()
    scores_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                               f'scores_epoch{epoch}_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.pickle')
    pickle.dump(scores, open(scores_path, 'wb'))
  
def sample_from_diffalign_(cfg):
    ''' 
    Sample one reaction.
    '''
    print(f'======= sampling =======\n')
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    print(f'parent_path {parent_path}')
    
    # NOTE: this is used to parallelize the sampling when multiple GPUs are available
    try:
        from mpi4py import MPI
    except ImportError: # mpi4py is not installed, for local experimentation
        MPI = None
        log.warning("mpi4py not found. MPI will not be used.")
        
    if MPI:
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        mpi_rank = comm.Get_rank() # this will be 0
    else:
        mpi_size = 1
        mpi_rank = 0
        
    start_index_for_current_gpu_device = cfg.test.condition_index*mpi_size + mpi_rank
    log.info(f'cfg.test.condition_first = {cfg.test.condition_first}, slurm array index = {cfg.test.condition_index}, start_index_for_current_gpu_device = {start_index_for_current_gpu_device}\n')
        
    # Extract only the command-line overrides
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    log.info(f"Random seed: {cfg.train.seed}")
    log.info(f"Shuffling on: {cfg.dataset.shuffle}")
    log.info(f"cfg.general.wandb.initialization_run_id: {cfg.general.wandb.initialization_run_id}")

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device_count: {device_count}, device: {device}\n')
    epoch_num = cfg.general.wandb.checkpoint_epochs[0]
    # condition_first: the first condition to be sampled overall
    # condition_index: defines the range of conditions to be sampled in this particular 
    # run (across multiple parallel ones)
    # So overall, we sample ranges [condition_first, condition_first+n_conditions], 
    # [condition_first+n_conditions, condition_first+2*n_conditions], etc.
    condition_start_for_job = int(cfg.test.condition_first) + int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    if condition_start_for_job is not None: # take only a slice of the 'true' edge conditional set
        log.info(f"Condition start: {int(cfg.test.condition_first)}+{int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)} = {condition_start_for_job}")
        data_slices = {'train': None, 'val': None, 'test': None}
        data_slices[cfg.diffusion.edge_conditional_set] = [int(condition_start_for_job), int(condition_start_for_job)+int(cfg.test.n_conditions)]

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], 
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, slices=data_slices)
    
    log.info("Getting the model and train objects...")
    model, optimizer, scheduler, scaler, _ = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': device_count>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=device_count)

    log.info("2!------------------------------------------------")
    log.info(f": {cfg}")
    log.info(f": {cfg.general}")
    log.info(f": {cfg.general.wandb}")

    # 4. load the weights to the model
    #savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    savedir = os.path.join(parent_path, 'RetroDiffuser', 'checkpoints')
    print(f'savedir_weights {savedir}\n')
    model, optimizer, scheduler, scaler, _ = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=device_count)

    # 5. sample n_conditions and n_samples_per_condition
    if cfg.diffusion.edge_conditional_set=='test':
        dataloader = datamodule.test_dataloader()
    elif cfg.diffusion.edge_conditional_set=='val':
        dataloader = datamodule.val_dataloader()
    elif cfg.diffusion.edge_conditional_set=='train':
        dataloader = datamodule.train_dataloader()
    else:
        raise ValueError(f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set}'+\
                         'is not a valid value.\n')
    t0 = time.time()
    print(f'About to sample n_conditions={cfg.test.n_conditions}\n')

    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = model.sample_n_conditions(dataloader=dataloader, epoch_num=epoch_num,
                                                                                                           device_to_use=None,  inpaint_node_idx=None, 
                                                                                                           inpaint_edge_idx=None)
    log.info(f'===== Total sampling time: {time.time()-t0}\n')
    
    return all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg

def get_scored_samples_for_multistep(cfg):
    ''' 
    Sample one reaction.
    '''
    print(f'======= sampling =======\n')
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    print(f'parent_path {parent_path}')
    
    # NOTE: this is used to parallelize the sampling when multiple GPUs are available
    try:
        from mpi4py import MPI
    except ImportError: # mpi4py is not installed, for local experimentation
        MPI = None
        log.warning("mpi4py not found. MPI will not be used.")
        
    if MPI:
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        mpi_rank = comm.Get_rank() # this will be 0
    else:
        mpi_size = 1
        mpi_rank = 0
        
    start_index_for_current_gpu_device = cfg.test.condition_index*mpi_size + mpi_rank
    log.info(f'cfg.test.condition_first = {cfg.test.condition_first}, slurm array index = {cfg.test.condition_index}, start_index_for_current_gpu_device = {start_index_for_current_gpu_device}\n')
        
    # Extract only the command-line overrides
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    log.info(f"Random seed: {cfg.train.seed}")
    log.info(f"Shuffling on: {cfg.dataset.shuffle}")
    log.info(f"cfg.general.wandb.initialization_run_id: {cfg.general.wandb.initialization_run_id}")

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device_count: {device_count}, device: {device}\n')
    epoch_num = cfg.general.wandb.checkpoint_epochs[0]
    # condition_first: the first condition to be sampled overall
    # condition_index: defines the range of conditions to be sampled in this particular 
    # run (across multiple parallel ones)
    # So overall, we sample ranges [condition_first, condition_first+n_conditions], 
    # [condition_first+n_conditions, condition_first+2*n_conditions], etc.
    condition_start_for_job = int(cfg.test.condition_first) + int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    if condition_start_for_job is not None: # take only a slice of the 'true' edge conditional set
        log.info(f"Condition start: {int(cfg.test.condition_first)}+{int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)} = {condition_start_for_job}")
        data_slices = {'train': None, 'val': None, 'test': None}
        data_slices[cfg.diffusion.edge_conditional_set] = [int(condition_start_for_job), int(condition_start_for_job)+int(cfg.test.n_conditions)]

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], 
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, slices=data_slices)
    
        # create the denoiser and load the pretraining weights
    # make copies of the input and output dims for the pretrained model
    pretrained_input_dims = {'X': 37, 'y': 13, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
    pretrained_output_dims = {'X': 29, 'y': 1, 'E': 7, 'atom_charges': 3, 'atom_chiral': 3, 'bond_dirs': 3}
    pretrained_model = GraphTransformerWithYAtomMapPosEmbInefficient(n_layers=cfg.neuralnet.n_layers,
                                                                     input_dims=pretrained_input_dims,
                                                                     hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims,
                                                                     hidden_dims=cfg.neuralnet.hidden_dims,
                                                                     output_dims=pretrained_output_dims, 
                                                                     act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(),
                                                                     pos_emb_permutations=cfg.neuralnet.pos_emb_permutations,
                                                                     improved=cfg.neuralnet.improved, cfg=cfg, 
                                                                     dropout=cfg.neuralnet.dropout,
                                                                     p_to_r_skip_connection=cfg.neuralnet.p_to_r_skip_connection,
                                                                     p_to_r_init=cfg.neuralnet.p_to_r_init, 
                                                                     alignment_type=cfg.neuralnet.alignment_type,
                                                                     input_alignment=cfg.neuralnet.input_alignment)
    # TODO: make this dynamic
    atom_encoder_input_dims = 85 # the size of the atom labels + PEs
    # TODO: decide what is an appropriate encoder here
    new_encoder = AtomEncoder(input_dims=atom_encoder_input_dims,
                              hidden_mlp_dims=cfg.neuralnet.hidden_mlp_dims['X'],
                              out_dim=pretrained_input_dims['X'])
    # TODO: could also experiment with output layer sizes
    denoiser_with_pretraining = GraphTransformerWithPretraining(pretrained_model, 
                                                                new_encoder,
                                                                pretrained_model_out_dim_X=pretrained_output_dims['X'],
                                                                output_hidden_dim_X=32,
                                                                output_dim_X=dataset_infos.output_dims['X'])
    
    log.info("Getting the model and train objects...")
    model, optimizer, scheduler, scaler, _ = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'],
                                                                                         model_kwargs={'dataset_infos': dataset_infos,
                                                                                                       'denoiser': denoiser_with_pretraining,
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': device_count>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'),
                                                                                         load_weights_bool=False, device=device, device_count=device_count)

    log.info("2!------------------------------------------------")
    log.info(f": {cfg}")
    log.info(f": {cfg.general}")
    log.info(f": {cfg.general.wandb}")

    # 4. load the weights to the model
    # #savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    # savedir = os.path.join(parent_path, 'RetroDiffuser', 'checkpoints')
    # print(f'savedir_weights {savedir}\n')
    # model, optimizer, scheduler, scaler, _ = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
    #                                                                                                         scheduler, scaler, device_count=device_count)
    savedir = os.path.join(parent_path, 'checkpoints')
    epoch_num = cfg.general.wandb.checkpoint_epochs[0]
    weights_path = os.path.join(savedir, f'pretrained_epoch{epoch_num}.pt')
    # For your model
    state_dict = torch.load(weights_path, map_location=device)['model_state_dict']
    #state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # 5. sample n_conditions and n_samples_per_condition
    if cfg.diffusion.edge_conditional_set=='test':
        dataloader = datamodule.test_dataloader()
    elif cfg.diffusion.edge_conditional_set=='val':
        dataloader = datamodule.val_dataloader()
    elif cfg.diffusion.edge_conditional_set=='train':
        dataloader = datamodule.train_dataloader()
    else:
        raise ValueError(f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set}'+\
                         'is not a valid value.\n')
    t0 = time.time()
    print(f'About to sample n_conditions={cfg.test.n_conditions}\n')

    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = model.sample_n_conditions(dataloader=dataloader, 
                                                                                                           epoch_num=epoch_num,
                                                                                                           device_to_use=None,  
                                                                                                           inpaint_node_idx=None,
                                                                                                           inpaint_edge_idx=None)
    reaction_data = {"gen": all_gen_rxn_pyg, "true": all_gen_rxn_pyg}
    epoch = cfg.general.wandb.checkpoint_epochs[0]
    condition_start_zero_indexed = int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    max_dataset_size = cfg.dataset.dataset_size.test if cfg.diffusion.edge_conditional_set=='test' \
                                                     else 4951 \
                                                     if cfg.diffusion.edge_conditional_set=='val' \
                                                     else cfg.dataset.dataset_size.train
    condition_range = [condition_start_zero_indexed, min(int(condition_start_zero_indexed)+int(cfg.test.n_conditions), max_dataset_size)]
    log.info(f'condition_range: {condition_range}\n')
    actual_n_conditions = condition_range[1] - condition_range[0] # handles the case where max_dataset_size < start+n_conditions
    scores, all_elbo_sorted_reactions, all_weighted_prob_sorted_rxns, placeholders_for_print = model.evaluate_from_artifact(reaction_data=reaction_data,
                                                                                                                            actual_n_conditions=actual_n_conditions,
                                                                                                                            device=device,
                                                                                                                            condition_range=condition_range,
                                                                                                                            epoch=epoch)
    log.info(f'===== Total sampling time: {time.time()-t0}\n')
    
    return all_weighted_prob_sorted_rxns

def sample_for_multistep(product_smiles,
                         config_file,
                         config_folder,
                         n_samples=1):
    '''
    Get samples for a multistep model.
    '''
    product_smiles_mapped = add_atom_mapping(product_smiles)
    reaction_smiles = f'{product_smiles_mapped}>>{product_smiles_mapped}'
    
    # parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    # config_folder = os.path.join(parent_path, 'configs', 'experiment')
    # experiment_file_path = os.path.join(config_folder, config_file)
    # print(f'Using experiment file {experiment_file_path}...')
    # assert os.path.exists(experiment_file_path),\
    #         f'Experiment file {experiment_file_path} does not exist!'
    #log.info(f'Using experiment file {experiment_file_path}...')
    
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    print(f'parent_path {parent_path}')
    raw_dir = f'{parent_path}/data/one_reaction/raw'
    processed_dir = f'{parent_path}/data/one_reaction/processed'
    # Delete directory and all its contents
    shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    log.info(f'Saving reaction string to {raw_dir}...')
    print(f'Saving reaction string to {raw_dir}...')
    open(f'{raw_dir}/train.csv', 'w').write(reaction_smiles)
    open(f'{raw_dir}/test.csv', 'w').write(reaction_smiles)
    open(f'{raw_dir}/val.csv', 'w').write(reaction_smiles)
    # Reset Hydra if it was already initialized
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    # Get the directory containing the yaml file
    GlobalHydra.instance().clear()
    # Initialize and load the config
    print(f'before the hydra config')
    # NOTE overrides/defaults does not work here,
    # the yaml file we use needs to contain all the parameters
    with initialize(version_base=None, job_name="test_app",
                    config_path='../../configs/experiment'):
        cfg = compose(config_name=f'{config_file}')
        print(f'testing config: {cfg.dataset.dataset_nb}')
        cfg.general.wandb.checkpoint_epochs = [100]
        cfg.general.wandb.run_id = '7ckmnkvc'
        cfg.dataset.datadir = 'data/one_reaction'
        cfg.diffusion.edge_conditional_set = 'test'
        cfg.dataset.shuffle = False
        cfg.dataset.dataset_nb= ''
        cfg.dataset.datadist_dir = 'data/one_reaction'
        cfg.test.num_samples_per_condition_subrepetitions = [100,100]
        cfg.test.num_samples_per_condition_subrepetition_ranges = [10]
        cfg.dataset.num_workers = 0
        cfg.test.reassign_atom_map_nums = False
        cfg.test.condition_first = 0
        cfg.test.total_cond_eval = 22000
        cfg.test.batch_size = 1
        cfg.test.elbo_samples = 1
        cfg.test.n_conditions = 1 # condition = reaction = product
        cfg.test.n_samples_per_condition = n_samples # number of samples/predictions/reactants for each product/reaction
        cfg.diffusion.diffusion_steps = 10 # the higher the better and the slower (training)
        cfg.diffusion.diffusion_steps_eval = 10 # the higher the better and the slower (evaluation)
        cfg.test.total_cond_eval = 1 # same as n_conditions, normally used when evaluation is parallelized

        all_weighted_prob_sorted_rxns = get_scored_samples_for_multistep(cfg)
        return all_weighted_prob_sorted_rxns

def sample_and_save_from_diffalign(cfg, output_folder):
    try:
        from mpi4py import MPI
    except ImportError: # mpi4py is not installed, for local experimentation
        MPI = None
        log.warning("mpi4py not found. MPI will not be used.")
        
    if MPI:
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        mpi_rank = comm.Get_rank() # this will be 0
    else:
        mpi_size = 1
        mpi_rank = 0
    sampling_steps = cfg.diffusion.diffusion_steps_eval
    start_index_for_current_gpu_device = cfg.test.condition_index*mpi_size + mpi_rank
    condition_start_for_job = int(cfg.test.condition_first) + int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    
    output_file_smiles = f'samples_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.txt'
    output_file_smiles_path = os.path.join(output_folder, f'samples_from_{cfg.dataset.datadir.split("/")[-1]}')
    print(f'output_file_smiles_path {output_file_smiles_path}\n')
    os.makedirs(output_file_smiles_path, exist_ok=True)
    output_file_smiles_path = os.path.join(output_file_smiles_path, output_file_smiles)
    output_file_pyg = f'samples_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.gz'
    output_file_pyg_path = os.path.join(output_folder, f'samples_from_{cfg.dataset.datadir.split("/")[-1]}')
    os.makedirs(output_file_pyg_path, exist_ok=True)
    output_file_pyg_path = os.path.join(output_file_pyg_path, output_file_pyg)

    
    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = sample_from_diffalign_(cfg)
    for i, true_rxn_smiles in enumerate(all_true_rxn_smiles):
        gen_rxn_smiles = all_gen_rxn_smiles[i]
        true_rcts_smiles = [rxn.split('>>')[0].split('.') for rxn in true_rxn_smiles]
        true_prods_smiles = [rxn.split('>>')[1].split('.') for rxn in true_rxn_smiles]
        graph.save_gen_rxn_smiles_to_file(output_file_smiles_path, condition_idx=condition_start_for_job+i,
                                          gen_rxns=gen_rxn_smiles, true_rcts=true_rcts_smiles[0], true_prods=true_prods_smiles[0])
    # Save the sparse format generated graphs to a file (includes atom-mapping information) all_true_rxn_pyg
    graph.save_gen_rxn_pyg_to_file(filename=output_file_pyg_path, gen_rxns_pyg=all_gen_rxn_pyg, true_rxns_pyg=all_true_rxn_pyg)
    
def eval_from_diffalign(cfg, output_folder):
    '''
    Evaluate one reaction.
    '''
    # NOTE: this is used to parallelize the sampling when multiple GPUs are available
    try:
        from mpi4py import MPI
    except ImportError: # mpi4py is not installed, for local experimentation
        MPI = None
        log.warning("mpi4py not found. MPI will not be used.")
        
    if MPI:
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        mpi_rank = comm.Get_rank() # this will be 0
    else:
        mpi_size = 1
        mpi_rank = 0
        
    start_index_for_current_gpu_device = cfg.test.condition_index*mpi_size + mpi_rank
    log.info('cfg.test.condition_first = %d, slurm array index = %d, start_index_for_current_gpu_device = %d\n', \
                cfg.test.condition_first, cfg.test.condition_index, start_index_for_current_gpu_device)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    epoch = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval
    # just a safeguard to be able to run this code on cpu as well
    num_gpus = torch.cuda.device_count() 

    dataset_infos = setup.get_dataset(cfg=cfg,
                                      dataset_class=setup.task_to_class_and_model[cfg.general.task]\
                                                    ['data_class'],
                                      shuffle=cfg.dataset.shuffle, 
                                      return_datamodule=False, 
                                      recompute_info=False)
    model, optimizer, scheduler, scaler, _ = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]\
                                                                                                ['model_class'], 
                                                                                model_kwargs={'dataset_infos': dataset_infos, 
                                                                                              'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                              'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                              'use_data_parallel': num_gpus>1},
                                                                                parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                load_weights_bool=False, device=device, device_count=num_gpus)
    # 4. load the weights to the model
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    savedir = os.path.join(parent_path, 'RetroDiffuser', 'checkpoints')
    print(f'savedir_weights {savedir}\n')
    #model, optimizer, scheduler, scaler, _ = setup.load_weights_from_wandb_no_download(cfg, epoch, savedir, model, optimizer, scheduler, scaler, device_count=num_gpus)   

    # Dataset & slice statistics
    assert cfg.diffusion.edge_conditional_set in ['test', 'val', 'train'], f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set} is not a valid value.\n'
    #TODO: Fix this, here the validation set size is hardcoded, which is not good
    max_dataset_size = cfg.dataset.dataset_size.test if cfg.diffusion.edge_conditional_set=='test' \
                                                     else 4951 \
                                                     if cfg.diffusion.edge_conditional_set=='val' \
                                                     else cfg.dataset.dataset_size.train
    total_conditions = min(max_dataset_size, cfg.test.total_cond_eval)
    condition_start_zero_indexed = int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions) # zero-indexed because no condition_first here
    condition_range = [condition_start_zero_indexed, min(int(condition_start_zero_indexed)+int(cfg.test.n_conditions), max_dataset_size)]
    log.info(f'condition_range: {condition_range}\n')
    actual_n_conditions = condition_range[1] - condition_range[0] # handles the case where max_dataset_size < start+n_conditions
   
    # Load the data
    # file_path = samples_from_wandb(cfg.general.wandb.entity, cfg.general.wandb.run_id, cfg.general.wandb.project,
    #                     sampling_steps, epoch, total_conditions, cfg.test.n_samples_per_condition)
    # Assumes that hydra.run.dir is set to the same location as the samples
    condition_start_for_job = int(cfg.test.condition_first) + int(start_index_for_current_gpu_device)*int(cfg.test.n_conditions)
    file_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                             f"samples_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.gz")
    # How to get to experiments/if3aizpe_sample_ts/samples_epoch??_steps??_cond??_samplercond??_s{condition_first thing from sample_array_job}.gz?
    # just need the if3aizpe_sample_ts part.
    # -> or preferably actually just the corresponding .gz file here, don't download everything.
    # ... but then the condition_range stuff will go funny? Or will it?
    # What is the format of file_path here? Just the one .gz file? -> Then we can replace it with another .gz file
    # TODO: Change this such that it uses the correct stuff
    reaction_data = io_utils.get_samples_from_file_pyg(cfg, file_path, condition_range=None) # None means: don't do additional slicing anymore    

    # TODO: Why is the condition_range input here as well? -> It's not, it's fine
    scores, all_elbo_sorted_reactions, all_weighted_prob_sorted_rxns, placeholders_for_print = model.evaluate_from_artifact(reaction_data=reaction_data, 
                                                                                                                            actual_n_conditions=actual_n_conditions, 
                                                                                                                            device=device,
                                                                                                                            condition_range=condition_range,
                                                                                                                            epoch=epoch)
    #for i in range(len(placeholders_for_print)):
    for i, original_data_placeholder in enumerate(placeholders_for_print):
        elbo_sorted_reactions = all_elbo_sorted_reactions[i]
        weighted_prob_sorted_rxns = all_weighted_prob_sorted_rxns[i]
        true_rxns = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=original_data_placeholder, cfg=cfg)
        # true_rcts, true_prods = mol.get_cano_list_smiles(X=original_data_placeholder.X, E=original_data_placeholder.E, atom_types=model.dataset_info.atom_decoder,
        #                                                  bond_types=model.dataset_info.bond_decoder, plot_dummy_nodes=cfg.test.plot_dummy_nodes)
        samples_without_weighted_prob_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                                                          f'eval_steps{sampling_steps}_s{condition_start_for_job}.txt')
        samples_with_weighted_prob_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                                                       f'eval_steps{sampling_steps}_s{condition_start_for_job}_resorted{cfg.test.sort_lambda_value}.txt')
        graph.save_samples_to_file_without_weighted_prob(samples_without_weighted_prob_path, i, elbo_sorted_reactions, true_rxns, overwrite=True)
        graph.save_samples_to_file(samples_with_weighted_prob_path, i, weighted_prob_sorted_rxns, true_rxns, overwrite=True)
    for score in scores:
        for key, value in score.items():
            if isinstance(value, torch.Tensor):
                score[key] = value.detach().cpu().numpy()
    scores_path = os.path.join(output_folder, f"samples_from_{cfg.dataset.datadir.split('/')[-1]}",
                               f'scores_epoch{epoch}_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.pickle')
    pickle.dump(scores, open(scores_path, 'wb'))
