'''
Evaluate the samples saved as wandb artifacts.
'''
# A script that takes a run id, (a list of epochs) that we have good evaluations for, and updates new re-ranked evaluations to wandb. Also the lambda re-ranking value.
# What we need:
# 1. The run id
# 2. The list of epochs for which we have results from wandb automatically (TODO: How to do this?)
# retrieve the correct data from wandb for a given run id and epoch
# get the config file for the run, and create a model based on it (from diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn)
# Then transform the samples.txt data into the elbo_ranked format
# Then get weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_rxns, self.cfg.test.sort_lambda_value)
# true_rcts, true_prods = graph.split_reactions_to_reactants_and_products(true_rxn_smiles)
# topk_weighted = graph.calculate_top_k(self, weighted_prob_sorted_rxns, true_rcts, true_prods)
import os
from diffalign.utils import wandb_utils, io_utils, graph
import wandb
from omegaconf import OmegaConf, DictConfig
import numpy as np
import pickle
import hydra
import random
import torch
import logging
import pathlib
import re
import sys
from diffalign.utils import setup
from diffalign.utils import mol
from diffalign.utils import data_utils
import copy

# A logger for this file
log = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except:
    MPI = None
    log.warning("mpi4py not found. MPI will not be used.")

# A logger for this file
log = logging.getLogger(__name__)
parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def samples_from_wandb(entity, project, run_id, sampling_steps, epoch, total_conditions, n_samples_per_condition):
    # 1. get samples artifact from run_id
    collection_name = f"{run_id}_samples"
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name='samples', project=f"{entity}/{project}").collections()
        if coll.name==collection_name
    ]
    assert len(collections)==1, f'Found {len(collections)} collections with name {collection_name}, expected 1.'
    
    coll = collections[0]
    # aliases = [alias for art in coll.versions() for alias in art.aliases \
    #                  if 'samples' in alias
    #                  and re.findall('steps\d+', alias)[0]==f'steps{sampling_steps}'
    #                  and re.findall('epoch\d+', alias)[0]==f'epoch{epoch}'
    #                  and re.findall('cond\d+', alias)[0]==f'cond{cfg.test.total_cond_eval}'
    #                  and re.findall('sampercond\d+', alias)[0]==f'sampercond{cfg.test.n_samples_per_condition}']
    aliases = [alias for art in coll.versions() for alias in art.aliases \
                    if 'samples' in alias
                    and re.search(f'steps{sampling_steps}', alias)
                    and re.search(f'epoch{epoch}', alias)
                    and re.search(f'cond{total_conditions}', alias)
                    and re.search(f'sampercond{n_samples_per_condition}', alias)]
    if len(aliases) == 0:
        aliases = coll.versions()[0].aliases
        for a in aliases:
            log.info(a)
            log.info(re.search(f'steps{sampling_steps}', a))
            log.info(re.search(f'epoch{epoch}', a))
            log.info(re.search(f'cond{total_conditions}', a))
            log.info(re.search(f'sampercond{n_samples_per_condition}', a))
            # log.info(re.findall('epoch\d+', a)[0]==f'epoch{epoch}')
            # log.info(re.findall('cond\d+', a)[0]==f'cond{cfg.test.total_cond_eval}')
            # log.info(re.findall('sampercond\d+', a)[0]==f'sampercond{cfg.test.n_samples_per_condition}')
        assert False, 'No aliases found'
    versions = [int(art.version.split('v')[-1]) for art in coll.versions()]

    aliases = [a for a,v in sorted(zip(aliases, versions), key=lambda pair: pair[1], reverse=True)]
    #log.info(f'cfg.general.wandb.sample_file_name {cfg.general.wandb.sample_file_name}\n')
    assert len(aliases)>0, f'No aliases found for given specs.'
    log.info(f'ordered aliases {aliases}\n')
    log.info(f'the script will be using the newest alias: {aliases[0]}\n')

    # get samples from wandb
    savedir = os.path.join(parent_path, "experiments", "trained_models", run_id)
    artifact_name = f"{entity}/{project}/{collection_name}:{aliases[0]}"
    samples_art = wandb.Api().artifact(artifact_name)
    samples_art.download(root=savedir)

    # sample_file_name = f'samples_epoch{epoch}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}.txt'
    file_path = os.path.join(savedir, aliases[0]+'.gz')
    return file_path

# 1. The run id, etc.
@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):     
    # Log the current working directory
    # log.info(os.getcwd())
    # log.info(os.listdir())
    orig_cfg = copy.deepcopy(cfg)
    
    print(f'cfg.test.batch_size: {cfg.test.batch_size}\n')
    print(f'cfg.test.elbo_samples: {cfg.test.elbo_samples}\n')
    # MPI related parameters (in case --ntasks>1)
    if MPI is not None:
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size() # if not --ntasks>1, this will be 1
        mpi_rank = comm.Get_rank() # this will be 0
    else:
        mpi_size = 8 # TODO: change back (?)
        mpi_rank = 0
    # mpi_size = 8 # TODO: remove
    # mpi_rank = 0

    # assert cfg.general.wandb.sample_file_name is not None, f'Need to give cfg.general.wandb.sample_file_name in the form epoch#_cond#_sampercond#_# (no extension, last number is the timestamp). Got {cfg.general.wandb.sample_file_name}.'
    # Extract only the command-line overrides
    cli_overrides = setup.capture_cli_overrides()

    if cfg.general.wandb.mode=='online': 
        run, cfg = setup.setup_wandb(cfg, job_type='ranking') # This creates a new wandb run or resumes a run given its id
    
    entity = cfg.general.wandb.entity
    project = cfg.general.wandb.project

    if cfg.general.wandb.load_run_config: 
        run_config = setup.load_wandb_config(orig_cfg)
        cfg = setup.merge_configs(default_cfg=orig_cfg, new_cfg=run_config, cli_overrides=cli_overrides)

    cfg.general.wandb.entity = entity
    cfg.general.wandb.project = project
    
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    
    epoch = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval
    num_gpus = torch.cuda.device_count() # just a safeguard to be able to run this code on cpu as well
    
    total_index = cfg.test.condition_index*mpi_size + mpi_rank
    log.info(f'cfg.test.condition_first & slurm array index & total condition index {cfg.test.condition_first}, {cfg.test.condition_index}, {total_index}\n')
    
    dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                      shuffle=cfg.dataset.shuffle, return_datamodule=False, recompute_info=False)

    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': num_gpus>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=num_gpus)
    
    # 4. load the weights to the model
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb_no_download(cfg, epoch, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=num_gpus)
    
    # Dataset & slice statistics
    assert cfg.diffusion.edge_conditional_set in ['test', 'val', 'train'], f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set} is not a valid value.\n'
    # TODO: Fix this, here the validation set size is hardcoded, which is not good
    max_dataset_size = cfg.dataset.dataset_size.test if cfg.diffusion.edge_conditional_set=='test' else 4951 if cfg.diffusion.edge_conditional_set=='val' else cfg.dataset.dataset_size.train
    total_conditions = min(max_dataset_size, cfg.test.total_cond_eval)
    condition_start_zero_indexed = int(total_index)*int(cfg.test.n_conditions) # zero-indexed because no condition_first here
    condition_range = [condition_start_zero_indexed, min(int(condition_start_zero_indexed)+int(cfg.test.n_conditions), max_dataset_size)]
    log.info(f'condition_range: {condition_range}\n')
    actual_n_conditions = condition_range[1] - condition_range[0] # handles the case where max_dataset_size < start+n_conditions

    # Load the data
    # file_path = samples_from_wandb(cfg.general.wandb.entity, cfg.general.wandb.run_id, cfg.general.wandb.project,
    #                     sampling_steps, epoch, total_conditions, cfg.test.n_samples_per_condition)
    # Assumes that hydra.run.dir is set to the same location as the samples
    condition_start_for_job = int(cfg.test.condition_first) + int(total_index)*int(cfg.test.n_conditions)
    file_path = f"samples_epoch{epoch}_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.gz"
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
    for i in range(len(placeholders_for_print)):
        original_data_placeholder = placeholders_for_print[i]
        elbo_sorted_reactions = all_elbo_sorted_reactions[i]
        weighted_prob_sorted_rxns = all_weighted_prob_sorted_rxns[i]
        
        '''
        (dense_data, rdkit_atom_types, rdkit_bond_types, rdkit_atom_charges, rdkit_atom_chiral_tags, 
                                          return_dict=False, plot_dummy_nodes=False)
        '''
        true_rxns = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=original_data_placeholder, cfg=cfg) 
        # true_rcts, true_prods = mol.get_cano_list_smiles(X=original_data_placeholder.X, E=original_data_placeholder.E, atom_types=model.dataset_info.atom_decoder, 
        #                                                  bond_types=model.dataset_info.bond_decoder, plot_dummy_nodes=cfg.test.plot_dummy_nodes)
        graph.save_samples_to_file_without_weighted_prob(f'eval_epoch{epoch}_steps{sampling_steps}_s{condition_start_for_job}.txt', i, elbo_sorted_reactions, true_rxns)
        graph.save_samples_to_file(f'eval_epoch{epoch}_steps{sampling_steps}_resorted_{cfg.test.sort_lambda_value}_s{condition_start_for_job}.txt', i, weighted_prob_sorted_rxns, true_rxns)
    
    for score in scores:
        for key, value in score.items():
            if isinstance(value, torch.Tensor):
                score[key] = value.detach().cpu().numpy()
    
    pickle.dump(scores, open(f'scores_epoch{epoch}_steps{sampling_steps}_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}_s{condition_start_for_job}.pickle', 'wb'))

if __name__ == '__main__':
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        try:
            main()
        except Exception as e:
            log.exception("main crashed. Error: %s", e)
    else:
        main()
