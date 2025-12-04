import os
import multiprocessing
import wandb
from omegaconf import DictConfig, OmegaConf
import pathlib
import pickle
import time
import hydra
from datetime import datetime
from src.utils import setup, io_utils
import numpy as np
import logging
import re
from os import listdir
from os.path import isfile, join
from src.utils.diffusion import helpers
from src.utils import wandb_utils, graph
import copy

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

os.environ['WANDB__SERVICE_WAIT'] = '1000'

def get_latest_alias_artifact_name_from_collection(cfg, coll, epoch, sampling_steps, collection_name):
    aliases = [alias for art in coll.versions() for alias in art.aliases \
                        if 'samples' in alias
                        and re.search(f'steps{sampling_steps}', alias)
                        and re.search(f'epoch{epoch}', alias)
                        and re.search(f'cond{cfg.test.total_cond_eval}', alias)
                        and re.search(f'sampercond{cfg.test.n_samples_per_condition}', alias)]
    versions = [int(art.version.split('v')[-1]) for art in coll.versions()]
    aliases = [a for a,v in sorted(zip(aliases, versions), key=lambda pair: pair[1], reverse=True)]
    log.info(f'ordered aliases {aliases}\n')
    log.info(f'the script will be using the newest alias: {aliases[0]}\n')
    artifact_name = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{collection_name}:{aliases[0]}"
    return artifact_name

def merge_results_and_log_to_wandb(run, cfg, e, sampling_steps, collection_name, collections):
    """Utility function to make the code in the actual script less repetitive / more readable"""
    # Merge all SMILES-encoded output files belonging to this epoch
    regex = r"eval_epoch" + str(e) + r"_steps" + str(sampling_steps) + r"_resorted_0.9_s\d+.txt" # NOTE: WE ASSUME THAT THE DEFAULT IS ALWAYS 0.9, only change this in this script
    files_to_merge = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if re.match(regex, f)]
    log.info(f"Files to merge for sampling_steps {sampling_steps}: {files_to_merge}")
    merged_output_file_name = f'eval_epoch{e}_steps{sampling_steps}_resorted_{cfg.test.sort_lambda_value}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}_lam{cfg.test.sort_lambda_value}.txt'
    
    scores = None
    if cfg.test.sort_lambda_value != 0.9:
        new_files_to_merge = []

        all_scores = []

        for filename in files_to_merge:
            f = open(filename, "r")
            samples = f.read()
            f.close()
            reactions = io_utils.read_saved_reaction_data(samples)
            true_reactants = [r[0].split(">>")[0].split('.') for r in reactions]
            # true_reactants = [r for reactants in true_reactants for r in reactants if len(r) < 2]
            true_products = [r[0].split(">>")[1].split('.') for r in reactions]
            elbo_sorted_rxns_no_duplicates = io_utils.restructure_reactions(reactions, with_count_and_prob=True)
            weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_rxns_no_duplicates, cfg.test.sort_lambda_value)
            weighted_prob_sorted_rxns = graph.filter_small_molecules(weighted_prob_sorted_rxns, filter_limit=1)
            # create new file name by replacing the
            pattern = r"(_resorted_)?\d+\.\d+"
            # The replacement function
            def replacement(match):
                return match.group(1) + str(cfg.test.sort_lambda_value)
            # Replace the matched patterns in the original string
            new_filename = re.sub(pattern, replacement, filename)
            graph.save_samples_to_file_with_overwrite(new_filename, weighted_prob_sorted_rxns, true_reactants, true_products)
            new_files_to_merge.append(new_filename)
            topk = graph.calculate_top_k(cfg, weighted_prob_sorted_rxns, true_reactants, true_products)
            scores = dict()
            for j, k_ in enumerate(cfg.test.topks):
                # scores[f'top-{k_}'] = topk[:,j].mean().item()
                # for i, sort_lambda_value in enumerate(sort_lambda_values):
                scores[f'top-{k_}_weighted_{cfg.test.sort_lambda_value}'] = topk[:,j].mean().item()            
            all_scores.append(scores)
        scores = dict(io_utils.merge_scores_from_dicts(all_scores))
        log.info(scores)
        # , condition_idx, gen_rxns, true_rcts, true_prods
        files_to_merge = new_files_to_merge
    else:
        # merge all precalculated score dicts belonging to this epoch
        all_score_files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if 'scores' in f and re.search(f'epoch{e}_', f) and re.search(f'steps{sampling_steps}_', f)]
        print(f'all_score_files {len(all_score_files)}\n')
        print(f'all_score_files {all_score_files}\n')
        scores = io_utils.merge_scores(file_scores_to_merge=all_score_files)

    io_utils.merge_smiles_sample_output_files(files_to_merge, merged_output_file_name)
    artifact = wandb.Artifact(f'{cfg.general.wandb.run_id}_eval', type='eval')
    artifact.add_file(merged_output_file_name, name=merged_output_file_name)
    run.log_artifact(artifact, aliases=[f'{merged_output_file_name.split(".txt")[0]}'])

    print(f'scores {scores}\n')
    dict_to_save = {k:v for k, v in scores.items() if k!='rxn_plots'}
    dict_to_save['epoch'] = e
    dict_to_save['sampling_steps'] = sampling_steps
    log.info(dict_to_save)
    run.log({'sample_eval/': dict_to_save})

    try:
        previous_artifact_name = get_latest_alias_artifact_name_from_collection(cfg, collections[0], e, sampling_steps, collection_name)
        run.use_artifact(previous_artifact_name)
    except Exception as error:
        log.info("Something went wrong with establishing the run.use_artifact. ")
        log.info(error)

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    
    '''
        - if already merged some scores, resume run
        - merge all saved score dicts
        - log one scores dict per epoch
    '''
    # get epochs corresponding to successful sampling runs 
    '''
        All code below is to link the eval run with the samples artifact => move to function
    '''
    epochs = cfg.general.wandb.checkpoint_epochs
    eval_sampling_steps = cfg.general.wandb.eval_sampling_steps
    assert eval_sampling_steps != None
    assert len(epochs) == 1 or len(eval_sampling_steps) == 1
    collection_name = f"{cfg.general.wandb.run_id}_samples"
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name='samples', project=f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}").collections()
        if coll.name==collection_name
    ]
    assert len(collections)==1, f'Found {len(collections)} collections with name {collection_name}, expected 1.'
    
    assert cfg.test.total_cond_eval != None
    assert cfg.test.n_samples_per_condition != None

    # Update the experiment_group field in the original run config
    run_path = wandb_utils.get_run_path(cfg.general.wandb.run_id, cfg.general.wandb.project, cfg.general.wandb.entity)
    orig_run = wandb_utils.get_run(run_path)
    orig_run.config["experiment_group"] = cfg.general.wandb.run_id
    orig_run.update()
    # Add the experiment_group field in the new eval run config for grouping with the original run
    cfg_to_save = copy.deepcopy(orig_run.config)

    # OmegaConf.set_struct(cfg_to_save, False)
    cfg_to_save['experiment_group'] = cfg.general.wandb.run_id

    if len(eval_sampling_steps) == 1:
        with wandb.init(name=f"eval_{cfg.general.wandb.run_id}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_loss0_{cfg.test.loss_0_repeat}_{cfg.diffusion.edge_conditional_set}_s{cfg.test.condition_first}_steps{eval_sampling_steps[0]}_sort_lambda_{cfg.test.sort_lambda_value}", 
                        project=cfg.general.wandb.project, entity=cfg.general.wandb.entity, 
                        resume='allow', job_type='ranking', config=cfg_to_save) as run:
            sampling_steps = eval_sampling_steps[0]
            for e in sorted(cfg.general.wandb.checkpoint_epochs):
                # Get the relevant numbers for this particular epoch

                # All of this is just to recover the original sample artifact to get the lineages right -> they used to be slightly wrong 
                # coll = collections[0]
                # # aliases = [alias for art in coll.versions() for alias in art.aliases \
                # #                 if 'samples' in alias
                # #                 and re.findall('epoch\d+', alias)[0]==f'epoch{epoch}'
                # #                 and re.findall('cond\d+', alias)[0]==f'cond{cfg.test.total_cond_eval}'
                # #                 and re.findall('sampercond\d+', alias)[0]==f'sampercond{cfg.test.n_samples_per_condition}']
                
                # aliases = [alias for art in coll.versions() for alias in art.aliases \
                #         if 'samples' in alias
                #         and re.search(f'steps{sampling_steps}', alias)
                #         and re.search(f'epoch{e}', alias)
                #         and re.search(f'cond{cfg.test.total_cond_eval}', alias)
                #         and re.search(f'sampercond{cfg.test.n_samples_per_condition}', alias)]

                # versions = [int(art.version.split('v')[-1]) for art in coll.versions()]

                # aliases = [a for a,v in sorted(zip(aliases, versions), key=lambda pair: pair[1], reverse=True)]
                # #log.info(f'cfg.general.wandb.sample_file_name {cfg.general.wandb.sample_file_name}\n')
                # log.info(f'ordered aliases {aliases}\n')
                # log.info(f'the script will be using the newest alias: {aliases[0]}\n')
                # #savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
                # artifact_name = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{collection_name}:{aliases[0]}"
                # #samples_art = wandb.Api().artifact(artifact_name)
                
                merge_results_and_log_to_wandb(run, cfg, e, sampling_steps, collection_name, collections)

                # Merge all SMILES-encoded output files belonging to this epoch
                # regex = r"eval_epoch" + str(e) + r"_steps" + str(sampling_steps) + r"\d+_resorted_?\d+\.\d+_s\d+.txt"
                # files_to_merge = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if re.match(regex, f)]
                # merged_output_file_name = f'eval_epoch{e}_steps{sampling_steps}_resorted_{cfg.test.sort_lambda_value}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}.txt'
                # io_utils.merge_smiles_sample_output_files(files_to_merge, merged_output_file_name)
                # artifact = wandb.Artifact(f'{cfg.general.wandb.run_id}_eval', type='eval')
                # artifact.add_file(merged_output_file_name, name=merged_output_file_name)
                # run.log_artifact(artifact, aliases=[f'{merged_output_file_name.split(".txt")[0]}'])

                # # merge all score dicts belonging to this epoch
                # all_score_files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if 'scores' in f and re.search(f'epoch{e}', f)]
                # print(f'all_score_files {len(all_score_files)}\n')
                # print(f'all_score_files {all_score_files}\n')
                # scores = io_utils.merge_scores(file_scores_to_merge=all_score_files)
                # print(f'scores {scores}\n')
                # dict_to_save = {k:v for k, v in scores.items() if k!='rxn_plots'}
                # dict_to_save['epoch'] = e
                # run.log({'sample_eval/': dict_to_save})

                # previous_artifact_name = get_latest_alias_artifact_name_from_collection(cfg, collections[0], e, sampling_steps, collection_name)
                # run.use_artifact(previous_artifact_name)
    else:
        with wandb.init(name=f"eval_{cfg.general.wandb.run_id}_differentsteps_epoch{epochs[0]}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_loss0_{cfg.test.loss_0_repeat}_{cfg.diffusion.edge_conditional_set}_s{cfg.test.condition_first}_sort_lambda_{cfg.test.sort_lambda_value}", 
                        project=cfg.general.wandb.project, entity=cfg.general.wandb.entity, 
                        resume='allow', job_type='ranking', config=cfg_to_save) as run:
            e = epochs[0]
            for sampling_steps in eval_sampling_steps:
                # All of this is just to recover the original sample artifact to get the lineages right
                # coll = collections[0]
                # aliases = [alias for art in coll.versions() for alias in art.aliases \
                #         if 'samples' in alias
                #         and re.search(f'steps{sampling_steps}', alias)
                #         and re.search(f'epoch{e}', alias)
                #         and re.search(f'cond{cfg.test.total_cond_eval}', alias)
                #         and re.search(f'sampercond{cfg.test.n_samples_per_condition}', alias)]
                # versions = [int(art.version.split('v')[-1]) for art in coll.versions()]
                # aliases = [a for a,v in sorted(zip(aliases, versions), key=lambda pair: pair[1], reverse=True)]
                # log.info(f'ordered aliases {aliases}\n')
                # log.info(f'the script will be using the newest alias: {aliases[0]}\n')
                # artifact_name = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{collection_name}:{aliases[0]}"

                log.info(f"Merging and logging sampling steps: {sampling_steps}")
                merge_results_and_log_to_wandb(run, cfg, e, sampling_steps, collection_name, collections)

                # # Merge all SMILES-encoded output files belonging to this epoch and with this amount of sampling steps
                # regex = r"eval_epoch" + str(e) + r"_steps" + str(sampling_steps) + r"\d+_resorted_?\d+\.\d+_s\d+.txt"
                # files_to_merge = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if re.match(regex, f)]
                # merged_output_file_name = f'eval_epoch{e}_steps{sampling_steps}_resorted_{cfg.test.sort_lambda_value}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}.txt'
                # io_utils.merge_smiles_sample_output_files(files_to_merge, merged_output_file_name)
                # artifact = wandb.Artifact(f'{cfg.general.wandb.run_id}_eval', type='eval')
                # artifact.add_file(merged_output_file_name, name=merged_output_file_name)
                # run.log_artifact(artifact, aliases=[f'{merged_output_file_name.split(".txt")[0]}'])

                # # merge all score dicts belonging to this sampling step count
                # all_score_files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if 'scores' in f and re.search(f'epoch{e}', f)]
                # print(f'all_score_files {len(all_score_files)}\n')
                # print(f'all_score_files {all_score_files}\n')
                # scores = io_utils.merge_scores(file_scores_to_merge=all_score_files)
                # print(f'scores {scores}\n')
                # dict_to_save = {k:v for k, v in scores.items() if k!='rxn_plots'}
                # dict_to_save['epoch'] = e
                # dict_to_save['sampling_steps'] = sampling_steps
                # run.log({'sample_eval/': dict_to_save})
                
                # previous_artifact_name = get_latest_alias_artifact_name_from_collection(cfg, collections[0], e, sampling_steps, collection_name)
                # run.use_artifact(previous_artifact_name)



if __name__ == '__main__':
    # main()
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
