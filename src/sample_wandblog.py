import os
import multiprocessing
import wandb
from omegaconf import DictConfig
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

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

os.environ['WANDB__SERVICE_WAIT'] = '1000'

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    # get epochs corresponding to successful sampling runs 
    epochs = cfg.general.wandb.checkpoint_epochs
    eval_sampling_steps = cfg.general.wandb.eval_sampling_steps
    assert eval_sampling_steps != None
    assert len(epochs) == 1 or len(eval_sampling_steps) == 1
    ts = int(round(datetime.now().timestamp()))
    max_dataset_size = cfg.dataset.dataset_size.test if cfg.diffusion.edge_conditional_set=='test' else cfg.dataset.dataset_size.val if cfg.diffusion.edge_conditional_set=='val' else cfg.dataset.dataset_size.train

    # NOTE: 
    # Input files are of format samples_epoch{e}_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}
    # Output files are of format
    # samples_epoch{e}_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}

    if len(eval_sampling_steps) == 1:
        with wandb.init(name=f"sample_{cfg.general.wandb.run_id}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}",
                        project=cfg.general.wandb.project, entity=cfg.general.wandb.entity, resume='allow', job_type='sampling') as run:
            for e in epochs:
                sampling_steps = eval_sampling_steps[0]
                # merge all SMILES-encoded output files belonging to this epoch
                regex = r"samples_epoch" + str(e) + r"_steps\d+_cond\d+_sampercond\d+_s\d+\.txt"
                all_output_files_smiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if re.match(regex, f)]#if 'samples' in f and re.search(f'epoch{e}', f) and re.search(f'.txt', f)]
                # cond = min(cfg.test.total_cond_eval, max_dataset_size)
                cond = min(len(all_output_files_smiles)*int(cfg.test.n_conditions), max_dataset_size)
                output_file_smiles = f'samples_epoch{e}_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}.txt'
                io_utils.merge_smiles_sample_output_files(files_to_merge=all_output_files_smiles, merged_output_file_name=output_file_smiles)
                
                # merge all PyG-encoded output files belonging to this epoch
                regex = r"samples_epoch" + str(e) + r"_steps\d+_cond\d+_sampercond\d+_s\d+\.gz"
                all_output_files_pyg = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if re.match(regex, f)]#if 'samples' in f and re.search(f'epoch{e}', f) and re.search(f'.pickle', f)]
                # cond = min(len(all_output_files_smiles)*int(cfg.test.n_conditions), max_dataset_size)
                output_file_pyg = f'samples_epoch{e}_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}.gz'
                io_utils.merge_pyg_sample_output_files(files_to_merge=all_output_files_pyg, merged_output_file_name=output_file_pyg)

                # get name of the artifact corresponding to the model weights to be added as input to the sampling run
                artifact_name_in_wandb = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{cfg.general.wandb.run_id}_model:epoch{e}"
                run.use_artifact(artifact_name_in_wandb)
                # define a whole artifact per epoch. Artifact versions would correspond to the number of samples in each file (or other variables)
                artifact = wandb.Artifact(f'{cfg.general.wandb.run_id}_samples', type='samples')
                assert os.path.exists(output_file_smiles), f'Could not find file {output_file_smiles}.'
                assert os.path.exists(output_file_pyg), f'Could not find file {output_file_pyg}.'
                artifact.add_file(output_file_smiles, name=output_file_smiles)
                artifact.add_file(output_file_pyg, name=output_file_pyg)
                run.log_artifact(artifact, aliases=[f'{output_file_smiles.split(".txt")[0]}'])
    else:
        with wandb.init(name=f"sample_{cfg.general.wandb.run_id}_differentsteps_epoch{epochs[0]}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}",
                        project=cfg.general.wandb.project, entity=cfg.general.wandb.entity, resume='allow', job_type='sampling') as run:
            for sampling_steps in eval_sampling_steps:
                e = epochs[0]
                # merge all SMILES-encoded output files belonging to this epoch
                regex = r"samples_epoch" + str(e) + r"_steps" + str(sampling_steps) + r"_cond\d+_sampercond\d+_s\d+\.txt"
                all_output_files_smiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if re.match(regex, f)]#if 'samples' in f and re.search(f'epoch{e}', f) and re.search(f'.txt', f)]
                # cond = min(cfg.test.total_cond_eval, max_dataset_size)
                cond = min(len(all_output_files_smiles)*int(cfg.test.n_conditions), max_dataset_size)
                output_file_smiles = f'samples_epoch{e}_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}.txt'
                io_utils.merge_smiles_sample_output_files(files_to_merge=all_output_files_smiles, merged_output_file_name=output_file_smiles)
                # merge all PyG-encoded output files belonging to this epoch
                regex = r"samples_epoch" + str(epochs[0]) + r"_steps" + str(sampling_steps) + r"_cond\d+_sampercond\d+_s\d+\.gz"
                all_output_files_pyg = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) if re.match(regex, f)]#if 'samples' in f and re.search(f'epoch{e}', f) and re.search(f'.pickle', f)]
                # cond = min(len(all_output_files_smiles)*int(cfg.test.n_conditions), max_dataset_size)
                output_file_pyg = f'samples_epoch{e}_steps{sampling_steps}_cond{cond}_sampercond{cfg.test.n_samples_per_condition}_{ts}.gz'
                io_utils.merge_pyg_sample_output_files(files_to_merge=all_output_files_pyg, merged_output_file_name=output_file_pyg)

                # get name of the artifact corresponding to the model weights to be added as input to the sampling run
                artifact_name_in_wandb = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{cfg.general.wandb.run_id}_model:epoch{e}"
                run.use_artifact(artifact_name_in_wandb)
                # define a whole artifact per epoch. Artifact versions would correspond to the number of samples in each file (or other variables)
                artifact = wandb.Artifact(f'{cfg.general.wandb.run_id}_samples', type='samples')
                assert os.path.exists(output_file_smiles), f'Could not find file {output_file_smiles}.'
                assert os.path.exists(output_file_pyg), f'Could not find file {output_file_pyg}.'
                artifact.add_file(output_file_smiles, name=output_file_smiles)
                artifact.add_file(output_file_pyg, name=output_file_pyg)
                run.log_artifact(artifact, aliases=[f'{output_file_smiles.split(".txt")[0]}'])

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
