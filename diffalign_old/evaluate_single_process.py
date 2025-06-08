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
from diffalign_old.utils import wandb_utils, io_utils, graph
import wandb
from omegaconf import OmegaConf, DictConfig
import numpy as np
import pickle
import hydra
import random
import torch
import logging
import pathlib
from diffalign_old.utils import setup

# A logger for this file
log = logging.getLogger(__name__)
parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]

def download_samples_artifact(savedir, run):
    # Download the checkpoint
    artifact_name_prefix = "samples"
    all_artifacts = run.logged_artifacts()
    artifact_name = None
    for a in all_artifacts:
        if a.name.startswith(artifact_name_prefix+":"):
            artifact_name = a.name
            a.download(root=savedir)
            
    assert artifact_name is not None, f"Artifact with prefix {artifact_name_prefix} not found for the specified run."

    return a, artifact_name, a.files()[0].name # change to smthg more robust

# 1. The run id, etc.
@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    
    # epoch_num = 10
    run = setup.setup_wandb(cfg, job_type='ranking') # This creates a new wandb run or resumes a run given its id
    sort_lambda_values = [0.9]
    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False)
    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, run=run, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False)
    # 1. get samples artifact from run_id
    collection_name = f"{cfg.general.wandb.run_id}_samples"
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name='samples', project=cfg.general.wandb.project).collections()
        if coll.name==collection_name
    ]
    assert len(collections)==1, f'Found {len(collections)} collections with the same name {collection_name}.'
    
    coll = collections[0]
    aliases = [alias for art in coll.versions() for alias in art.aliases if alias!='latest']
    for alias in aliases:
        print(alias)
        epoch_num = int(alias.split('_')[0].split('epoch')[-1])
        savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_name + cfg.general.wandb.run_id)
        model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb(cfg, run, epoch_num, savedir, model, optimizer, scheduler, scaler)
        artifact_name = f"{cfg.general.wandb.entity}/{cfg.general.wandb.project}/{collection_name}:{alias}"
        samples_art = wandb.Api().artifact(artifact_name)
        run.use_artifact(samples_art)
        samples_art.download(root=savedir)
        final_samples = pickle.load(open(os.path.join(savedir, f'all_final_samples_epoch{epoch_num}.pickle'), 'rb'))
        dense_data = pickle.load(open(os.path.join(savedir, f'all_dense_data_epoch{epoch_num}.pickle'), 'rb'))
        # samples_path = os.path.join(savedir, f'samples_epoch{cfg.general.wandb.checkpoint_epoch}.txt')
        # samples = open(samples_path, 'r').read()
        scores = model.evaluate_from_artifact(dense_data=dense_data, final_samples=final_samples)
        wandb.log({'new-pipeline/': scores})
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
