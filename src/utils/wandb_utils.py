import wandb
import os
import yaml
from omegaconf import OmegaConf
import omegaconf
from src.utils import setup

def get_run_path(run_id, project, entity):
    return f"{entity}/{project}/{run_id}"

# 1. Access the run using the run path
def get_run(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    return run

def resume_run(cfg):
    assert cfg.general.wandb_id != "" and cfg.general.wandb_id != None, "wandb_id must be set if wandb_resume is True"
    return wandb.init(id=cfg.general.wandb_id, project=cfg.general.project, entity=cfg.general.wandb_team, resume="allow")

def download_config_to(savedir, run):
    # Downloads the run config (shared between processes) and creates the directory structure
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    config_path = os.path.join(savedir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(run.config, f)
    return savedir

def load_config_from(savedir):
    config_path = os.path.join(savedir, 'config.yaml')
    config = OmegaConf.load(config_path)
    return config

def download_checkpoint(savedir, run, epoch_num):
    # Download the checkpoint
    artifact_name_prefix = f"eval_epoch{epoch_num}"
    all_artifacts = run.logged_artifacts()
    artifact_name = None
    for a in all_artifacts:
        if a.name.startswith(artifact_name_prefix + ":"):
            artifact_name = a.name
            a.download(root=savedir)
    assert artifact_name is not None, f"Artifact with prefix {artifact_name_prefix} not found for the specified run."

    # Get the name of the downloaded file
    downloaded_file = os.path.join(savedir, "artifacts", artifact_name, artifact_name.split(":")[0] + ".pt")

    return downloaded_file

# def eval_key_old(cfg, with_elborep=False):
#     key = f'eval_ncond_{cfg.diffusion.edge_conditional_set}_{cfg.test.n_conditions}'\
#         + (f"_clfgw_{cfg.diffusion.classifier_free_guidance_weight}" if cfg.diffusion.classifier_free_guidance_weight != 0.1 else "")\
#         + (f"_lambdatest_{cfg.diffusion.lambda_test}" if cfg.diffusion.lambda_test != 1 else "") \
#         + '/'
#     return key

def eval_key(cfg, new_key=False):
    if new_key:
        key = f'eval_noduplicates_ncond_{cfg.diffusion.edge_conditional_set}_{cfg.test.n_conditions}'\
            + (f"_clfgw_{cfg.diffusion.classifier_free_guidance_weight}" if cfg.diffusion.classifier_free_guidance_weight != 0.1 else "")\
            + (f"_lambdatest_{cfg.diffusion.lambda_test}" if cfg.diffusion.lambda_test != 1 else "") \
            + (f"_elborep_{cfg.test.repeat_elbo}") + '/'
    else:
        key = f'eval_ncond_{cfg.diffusion.edge_conditional_set}_{cfg.test.n_conditions}'\
            + (f"_clfgw_{cfg.diffusion.classifier_free_guidance_weight}" if cfg.diffusion.classifier_free_guidance_weight != 0.1 else "")\
            + (f"_lambdatest_{cfg.diffusion.lambda_test}" if cfg.diffusion.lambda_test != 1 else "") \
            + '/'
        
    return key

def save_file(run_id, project, entity, file_path):
    # e.g., for saving the modified config file
    with wandb.init(id=run_id, project=project, entity=entity, resume="allow") as run:
        # Move back to the original directory so that wandb.save works properly
        wandb.save(file_path)

def save_results(run_id, project, entity, results, directory, file_prefix, file_postfix):
    # This saves results in the form of a dictionary, where the keys are the epoch numbers and the values are the results
    # TODO: Add a filtering function to filter out the epochs that don't have results
    with wandb.init(id=run_id, project=project, entity=entity, resume="allow") as run:
        # Move back to the original directory so that wandb.save works properly
        for i, epoch in enumerate(results.keys()):
            # os.chdir(orig_dir)
            wandb.log({epoch:results[epoch]})
            # wandb.log({key: })
            if os.path.exists(os.path.join(directory, file_prefix + f"{epoch}" + file_postfix)): # also save accompanying file if it exists
                wandb.save(os.path.join(directory, file_prefix + f"{epoch}" + file_postfix))

# TODO: Maybe this one shouldn't be here, but instead setup.py, since setup.py is more like the one that contains these messy functions useful for setting up training
# def setup_wandb(cfg, parent_path):
#     config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
#     # if cfg.general.wandb_resume -> cfg.general.wandb_id != ''
    
#     assert (cfg.general.wandb_resume == False) or (cfg.general.wandb_resume != False and cfg.general.wandb_id != ''), "If wandb_resume is True, wandb_id must be set"
#     # tags and groups
#     kwargs = {'name': cfg.general.name, 'project': cfg.general.project, 'config': config_dict,
#               'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb,
#               'entity': cfg.general.wandb_team, 'resume': cfg.general.wandb_resume}
#     if cfg.general.wandb_resume:
#         assert cfg.general.wandb_id!='', "If wandb_resume is True, wandb_id must be set"
#         kwargs['id'] = cfg.general.wandb_id
#     wandb.init(**kwargs)
#     wandb.save('*.txt')
#     setup.update_config_with_run_id(parent_path, wandb.run.id)