'''
Take the samples from text files in old run and turn them into wandb artifacts.
'''

'''
Workflow idea:
- run = experiment with hyperparams
- each run produces n < epochs checkpoints
- some checkpoints are better than others => want to evaluate more and choose the best one
    (ideally, after the model converges it's enough to use the model's last checkpoint)
- choose checkpoints to extra evaluate (last n or best based on topk) => save to model registry (all of them?) 
=> sample/ or fake sample for old runs (parse samples from txt file) => save samples as artifacts
(as versions of the same artifact?) => evaluate samples with new evaluation script from samples artifact
=> save best samples in model registry?
'''
import wandb
from diffalign.utils import io_utils, wandb_utils
import os
from omegaconf import OmegaConf, DictConfig
import pickle
import logging
import hydra

# A logger for this file
log = logging.getLogger(__name__)

def save_samples_as_artifact_to_wandb(project, entity, model_art, cfg={}, artifactname='samples', filename='samples.pickle'):
    '''
        Uploads samples as artifact to wandb and returns the run id.
    '''
    run = wandb.init(project=project, entity=entity, config=OmegaConf.to_container(cfg), job_type='sampling')
    run.use_artifact(model_art)
    artifact = wandb.Artifact(artifactname, type='samples')
    artifact.add_file(filename, name=filename)
    run.log_artifact(artifact, aliases=[f'seed{cfg.train.seed}'])
    run.finish()

def get_run_path(run_id, project, entity):
    return f"{entity}/{project}/{run_id}"

def get_run(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    return run

def download_files(savedir, run, epoch_num):
    # Download the checkpoint
    artifact_name_prefix = f"eval_epoch{epoch_num}"

    all_artifacts = run.logged_artifacts()
    artifact_name = None
    for a in all_artifacts:
        if a.name.startswith(artifact_name_prefix + ":"):
            artifact_name = a.name
            a.download(root=savedir)
            
    assert artifact_name is not None, f"Artifact with prefix {artifact_name_prefix} not found for the specified run."

    return a, artifact_name

def add_to_model_registry(project, entity, model_registry, artifact):
    artifact.link(f'{entity}/model-registry/{model_registry}')

@hydra.main(version_base='1.1', config_path='../../configs', config_name=f'default')
def main(cfg: DictConfig):
    run_id = "b6f8j287" 
    project = "retrodiffuser"
    model_registry = 'semi-template'
    entity = "najwalb"
    epoch_nums = [9] #319
    # whether to use the new or old key system (for choosing samples to reeval)
    # new is noduplicates
    new_key = False 
    # whether the numbers for each reaction contains the duplicates count
    # True for new samples (with noduplicates)
    with_count = new_key
    sort_lambda_values = [0.9]
    num_conditions = 128

    run_path = get_run_path(run_id, project, entity)
    run = get_run(run_path)

    art, artifact_name = download_files(savedir='.', run=run, epoch_num=epoch_nums[0])

    # add_to_model_registry(project, entity, model_registry=model_registry, artifact=art)

    # get samples corresponding to these checkpoints
    # parse the samples into an artifact object...
    ## what do we want in it: 
    ### samples per product
    ### list of numbers representing the stuff computed for it? => no
    #   (probably shld only be saved in metrics, maybe in text files too for extra caution, but not in artifacts)
    # save sample in samples artifact 
    ## get the model artifact just so we can have the link in model registry
    ## bonus: if can get ancestor run/artifact from samples' run (or any current run), 
    # otherwise need to save the model name somewhere (maybe config?)
    # eval samples (also test in this script)

    epoch = epoch_nums[0]
    cfg = OmegaConf.create(run.config)
    cfg.test.n_conditions = num_conditions
    run_key = wandb_utils.eval_key(cfg, new_key=new_key) # TODO: This needs to be changed for saving the new results later. For now should be "eval_ncond_val_128"
    run_key = ''
    savedir = os.path.join("experiments", "trained_models", run.name + run_id)

    files = run.files()
    epochs = []
    for file in files:
        # if file.name.startswith(os.path.join(savedir, run_key)) and os.path.basename(file.name).startswith(f"samples"):
        if os.path.basename(file.name).startswith(f"samples"):
            # Download the file
            # os.makedirs(os.path.dirname(file.name), exist_ok=True)
            file.download(replace=True)

    # with open(os.path.join(savedir, f"{run_key}samples_epoch{epoch}.txt")) as f:
    with open(f"samples_epoch{epoch}.txt") as f:
        samples = f.read()

    # Then transform the samples.txt data into the elbo_ranked format
    reactions = io_utils.read_saved_reaction_data(samples)
    # art_name = os.path.join(savedir, f"{run_key}samples_epoch{epoch}.txt")
    art_name = f"samples_epoch{epoch}.pickle"
    pickle.dump(reactions, open(art_name, 'wb'))
    save_samples_as_artifact_to_wandb(project, entity, cfg=cfg, model_art=artifact_name, 
                                    artifactname='samples', filename=art_name)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
