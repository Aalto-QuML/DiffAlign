import wandb
from diffalign.utils import setup
from datetime import datetime

entity = 'najwalb'
project = 'retrodiffuser'
run_id = 'b4nxbanv'
epochs = [140, 180, 220, 260, 300]
type_ = 'samples'
cond = 128
sampercond = 100
ts = int(round(datetime.now().timestamp()))

for e in epochs:
    filepath = f'/scratch/project_2006950/RetroDiffuser/experiments/train_rxn_q717nhz1/samples_epoch{e}.txt'
    filename = f'samples_epoch{e}_cond{cond}_sampercond{sampercond}.txt'
    alias = f'epoch{e}_cond{cond}_sampercond{sampercond}_{ts}'
    artifact_name = f'{run_id}_{type_}'

    run = wandb.init(entity=entity, project=project)
    artifact_name_in_wandb = f"{entity}/{project}/{run_id}_model:epoch{e}"
    run.use_artifact(artifact_name_in_wandb)
    setup.save_file_as_artifact_to_wandb(run, artifactname=f'{artifact_name}', type_=type_, alias=alias, filepath=filepath, filename=filename)
