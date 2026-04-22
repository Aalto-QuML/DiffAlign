'''
    Sampling from a trained model.
'''
import logging
import os
import pathlib
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign.sampling.helpers import (
    build_data_slices,
    build_output_paths,
    compute_condition_range,
    load_weights_for_inference,
    run_sampling,
    save_sampling_outputs,
)
from diffalign.training.helpers import (
    build_dataloaders_and_infos,
    build_model_and_train_objects,
    setup_experiment,
)

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]

os.environ['WANDB__SERVICE_WAIT'] = '1000'


@hydra.main(version_base='1.1', config_path='../configs', config_name='default')
def sample(cfg: DictConfig):
    _run, cfg, _cli = setup_experiment(cfg, job_type='ranking')

    log.info(f"Random seed: {cfg.train.seed}")
    log.info(f"Shuffling on: {cfg.dataset.shuffle}")

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device_count: {device_count}, device: {device}\n')

    epoch_num = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval

    total_index, condition_start_for_job = compute_condition_range(cfg)
    log.info(f'cfg.test.condition_first & slurm array index & total condition index '
             f'{cfg.test.condition_first}, {cfg.test.condition_index}, {total_index}\n')
    log.info(f"Condition start: {condition_start_for_job}")
    data_slices = build_data_slices(cfg, condition_start_for_job)
    print(f'data_slices {data_slices}\n')

    dataloaders, dataset_infos = build_dataloaders_and_infos(cfg, slices=data_slices)
    model, optimizer, scheduler, scaler, _start_epoch = build_model_and_train_objects(
        cfg, run=None, dataset_infos=dataset_infos,
        parent_path=parent_path,
        savedir=os.path.join(parent_path, 'experiments'),
        device=device, device_count=device_count,
        use_data_parallel=device_count > 1,
        load_weights_bool=False,
    )
    log.info("2!------------------------------------------------")
    log.info(f": {cfg}")
    log.info(f": {cfg.general}")
    log.info(f": {cfg.general.wandb}")

    model, optimizer, scheduler, scaler, _artifact = load_weights_for_inference(
        cfg, epoch_num, parent_path=parent_path, model=model,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, device_count=device_count,
    )

    output_file_smiles, output_file_pyg = build_output_paths(
        cfg, parent_path=parent_path,
        epoch_num=epoch_num, sampling_steps=sampling_steps,
        condition_start_for_job=condition_start_for_job,
    )

    t0 = time.time()
    log.info(f'About to sample n_conditions={cfg.test.n_conditions}\n')
    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = run_sampling(
        model, dataloaders, cfg, epoch_num,
    )
    save_sampling_outputs(
        output_file_smiles=output_file_smiles, output_file_pyg=output_file_pyg,
        all_gen_rxn_smiles=all_gen_rxn_smiles, all_true_rxn_smiles=all_true_rxn_smiles,
        all_gen_rxn_pyg=all_gen_rxn_pyg, all_true_rxn_pyg=all_true_rxn_pyg,
        condition_start_for_job=condition_start_for_job,
    )

    log.info(f'===== Total sampling time: {time.time()-t0}\n')


if __name__ == '__main__':
    try:
        sample()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
