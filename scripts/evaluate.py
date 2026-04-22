'''
Evaluate the samples saved as wandb artifacts.
'''
import logging
import os
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import torch
from omegaconf import DictConfig

from diffalign.evaluation.helpers import (
    build_samples_filepath,
    compute_condition_range,
    dump_scores_pickle,
    load_and_reshape_samples,
    write_evaluation_outputs,
)
from diffalign.sampling.helpers import load_weights_for_inference
from diffalign.training.helpers import (
    build_dataloaders_and_infos,
    build_model_and_train_objects,
    setup_experiment,
)

log = logging.getLogger(__name__)
parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base='1.1', config_path='../configs', config_name='default')
def main(cfg: DictConfig):
    _run, cfg, _cli = setup_experiment(cfg, job_type='ranking', preserve_entity_project=True)

    epoch = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval
    num_gpus = torch.cuda.device_count()

    total_index, condition_start_for_job = compute_condition_range(cfg)
    log.info(f'cfg.test.condition_first & slurm array index & total condition index '
             f'{cfg.test.condition_first}, {cfg.test.condition_index}, {total_index}\n')

    _, dataset_infos = build_dataloaders_and_infos(cfg, return_datamodule=False)
    model, optimizer, scheduler, scaler, _start_epoch = build_model_and_train_objects(
        cfg, run=None, dataset_infos=dataset_infos,
        parent_path=parent_path,
        savedir=os.path.join(parent_path, 'experiments'),
        device=device, device_count=num_gpus,
        use_data_parallel=num_gpus > 1,
        load_weights_bool=False,
    )

    model, optimizer, scheduler, scaler, _artifact = load_weights_for_inference(
        cfg, epoch, parent_path=parent_path, model=model,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, device_count=num_gpus,
    )

    assert cfg.diffusion.edge_conditional_set in ['test', 'val', 'train'], (
        f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set} '
        f'is not a valid value.\n')

    # Assumes hydra.run.dir is the same location as the samples.
    file_path = build_samples_filepath(
        cfg, epoch=epoch, sampling_steps=sampling_steps,
        condition_start_for_job=condition_start_for_job,
    )

    # NOTE: keep dense_data/final_samples on CPU here. evaluate_from_artifact slices
    # them per condition and moves only the slice to `device`, so the full (bs*n_samples)
    # tensors never need to live on the GPU.
    true_graph_data, sample_graph_data, _actual_n, condition_range = load_and_reshape_samples(
        cfg, file_path, condition_start_for_job=condition_start_for_job,
    )

    scores, all_elbo_sorted_reactions, all_weighted_prob_sorted_rxns, placeholders_for_print = \
        model.evaluate_from_artifact(
            dense_data=true_graph_data, final_samples=sample_graph_data, device=device,
            condition_range=condition_range, epoch=epoch,
        )

    write_evaluation_outputs(
        cfg, model,
        placeholders_for_print=placeholders_for_print,
        all_elbo_sorted_reactions=all_elbo_sorted_reactions,
        all_weighted_prob_sorted_rxns=all_weighted_prob_sorted_rxns,
        epoch=epoch, sampling_steps=sampling_steps,
        condition_start_for_job=condition_start_for_job,
    )

    dump_scores_pickle(
        cfg, scores, epoch=epoch, sampling_steps=sampling_steps,
        condition_start_for_job=condition_start_for_job,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
