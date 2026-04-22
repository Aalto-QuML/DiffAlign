"""Helpers for scripts/evaluate.py."""
import logging
import pickle

import torch

from diffalign.sampling.helpers import compute_condition_range  # re-export
from diffalign.utils import graph, io_utils, mol

__all__ = [
    'compute_condition_range',
    'build_samples_filepath',
    'load_and_reshape_samples',
    'write_evaluation_outputs',
    'dump_scores_pickle',
]

log = logging.getLogger(__name__)


def build_samples_filepath(cfg, *, epoch, sampling_steps, condition_start_for_job) -> str:
    """Path to the .gz file written by sample.py for this condition range."""
    return (f"samples_epoch{epoch}_steps{sampling_steps}"
            f"_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}"
            f"_s{condition_start_for_job}.gz")


def load_and_reshape_samples(cfg, file_path: str, *, condition_start_for_job: int):
    """Load true/sample pyg data, mask-collapse, reshape by (bs, n_samples).

    Returns (true_graph_data, sample_graph_data, actual_n_conditions, condition_range).
    actual_n_conditions is derived from the loaded data (source of truth).
    """
    true_graph_data, sample_graph_data = io_utils.get_samples_from_file_pyg(
        cfg, file_path, condition_range=None,
    )
    log.info(f'true_graph_data.X.shape {true_graph_data.X.shape}\n')
    log.info(f'sample_graph_data.X.shape {sample_graph_data.X.shape}\n')
    true_graph_data = true_graph_data.mask(collapse=True)
    sample_graph_data = sample_graph_data.mask(collapse=True)

    actual_n_conditions = true_graph_data.X.shape[0] // cfg.test.n_samples_per_condition
    condition_range = [condition_start_for_job, condition_start_for_job + actual_n_conditions]
    log.info(f'actual_n_conditions (from data): {actual_n_conditions}\n')

    true_graph_data.reshape_bs_n_samples(
        bs=actual_n_conditions, n_samples=cfg.test.n_samples_per_condition,
        n=true_graph_data.X.shape[1],
    )
    sample_graph_data.reshape_bs_n_samples(
        bs=actual_n_conditions, n_samples=cfg.test.n_samples_per_condition,
        n=sample_graph_data.X.shape[1],
    )
    return true_graph_data, sample_graph_data, actual_n_conditions, condition_range


def write_evaluation_outputs(cfg, model, *, placeholders_for_print,
                             all_elbo_sorted_reactions, all_weighted_prob_sorted_rxns,
                             epoch, sampling_steps, condition_start_for_job):
    """For each placeholder, write both the elbo-sorted and weighted-sorted txt files."""
    for i, original_data_placeholder in enumerate(placeholders_for_print):
        elbo_sorted_reactions = all_elbo_sorted_reactions[i]
        weighted_prob_sorted_rxns = all_weighted_prob_sorted_rxns[i]
        true_rcts, true_prods = mol.get_cano_list_smiles(
            X=original_data_placeholder.X, E=original_data_placeholder.E,
            atom_types=model.dataset_info.atom_decoder,
            bond_types=model.dataset_info.bond_decoder,
            plot_dummy_nodes=cfg.test.plot_dummy_nodes,
        )
        graph.save_samples_to_file_without_weighted_prob(
            f'eval_epoch{epoch}_steps{sampling_steps}_s{condition_start_for_job}.txt',
            i, elbo_sorted_reactions, true_rcts, true_prods,
        )
        weighted_file = (f'eval_epoch{epoch}_steps{sampling_steps}'
                         f'_resorted_{cfg.test.sort_lambda_value}'
                         f'_s{condition_start_for_job}.txt')
        graph.save_samples_to_file(
            weighted_file, i, weighted_prob_sorted_rxns, true_rcts, true_prods,
        )


def dump_scores_pickle(cfg, scores, *, epoch, sampling_steps, condition_start_for_job):
    """Move any tensor scores to CPU numpy, then pickle.dump."""
    print(f'scores {len(scores)}\n')
    print(f'scores[0] {scores[0]}\n')
    for score in scores:
        for key, value in score.items():
            if isinstance(value, torch.Tensor):
                score[key] = value.detach().cpu().numpy()
    filename = (f'scores_epoch{epoch}_steps{sampling_steps}'
                f'_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}'
                f'_s{condition_start_for_job}.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(scores, f)
