"""Helpers for scripts/sample.py."""
import logging
import os

from diffalign.utils import graph, setup

log = logging.getLogger(__name__)


def compute_condition_range(cfg, mpi_rank: int = 0, mpi_size: int = 1):
    """Per-job condition indexing for array/MPI parallelism.

    Returns (total_index, condition_start_for_job).
    """
    total_index = cfg.test.condition_index * mpi_size + mpi_rank
    condition_start = int(cfg.test.condition_first) + int(total_index) * int(cfg.test.n_conditions)
    return total_index, condition_start


def build_data_slices(cfg, condition_start_for_job: int) -> dict:
    """{'train': None, 'val': None, 'test': None} with the edge_conditional_set slot filled."""
    data_slices = {'train': None, 'val': None, 'test': None}
    data_slices[cfg.diffusion.edge_conditional_set] = [
        int(condition_start_for_job),
        int(condition_start_for_job) + int(cfg.test.n_conditions),
    ]
    return data_slices


def load_weights_for_inference(cfg, epoch_num, *, parent_path, model, optimizer,
                               scheduler, scaler, device_count):
    """Load checkpoint weights from the local checkpoints/ dir (falling back to wandb)."""
    savedir = os.path.join(parent_path, "checkpoints")
    return setup.load_weights_from_wandb_no_download(
        cfg, epoch_num, savedir, model, optimizer, scheduler, scaler,
        device_count=device_count,
    )


def build_output_paths(cfg, *, parent_path, epoch_num, sampling_steps, condition_start_for_job):
    """Build (smiles_txt_path, pyg_gz_path); ensure output_dir exists."""
    output_dir = cfg.test.output_dir or os.path.join(parent_path, "experiments")
    os.makedirs(output_dir, exist_ok=True)
    stem = (f'samples_epoch{epoch_num}_steps{sampling_steps}'
            f'_cond{cfg.test.n_conditions}_sampercond{cfg.test.n_samples_per_condition}'
            f'_s{condition_start_for_job}')
    return os.path.join(output_dir, stem + '.txt'), os.path.join(output_dir, stem + '.gz')


def run_sampling(model, dataloaders, cfg, epoch_num):
    """Run model.sample_n_conditions over cfg.diffusion.edge_conditional_set dataloader."""
    dataloader = dataloaders[cfg.diffusion.edge_conditional_set]
    return model.sample_n_conditions(
        dataloader=dataloader, epoch_num=epoch_num, device_to_use=None,
        inpaint_node_idx=None, inpaint_edge_idx=None,
    )


def save_sampling_outputs(*, output_file_smiles, output_file_pyg,
                          all_gen_rxn_smiles, all_true_rxn_smiles,
                          all_gen_rxn_pyg, all_true_rxn_pyg, condition_start_for_job):
    """Write per-condition SMILES file and the combined pyg .gz file."""
    for i, gen_rxn_smiles in enumerate(all_gen_rxn_smiles):
        true_rxn_smiles = all_true_rxn_smiles[i]
        true_rcts_smiles = [rxn.split('>>')[0].split('.') for rxn in true_rxn_smiles]
        true_prods_smiles = [rxn.split('>>')[1].split('.') for rxn in true_rxn_smiles]
        print(f'saving to file {output_file_smiles}')
        graph.save_gen_rxn_smiles_to_file(
            output_file_smiles,
            condition_idx=condition_start_for_job + i,
            gen_rxns=gen_rxn_smiles,
            true_rcts=true_rcts_smiles[0],
            true_prods=true_prods_smiles[0],
        )
    graph.save_gen_rxn_pyg_to_file(
        filename=output_file_pyg, gen_rxns_pyg=all_gen_rxn_pyg, true_rxns_pyg=all_true_rxn_pyg,
    )
