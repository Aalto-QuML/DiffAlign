"""Shared setup helpers for train/sample/evaluate scripts."""
import logging
from diffalign.helpers import set_seed
from diffalign.utils import setup

log = logging.getLogger(__name__)


def setup_experiment(cfg, job_type):
    """Common experiment setup: wandb init, config loading from wandb run, seed.

    Returns (run_or_None, cfg, cli_overrides).
    """
    run = None
    cli_overrides = setup.capture_cli_overrides()

    if cfg.general.wandb.mode == 'online':
        run, cfg = setup.setup_wandb(cfg, job_type=job_type)

    if cfg.general.wandb.load_run_config:
        run_config = setup.load_wandb_config(cfg)
        cfg = setup.merge_configs(default_cfg=cfg, new_cfg=run_config, cli_overrides=cli_overrides)

    set_seed(cfg.train.seed)
    return run, cfg, cli_overrides


def compute_condition_range(cfg, mpi_rank=0, mpi_size=1):
    """Unified condition index computation for sampling/evaluation.

    Returns (total_index, condition_start_for_job).
    """
    total_index = cfg.test.condition_index * mpi_size + mpi_rank
    condition_start = int(cfg.test.condition_first) + int(total_index) * int(cfg.test.n_conditions)
    return total_index, condition_start


def get_dataset_size(cfg, split):
    """Get dataset size for a given split from config."""
    return getattr(cfg.dataset.dataset_size, split)
