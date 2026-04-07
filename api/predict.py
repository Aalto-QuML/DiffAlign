"""Web API prediction module — thin re-export from diffalign.inference.

All inference logic lives in ``diffalign.inference``. This module re-exports
the public API and eagerly loads the model at import time so the first HTTP
request doesn't pay the cold-start cost.
"""

import torch

# ── CPU inference optimizations (API-specific) ────────────────────────────
torch.set_num_threads(4)  # fewer threads = less contention on shared cloud vCPUs
torch.backends.mkldnn.enabled = True

from diffalign.inference import (  # noqa: E402, F401
    BOND_ORDERS,
    BOND_TYPES,
    PROJECT_ROOT,
    _get_or_build_transition_model_eval,
    device,
    get_model_and_cfg,
    load_model,
    predict_precursors,
    predict_precursors_from_diffalign,
    predict_with_inpainting,
    smiles_to_dense_data,
)

# ── Eager load at import time (runs once, before first request) ────────────
get_model_and_cfg()
