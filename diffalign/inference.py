"""Core inference pipeline for DiffAlign retrosynthesis model.

Provides model loading, SMILES-to-graph conversion, and prediction functions.
This module does NOT eagerly load the model at import time — callers decide
when to trigger loading via ``get_model_and_cfg()``.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from diffalign.constants import BOND_TYPES
from diffalign.utils.setup import dotdict, load_weights
from diffalign.utils.mol import get_cano_smiles_from_dense
from diffalign.utils.graph_builder import build_rxn_graph
from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign.datasets.abstract_dataset import DistributionNodes
from diffalign.utils.graph import to_dense, get_index_from_states
from diffalign.diffusion.noise_schedule import AbsorbingStateTransitionMaskNoEdge

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
BOND_ORDERS = [0, 1, 2, 3, 0, 0, 0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model caching ──────────────────────────────────────────────────────────
_cached_model = None
_cached_cfg = None


def get_model_and_cfg():
    """Load model and config once, return cached copies on subsequent calls."""
    global _cached_model, _cached_cfg
    if _cached_model is not None:
        return _cached_model, _cached_cfg

    log.info("Loading DiffAlign model (one-time)...")
    GlobalHydra.instance().clear()
    with initialize(config_path="../configs"):
        cfg = compose(
            config_name="default",
            overrides=["+experiment=align_absorbing"]
        )
    # Default eval steps — cheap to rebuild later if caller wants different value
    cfg.diffusion.diffusion_steps_eval = 1

    model = load_model(cfg)
    model.eval()
    log.info("DiffAlign model loaded and cached.")

    _cached_model = model
    _cached_cfg = cfg
    return _cached_model, _cached_cfg


# ── Transition-model cache (keyed by diffusion_steps_eval) ─────────────────
_transition_cache = {}


def _get_or_build_transition_model_eval(model, cfg, diffusion_steps_eval):
    """Rebuild transition_model_eval for a different step count. Cheap (just matrices)."""
    if diffusion_steps_eval in _transition_cache:
        return _transition_cache[diffusion_steps_eval]

    abs_state_position_x = model.dataset_info.atom_decoder.index('Au')
    abs_state_position_e = 0
    node_idx_to_mask, edge_idx_to_mask = get_index_from_states(
        atom_decoder=model.dataset_info.atom_decoder,
        bond_decoder=model.dataset_info.bond_decoder,
        node_states_to_mask=cfg.diffusion.node_states_to_mask,
        edge_states_to_mask=cfg.diffusion.edge_states_to_mask,
        device=device,
    )

    tm = AbsorbingStateTransitionMaskNoEdge(
        x_classes=model.Xdim_output,
        e_classes=model.Edim_output,
        y_classes=model.ydim_output,
        timesteps=diffusion_steps_eval,
        diffuse_edges=cfg.diffusion.diffuse_edges,
        abs_state_position_x=abs_state_position_x,
        abs_state_position_e=abs_state_position_e,
        node_idx_to_mask=node_idx_to_mask,
        edge_idx_to_mask=edge_idx_to_mask,
    )
    _transition_cache[diffusion_steps_eval] = tm
    return tm


def smiles_to_dense_data(
    product_smiles: str,
    max_nodes_more_than_product: int,
    with_explicit_h: bool,
    with_formal_charge: bool,
    canonicalize_molecule: bool,
    permute_mols: bool,
    add_supernode_edges: bool,
    atom_types: List[str],
    bond_types: List[BT],
):
    # Add atom mapping numbers so positional encodings work correctly.
    # Without mappings, all atoms get identical encodings (catastrophically OOD).
    mol = Chem.MolFromSmiles(product_smiles)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    product_smiles_mapped = Chem.MolToSmiles(mol)
    rxn_smiles = f"{product_smiles_mapped}>>{product_smiles_mapped}"
    reactants = [r for r in rxn_smiles.split('>>')[0].split('.')]
    products = [p for p in rxn_smiles.split('>>')[1].split('.')]

    data, cannot_generate = build_rxn_graph(
        reactants=reactants,
        products=products,
        atom_types=atom_types,
        bond_types=bond_types,
        max_nodes_more_than_product=max_nodes_more_than_product,
        with_explicit_h=with_explicit_h,
        with_formal_charge=with_formal_charge,
        add_supernode_edges=False,  # must match dataset process() which hardcodes False
        canonicalize_molecule=canonicalize_molecule,
        permute_mols=permute_mols,
        scramble_atom_mapping=True,
        idx=0,
    )

    # Wrap in a Batch to match the working pipeline (to_dense expects data.batch)
    batch = Batch.from_data_list([data])
    dense_data = to_dense(batch)
    return dense_data


def load_model(cfg):
    device_count = 1
    # TODO: get rid of dataset_infos and maybe model_kwargs
    atom_types = cfg.dataset.atom_types
    allowed_bonds = cfg.dataset.allowed_bonds
    valencies = [0] + list(abs[0] for atom_type, abs in allowed_bonds.items() if atom_type in atom_types) + [0]
    periodic_table = Chem.rdchem.GetPeriodicTable()
    atom_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in atom_types[1:-1]] + [0] # discard charge
    atom_weights = {atom_type: weight for atom_type, weight in zip(atom_types, atom_weights)}

    # Look in checkpoints/ first, fall back to data processed dir
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    data_dir = os.path.join(PROJECT_ROOT, cfg.dataset.datadir + '-' + str(cfg.dataset.dataset_nb), 'processed')
    def _find(name):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
        return None
    node_count_path = _find('n_counts.txt')
    atom_type_path = _find('atom_types.txt')
    edge_type_path = _find('edge_types.txt')
    atom_type_unnorm_path = _find('atom_types_unnorm_mol.txt')
    edge_type_unnorm_path = _find('edge_types_unnorm_mol.txt')
    paths_exist = all(p is not None for p in [node_count_path, atom_type_path, edge_type_path,
                                               atom_type_unnorm_path, edge_type_unnorm_path])

    if paths_exist:
        # use the same distributions for all subsets of the dataset
        n_nodes = torch.from_numpy(np.loadtxt(node_count_path))
        node_types = torch.from_numpy(np.loadtxt(atom_type_path))
        edge_types = torch.from_numpy(np.loadtxt(edge_type_path))
        node_types_unnormalized = torch.from_numpy(np.loadtxt(atom_type_unnorm_path)).long()
        edge_types_unnormalized = torch.from_numpy(np.loadtxt(edge_type_unnorm_path)).long()

    dataset_infos = dotdict({
        'node_types_unnormalized': node_types_unnormalized,
        'edge_types_unnormalized': edge_types_unnormalized,
        'num_classes': len(node_types),
        'max_n_nodes': len(n_nodes) - 1,
        'nodes_dist': DistributionNodes(n_nodes),
        'atom_decoder': cfg.dataset.atom_types,
        'bond_decoder': BOND_TYPES,
        'valencies': valencies,
        'atom_weights': atom_weights,
        'max_weight': cfg.dataset.max_atom_weight,
        'bond_orders': BOND_ORDERS,
        'remove_h': cfg.dataset.remove_h,
        'input_dims': dotdict({
            'X': len(cfg.dataset.atom_types),
            'E': len(BOND_TYPES),
            'y': 1
        }),
        'output_dims': dotdict({
            'X': len(cfg.dataset.atom_types),
            'E': len(BOND_TYPES),
            'y': 0
        })
    })
    model_kwargs={
        'dataset_infos': dataset_infos,
        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
        'use_data_parallel': device_count>1
    }
    model = DiscreteDenoisingDiffusionRxn(cfg=cfg, **model_kwargs)
    model = model.to(device)
    checkpoint_file = os.path.join(PROJECT_ROOT, 'checkpoints', f'epoch760.pt')
    print(f'loading model from {checkpoint_file}')
    state_dict = torch.load(checkpoint_file, map_location=device)['model_state_dict']
    model = load_weights(model, state_dict, device_count=1)
    return model

def predict_precursors_from_diffalign(
    product_smiles: str,
    n_precursors: int = 1,
    diffusion_steps: int = 1
):
    model, cfg = get_model_and_cfg()

    # Validate diffusion_steps divides T=500
    T = cfg.diffusion.diffusion_steps  # 500
    assert T % diffusion_steps == 0, (
        f"diffusion_steps={diffusion_steps} must divide T={T}. "
        f"Valid values: {[d for d in range(1, T+1) if T % d == 0]}"
    )

    # Swap transition_model_eval if steps differ from current
    if diffusion_steps != cfg.diffusion.diffusion_steps_eval:
        model.transition_model_eval = _get_or_build_transition_model_eval(
            model, cfg, diffusion_steps
        )
        cfg.diffusion.diffusion_steps_eval = diffusion_steps
    log.info(f"Running inference with diffusion_steps_eval={diffusion_steps}")
    # create dense data
    dense_data = smiles_to_dense_data(
        max_nodes_more_than_product=cfg.dataset.nb_rct_dummy_nodes,
        product_smiles=product_smiles,
        atom_types=cfg.dataset.atom_types,
        bond_types=BOND_TYPES,
        with_explicit_h=cfg.dataset.with_explicit_h,
        with_formal_charge=cfg.dataset.with_formal_charge,
        add_supernode_edges=cfg.dataset.add_supernode_edges,
        canonicalize_molecule=cfg.dataset.canonicalize_molecule,
        permute_mols=cfg.dataset.permute_mols
    )
    dense_data = dense_data.to_device(device)
    print('done creating dense data.')
    # sample
    with torch.inference_mode():
        final_samples = model.sample_for_condition(
            dense_data=dense_data,
            n_samples=n_precursors,
            inpaint_node_idx=None,
            inpaint_edge_idx=None,
            device=device
        )
    print('done sampling.')

    # Convert dense samples to reaction SMILES strings
    all_rxn_str = get_cano_smiles_from_dense(
        final_samples.X, final_samples.E,
        cfg.dataset.atom_types, BOND_TYPES
    )

    # Extract reactant SMILES (left side of >>), deduplicate, and score by frequency
    reactant_counts = {}
    for rxn_str in all_rxn_str:
        if '>>' in rxn_str:
            reactants = rxn_str.split('>>')[0]
        else:
            reactants = rxn_str
        if reactants:
            reactant_counts[reactants] = reactant_counts.get(reactants, 0) + 1

    total = len(all_rxn_str) if all_rxn_str else 1
    results = []
    for precursors, count in reactant_counts.items():
        results.append({'precursors': precursors, 'score': count / total})
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:n_precursors]


def predict_precursors(
    product_smiles: str,
    n_precursors: int = 1,
    diffusion_steps: int = 1,
) -> List[Dict[str, Any]]:
    """Predict retrosynthesis precursors for a given product SMILES."""
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return []

    results = predict_precursors_from_diffalign(
        product_smiles=product_smiles,
        n_precursors=n_precursors,
        diffusion_steps=diffusion_steps,
    )

    # Validate SMILES in results
    validated = []
    for result in results:
        all_valid = True
        for smi in result['precursors'].split('.'):
            if Chem.MolFromSmiles(smi) is None:
                all_valid = False
                break
        if all_valid:
            validated.append(result)
    return validated
