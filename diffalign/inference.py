"""Core inference pipeline for DiffAlign retrosynthesis model.

Provides model loading, SMILES-to-graph conversion, and prediction functions.
This module does NOT eagerly load the model at import time — callers decide
when to trigger loading via ``get_model_and_cfg()``.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from diffalign.constants import BOND_TYPES
from diffalign.utils.setup import dotdict, load_weights
from diffalign.utils.mol import (
    get_cano_smiles_from_dense,
    get_cano_smiles_with_atom_mapping,
    match_atom_mapping_without_stereo,
    transfer_chirality_from_product_to_reactant,
    transfer_bond_dir_from_product_to_reactant,
    remove_atom_mapping_from_mol,
)
from diffalign.utils.placeholder import PlaceHolder
from diffalign.utils.graph_builder import build_rxn_graph
from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from diffalign.datasets.abstract_dataset import DistributionNodes
from diffalign.datasets.supernode_dataset import build_dataset_info
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

    # Try checkpoints/ first for stats files (without /processed suffix),
    # fall back to standard data dir via build_dataset_info
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    stats_files = ['n_counts.txt', 'atom_types.txt', 'edge_types.txt',
                   'atom_types_unnorm_mol.txt', 'edge_types_unnorm_mol.txt']
    ckpt_has_stats = all(os.path.exists(os.path.join(ckpt_dir, f)) for f in stats_files)

    if ckpt_has_stats:
        # Symlink or copy approach: build_dataset_info expects files in {dir}/processed/
        # Create a temp symlink so build_dataset_info can find them
        import tempfile
        tmp_base = tempfile.mkdtemp()
        os.symlink(ckpt_dir, os.path.join(tmp_base, 'processed'))
        # datadist_dir is relative to 2 parents up from supernode_dataset.py
        base_path = Path(os.path.realpath(__file__)).parents[1]
        rel_path = os.path.relpath(tmp_base, base_path)
        dataset_infos = build_dataset_info(cfg, datadist_dir=rel_path)
    else:
        dataset_infos = build_dataset_info(cfg)
    model_kwargs = {
        'dataset_infos': dataset_infos,
        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
        'use_data_parallel': device_count > 1,
    }
    model = DiscreteDenoisingDiffusionRxn(cfg=cfg, **model_kwargs)
    model = model.to(device)
    checkpoint_file = os.path.join(PROJECT_ROOT, 'checkpoints', f'epoch760.pt')
    print(f'loading model from {checkpoint_file}')
    state_dict = torch.load(checkpoint_file, map_location=device)['model_state_dict']
    model = load_weights(model, state_dict, device_count=1)
    return model

def _ensure_transition_model(model, cfg, diffusion_steps):
    """Swap transition_model_eval if steps differ from current."""
    T = cfg.diffusion.diffusion_steps  # 500
    assert T % diffusion_steps == 0, (
        f"diffusion_steps={diffusion_steps} must divide T={T}. "
        f"Valid values: {[d for d in range(1, T+1) if T % d == 0]}"
    )
    if diffusion_steps != cfg.diffusion.diffusion_steps_eval:
        model.transition_model_eval = _get_or_build_transition_model_eval(
            model, cfg, diffusion_steps
        )
        cfg.diffusion.diffusion_steps_eval = diffusion_steps


def _set_atom_map_from_dense(mol_obj, rdkit_to_dense, am_numbers):
    """Set RDKit atom map numbers on *mol_obj* using the dense graph's atom_map_numbers.

    Args:
        mol_obj: RDKit Mol to annotate (modified in-place).
        rdkit_to_dense: dict {rdkit_atom_idx: dense_node_idx} from
            ``get_cano_smiles_with_atom_mapping``.
        am_numbers: 1-D tensor of per-node atom-map numbers for this sample.
    """
    for rdkit_idx, dense_idx in rdkit_to_dense.items():
        am = int(am_numbers[dense_idx])
        if am > 0:
            mol_obj.GetAtomWithIdx(rdkit_idx).SetAtomMapNum(am)


def _transfer_stereo_to_reactant(rct_smiles, gen_prod_smiles, orig_product_mol, am_numbers, mol_infos):
    """Transfer stereochemistry from the original product to a generated reactant.

    Processes each reactant molecule individually to avoid atom-reordering issues
    when parsing dot-separated SMILES.

    Returns the stereo-corrected reactant SMILES, or the original if transfer fails.
    """
    try:
        gen_prod_mol = Chem.MolFromSmiles(gen_prod_smiles)
        if gen_prod_mol is None:
            return rct_smiles

        prod_info = mol_infos[-1]
        _set_atom_map_from_dense(gen_prod_mol, prod_info['atom_map'], am_numbers)

        orig_prod_copy = Chem.RWMol(Chem.MolFromSmiles(Chem.MolToSmiles(orig_product_mol)))
        match_atom_mapping_without_stereo(gen_prod_mol, orig_prod_copy)

        orig_smi = Chem.MolToSmiles(orig_prod_copy)
        has_chiral = "@" in orig_smi
        has_cistrans = "/" in orig_smi or "\\" in orig_smi

        rct_infos = mol_infos[:-1] if len(mol_infos) > 1 else mol_infos
        prod_side_ams = set(a.GetAtomMapNum() for a in gen_prod_mol.GetAtoms())

        corrected_parts = []
        for mi in rct_infos:
            rct_mol = Chem.MolFromSmiles(mi['smiles'])
            if rct_mol is None:
                corrected_parts.append(mi['smiles'])
                continue

            _set_atom_map_from_dense(rct_mol, mi['atom_map'], am_numbers)
            Chem.RemoveStereochemistry(rct_mol)

            for a in rct_mol.GetAtoms():
                if a.GetAtomMapNum() not in prod_side_ams:
                    a.ClearProp('molAtomMapNumber')

            if has_chiral:
                rct_mol = transfer_chirality_from_product_to_reactant(rct_mol, orig_prod_copy)
            if has_cistrans:
                rct_mol = transfer_bond_dir_from_product_to_reactant(rct_mol, orig_prod_copy)

            remove_atom_mapping_from_mol(rct_mol)
            corrected_parts.append(Chem.MolToSmiles(rct_mol, canonical=True))

        return '.'.join(corrected_parts)
    except Exception:
        return rct_smiles


def _decode_samples(final_samples, cfg, product_smiles=None):
    """Decode final_samples into deduplicated results with atom mappings.

    If *product_smiles* is provided and contains stereochemistry, transfers
    stereo from the original product to each generated reactant via atom
    mapping correspondence.

    Returns list of dicts: {precursors, score, sample_data, atom_mapping}
    """
    all_rxn_str, all_atom_mappings = get_cano_smiles_with_atom_mapping(
        final_samples.X, final_samples.E,
        cfg.dataset.atom_types, BOND_TYPES,
    )

    orig_product_mol = None
    has_stereo = False
    if product_smiles:
        orig_product_mol = Chem.MolFromSmiles(product_smiles)
        if orig_product_mol is not None:
            smi = Chem.MolToSmiles(orig_product_mol)
            has_stereo = ("@" in smi or "/" in smi or "\\" in smi)

    am_numbers = final_samples.atom_map_numbers

    # Deduplicate by reactant SMILES, track which sample index produced each
    reactant_info = {}  # precursors -> {count, first_sample_idx}
    for idx, rxn_str in enumerate(all_rxn_str):
        reactants = rxn_str.split('>>')[0] if '>>' in rxn_str else rxn_str
        if reactants:
            if reactants not in reactant_info:
                reactant_info[reactants] = {'count': 0, 'sample_idx': idx}
            reactant_info[reactants]['count'] += 1

    total = len(all_rxn_str) if all_rxn_str else 1
    results = []
    for precursors, info in reactant_info.items():
        sample_idx = info['sample_idx']
        sample_ph = final_samples.select_by_batch_idx(sample_idx)

        mol_infos = all_atom_mappings[sample_idx]
        reactant_mol_infos = mol_infos[:-1] if len(mol_infos) > 1 else mol_infos

        if has_stereo and am_numbers is not None:
            gen_prod = all_rxn_str[sample_idx].split('>>')[-1] if '>>' in all_rxn_str[sample_idx] else ''
            precursors = _transfer_stereo_to_reactant(
                precursors, gen_prod, orig_product_mol,
                am_numbers[sample_idx], mol_infos,
            )

        results.append({
            'precursors': precursors,
            'score': info['count'] / total,
            'sample_data': sample_ph.serialize(),
            'atom_mapping': [
                {'smiles': mi['smiles'], 'atom_map': {str(k): v for k, v in mi['atom_map'].items()}}
                for mi in reactant_mol_infos
            ],
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def predict_precursors_from_diffalign(
    product_smiles: str,
    n_precursors: int = 1,
    diffusion_steps: int = 1
):
    model, cfg = get_model_and_cfg()
    _ensure_transition_model(model, cfg, diffusion_steps)
    log.info(f"Running inference with diffusion_steps_eval={diffusion_steps}")

    dense_data = smiles_to_dense_data(
        max_nodes_more_than_product=cfg.dataset.nb_rct_dummy_nodes,
        product_smiles=product_smiles,
        atom_types=cfg.dataset.atom_types,
        bond_types=BOND_TYPES,
        with_explicit_h=cfg.dataset.with_explicit_h,
        with_formal_charge=cfg.dataset.with_formal_charge,
        add_supernode_edges=cfg.dataset.add_supernode_edges,
        canonicalize_molecule=cfg.dataset.canonicalize_molecule,
        permute_mols=cfg.dataset.permute_mols,
    )
    dense_data = dense_data.to_device(device)
    log.info('Done creating dense data.')

    with torch.inference_mode():
        final_samples = model.sample_for_condition(
            dense_data=dense_data,
            n_samples=n_precursors,
            inpaint_node_idx=None,
            inpaint_edge_idx=None,
            device=device,
        )
    log.info('Done sampling.')

    results = _decode_samples(final_samples, cfg, product_smiles=product_smiles)
    return results[:n_precursors]


def predict_with_inpainting(
    product_smiles: str,
    previous_sample_data: dict,
    inpaint_node_indices: List[int],
    n_precursors: int = 1,
    diffusion_steps: int = 1,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Re-generate precursors while keeping selected atoms fixed (inpainting).

    A sample is considered valid only if at least one of the user-requested
    change atoms (real atoms NOT in *inpaint_node_indices*) actually differs
    from the original. Samples that regenerated to identical structures on
    every change-atom are filtered out.

    Args:
        product_smiles: Target product SMILES.
        previous_sample_data: Serialized PlaceHolder from a previous prediction.
        inpaint_node_indices: Dense graph node indices to keep fixed.
        n_precursors: Number of samples to generate.
        diffusion_steps: Number of diffusion steps.

    Returns:
        Tuple (results, failure_info).
        - results: list of result dicts with precursors, score, sample_data,
          atom_mapping. Empty if every sample failed the constraint.
        - failure_info: None on success. When results is empty, a dict with:
            n_samples: how many samples were generated.
            requested_change_atoms: [{index, element}, ...] — atoms the user
              asked to change.
            stuck_atoms: [{index, element}, ...] — subset that stayed the same
              in every sample.
    """
    model, cfg = get_model_and_cfg()
    _ensure_transition_model(model, cfg, diffusion_steps)
    log.info(f"Running inpainting inference with diffusion_steps_eval={diffusion_steps}")

    dense_data = PlaceHolder.deserialize(previous_sample_data).to_device(device)

    # Auto-derive edge indices from the collapsed E tensor
    E_collapsed = dense_data.E[0] if dense_data.E.ndim >= 3 else dense_data.E
    inpaint_edge_indices = []
    for i_pos, ni in enumerate(inpaint_node_indices):
        for nj in inpaint_node_indices[i_pos + 1:]:
            if 0 <= ni < E_collapsed.shape[0] and 0 <= nj < E_collapsed.shape[1]:
                if int(E_collapsed[ni, nj]) != 0:
                    inpaint_edge_indices.append([ni, nj])

    # Snapshot the collapsed original atom types AND edges BEFORE one-hot —
    # used later for the constraint check (atom-type diff OR bond diff) and
    # failure diagnostics.
    orig_X_collapsed = (
        dense_data.X[0].clone() if dense_data.X.ndim == 2
        else dense_data.X[0].argmax(dim=-1).clone()
    )
    orig_E_collapsed = (
        dense_data.E[0].clone() if dense_data.E.ndim == 3
        else dense_data.E[0].argmax(dim=-1).clone()
    )
    node_mask_row = dense_data.node_mask[0] if dense_data.node_mask is not None else None

    # The serialized sample was collapsed (argmax'd): X is (1, n), E is (1, n, n).
    # sample_for_condition expects one-hot: X (1, n, dx), E (1, n, n, de).
    # Convert back to one-hot.
    import torch.nn.functional as F
    dx = model.Xdim_output
    de = model.Edim_output
    if dense_data.X.ndim == 2:
        dense_data.X = F.one_hot(dense_data.X.long(), num_classes=dx).float()
    if dense_data.E.ndim == 3:
        dense_data.E = F.one_hot(dense_data.E.long(), num_classes=de).float()

    # Format as list-of-lists (one per batch element)
    inpaint_node_idx = [inpaint_node_indices]
    inpaint_edge_idx = [inpaint_edge_indices] if inpaint_edge_indices else None

    with torch.inference_mode():
        final_samples = model.sample_for_condition(
            dense_data=dense_data,
            n_samples=n_precursors,
            inpaint_node_idx=inpaint_node_idx,
            inpaint_edge_idx=inpaint_edge_idx,
            device=device,
        )
    log.info('Done inpainting sampling.')

    # ── Hard constraint check: at least one change-atom must differ ────────
    # sample_for_condition can return X either one-hot (n_samples, n_nodes, dx)
    # or already-collapsed (n_samples, n_nodes). Normalize to collapsed 2D.
    if final_samples.X.ndim == 3:
        new_X_collapsed = final_samples.X.argmax(dim=-1)
    elif final_samples.X.ndim == 2:
        new_X_collapsed = final_samples.X.long()
    else:
        new_X_collapsed = final_samples.X.long().unsqueeze(0)
    n_samples_actual = int(new_X_collapsed.shape[0])

    if node_mask_row is not None:
        real_idx_set = set(torch.where(node_mask_row)[0].tolist())
    else:
        real_idx_set = set(range(int(new_X_collapsed.shape[1])))
    keep_set = {int(i) for i in inpaint_node_indices if int(i) in real_idx_set}
    change_idx_list = sorted(real_idx_set - keep_set)

    # Filter out structural placeholders. 'none', 'U', 'Au', and 'SuNo' are
    # padding / supernode / absorbing-state types — they never correspond to
    # atoms the user can see or change, so they must not contaminate either
    # the strict "all change-atoms must differ" check or the stuck-atoms
    # failure message.
    atom_decoder = cfg.dataset.atom_types
    _PLACEHOLDER_TYPES = {'none', 'U', 'Au', 'SuNo'}
    change_idx_list = [
        i for i in change_idx_list
        if atom_decoder[int(orig_X_collapsed[i])] not in _PLACEHOLDER_TYPES
    ]

    failure_info: Optional[Dict[str, Any]] = None

    if len(change_idx_list) == 0:
        # Degenerate case: user fixed every real atom. The API layer should
        # have rejected this already, but if we reach here keep all samples.
        log.warning("Inpainting called with no change-atoms; skipping constraint check.")
        filtered_samples = final_samples
    else:
        change_idx_tensor = torch.tensor(
            change_idx_list, device=new_X_collapsed.device, dtype=torch.long,
        )
        orig_at_change = orig_X_collapsed.to(new_X_collapsed.device)[change_idx_tensor]
        # (n_samples, |change|): True where the new atom TYPE differs.
        atom_diff = new_X_collapsed[:, change_idx_tensor] != orig_at_change

        # Bond-change check: for each change-atom, True if any bond incident to
        # it differs from the original. This catches the case where the sampler
        # kept the same atom type but rewired the connectivity — keeping "CCC"
        # at those indices but with different neighbors is still a real change.
        new_E = final_samples.E
        if new_E.ndim == 4:
            new_E_collapsed = new_E.argmax(dim=-1)
        elif new_E.ndim == 3:
            new_E_collapsed = new_E.long()
        else:
            new_E_collapsed = new_E.long().unsqueeze(0)
        orig_E_dev = orig_E_collapsed.to(new_E_collapsed.device)
        # (n_samples, |change|, n_nodes)
        bond_rows_diff = (
            new_E_collapsed[:, change_idx_tensor, :] != orig_E_dev[change_idx_tensor, :].unsqueeze(0)
        )
        bond_diff = bond_rows_diff.any(dim=-1)

        # An atom counts as "changed" if its type differs OR any of its bonds do.
        diff_per_atom = atom_diff | bond_diff
        # Strict constraint: every selected change-atom must differ.
        changed_per_sample = diff_per_atom.all(dim=1)          # (n_samples,)
        changed_per_atom = diff_per_atom.any(dim=0)            # (|change|,)

        n_kept = int(changed_per_sample.sum())
        log.info(
            f"Inpainting constraint: {n_kept}/{n_samples_actual} samples "
            f"changed every selected atom (type or bond)."
        )

        if n_kept == 0:
            # stuck = atoms that stayed unchanged in every sample
            stuck_positions = [
                change_idx_list[i] for i in range(len(change_idx_list))
                if not bool(changed_per_atom[i])
            ]
            stuck_atoms = [
                {'index': pos, 'element': atom_decoder[int(orig_X_collapsed[pos])]}
                for pos in stuck_positions
            ]
            requested_change_atoms = [
                {'index': pos, 'element': atom_decoder[int(orig_X_collapsed[pos])]}
                for pos in change_idx_list
            ]
            failure_info = {
                'n_samples': n_samples_actual,
                'requested_change_atoms': requested_change_atoms,
                'stuck_atoms': stuck_atoms,
            }
            return [], failure_info

        filtered_samples = final_samples.select_subset(changed_per_sample)

    results = _decode_samples(filtered_samples, cfg, product_smiles=product_smiles)
    return results[:n_precursors], None


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
