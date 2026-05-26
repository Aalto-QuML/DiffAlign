"""Syntheseus adapter for DiffAlign retrosynthesis model."""

import logging
from pathlib import Path
from typing import List, Sequence

import torch

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_backwards

from diffalign.inference import (
    BOND_TYPES,
    PROJECT_ROOT,
    _decode_samples,
    _ensure_transition_model,
    get_model_and_cfg,
    smiles_to_dense_data,
)
from diffalign.utils.mol import get_cano_smiles_from_dense

log = logging.getLogger(__name__)


class DiffAlignModel(ExternalBackwardReactionModel):
    """Syntheseus wrapper around the DiffAlign graph-diffusion retrosynthesis model.

    This adapter reuses the existing inference functions from ``inference.py`` and
    exposes them through the standard ``BackwardReactionModel`` interface so that
    DiffAlign can participate in Syntheseus retrosynthesis planning.

    Args:
        diffusion_steps: Number of reverse-diffusion steps at inference time.
            Must divide the training diffusion steps (500). Fewer steps = faster
            but lower quality.
        samples_per_product: Minimum number of stochastic samples to draw per
            product molecule. The model deduplicates by frequency, so more
            samples yield better coverage and score estimates.
        rich_metadata: When True (default, used by the RxnLab web app), decode via
            ``_decode_samples`` and attach ``sample_data``/``atom_mapping``/
            ``mapped_rxn`` to each reaction's metadata. When False (multi-step
            *search*, which only needs reactant SMILES + score), use the cheaper
            frequency decode that skips atom-mapping, mapped-rxn assembly, and
            stereo transfer.
        *args, **kwargs: Forwarded to ``ExternalBackwardReactionModel`` (accepts
            ``model_dir``, ``device``, ``remove_duplicates``, ``use_cache``,
            ``default_num_results``, etc.).
    """

    def __init__(
        self,
        *args,
        diffusion_steps: int = 1,
        samples_per_product: int = 100,
        rich_metadata: bool = True,
        **kwargs,
    ):
        self.diffusion_steps = diffusion_steps
        self.samples_per_product = samples_per_product
        self.rich_metadata = rich_metadata
        super().__init__(*args, **kwargs)

        # Load model and Hydra config via existing caching mechanism
        self._model, self._cfg = get_model_and_cfg()

    @property
    def name(self) -> str:
        return "DiffAlign"

    def get_default_model_dir(self) -> Path:
        return PROJECT_ROOT / "checkpoints"

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        """Run reverse diffusion and decode via the SAME pipeline as the native
        ``inference.predict_precursors`` path (``_decode_samples``), so the web app
        gets byte-identical predictions/scores when routed through this wrapper.

        The rich per-prediction fields the app depends on — ``score``,
        ``sample_data`` (for inpainting), ``atom_mapping`` and ``mapped_rxn`` (for
        stereo display + CSV export) — are carried on each reaction's ``metadata``.
        ``metadata`` is declared ``compare=False`` on syntheseus' ``Reaction``, so
        these extra payloads never affect dedup/equality.
        """
        results = []
        n_samples = max(num_results, self.samples_per_product)

        for mol in inputs:
            product_smiles = mol.smiles
            cfg = self._cfg

            _ensure_transition_model(self._model, cfg, self.diffusion_steps)

            dense_data = smiles_to_dense_data(
                product_smiles=product_smiles,
                max_nodes_more_than_product=cfg.dataset.nb_rct_dummy_nodes,
                atom_types=cfg.dataset.atom_types,
                bond_types=BOND_TYPES,
                with_explicit_h=cfg.dataset.with_explicit_h,
                with_formal_charge=cfg.dataset.with_formal_charge,
                add_supernode_edges=cfg.dataset.add_supernode_edges,
                canonicalize_molecule=cfg.dataset.canonicalize_molecule,
                permute_mols=cfg.dataset.permute_mols,
            )
            dense_data = dense_data.to_device(self.device)

            with torch.inference_mode():
                final_samples = self._model.sample_for_condition(
                    dense_data=dense_data,
                    n_samples=n_samples,
                    inpaint_node_idx=None,
                    inpaint_edge_idx=None,
                    device=self.device,
                )

            if self.rich_metadata:
                output_list, metadata_list = self._decode_rich(
                    final_samples, cfg, product_smiles, num_results
                )
            else:
                output_list, metadata_list = self._decode_lean(
                    final_samples, cfg, num_results
                )

            reactions = process_raw_smiles_outputs_backwards(
                input=mol, output_list=output_list, metadata_list=metadata_list
            )
            results.append(reactions)

        return results

    def _decode_rich(self, final_samples, cfg, product_smiles, num_results):
        """Full decode (web path): same pipeline as native ``predict_precursors``."""
        decoded = _decode_samples(final_samples, cfg, product_smiles=product_smiles)[:num_results]
        output_list = [d["precursors"] for d in decoded]
        metadata_list = [
            {
                "probability": d["score"],
                "score": d["score"],
                "precursors": d["precursors"],
                "sample_data": d.get("sample_data"),
                "atom_mapping": d.get("atom_mapping"),
                "mapped_rxn": d.get("mapped_rxn"),
            }
            for d in decoded
        ]
        return output_list, metadata_list

    def _decode_lean(self, final_samples, cfg, num_results):
        """Cheap decode (search path): reactant SMILES + frequency score only.
        Skips atom-mapping, mapped-rxn assembly, and stereo transfer."""
        all_rxn_str = get_cano_smiles_from_dense(
            final_samples.X, final_samples.E, cfg.dataset.atom_types, BOND_TYPES,
        )
        counts: dict = {}
        for rxn_str in all_rxn_str:
            reactants = rxn_str.split(">>")[0] if ">>" in rxn_str else rxn_str
            if reactants:
                counts[reactants] = counts.get(reactants, 0) + 1
        total = len(all_rxn_str) if all_rxn_str else 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:num_results]
        output_list = [smiles for smiles, _ in top]
        metadata_list = [{"probability": c / total, "score": c / total} for _, c in top]
        return output_list, metadata_list

    def get_parameters(self):
        return self._model.parameters()
