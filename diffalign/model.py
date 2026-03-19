"""Syntheseus adapter for DiffAlign retrosynthesis model."""

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_backwards

from diffalign.inference import (
    BOND_TYPES,
    PROJECT_ROOT,
    _get_or_build_transition_model_eval,
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
        *args, **kwargs: Forwarded to ``ExternalBackwardReactionModel`` (accepts
            ``model_dir``, ``device``, ``remove_duplicates``, ``use_cache``,
            ``default_num_results``, etc.).
    """

    def __init__(
        self,
        *args,
        diffusion_steps: int = 1,
        samples_per_product: int = 100,
        **kwargs,
    ):
        self.diffusion_steps = diffusion_steps
        self.samples_per_product = samples_per_product
        super().__init__(*args, **kwargs)

        # Load model and Hydra config via existing caching mechanism
        self._model, self._cfg = get_model_and_cfg()
        self._transition_cache: Dict[int, object] = {}

    @property
    def name(self) -> str:
        return "DiffAlign"

    def get_default_model_dir(self) -> Path:
        return PROJECT_ROOT / "checkpoints"

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        results = []
        n_samples = max(num_results, self.samples_per_product)

        for mol in inputs:
            product_smiles = mol.smiles
            cfg = self._cfg

            # Encode product as dense graph data
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

            # Ensure transition model matches requested diffusion steps
            T = cfg.diffusion.diffusion_steps
            assert T % self.diffusion_steps == 0, (
                f"diffusion_steps={self.diffusion_steps} must divide T={T}. "
                f"Valid values: {[d for d in range(1, T + 1) if T % d == 0]}"
            )
            if self.diffusion_steps != cfg.diffusion.diffusion_steps_eval:
                self._model.transition_model_eval = _get_or_build_transition_model_eval(
                    self._model, cfg, self.diffusion_steps
                )
                cfg.diffusion.diffusion_steps_eval = self.diffusion_steps

            # Run reverse diffusion
            with torch.inference_mode():
                final_samples = self._model.sample_for_condition(
                    dense_data=dense_data,
                    n_samples=n_samples,
                    inpaint_node_idx=None,
                    inpaint_edge_idx=None,
                    device=self.device,
                )

            # Convert dense tensors back to reaction SMILES
            all_rxn_str = get_cano_smiles_from_dense(
                final_samples.X, final_samples.E,
                cfg.dataset.atom_types, BOND_TYPES,
            )

            # Extract reactant SMILES, deduplicate, and score by frequency
            reactant_counts: Dict[str, int] = {}
            for rxn_str in all_rxn_str:
                reactants = rxn_str.split(">>")[0] if ">>" in rxn_str else rxn_str
                if reactants:
                    reactant_counts[reactants] = reactant_counts.get(reactants, 0) + 1

            total = len(all_rxn_str) if all_rxn_str else 1
            scored = sorted(reactant_counts.items(), key=lambda x: x[1], reverse=True)
            top = scored[:num_results]

            # Convert to SingleProductReaction via syntheseus utility
            output_list = [smiles for smiles, _ in top]
            metadata_list = [{"probability": count / total} for _, count in top]
            reactions = process_raw_smiles_outputs_backwards(
                input=mol, output_list=output_list, metadata_list=metadata_list
            )
            results.append(reactions)

        return results

    def get_parameters(self):
        return self._model.parameters()
