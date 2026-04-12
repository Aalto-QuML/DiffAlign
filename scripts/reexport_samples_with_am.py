"""Re-export existing eval .txt files with atom mappings included in the SMILES.

The existing eval_*.txt files contain generated reactions without atom mappings.
The corresponding samples_*.gz files contain the full graph data with atom mapping
info (mask_atom_mapping). This script uses the .gz files to regenerate SMILES
with atom mappings, then rewrites the eval .txt files preserving the ELBO ranking.

Usage:
    python scripts/reexport_samples_with_am.py \\
        --eval_dir /path/to/experiment_dir \\
        --epoch 760 --steps 100 --lambda_value 0.9 \\
        --n_samples_per_condition 100
"""
import argparse
import glob
import gzip
import os
import pickle
import re
from collections import defaultdict

import torch
from rdkit import Chem

# graph must be imported before mol to resolve the circular import through
# diffalign.utils.graph_builder -> mol
from diffalign.utils import graph as graph_mod
from diffalign.utils import mol as mol_mod
from omegaconf import OmegaConf


def canonicalize_without_am(smi):
    """Return the canonical SMILES with atom mapping numbers stripped."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        [a.ClearProp('molAtomMapNumber') for a in m.GetAtoms()]
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return smi


def canonicalize_rxn_without_am(rxn_smi):
    rcts, prods = rxn_smi.split('>>')
    return canonicalize_without_am(rcts) + '>>' + canonicalize_without_am(prods)


def read_eval_txt(path):
    """Parse eval .txt file. Returns list of (cond_idx, gt_rxn, [(smi, [elbo, loss_t, loss_0, count, prob]), ...])."""
    data = open(path, 'r').read()
    blocks = []
    pattern = re.compile(r'\(cond (\d+)\)')
    parts = pattern.split(data)
    # parts = ['', cond_idx_0, block_0, cond_idx_1, block_1, ...]
    for i in range(1, len(parts), 2):
        cond_idx = int(parts[i])
        block_text = parts[i + 1]
        lines = block_text.strip().split('\n')
        gt_rxn = lines[0].split(':')[0].strip()
        samples = []
        for line in lines[1:]:
            m = re.match(r"\t\('([^']+)', \[([^\]]+)\]\)", line)
            if m:
                rxn_smi = m.group(1)
                numbers = list(map(float, m.group(2).split(',')))
                samples.append((rxn_smi, numbers))
        blocks.append((cond_idx, gt_rxn, samples))
    return blocks


def build_am_smiles_for_gz(gz_path, atom_types, bond_types, n_samples_per_condition):
    """Load a .gz sample file and build a list per condition of
    (non_am_canonical_rxn -> am_rxn) lookup dicts."""
    with gzip.open(gz_path, 'rb') as f:
        reactions = pickle.load(f)

    # reactions = {'gen': [DataBatch, ...], 'true': [DataBatch, ...]}
    # Each DataBatch contains n_samples_per_condition samples for one condition.
    per_condition_lookups = []
    per_condition_gt = []
    for cond_idx in range(len(reactions['gen'])):
        gen_pyg = graph_mod.pyg_to_full_precision_expanded(reactions['gen'][cond_idx], atom_types=atom_types)
        true_pyg = graph_mod.pyg_to_full_precision_expanded(reactions['true'][cond_idx], atom_types=atom_types)

        gen_dense = graph_mod.to_dense(gen_pyg)
        true_dense = graph_mod.to_dense(true_pyg)

        # collapse from one-hot to indices
        gen_dense = gen_dense.mask(collapse=True)
        true_dense = true_dense.mask(collapse=True)

        # Generate AM-aware SMILES for all samples in this condition
        am_rxn_smiles = mol_mod.get_cano_smiles_from_dense(
            X=gen_dense.X, E=gen_dense.E,
            atom_types=atom_types, bond_types=bond_types,
            return_dict=False, atom_map_numbers=gen_dense.atom_map_numbers,
        )
        # Also get one GT reaction (same for all samples in a condition)
        gt_am_rxn_smiles = mol_mod.get_cano_smiles_from_dense(
            X=true_dense.X[:1], E=true_dense.E[:1],
            atom_types=atom_types, bond_types=bond_types,
            return_dict=False, atom_map_numbers=true_dense.atom_map_numbers[:1],
        )[0]

        # Build lookup: non-AM canonical form -> AM-including SMILES
        lookup = {}
        for am_rxn in am_rxn_smiles:
            key = canonicalize_rxn_without_am(am_rxn)
            if key not in lookup:  # keep first occurrence
                lookup[key] = am_rxn

        per_condition_lookups.append(lookup)
        per_condition_gt.append(gt_am_rxn_smiles)

    return per_condition_lookups, per_condition_gt


def write_eval_txt(path, blocks):
    """Write eval .txt file in the same format as save_samples_to_file."""
    with open(path, 'w') as f:
        for cond_idx, gt_rxn, samples in blocks:
            f.write(f'(cond {cond_idx}) {gt_rxn}:\n')
            for rxn_smi, numbers in samples:
                numbers_str = ', '.join(str(n) for n in numbers)
                f.write(f"\t('{rxn_smi}', [{numbers_str}])\n")


def main():
    parser = argparse.ArgumentParser(description="Re-export eval .txt files with atom mappings.")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lambda_value", type=float, default=0.9)
    parser.add_argument("--n_samples_per_condition", type=int, default=100)
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the experiment config YAML (for atom_types / bond_types).")
    parser.add_argument("--suffix", type=str, default="_withAM",
                        help="Suffix to add to output filenames (so originals aren't overwritten).")
    args = parser.parse_args()

    # Load config to get atom_types and bond_types
    cfg = OmegaConf.load(args.config_path)
    atom_types = list(cfg.dataset.atom_types)
    # bond_types are defined in graph.py (constant)
    bond_types = graph_mod.bond_types

    # Find all samples .gz files
    gz_pattern = os.path.join(args.eval_dir, f"samples_epoch{args.epoch}_steps{args.steps}_cond*_sampercond{args.n_samples_per_condition}_s*.gz")
    gz_files = sorted(glob.glob(gz_pattern), key=lambda p: int(re.search(r'_s(\d+)\.gz$', p).group(1)))
    if not gz_files:
        raise FileNotFoundError(f"No .gz files found matching: {gz_pattern}")

    print(f"Found {len(gz_files)} .gz sample files")

    for gz_path in gz_files:
        start_idx = int(re.search(r'_s(\d+)\.gz$', gz_path).group(1))
        print(f"\nProcessing chunk s{start_idx}...")

        # Corresponding eval .txt file (resorted version, to reuse weighted_prob ranking)
        eval_txt = os.path.join(args.eval_dir, f"eval_epoch{args.epoch}_steps{args.steps}_resorted_{args.lambda_value}_s{start_idx}.txt")
        if not os.path.exists(eval_txt):
            print(f"  WARNING: missing {eval_txt}, skipping")
            continue

        # Build AM lookup from .gz
        lookups, gt_rxns = build_am_smiles_for_gz(gz_path, atom_types, bond_types, args.n_samples_per_condition)
        print(f"  Built lookups for {len(lookups)} conditions from .gz")

        # Read eval text file
        blocks = read_eval_txt(eval_txt)
        print(f"  Read {len(blocks)} condition blocks from eval .txt")

        if len(blocks) != len(lookups):
            # Eval files may have duplicates (append mode). Handle gracefully.
            print(f"  Note: eval .txt has {len(blocks)} blocks, .gz has {len(lookups)}. "
                  f"Processing first {min(len(blocks), len(lookups))}.")

        new_blocks = []
        rewritten = 0
        total = 0
        for i, (cond_idx, gt_rxn, samples) in enumerate(blocks):
            if i >= len(lookups):
                break
            lookup = lookups[i]
            gt_am = gt_rxns[i]
            new_samples = []
            for rxn_smi, numbers in samples:
                total += 1
                key = canonicalize_rxn_without_am(rxn_smi)
                if key in lookup:
                    new_samples.append((lookup[key], numbers))
                    rewritten += 1
                else:
                    new_samples.append((rxn_smi, numbers))
            new_blocks.append((cond_idx, gt_am, new_samples))

        # Write new eval .txt file
        out_path = eval_txt.replace('.txt', f'{args.suffix}.txt')
        write_eval_txt(out_path, new_blocks)
        print(f"  Wrote {out_path} ({rewritten}/{total} samples rewritten with AM)")

    print("\nDone.")


if __name__ == "__main__":
    main()
