"""Recompute top-k accuracy with stereochemistry transferred from products to reactants.

This script takes:
  1. The raw test CSV (with original stereochemistry and atom mappings)
  2. The evaluation output text files (resorted, with atom mappings but no stereochemistry)
And recomputes top-k by transferring stereochemistry from the product to the generated
reactants via atom mapping, then comparing against the stereo ground truth.

Self-contained: only depends on rdkit, numpy, tqdm. Does not import from diffalign.

Usage:
    python scripts/transfer_stereo.py \\
        --raw_data_path /path/to/test.csv \\
        --eval_dir /path/to/experiment_dir \\
        --epoch 760 --steps 100 --lambda_value 0.9 \\
        --topks 1 3 5 10 50 100
"""
import argparse
import glob
import os
import re
import copy
import logging
import multiprocessing
from functools import partial
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem

log = logging.getLogger(__name__)


# --- Inlined from diffalign/utils/io_utils.py ---

def read_saved_reaction_data(data):
    blocks = re.split(r'\(cond \d+\)', data)[1:]
    reactions = []
    for block in blocks:
        lines = block.strip().split('\n')
        # Use rsplit because atom-mapped SMILES contain `:` characters (e.g. [CH2:1]).
        # The GT reaction line ends with `:` as a trailing delimiter.
        original_reaction = lines[0].rsplit(':', 1)[0].strip()
        generated_reactions = []
        for line in lines[1:]:
            match = re.match(r"\t\('([^']+)', \[([^\]]+)\]\)", line)
            if match:
                reaction_smiles = match.group(1)
                numbers = list(map(float, match.group(2).split(',')))
                generated_reactions.append((reaction_smiles, numbers))
        reactions.append((original_reaction, generated_reactions))
    return reactions


# --- Inlined stereo helpers from diffalign/utils/mol.py ---

def are_same_cycle(l1, l2):
    l1 = l1 * 2
    for i in range(len(l1) - 2):
        if l1[i] == l2[0] and l1[i + 1] == l2[1] and l1[i + 2] == l2[2]:
            return True
    return False


def order_of_pair_of_indices_in_cycle(idx1, idx2, cycle):
    cycle = cycle * 2
    for i in range(len(cycle * 2) - 1):
        if cycle[i] == idx1 and cycle[i + 1] == idx2:
            return (idx1, idx2)
        if cycle[i] == idx2 and cycle[i + 1] == idx1:
            return (idx2, idx1)
    raise ValueError("The indices are not in the cycle")


def get_opposite_chiral_tag(atom):
    if atom == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CW
    elif atom == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    return None


def switch_between_bond_cw_ccw_label_and_cip_based_label(atom, cw_ccw_label, cip_ranking):
    if cw_ccw_label == Chem.ChiralType.CHI_UNSPECIFIED:
        return Chem.ChiralType.CHI_UNSPECIFIED
    nbrs = [(x.GetOtherAtomIdx(atom.GetIdx()), x.GetIdx()) for x in atom.GetBonds()]
    s_nbrs = sorted(nbrs, key=lambda x: cip_ranking[x[0]])
    if len(nbrs) == 3:
        order_based_on_bonds = [x.GetOtherAtomIdx(atom.GetIdx()) for x in atom.GetBonds()]
        order_based_on_bonds_with_cip = [(idx, cip_ranking[idx]) for idx in order_based_on_bonds]
        order_based_on_cip = list(map(lambda x: x[0], sorted(order_based_on_bonds_with_cip, key=lambda x: x[1])))
        if are_same_cycle(order_based_on_bonds, order_based_on_cip):
            return cw_ccw_label
        return get_opposite_chiral_tag(cw_ccw_label)
    elif len(nbrs) == 4:
        leading_bond_order_representation = nbrs[0]
        leading_atom_representation = s_nbrs[0]
        if leading_bond_order_representation == leading_atom_representation:
            order_based_on_bonds = [x.GetOtherAtomIdx(atom.GetIdx()) for x in atom.GetBonds()][1:]
            order_based_on_bonds_with_cip = [(idx, cip_ranking[idx]) for idx in order_based_on_bonds]
            order_based_on_cip = list(map(lambda x: x[0], sorted(order_based_on_bonds_with_cip, key=lambda x: x[1])))
            if are_same_cycle(order_based_on_bonds, order_based_on_cip):
                return cw_ccw_label
            return get_opposite_chiral_tag(cw_ccw_label)
        else:
            remaining_neighbor_indices_bond_order_based = [x[0] for x in nbrs[1:]]
            remaining_neighbor_indices_cip_based = [x[0] for x in s_nbrs[1:]]
            remaining_two_atoms = [x for x in remaining_neighbor_indices_bond_order_based if x in remaining_neighbor_indices_cip_based]
            order_of_remaining_pair_bond_order_based = order_of_pair_of_indices_in_cycle(
                remaining_two_atoms[0], remaining_two_atoms[1], remaining_neighbor_indices_bond_order_based)
            order_of_remaining_pair_atom_index_based = order_of_pair_of_indices_in_cycle(
                remaining_two_atoms[0], remaining_two_atoms[1], remaining_neighbor_indices_cip_based)
            if order_of_remaining_pair_bond_order_based == order_of_remaining_pair_atom_index_based:
                return get_opposite_chiral_tag(cw_ccw_label)
            else:
                return cw_ccw_label
    else:
        return Chem.ChiralType.CHI_UNSPECIFIED


def match_atom_mapping(mol1, mol2):
    match = mol2.GetSubstructMatch(mol1)
    for idx1, idx2 in enumerate(match):
        if idx2 >= 0:
            mol2.GetAtomWithIdx(idx2).SetAtomMapNum(
                mol1.GetAtomWithIdx(idx1).GetAtomMapNum()
            )


def match_atom_mapping_without_stereo(mol1, mol2):
    mol2_copy = copy.deepcopy(mol2)
    Chem.RemoveStereochemistry(mol2_copy)
    match_atom_mapping(mol1, mol2_copy)
    for atom2_copy, atom2 in zip(mol2_copy.GetAtoms(), mol2.GetAtoms()):
        atom2.SetAtomMapNum(atom2_copy.GetAtomMapNum())


def transfer_bond_dir_from_product_to_reactant(r_mol, p_mol):
    r_mol_new = copy.deepcopy(r_mol)
    am_pair_to_bonddir = {}
    for bond in p_mol.GetBonds():
        if bond.GetBondDir() != Chem.BondDir.NONE:
            am_pair_to_bonddir[
                (bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())
            ] = bond.GetBondDir()

    for bond in r_mol_new.GetBonds():
        key_fwd = (bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())
        key_rev = (bond.GetEndAtom().GetAtomMapNum(), bond.GetBeginAtom().GetAtomMapNum())
        if key_fwd in am_pair_to_bonddir:
            bond.SetBondDir(am_pair_to_bonddir[key_fwd])
        elif key_rev in am_pair_to_bonddir:
            bond.SetBondDir(am_pair_to_bonddir[key_rev])
    return r_mol_new


def transfer_chirality_from_product_to_reactant(r_mol, p_mol):
    r_mol_new = copy.deepcopy(r_mol)
    am_to_am_of_neighbors_prod = {}
    am_to_chiral_tag = {}
    for atom in p_mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            nbrs = [(x.GetOtherAtomIdx(atom.GetIdx()), x.GetIdx()) for x in atom.GetBonds()]
            atom_maps_of_neighbors = [p_mol.GetAtomWithIdx(x[0]).GetAtomMapNum() for x in nbrs]
            am_to_am_of_neighbors_prod[atom.GetAtomMapNum()] = atom_maps_of_neighbors
            am_to_chiral_tag[atom.GetAtomMapNum()] = atom.GetChiralTag()

    for atom in r_mol_new.GetAtoms():
        if atom.GetAtomMapNum() in am_to_am_of_neighbors_prod:
            am_nbrs_prod = am_to_am_of_neighbors_prod[atom.GetAtomMapNum()]
            nbrs = [x.GetOtherAtomIdx(atom.GetIdx()) for x in atom.GetBonds()]
            am_to_rank_prod = {am: idx for idx, am in enumerate(am_nbrs_prod)}
            try:
                nbr_idx_to_rank_prod = {
                    nbr_idx: am_to_rank_prod[r_mol.GetAtomWithIdx(nbr_idx).GetAtomMapNum()]
                    for nbr_idx in nbrs
                }
                cw_ccw = switch_between_bond_cw_ccw_label_and_cip_based_label(
                    atom, am_to_chiral_tag[atom.GetAtomMapNum()], nbr_idx_to_rank_prod
                )
            except:
                cw_ccw = Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            atom.SetChiralTag(cw_ccw)
    return r_mol_new


def remove_atom_mapping_from_mol(mol):
    [a.ClearProp("molAtomMapNumber") for a in mol.GetAtoms()]
    return mol


# --- Main script logic ---

def remove_atom_mapping_and_stereo(smi):
    m = Chem.MolFromSmiles(smi)
    [a.ClearProp('molAtomMapNumber') for a in m.GetAtoms()]
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, canonical=True)


def remove_atom_mapping(smi):
    m = Chem.MolFromSmiles(smi)
    [a.ClearProp('molAtomMapNumber') for a in m.GetAtoms()]
    return Chem.MolToSmiles(m, canonical=True)


def undo_kekulize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=True)


def get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_stereo):
    for true_rxn_without_stereo, true_rxn_with_stereo in zip(true_rxns_without_stereo, true_rxns_with_stereo):
        if old_rxn == true_rxn_without_stereo:
            return true_rxn_with_stereo
    return None


def match_old_rxns(old_rxn, true_rxns_without_stereo, true_rxns_with_stereo, true_rxns_with_am_and_stereo):
    return (
        get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_stereo),
        get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_am_and_stereo)
    )


def group_indices(strings):
    indices = defaultdict(list)
    for index, string in enumerate(strings):
        indices[string].append(index)
    return indices


counter = None
lock = None

def init_globals(c, l):
    global counter, lock
    counter = c
    lock = l

def update_progress():
    global counter, lock
    with lock:
        counter.value += 1
        return counter.value


def process_reaction(i, matched_database_rxns_with_am, matched_database_rxns, sampled_rxns, topks):
    matched_rxn_with_am = matched_database_rxns_with_am[i]
    matched_rxn = matched_database_rxns[i]
    samples = sampled_rxns[i]

    chiral_reactions = 0
    cistrans_reactions = 0
    topk_local = {k: 0 for k in topks}
    topk_among_chiral_local = {k: 0 for k in topks}
    topk_among_cistrans_local = {k: 0 for k in topks}

    if matched_rxn is not None:
        matched_prod_mol = Chem.MolFromSmiles(matched_rxn_with_am.split('>>')[1])
        samples_with_chirality = []
        for j in range(len(samples)):
            prod_smi_without_am = matched_rxn.split('>>')[1]
            if "@" in matched_rxn_with_am or "/" in matched_rxn_with_am or "\\" in matched_rxn_with_am:
                sample = samples[j]
                try:
                    prod_side_ams = set([a.GetAtomMapNum() for a in Chem.MolFromSmiles(sample.split('>>')[1]).GetAtoms()])
                    sample_rct_mol = Chem.MolFromSmiles(sample.split('>>')[0])
                    sample_prod_mol = Chem.MolFromSmiles(sample.split('>>')[1])
                    if sample_rct_mol is not None:
                        Chem.RemoveStereochemistry(sample_rct_mol)
                        for a in sample_rct_mol.GetAtoms():
                            if a.GetAtomMapNum() not in prod_side_ams:
                                a.ClearProp('molAtomMapNumber')
                        match_atom_mapping_without_stereo(sample_prod_mol, matched_prod_mol)
                        if "@" in matched_rxn_with_am:
                            sample_rct_mol = transfer_chirality_from_product_to_reactant(sample_rct_mol, matched_prod_mol)
                        if "/" in matched_rxn_with_am or "\\" in matched_rxn_with_am:
                            sample_rct_mol = transfer_bond_dir_from_product_to_reactant(sample_rct_mol, matched_prod_mol)
                        remove_atom_mapping_from_mol(sample_rct_mol)
                        r_smiles = Chem.MolToSmiles(sample_rct_mol, canonical=True)
                    else:
                        r_smiles = ""
                except Exception:
                    r_smiles = sample.split('>>')[0]
            else:
                try:
                    r_mol = Chem.MolFromSmiles(samples[j].split('>>')[0])
                    [r_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(r_mol.GetNumAtoms())]
                    r_smiles = Chem.MolToSmiles(r_mol, canonical=True)
                except:
                    r_smiles = samples[j].split('>>')[0]

            samples_with_chirality.append(r_smiles + ">>" + prod_smi_without_am)

        if "@" in matched_rxn:
            chiral_reactions = 1
            for k in topks:
                topk_among_chiral_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])
        if "/" in matched_rxn or "\\" in matched_rxn:
            cistrans_reactions = 1
            for k in topks:
                topk_among_cistrans_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])

        for k in topks:
            topk_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])

    progress = update_progress()
    return topk_local, topk_among_chiral_local, topk_among_cistrans_local, chiral_reactions, cistrans_reactions, progress


def main():
    parser = argparse.ArgumentParser(description="Recompute top-k with stereochemistry transfer.")
    parser.add_argument("--raw_data_path", type=str, required=True,
                        help="Path to the raw test CSV with stereochemistry and atom mappings.")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Path to the experiment directory containing eval_*.txt files.")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lambda_value", type=float, default=0.9)
    parser.add_argument("--topks", type=int, nargs='+', default=[1, 3, 5, 10, 50, 100])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix for input eval files (e.g. '_withAM' to use files with atom mappings).")
    args = parser.parse_args()

    # Match files ending in `_s<digits>{suffix}.txt`. Using regex because globs can't
    # distinguish _s0.txt from _s0_withAM.txt when suffix is empty.
    all_candidates = sorted(glob.glob(os.path.join(
        args.eval_dir, f"eval_epoch{args.epoch}_steps{args.steps}_resorted_{args.lambda_value}_s*.txt"
    )))
    file_re = re.compile(rf"_s\d+{re.escape(args.suffix)}\.txt$")
    eval_files = [f for f in all_candidates if file_re.search(f)]
    if not eval_files:
        raise FileNotFoundError(f"No eval files found for suffix={args.suffix!r} in {args.eval_dir}")
    print(f"Found {len(eval_files)} eval files (suffix={args.suffix!r})")

    all_eval_data = ""
    for f in eval_files:
        all_eval_data += open(f, 'r').read()

    sampled_rxns_blocks = read_saved_reaction_data(all_eval_data)
    print(f"Total conditions (before dedup): {len(sampled_rxns_blocks)}")

    # Dedupe by ground truth reaction — eval .txt files are written in append mode,
    # so re-running a chunk produces duplicates. Keep the first occurrence.
    seen = set()
    deduped = []
    for block in sampled_rxns_blocks:
        gt = block[0]
        if gt not in seen:
            seen.add(gt)
            deduped.append(block)
    sampled_rxns_blocks = deduped
    print(f"Total conditions (after dedup): {len(sampled_rxns_blocks)}")

    print(f"Reading raw data from {args.raw_data_path}")
    raw_true_rxns = open(args.raw_data_path, 'r').readlines()
    raw_true_rxns = [r.strip() for r in raw_true_rxns if r.strip()]

    database_rxns_with_am_and_stereo = []
    database_rxns_with_stereo = []
    database_rxns_without_stereo = []
    for rxn in raw_true_rxns:
        reactants = rxn.split('>>')[0]
        products = rxn.split('>>')[1]
        database_rxns_with_am_and_stereo.append(rxn)
        database_rxns_with_stereo.append(remove_atom_mapping(reactants) + '>>' + remove_atom_mapping(products))
        database_rxns_without_stereo.append(remove_atom_mapping_and_stereo(reactants) + '>>' + remove_atom_mapping_and_stereo(products))

    raw_old_true_rxns = [sample[0] for sample in sampled_rxns_blocks]
    old_true_rxns = []
    for rxn in raw_old_true_rxns:
        reactants = rxn.split('>>')[0]
        products = rxn.split('>>')[1]
        old_true_rxns.append(remove_atom_mapping(reactants) + '>>' + remove_atom_mapping(products))

    raw_sampled_rxns = [sample[1] for sample in sampled_rxns_blocks]
    sampled_rxns = []
    for sample in raw_sampled_rxns:
        sampled_rxns_per_true_rxn = []
        for sample_info in sample:
            rxn = sample_info[0]
            reactants = rxn.split('>>')[0]
            products = rxn.split('>>')[1]
            sampled_rxns_per_true_rxn.append(undo_kekulize(reactants) + '>>' + undo_kekulize(products))

        rxn_indices_grouped = group_indices(sampled_rxns_per_true_rxn)
        new_counts = {}
        new_elbos = {}
        for rxn, indices in rxn_indices_grouped.items():
            counts = sum([int(sample[i][1][-2]) for i in indices])
            elbos = sum([float(sample[i][1][0]) for i in indices]) / len(indices)
            new_elbos[rxn] = elbos
            new_counts[rxn] = counts

        sum_exp_elbo = sum(np.exp(-elbo) for elbo in new_elbos.values())
        sum_counts = sum(new_counts.values())
        new_weighted_probs = {}
        for rxn in sampled_rxns_per_true_rxn:
            exp_elbo = np.exp(-new_elbos[rxn])
            weighted_prob = (exp_elbo / sum_exp_elbo) * args.lambda_value + (new_counts[rxn] / sum_counts) * (1 - args.lambda_value)
            new_weighted_probs[rxn] = weighted_prob

        new_sampled_rxns_per_true_rxn = sorted(list(set(sampled_rxns_per_true_rxn)), key=lambda x: new_weighted_probs[x], reverse=True)
        sampled_rxns.append(new_sampled_rxns_per_true_rxn)

    print("Matching generated ground-truth to database reactions...")
    partial_match = partial(
        match_old_rxns,
        true_rxns_without_stereo=database_rxns_without_stereo,
        true_rxns_with_stereo=database_rxns_with_stereo,
        true_rxns_with_am_and_stereo=database_rxns_with_am_and_stereo
    )
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = pool.map(partial_match, old_true_rxns)

    gen_index_to_database_rxn = []
    gen_index_to_database_rxn_with_am = []
    for matched_rxn, matched_rxn_with_am in results:
        gen_index_to_database_rxn.append(matched_rxn)
        gen_index_to_database_rxn_with_am.append(matched_rxn_with_am)

    n_unmatched = sum(1 for x in gen_index_to_database_rxn if x is None)
    if n_unmatched > 0:
        print(f"Warning: {n_unmatched} reactions could not be matched to database.")

    print("Computing top-k with stereochemistry transfer...")
    manager = multiprocessing.Manager()
    counter_obj = manager.Value('i', 0)
    lock_obj = manager.Lock()

    partial_process = partial(
        process_reaction,
        matched_database_rxns_with_am=gen_index_to_database_rxn_with_am,
        matched_database_rxns=gen_index_to_database_rxn,
        sampled_rxns=sampled_rxns,
        topks=args.topks
    )

    total_tasks = len(gen_index_to_database_rxn)
    topk = {k: 0 for k in args.topks}
    topk_among_chiral = {k: 0 for k in args.topks}
    topk_among_cistrans = {k: 0 for k in args.topks}
    total_chiral_reactions = 0
    total_cistrans_reactions = 0

    with multiprocessing.Pool(processes=args.num_workers, initializer=init_globals, initargs=(counter_obj, lock_obj)) as pool:
        all_results = []
        with tqdm(total=total_tasks) as pbar:
            for result in pool.imap(partial_process, range(total_tasks)):
                all_results.append(result[:5])
                pbar.update(result[-1] - pbar.n)

    for topk_local, topk_chiral_local, topk_cistrans_local, chiral, cistrans in all_results:
        for k in args.topks:
            topk[k] += topk_local[k]
            topk_among_chiral[k] += topk_chiral_local[k]
            topk_among_cistrans[k] += topk_cistrans_local[k]
        total_chiral_reactions += chiral
        total_cistrans_reactions += cistrans

    n = len(sampled_rxns)
    topk_norm = {k: v / n for k, v in topk.items()}
    # Also compute MRR
    mrr_sum = 0.0
    for topk_local, _, _, _, _ in all_results:
        for k in sorted(args.topks):
            if topk_local[k] > 0:
                mrr_sum += 1.0 / k
                break
    mrr_approx = mrr_sum / n

    print(f"\n=== Top-K Accuracy (with stereochemistry, n={n}) ===")
    for k in args.topks:
        print(f"  top-{k:3d}: {topk_norm[k]:.4f}  ({topk[k]}/{n})")
    print(f"  MRR (approx, from topks): {mrr_approx:.4f}")

    if total_chiral_reactions > 0:
        topk_chiral_norm = {k: v / total_chiral_reactions for k, v in topk_among_chiral.items()}
        print(f"\n=== Top-K among chiral reactions (n={total_chiral_reactions}) ===")
        for k in args.topks:
            print(f"  top-{k:3d}: {topk_chiral_norm[k]:.4f}")

    if total_cistrans_reactions > 0:
        topk_cistrans_norm = {k: v / total_cistrans_reactions for k, v in topk_among_cistrans.items()}
        print(f"\n=== Top-K among cis/trans reactions (n={total_cistrans_reactions}) ===")
        for k in args.topks:
            print(f"  top-{k:3d}: {topk_cistrans_norm[k]:.4f}")


if __name__ == "__main__":
    main()
