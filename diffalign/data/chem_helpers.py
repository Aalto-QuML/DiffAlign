'''
    Helper functions for chemical data.
'''

import copy
import logging

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from torch_geometric.data import Data

log = logging.getLogger(__name__)




def order_of_pair_of_indices_in_cycle(idx1, idx2, cycle):
    """Inputs:
    idx1, idx2: indices of atoms that should be present in the list 'cycle'
    cycle: list of 3 atom indices that are interpreted to be cyclical, e.g., [2,3,1,2,3,1,...]
    Outputs:
    A tuple consisting of idx1 and idx2, where the order is is the one in which they appear next to each other in the extended cycle
    (e.g., idx1=2, idx2=1, cycle=[2,3,1] -> (1,2))
    """
    cycle = cycle * 2
    for i in range(len(cycle * 2) - 1):
        if cycle[i] == idx1 and cycle[i + 1] == idx2:
            return (idx1, idx2)
        if cycle[i] == idx2 and cycle[i + 1] == idx1:
            return (idx2, idx1)
    raise ValueError("The indices are not in the cycle")

def get_opposite_chiral_tag(atom):
    '''
        Get the opposite chiral tag.
    '''
    if atom == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CW
    elif atom == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    return None

def are_same_cycle(l1, l2):
    """Inputs:
    l1, l2: Lists of indices of length 3
    Outputs:
    True if the two lists follow the same extended cycle, False otherwise.
    'cycle' meaning, e.g., l1=(2,1,3) -> (...,2,3,1,2,3,1,...)
    """
    # Assuming both l1 and l2 are lists of atom indices of length 3
    l1 = l1 * 2
    for i in range(len(l1) - 2):
        if l1[i] == l2[0] and l1[i + 1] == l2[1] and l1[i + 2] == l2[2]:
            return True
    return False

def switch_between_bond_cw_ccw_label_and_cip_based_label(
    atom, cw_ccw_label, cip_ranking
):
    """
    Inputs:
    atom: RDKit atom object
    cw_ccw_label: Chem.ChiralType.CHI_TETRAHEDRAL_CCW or Chem.ChiralType.CHI_TETRAHEDRAL_CW, defining the chirality of the atom either in the bond-index based representation or the cip-based representation
    cip_ranking: A list of the CIP rankings of the atoms in the molecule (integers)
    Outputs:
    A transformed CW/CCW label. If CW/CCW was originally bond-based, the output is the corresponding CIP-based label, and vice versa.

    Uses the fact that rdkit assigns the CW/CCW label based on the order the bonds are attached to the atom in the atom.GetBonds() data structure. The function transforms this to a label
    that is based on the CIP ranking of the atoms attached to the atom, which does not depend on the particular data structures in the molecule object. This way, it can be used as a
    'canonical' representation for the chirality of the atom. Assumes that cip_ranking is unique for all the atoms in the molecule (note: there are many rdkit methods to get the CIP ranking,
    some do not guarantee uniqueness. Our get_cip_ranking accomplishes this).

    The convention (for rdkit bond-order-based label and our cip-based label) is that we point the lowest-ranking atom away from us, and see whether the rest of the atoms are arranged clockwise or
    counter-clockwise in increasing order of their bond index / CIP ranking.

    This function can also handle other types of rankings than the CIP ranking. Just replace cip_ranking with the ranking you want to use.
    """
    if cw_ccw_label == Chem.ChiralType.CHI_UNSPECIFIED:
        return Chem.ChiralType.CHI_UNSPECIFIED
    nbrs = [(x.GetOtherAtomIdx(atom.GetIdx()), x.GetIdx()) for x in atom.GetBonds()]
    s_nbrs = sorted(nbrs, key=lambda x: cip_ranking[x[0]])
    if (
        len(nbrs) == 3
    ):  # case where the 'leading atom' is the implicit hydrogen not in the bonds
        # See if the cycle of cip-ranked indices is the same as the cycle of b-index ranked indices (pointing to the same atoms)
        order_based_on_bonds = [
            x.GetOtherAtomIdx(atom.GetIdx()) for x in atom.GetBonds()
        ]
        order_based_on_bonds_with_cip = [
            (idx, cip_ranking[idx]) for idx in order_based_on_bonds
        ]
        order_based_on_cip = list(
            map(
                lambda x: x[0],
                sorted(order_based_on_bonds_with_cip, key=lambda x: x[1]),
            )
        )
        if are_same_cycle(order_based_on_bonds, order_based_on_cip):
            return cw_ccw_label
        return get_opposite_chiral_tag(cw_ccw_label)
    elif len(nbrs) == 4:
        # Get the (atomindex, bondindex) pair for the bond that is not in the cycle for the bond-based representation
        leading_bond_order_representation = nbrs[0]
        # Get the (atomindex, bondindex) pair for the bond that is not in the cycle for the cip-based representation
        leading_atom_representation = s_nbrs[0]
        if (
            leading_bond_order_representation == leading_atom_representation
        ):  # case where the 'leading atom' is the same in both representations (leading atom = atom not in the cycle that defines clockwise/counter-clockwise)
            # See if the cycle of cip-ranked indices is the same as the cycle of b-index ranked indices (pointing to the same atoms)
            order_based_on_bonds = [
                x.GetOtherAtomIdx(atom.GetIdx()) for x in atom.GetBonds()
            ][1:]
            order_based_on_bonds_with_cip = [
                (idx, cip_ranking[idx]) for idx in order_based_on_bonds
            ]
            order_based_on_cip = list(
                map(
                    lambda x: x[0],
                    sorted(order_based_on_bonds_with_cip, key=lambda x: x[1]),
                )
            )
            if are_same_cycle(order_based_on_bonds, order_based_on_cip):
                return cw_ccw_label
            return get_opposite_chiral_tag(cw_ccw_label)
        else:  # case where the 'leading atom' is different in the representations
            # Get the two remaining atoms after taking out the leading atoms/bonds (the ones that are not in the cycle) for both representations
            remaining_neighbor_indices_bond_order_based = [
                x[0] for x in nbrs[1:]
            ]  # contains 3 atom indices
            remaining_neighbor_indices_cip_based = [
                x[0] for x in s_nbrs[1:]
            ]  # contains 3 atom indices
            remaining_two_atoms = [
                x
                for x in remaining_neighbor_indices_bond_order_based
                if x in remaining_neighbor_indices_cip_based
            ]  # contains 2 atom indices
            order_of_remaining_pair_bond_order_based = (
                order_of_pair_of_indices_in_cycle(
                    remaining_two_atoms[0],
                    remaining_two_atoms[1],
                    remaining_neighbor_indices_bond_order_based,
                )
            )
            order_of_remaining_pair_atom_index_based = (
                order_of_pair_of_indices_in_cycle(
                    remaining_two_atoms[0],
                    remaining_two_atoms[1],
                    remaining_neighbor_indices_cip_based,
                )
            )
            if (
                order_of_remaining_pair_bond_order_based
                == order_of_remaining_pair_atom_index_based
            ):  # This is how it works after you think hard about it
                return get_opposite_chiral_tag(cw_ccw_label)
            else:
                return cw_ccw_label
    else:  # case where the atom has less than 3 or more than 4 neighbors but still has a chiral tag, could happen in generation
        return Chem.ChiralType.CHI_UNSPECIFIED


def get_cip_ranking(mol):
    """
        Gets a type of CIP ranking of the atoms in the molecule such that the ranking is 
        unique for each atom in the molecule.
        The ranking ignores the stereochemistry of the molecule, since we want to get the ranking 
        for sampled molecules precicely to be able to set the stereochemistry consistently.
        In this sense, it is not the 'true' CIP label, but it is still a unique ranking of 
        the atoms that doesn't reference the order of the atoms or bonds in the data structure.
    """
    # Okay so this part is a bit tricky when the molecules we generate are crap, 
    # UpdatePropertyCache() can throw an error.
    # But I guess can just create some dummy CIP ranking in that case, 
    # since the generated molecules are not going to be valid anyways?
    m_copy = copy.deepcopy(mol)
    for (
        atom
    ) in m_copy.GetAtoms():  # Atom mapping will affect the AllChem.CanonicalRankAtoms
        atom.SetAtomMapNum(0)
    # Remove stereochemistry information from the molecule to ensure that the same 3D structure can be recovered after creating a molecule from scratch without the chiral tags yet set
    try:
        rdmolops.RemoveStereochemistry(m_copy)
        m_copy.UpdatePropertyCache()  # required for the next step to work after removing the steroechemistry
        AllChem.EmbedMolecule(
            m_copy
        )  # Generate 3D coordinates for correct stereochemistry. This gets us a unique CIP ranking.
        cip_ranks = list(
            AllChem.CanonicalRankAtoms(m_copy)
        )  # Calculate CIP values for atoms
    except Exception as e:
        print(f"Caught an exception while trying to get CIP ranking: {e}")
        cip_ranks = list(range(m_copy.GetNumAtoms()))
    return cip_ranks

def get_atom_symbol(atom, with_formal_charge_in_atom_symbols):
    if with_formal_charge_in_atom_symbols:
        return (
            atom.GetSymbol()
            if atom.GetFormalCharge() == 0
            else atom.GetSymbol() + f"{atom.GetFormalCharge():+}"
        )
    else:
        return atom.GetSymbol()

def create_canonicalized_mol(mol):
    """
    outputs mol in kekulized format, but the ordering is not based on the atom mapping but instead on the
    SMILES canonicalization. Removes potential information leaks :)
    """
    atom_mapping = {}
    # mol_ = Standardizer().standardize(copy.deepcopy(mol)) # all comparisons should be done in the true canonical format, not kekulized format
    # kekulization is done only so that
    for atom in mol.GetAtoms():
        atom_mapping[atom.GetIdx()] = atom.GetAtomMapNum()

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    smi = Chem.MolToSmiles(mol)  # , kekuleSmiles=True, isomericSmiles=True)
    mol_cano = Chem.MolFromSmiles(smi)
    # standardize the 'mol' object here
    # kekulize the mol
    # Chem.Kekulize(mol_cano) # ... why is this here?
    matches = mol.GetSubstructMatches(
        mol_cano
    )  # This maps from the canonical atom order to the original atom order
    if matches:  # if no matches, then no atom map nums at all
        # mapnums_old2new = {}
        # for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
        #     mapnums_old2new[atom_mapping[mat]] = 1 + atom.GetIdx()
        # update product mapping numbers according to canonical atom order
        # to completely remove potential information leak
        # atom.SetAtomMapNum(1 + atom.GetIdx())
        for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
            atom.SetAtomMapNum(atom_mapping[mat])

    Chem.Kekulize(mol_cano)

    return mol_cano
