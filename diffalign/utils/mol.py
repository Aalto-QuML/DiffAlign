import re
from multiset import *
import torch
from rdkit import Chem
import numpy as np
import torch.nn.functional as F
from typing import List, Optional, Any, Tuple
import os
import pathlib
import copy
from diffalign.utils import graph
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import rdkit
import time
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import logging

log = logging.getLogger(__name__)
from PIL import Image, ImageDraw, ImageFont
from diffalign.utils import data_utils

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

# NOTE: should these be in config files?
rdkit_bond_types = [
    0,
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
rdkit_bond_dirs = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
rdkit_bond_configs = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]
rdkit_atom_chiral_tags = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

# modified I (iodine) based on this: https://socratic.org/questions/how-many-single-bonds-can-iodine-form
# TODO: check others and come up with general rule for regular atom and charged one
# allowed_bonds = {'O': [2], 'C': [4], 'N': [3], 'I': [1, 3, 5, 7], 'H': [1], 'Cl': [1], 'Si': [4, 6], 'F': [1],
#                  'Br': [1], 'N+1': [4], 'O-1': [1], 'S': [2, 4, 6], 'B': [3], 'N-1': [2], 'Zn+1': [3], 'Cu': [1, 2],
#                  'Sn': [2, 4], 'P+1': [4, 6, 8], 'Mg+1': [3], 'C-1': [3], 'P': [3, 5, 7], 'S+1': [3, 5, 7], 'S-1': [1, 3, 5],
#                  'Se': [2, 4, 6], 'Zn': [2], 'Mg': [2], 'Au': [0]}

## need the previous encoding for evaluating old models
# atom_types = ['O', 'C', 'N', 'I', 'H', 'Cl', 'Si', 'F', 'Br', 'N+1', 'O-1', 'S', 'B', 'N-1', 'Zn+1',
#               'Cu', 'Sn', 'P+1', 'Mg+1', 'C-1', 'P', 'S+1', 'S-1', 'Se', 'Zn', 'Mg']

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_smiles_like_diffalign_output(smiles: str) -> str:
    '''
        Get the smiles string like the one used in the diffalign output.
    '''
    prod_m = Chem.MolFromSmiles(smiles)
    for a in prod_m.GetAtoms():
        a.SetAtomMapNum(0)
    prod_m = Chem.RemoveAllHs(prod_m)
    Chem.RemoveStereochemistry(prod_m)
    return Chem.MolToSmiles(prod_m, canonical=True)

def get_canonical_smiles_from_rdkit_mol(mol: Chem.Mol) -> Chem.Mol:
    '''
        Given an rdkit mol, return the canonicalized smiles string.
    '''
    
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(mol,
                            canonical=True,      # default, ensures consistent atom ordering
                            kekuleSmiles=True,  # consistent representation of aromatic bonds
                            isomericSmiles=True,  # keep stereochemistry
                            allBondsExplicit=False,  # no need to write all single bonds
                            allHsExplicit=False,     # no need to write all hydrogens
                            doRandom=False)  

def get_reactants_and_products_from_reaction_with_rdkit(reaction_smiles: str) -> Tuple[List[str], List[str]]:
    '''
        Get the reactants and products from a reaction smiles string.
    '''
    reaction = Reactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
    return list(reaction.GetReactants()), list(reaction.GetProducts())

def remove_stereochem_from_smiles(smi):
    rcts = smi.split(">>")[0]
    prods = smi.split(">>")[1]
    rcts_mol = Chem.MolFromSmiles(rcts)
    prods_mol = Chem.MolFromSmiles(prods)
    Chem.RemoveStereochemistry(rcts_mol)
    Chem.RemoveStereochemistry(prods_mol)
    return (
        data_utils.create_canonical_smiles_from_mol(rcts_mol)
        + ">>"
        + data_utils.create_canonical_smiles_from_mol(prods_mol)
    )


def remove_charges_from_smiles(smi):
    rcts = smi.split(">>")[0]
    prods = smi.split(">>")[1]
    rcts_mol = Chem.MolFromSmiles(rcts)
    prods_mol = Chem.MolFromSmiles(prods)
    for atom in rcts_mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
    for atom in prods_mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
    return (
        data_utils.create_canonical_smiles_from_mol(rcts_mol)
        + ">>"
        + data_utils.create_canonical_smiles_from_mol(prods_mol)
    )


# def remove_stereochem_from_smiles(smi):
#     '''
#         Removes stereochemistry information from a smiles string.
#     '''
#     rcts = smi.split('>>')[0].split('.')
#     prods = smi.split('>>')[1].split('.')

#     new_rxn_smi = ''
#     for i, r in enumerate(rcts):
#         mol = Chem.MolFromSmiles(r)
#         if mol is None: assert f'{r} is not a valid smiles string.\n'
#         Chem.RemoveStereochemistry(mol)
#         Chem.Kekulize(mol, clearAromaticFlags=True)
#         smi = Chem.MolToSmiles(mol)
#         if i<len(rcts)-1:
#             new_rxn_smi += smi + '.'
#         else:
#             new_rxn_smi += smi + '>>'

#     for i, p in enumerate(prods):
#         mol = Chem.MolFromSmiles(p)
#         if mol is None: assert f'{p} is not a valid smiles string.\n'
#         Chem.RemoveStereochemistry(mol)
#         Chem.Kekulize(mol, clearAromaticFlags=True)
#         smi = Chem.MolToSmiles(mol)
#         if i<len(prods)-1:
#             new_rxn_smi += smi + '.'
#         else:
#             new_rxn_smi += smi

#     return new_rxn_smi


def get_rdkit_bond_types(bond_types):
    """
    Add the bond types to the list of bond types.
    """
    new_bond_types = []

    for b in bond_types:
        if b == "SINGLE":
            new_bond_types.append(BT.SINGLE)
        elif b == "DOUBLE":
            new_bond_types.append(BT.DOUBLE)
        elif b == "TRIPLE":
            new_bond_types.append(BT.TRIPLE)
        elif b == "AROMATIC":
            new_bond_types.append(BT.AROMATIC)
        else:
            new_bond_types.append(b)

    return new_bond_types


def get_bond_orders(bond_types):
    """
    Get the bond orders from the bond types.
    """
    bond_orders = []
    for b in bond_types:
        if b == BT.SINGLE:
            bond_orders.append(1)
        if b == BT.DOUBLE:
            bond_orders.append(2)
        if b == BT.TRIPLE:
            bond_orders.append(3)
        if b == BT.AROMATIC:
            bond_orders.append(1.5)
        else:
            bond_orders.append(0)

    return bond_orders


def get_bond_orders_correct(bond_types):
    """
    Get the bond orders from the bond types.
    """
    bond_orders = []

    for b in bond_types:
        if b == BT.SINGLE or b == "SINGLE":
            bond_orders.append(1)
        elif b == BT.DOUBLE or b == "DOUBLE":
            bond_orders.append(2)
        elif b == BT.TRIPLE or b == "TRIPLE":
            bond_orders.append(3)
        elif b == BT.AROMATIC or b == "AROMATIC":
            bond_orders.append(1.5)
        else:
            bond_orders.append(0)

    return bond_orders


def get_rdkit_chiral_tags(chiral_tags):
    rdkit_chiral_tags = []

    # TODO: could add more types here
    for tag in chiral_tags:
        if tag == "ccw":
            rdkit_chiral_tags.append(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == "cw":
            rdkit_chiral_tags.append(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        else:
            rdkit_chiral_tags.append(Chem.ChiralType.CHI_UNSPECIFIED)

    return rdkit_chiral_tags


def get_rdkit_bond_dirs(bond_dirs):
    rdkit_bond_dirs = []

    # rdkit_bond_dirs = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT]
    for dir in bond_dirs:
        if dir == "ENDUPRIGHT":
            rdkit_bond_dirs.append(Chem.rdchem.BondDir.ENDUPRIGHT)
        elif dir == "ENDDOWNRIGHT":
            rdkit_bond_dirs.append(Chem.rdchem.BondDir.ENDDOWNRIGHT)
        else:
            rdkit_bond_dirs.append(Chem.rdchem.BondDir.NONE)

    return rdkit_bond_dirs


def get_atom_symbol(atom, with_formal_charge_in_atom_symbols):
    if with_formal_charge_in_atom_symbols:
        return (
            atom.GetSymbol()
            if atom.GetFormalCharge() == 0
            else atom.GetSymbol() + f"{atom.GetFormalCharge():+}"
        )
    else:
        return atom.GetSymbol()


def split_atom_symbol_from_formal_charge(symbol):
    """
    Input: An atom symbol with a formal charge attached to it, e.g., 'N-1'
    Output: The atom symbol and the formal charge, e.g., 'N', -1
    """
    atom_symbol = re.split("[-+]\d+", symbol)[0]
    formal_charge_matches = re.findall("[-+]\d+", symbol)

    if not formal_charge_matches:
        formal_charge = 0
    else:
        formal_charge = int(formal_charge_matches[0])

    return atom_symbol, formal_charge


# def mol_to_graph_with_stereochem(m, atom_types, atom_charges, rdkit_atom_chiral_tags,
#                                   rdkit_bond_types, rdkit_bond_dirs, with_formal_charge_in_atom_symbols=False):
#     '''
#         m: rdkit molecule object
#         returns: atom_symbols, atom_charges, atom_chiral, edge_index, bond_type, bond_dir, atom_map
#     '''
#     # TODO: add case where input is a smiles string?
#     # TODO: should reprocess m anyway, better to make this expect a smiles string because reprocessing does not lead to the same thing

#     if BT.AROMATIC not in rdkit_bond_types: Chem.Kekulize(m)

#     atom_symbols = F.one_hot(torch.tensor([atom_types.index(get_atom_symbol(atom, with_formal_charge_in_atom_symbols)) for atom in m.GetAtoms()]),
#                                           num_classes=len(atom_types)).float()

#     atom_map = torch.tensor([atom.GetAtomMapNum() for atom in m.GetAtoms()])

#     # get atom charges
#     # atom_charges = [atom.GetFormalCharge() for atom in m.GetAtoms()]
#     atom_charges = F.one_hot(torch.tensor([atom_charges.index(atom.GetFormalCharge()) for atom in m.GetAtoms()]),
#                                            num_classes=len(atom_charges)).float()

#     # get atom chirality
#     # atom_chiral = [atom.GetChiralTag() for atom in m.GetAtoms()]
#     atom_chiral = F.one_hot(torch.tensor([rdkit_atom_chiral_tags.index(atom.GetChiralTag()) for atom in m.GetAtoms()]),
#                                           num_classes=len(rdkit_atom_chiral_tags)).float()

#     # get bonds' end indices
#     # TODO: duplicate and turn to torch tensor
#     # bond_end_indices = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in m.GetBonds()]
#     # TODO: why long and not smthg else
#     edge_index = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]+[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] for bond in m.GetBonds()]).flatten().reshape(-1,2).t().contiguous().long()

#     # get bond types
#     # TODO: one-hot encode 2D tensor
#     # bond_types = {bond.GetIdx():bond.GetBondType() for bond in m.GetBonds()}

#     bond_types = F.one_hot(torch.tensor([rdkit_bond_types.index(bond.GetBondType()) for bond in m.GetBonds()]).repeat_interleave(2),
#                                 num_classes=len(rdkit_bond_types)).float()
#     #bond_dirs = {bond.GetIdx():bond.GetBondDir() for bond in m.GetBonds()}
#     bond_dirs = F.one_hot(torch.tensor([rdkit_bond_dirs.index(bond.GetBondDir()) for bond in m.GetBonds()]).repeat_interleave(2),
#                                 num_classes=len(rdkit_bond_dirs)).float()

#     # if Chem.rdchem.BondType.AROMATIC in bond_types.values():
#     #     new_smi = Chem.MolToSmiles(m)
#     #     print(f'still aromatic smi {new_smi}\n')
#     #     print(f'bond_types {bond_types}\n')

#     return atom_symbols, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map


def rebuild_mol_with_stereochem_LEGACY(
    atom_symbols, atom_charges, atom_chiral, bond_end_indices, bond_types, bond_dirs
):
    rw_mol = Chem.RWMol()

    # add atoms
    for atom_symbol, atom_charge, tag in zip(atom_symbols, atom_charges, atom_chiral):
        atom = Chem.Atom(atom_symbol)
        atom.SetFormalCharge(atom_charge)
        atom.SetChiralTag(tag)
        rw_mol.AddAtom(atom)

    # add bonds
    for (begatom, endatom), bond_type, bond_dir in zip(
        bond_end_indices, bond_types.values(), bond_dirs.values()
    ):
        rw_mol.AddBond(begatom, endatom, bond_type)
        rw_mol.GetBondBetweenAtoms(begatom, endatom).SetBondDir(bond_dir)

    # add implicit info
    new_mol = rw_mol.GetMol()
    new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
    # new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True, kekuleSmiles=True))
    if new_mol is None:
        return "invalid"
    try:
        Chem.Kekulize(new_mol, clearAromaticFlags=True)
    except:
        print(f"bond_types {bond_types}\n")

    # # add stereo info
    # t = time.time()
    # stereo_vec = [e for e in Chem.rdmolops.FindPotentialStereo(new_mol) if e.type==Chem.rdchem.StereoType.Bond_Double]
    # #print(f'==== rebuild stereochem algo total time: {datetime.time()-t}\n')

    # for stereo_info in stereo_vec:
    #     bond_idx = stereo_info.centeredOn
    #     print(f'bond_idx {bond_idx}\n')
    #     stereo_atoms = [a for a in stereo_info.controllingAtoms]
    #     print(f'stereo_atoms {stereo_atoms}\n')
    #     new_mol.GetBondWithIdx(bond_idx).SetStereo(bond_stereo[bond_idx])
    #     new_mol.GetBondWithIdx(bond_idx).SetStereoAtoms(stereo_atoms[0], stereo_atoms[2])

    # recovered_smi = Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True, kekuleSmiles=True)
    recovered_smi = Chem.MolToSmiles(
        Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)), canonical=True
    )

    return recovered_smi


def parse_mol_with_stereo_chem_LEGACY(m):
    placeholder = 4294967295
    #    mol = Chem.MolFromSmiles(smi)
    #    Chem.Kekulize(mol, clearAromaticFlags=True)
    #    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, kekuleSmiles=True))

    # get atom symbols
    atom_symbols = [atom.GetSymbol() for atom in m.GetAtoms()]

    # get atom charges
    atom_charges = [atom.GetFormalCharge() for atom in m.GetAtoms()]
    # get atom chirality
    atom_chiral = [atom.GetChiralTag() for atom in m.GetAtoms()]
    # get bonds' end indices
    # TODO: duplicate and turn to torch tensor
    bond_end_indices = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in m.GetBonds()
    ]
    # get bond types
    # TODO: one-hot encode 2D tensor
    bond_types = {bond.GetIdx(): bond.GetBondType() for bond in m.GetBonds()}
    bond_dirs = {bond.GetIdx(): bond.GetBondDir() for bond in m.GetBonds()}

    if Chem.rdchem.BondType.AROMATIC in bond_types.values():
        new_smi = Chem.MolToSmiles(m)
        print(f"still aromatic smi {new_smi}\n")
        print(f"bond_types {bond_types}\n")

    # # get bond e/z stereo
    # bond_stereo_config = {bond.GetIdx():bond.GetStereo() for bond in m.GetBonds()}
    # #print(f'bond_stereo_config {bond_stereo_config}')
    # bond_stereo_config_true = {bond.GetIdx():bond.GetStereo() for bond in m.GetBonds() if bond.GetStereo()!=Chem.rdchem.BondStereo.STEREONONE}
    # # get stereo atoms from smiles
    # all_smi_stereo_atoms = {bond.GetIdx():[a for a in bond.GetStereoAtoms()] for bond in m.GetBonds() if len(bond.GetStereoAtoms())>0}

    # # NOTE: a small sanity check that the algo does not find more stereochem than is parsed by rdkit from smiles?
    # #       can remove it if is becomes costly
    # # NOTE: check if algo returns any other types of stereochem? everything else is ignored atm.
    # t = time.time()
    # stereo_info_double_bond = [e for e in Chem.rdmolops.FindPotentialStereo(m) if e.type==Chem.rdchem.StereoType.Bond_Double]
    # all_algo_stereo_atoms = {e.centeredOn:[c for c in e.controllingAtoms] for e in stereo_info_double_bond}
    # # also sort by index the dict
    # all_algo_stereo_atoms = dict(all_algo_stereo_atoms.items()) # NOTE: correct?
    # # print(f'==== stereochem algo total time: {datetime.time()-t}\n')

    # if all_algo_stereo_atoms.keys()==all_smi_stereo_atoms.keys():
    #     # NOTE: fixing (canonicalizing) the stereo atoms might not be necessary since rdkit
    #     #       parses according to cip rank already (see example in notebooks/encode_stereochem.ipynb)
    #     # TODO: would be good to understand how rdkit processes stereochem from smiles officially
    #     #       to see if we need to rerun the algorithm in any edge case
    #     ### check that the stereo atoms at the ends of the double bond are correct
    #     for bond_idx, smi_stereo_atoms in all_smi_stereo_atoms.items(): # NOTE: what if the order is different?, could try sets
    #         # NOTE: indices 0,2 are assumed for the double bond case, need to be checked in rdkit source code/documentation
    #         if smi_stereo_atoms[0]!=all_algo_stereo_atoms[bond_idx][0] or smi_stereo_atoms[1]!=all_algo_stereo_atoms[bond_idx][2]:
    #                 print(f'smi_stereo_atoms {smi_stereo_atoms} and all_algo_stereo_atoms {all_algo_stereo_atoms[bond_idx]} are different for bond {bond_idx}\n')

    #                 if smi_stereo_atoms[0]!=all_algo_stereo_atoms[bond_idx][0] and smi_stereo_atoms[1]!=all_algo_stereo_atoms[bond_idx][2]:
    #                         print(f'changing stereo atoms\n')
    #                         # no need to change the stereo type (E/Z)
    #                         all_smi_stereo_atoms[bond_idx] = all_algo_stereo_atoms[bond_idx]
    #                 else: # only one of the stereo atoms is different
    #                         print(f'changing stereo config\n')
    #                         print(f'====== before: bond_stereo_config[bond_idx] {bond_stereo_config[bond_idx]}\n')
    #                         bond_stereo_config[bond_idx] = Chem.rdchem.BondStereo.STEREOE if bond_stereo_config[bond_idx]==Chem.rdchem.BondStereo.STEREOZ else Chem.rdchem.BondStereo.STEREOZ
    #                         print(f'====== after: bond_stereo_config[bond_idx] {bond_stereo_config[bond_idx]}\n')
    #                         all_smi_stereo_atoms[bond_idx] = all_algo_stereo_atoms[bond_idx]
    #                         print(f'====== after: all_smi_stereo_atoms[bond_idx] {all_smi_stereo_atoms[bond_idx]}\n')
    #         #else:
    #         #print(f'Found {all_algo_stereo_atoms.keys()} stereo bonds through the algorithm, but {all_smi_stereo_atoms.keys()}.')

    return (
        atom_symbols,
        atom_charges,
        atom_chiral,
        bond_end_indices,
        bond_types,
        bond_dirs,
    )


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


from diffalign.utils import mol
import copy


def match_atom_mapping_without_stereo(mol1, mol2):
    # changes the atom mapping in mol2 to match with the convention in mol1,
    # assumes that mol1 and mol2 are the same up to stereochemistry. So mol2 can have stereochemistry.
    mol2_copy = copy.deepcopy(mol2)
    Chem.RemoveStereochemistry(mol2_copy)
    match_atom_mapping(mol1, mol2_copy)
    # transfer the atom mapping from mol2_copy to mol2
    for atom2_copy, atom2 in zip(mol2_copy.GetAtoms(), mol2.GetAtoms()):
        atom2.SetAtomMapNum(atom2_copy.GetAtomMapNum())


def match_atom_mapping(mol1, mol2):
    # changes the atom mapping in mol2 to match with the convention in mol1
    match = mol2.GetSubstructMatch(mol1)
    # Transfer atom mappings from mol1 to mol2 based on the match
    for idx1, idx2 in enumerate(match):
        if idx2 >= 0:
            mol2.GetAtomWithIdx(idx2).SetAtomMapNum(
                mol1.GetAtomWithIdx(idx1).GetAtomMapNum()
            )


def transfer_bond_dir_from_product_to_reactant(r_mol, p_mol):

    # Extract relevant info from the product molecule
    r_mol_new = copy.deepcopy(r_mol)

    am_pair_to_bonddir = {}
    # am_to_chiral_tag = {}
    for idx, bond in enumerate(p_mol.GetBonds()):
        # check whether the atom has chirality
        # print(bond.GetBondDir())
        if bond.GetBondDir() != Chem.BondDir.NONE:
            # print(bond.GetBondDir())
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            am_pair_to_bonddir[
                (begin_atom.GetAtomMapNum(), end_atom.GetAtomMapNum())
            ] = bond.GetBondDir()

    # transfer the bond dir to the reactant
    for idx, bond in enumerate(r_mol_new.GetBonds()):
        if (
            bond.GetBeginAtom().GetAtomMapNum(),
            bond.GetEndAtom().GetAtomMapNum(),
        ) in am_pair_to_bonddir:
            bond.SetBondDir(
                am_pair_to_bonddir[
                    (
                        bond.GetBeginAtom().GetAtomMapNum(),
                        bond.GetEndAtom().GetAtomMapNum(),
                    )
                ]
            )
        elif (
            bond.GetEndAtom().GetAtomMapNum(),
            bond.GetBeginAtom().GetAtomMapNum(),
        ) in am_pair_to_bonddir:
            bond.SetBondDir(
                am_pair_to_bonddir[
                    (
                        bond.GetEndAtom().GetAtomMapNum(),
                        bond.GetBeginAtom().GetAtomMapNum(),
                    )
                ]
            )

    return r_mol_new


def transfer_chirality_from_product_to_reactant(r_mol, p_mol):
    """Transfers the chiral tags with atom mapping from the product side to the reactant side"""

    # Extract relevant info from the product molecule
    r_mol_new = copy.deepcopy(r_mol)
    am_to_am_of_neighbors_prod = {}
    am_to_chiral_tag = {}
    for idx, atom in enumerate(p_mol.GetAtoms()):
        # check whether the atom has chirality
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            # get the atom maps of the neighbors in the order that they are in the bonds
            nbrs = [
                (x.GetOtherAtomIdx(atom.GetIdx()), x.GetIdx()) for x in atom.GetBonds()
            ]
            atom_maps_of_neighbors = [
                p_mol.GetAtomWithIdx(x[0]).GetAtomMapNum() for x in nbrs
            ]
            am_to_am_of_neighbors_prod[atom.GetAtomMapNum()] = atom_maps_of_neighbors
            am_to_chiral_tag[atom.GetAtomMapNum()] = atom.GetChiralTag()

    for idx, atom in enumerate(r_mol_new.GetAtoms()):
        if atom.GetAtomMapNum() in am_to_am_of_neighbors_prod:
            # Get the atom maps of the neighbors in the product
            am_nbrs_prod = am_to_am_of_neighbors_prod[atom.GetAtomMapNum()]
            # Get the indices of the neighbors of the atom in the reactant
            nbrs = [x.GetOtherAtomIdx(atom.GetIdx()) for x in atom.GetBonds()]
            # Get the map from atom map to rank in the product
            am_to_rank_prod = {am: idx for idx, am in enumerate(am_nbrs_prod)}
            # Get the map from reactant idx to rank in the product (assuming that the same atom mappings are present in the reactant side)
            try:
                nbr_idx_to_rank_prod = {
                    nbr_idx: am_to_rank_prod[
                        r_mol.GetAtomWithIdx(nbr_idx).GetAtomMapNum()
                    ]
                    for nbr_idx in nbrs
                }
                # Flip the cw/ccw label if necessary
                cw_ccw = mol.switch_between_bond_cw_ccw_label_and_cip_based_label(
                    atom, am_to_chiral_tag[atom.GetAtomMapNum()], nbr_idx_to_rank_prod
                )
            except:
                # If the atom mappings are not present in t
                # he reactant, just guess one of the two
                cw_ccw = Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            atom.SetChiralTag(cw_ccw)
    return r_mol_new


def remove_atom_mapping_from_mol(mol):
    [a.ClearProp("molAtomMapNumber") for a in mol.GetAtoms()]
    return mol


def transfer_stereo_from_product_to_reactant(
    r_smiles_with_atom_map, p_smiles_with_stereo_and_atom_map
):
    try:
        # breakpoint()
        prod_side_ams = set(
            [
                a.GetAtomMapNum()
                for a in Chem.MolFromSmiles(
                    p_smiles_with_stereo_and_atom_map
                ).GetAtoms()
            ]
        )
        rct_mol = Chem.MolFromSmiles(r_smiles_with_atom_map)
        prod_mol = Chem.MolFromSmiles(p_smiles_with_stereo_and_atom_map)

        Chem.RemoveStereochemistry(
            rct_mol
        )  # This does some kind of sanitization, otherwise transferring the bond_dirs doesn't work reliably
        for (
            a
        ) in (
            rct_mol.GetAtoms()
        ):  # remove atom mappings that are not on the product side
            if a.GetAtomMapNum() not in prod_side_ams:
                a.ClearProp("molAtomMapNumber")
        if "@" in p_smiles_with_stereo_and_atom_map:
            rct_mol = mol.transfer_chirality_from_product_to_reactant(rct_mol, prod_mol)
        if (
            "/" in p_smiles_with_stereo_and_atom_map
            or "\\" in p_smiles_with_stereo_and_atom_map
        ):
            rct_mol = mol.transfer_bond_dir_from_product_to_reactant(rct_mol, prod_mol)
        remove_atom_mapping_from_mol(rct_mol)
        r_smiles = Chem.MolToSmiles(rct_mol, canonical=True)
    except:
        # remove the atom mappings from the reactant

        # Regular expression to match atom mappings in the format [atom:digit]
        pattern = r"\[([A-Za-z0-9@+-]+):\d+\]"
        # Replace the matched pattern with just the atom without the mapping
        cleaned_smiles = re.sub(pattern, r"[\1]", r_smiles_with_atom_map)
        # Additionally, handle cases where the atom mapping is on a single atom like [C:1] -> C
        cleaned_smiles = re.sub(r"\[([A-Za-z0-9@+-]+)\]", r"\1", cleaned_smiles)
        # r_smiles_with_atom_map = re.sub(r':\d+', '', r_smiles_with_atom_map)
        r_smiles = cleaned_smiles
    return r_smiles


def get_opposite_chiral_tag(atom):
    if atom == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CW
    elif atom == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    return None


def get_cip_ranking(mol):
    """
    Gets a type of CIP ranking of the atoms in the molecule such that the ranking is unique for each atom in the molecule.
    The ranking ignores the stereochemistry of the molecule, since we want to get the ranking for sampled molecules precicely to be able to set the stereochemistry consistently.
    In this sense, it is not the 'true' CIP label, but it is still a unique ranking of the atoms that doesn't reference the order of the atoms or bonds in the data structure.
    """
    # Okay so this part is a bit tricky when the molecules we generate are crap, UpdatePropertyCache() can throw an error.
    # But I guess can just create some dummy CIP ranking in that case, since the generated molecules are not going to be valid anyways?
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
        log.info(f"Caught an exception while trying to get CIP ranking: {e}")
        cip_ranks = list(range(m_copy.GetNumAtoms()))
    return cip_ranks


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


def sanity_check_and_fix_atom_mapping(mask_atom_mapping, g_nodes):
    # DEPRECATED
    """
    Checks if the atom mapping is valid and fixes it if it's not.

    Valid atom-mapping:
        - the number of mapped atoms should be exactly 2
        - mapped atoms should be the same

    Any atom mapping that doesn't satisfy the above conditions is set to 0.
    """
    atom_mapping_dict = {}
    for i, (atom, map_num) in enumerate(zip(g_nodes.argmax(-1), mask_atom_mapping)):
        if map_num.item() not in atom_mapping_dict.keys():
            atom_mapping_dict[map_num.item()] = {
                "atoms": [atom.item()],
                "map_num_idx": [i],
            }
        else:
            atom_mapping_dict[map_num.item()]["atoms"].append(atom.item())
            atom_mapping_dict[map_num.item()]["map_num_idx"].append(i)

    for map_num in atom_mapping_dict.keys():
        # atom-map numbers should be exactly 2
        if len(atom_mapping_dict[map_num]["atoms"]) != 2:
            for idx in atom_mapping_dict[map_num]["map_num_idx"]:
                mask_atom_mapping[idx] = 0

        # mapped atoms should be the same
        elif (
            atom_mapping_dict[map_num]["atoms"][0]
            != atom_mapping_dict[map_num]["atoms"][1]
        ):
            for idx in atom_mapping_dict[map_num]["map_num_idx"]:
                mask_atom_mapping[idx] = 0

    # renumber atom mapping to be from 0 to n (smallest to largest)
    for i, map_num in enumerate(torch.unique(mask_atom_mapping)):
        mask_atom_mapping[mask_atom_mapping == map_num] = i

    return mask_atom_mapping


def filter_out_nodes(out_node_idx, nodes, edge_index, edge_attr, **kwargs):
    """
    Cut the first n nodes from the node and edge features of a graph.
    Kwargs contains other node feature tensors we might want to cut as well.
    Primarily used in the supernode_dataset_preprocessing function.

    out_node_idx: the indices of the nodes to remove.
    """
    cut_nodes = nodes[np.setdiff1d(range(nodes.shape[0]), out_node_idx)]

    # (2, n_edges)
    cut_edges = [
        (edge_index[:, i], edge_attr[i])
        for i in range(edge_index.shape[1])
        if (
            edge_index[0, i] not in out_node_idx
            and edge_index[1, i] not in out_node_idx
        )
    ]
    cut_edge_index = torch.cat(
        [edge_info[0].unsqueeze(-1) for edge_info in cut_edges], dim=-1
    )
    cut_edge_attr = torch.cat(
        [edge_info[1].unsqueeze(0) for edge_info in cut_edges], dim=0
    )

    not_a_tensor = [
        k
        for k in kwargs.values()
        if not isinstance(k, torch.Tensor)
        or (isinstance(k, torch.Tensor) and k.ndim > 1)
    ]
    assert (
        len(not_a_tensor) == 0
    ), "cut_first_n_nodes was given a variable other than a tensor or a multi-dimensional tensor."

    new_kwargs = {
        var_name: t[np.setdiff1d(range(nodes.shape[0]), out_node_idx)]
        for var_name, t in kwargs.items()
    }

    return cut_nodes, cut_edge_index, cut_edge_attr, new_kwargs


def rxn_list_to_str(rcts, prods):
    rxn_str = ""

    for i, m in enumerate(rcts):
        if i == len(rcts) - 1:  # last molecule is product
            rxn_str += m
        else:
            rxn_str += m + "."

    rxn_str += ">>"

    for i, m in enumerate(prods):
        if i == len(prods) - 1:  # last molecule is product
            rxn_str += m
        else:
            rxn_str += m + "."

    return rxn_str


def get_cano_list_smiles(
    X, E, atom_types, bond_types, mol_assignment, plot_dummy_nodes=False
):
    """
    Returns canonical smiles of all the molecules in a reaction
    given a dense matrix representation of said reaction.
    Invidual molecules are identified by splitting their smiles by '.'.
    A set of canonical smiles is returned for each rxn.
    Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
    Handles batched reactions.

    X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
    E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

    return: list of smiles of valid molecules from rxn.
    """
    # DEPRECATED!!

    kekulize_molecule = BT.AROMATIC not in bond_types

    assert X.ndim == 2 and E.ndim == 3, (
        "Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n)."
        + f" Got X.shape={X.shape} and E.shape={E.shape} instead."
    )

    # suno_idx = atom_types.index('SuNo') # offset because index 0 is for no node

    all_rcts = []
    all_prods = []
    for j in range(X.shape[0]):  # for each rxn in batch
        product_mol_index = mol_assignment[j].max().item()
        product_start_index = (
            (mol_assignment[j] == product_mol_index)
            .nonzero(as_tuple=True)[0]
            .min()
            .item()
        )

        # suno_indices = (X[j,:]==suno_idx).nonzero(as_tuple=True)[0].cpu()
        cutoff = (
            1 if 0 in suno_indices else 0
        )  # relevant in case there's a SuNo node in the first position
        atoms = torch.tensor_split(X[j, :], suno_indices, dim=-1)[
            cutoff:
        ]  # ignore first set (SuNo)
        edges = torch.tensor_split(E[j, :, :], suno_indices, dim=-1)[cutoff:]

        rct_smiles = []
        prod_smiles = []
        # TODO: The references to the supernode here are old as well, but works for now, I guess?
        for i, mol_atoms in enumerate(atoms):  # for each mol in rxn
            mol_edges_to_all = edges[i]
            cutoff = (
                1 if 0 in suno_indices else 0
            )  # relevant in case there's a SuNo node in the first position
            mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[
                cutoff:
            ]
            mol_edges = mol_edges_t[
                i
            ]  # the edges from the molecule to the entire reaction
            cutoff = 1 if suno_idx in mol_atoms else 0
            mol_atoms = mol_atoms[cutoff:]  # (n-1)
            mol_edges = mol_edges[cutoff:, :][:, cutoff:]  # (n-1, n-1)
            mol = mol_from_graph(
                node_list=mol_atoms,
                adjacency_matrix=mol_edges,
                atom_types=atom_types,
                bond_types=bond_types,
                plot_dummy_nodes=plot_dummy_nodes,
            )
            if kekulize_molecule:
                smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)
            else:
                smiles = Chem.MolToSmiles(mol)
            set_mols = smiles.split(".")
            if i == len(atoms) - 1:
                prod_smiles.extend(set_mols)
            else:
                rct_smiles.extend(set_mols)
        all_rcts.append(rct_smiles)
        all_prods.append(prod_smiles)

    return all_rcts, all_prods


def get_cano_smiles_from_dense_legacy(X, E, atom_types, bond_types, return_dict=False):
    """
    Returns canonical smiles of all the molecules in a reaction
    given a dense matrix representation of said reaction.
    Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
    Handles batched reactions.

    X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
    E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

    return: list of smiles of valid molecules from rxn.
    """
    kekulize_molecule = BT.AROMATIC not in bond_types

    assert X.ndim == 2 and E.ndim == 3, (
        "Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n)."
        + f" Got X.shape={X.shape} and E.shape={E.shape} instead."
    )

    suno_idx = atom_types.index("SuNo")  # offset because index 0 is for no node

    all_smiles = {}
    all_rxn_str = []
    for j in range(X.shape[0]):  # for each rxn in batch
        # print(f'j {j}\n')
        suno_indices = (X[j, :] == suno_idx).nonzero(as_tuple=True)[0].cpu()
        cutoff = 1 if 0 in suno_indices else 0
        atoms = torch.tensor_split(X[j, :], suno_indices, dim=-1)[
            cutoff:
        ]  # ignore first set (SuNo)
        edges = torch.tensor_split(E[j, :, :], suno_indices, dim=-1)[cutoff:]

        rxn_smiles = []
        rxn_str = ""
        for i, mol_atoms in enumerate(atoms):  # for each mol in rxn
            cutoff = 1 if 0 in suno_indices else 0
            mol_edges_to_all = edges[i]
            mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[
                cutoff:
            ]
            mol_edges = mol_edges_t[i]
            cutoff = 1 if suno_idx in mol_atoms else 0
            mol_atoms = mol_atoms[cutoff:]  # (n-1)
            mol_edges = mol_edges[cutoff:, :][:, cutoff:]  # (n-1, n-1)
            mol = mol_from_graph(
                node_list=mol_atoms,
                adjacency_matrix=mol_edges,
                atom_types=atom_types,
                bond_types=bond_types,
            )
            if kekulize_molecule:
                smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)
            else:
                smiles = Chem.MolToSmiles(mol)

            if i < len(atoms) - 2:  # if the mol is not the last reactant
                rxn_str += smiles + "."  # instead of dot to make it easier to read rxn
            elif i == len(atoms) - 2:  # if the mol is the last reactant
                rxn_str += smiles + ">>"
            elif i == len(atoms) - 1:  # if the mol is the product
                rxn_str += smiles

            rxn_smiles.append(smiles)
        all_rxn_str.append(rxn_str)
        all_smiles[j] = rxn_smiles

    return all_smiles if return_dict else all_rxn_str


# def get_cano_smiles_with_atom_mapping_from_dense_stereochem(dense_data, rdkit_atom_types, rdkit_bond_types,
#                                                             rdkit_atom_charges, rdkit_atom_chiral_tags, with_formal_charge_in_atom_symbols,
#                                                             return_dict=False, plot_dummy_nodes=False):
#     '''
#         Returns canonical smiles of all the molecules in a reaction with atom mapping
#         given a dense matrix representation of said reaction.
#         Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
#         Handles batched reactions.

#         X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
#         E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

#         return: list of smiles of valid molecules from rxn.
#     '''
#     # TODO: make some of this stuff optinal
#     X_atom_types, X_atom_charges, X_chiral_tags = dense_data.X, dense_data.atom_charges, dense_data.atom_chiral
#     E_bond_types, E_bond_dirs = dense_data.E, dense_data.bond_dirs
#     atom_map_numbers = dense_data.atom_map_numbers

#     assert X_atom_types.ndim==2 and E_bond_types.ndim==3,\
#             'Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n).' \
#             + f' Got X.shape={X_atom_types.shape} and E.shape={E_bond_types.shape} instead.'

#     # remove product
#     kekulize_molecule = BT.AROMATIC not in rdkit_bond_types

#     suno_idx = rdkit_atom_types.index('SuNo')
#     all_rxn_smiles = []
#     for j in range(X_atom_types.shape[0]): # for each rxn in batch
#         suno_indices = (X_atom_types[j,:]==suno_idx).nonzero(as_tuple=True)[0].cpu()
#         product_suno_idx = max(suno_indices)
#         # parse reactants
#         rcts_X_atom_types = X_atom_types[j,:product_suno_idx]
#         rcts_X_atom_charges = X_atom_charges[j,:product_suno_idx]
#         rcts_X_chiral_tags = X_chiral_tags[j,:product_suno_idx]

#         rcts_E = E_bond_types[j,:product_suno_idx, :product_suno_idx]
#         rcts_E_bond_dirs = E_bond_dirs[j,:product_suno_idx, :product_suno_idx]

#         rcts_atom_map_nums = atom_map_numbers[j, :product_suno_idx]

#         # (atom_symbols, atom_charges, atom_chiral, bond_adjacency, bond_dirs, rdkit_atom_types, rdkit_atom_charges, rdkit_atom_chiral_tags, rdkit_bond_types, rdkit_bond_dirs)
#         rcts_mol = mol_from_graph_with_stereochem(atom_symbols=rcts_X_atom_types, atom_charges=rcts_X_atom_charges, atom_chiral=rcts_X_chiral_tags,
#                                                   bond_adjacency=rcts_E, bond_dirs=rcts_E_bond_dirs,
#                                                   rdkit_atom_types=rdkit_atom_types, rdkit_atom_charges=rdkit_atom_charges,
#                                                   rdkit_atom_chiral_tags=rdkit_atom_chiral_tags, rdkit_bond_types=rdkit_bond_types,
#                                                   rdkit_bond_dirs=rdkit_bond_dirs, plot_dummy_nodes=plot_dummy_nodes,
#                                                   atom_map_numbers=rcts_atom_map_nums.detach().cpu(),
#                                                   with_formal_charge_in_atom_symbols=with_formal_charge_in_atom_symbols)

#         # TODO: figure out how to canonicalize the molecule correctly to compare it to ground truth
#         if kekulize_molecule: rcts_smiles = Chem.MolToSmiles(rcts_mol, kekuleSmiles=True, isomericSmiles=True) # TODO: add canonical?
#         else: rcts_smiles = Chem.MolToSmiles(rcts_mol, isomericSmiles=True)

#         # parse product
#         prod_X_atom_types = X_atom_types[j, product_suno_idx:]
#         prod_X_atom_charges = X_atom_charges[j,product_suno_idx:]
#         prod_X_chiral_tags = X_chiral_tags[j,product_suno_idx:]
#         prod_E_bond_types = E_bond_types[j, product_suno_idx:, product_suno_idx:]
#         prod_E_bond_dirs = E_bond_dirs[j,product_suno_idx:, product_suno_idx:]
#         prod_atom_map_nums =  atom_map_numbers[j, product_suno_idx:]
#         prod_mol = mol_from_graph_with_stereochem(atom_symbols=prod_X_atom_types, atom_charges=prod_X_atom_charges,
#                                                   atom_chiral=prod_X_chiral_tags, bond_adjacency=prod_E_bond_types,
#                                                   bond_dirs=prod_E_bond_dirs,
#                                                   rdkit_atom_types=rdkit_atom_types, rdkit_atom_charges=rdkit_atom_charges,
#                                                   rdkit_atom_chiral_tags=rdkit_atom_chiral_tags, rdkit_bond_types=rdkit_bond_types,
#                                                   rdkit_bond_dirs=rdkit_bond_dirs, plot_dummy_nodes=plot_dummy_nodes,
#                                                   atom_map_numbers=prod_atom_map_nums.detach().cpu(),
#                                                   with_formal_charge_in_atom_symbols=with_formal_charge_in_atom_symbols)

#         if kekulize_molecule: prod_smiles = Chem.MolToSmiles(prod_mol, kekuleSmiles=True, isomericSmiles=True)
#         else: prod_smiles = Chem.MolToSmiles(prod_mol, isomericSmiles=True)

#         all_rxn_smiles.append(f'{rcts_smiles}>>{prod_smiles}')

#     return all_rxn_smiles


def get_cano_smiles_with_atom_mapping_from_dense(
    X,
    E,
    atom_types,
    bond_types,
    atom_map_numbers,
    return_dict=False,
    plot_dummy_nodes=False,
):
    """
    Returns canonical smiles of all the molecules in a reaction with atom mapping
    given a dense matrix representation of said reaction.
    Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
    Handles batched reactions.

    X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
    E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

    return: list of smiles of valid molecules from rxn.
    """
    assert X.ndim == 2 and E.ndim == 3, (
        "Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n)."
        + f" Got X.shape={X.shape} and E.shape={E.shape} instead."
    )
    # DEPRECATED!
    # REMOVE ATOM MAPPING
    # remove product

    kekulize_molecule = BT.AROMATIC not in bond_types

    suno_idx = atom_types.index("SuNo")
    all_rxn_smiles = []
    for j in range(X.shape[0]):  # for each rxn in batch
        suno_indices = (X[j, :] == suno_idx).nonzero(as_tuple=True)[0].cpu()
        product_suno_idx = max(suno_indices)
        # parse reactants
        rcts_X = X[j, :product_suno_idx]
        rcts_E = E[j, :product_suno_idx, :product_suno_idx]
        rcts_atom_map_nums = atom_map_numbers[j, :product_suno_idx]
        rcts_mol = mol_from_graph(
            node_list=rcts_X,
            adjacency_matrix=rcts_E,
            atom_types=atom_types,
            bond_types=bond_types,
            plot_dummy_nodes=plot_dummy_nodes,
            atom_map_numbers=rcts_atom_map_nums.detach().cpu(),
        )
        if kekulize_molecule:
            rcts_smiles = Chem.MolToSmiles(
                rcts_mol, kekuleSmiles=True, isomericSmiles=True
            )
        else:
            rcts_smiles = Chem.MolToSmiles(rcts_mol, isomericSmiles=True)
        # parse product
        prod_X = X[j, product_suno_idx:]
        prod_E = E[j, product_suno_idx:, product_suno_idx:]
        prod_atom_map_nums = atom_map_numbers[j, product_suno_idx:]
        prod_mol = mol_from_graph(
            node_list=prod_X,
            adjacency_matrix=prod_E,
            atom_types=atom_types,
            bond_types=bond_types,
            atom_map_numbers=prod_atom_map_nums.detach().cpu(),
        )
        if kekulize_molecule:
            prod_smiles = Chem.MolToSmiles(
                prod_mol, kekuleSmiles=True, isomericSmiles=True
            )
        else:
            prod_smiles = Chem.MolToSmiles(prod_mol, isomericSmiles=True)

        all_rxn_smiles.append(f"{rcts_smiles}>>{prod_smiles}")

    return all_rxn_smiles


def get_cano_smiles_from_dense(
    X,
    E,
    mol_assignment,
    atom_types,
    bond_types,
    return_dict=False,
    plot_dummy_nodes=False,
):
    """
    Returns canonical smiles of all the molecules in a reaction
    given a dense matrix representation of said reaction.
    Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
    Handles batched reactions.

    X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
    E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

    return: list of smiles of valid molecules from rxn.
    """
    assert X.ndim == 2 and E.ndim == 3, (
        "Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n)."
        + f" Got X.shape={X.shape} and E.shape={E.shape} instead."
    )
    # DEPRECATED AT THE MOMENT! (canonicalization process should use data_utils.canonicalize_smiles or data_utils.create_canonical_smiles_from_mol)
    # remove product

    kekulize_molecule = BT.AROMATIC not in bond_types

    # suno_idx = atom_types.index('SuNo')
    all_rxn_smiles = []
    for j in range(X.shape[0]):  # for each rxn in batch
        # suno_indices = (X[j,:]==suno_idx).nonzero(as_tuple=True)[0].cpu()
        # product_suno_idx = max(suno_indices)
        # parse reactants
        product_mol_index = mol_assignment[j].max().item()
        product_start_index = (
            (mol_assignment[j] == product_mol_index)
            .nonzero(as_tuple=True)[0]
            .min()
            .item()
        )

        rcts_X = X[j, :product_start_index]
        rcts_E = E[j, :product_start_index, :product_start_index]
        rcts_mol = mol_from_graph(
            node_list=rcts_X,
            adjacency_matrix=rcts_E,
            atom_types=atom_types,
            bond_types=bond_types,
            plot_dummy_nodes=plot_dummy_nodes,
        )
        if kekulize_molecule:
            rcts_smiles = Chem.MolToSmiles(
                rcts_mol, kekuleSmiles=True, isomericSmiles=True, canonical=True
            )
        else:
            rcts_smiles = Chem.MolToSmiles(rcts_mol, canonical=True)
        # parse product
        prod_X = X[j, product_start_index:]
        prod_E = E[j, product_start_index:, product_start_index:]
        prod_mol = mol_from_graph(
            node_list=prod_X,
            adjacency_matrix=prod_E,
            atom_types=atom_types,
            bond_types=bond_types,
        )
        if kekulize_molecule:
            prod_smiles = Chem.MolToSmiles(
                prod_mol, kekuleSmiles=True, isomericSmiles=True, canonical=True
            )
        else:
            prod_smiles = Chem.MolToSmiles(prod_mol, canonical=True)

        all_rxn_smiles.append(f"{rcts_smiles}>>{prod_smiles}")

    return all_rxn_smiles


def get_cano_smiles_from_dense_with_stereochem(
    dense_data, cfg, like_raw=False, with_atom_mapping=False, return_dict=False
):
    """
    wrapper for get_cano_smiles_from_dense_stereochem_, turns the dense_data object into a list of smiles.
    """
    gen_rxn_smiles = get_cano_smiles_from_dense_with_stereochem_(
        dense_data,
        rdkit_atom_types=cfg.dataset.atom_types,
        rdkit_bond_types=get_rdkit_bond_types(cfg.dataset.bond_types),
        rdkit_atom_charges=cfg.dataset.atom_charges,
        rdkit_atom_chiral_tags=get_rdkit_chiral_tags(cfg.dataset.atom_chiral_tags),
        rdkit_bond_dirs=get_rdkit_bond_dirs(cfg.dataset.bond_dirs),
        return_dict=return_dict,
        plot_dummy_nodes=cfg.test.plot_dummy_nodes,
        use_stereochemistry=cfg.dataset.use_stereochemistry,
        with_formal_charge_in_atom_symbols=cfg.dataset.with_formal_charge_in_atom_symbols,
        with_atom_mapping=with_atom_mapping,
        like_raw=like_raw,
    )

    return gen_rxn_smiles


def get_cano_smiles_from_dense_with_stereochem_(
    dense_data,
    rdkit_atom_types,
    rdkit_bond_types,
    rdkit_atom_charges,
    rdkit_atom_chiral_tags,
    with_formal_charge_in_atom_symbols,
    rdkit_bond_dirs,
    return_dict,
    use_stereochemistry,
    plot_dummy_nodes,
    with_atom_mapping,
    like_raw,
):
    """
    Returns canonical smiles of all the molecules in a reaction
    given a dense matrix representation of said reaction.
    Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
    Handles batched reactions.

    X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
    E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

    return: list of smiles of valid molecules from rxn.
    """
    X_atom_types, X_atom_charges, X_chiral_tags = (
        dense_data.X,
        dense_data.atom_charges,
        dense_data.atom_chiral,
    )
    E_bond_types, E_bond_dirs = dense_data.E, dense_data.bond_dirs
    atom_map_numbers, mol_assignment = (
        dense_data.atom_map_numbers,
        dense_data.mol_assignment,
    )

    assert X_atom_types.ndim == 2 and E_bond_types.ndim == 3, (
        "Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n)."
        + f" Got X.shape={X_atom_types.shape} and E.shape={E_bond_types.shape} instead."
    )

    # remove product
    # WRONG
    kekulize_molecule = BT.AROMATIC not in rdkit_bond_types
    kekulize_molecule = True
    all_rxn_smiles = []

    for j in range(X_atom_types.shape[0]):  # for each rxn in batch
        product_mol_index = mol_assignment[j].max().item()
        product_start_index = (
            (mol_assignment[j] == product_mol_index)
            .nonzero(as_tuple=True)[0]
            .min()
            .item()
        )

        # parse reactants
        rcts_X_atom_types = X_atom_types[j, :product_start_index]
        rcts_X_atom_charges = X_atom_charges[j, :product_start_index]
        rcts_X_chiral_tags = X_chiral_tags[j, :product_start_index]

        rcts_E = E_bond_types[j, :product_start_index, :product_start_index]
        rcts_E_bond_dirs = E_bond_dirs[j, :product_start_index, :product_start_index]
        if with_atom_mapping:
            rcts_atom_map_numbers = atom_map_numbers[j, :product_start_index]
        else:
            rcts_atom_map_numbers = None

        # # 17 and 49 (index of not-working)
        # if j == 49:
        #     breakpoint()

        # (atom_symbols, atom_charges, atom_chiral, bond_adjacency, bond_dirs, rdkit_atom_types, rdkit_atom_charges, rdkit_atom_chiral_tags, rdkit_bond_types, rdkit_bond_dirs)
        rcts_mol = mol_from_graph_with_stereochem_(
            atom_symbols=rcts_X_atom_types,
            atom_charges=rcts_X_atom_charges,
            atom_chiral=rcts_X_chiral_tags,
            bond_types_adj=rcts_E,
            bond_dirs_adj=rcts_E_bond_dirs,
            rdkit_atom_types=rdkit_atom_types,
            rdkit_atom_charges=rdkit_atom_charges,
            rdkit_atom_chiral_tags=rdkit_atom_chiral_tags,
            rdkit_bond_types=rdkit_bond_types,
            rdkit_bond_dirs=rdkit_bond_dirs,
            plot_dummy_nodes=plot_dummy_nodes,
            atom_map_numbers=rcts_atom_map_numbers,
            use_stereochemistry=use_stereochemistry,
            with_formal_charge_in_atom_symbols=with_formal_charge_in_atom_symbols,
        )

        # TODO: figure out how to canonicalize the molecule correctly to compare it to ground truth
        # if kekulize_molecule:
        #     # Chem.Kekulize(rcts_mol, clearAromaticFlags=True)
        #     rcts_smiles = Chem.MolToSmiles(rcts_mol, kekuleSmiles=True, isomericSmiles=True, canonical=True) # TODO: add canonical?
        # else: rcts_smiles = Chem.MolToSmiles(rcts_mol, canonical=True)
        if with_atom_mapping:
            rcts_smiles = data_utils.create_smiles_like_raw_from_mol(rcts_mol)
        else:
            rcts_smiles = data_utils.create_canonical_smiles_from_mol(rcts_mol)

        # parse product
        prod_X_atom_types = X_atom_types[j, product_start_index:]
        prod_X_atom_charges = X_atom_charges[j, product_start_index:]
        prod_X_chiral_tags = X_chiral_tags[j, product_start_index:]
        prod_E_bond_types = E_bond_types[j, product_start_index:, product_start_index:]
        prod_E_bond_dirs = E_bond_dirs[j, product_start_index:, product_start_index:]
        if with_atom_mapping:
            prod_atom_map_numbers = atom_map_numbers[j, product_start_index:]
        else:
            prod_atom_map_numbers = None

        # breakpoint()
        prod_mol = mol_from_graph_with_stereochem_(
            atom_symbols=prod_X_atom_types,
            atom_charges=prod_X_atom_charges,
            atom_chiral=prod_X_chiral_tags,
            bond_types_adj=prod_E_bond_types,
            bond_dirs_adj=prod_E_bond_dirs,
            rdkit_atom_types=rdkit_atom_types,
            rdkit_atom_charges=rdkit_atom_charges,
            rdkit_atom_chiral_tags=rdkit_atom_chiral_tags,
            rdkit_bond_types=rdkit_bond_types,
            rdkit_bond_dirs=rdkit_bond_dirs,
            plot_dummy_nodes=plot_dummy_nodes,
            atom_map_numbers=prod_atom_map_numbers,
            use_stereochemistry=use_stereochemistry,
            with_formal_charge_in_atom_symbols=with_formal_charge_in_atom_symbols,
        )
        # if kekulize_molecule:
        #     # Chem.Kekulize(prod_mol, clearAromaticFlags=True)
        #     prod_smiles = Chem.MolToSmiles(prod_mol, canonical=True)
        # else: prod_smiles = Chem.MolToSmiles(prod_mol, isomericSmiles=True)
        if with_atom_mapping:
            prod_smiles = data_utils.create_smiles_like_raw_from_mol(prod_mol)
        else:
            prod_smiles = data_utils.create_canonical_smiles_from_mol(prod_mol)

        all_rxn_smiles.append(f"{rcts_smiles}>>{prod_smiles}")

    return all_rxn_smiles


def get_mol_nodes(mol, atom_types, with_formal_charge=True, get_atom_mapping=False):
    atoms = mol.GetAtoms()
    atom_mapping = torch.zeros(len(atoms), dtype=torch.long)

    for i, atom in enumerate(atoms):
        if with_formal_charge:
            at = (
                atom.GetSymbol()
                if atom.GetFormalCharge() == 0
                else atom.GetSymbol() + f"{atom.GetFormalCharge():+}"
            )
        else:
            at = atom.GetSymbol()
        try:
            atom_type = torch.tensor(
                [atom_types.index(at)], dtype=torch.long
            )  # needs to be int for one hot
        except:
            log.info(f"at {at}\n")
            log.info(f"atom types: {atom_types}")
            # exit()
        atom_types_ = torch.cat((atom_types_, atom_type), dim=0) if i > 0 else atom_type
        atom_mapping[i] = atom.GetAtomMapNum()

    atom_feats = F.one_hot(atom_types_, num_classes=len(atom_types)).float()

    if get_atom_mapping:
        return atom_feats, atom_mapping

    return atom_feats


def get_mol_edges(mol, bond_types, offset=1):
    """
    Input:
        offset (optional): default: 1. To account for SuNo added at the beginning of the graph.
    """
    # print(f'len(mol.GetBonds()) {len(mol.GetBonds())}\n')
    for i, b in enumerate(mol.GetBonds()):
        beg_atom_idx = b.GetBeginAtom().GetIdx()
        end_atom_idx = b.GetEndAtom().GetIdx()
        e_beg = torch.tensor(
            [beg_atom_idx + offset, end_atom_idx + offset], dtype=torch.long
        ).unsqueeze(-1)
        e_end = torch.tensor(
            [end_atom_idx + offset, beg_atom_idx + offset], dtype=torch.long
        ).unsqueeze(-1)
        e_type = torch.tensor(
            [bond_types.index(b.GetBondType()), bond_types.index(b.GetBondType())],
            dtype=torch.long,
        )  # needs to be int for one hot
        begs = torch.cat((begs, e_beg), dim=0) if i > 0 else e_beg
        ends = torch.cat((ends, e_end), dim=0) if i > 0 else e_end
        edge_type = torch.cat((edge_type, e_type), dim=0) if i > 0 else e_type

    if len(mol.GetBonds()) > 0:
        edge_index = torch.cat((begs, ends), dim=1).mT.contiguous()
        edge_attr = F.one_hot(
            edge_type, num_classes=len(bond_types)
        ).float()  # add 1 to len of bonds to account for no edge
    else:
        edge_index = torch.tensor([]).long().reshape(2, 0)
        edge_attr = torch.tensor([]).float().reshape(0, len(bond_types))

    return edge_index, edge_attr


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


def mol_to_graph(
    mol,
    atom_types,
    bond_types,
    offset=0,
    with_explicit_h=True,
    with_formal_charge=True,
    get_atom_mapping=False,
    canonicalize_molecule=True,
):
    # MOSTLY DEPRECATED. See mol_to_graph_with_stereochem
    kekulize_molecule = BT.AROMATIC not in bond_types

    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    if canonicalize_molecule:
        mol = create_canonicalized_mol(mol)

    if kekulize_molecule:
        Chem.RemoveStereochemistry(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)

    if with_explicit_h:
        mol = Chem.AddHs(mol, explicitOnly=True)

    if not get_atom_mapping:
        m_nodes = get_mol_nodes(
            mol=mol,
            atom_types=atom_types,
            with_formal_charge=with_formal_charge,
            get_atom_mapping=get_atom_mapping,
        )
    else:
        m_nodes, atom_map = get_mol_nodes(
            mol=mol,
            atom_types=atom_types,
            with_formal_charge=with_formal_charge,
            get_atom_mapping=get_atom_mapping,
        )

    m_edge_index, m_edge_attr = get_mol_edges(
        mol=mol, bond_types=bond_types, offset=offset
    )

    if not get_atom_mapping:
        return m_nodes, m_edge_index, m_edge_attr
    else:
        return m_nodes, m_edge_index, m_edge_attr, atom_map


def mol_from_sparse_graph(node_list, edge_index, edge_attr, atom_types, bond_types):
    """A wrapper around mol_from_graph. Assumes that the first node index is 0 in edge_index, and that there is only one molecule"""
    adjacency_matrix = np.zeros(
        (node_list.shape[0], node_list.shape[0]), dtype=np.int64
    )
    # edge_index = edge_index - edge_index.min()
    edge_attr = (
        edge_attr.argmax(-1).detach().numpy()
        if type(edge_attr) == torch.Tensor
        else edge_attr.argmax(-1)
    )
    for idx, edge in enumerate(edge_index.T):
        adjacency_matrix[edge[0].item(), edge[1].item()] = edge_attr[idx]
        adjacency_matrix[edge[1].item(), edge[0].item()] = edge_attr[idx]
    molecule = mol_from_graph(node_list, adjacency_matrix, atom_types, bond_types)
    return molecule


def mol_from_graph(
    node_list,
    adjacency_matrix,
    atom_types,
    bond_types,
    plot_dummy_nodes=False,
    atom_map_numbers=None,
):
    """
    Convert graphs to RDKit molecules.

    node_list: the nodes of one molecule (n)
    adjacency_matrix: the adjacency_matrix of the molecule (n, n)
    atom_map_numbers: (optional) atom map numbers of one molecule (n)

    return: RDKit's editable mol object.
    """
    # fc = node_list[...,-1] # get formal charge of each atom
    # node_list = torch.argmax(node_list[...,:-1], dim=-1)
    # adjacency_matrix = torch.argmax(adjacency_matrix, dim=-1)
    # create empty editable mol object
    suno_type = atom_types.index("SuNo")

    mol = Chem.RWMol()
    if not plot_dummy_nodes:
        masking_atom = atom_types.index("U") if "U" in atom_types else 0
    else:
        masking_atom = 0

    node_to_idx = {}  # needed because using 0 to mark node paddings
    # add atoms to mol and keep track of index
    for i in range(len(node_list)):
        # ignore padding nodes
        if (
            node_list[i] == 0
            or node_list[i] == masking_atom
            or node_list[i] == suno_type
        ):
            continue
        at = atom_types[int(node_list[i])]
        fc = re.findall("[-+]\d+", at)
        s = re.split("[-+]\d+", at)[0]
        a = Chem.Atom(s)
        if len(fc) != 0:
            a.SetFormalCharge(int(fc[0]))
        if atom_map_numbers != None:
            a.SetAtomMapNum(atom_map_numbers[i].detach().cpu().item())
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    if type(adjacency_matrix) == torch.Tensor:  # hack to get this to work with
        adjacency_matrix = adjacency_matrix.detach().cpu().numpy().tolist()
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the symmetric matrix
            if iy <= ix:
                continue
            # only consider nodes parsed earlier (ignore empty nodes)
            if (ix not in node_to_idx.keys()) or (iy not in node_to_idx.keys()):
                continue
            # only consider valid edges types

            bond_type = bond_types[bond]
            if bond_type not in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]:
                continue
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    return mol


def smiles_to_graph_with_stereochem(smi, cfg):
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"Could not get rdkit mol object from mol_input={smi}\n"

    nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = (
        mol_to_graph_with_stereochem_(
            cfg,
            m,
            rdkit_atom_types=cfg.dataset.atom_types,
            rdkit_atom_charges=cfg.dataset.atom_charges,
            rdkit_atom_chiral_tags=get_rdkit_chiral_tags(cfg.dataset.atom_chiral_tags),
            rdkit_bond_types=get_rdkit_bond_types(cfg.dataset.bond_types),
            rdkit_bond_dirs=get_rdkit_bond_dirs(cfg.dataset.bond_dirs),
            with_formal_charge_in_atom_symbols=cfg.dataset.with_formal_charge_in_atom_symbols,
        )
    )

    return nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map


def mol_to_graph_with_stereochem(mol_input, cfg):
    """
    Wrapper for mol_to_graph_with_stereochem_ that takes a rdkit mol object as input and whole hydra cfg.

    mol_input: rdkit mol object
    cfg: hydra cfg dictionary
    """
    nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = (
        mol_to_graph_with_stereochem_(
            cfg,
            mol_input,
            rdkit_atom_types=cfg.dataset.atom_types,
            rdkit_atom_charges=cfg.dataset.atom_charges,
            rdkit_atom_chiral_tags=get_rdkit_chiral_tags(cfg.dataset.atom_chiral_tags),
            rdkit_bond_types=get_rdkit_bond_types(cfg.dataset.bond_types),
            rdkit_bond_dirs=get_rdkit_bond_dirs(cfg.dataset.bond_dirs),
            with_formal_charge_in_atom_symbols=cfg.dataset.with_formal_charge_in_atom_symbols,
        )
    )

    return nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map


def mol_to_graph_with_stereochem_(
    cfg,
    mol_input,
    rdkit_atom_types,
    rdkit_atom_charges,
    rdkit_atom_chiral_tags,
    rdkit_bond_types,
    rdkit_bond_dirs,
    with_formal_charge_in_atom_symbols=False,
):
    """
    m: rdkit molecule object
    returns: atom_symbols, atom_charges, atom_chiral, edge_index, bond_type, bond_dir, atom_map
    """
    assert (
        not with_formal_charge_in_atom_symbols
        or len([at for at in rdkit_atom_types if "+" in at or "-" in at]) > 0
    ), f"with_formal_charge_in_atom_symbols={with_formal_charge_in_atom_symbols} but no formal charges in rdkit_atom_types={rdkit_atom_types}.\n"
    assert (
        with_formal_charge_in_atom_symbols
        or len([at for at in rdkit_atom_types if "+" in at or "-" in at]) == 0
    ), f"with_formal_charge_in_atom_symbols={with_formal_charge_in_atom_symbols} but formal charges in rdkit_atom_types={rdkit_atom_types}.\n"

    # TODO: Could add a flag for this to be turned on or off

    if cfg.dataset.canonicalize_molecule:
        mol_input = create_canonicalized_mol(mol_input)

    Chem.Kekulize(
        mol_input, clearAromaticFlags=True
    )  # we need this because the graph will be represented in kekulized format

    # get atom symbols
    atom_symbols = F.one_hot(
        torch.tensor(
            [
                rdkit_atom_types.index(
                    get_atom_symbol(atom, with_formal_charge_in_atom_symbols)
                )
                for atom in mol_input.GetAtoms()
            ]
        ),
        num_classes=len(rdkit_atom_types),
    ).float()
    # get atom map number
    atom_map = torch.tensor([atom.GetAtomMapNum() for atom in mol_input.GetAtoms()])

    # get atom charges
    atom_charges = F.one_hot(
        torch.tensor(
            [
                rdkit_atom_charges.index(atom.GetFormalCharge())
                for atom in mol_input.GetAtoms()
            ]
        ),
        num_classes=len(rdkit_atom_charges),
    ).float()

    # get atom chirality
    atom_chiral = F.one_hot(
        torch.tensor(
            [
                rdkit_atom_chiral_tags.index(atom.GetChiralTag())
                for atom in mol_input.GetAtoms()
            ]
        ),
        num_classes=len(rdkit_atom_chiral_tags),
    ).float()

    # get bonds' end indices
    # TODO: duplicate and turn to torch tensor
    # bond_end_indices = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in m.GetBonds()]
    # TODO: why long and not smthg else
    edge_index = (
        torch.tensor(
            [
                [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                + [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
                for bond in mol_input.GetBonds()
            ]
        )
        .flatten()
        .reshape(-1, 2)
        .t()
        .contiguous()
        .long()
    )

    # get bond types
    # TODO: one-hot encode 2D tensor
    bond_types_list = [bond.GetBondType() for bond in mol_input.GetBonds()]
    # print(f'mol_input {mol_input}\n')
    # print(f'b_types {b_types}\n')
    if len(bond_types_list) > 0:
        bond_types = F.one_hot(
            torch.tensor(
                [
                    rdkit_bond_types.index(bond.GetBondType())
                    for bond in mol_input.GetBonds()
                ]
            ).repeat_interleave(2),
            num_classes=len(rdkit_bond_types),
        ).float()
    else:
        bond_types = torch.zeros((0, len(rdkit_bond_types)))

    bond_dirs_list = [
        rdkit_bond_dirs.index(bond.GetBondDir()) for bond in mol_input.GetBonds()
    ]
    if len(bond_dirs_list) > 0:
        bond_dirs = F.one_hot(
            torch.tensor(
                [
                    rdkit_bond_dirs.index(bond.GetBondDir())
                    for bond in mol_input.GetBonds()
                ]
            ).repeat_interleave(2),
            num_classes=len(rdkit_bond_dirs),
        ).float()
    else:
        bond_dirs = torch.zeros((0, len(rdkit_bond_dirs)))

    # get atom chirality
    cip_ranking = get_cip_ranking(
        mol_input
    )  # Note: this could be slightly slow for large molecules
    chiral_labels = []
    for atom in mol_input.GetAtoms():
        chiral_labels.append(
            switch_between_bond_cw_ccw_label_and_cip_based_label(
                atom, atom.GetChiralTag(), cip_ranking
            )
        )
    atom_chiral = F.one_hot(
        torch.tensor([rdkit_atom_chiral_tags.index(label) for label in chiral_labels]),
        num_classes=len(rdkit_atom_chiral_tags),
    ).float()

    # if Chem.rdchem.BondType.AROMATIC in bond_types.values():
    #     new_smi = Chem.MolToSmiles(m)
    #     print(f'still aromatic smi {new_smi}\n')
    #     print(f'bond_types {bond_types}\n')

    return (
        atom_symbols,
        atom_charges,
        atom_chiral,
        edge_index,
        bond_types,
        bond_dirs,
        atom_map,
    )


def mol_from_graph_with_stereochem_edge_idx_(
    atom_symbols,
    atom_charges,
    atom_chiral,
    edge_index,
    bond_types,
    bond_dirs,
    rdkit_atom_types,
    rdkit_atom_charges,
    rdkit_atom_chiral_tags,
    rdkit_bond_types,
    rdkit_bond_dirs,
):
    rw_mol = Chem.RWMol()

    # NOTE: assumes input is one-hot encoded
    symbols = atom_symbols.argmax(-1)
    charges = atom_charges.argmax(-1)
    chiral_tags = atom_chiral.argmax(-1)

    # NOTE: for now bond info is not one-hot encoded

    # add atoms
    node_to_idx = {}
    for i, (atom_symbol, atom_charge, tag) in enumerate(
        zip(symbols, charges, chiral_tags)
    ):
        atom = Chem.Atom(rdkit_atom_types[atom_symbol.item()])
        atom.SetFormalCharge(rdkit_atom_charges[atom_charge.item()])
        # atom.SetChiralTag(rdkit_atom_chiral_tags[tag.item()])
        molidx = rw_mol.AddAtom(atom)
        node_to_idx[i] = molidx

    # add bonds
    for i, (bond_type, bond_dir) in enumerate(zip(bond_types, bond_dirs)):
        if bond_type == 0:
            continue
        if edge_index[0, i] > edge_index[1, i]:
            continue
        beg_atom, end_atom = edge_index[0, i].item(), edge_index[1, i].item()
        rw_mol.AddBond(
            node_to_idx[beg_atom], node_to_idx[end_atom], rdkit_bond_types[bond_type]
        )
        rw_mol.GetBondBetweenAtoms(
            node_to_idx[beg_atom], node_to_idx[end_atom]
        ).SetBondDir(rdkit_bond_dirs[bond_dir])

    # add implicit info
    new_mol = rw_mol.GetMol()
    # new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

    # try:
    #     Chem.Kekulize(new_mol, clearAromaticFlags=True)
    # except:
    #     print(f'bond_types {bond_types}\n') # TODO: better error message

    return new_mol


def mol_from_graph_with_stereochem_(
    atom_symbols,
    atom_charges,
    atom_chiral,
    bond_types_adj,
    bond_dirs_adj,
    rdkit_atom_types,
    rdkit_atom_charges,
    rdkit_atom_chiral_tags,
    rdkit_bond_types,
    rdkit_bond_dirs,
    with_formal_charge_in_atom_symbols,
    use_stereochemistry,
    plot_dummy_nodes,
    atom_map_numbers,
):
    if "SuNo" in rdkit_atom_types:
        suno_type = rdkit_atom_types.index("SuNo")
    else:
        suno_type = 0

    rw_mol = Chem.RWMol()
    if not plot_dummy_nodes:
        dummy_atom = rdkit_atom_types.index("U") if "U" in rdkit_atom_types else 0
    else:
        dummy_atom = 0

    # symbols = atom_symbols.argmax(-1)
    # charges = atom_charges.argmax(-1)
    # chiral_tags = atom_chiral.argmax(-1)

    # TODO: This should handle the case where the charges are represented in the atom symbols

    # add atoms
    node_to_idx = {}
    for i, (atom_symbol, atom_charge) in enumerate(zip(atom_symbols, atom_charges)):
        if atom_symbol == 0 or atom_symbol == dummy_atom or atom_symbol == suno_type:
            continue
        if with_formal_charge_in_atom_symbols:
            atom_type, charge = split_atom_symbol_from_formal_charge(
                rdkit_atom_types[atom_symbol.item()]
            )
        else:
            atom_type, charge = (
                rdkit_atom_types[atom_symbol.item()],
                rdkit_atom_charges[atom_charge.item()],
            )

        atom = Chem.Atom(atom_type)
        atom.SetFormalCharge(charge)
        if atom_map_numbers != None:
            atom.SetAtomMapNum(atom_map_numbers[i].detach().cpu().item())
        molidx = rw_mol.AddAtom(atom)
        node_to_idx[i] = molidx

    # print(f'node_to_idx {node_to_idx}\n')

    # add bonds
    if type(bond_types_adj) == torch.Tensor:  # hack to get this to work with
        bond_types_adj = bond_types_adj.detach().cpu().numpy().tolist()
        bond_dirs_adj = bond_dirs_adj.detach().cpu().numpy().tolist()
    for ix, row in enumerate(bond_types_adj):
        for iy, bond in enumerate(row):
            # only traverse half the symmetric matrix
            if iy > ix:
                continue
            # only consider nodes parsed earlier (ignore empty nodes)
            if (ix not in node_to_idx.keys()) or (iy not in node_to_idx.keys()):
                continue
            # only consider valid edges types
            bond_type = rdkit_bond_types[bond]
            if bond_type not in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]:
                continue
            rw_mol.AddBond(node_to_idx[iy], node_to_idx[ix], bond_type)
            # add bond dirs
            bond_dir = rdkit_bond_dirs[bond_dirs_adj[iy][ix]]
            if use_stereochemistry:
                rw_mol.GetBondBetweenAtoms(node_to_idx[iy], node_to_idx[ix]).SetBondDir(
                    bond_dir
                )

    # Add the chiral tags
    if use_stereochemistry:
        cip_ranking = get_cip_ranking(rw_mol)
        for i, chiral_tag in enumerate(atom_chiral):
            if i not in node_to_idx.keys():
                continue
            atom = rw_mol.GetAtomWithIdx(node_to_idx[i])
            try:
                atom.SetChiralTag(
                    switch_between_bond_cw_ccw_label_and_cip_based_label(
                        atom, rdkit_atom_chiral_tags[chiral_tag.item()], cip_ranking
                    )
                )
            except:
                switch_between_bond_cw_ccw_label_and_cip_based_label(
                    atom, rdkit_atom_chiral_tags[chiral_tag.item()], cip_ranking
                )

    # add implicit info
    # new_mol = rw_mol.GetMol()
    # new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

    # try:
    #     Chem.Kekulize(new_mol, clearAromaticFlags=True)
    # except:
    #     print(f'bond_types\n') # TODO: better error message

    return rw_mol


def connect_mol_to_supernode(mol, atom_types, bond_types, supernode_nb=1):
    s_nodes = F.one_hot(
        torch.tensor([atom_types.index("SuNo")], dtype=torch.long),
        num_classes=len(atom_types),
    ).float()

    # connect all atoms to supernode (idx supernode_nb - 1)
    # print(f'len(mol.GetAtoms()) {len(mol.GetAtoms())}\n')
    for i, a in enumerate(mol.GetAtoms()):
        e_beg = torch.tensor(
            [supernode_nb - 1, a.GetIdx() + supernode_nb], dtype=torch.long
        ).unsqueeze(-1)
        e_end = torch.tensor(
            [a.GetIdx() + supernode_nb, supernode_nb - 1], dtype=torch.long
        ).unsqueeze(-1)

        begs = torch.cat((begs, e_beg), dim=0) if i > 0 else e_beg
        ends = torch.cat((ends, e_end), dim=0) if i > 0 else e_end

    s_edge_index = torch.cat((begs, ends), dim=1).mT.contiguous()
    edge_type = torch.full(
        size=(begs.shape[0],), fill_value=bond_types.index("mol"), dtype=torch.long
    )  # needs to be int for one hot
    s_edge_attr = F.one_hot(
        edge_type, num_classes=len(bond_types)
    ).float()  # add 1 to len of bonds to account for no edge

    # print(f's_edge_index.shape {s_edge_index.shape}\n')

    return s_nodes, s_edge_index, s_edge_attr


def rxn_plots(rxns, atom_types, bond_types):
    """rxns is a Placeholder object that contains multiple reactions"""
    num_rxns = len(rxns.X)
    rxn_imgs = []
    for i in range(num_rxns):
        rxn = graph.PlaceHolder(
            X=rxns.X[i : i + 1],
            E=rxns.E[i : i + 1],
            y=rxns.y[i : i + 1],
            node_mask=rxns.node_mask[i : i + 1],
        )
        rxn_img = rxn_plot(
            rxn, atom_types, bond_types, filename="test.png"
        )  # For now the filename is hardcoded, doesn't do anything interesting
        rxn_imgs.append(rxn_img)
    return rxn_imgs


def rxn_plot(rxn, cfg, filename="test.png", return_smarts=False):
    """
    Return a plot of a rxn given a rxn graph (with supernodes).
    """
    # rxn_smrts = rxn_from_graph_supernode(data=rxn, atom_types=atom_types, bond_types=bond_types, plot_dummy_nodes=plot_dummy_nodes)
    rxn_smrts = get_cano_smiles_from_dense_with_stereochem(rxn, cfg)[0]

    # try:
    #     rxn_obj = Reactions.ReactionFromSmarts(rxn_smrts)
    # except:
    #     log.info(f'Could not turn this rxn_smrts to rxn_obj when trying to plot: {rxn_smrts}\n')
    #     rxn_obj = Reactions.ReactionFromSmarts(''+'>>'+rxn_smrts.split('>>')[-1])

    # drawer = rdMolDraw2D.MolDraw2DCairo(800, 200)
    # drawer.SetFontSize(1.0)
    # drawer.DrawReaction(rxn_obj)
    # drawer.FinishDrawing()
    # drawer.WriteDrawingText(filename)

    rct_images, prod_image = draw_molecules_from_reaction(rxn_smrts)
    rxn_img = combine_reaction_image_from_reactant_product_images(
        rct_images, prod_image
    )

    # rxn_img = Draw.ReactionToImage(rxn_obj, wedgeBonds=True) # TODO: investigate fancy reaction plotting

    if return_smarts:
        return rxn_img, rxn_smrts
    return rxn_img


def draw_molecules_from_reaction(rxn_smrts, draw_with_atom_mapping=False):
    rxn_obj = Reactions.ReactionFromSmarts(rxn_smrts)

    # Function to highlight and draw a molecule
    def draw_mol(mol):
        try:
            smi = Chem.MolToSmiles(mol)
            mol_ = Chem.MolFromSmiles(smi)
            # if mol_ is None:
            #     return Draw.MolToImage(mol, size=(int(30*np.sqrt(mol.GetNumAtoms())), int(30*np.sqrt(mol.GetNumAtoms()))), wedgeBonds=True, includeAtomNumbers=draw_with_atom_mapping)
            if mol_ is None:
                # Return an empty PIL image (if we don't handle this possibility, the will crash in Draw.MolToImage without saying anything)
                # turn off stereochem
                Chem.RemoveStereochemistry(mol)
                return Draw.MolToImage(
                    mol,
                    size=(
                        int(30 * np.sqrt(mol.GetNumAtoms())),
                        int(30 * np.sqrt(mol.GetNumAtoms())),
                    ),
                    wedgeBonds=False,
                    includeAtomNumbers=draw_with_atom_mapping,
                )
                smi = Chem.MolToSmiles(mol)
                mol_ = Chem.MolFromSmiles(smi)
                if mol_ is None:
                    return Image.new("RGB", (20, 20), "black")
                return Draw.MolToImage(
                    mol,
                    size=(
                        int(30 * np.sqrt(mol.GetNumAtoms())),
                        int(30 * np.sqrt(mol.GetNumAtoms())),
                    ),
                    wedgeBonds=False,
                    includeAtomNumbers=draw_with_atom_mapping,
                )
            return Draw.MolToImage(
                mol_,
                size=(
                    int(30 * np.sqrt(mol.GetNumAtoms())),
                    int(30 * np.sqrt(mol.GetNumAtoms())),
                ),
                wedgeBonds=True,
                includeAtomNumbers=draw_with_atom_mapping,
            )
        except:
            # This can't draw stereochemistry, sadly
            return Draw.MolToImage(
                mol,
                size=(
                    int(30 * np.sqrt(mol.GetNumAtoms())),
                    int(30 * np.sqrt(mol.GetNumAtoms())),
                ),
                wedgeBonds=True,
                includeAtomNumbers=draw_with_atom_mapping,
            )

    reactant_images = []
    for i in range(len(rxn_obj.GetReactants())):
        # create the atommapnum into an index...?
        if not draw_with_atom_mapping:
            for atom in rxn_obj.GetReactants()[i].GetAtoms():
                atom.SetAtomMapNum(0)
        # highlight_atoms = list(range(rxn_obj.GetReactants()[highlighted_molecule_idx].GetNumAtoms()))
        img_data = draw_mol(rxn_obj.GetReactants()[i])
        if img_data is not None:
            reactant_images.append(img_data)

    if not draw_with_atom_mapping:
        for atom in rxn_obj.GetProducts()[0].GetAtoms():
            atom.SetAtomMapNum(0)
    prod_mol = rxn_obj.GetProducts()[0]
    product_image = draw_mol(rxn_obj.GetProducts()[0])
    return reactant_images, product_image


def combine_reaction_image_from_reactant_product_images(reactant_images, product_image):
    # Let's assume `images` is a list of your PIL images of the reactants and products in order
    images = reactant_images + [product_image]

    # Calculate total width and max height
    total_width = sum(image.width for image in images) + 60 * (
        len(images) - 1
    )  # Adding space for arrows or text
    max_height = max(image.height for image in images)

    # Create a new image with enough space
    combined_image = Image.new("RGB", (total_width, max_height), "white")

    # Paste each image onto the combined image
    x_offset = 0
    for image in images:
        y_offset = int((max_height - image.height) / 2)
        combined_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + 60  # Space for arrows or text between molecules

    # Draw the reaction arrow or text
    draw = ImageDraw.Draw(combined_image)
    arrow_start = sum([img.width for img in images[:-1]]) + 60 * (len(images) - 2) + 20
    # arrow_end = 25
    # draw_text(combined_image, position=(arrow_start, max_height // 2), text="->", scale_factor=6)
    # font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", 28)
    # font_path = os.path.expanduser("~/.local/share/fonts/roboto/Roboto-Thin.ttf")
    # font = ImageFont.truetype(font_path, 28)

    draw.text(((arrow_start), max_height // 2 - 14), "->", fill=(0, 0, 0))

    plus_starts = [
        sum([img.width for img in images[: -1 - i]]) + 60 * (len(images) - 2 - i) + 20
        for i in range(1, len(images) - 1)
    ]
    for start in plus_starts:
        draw.text(((start), max_height // 2 - 14), "+", fill=(0, 0, 0))

    return combined_image


def rxn_to_graph_supernode(
    mol,
    atom_types,
    bond_types,
    supernode_nb=1,
    with_explicit_h=True,
    with_formal_charge=True,
    add_supernode_edges=True,
    get_atom_mapping=False,
    canonicalize_molecule=True,
):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    if not get_atom_mapping:
        m_nodes, m_edge_index, m_edge_attr = mol_to_graph(
            mol=mol,
            atom_types=atom_types,
            bond_types=bond_types,
            offset=supernode_nb,
            with_explicit_h=with_explicit_h,
            with_formal_charge=with_formal_charge,
            get_atom_mapping=get_atom_mapping,
            canonicalize_molecule=canonicalize_molecule,
        )
    else:
        m_nodes, m_edge_index, m_edge_attr, atom_map = mol_to_graph(
            mol=mol,
            atom_types=atom_types,
            bond_types=bond_types,
            offset=supernode_nb,
            with_explicit_h=with_explicit_h,
            with_formal_charge=with_formal_charge,
            get_atom_mapping=get_atom_mapping,
            canonicalize_molecule=canonicalize_molecule,
        )
        # add 0 for SuNo node
        atom_map = torch.cat((torch.zeros(1, dtype=torch.long), atom_map), dim=0)
        # print(f'm_edge_index {m_edge_index.shape}\n')

    print(f"==== ABOUT TO ADD SN EDGES: add_supernode_edges {add_supernode_edges}\n")
    if add_supernode_edges:
        s_nodes, s_edge_index, s_edge_attr = connect_mol_to_supernode(
            mol=mol,
            atom_types=atom_types,
            bond_types=bond_types,
            supernode_nb=supernode_nb,
        )
        # print(f'm_nodes.shape {m_nodes.shape}\n')
        g_nodes = torch.cat([s_nodes, m_nodes], dim=0)
        g_edge_index = torch.cat(
            [s_edge_index, m_edge_index], dim=1
        )  # s/m_edge_index: (2, n_edges)
        g_edge_attr = torch.cat([s_edge_attr, m_edge_attr], dim=0)
    else:
        s_nodes = F.one_hot(
            torch.tensor([atom_types.index("SuNo")], dtype=torch.long),
            num_classes=len(atom_types),
        ).float()
        g_nodes = torch.cat([s_nodes, m_nodes], dim=0)
        g_edge_index = m_edge_index
        g_edge_attr = m_edge_attr

    if not get_atom_mapping:
        return g_nodes, g_edge_index, g_edge_attr
    else:
        return g_nodes, g_edge_index, g_edge_attr, atom_map


def rxn_from_graph_supernode(data, atom_types, bond_types, plot_dummy_nodes=True):
    if type(data) != graph.PlaceHolder:
        data_ = graph.to_dense(data)
        data_ = data_.mask(data_.node_mask, collapse=True)
    else:
        data_ = copy.deepcopy(data)

    assert (
        data_.X.shape[0] == 1
    ), "Function expects a single example, batch given instead."

    all_rxn_smiles = get_cano_smiles_from_dense(
        data_.X,
        data_.E,
        data_.mol_assignment,
        atom_types,
        bond_types,
        return_dict=False,
        plot_dummy_nodes=plot_dummy_nodes,
    )

    # suno_idx = atom_types.index('SuNo') # offset because index 0 is for no node
    # suno_indices = (X.squeeze()==suno_idx).nonzero(as_tuple=True)[0].cpu()
    # cutoff = 1 if 0 in suno_indices else 0
    # mols_atoms = torch.tensor_split(X.squeeze(), suno_indices, dim=-1)[cutoff:] # ignore first set (SuNo)
    # mols_edges = torch.tensor_split(E.squeeze(), suno_indices, dim=-1)[cutoff:]

    # smiles = []
    # for i, mol_atoms in enumerate(mols_atoms): # for each mol in sample
    #     mol_edges_to_all = mols_edges[i]
    #     mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[cutoff:] # ignore first because empty SuNo set
    #     mol_edges = mol_edges_t[i]
    #     cutoff = 1 if suno_idx in mol_atoms else 0
    #     mol_atoms = mol_atoms[cutoff:] # (n-1)
    #     mol_edges = mol_edges[cutoff:,:][:,cutoff:] # (n-1, n-1)

    #     mol = mol_from_graph(node_list=mol_atoms, adjacency_matrix=mol_edges,
    #                          atom_types=atom_types, bond_types=bond_types,
    #                          plot_dummy_nodes=plot_dummy_nodes)
    #     #Chem.SanitizeMol(mol)
    #     smi = Chem.MolToSmiles(mol)
    #     smiles.append(smi)

    # rxn_smrts = '>>' + smiles[-1]

    # for smi in smiles[1:-1]:
    #     rxn_smrts = '.' + smi + rxn_smrts

    # rxn_smrts = smiles[0] + rxn_smrts

    return all_rxn_smiles[0]


def rxn_from_graph_supernode_legacy(
    data, atom_types, bond_types, plot_dummy_nodes=True
):
    if type(data) != graph.PlaceHolder:
        data_ = graph.to_dense(data)
        data_ = data_.mask(data_.node_mask, collapse=True)
    else:
        data_ = copy.deepcopy(data)

    assert (
        data_.X.shape[0] == 1
    ), "Function expects a single example, batch given instead."

    X = data_.X.squeeze()
    E = data_.E.squeeze()
    suno_idx = atom_types.index("SuNo")  # offset because index 0 is for no node
    suno_indices = (X.squeeze() == suno_idx).nonzero(as_tuple=True)[0].cpu()
    cutoff = 1 if 0 in suno_indices else 0
    mols_atoms = torch.tensor_split(X.squeeze(), suno_indices, dim=-1)[
        cutoff:
    ]  # ignore first set (SuNo)
    mols_edges = torch.tensor_split(E.squeeze(), suno_indices, dim=-1)[cutoff:]

    smiles = []
    for i, mol_atoms in enumerate(mols_atoms):  # for each mol in sample
        mol_edges_to_all = mols_edges[i]
        mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[
            cutoff:
        ]  # ignore first because empty SuNo set
        mol_edges = mol_edges_t[i]
        cutoff = 1 if suno_idx in mol_atoms else 0
        mol_atoms = mol_atoms[cutoff:]  # (n-1)
        mol_edges = mol_edges[cutoff:, :][:, cutoff:]  # (n-1, n-1)

        mol = mol_from_graph(
            node_list=mol_atoms,
            adjacency_matrix=mol_edges,
            atom_types=atom_types,
            bond_types=bond_types,
            plot_dummy_nodes=plot_dummy_nodes,
        )
        # Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        smiles.append(smi)

    rxn_smrts = ">>" + smiles[-1]

    for smi in smiles[1:-1]:
        rxn_smrts = "." + smi + rxn_smrts

    rxn_smrts = smiles[0] + rxn_smrts

    return rxn_smrts


def check_valid_molecule(mol_atoms, mol_edges, atom_types, bond_types):
    """
    Checks if a molecule graph represents a valid molecule by synthesizing it using RDKit.

    Input:
        mol_atoms: list of atom types. (n,
        mol_edges: adjacency matrix with edge types. (n, n)

    Output:
        smiles (str): smiles string from the molecule graph.
        mol_valid (bool): boolean representing molecule validity. 1 if valid, None if not valid.
    """
    assert mol_atoms.ndim == 1 and mol_edges.ndim == 2, (
        "Expected mol_atoms to be of dimension (n,) and mol_edges to be of dimension (n,n)."
        + f"Instead got mol_atoms.shape={mol_atoms.shape} and mol_edges.shape={mol_edges.shape}"
    )

    mol = mol_from_graph(
        node_list=mol_atoms,
        adjacency_matrix=mol_edges,
        atom_types=atom_types,
        bond_types=bond_types,
    )
    smiles = Chem.MolToSmiles(mol, canonical=True)
    # check if molecule is valid by synthesizing it (return None if invalid).
    mol_valid = Chem.MolFromSmiles(smiles)

    if mol_valid == None:
        issue = "invalid"
    else:
        issue = "no_issue"

    return smiles, mol_valid, issue


def canonicalize_smiles_fcd(smiles):
    # Canonicalize smiles with the convention used for the FCD computations
    new_smiles = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            s = Chem.MolToSmiles(mol)
        new_smiles.append(s)
    return new_smiles


def canonicalize_smiles_our(smiles, kekulize_molecule=True):
    # DEPRECATED, SOME OLD STUFF
    # Canonicalize smiles with our convention
    new_smiles = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None and kekulize_molecule:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            Chem.RemoveStereochemistry(mol)
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
        else:
            s = Chem.MolToSmiles(mol)
        new_smiles.append(s)
    return new_smiles


def check_valid_molecule_fragment(mol_atoms, mol_edges, atom_types, bond_types):
    """generated: list of couples (positions, atom_types)"""
    # DEPRECATED, SOME OLD STUFF
    kekulize_molecule = BT.AROMATIC not in bond_types
    mol = mol_from_graph(
        node_list=mol_atoms,
        adjacency_matrix=mol_edges,
        atom_types=atom_types,
        bond_types=bond_types,
    )
    if kekulize_molecule:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.RemoveStereochemistry(mol)
        mol = Chem.AddHs(mol, explicitOnly=True)
        smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
    else:
        smi = Chem.MolToSmiles(mol)
    if smi is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            # num_components = len(mol_frags)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            # What's not here that is in the FCD calculations?
            if kekulize_molecule:
                smi = Chem.MolToSmiles(largest_mol, kekuleSmiles=True, canonical=True)
            else:
                smi = Chem.MolToSmiles(largest_mol)
            # Additional step of canonization
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
            issue = "no_issue"
        except Chem.rdchem.AtomValenceException:
            # print("Valence error in GetmolFrags")
            mol = None
            issue = "wrong_valence"
        except Chem.rdchem.KekulizeException:
            # print("Can't kekulize molecule")
            mol = None
            issue = "wrong_kekule"
    else:
        issue = "no_smiles"
    return smi, mol, issue


def check_valid_mol_batch(X, E, atom_types, bond_types):
    """
    Check the validity of molecules in a batch.

    Input:
        X, E: node and edges of molecules in batch format.
    Output:
        valid: the average validity over the batch
    """
    all_smis = []
    all_valid_smis = []
    for i in range(X.shape[0]):
        smi, mol, issue = check_valid_molecule_fragment(
            mol_atoms=X[i, :],
            mol_edges=E[i, :, :],
            atom_types=atom_types,
            bond_types=bond_types,
        )
        all_smis.append(smi)
        if issue == "no_issue":
            all_valid_smis.append(smi)

    return len(all_valid_smis) / X.shape[0], all_valid_smis, all_smis


def check_unique_mols(all_smis):
    if len(all_smis) == 0:
        return 0.0, []
    return len(set(all_smis)) / len(all_smis), set(all_smis)


def check_novel_mols(smiles, train_dataset_path):
    train_smiles = open(train_dataset_path, "r").readlines()
    # TODO: This is pretty slow doing this canonicalization again and again
    # Could precompute them and try to see the optimal places where to apply
    # Also should maybe refactorize anyways a bit to make the different
    # canonicalization steps more explicit

    # NOTE: Now uses multiset to also take into account the number of occurences of each molecule
    # in the generated data
    train_smiles = Multiset(canonicalize_smiles_our(train_smiles))
    smiles = Multiset(canonicalize_smiles_our(smiles))
    if len(smiles) == 0:
        return 0.0
    not_novel = smiles.intersection(train_smiles)
    novelty = 1 - len(not_novel) / len(smiles)

    return novelty


def check_stable_mols(atoms, edges, atom_types, bond_types):
    n_mols_stability, n_atoms_stability, n_atoms = 0, 0, 0

    for i in range(atoms.shape[0]):
        stable_mol, stable_atoms, all_atoms = check_stable_single_mol(
            atoms[i, ...], edges[i, ...], atom_types, bond_types
        )
        n_mols_stability += stable_mol
        n_atoms_stability += len(stable_atoms)
        n_atoms += len(all_atoms)

    return n_mols_stability / atoms.shape[0], n_atoms_stability / n_atoms


def check_stable_single_mol(atoms, edges, atom_types, bond_types, allowed_bonds):
    # compute the number of edges per atom
    # compare to the number of allowed edges of the given atom type
    # DEPRECATED, SOME OLD STUFF
    kekulize_molecule = BT.AROMATIC not in bond_types

    assert (
        len(allowed_bonds) > 0
    ), "allowed_bonds should be a dictionary with atom types as keys and allowed bonds as values. Got len(allowed_bonds)==0."
    idx = atoms != 0
    atoms = atoms[idx]
    edges = edges[idx, :][:, idx]

    # need to get rdkit objects to fill in implicit hydrogens
    mol = mol_from_graph(
        node_list=atoms,
        adjacency_matrix=edges,
        atom_types=atom_types,
        bond_types=bond_types,
    )

    if kekulize_molecule:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.RemoveStereochemistry(mol)
        mol = Chem.AddHs(mol, explicitOnly=True)
    # smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)

    stable_atoms = []
    all_atoms = []
    for i, a in enumerate(mol.GetAtoms()):
        symb = (
            a.GetSymbol()
            if a.GetFormalCharge() == 0
            else a.GetSymbol() + f"{a.GetFormalCharge():+}"
        )
        all_atoms.append(symb)
        if a.GetTotalValence() in allowed_bonds[symb]:
            stable_atoms.append(symb)

    stable_mol = len(stable_atoms) == atoms.shape[0]

    return stable_mol, stable_atoms, all_atoms


def check_valid_product_in_rxn(X, E, true_rxn_smiles, atom_types, bond_types):
    """
    Checks if the product given in dense tensor format is valid.

    Input:
        X: nodes of a reaction in (discrete) matrix dense format. (bs*n_samples, n)
        E: Edges of a reaction in (discrete) matrix dense format. (bs*n_samples, n, n)
        n_samples: number of samples generated for each rxn.

    Output:
        avg_validity: avg validity of each set of precursors generated for each test product. (bs*samples,)
    """
    # DEPRECATED, OLD STUFF
    kekulize_molecule = BT.AROMATIC not in bond_types

    assert X.ndim == 2 and E.ndim == 3, (
        "Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n)."
        + f" Got X.shape={X.shape} and E.shape={E.shape} instead."
    )

    all_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(
        device
    )  # all generated precursors are valid
    atleastone_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(
        device
    )  # at least one generated precursor is valid
    suno_idx = atom_types.index("SuNo")
    gen_rxns = {}
    for j in range(X.shape[0]):  # for each rxn in batch
        log.debug(f"True rxn: {true_rxn_smiles[j]}\n")
        suno_indices = (X[j, :] == suno_idx).nonzero(as_tuple=True)[0].cpu()
        log.debug(f"\nChecking precursors {j}\n")

        # TODO: refactor to make more generic => ignore whatever is masked
        mol_atoms = torch.tensor_split(X[j, :], suno_indices, dim=-1)[
            -1
        ]  # get only last set (product)
        mol_atoms = mol_atoms[1:]  # ignore first atom because it's SuNo, (n-1)
        mol_edges_to_all = torch.tensor_split(E[j, :, :], suno_indices, dim=-1)[-1]
        mol_edges = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[
            -1
        ]  # ignore first because empty SuNo set
        mol_edges = mol_edges[1:, :][
            :, 1:
        ]  # ignore edges to/from first atom because SuNo, (n-1,n-1)
        # smi, mol, issue = check_valid_molecule(mol_atoms, mol_edges, atom_types, bond_types)
        ## use the same function as when evaluating single molecules
        # smi, mol, issue = check_valid_molecule_fragment(mol_atoms=mol_atoms, mol_edges=mol_edges,
        #                                                 atom_types=atom_types, bond_types=bond_types)
        mol = mol_from_graph(
            node_list=mol_atoms,
            adjacency_matrix=mol_edges,
            atom_types=atom_types,
            bond_types=bond_types,
        )
        try:
            if kekulize_molecule:
                smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
            else:
                smi = Chem.MolToSmiles(mol)
            issue = "no_issue"
        except:
            issue = "invalid"

        all_valid_per_sample = 0
        all_mols_in_prods = 0
        log.debug(f"Product #{j}: {smi}\n")
        if issue == "no_issue":
            log.debug(f"valid products!\n")
            prods = smi.split(".")
            all_mols_in_prods += len(prods)
            for i_p, p in enumerate(prods):
                mol_p = Chem.MolFromSmiles(p)
                try:
                    if kekulize_molecule:
                        smi_p = Chem.MolToSmiles(mol_p, kekuleSmiles=True)
                    else:
                        smi_p = Chem.MolToSmiles(mol_p)
                    all_valid_per_sample += 1
                except:
                    log.debug(f"p {i_p} is invalid\n")

        rct = true_rxn_smiles[j].split(">>")[0]
        gen_rxn = rct + ">>" + smi

        if rct in gen_rxns.keys():
            gen_rxns[rct].append(gen_rxn)
        else:
            gen_rxns[rct] = [gen_rxn]

        all_valid[j] = float(
            (all_valid_per_sample == all_mols_in_prods) and all_mols_in_prods > 0
        )
        atleastone_valid[j] = float(
            (all_valid_per_sample > 0) and all_mols_in_prods > 0
        )

    return all_valid, atleastone_valid, gen_rxns


def check_valid_reactants_in_rxn(
    X, E, true_rxn_smiles, n_samples, atom_types, bond_types
):
    """
    Checks if the molecules given in dense tensor format are valid.

    Input:
        X: nodes of a reaction in (discrete) matrix dense format. (bs*n_samples, n)
        E: Edges of a reaction in (discrete) matrix dense format. (bs*n_samples, n, n)
        n_samples: number of samples generated for each rxn.

    Output:
        avg_validity: avg validity of each set of precursors generated for each test product. (bs*samples,)
    """
    # DEPRECATED, OLD STUFF
    kekulize_molecule = BT.AROMATIC not in bond_types

    assert X.ndim == 2 and E.ndim == 3, (
        "Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n)."
        + f" Got X.shape={X.shape} and E.shape={E.shape} instead."
    )

    all_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(
        device
    )  # all generated precursors are valid
    atleastone_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(
        device
    )  # at least one generated precursor is valid
    suno_idx = atom_types.index("SuNo")
    gen_rxns = {}
    for j in range(X.shape[0]):  # for each rxn in batch
        log.debug(f"True rxn: {true_rxn_smiles[j]}\n")
        suno_indices = (X[j, :] == suno_idx).nonzero(as_tuple=True)[0].cpu()
        cutoff = 1 if 0 in suno_indices else 0
        # TODO: refactor to make more generic => ignore whatever is masked
        mols_atoms = torch.tensor_split(X[j, :], suno_indices, dim=-1)[
            cutoff:-1
        ]  # ignore first set (SuNo) and last set (product)
        mols_edges = torch.tensor_split(E[j, :, :], suno_indices, dim=-1)[cutoff:-1]

        log.debug(
            f"\nChecking precursors {j}, total nb of molecules: {len(mols_atoms)}\n"
        )
        all_valid_per_sample = 0
        all_mols_in_rcts = 0
        gen_rxn = ""
        for i, mol_atoms in enumerate(mols_atoms):  # for each mol in sample
            cutoff = 1 if 0 in suno_indices else 0
            mol_edges_to_all = mols_edges[i]
            mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[
                cutoff:
            ]  # ignore first because empty SuNo set
            mol_edges = mol_edges_t[i]
            cutoff = 1 if suno_idx in mol_atoms else 0
            mol_atoms = mol_atoms[cutoff:]  # (n-1)
            mol_edges = mol_edges[cutoff:, :][:, cutoff:]  # (n-1, n-1)
            # smi, mol, issue = check_valid_molecule(mol_atoms, mol_edges, atom_types, bond_types)
            ## use the same function as when evaluating single molecules
            # smi, mol, issue = check_valid_molecule_fragment(mol_atoms=mol_atoms, mol_edges=mol_edges,
            #                                                 atom_types=atom_types, bond_types=bond_types)
            mol = mol_from_graph(
                node_list=mol_atoms,
                adjacency_matrix=mol_edges,
                atom_types=atom_types,
                bond_types=bond_types,
            )
            try:
                if kekulize_molecule:
                    smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
                else:
                    smi = Chem.MolToSmiles(mol)
                issue = "no_issue"
                log.debug(f"Molecule #{i}: {smi}\n")
            except:
                issue = "invalid"

            if issue == "no_issue":
                log.debug(f"valid reactants!\n")
                rcts = smi.split(".")
                all_mols_in_rcts += len(rcts)
                for i_r, r in enumerate(rcts):
                    mol_r = Chem.MolFromSmiles(r)
                    try:
                        if kekulize_molecule:
                            smi_r = Chem.MolToSmiles(mol_r, kekuleSmiles=True)
                        else:
                            smi_r = Chem.MolToSmiles(mol_r)
                        all_valid_per_sample += 1
                    except:
                        log.debug(f"r {i_r} is invalid\n")

            gen_rxn = smi if gen_rxn == "" else gen_rxn + "." + smi

        product = true_rxn_smiles[j].split(">>")[-1]
        gen_rxn += ">>" + product + "\n"
        if product in gen_rxns.keys():
            gen_rxns[product].append(gen_rxn)
        else:
            gen_rxns[product] = [gen_rxn]

        all_valid[j] = float(
            (all_valid_per_sample == all_mols_in_rcts) and all_mols_in_rcts > 0
        )
        atleastone_valid[j] = float((all_valid_per_sample > 0) and all_mols_in_rcts > 0)

    return all_valid, atleastone_valid, gen_rxns


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(x):
    xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            print(f"atomid_valence {atomid_valence}\n")
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            an = mol.GetAtomWithIdx(idx).GetAtomicNum()
            print("atomic num of atom with a large valence", an)
            if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                mol.GetAtomWithIdx(idx).SetFormalCharge(1)
            # queue = []
            # for b in mol.GetAtomWithIdx(idx).GetBonds():
            #     queue.append(
            #         (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            #     )
            # queue.sort(key=lambda tup: tup[1], reverse=True)
            # print(f'queue {queue}\n')
            # if len(queue) > 0:
            #     start = queue[0][2]
            #     end = queue[0][3]
            #     t = queue[0][1] - 1
            #     mol.RemoveBond(start, end)
            #     if t >= 1:
            #         mol.AddBond(start, end, bond_decoder_m[t])
    return
