from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

def get_rdkit_bond_dirs(bond_dirs):
    rdkit_bond_dirs = []
    # rdkit_bond_dirs = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT]
    for direction in bond_dirs:
        if direction == "ENDUPRIGHT":
            rdkit_bond_dirs.append(Chem.rdchem.BondDir.ENDUPRIGHT)
        elif direction == "ENDDOWNRIGHT":
            rdkit_bond_dirs.append(Chem.rdchem.BondDir.ENDDOWNRIGHT)
        else:
            rdkit_bond_dirs.append(Chem.rdchem.BondDir.NONE)
    return rdkit_bond_dirs

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

def get_rdkit_chiral_tags(chiral_tags):
    rdkit_chiral_tags = []

    # TODO: could add more types here
    for tag in chiral_tags:
        if tag == "CHI_TETRAHEDRAL_CCW":
            rdkit_chiral_tags.append(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == "CHI_TETRAHEDRAL_CW":
            rdkit_chiral_tags.append(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        else:
            rdkit_chiral_tags.append(Chem.ChiralType.CHI_UNSPECIFIED)

    return rdkit_chiral_tags