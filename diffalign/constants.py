"""
Centralized constants used across the DiffAlign codebase.
"""
from rdkit.Chem.rdchem import BondType as BT

MAX_ATOMS_RXN = 300
MAX_NODES = 300
DUMMY_RCT_NODE_TYPE = 'U'
BOND_TYPES = ['none', BT.SINGLE, BT.DOUBLE, BT.TRIPLE, 'mol', 'within', 'across']
