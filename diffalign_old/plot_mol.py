from rdkit import Chem
from rdkit.Chem import Draw

# Define the SMILES string
smiles = "COC1=CC([N+](=O)[O-])=CC=C1(Cl)N1CCC(=O)C1"

# Convert the SMILES string to an RDKit molecule
molecule = Chem.MolFromSmiles(smiles, sanitize=False)


# Draw the molecule and save as a PNG
Draw.MolToFile(molecule, 'molecule.png')
