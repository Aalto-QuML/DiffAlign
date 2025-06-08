from rdkit import Chem
from rdkit.Chem import Draw

# SMILES strings
#smiles_without_pos_enc = "B.C.C.C.C.C=N.N"
smiles_without_pos_enc = "CC1=NC=CC=C1CO"
smiles_with_pos_enc = "CC1=NC=CC=C1CO.O"

# Converting SMILES to RDKit molecule objects
molecule_without_pos_enc = Chem.MolFromSmiles(smiles_without_pos_enc)
molecule_with_pos_enc = Chem.MolFromSmiles(smiles_with_pos_enc)

# Drawing the molecules
img1 = Draw.MolToImage(molecule_without_pos_enc)
img2 = Draw.MolToImage(molecule_with_pos_enc)

# Save or display the images
img1.save("molecule_without_pos_enc.png")
img2.save("molecule_with_pos_enc.png")
