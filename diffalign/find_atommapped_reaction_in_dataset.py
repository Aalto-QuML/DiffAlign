file = "data/uspto-50k-block-15/raw/val.csv"
# Load the dataset by just reading in all the lines:
lines = []
with open(file, 'r') as f:
    lines = f.readlines()
# Now, we can search for a specific reaction in the dataset:
# ground_truth_reaction = "O=S(C1=CC=C(C=CC2=NC(CCl)=CO2)C=C1)C(F)(F)F.OC1=CC=C(COCCN2C=CN=N2)C=C1>>O=S(C1=CC=C(C=CC2=NC(COC3=CC=C(COCCN4C=CN=N4)C=C3)=CO2)C=C1)C(F)(F)F"

from rdkit import Chem
# Find the reaction in the dataset:
for idx, line in enumerate(lines):

    # line = "[N-:1]=[N+:2]=[N-:3].[N:4]#[C:5][CH2:6][c:7]1[cH:8][cH:9][c:10]2[c:11]([cH:12]1)[S:13][c:14]1[cH:15][cH:16][cH:17][cH:18][c:19]1[CH2:20][C:21]2=[O:22]>>[n:1]1[n:2][nH:3][c:5]([CH2:6][c:7]2[cH:8][cH:9][c:10]3[c:11]([cH:12]2)[S:13][c:14]2[cH:15][cH:16][cH:17][cH:18][c:19]2[CH2:20][C:21]3=[O:22])[n:4]1"

    reactants, product = line.strip().split(">>")
    reactants = reactants.split(".")
    reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
    product_mol = Chem.MolFromSmiles(product)
    # REMOVE ATOM MAPPING
    for mol in reactant_mols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    for atom in product_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # TURN BACK TO SMILES
    reactants = [Chem.MolToSmiles(r, isomericSmiles=True, canonical=True) for r in reactant_mols]
    product = Chem.MolToSmiles(product_mol, isomericSmiles=True, canonical=True)
    reaction = ".".join(reactants) + ">>" + product

    g_product = "CCCCOC(=O)COCCCNC(=O)C1=C(C)N=C(NC(=N)N)S1"
    g_product = "O=S(C1=CC=C(C=CC2=NC(COC3=CC=C(COCCN4C=CN=N4)C=C3)=CO2)C=C1)C(F)(F)F" # position 0 in the incorrect data file
    g_product = "C[SiH](C)C1=CC(C(C)(C)C)=CN=C1C1=CC=CC=N1"
    g_product = "O=C1CC2=CC=CC=C2SC2=CC(CC3=NN=NN3)=CC=C12" # position 1 (?) in the incorrect data file
    g_product = "CC1=CC(C2=NN=CN2)=CC=C1C1=CN=C2NC(=O)C3(CC3)NC2=N1" # position 2 in the incorrect data file
    g_product_mol = Chem.MolFromSmiles(g_product)
    for atom in g_product_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    g_product = Chem.MolToSmiles(g_product_mol, isomericSmiles=True, canonical=True)

    if product == g_product:
        print("Found the reaction in the dataset. Idx: ", idx)
        break
# )    if reaction == ground_truth_reaction:
#         print("Found the reaction in the dataset.")
#         break