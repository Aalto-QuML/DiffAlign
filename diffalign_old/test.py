from rdkit import Chem

smi = '[C:1](\[CH3:2])([c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1)=[N:20]/[O:19][CH2:18][c:17]1[cH:16][cH:15][c:14]([N+:11](=[O:12])[O-:13])[cH:22][cH:21]1'
#smi = '[C:1](\[CH3:2])([c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1)=[N:20]\[O:19][CH2:18][c:17]1[cH:16][cH:15][c:14]([N+:11](=[O:12])[O-:13])[cH:22][cH:21]1'
m = Chem.MolFromSmiles(smi)
Chem.SanitizeMol(m)
smi_out = Chem.MolToSmiles(m, kekuleSmiles=False, canonical=True)
print(f'smi_out {smi_out}\n')
