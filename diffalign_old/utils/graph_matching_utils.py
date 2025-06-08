from rdkit import Chem
from collections import defaultdict

def get_all_matches(mol1, mol2):
    return mol2.GetSubstructMatches(mol1, uniquify=False)

def score_matching(mol1, mol2, matching):
    score = 0
    for i, j in enumerate(matching):
        atom1 = mol1.GetAtomWithIdx(i)
        atom2 = mol2.GetAtomWithIdx(j)
        if atom1.GetAtomMapNum() == atom2.GetAtomMapNum() and atom1.GetAtomMapNum() != 0:
            score += 1
    return score

def get_best_matching(mol1, mol2):
    matches = get_all_matches(mol1, mol2)
    if not matches:
        return None
    scores = [score_matching(mol1, mol2, match) for match in matches]
    return matches[scores.index(max(scores))]

def aggregate_matchings(mol_list):
    if not mol_list:
        return []

    n_atoms = mol_list[0].GetNumAtoms()
    aggregated_mappings = [defaultdict(int) for _ in range(n_atoms)]
    
    # First, aggregate the atom mappings from all molecules
    for mol in mol_list:
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            map_num = atom.GetAtomMapNum()
            if map_num != 0:
                aggregated_mappings[i][map_num] += 1
    
    # Then, aggregate matchings between pairs of molecules
    for i in range(len(mol_list)):
        for j in range(i+1, len(mol_list)):
            matching = get_best_matching(mol_list[i], mol_list[j])
            if matching:
                for k, l in enumerate(matching):
                    atom_i = mol_list[i].GetAtomWithIdx(k)
                    atom_j = mol_list[j].GetAtomWithIdx(l)
                    map_i = atom_i.GetAtomMapNum()
                    map_j = atom_j.GetAtomMapNum()
                    
                    if map_i != 0:
                        aggregated_mappings[k][map_i] += 1
                    if map_j != 0:
                        aggregated_mappings[l][map_j] += 1  # Note the change from k to l here
    
    return aggregated_mappings

def get_unique_most_likely_mappings(aggregated_mappings):
    n_atoms = len(aggregated_mappings)
    most_likely_mappings = [0] * n_atoms
    used_mappings = set()
    
    # Sort atom indices by the maximum likelihood of their mappings
    sorted_indices = sorted(range(n_atoms), 
                            key=lambda i: max(aggregated_mappings[i].values()) if aggregated_mappings[i] else 0, 
                            reverse=True)
    
    for idx in sorted_indices:
        if not aggregated_mappings[idx]:
            continue
        
        # Sort mappings by likelihood
        sorted_mappings = sorted(aggregated_mappings[idx].items(), key=lambda x: x[1], reverse=True)
        
        # Assign the most likely unused mapping
        for mapping, _ in sorted_mappings:
            if mapping not in used_mappings:
                most_likely_mappings[idx] = mapping
                used_mappings.add(mapping)
                break
        
        # If no unused mapping found, assign 0 (unmapped)
        if most_likely_mappings[idx] == 0:
            print(f"Warning: Could not assign unique mapping for atom {idx}")
    
    return most_likely_mappings

def resolve_atom_mappings(smiles_list):
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    if all(mol is None for mol in mol_list):
        print("Error: No valid molecules found in the input list.")
        return None, None
    
    aggregated_mappings = aggregate_matchings(mol_list)
    most_likely_mappings = get_unique_most_likely_mappings(aggregated_mappings)
    
    # Apply the most likely mappings to all molecules
    for mol in mol_list:
        for i, mapping in enumerate(most_likely_mappings):
            mol.GetAtomWithIdx(i).SetAtomMapNum(mapping)
    
    return mol_list, most_likely_mappings