"""
Get the dataset info needed for the model:
- atom_types, atom_charges, atom_chiral_tags
- bond_types, bond_dirs, allowed_bonds?
- get size of largest reaction => to adjust max num of nodes

preprocessing:
- split or remove any reaction with more than one product
- fix atom-mapping if needed: 1) all numbers from rct/prod sides match, 2) all atoms matched by number are of the same atom type, 3) numbers in ascending order 
"""
from rdkit import Chem
from matplotlib import pyplot as plt

def get_possible_valences(atom_type):
    '''
        Takes as input an atom_type in string format. Formal charge is expect to be in the format of O+1, N-2, etc.
        Returns a list of possible valences for the atom type, while taking into account the formal charge if present.
    '''
    # TODO: is this buggy somehow?
    get_charge = lambda x: int(x.split('+')[1]) if '+' in x else -int(x.split('-')[1]) if '-' in x else 0
    get_atom_type = lambda x: x.split('+')[0] if '+' in x else x.split('-')[0] if '-' in x else x
    # get valence of atom type
    pt = Chem.GetPeriodicTable()
    atom_type_valence = pt.GetValenceList(get_atom_type(atom_type))
    # add charge to valence if applicable
    valence_list = [v + get_charge(atom_type) for v in atom_type_valence]
    
    return list(valence_list)

folder = '/Users/laabidn1/RetroDiffuser/data/uspto-full-ring-openings/raw/'
subsets = ['train', 'test', 'val']
# TODO: what is the reaction dataset is too large to process all subsets in one go??
atom_types_without_charges, atom_types_with_charges, allowed_bonds, formal_charges = [], [], [], []   
all_nodes_in_reactions, dummy_nodes_in_reactions = [], []
cut_off = 10 # add comments explaining how to use this
for subset in subsets:
    rxns_larger_than_cut_off = []
    rxns = open(folder + subset + '.csv', 'r').readlines()
    for rxn in rxns:
        # get molecules
        rct, prod = rxn.split('>>')
        rct_mol = Chem.MolFromSmiles(rct)
        prod_mol = Chem.MolFromSmiles(prod)
        
        # check number of nodes and dummy nodes needed
        n_nodes = len(rct_mol.GetAtoms()) + len(prod_mol.GetAtoms())
        n_dummy_nodes = len(rct_mol.GetAtoms()) - len(prod_mol.GetAtoms())
        if n_dummy_nodes > cut_off: 
            rxns_larger_than_cut_off.append(rxn)
            # TODO: save the reactions ignored here?
        all_nodes_in_reactions.append(n_nodes)
        dummy_nodes_in_reactions.append(n_dummy_nodes)
        
        # get atom types and charges
        for m in [rct_mol, prod_mol]:
            for a in m.GetAtoms():
                symb_without_charge = a.GetSymbol()
                symb_with_charge = a.GetSymbol() 
                formal_charge = a.GetFormalCharge()
                # add formal charge to atom type if applicable
                if formal_charge != 0: 
                    sign = '+' if formal_charge>0 else '-'
                    symb_with_charge += sign + str(abs(formal_charge)) 
                atom_types_without_charges.extend([symb_without_charge])
                atom_types_with_charges.extend([symb_with_charge])
                formal_charges.extend([formal_charge])
    print(f'{subset}: len(rxns)={len(rxns)}, len(rxns_larger_than_cut_off)={len(rxns_larger_than_cut_off)}')
    
atom_types_without_charges = list(set(atom_types_without_charges))
atom_types_with_charges = list(set(atom_types_with_charges))
formal_charges = list(set(formal_charges))
print(f'atom_types_with_charges: {len(atom_types_with_charges)}, {atom_types_with_charges}')
allowed_bonds_without_charges = {a:get_possible_valences(a) for a in atom_types_without_charges}
allowed_bonds_with_charges = {a:get_possible_valences(a) for a in atom_types_with_charges}
atom_types_without_charges.extend(['none', 'SuNo', 'U', 'Au'])
atom_types_with_charges.extend(['none', 'SuNo', 'U', 'Au'])

# NOTE: for now (09.09) we don't care too much about the number of nodes because it's handled by supernodedataset
# the plan is to ignore large reactions by default, s.t. we only try 5, 10, 15, and 40 dummy nodes regardless of how many are needed 
# by the largest reactions in this dataset.
dummy_nodes_count = {dummy_nodes:dummy_nodes_in_reactions.count(dummy_nodes) for dummy_nodes in set(dummy_nodes_in_reactions)}
n_nodes_count = {n_nodes:all_nodes_in_reactions.count(n_nodes) for n_nodes in set(all_nodes_in_reactions)}
filtered_dummy_nodes_count = {dummy_nodes:dummy_nodes_count[dummy_nodes] for dummy_nodes in dummy_nodes_count if dummy_nodes_count[dummy_nodes] > cut_off}
filtered_n_nodes_count = {n_nodes:n_nodes_count[n_nodes] for n_nodes in n_nodes_count if n_nodes_count[n_nodes] > cut_off}
plt.hist(filtered_dummy_nodes_count, bins=100)
plt.savefig('filtered_dummy_nodes_count.png')
plt.close()
plt.hist(filtered_n_nodes_count, bins=100)
plt.savefig('filtered_n_nodes_count.png')
plt.close()
print(f'dummy_nodes_count: {filtered_dummy_nodes_count}')
print(f'n_nodes_count: {filtered_n_nodes_count}')
print(f'atom_types_without_charges: {len(atom_types_without_charges)}, {atom_types_without_charges}')
print(f'atom_types_with_charges: {len(atom_types_with_charges)}, {atom_types_with_charges}')
print(f'allowed_bonds_without_charges: {len(allowed_bonds_without_charges)}, {allowed_bonds_without_charges}')
print(f'allowed_bonds_with_charges: {len(allowed_bonds_with_charges)}, {allowed_bonds_with_charges}')
print(f'formal_charges: {len(formal_charges)}, {formal_charges}')
print(f'largest_n_nodes: {max(all_nodes_in_reactions)}')
print(f'dummy_nodes_needed: {max(dummy_nodes_in_reactions)}')