import pandas as pd
from setup_path import *
from diffalign.data.helpers import clear_atom_map

def canonicalize_rxn(rxn, remove_atom_map=True):
    # NOTE; need to remove atom_map always because otherwise it's not canonical
    # TODO: also need to check if rdkit canonicalization cannot happen with atom mapping?
    if '>>' in rxn:
        token = '>>'
    elif '>' in rxn:
        token = '>'
    else:
        raise ValueError(f"Invalid reaction format: {rxn}")

    reactants = rxn.split(token)[0].strip().split('.')
    products = rxn.split(token)[1].strip().split('.')
    
    reactants = [clear_atom_map(r) if remove_atom_map else r for r in reactants]
    products = [clear_atom_map(p) if remove_atom_map else p for p in products]
    
    return f"{'.'.join(reactants)}{token}{'.'.join(products)}"

def canonicalize_reactions_without_atom_mapping():
    """
    Canonicalize reactions without atom mapping.
    """
    path = '/Users/laabidn1/DiffAlign/dataset/uspto_50k/raw/test.csv'
    reactions = pd.read_csv(path)['reactants>reagents>production'].tolist()

    cano_reactions = [canonicalize_rxn(r) for r in reactions]
    out_file = '/Users/laabidn1/DiffAlign/dataset/uspto_50k/raw/test_cano.csv'
    pd.DataFrame({'reactants>reagents>production': cano_reactions}).to_csv(out_file, index=False)

if __name__ == "__main__":
    canonicalize_reactions_without_atom_mapping()