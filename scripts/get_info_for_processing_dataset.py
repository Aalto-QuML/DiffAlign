'''
Info includes: atom types, atom charges, atom chiral tags, bond types, bond dirs.
'''
import enum
from setup_path import *
import os
import hydra
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

from diffalign.helpers import PROJECT_ROOT
from diffalign.data.helpers import get_reactant_and_product_from_reaction_smiles, get_formal_charge_as_str

@hydra.main(config_path='../configs', config_name='config.yaml')
def get_info_for_processing_dataset(cfg):
    '''
        Get info for processing dataset.
    '''
    atom_types = []
    atom_types_charged = []
    atom_charges = []
    atom_chiral_tags = []
    bond_types = []
    bond_dirs = []
    # should only apply to training set
    # reactions_train = open(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
    #                                     'raw', 'train.csv'), encoding='utf-8').readlines()
    reactions_train = pd.read_csv(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
                                                'raw', 'train.csv'))['reactants>reagents>production'].tolist()
    for idx, reaction_smiles in enumerate(reactions_train):
        reactants, products = get_reactant_and_product_from_reaction_smiles(reaction_smiles)
        for molecule in reactants+products:
            mol = Chem.MolFromSmiles(molecule)
            if not cfg.dataset.use_aromatic_bond:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            for atom in mol.GetAtoms():
                atom_types.append(atom.GetSymbol())
                atom_charges.append(get_formal_charge_as_str(atom.GetFormalCharge(), no_charge_str='0'))
                atom_chiral_tags.append(str(atom.GetChiralTag()))
                atom_types_charged.append(atom.GetSymbol()+get_formal_charge_as_str(atom.GetFormalCharge()))
            for bond in mol.GetBonds():
                bond_types.append(str(bond.GetBondType()))
                bond_dirs.append(str(bond.GetBondDir()))
    # process the lists
    additional_atom_types = ['U', 'Au', 'SuNo', 'None']
    additional_bond_types = ['none', 'mol', 'within', 'across']
    atom_types_charged = [a+'\n' for a in set(atom_types_charged + additional_atom_types)]
    atom_types = [a+'\n' for a in set(atom_types + additional_atom_types)]
    atom_charges = [a+'\n' for a in set(atom_charges)]
    atom_chiral_tags = [a+'\n' for a in set(atom_chiral_tags)]
    bond_types = [a+'\n' for a in set(bond_types + additional_bond_types)]
    bond_dirs = [a+'\n' for a in set(bond_dirs)]
    # if cfg.dataset.dummy_node_type in atom_types:
    #     raise ValueError(f'{cfg.dataset.dummy_node_type} is in atom_types.')
    # else:
    #     atom_types.append(cfg.dataset.dummy_node_type)
    # save info
    os.makedirs(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir, 'processed'), exist_ok=True)
    open(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
                      'processed', 'atom_types.txt'), 'w', encoding='utf-8').writelines(atom_types)
    open(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
                      'processed', 'atom_types_charged.txt'), 'w', encoding='utf-8').writelines(atom_types_charged)
    open(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
                      'processed', 'atom_charges.txt'), 'w', encoding='utf-8').writelines(atom_charges)
    open(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
                      'processed', 'atom_chiral_tags.txt'), 'w', encoding='utf-8').writelines(set(atom_chiral_tags))
    open(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
                      'processed', 'bond_types.txt'), 'w', encoding='utf-8').writelines(set(bond_types))
    open(os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir,
                      'processed', 'bond_dirs.txt'), 'w', encoding='utf-8').writelines(set(bond_dirs))

if __name__ == '__main__':
    get_info_for_processing_dataset()
