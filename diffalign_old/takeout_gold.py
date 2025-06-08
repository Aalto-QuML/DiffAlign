'''
    Training script: used to train new models or resume training runs from wandb.
'''
import time
import os
import sys
import datetime
import pathlib
import warnings
import random
import numpy as np
import torch
import wandb
import hydra
import logging
import copy
from rdkit import Chem

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign_old.utils import setup, mol
from hydra.core.hydra_config import HydraConfig
from diffalign_old.utils import setup
from datetime import date

warnings.filterwarnings("ignore", category=PossibleUserWarning)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    
    '''
        Expects as input file a list reactions, one per file
    '''
    
    for s in ['train', 'test', 'val']:
        dataset_folder = f'{cfg.dataset.name}-{cfg.dataset.dataset_nb}' if cfg.dataset.dataset_nb!='' else cfg.dataset.name
        in_file = open(os.path.join(parent_path, 'data', dataset_folder, 'raw', f'{s}.csv'), 'r')
    
        #rxns = [l.split(',')[-1] for l in in_file.readlines()[1:]]
        rxns = in_file.readlines()
        missing_rcts, missing_prod = 0,0 
        atom_types, no_gold_rxns = [], []
        for i, rxn in enumerate(rxns):
            #print(f'rxn {i}\n')
            has_gold = False
            rcts = rxn.split('>>')[0].split('.')
            prod = rxn.split('>>')[1].split('.')
            if len([m for m in rcts if m!=''])==0: missing_rcts += 1
            if len([m for m in prod if m!=''])==0: missing_prod += 1
            
            #print(f'total mols {len(rcts+prod)}\n')
        
            for smi in rcts+prod:
                m = Chem.MolFromSmiles(smi)
                
                # print(f'==== cano {smi} \n')
                # if cfg.dataset.canonicalize_molecule:
                #     cano_m = mol.create_canonicalized_mol(m)
                # print(f'Done cano\n')
        
                Chem.RemoveStereochemistry(m)
                Chem.Kekulize(m, clearAromaticFlags=True)
                if cfg.dataset.with_explicit_h: m = Chem.AddHs(m, explicitOnly=True)

                #print(f'==== total atoms {len(m.GetAtoms())}\n')
                for atom in m.GetAtoms():
                    if cfg.dataset.with_formal_charge:
                        at = atom.GetSymbol() if atom.GetFormalCharge()==0 else atom.GetSymbol()+f'{atom.GetFormalCharge():+}'
                    else:
                        at = atom.GetSymbol()
                    if at=='Au':
                        has_gold = True
                    else:
                        atom_types.append(at)
            #print(f'==== next rxn \n')
            
            if has_gold:
                no_gold_rxns.append(rxn)
        
        print(f's={s} len(rxns)={len(rxns)} len(no_gold_rxns)={len(no_gold_rxns)}\n')
    # print(f'Formal charge = {cfg.dataset.with_formal_charge}\n')
    # print(set(atom_types))
    # print(len(set(atom_types)))
    # print(f'missing_rcts {missing_rcts}\n')
    # print(f'missing_prod {missing_prod}\n')
    
    # print(f'cfg.dataset.name {cfg.dataset.name}\n')
    # print(f'cfg.dataset.name {cfg.dataset.atom_types}\n')
    # exit()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception("main crashed. Error: %s", e)
