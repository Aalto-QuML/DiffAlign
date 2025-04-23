'''
   recomputing topk of old runs while taking stereochem into account.
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
from torch.profiler import profile, record_function, ProfilerActivity
from diffalign.utils import io_utils, mol
import multiprocessing
from functools import partial
from chython import smiles
from tqdm import tqdm

# A logger for this file
log = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffalign.utils import setup
from hydra.core.hydra_config import HydraConfig
from diffalign.utils import setup
from datetime import date
import re
from rdkit import Chem

warnings.filterwarnings("ignore", category=PossibleUserWarning)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB__SERVICE_WAIT"] = "300"

def remove_atom_mapping_and_stereo(smi):
    m = Chem.MolFromSmiles(smi)
    [a.ClearProp('molAtomMapNumber') for a in m.GetAtoms()]
    Chem.RemoveStereochemistry(m)
    
    return Chem.MolToSmiles(m, canonical=True)

def remove_atom_mapping(smi):
    m = Chem.MolFromSmiles(smi)
    [a.ClearProp('molAtomMapNumber') for a in m.GetAtoms()]
    return Chem.MolToSmiles(m, canonical=True)

def remove_atom_mapping_from_mol(mol):
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return mol

def undo_kekulize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return smi
    aromatic_smi = Chem.MolToSmiles(m, canonical=True)
    return aromatic_smi

def get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_stereo):
    
    for  true_rxn_without_stereo, true_rxn_with_stereo in zip(true_rxns_without_stereo, true_rxns_with_stereo):
        if  old_rxn==true_rxn_without_stereo:
            return true_rxn_with_stereo
    
    print(f'old rxn {old_rxn}\n')
    print(f'true_rxn_with_stereo {true_rxn_with_stereo}\n')
    
    return None

def match_old_rxns(old_rxn, true_rxns_without_stereo, true_rxns_with_stereo, true_rxns_with_am_and_stereo):
    return (
        get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_stereo),
        get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_am_and_stereo)
    )

# Define global variables for shared objects
counter = None
lock = None

def init_globals(c, l):
    global counter
    global lock
    counter = c
    lock = l

def update_progress():
    global counter
    global lock
    with lock:
        counter.value += 1
        return counter.value

from collections import defaultdict

def group_indices(strings):
    # Create a dictionary to store the indices for each string
    indices = defaultdict(list)
    
    # Populate the dictionary with indices
    for index, string in enumerate(strings):
        indices[string].append(index)
    
    return indices

# def process_reaction(i, matched_database_rxns_with_am, matched_database_rxns, sampled_rxns, cfg):
#     # transfers stereochemistry from product to reactant in the sampled_rxns using atom mapping
#     # and outputs matched top-k values compared to the reactant side of the matched_database_rxns
#     # ... also take the stereochemistry from the matched product from matched_database_rxns_with_am
    
#     matched_rxn_with_am = matched_database_rxns_with_am[i]
#     matched_rxn = matched_database_rxns[i]
#     samples = sampled_rxns[i]
    
#     matched_prod_mol = Chem.MolFromSmiles(matched_rxn_with_am.split('>>')[1])
#     matched_rct_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(matched_rxn_with_am.split('>>')[0]), canonical=True)
#     matched_prod_smiles = Chem.MolToSmiles(matched_prod_mol, canonical=True)

#     samples_with_chirality = []
#     for j in range(len(samples)):
#         prod_smi_without_am = matched_rxn.split('>>')[1]
#         if "@" in matched_rxn_with_am or "/" in matched_rxn_with_am or "\\" in matched_rxn_with_am:
#             sample = samples[j]
            
#             # TODO: Change this so that it works with the atom mappings from the generations directly
#             # sample = smiles(sample)
#             # sample.reset_mapping()
#             # sample = format(sample, 'm')
            
#             prod_side_ams = set([a.GetAtomMapNum() for a in Chem.MolFromSmiles(sample.split('>>')[1]).GetAtoms()])
#             sample_rct_mol = Chem.MolFromSmiles(sample.split('>>')[0])
#             sample_prod_mol = Chem.MolFromSmiles(sample.split('>>')[1])

#             if sample_rct_mol is not None:
#                 # matched_rxn_mol = Chem.MolFromSmiles(matched_rxn.split(">>")[0])
#                 # Chem.RemoveStereochemistry(matched_rxn_mol)
#                 # if ("\\" in matched_rxn or "/" in matched_rxn) and (Chem.MolToSmiles(remove_atom_mapping_from_mol(copy.deepcopy(sample_rct_mol))) == Chem.MolToSmiles(matched_rxn_mol)):
#                 #     print("heihi")
#                 #     print("hoihi")
#                 Chem.RemoveStereochemistry(sample_rct_mol) # This does some kind of sanitization, otherwise transferring the bond_dirs doesn't work reliably

#                 for a in sample_rct_mol.GetAtoms():# remove atom mappings that are not on the product side
#                     if a.GetAtomMapNum() not in prod_side_ams:
#                         a.ClearProp('molAtomMapNumber')
#                 mol.match_atom_mapping_without_stereo(sample_prod_mol, matched_prod_mol) # temporarily change the atom mapping in matched_prod_mol
#                 if "@" in matched_rxn_with_am:
#                     sample_rct_mol = mol.transfer_chirality_from_product_to_reactant(sample_rct_mol, matched_prod_mol)
#                 if "/" in matched_rxn_with_am or "\\" in matched_rxn_with_am:
#                     sample_rct_mol = mol.transfer_bond_dir_from_product_to_reactant(sample_rct_mol, matched_prod_mol)
#                 remove_atom_mapping_from_mol(sample_rct_mol)
#                 r_smiles = Chem.MolToSmiles(sample_rct_mol, canonical=True)
#             else:
#                 r_smiles = ""
#         else:
#             r_smiles = samples[j].split('>>')[0]

#         # assert Chem.MolToSmiles(remove_atom_mapping_from_mol(copy.deepcopy(matched_prod_mol))) == prod_smi_without_am

#         samples_with_chirality.append(r_smiles + ">>" + prod_smi_without_am)

#     chiral_reactions = 0
#     cistrans_reactions = 0
#     topk_local = {k: 0 for k in cfg.test.topks}
#     topk_among_chiral_local = {k: 0 for k in cfg.test.topks}
#     topk_among_cistrans_local = {k: 0 for k in cfg.test.topks}

#     if "@" in matched_rxn:
#         chiral_reactions = 1
#         for k in cfg.test.topks:
#             topk_among_chiral_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])
#     if "/" in matched_rxn or "\\" in matched_rxn:
#         cistrans_reactions = 1
#         for k in cfg.test.topks:
#             topk_among_cistrans_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])
#         # if topk_among_cistrans_local[100] == 0:
#         #     print("hei") # checking whether it would have worked with stereochemistry
#         #     matched_rxn_mol_rct = Chem.MolFromSmiles(matched_rxn.split(">>")[0])
#         #     matched_rxn_mol_prod = Chem.MolFromSmiles(matched_rxn.split(">>")[1])
#         #     Chem.RemoveStereochemistry(matched_rxn_mol_rct)
#         #     Chem.RemoveStereochemistry(matched_rxn_mol_prod)
#         #     matched_smiles_without_stereo = Chem.MolToSmiles(matched_rxn_mol_rct) + ">>" + Chem.MolToSmiles(matched_rxn_mol_prod)
#         #     if matched_smiles_without_stereo in samples:
#         #         print(matched_smiles_without_stereo in samples)

#     for k in cfg.test.topks:
#         topk_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])

#     return topk_local, topk_among_chiral_local, topk_among_cistrans_local, chiral_reactions, cistrans_reactions

def process_reaction(i, matched_database_rxns_with_am, matched_database_rxns, sampled_rxns, cfg):
    """Inputs:
    i: index of the reaction
    matched_old_true_rxns_with_am: list of matched old reactions with atom mappings
    matched_old_true_rxns: list of matched old reactions without atom mappings
    sampled_rxns: list of sampled reactions. Should have atom mappings!
    cfg: configuration object
    Returns:
    topk_local: dictionary of topk for the current reaction
    topk_among_chiral_local: dictionary of topk for chiral reactions, for logging purposes
    topk_among_cistrans_local: dictionary of topk for cistrans reactions, for logging purposes
    chiral_reactions: number of chiral reactions (1/0)
    cistrans_reactions: number of cistrans reactions (1/0)
    progress: current progress counter
    """
    matched_rxn_with_am = matched_database_rxns_with_am[i]
    matched_rxn = matched_database_rxns[i]
    samples = sampled_rxns[i]
    
    matched_prod_mol = Chem.MolFromSmiles(matched_rxn_with_am.split('>>')[1])

    samples_with_chirality = []
    for j in range(len(samples)):
        prod_smi_without_am = matched_rxn.split('>>')[1]
        if "@" in matched_rxn_with_am or "/" in matched_rxn_with_am or "\\" in matched_rxn_with_am:
            sample = samples[j]
            
            try:
                r_mol = Chem.MolFromSmiles(sample.split('>>')[0])
                [r_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(r_mol.GetNumAtoms())]
                p_mol = Chem.MolFromSmiles(sample.split('>>')[1])
                [p_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(p_mol.GetNumAtoms())]
                sample2 = Chem.MolToSmiles(r_mol, canonical=True) + ">>" + Chem.MolToSmiles(p_mol, canonical=True)
                sample2 = smiles(sample2)
                sample2.reset_mapping()
                sample2 = format(sample2, 'm')
            except:
                pass

            def a(sample):
                prod_side_ams = set([a.GetAtomMapNum() for a in Chem.MolFromSmiles(sample.split('>>')[1]).GetAtoms()])
                sample_rct_mol = Chem.MolFromSmiles(sample.split('>>')[0])
                sample_prod_mol = Chem.MolFromSmiles(sample.split('>>')[1])

                if sample_rct_mol is not None:
                    Chem.RemoveStereochemistry(sample_rct_mol) # This does some kind of sanitization, otherwise transferring the bond_dirs doesn't work reliably
                    for a in sample_rct_mol.GetAtoms():# remove atom mappings that are not on the product side
                        if a.GetAtomMapNum() not in prod_side_ams:
                            a.ClearProp('molAtomMapNumber')
                    mol.match_atom_mapping_without_stereo(sample_prod_mol, matched_prod_mol) # temporarily change the atom mapping in matched_prod_mol
                    if "@" in matched_rxn_with_am:
                        sample_rct_mol = mol.transfer_chirality_from_product_to_reactant(sample_rct_mol, matched_prod_mol)
                    if "/" in matched_rxn_with_am or "\\" in matched_rxn_with_am:
                        sample_rct_mol = mol.transfer_bond_dir_from_product_to_reactant(sample_rct_mol, matched_prod_mol)
                    remove_atom_mapping_from_mol(sample_rct_mol)
                    r_smiles = Chem.MolToSmiles(sample_rct_mol, canonical=True)
                else:
                    r_smiles = ""
                return r_smiles
            
            #r_smiles = a(sample)
            r_smiles = a(sample2)
        else:
            try:
                r_mol = Chem.MolFromSmiles(samples[j].split('>>')[0])
                [r_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(r_mol.GetNumAtoms())]
                r_smiles = Chem.MolToSmiles(r_mol, canonical=True)
            except: 
                r_smiles = samples[j].split('>>')[0]

        samples_with_chirality.append(r_smiles+">>"+prod_smi_without_am)

    chiral_reactions = 0
    cistrans_reactions = 0
    topk_local = {k: 0 for k in cfg.test.topks}
    topk_among_chiral_local = {k: 0 for k in cfg.test.topks}
    topk_among_cistrans_local = {k: 0 for k in cfg.test.topks}

    if "@" in matched_rxn:
        chiral_reactions = 1
        for k in cfg.test.topks:
            topk_among_chiral_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])
    if "/" in matched_rxn or "\\" in matched_rxn:
        cistrans_reactions = 1
        for k in cfg.test.topks:
            topk_among_cistrans_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])

    for k in cfg.test.topks:
        topk_local[k] += int(matched_rxn in samples_with_chirality[:int(k)])

    # Update progress counter
    progress = update_progress()

    return topk_local, topk_among_chiral_local, topk_among_cistrans_local, chiral_reactions, cistrans_reactions, progress, samples_with_chirality


@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    """Usage: You must specify:
    cfg.general.wandb.checkpoint_epochs (list with only one element in this case)
    cfg.diffusion.diffusion_steps_eval: number of steps in the diffusion model for generating the samples
    cfg.test.sort_lambda_value: the lambda value used for sorting the samples in ranking the samples
    cfg.test.total_cond_eval: the number of conditions used for evaluation
    cfg.test.n_samples_per_condition: the number of samples per condition used for evaluation
    cfg.diffusion.edge_conditional_set: the edge conditional set used for the diffusion model
    cfg.hydra.run.dir: the directory where the generated data is stored 
    """

    parent_path = pathlib.Path(os.path.realpath(__file__)).parents[3] # hacky for now

    epoch = cfg.general.wandb.checkpoint_epochs[0]
    sampling_steps = cfg.diffusion.diffusion_steps_eval

    # where to get the actual data location? 
    raw_data_path = os.path.join(parent_path, cfg.dataset.datadir + '-'  +str(cfg.dataset.dataset_nb), 'raw', cfg.diffusion.edge_conditional_set + '.csv')
    # assumes that hydra.run.dir is set to the location where you have generated data
    old_samples_path = os.path.join(f'eval_epoch{epoch}_steps{sampling_steps}_resorted_{cfg.test.sort_lambda_value}_cond{cfg.test.total_cond_eval}_sampercond{cfg.test.n_samples_per_condition}_{cfg.diffusion.edge_conditional_set}_lam{cfg.test.sort_lambda_value}.txt') 

    # load true raw data reactions
    print(f'reading raw data from {raw_data_path}\n')
    raw_true_rxns = open(raw_data_path, 'r').readlines()
    print(f'reading old samples from {old_samples_path}\n')
    sampled_rxns_in_eval_format = open(old_samples_path, 'r').read()
    sampled_rxns_blocks = io_utils.read_saved_reaction_data(sampled_rxns_in_eval_format)
    #sampled_rxns_blocks = sampled_rxns_blocks[:110]

    # this data contains the atom mappings, stereochemistry, everything. We want to remove atom mapping, stereochemistry, etc., to match
    # to the generated data on. But also we want to keep the reactions with stereochemistry 
    database_rxns_with_am_and_stereo, database_rxns_with_stereo, database_rxns_without_stereo = [], [], []
    for rxn in raw_true_rxns:
        reactants = rxn.split('>>')[0]
        products = rxn.split('>>')[1]
        database_rxns_with_am_and_stereo.append(rxn)
        rxn_with_stereo = remove_atom_mapping(reactants) + '>>' + remove_atom_mapping(products)
        database_rxns_with_stereo.append(rxn_with_stereo)
        rxn_without_stereo = remove_atom_mapping_and_stereo(reactants) + '>>' + remove_atom_mapping_and_stereo(products)
        database_rxns_without_stereo.append(rxn_without_stereo)

    # This is the ground-truth reaction saved in the sampled data. We want to match these to the original reaction database
    # to find out more details about what are we trying to generate in the first place here, in the case that the generated data
    # structure is lacking some information about stereochemistry
    raw_old_true_rxns = [sample[0] for sample in sampled_rxns_blocks]
    old_true_rxns = [] # the format that we used to have
    # This transforms the data to the format 
    for rxn in raw_old_true_rxns:
        reactants = rxn.split('>>')[0]
        products = rxn.split('>>')[1]
        old_true_rxn = remove_atom_mapping(reactants) + '>>' + remove_atom_mapping(products)
        old_true_rxns.append(old_true_rxn)

    # This is the generated data. We want to match these to the original reaction database
    # using the matching we get from raw_old_true_rxns and true_rxns_without_stereo
    # The code here mainly cleans up the data into a canonical format that can be used for matching. 
    # sampled_rxns is supposed to contain atom mapping
    # sampled_rxns is a list of lists, where each list contains the samples for a product (single reaction)
    raw_sampled_rxns = [sample[1] for sample in sampled_rxns_blocks]

    # TODO: CHECK THAT THE DEDUPLICATION WORKS WITH THE ATOM MAPPING HERE -> probably choose the most common atom mapping...

    sampled_rxns = []
    for sample in raw_sampled_rxns:
        sampled_rxns_per_true_rxn = []
        for sample_info in sample:
            rxn = sample_info[0] # sample_info also contains numbers like elbo etc
            reactants = rxn.split('>>')[0]
            products = rxn.split('>>')[1]
            sampled_rxn = undo_kekulize(reactants) + '>>' + undo_kekulize(products)
            sampled_rxns_per_true_rxn.append(sampled_rxn)
        
        # For the case where the 'samples' file was encoded in a way that contained duplicates, we group them together here and calculate the weighted probability
        rxn_indices_grouped = group_indices(sampled_rxns_per_true_rxn) # group indices of duplicates
        new_counts = {}
        new_elbos = {}
        for rxn, indices in rxn_indices_grouped.items():
            counts = sum([int(sample[i][1][-2]) for i in indices])
            elbos = sum([float(sample[i][1][0]) for i in indices]) / len(indices)
            new_elbos[rxn] = elbos
            new_counts[rxn] = counts

        # recalculate the weighted probability
        sum_exp_elbo = sum(np.exp(-elbo) for elbo in new_elbos.values())
        sum_counts = sum(new_counts.values())
        new_weighted_probs = {}
        for rxn in sampled_rxns_per_true_rxn:
            exp_elbo = np.exp(-new_elbos[rxn])
            weighted_prob = (exp_elbo / sum_exp_elbo) * cfg.test.sort_lambda_value + (new_counts[rxn] / sum_counts) * (1 - cfg.test.sort_lambda_value)
            new_weighted_probs[rxn] = weighted_prob
        
        # sort the list of reactions for the current product based on weighted_prob
        new_sampled_rxns_per_true_rxn = sorted(list(set(sampled_rxns_per_true_rxn)), key=lambda x: new_weighted_probs[x], reverse=True)

        sampled_rxns.append(new_sampled_rxns_per_true_rxn)

    # match sample data ground-truth and database reactions
    print(f'matching sample data ground-truth and database reactions\n')
    # Define the partial function
    partial_match_old_rxns = partial(
        match_old_rxns,
        true_rxns_without_stereo=database_rxns_without_stereo,
        true_rxns_with_stereo=database_rxns_with_stereo,
        true_rxns_with_am_and_stereo=database_rxns_with_am_and_stereo
    )
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=8) as pool:
        # Map the match_old_rxns function to the old_true_rxns list
        # The results is a list of tuples, where the first element 
        results = pool.map(partial_match_old_rxns, old_true_rxns)

    # Process the results
    # the lists that we get out in the end here are generated data index -> database data elements (with or without AM)
    gen_index_to_database_rxn = []
    gen_index_to_database_rxn_with_am = []
    for i, (matched_rxn, matched_rxn_with_am) in enumerate(results):
        gen_index_to_database_rxn.append(matched_rxn)
        gen_index_to_database_rxn_with_am.append(matched_rxn_with_am)
        print(f'i {i}\n')

    # for i, old_rxn in enumerate(old_true_rxns):
    #     print(f'i {i}\n')
    #     # TODO: This double loop is a bit slow
    #     matched_old_true_rxns.append(get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_stereo))
    #     matched_old_true_rxns_with_am.append(get_rxn_with_stereo(old_rxn, true_rxns_without_stereo, true_rxns_with_am_and_stereo))

    assert None not in gen_index_to_database_rxn, 'Some old reactions could not be matched with true reactions.'
    
    import time
    t0 = time.time()
    # calculate topk
    # assumes old_samples are sorted
    print(f'calculating topk\n')
    topk = {k:0 for k in cfg.test.topks}
    topk_among_chiral = {k:0 for k in cfg.test.topks}
    topk_among_cistrans = {k:0 for k in cfg.test.topks}
    total_chiral_reactions = 0
    total_cistrans_reactions = 0
    num_processes = multiprocessing.cpu_count()
    num_processes = 1

    manager = multiprocessing.Manager()
    global counter, lock
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    print("num_processes", num_processes)
    partial_process_reaction = partial(
        process_reaction,
        matched_database_rxns_with_am=gen_index_to_database_rxn_with_am, 
            matched_database_rxns=gen_index_to_database_rxn, 
            sampled_rxns=sampled_rxns, cfg=cfg)
    
    total_tasks = len(gen_index_to_database_rxn)
    with multiprocessing.Pool(processes=num_processes, initializer=init_globals, initargs=(counter, lock)) as pool:
        results = []
        with tqdm(total=total_tasks) as pbar:
            for result in pool.imap(partial_process_reaction, range(total_tasks)):
                results.append(result[:5])  # Append only the topk_local, topk_among_chiral_local, and chiral_reactions
                pbar.update(result[-1] - pbar.n)  # Update the progress bar

    for topk_local, topk_among_chiral_local, topk_among_cistrans_local, chiral_reactions, cistrans_reactions in results:
        for k in cfg.test.topks:
            topk[k] += topk_local[k]
            topk_among_chiral[k] += topk_among_chiral_local[k]
            topk_among_cistrans[k] += topk_among_cistrans_local[k]
        total_chiral_reactions += chiral_reactions
        total_cistrans_reactions += cistrans_reactions

    # topk = {k:0 for k in cfg.test.topks}
    # topk_among_chiral = {k:0 for k in cfg.test.topks}
    # topk_among_cistrans = {k:0 for k in cfg.test.topks}
    # total_chiral_reactions = 0
    # total_cistrans_reactions = 0
    # num_processes = 1#multiprocessing.cpu_count()
    # print("num_processes", num_processes)
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.starmap(
    #         partial(process_reaction, 
    #                 matched_database_rxns_with_am=gen_index_to_database_rxn_with_am, 
    #                 matched_database_rxns=gen_index_to_database_rxn, 
    #                 sampled_rxns=sampled_rxns, cfg=cfg), 
    #         [(i,) for i in range(len(gen_index_to_database_rxn))]
    #     )
    # for topk_local, topk_among_chiral_local, topk_among_cistrans_local, chiral_reactions, cistrans_reactions in results:
    #     for k in cfg.test.topks:
    #         topk[k] += topk_local[k]
    #         topk_among_chiral[k] += topk_among_chiral_local[k]
    #         topk_among_cistrans[k] += topk_among_cistrans_local[k]
    #     total_chiral_reactions += chiral_reactions
    #     total_cistrans_reactions += cistrans_reactions

    print(f'time taken {time.time()-t0}\n')
    print(f'unnormalized topk {topk}\n')   
    print(f'unnormalized topk_among_chiral {topk_among_chiral}\n')
    print(f'total_chiral_reactions {total_chiral_reactions}\n')
    topk = {k:v/len(sampled_rxns) for k,v in topk.items()}
    topk_among_chiral = {k:v/total_chiral_reactions for k,v in topk_among_chiral.items()}
    topk_among_cistrans = {k:v/total_cistrans_reactions for k,v in topk_among_cistrans.items()}
    print(f'normalized topk {topk}\n')  
    print(f'normalized topk_among_chiral {topk_among_chiral}\n')
    print(f'normalized topk_among_cistrans {topk_among_cistrans}\n')

if __name__ == '__main__':
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        try:
            main()
        except Exception as e:
            log.exception("main crashed. Error: %s", e)
    else:
        main()


