import re
import copy
import random
import numpy as np
import os
import torch.distributed as dist
from diffalign.utils import graph, mol
from torch_geometric.data import Batch, Data
import torch.nn.functional as F
import logging
import torch
import itertools
import pickle
import gzip
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from diffalign.utils import data_utils, setup
from torch.utils.data.distributed import DistributedSampler  

log = logging.getLogger(__name__)
from diffalign.utils.diffusion import helpers
from collections import defaultdict

def create_evaluation_dataloader(rank, world_size, cfg, dataset_class, batch_size):
    # Get the original datamodule and dataset_infos
    data_slices = {'train': None, 'val': None, 'test': None}
    datamodule, dataset_infos = setup.get_dataset(
        cfg=cfg,
        dataset_class=dataset_class,
        shuffle=cfg.dataset.shuffle,
        return_datamodule=True,
        recompute_info=False,
        slices=data_slices
    )
    
    # Get the appropriate dataset based on cfg setting
    if cfg.diffusion.edge_conditional_set == 'test':
        dataset = datamodule.datasets['test']
    elif cfg.diffusion.edge_conditional_set == 'val':
        dataset = datamodule.datasets['val']
    elif cfg.diffusion.edge_conditional_set == 'train':
        dataset = datamodule.datasets['train']
    else:
        raise ValueError(f'cfg.diffusion.edge_conditional_set={cfg.diffusion.edge_conditional_set} is not valid')

    # Create DistributedSampler for the dataset
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=False  # Keep deterministic for evaluation
    # )
    sampler = None
    
    # Create and return the DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.dataset.num_workers,  # Adjust based on CPU availability
        pin_memory=True
    ), dataset_infos
            
def gather_distributed_results(world_size, results_to_gather):
    """
    Gather multiple arrays from all processes.
    
    Args:
        world_size: Number of processes
        *arrays: Arrays to gather from each process
        
    Returns:
        List of gathered results, one for each input array
    """
    gathered_results = {}
    for name, arr in results_to_gather.items():
        buffer = [None for _ in range(world_size)]
        dist.all_gather_object(buffer, arr)
        gathered_results[name] = buffer
    return gathered_results

def save_to_disk_distributed(data, filepath):
    for key, value in data.items():
        with open(os.path.join(filepath, f'{key}.pkl'), 'wb') as f:
            pickle.dump(value, f)

def get_samples_from_file_pyg(cfg, filepath, condition_range):
    """Get samples from a file in the format specified in dense_from_pyg_file_data function. 
    So can later then get the dense data from the samples. 
    Format: dict {'gen':gen_rxns, 'true':true_rxns} where gen_rxns and true_rxns are lists of DataBatches that contain n_samples_per_condition each
    """
    data = gzip.open(filepath, 'rb')
    # The format of the file is:
    # Dict {'gen':gen_rxns, 'true':true_rxns} where gen_rxns and true_rxns are lists DataBatches that
    # contain n_samples_per_condition samples each.
    reactions = pickle.load(data)

    # condition_range is a tuple of the form (start, end) that indicates the range of conditions to take from the file
    # ... but wait this is not necessary at all in case we just load the file with the correct starting index? 
    if condition_range: # TODO: This is deprecated
        reactions['gen'] = reactions['gen'][condition_range[0]:condition_range[1]]
        reactions['true'] = reactions['true'][condition_range[0]:condition_range[1]]

    # TODO: Use this instead
    # true_graph_data = graph.concatenate_databatches(reactions['true'])
    # sample_graph_data = graph.concatenate_databatches(reactions['gen'])
    # true_graph_data = graph.pyg_to_full_precision_expanded(true_graph_data, cfg.dataset.atom_types)
    # sample_graph_data = graph.pyg_to_full_precision_expanded(sample_graph_data, cfg.dataset.atom_types)
    # print(f'filepath {filepath}\n')
    # print(f'reactions {len(reactions)}\n')
    # print(f'reactions {reactions}\n')
    # true_graph_data, sample_graph_data = dense_from_pyg_file_data(cfg, reactions)
    
    return reactions #true_graph_data, sample_graph_data

def get_samples_from_file(cfg, filepath, condition_range=None):
    data = open(filepath, 'r').read()
    reactions = read_saved_reaction_only_data(data, condition_range=condition_range)
    true_graph_data, sample_graph_data = dense_from_file_data(cfg, reactions)
    
    return true_graph_data, sample_graph_data

def merge_scores_from_dicts(score_dicts_to_merge):
    scores = defaultdict(lambda:[])
    for score_dict in score_dicts_to_merge:
        for k,v in score_dict.items():
            scores[k].append(v)
    for k in scores.keys():
        for i,s in enumerate(scores[k]):
            if type(s) == type(np.array(1)): # make sure that there are no lingering np arrays
                scores[k][i] = s[0]
        # print(scores[k])
        scores[k] = sum(scores[k])/len(scores[k])
    return scores

def merge_scores(file_scores_to_merge):
    '''
        expects list of unaveraged scores (one per batch, i.e. each score is a tensor of shape (bs,))
    '''
    scores = defaultdict(lambda:[]) 
    
    for scores_f in file_scores_to_merge:
        new_scores  = pickle.load(open(scores_f, 'rb'))

        for score in new_scores:
            for k,v in score.items():
                scores[k].append(v)
        
    for k in scores.keys():
        for i,s in enumerate(scores[k]):
            if type(s) == type(np.array(1)): # make sure that there are no lingering np arrays
                scores[k][i] = s[0]
        # print(scores[k])
        scores[k] = sum(scores[k])/len(scores[k])#np.concatenate(scores[k], axis=0).mean(0)
            
    return scores


def merge_smiles_sample_output_files(files_to_merge, merged_output_file_name):
    '''
        Utility to merge sample output files into a single file. Useful when e.g. sampling from the full dataset in a multiprocessing run.
        Files are made of condition blocks of the following format: (cond cond_nb) true_rxn:\n\tsample_i\n\t
        For now, the function does not sort the conditions before dumping them in the shared file.
    '''
    files_to_merge = sorted(files_to_merge, key=lambda x: int(re.findall('\d+', x)[-1]))

    all_lines = []
    for m in files_to_merge:
        f = open(m, 'r')
        lines = f.readlines()
        f.close()
        all_lines.extend(lines)
        
    mgd_file = open(merged_output_file_name, 'w')
    mgd_file.writelines(all_lines)
    mgd_file.close()

def merge_pyg_sample_output_files(files_to_merge, merged_output_file_name):
    '''
        Utility to merge sample output files in a pickled list of PyG Data() format into a single file. 
    '''
    # The last number should indicate the order of the files. Not super important here, 
    # but including for consistency with merge_smiles_sample_output_files 
    files_to_merge = sorted(files_to_merge, key=lambda x: int(re.findall('\d+', x)[-1])) 
    all_true_reactions = []
    all_gen_reactions = []
    for m in files_to_merge:
        f = gzip.open(m, 'rb')
        data = pickle.load(f)
        # Format of data:
        # dict {'gen':gen_databatches, 'true':true_databatches] where gen_databatches and true_databatches are lists of DataBatches
        gen_databatches = data['gen']
        true_databatches = data['true']
        # TODO: PROBLEM: int16 in edge_index might not be enough here -> maybe do the combining to 
        # all other parts only later, when loading the data for evaluation
        all_gen_reactions.extend(gen_databatches)
        all_true_reactions.extend(true_databatches)
        # Remove databatches here and put them all to a single list
        # for i in range(len(gen_databatches)):
        #     all_gen_reactions.extend(gen_databatches[i].to_data_list())
        #     all_true_reactions.extend(true_databatches[i].to_data_list())
        f.close()
    # all_gen_reactions = Batch.from_data_list(all_gen_reactions)
    # all_true_reactions = Batch.from_data_list(all_true_reactions)
    all_data = {'gen':all_gen_reactions, 'true':all_true_reactions}
    with gzip.open(merged_output_file_name, 'wb') as f:
        pickle.dump(all_data, f)

def dense_from_file_data(cfg, reactions):
    true_rxn_graphs, samples_graphs = [], []
    
    for i, (true_rxn, samples) in enumerate(reactions):
        t_graph = graph.rxn_smi_to_graph_data(cfg, true_rxn)
        for s in samples:
            try:
                s_graph = graph.rxn_smi_to_graph_data(cfg, s)
            except:
                log.info(f'Could not parse sample {s}')
                # continue 
                # add dummy parsed nodes (e.g. all Au with no edges in between)
                # s_1 = '[C].[C].[Au].[Au].[Au]>>'+s.split('>>')[-1]
                p_nodes, p_edge_index, p_edge_attr, atom_map = mol.rxn_to_graph_supernode(mol=true_rxn.split('>>')[-1], atom_types=cfg.dataset.atom_types, 
                                                                                          bond_types=cfg.dataset.bond_types, with_explicit_h=cfg.dataset.with_explicit_h, 
                                                                                          supernode_nb=1, with_formal_charge=cfg.dataset.with_formal_charge,
                                                                                          add_supernode_edges=cfg.dataset.add_supernode_edges, get_atom_mapping=True,
                                                                                          canonicalize_molecule=cfg.dataset.canonicalize_molecule)
        
                n_nodes = t_graph.x.shape[0] - p_nodes.shape[0]
                # n_nodes = 1
                r_nodes = F.one_hot(torch.ones((n_nodes,), dtype=torch.long)*cfg.dataset.atom_types.index('U'), num_classes=len(cfg.dataset.atom_types)).float()
                r_edge_index = torch.tensor(list(itertools.combinations(range(n_nodes), 2))).T.long()
                r_edge_attr = torch.zeros((r_edge_index.shape[1], len(cfg.dataset.bond_types))).float()
                r_edge_attr[:,0] = 1.
                # r_edge_index = torch.tensor([]).long().reshape(2,0)
                # r_edge_attr = torch.tensor([]).float().reshape(0, len(graph.bond_types))
                offset = n_nodes
                
                # s_nodes = F.one_hot(torch.tensor([graph.atom_types.index('SuNo')], dtype=torch.long), 
                # num_classes=len(graph.atom_types)).float()
                g_nodes = torch.cat((r_nodes, p_nodes+offset), dim=0)
                g_edge_index = torch.cat((r_edge_index, p_edge_index+offset), dim=1)
                g_edge_attr = torch.cat((r_edge_attr, p_edge_attr), dim=0)
                total_n_nodes = t_graph.x.shape[0]
                        
                s_graph = Data(x=g_nodes, edge_index=g_edge_index, edge_attr=g_edge_attr, y=torch.zeros((1, 0), dtype=torch.float), idx=0,
                               mask_sn=torch.zeros(total_n_nodes, dtype=torch.long), mask_reactant_and_sn=torch.zeros(total_n_nodes, dtype=torch.long), 
                               mask_product_and_sn=torch.zeros(total_n_nodes, dtype=torch.long), mask_atom_mapping=torch.zeros(total_n_nodes, dtype=torch.long),
                               mol_assignment=torch.zeros(total_n_nodes, dtype=torch.long), cannot_generate=False)
    
            # add true_rxn for each sample reaction to get true data of the shape (bs, n_samples)
            # n_samples would be equal to the number of correct samples here. 
            # TODO: change the evaluation script to work on a subset of n_samples and average on n_samples
            true_rxn_graphs.append(t_graph)
            samples_graphs.append(s_graph)
        
    true_pyg_data = Batch.from_data_list(true_rxn_graphs)
    sample_pyg_data = Batch.from_data_list(samples_graphs)
    true_graph_data = graph.to_dense(true_pyg_data)
    sample_graph_data = graph.to_dense(sample_pyg_data)
    
    return true_graph_data, sample_graph_data

def dense_from_pyg_file_data(cfg, reaction):
    data_list = graph.pyg_to_full_precision_expanded(reaction, atom_types=cfg.dataset.atom_types, bond_types=cfg.dataset.bond_types).to_data_list()
    data_batch = Batch.from_data_list(data_list)
    dense_data = graph.to_dense(data_batch)
    return dense_data

def dense_from_pyg_file_data_for_reaction_list(cfg, reactions):
    # Format of reactions:
    # 2-length list [gen_rxns, true_rxns], where gen_rxns and true_rxns are lists of DataBatches that 
    # contain n_samples_per_condition samples each.
    samples_graphs, true_rxn_graphs = [], []
    for i in range(len(reactions['gen'])):
        #print(f"reactions['gen'][i] {reactions['gen'][i]}\n")
        true_rxn_graphs.extend(graph.pyg_to_full_precision_expanded(reactions['true'][i], cfg=cfg).to_data_list())
        samples_graphs.extend(graph.pyg_to_full_precision_expanded(reactions['gen'][i], cfg=cfg).to_data_list())

    if (cfg.neuralnet.pos_encoding_type != 'none' and cfg.neuralnet.pos_encoding_type != 'no_pos_enc'): # recalculate the positional encodings here
        pos_encoding_size = data_utils.get_pos_enc_size(cfg)
        for i in range(len(samples_graphs)):
            samples_graphs[i] = data_utils.positional_encoding_adding_transform(samples_graphs[i], cfg.neuralnet.pos_encoding_type, pos_encoding_size)

    true_pyg_data = Batch.from_data_list(true_rxn_graphs)
    sample_pyg_data = Batch.from_data_list(samples_graphs)
    true_graph_data = graph.to_dense(true_pyg_data)
    sample_graph_data = graph.to_dense(sample_pyg_data)
    
    return true_graph_data, sample_graph_data

    # for i, (true_rxn, samples) in enumerate(reactions):
    #     true_rxn_graphs.extend(true_rxn.to_data_list())
    #     samples_graphs.extend(samples.to_data_list())
    # true_pyg_data = Batch.from_data_list(true_rxn_graphs)
    # sample_pyg_data = Batch.from_data_list(samples_graphs)
    # true_graph_data = graph.to_dense(true_pyg_data)
    # sample_graph_data = graph.to_dense(sample_pyg_data)
    # return true_graph_data, sample_graph_data

def read_saved_reaction_data(data):
    # Reads the saved reaction data from the samples.txt file
    # Split the data into individual blocks based on '(cond ?)' pattern
    blocks = re.split(r'\(cond \d+\)', data)[1:]
    reactions = []
    for block in blocks:
        lines = block.strip().split('\n')
        original_reaction = lines[0].split(':')[0].strip()
        generated_reactions = []
        for line in lines[1:]:
            match = re.match(r"\t\('([^']+)', \[([^\]]+)\]\)", line)
            if match:
                reaction_smiles = match.group(1)
                numbers = list(map(float, match.group(2).split(',')))
                generated_reactions.append((reaction_smiles, numbers))
        reactions.append((original_reaction, generated_reactions))

    return reactions

def read_saved_reaction_only_data(data, condition_range=None):
    # Reads the saved reaction data from the samples.txt file
    # Split the data into individual blocks based on '(cond ?)' pattern
    blocks = re.split(r'\(cond \d+\)', data)[1:]
    if condition_range: blocks = blocks[int(condition_range[0]):int(condition_range[1])]
    reactions = []
    for block in blocks:
        lines = block.strip().split('\n')
        original_reaction = lines[0].split(':')[0].strip()
        # lines[0] is (cond #)
        generated_reactions = [rxn.strip() for rxn in lines[1:]]
        reactions.append((original_reaction, generated_reactions))
    
    return reactions

def restructure_reactions(reactions, with_count=True, with_count_and_prob=False):
    """
    Transform from format
        reactions = [
        ("C.C>>CC", [( "A.A>>CC", [1, 2, 3, 4] ), ( "B.B>>CC", [5, 6, 7, 8] )]),
        ("D.D>>DD", [( "X.X>>DD", [9, 10, 11, 12] ), ( "Y.Y>>DD", [13, 14, 15, 16] )])
        ]

        to the format in elbo_sorted_rxns, that is, a dictionary with the product as key and a list of dictionaries as value, where each dictionary is of the form
        {'rcts': [rct1, rct2, ...], 'prod': [prod],
        Example usage:
        reactions = [
            ("C.C>>CC", [( "A.A>>CC", [1, 2, 3, 4] ), ( "B.B>>CC", [5, 6, 7, 8] )]),
            ("D.D>>DD", [( "X.X>>DD", [9, 10, 11, 12] ), ( "Y.Y>>DD", [13, 14, 15, 16] )])
        ]
        output = restructure_reactions(reactions)
        print(output)   
    """
    # Initialize the dictionary to store the restructured data
    restructured_data = {}

    # Iterate over each original_reaction and its generated_reactions
    for original_reaction, generated_reactions in reactions:
        # Split the original reaction on ">>" to separate reactants and product
        _, original_product = original_reaction.split(">>")
        # print("Original product: ", original_product)

        # Initialize the list that will hold the dictionaries for each generated reaction
        generated_list = []

        # Iterate over each generated reaction
        for reaction_smiles, numbers in generated_reactions:
            # Split the generated reaction on ">>" to separate reactants and product
            generated_reactants, generated_product = reaction_smiles.split(">>")
            
            # print("Reaction: ", reaction_smiles)

            # Ensure that the generated product matches the original product
            assert generated_product.strip()==original_product.strip(),\
                   f'Original product {original_product.strip()} and generated product {generated_product.strip()} do not match.\n'
            generated_reactants = generated_reactants.split('.') # Split the reactants on '.' to get a list of reactants
            generated_product = [generated_product] # Convert the product to a list to match with the reactants

            # Extract the numbers
            if with_count_and_prob:
                elbo, loss_t, loss_0, count, prob = numbers
                # Create a dictionary for the generated reaction
                generated_dict = {
                    'rcts': generated_reactants,
                    'prod': generated_product,
                    'elbo': elbo,
                    'loss_t': loss_t,
                    'loss_0': loss_0,
                    'count': count,
                    'prob': prob
                }
            elif with_count:
                elbo, loss_t, loss_0, count = numbers
                # Create a dictionary for the generated reaction
                generated_dict = {
                    'rcts': generated_reactants,
                    'prod': generated_product,
                    'elbo': elbo,
                    'loss_t': loss_t,
                    'loss_0': loss_0,
                    'count': count
                }
            else:
                elbo, loss_t, loss_0 = numbers
                # Create a dictionary for the generated reaction
                generated_dict = {
                    'rcts': generated_reactants,
                    'prod': generated_product,
                    'elbo': elbo,
                    'loss_t': loss_t,
                    'loss_0': loss_0
                }

            # Append the dictionary to the list for this product
            generated_list.append(generated_dict)

        # Add the list of dictionaries to the restructured data under the product key
        restructured_data[original_product.strip()] = generated_list

    return restructured_data

def remove_duplicates(elbo_sorted_rxns):
    elbo_sorted_rxns = copy.deepcopy(elbo_sorted_rxns)
    new_data = {}
    for prod, reactions in elbo_sorted_rxns.items():
        seen_reactions = {}
        for reaction in reactions:
            # Convert the reactants list to a tuple so it can be used as a dictionary key
            reactants_tuple = tuple(reaction['rcts'])
            if reactants_tuple not in seen_reactions:
                # Add the reaction with a count of 1 if it's not a duplicate
                seen_reactions[reactants_tuple] = reaction.copy()
                seen_reactions[reactants_tuple]['count'] = 1
            else:
                # Increment the count if it's a duplicate
                seen_reactions[reactants_tuple]['count'] += 1
        # Add the unique reactions to the new data structure
        new_data[prod] = list(seen_reactions.values())
    return new_data

def remove_duplicates_and_select_random(elbo_sorted_rxns):
    # Selects a random set of (elbo, loss_t, loss_0) from the duplicates
    elbo_sorted_rxns = copy.deepcopy(elbo_sorted_rxns)
    new_data = {}
    for prod, reactions in elbo_sorted_rxns.items():
        seen_reactions = {}
        ordered_unique_reactions = []
        
        for reaction in reactions:
            reactants_tuple = tuple(reaction['rcts'])
            if reactants_tuple not in seen_reactions:
                # Add the reaction with a count of 1 if it's not a duplicate
                seen_reactions[reactants_tuple] = (reaction, 1)
                ordered_unique_reactions.append(reactants_tuple)
            else:
                # Update the reaction with a count incremented by 1 and replace the numbers with the current reaction's
                _, count = seen_reactions[reactants_tuple]
                seen_reactions[reactants_tuple] = (reaction, count + 1)
        
        # Now, build the final list of reactions with random numbers from duplicates
        for reactants_tuple in ordered_unique_reactions:
            reaction, count = seen_reactions[reactants_tuple]
            # If there are duplicates, randomly select one of the duplicates' numbers
            if count > 1:
                duplicates = [r for r in reactions if tuple(r['rcts']) == reactants_tuple]
                reaction = random.choice(duplicates)
            reaction['count'] = count
            new_data.setdefault(prod, []).append(reaction)
            
    return new_data

def remove_duplicates_and_average_numbers(elbo_sorted_rxns):
    elbo_sorted_rxns = copy.deepcopy(elbo_sorted_rxns)
    new_data = {}
    for prod, reactions in elbo_sorted_rxns.items():
        seen_reactions = {}
        ordered_unique_reactions = []
        
        for reaction in reactions:
            reactants_tuple = tuple(reaction['rcts'])
            if reactants_tuple not in seen_reactions:
                # Add the reaction with a count of 1 and initialize the sums of numbers
                seen_reactions[reactants_tuple] = {
                    'reaction': reaction,
                    'count': 1,
                    'sum_elbo': reaction['elbo'],
                    'sum_loss_t': reaction['loss_t'],
                    'sum_loss_0': reaction['loss_0']
                }
                ordered_unique_reactions.append(reactants_tuple)
            else:
                # Update the counts and sums of numbers
                seen_reaction = seen_reactions[reactants_tuple]
                seen_reaction['count'] += 1
                seen_reaction['sum_elbo'] += reaction['elbo']
                seen_reaction['sum_loss_t'] += reaction['loss_t']
                seen_reaction['sum_loss_0'] += reaction['loss_0']
        
        # Now, build the final list of reactions with average numbers
        for reactants_tuple in ordered_unique_reactions:
            seen_reaction = seen_reactions[reactants_tuple]
            count = seen_reaction['count']
            # Calculate the average of the numbers
            avg_elbo = seen_reaction['sum_elbo'] / count
            avg_loss_t = seen_reaction['sum_loss_t'] / count
            avg_loss_0 = seen_reaction['sum_loss_0'] / count
            
            # Update the reaction with the average numbers and count
            reaction = seen_reaction['reaction']
            reaction.update({
                'elbo': avg_elbo,
                'loss_t': avg_loss_t,
                'loss_0': avg_loss_0,
                'count': count
            })
            new_data.setdefault(prod, []).append(reaction)
            
    return new_data

# Example usage:
# original_data = {
#     'CC': [{'rcts': ['A', 'A'], 'prod': ['CC'], 'elbo': 1, 'loss_t': 2, 'loss_0': 3},
#            {'rcts': ['B', 'B'], 'prod': ['CC'], 'elbo': 5, 'loss_t': 6, 'loss_0': 7},
#            {'rcts': ['A', 'A'], 'prod': ['CC'], 'elbo': 1.1, 'loss_t': 2.1, 'loss_0': 3.1}],  # Duplicate for demonstration
#     'DD': [{'rcts': ['X', 'X'], 'prod': ['DD'], 'elbo': 9, 'loss_t': 10, 'loss_0': 11},
#            {'rcts': ['Y', 'Y'], 'prod': ['DD'], 'elbo': 13, 'loss_t': 14, 'loss_0': 15}]
# }

# new_data = remove_duplicates(original_data)
# print(new_data)