import re
import copy
import random
import numpy as np
from diffalign.utils import graph, mol
from torch_geometric.data import InMemoryDataset, Batch, Data
import torch.nn.functional as F
import logging
import torch
import itertools
import pickle
import gzip

log = logging.getLogger(__name__)
from diffalign.utils.diffusion import helpers
from collections import defaultdict

def get_samples_from_file_pyg(cfg, filepath, condition_range):
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

    true_graph_data, sample_graph_data = dense_from_pyg_file_data(cfg, reactions)
    return true_graph_data, sample_graph_data

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
                                                                                          bond_types=graph.bond_types, with_explicit_h=cfg.dataset.with_explicit_h, 
                                                                                          supernode_nb=1, with_formal_charge=cfg.dataset.with_formal_charge,
                                                                                          add_supernode_edges=cfg.dataset.add_supernode_edges, get_atom_mapping=True,
                                                                                          canonicalize_molecule=cfg.dataset.canonicalize_molecule)
        
                n_nodes = t_graph.x.shape[0] - p_nodes.shape[0]
                # n_nodes = 1
                r_nodes = F.one_hot(torch.ones((n_nodes,), dtype=torch.long)*cfg.dataset.atom_types.index('U'), num_classes=len(cfg.dataset.atom_types)).float()
                r_edge_index = torch.tensor(list(itertools.combinations(range(n_nodes), 2))).T.long()
                r_edge_attr = torch.zeros((r_edge_index.shape[1], len(graph.bond_types))).float()
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

def dense_from_pyg_file_data(cfg, reactions):
    # Format of reactions:
    # 2-length list [gen_rxns, true_rxns], where gen_rxns and true_rxns are lists of DataBatches that 
    # contain n_samples_per_condition samples each.
    samples_graphs, true_rxn_graphs = [], []
    for i in range(len(reactions['gen'])):
        true_rxn_graphs.extend(graph.pyg_to_full_precision_expanded(reactions['true'][i], atom_types=cfg.dataset.atom_types).to_data_list())
        samples_graphs.extend(graph.pyg_to_full_precision_expanded(reactions['gen'][i], atom_types=cfg.dataset.atom_types).to_data_list())
    true_pyg_data = Batch.from_data_list(true_rxn_graphs)
    sample_pyg_data = Batch.from_data_list(samples_graphs)
    true_graph_data = graph.to_dense(true_pyg_data)
    sample_graph_data = graph.to_dense(sample_pyg_data)
    return true_graph_data, sample_graph_data

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
    """Transform list of (original_rxn, [(gen_rxn, numbers), ...]) into a dict
    keyed by product SMILES with list-of-dict values."""
    restructured_data = {}

    for original_reaction, generated_reactions in reactions:
        _, original_product = original_reaction.split(">>")
        generated_list = []

        for reaction_smiles, numbers in generated_reactions:
            generated_reactants, generated_product = reaction_smiles.split(">>")
            assert generated_product.strip() == original_product.strip(), \
                   f'Product mismatch: {original_product.strip()} vs {generated_product.strip()}'

            generated_dict = {
                'rcts': generated_reactants.split('.'),
                'prod': [generated_product],
                'elbo': numbers[0],
                'loss_t': numbers[1],
                'loss_0': numbers[2],
            }
            if with_count_and_prob and len(numbers) >= 5:
                generated_dict['count'] = numbers[3]
                generated_dict['prob'] = numbers[4]
            elif with_count and len(numbers) >= 4:
                generated_dict['count'] = numbers[3]

            generated_list.append(generated_dict)

        restructured_data[original_product.strip()] = generated_list

    return restructured_data


def remove_duplicates(elbo_sorted_rxns, strategy='keep_first'):
    """Remove duplicate reactions (by reactants), keeping unique entries with counts.

    Args:
        strategy: 'keep_first' keeps the first occurrence's numbers,
                  'random' picks a random duplicate's numbers,
                  'average' averages elbo/loss_t/loss_0 across duplicates.
    """
    elbo_sorted_rxns = copy.deepcopy(elbo_sorted_rxns)
    new_data = {}

    for prod, reactions in elbo_sorted_rxns.items():
        seen_reactions = {}
        ordered_keys = []

        for reaction in reactions:
            key = tuple(reaction['rcts'])
            if key not in seen_reactions:
                seen_reactions[key] = {'first': reaction.copy(), 'all': [reaction], 'count': 1}
                ordered_keys.append(key)
            else:
                seen_reactions[key]['count'] += 1
                seen_reactions[key]['all'].append(reaction)

        result = []
        for key in ordered_keys:
            entry = seen_reactions[key]
            count = entry['count']

            if strategy == 'random' and count > 1:
                chosen = random.choice(entry['all'])
            elif strategy == 'average' and count > 1:
                chosen = entry['first']
                chosen['elbo'] = sum(r['elbo'] for r in entry['all']) / count
                chosen['loss_t'] = sum(r['loss_t'] for r in entry['all']) / count
                chosen['loss_0'] = sum(r['loss_0'] for r in entry['all']) / count
            else:
                chosen = entry['first']

            chosen['count'] = count
            result.append(chosen)

        new_data[prod] = result

    return new_data