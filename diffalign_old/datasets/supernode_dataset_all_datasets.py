import os
import os.path as osp
import pathlib
from typing import Any, Sequence
import pickle
import copy
import re
from multiprocessing import Pool
import sys
from multiprocessing import Lock, Process, Queue, current_process
import time
from rdkit.Chem import PeriodicTable
from rdkit.Chem import rdChemReactions
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from torch_geometric.utils import subgraph
from diffalign_old.datasets.abstract_dataset import AbstractDataModule, seed_worker
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
import rdkit
from diffalign_old.utils import graph, mol, setup
from diffalign_old.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos, DistributionNodes
# from src.utils.rdkit import  mol2smiles, build_molecule_with_partial_charges
# from src.utils.rdkit import compute_molecular_metrics
from diffalign_old.utils import data_utils, graph as graph_utils
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from diffalign_old.utils.mol import get_rdkit_chiral_tags, get_rdkit_bond_types, get_bond_orders, get_bond_orders_correct, get_rdkit_bond_dirs

MAX_ATOMS_RXN = 1000
# MAX_NODES_MORE_THAN_PRODUCT = 35 <- this shouldn't be used!

DUMMY_RCT_NODE_TYPE = 'U'

#rdkit_atom_chiral_tags = [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
#rdkit_bond_types = [0, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
#rdkit_bond_dirs = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT]
#rdkit_bond_configs = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]

# THESE ARE NOT USED ANYMORE
# size_bins = {
#     'train': [64, 83, 102], # [64,83,102]
#     'test': [250],
#     'val': [250]
# }

# batchsize_bins = { 
#     'train': [32, 16, 8], # [128, 64, 16]
#     'test': [32], # [64]
#     'val': [32] # [64]
# }

raw_files = ['train.csv', 'test.csv', 'val.csv']
processed_files = ['train.pt', 'test.pt', 'val.pt']

# raw_files = ['test.csv', 'val.csv']
# processed_files = ['test.pt', 'val.pt']

# raw_files = ['test.csv']
# processed_files = ['test.pt']

logging.basicConfig(filename='subprocess_log.txt', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')

class Dataset(InMemoryDataset):
    def __init__(self, stage, cfg):
        self.datadir = cfg.dataset.datadir
        self.datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.dataset_nb!='':
            self.datadir += '-'+str(cfg.dataset.dataset_nb)
            self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        self.stage = stage
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.root = os.path.join(base_path, self.datadir)
        self.cfg = cfg
        self.config_path = os.path.join(self.root, 'processed', 'dataset_config.yaml')
        # self.size_test_splits = cfg.size_test_splits
        # self.with_explicit_h = cfg.dataset.with_explicit_h
        # self.with_formal_charge = cfg.dataset.with_formal_charge
        # self.max_nodes_more_than_product = cfg.dataset.nb_rct_dummy_nodes
        # self.canonicalize_molecule = cfg.dataset.canonicalize_molecule
        # self.add_supernode_edges = cfg.dataset.add_supernode_edges
        # self.atom_types = cfg.dataset.atom_types
        # self.bond_types = add_chem_bond_types(cfg.dataset.bond_types)
        # self.atom_charges = cfg.dataset.atom_charges
        # self.rdkit_atom_chiral_tags = get_atom_chiral_tags(cfg.dataset.atom_chiral_tags)
        # self.rdkit_bond_dirs = get_bond_dirs(cfg.dataset.bond_dirs)
        # self.permute_mols = cfg.dataset.permute_mols
        # self.num_processes = cfg.dataset.num_processes
        # transform_1 = lambda x: x
        # if (cfg.neuralnet.pos_encoding_type != 'none' and cfg.neuralnet.pos_encoding_type != 'no_pos_enc'):
        #     pos_encoding_size = data_utils.get_pos_enc_size(cfg)
        #     transform_1 = lambda data: data_utils.positional_encoding_adding_transform(data, cfg.neuralnet.pos_encoding_type, pos_encoding_size)
        # transform_2 = lambda x: x
        # if cfg.dataset.drop_atom_maps_rate > 0:
        #     drop_transform = lambda data: data_utils.drop_atom_maps(data, cfg.dataset.drop_atom_maps_rate)
        #     transform_2 = lambda data: drop_transform(data)
        # self.transform = lambda x: transform_2(transform_1(x))
        # self.transform = transform_1
        
        if self.stage == 'train': self.file_idx = 0
        elif self.stage == 'test': self.file_idx = 1
        else: self.file_idx = 2

        super().__init__(self.root, transform=self.transform)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])     
        print(f'end of loading')

    def transform(self, x):
        transform_1 = lambda x: x
        if (self.cfg.neuralnet.pos_encoding_type != 'none' and self.cfg.neuralnet.pos_encoding_type != 'no_pos_enc'):
            pos_encoding_size = data_utils.get_pos_enc_size(self.cfg)
            transform_1 = lambda data: data_utils.positional_encoding_adding_transform(data, self.cfg.neuralnet.pos_encoding_type, pos_encoding_size)
        transform_2 = lambda x: x
        if self.cfg.dataset.drop_atom_maps_rate > 0:
            drop_transform = lambda data: data_utils.drop_atom_maps(data, self.cfg.dataset.drop_atom_maps_rate)
            transform_2 = lambda data: drop_transform(data)
        transform_3 = lambda x: x
        if self.cfg.neuralnet.add_product_identifier:
            product_id_transform = lambda data: data_utils.add_product_id(data)
            transform_3 =product_id_transform
        transform_4 = lambda x: x
        if self.cfg.dataset.add_supernodes:
            add_supernodes_transform = lambda data: data_utils.add_supernodes(self.cfg, data)
            transform_4 = add_supernodes_transform
        transform_5 = lambda x: x
        if self.cfg.dataset.add_supernode_edges:
            add_supernode_edges_transform = lambda data: data_utils.add_supernode_edges(self.cfg, data)
            transform_5 = add_supernode_edges_transform
        return transform_5(transform_4(transform_3(transform_2(transform_1(x)))))

    @property
    def raw_file_names(self):
        return raw_files

    @property
    def processed_file_names(self):
        return processed_files

    # Function to split your dataset into chunks
    def split_dataset(self, dataset, num_chunks):
        #print(f'dataset len: {len(dataset)}, dataset: {dataset}, num_chunks: {num_chunks}')
        # Split the dataset into 'num_chunks' parts and return a list of chunks
        # chunk_size = len(dataset) // min(num_chunks, len(dataset))
        print(f'len dataset: {len(dataset)}')
        num_chunks = min(num_chunks+1, len(dataset))
        print(f'num_chunks: {num_chunks}')
        chunk_borders = np.linspace(0,len(dataset),num_chunks).astype(int)
        print(f'chunk_borders: {chunk_borders}')
        chunk_min = chunk_borders[:-1] if len(chunk_borders) > 1 else [0]
        chunk_max = chunk_borders[1:] if len(chunk_borders) > 1 else [1]
        print(f'chunk_min: {chunk_min}, chunk_max: {chunk_max}')
        return [(dataset[i:j], i) for i,j in zip(chunk_min, chunk_max)]#range(0, len(dataset), chunk_size)]

    def sub_process_with_threading(self, dataset_chunk):
        assert DUMMY_RCT_NODE_TYPE in self.cfg.dataset.atom_types, 'DUMMY_RCT_NODE_TYPE not in atom_types.'

        def process_reaction(rxn_data):
            rxn_, idx = rxn_data
            reactants = [r for r in rxn_.split('>>')[0].split('.') if r]
            products = [p for p in rxn_.split('>>')[1].split('.') if p]
            
            if not reactants or not products:
                return None, None

            g = graph.turn_reactants_and_product_smiles_into_graphs(
                self.cfg, reactants, products, idx + dataset_chunk[1], self.stage
            )
            return g, rxn_ if g is not None else None

        def chunk_generator(data, chunk_size=1000):
            for i in range(0, len(data), chunk_size):
                yield data[i:i+chunk_size]

        def process_chunk(chunk):
            graphs = []
            filtered_raw_rxns = []
            for result in chunk:
                if result[0] is not None:
                    graphs.append(result[0])
                    filtered_raw_rxns.append(result[1])
            return graphs, filtered_raw_rxns

        def sub_function():
            all_graphs = []
            all_filtered_raw_rxns = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # process the reactions in parallel
                futures = [executor.submit(process_reaction, (rxn, i)) 
                        for i, rxn in enumerate(dataset_chunk[0])]
                
                # split the processed reactions into chunks and process them in parallel
                # this is mainly to limit memory usage: 
                for chunk in chunk_generator(futures):
                    chunk_graphs, chunk_filtered_raw = process_chunk([future.result() for future in as_completed(chunk)])
                    all_graphs.extend(chunk_graphs)
                    all_filtered_raw_rxns.extend(chunk_filtered_raw)
                    
                    if len(all_graphs) >= 10000:  # Adjust this threshold as needed
                        self.save_intermediate_results(all_graphs, all_filtered_raw_rxns, dataset_chunk[1])
                        all_graphs = []
                        all_filtered_raw_rxns = []

            if all_graphs:
                self.save_intermediate_results(all_graphs, all_filtered_raw_rxns, dataset_chunk[1])

            return all_graphs, all_filtered_raw_rxns

        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is not None:  # In debugger:
            return sub_function()
        else:
            try:
                return sub_function()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f'Error in sub_process: {e}', exc_info=True)
                return [], []  # Return empty lists in case of an error

    def save_intermediate_results(self, graphs, filtered_raw_rxns, chunk_id):
        os.makedirs(self.processed_paths[self.file_idx].split('.pt')[0], exist_ok=True)
        subprocess_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'graphs_{chunk_id}_{chunk_id+len(graphs)}.pickle')
        subprocess_filtered_raw_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'filtered_raw_{chunk_id}_{chunk_id+len(filtered_raw_rxns)}.csv')
        
        if graphs:
            with open(subprocess_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f'Saved {len(graphs)} graphs in {subprocess_path}')
            
            with open(subprocess_filtered_raw_path, 'w') as f:
                f.writelines(filtered_raw_rxns)
            print(f'Saved {len(filtered_raw_rxns)} filtered raw reactions in {subprocess_filtered_raw_path}')
        else:
            print(f'No graphs found for dataset chunk {chunk_id}. Skipping this chunk.')
            
    # Main function to process the dataset in parallel
    def process(self):
        # merge only
        # Collect results
        if self.cfg.dataset.merge_only:
            all_graphs = []
            all_filtered_raw = []
            for filename in os.listdir(self.processed_paths[self.file_idx].split('.pt')[0]):
                if filename.startswith('graphs_'):
                    with open(os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], filename), 'rb') as f:
                        all_graphs.extend(pickle.load(f))
                elif filename.startswith('filtered_raw_'):
                    with open(os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], filename), 'r') as f:
                        all_filtered_raw.extend(f.readlines())
                                    
            torch.save(self.collate(all_graphs), self.processed_paths[self.file_idx])
            with open(self.processed_paths[self.file_idx].split('.pt')[0]+'_filtered_raw.csv', 'w') as f:
                f.writelines(all_filtered_raw)
            OmegaConf.save(self.cfg.dataset, self.config_path)
            return 
            
        # Split dataset into chunks based on the number of processes
        # if self.file_idx != 1: # TODO: change
        #     return

        # Create a pool of worker processes
        # with Pool(self.num_processes) as pool:
        #     # Parallelize the sub_process function across the dataset chunks
        #     all_graphs_nested_list = pool.map(self.sub_process, dataset_chunks)
        
        # TODO this is now sequential
        # for dataset_chunk in dataset_chunks:
        #     self.sub_process(dataset_chunk)

        # dataset_chunks # <- we want to merge these chunks
        if self.cfg.dataset.chunk_idx is not None and self.cfg.dataset.chunk_idx != 'None':
            self.cfg.dataset.chunk_idx  = int(self.cfg.dataset.chunk_idx)
            dataset = open(self.raw_paths[self.file_idx],'r').readlines()
            # assumes recalculate_atom_maps is done
            # this is a quick fix to only process a subset of the dataset
            num_chunks = min(self.cfg.dataset.num_processes+1, len(dataset))
            # get all the chunk borders
            chunk_borders = np.linspace(0,len(dataset),num_chunks).astype(int)
            chunk_idx_loc = np.where(chunk_borders == self.cfg.dataset.chunk_idx)[0][0]
            min_chunk_idx = chunk_borders[chunk_idx_loc]
            max_chunk_idx = chunk_borders[chunk_idx_loc+1] if chunk_idx_loc+1 < len(chunk_borders) else len(dataset)
            dataset_chunk = (dataset[min_chunk_idx:max_chunk_idx], min_chunk_idx)
            log.info(f'num_chunks: {num_chunks}')
            log.info(f'# of dataset chunks: {len(dataset_chunk)}')
            log.info(f'start_idx: {min_chunk_idx}')
            log.info(f'end_idx: {max_chunk_idx }')
            log.info(f'chunk size: {max_chunk_idx-min_chunk_idx}')
            log.info(f'dataset_chunk length: {len(dataset_chunk[0])}')
            graphs_, filtered_raw_rxns_ = self.sub_process(dataset_chunk)
        else:
            log.info(f'processing all chunks')
            if self.cfg.dataset.recalculate_atom_maps:
                log.info(f'recalculating atom maps')
                recalculated_am_rxns_path = self.processed_paths[self.file_idx].split('.pt')[0]+ '_recalculated_am.txt'
                # if recalculated_am_rxns_path exists, load the list of strings there
                if os.path.exists(recalculated_am_rxns_path):
                    log.info(f'recalculated_am_rxns_path found, loading')
                    with open(recalculated_am_rxns_path, 'r') as f:
                        lines = f.readlines()
                    dataset_chunks = self.split_dataset(lines, int(self.cfg.dataset.num_processes))
                else:
                    log.info(f'recalculate_atom_maps is true, but no recalculated_am_rxns_path found, so recalculating')
                    num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
                    dataset_chunks = self.split_dataset(open(self.raw_paths[self.file_idx], 'r').readlines(), int(num_processes))
                    
                    log.info(f'after splitting data in recalc am')
                    print(f'# chunks: {len(dataset_chunks)}')
                    #recalculated_am_rxns = self.sub_process_atom_mapping(0, dataset_chunks[0])
                    gettrace = getattr(sys, 'gettrace', None) # TODO: change
                    if gettrace is None:
                        import multiprocessing as mp
                        mp.set_start_method('spawn', force=True)
                        with mp.Pool(processes=num_processes) as pool:
                            log.info(f'recalculating atom maps with {num_processes} processes for len dataset_chunks {len(dataset_chunks)}')
                            recalculated_am_rxns = pool.starmap(self.sub_process_atom_mapping, enumerate(dataset_chunks))
                            log.info(f'back from process')
                    else:
                        recalculated_am_rxns = self.sub_process_atom_mapping(0, dataset_chunks[0])
                        recalculated_am_rxns = [recalculated_am_rxns]
                    #dataset_chunks = recalculated_am_rxns
                    with open(recalculated_am_rxns_path, 'w', encoding='utf-8') as file: # save the strings
                        for chunk in recalculated_am_rxns:
                            for string in chunk[0]:
                                file.write(string + '\n')
                    print(f'after recalculating atom maps')
                    dataset_chunks = self.split_dataset(open(recalculated_am_rxns_path,'r').readlines(), int(self.cfg.dataset.num_processes))
            else:
                log.info(f'not recalculating atom maps')
                dataset_chunks = self.split_dataset(open(self.raw_paths[self.file_idx],'r').readlines(), int(self.cfg.dataset.num_processes))

            gettrace = getattr(sys, 'gettrace', None) # TODO: change
            if gettrace is None:
                log.info(f'processing chunks in parallel')
                with Pool(self.cfg.dataset.num_processes) as pool:
                    # Parallelize the sub_process function across the dataset chunks
                    # async_results = []
                    for chunk_idx, dataset_chunk in enumerate(dataset_chunks):
                        log.info(f'processing chunk_idx {chunk_idx}, start index {dataset_chunk[1]}\n')
                        async_result = pool.apply_async(self.sub_process, args=(dataset_chunk,))
                        # async_results.append(async_result)
                    pool.close()
                    pool.join()
                    
                    # Collect results
                    log.info(f'collecting results')
                    all_graphs = []
                    all_filtered_raw = []
                    for filename in os.listdir(self.processed_paths[self.file_idx].split('.pt')[0]):
                        if filename.startswith('graphs_'):
                            with open(os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], filename), 'rb') as f:
                                all_graphs.extend(pickle.load(f))
                        elif filename.startswith('filtered_raw_'):
                            with open(os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], filename), 'r') as f:
                                all_filtered_raw.extend(f.readlines())
                    
                    # all_graphs = []
                    # all_filtered_raw = []
                    # for i, async_result in enumerate(async_results):
                    #     print(f'asyn_result # {i}')
                    #     try:
                    #         graphs, filtered_raw = async_result.get()
                    #         if graphs:  # Only append non-empty results
                    #             all_graphs.extend(graphs)
                    #             all_filtered_raw.extend(filtered_raw)
                    #     except Exception as e:
                    #         log.info(f'Error processing chunk: {e}')
            else:
                log.info(f'processing chunks sequentially')
                all_graphs = []
                all_filtered_raw = []
                print(f'len dataset chunks: {len(dataset_chunks)}')
                for dataset_chunk in dataset_chunks:
                    graphs, filtered_raw = self.sub_process(dataset_chunk)
                    if graphs:  # Only append non-empty results
                        all_graphs.extend(graphs)
                        all_filtered_raw.extend(filtered_raw)

            if not all_graphs:
                log.warning(f"No graphs were generated for any chunks in {self.processed_paths[self.file_idx]}.")
                return  # Exit the function if no graphs were generated

            torch.save(self.collate(all_graphs), self.processed_paths[self.file_idx])
            with open(self.processed_paths[self.file_idx].split('.pt')[0]+'_filtered_raw.csv', 'w') as f:
                f.writelines(all_filtered_raw)
            OmegaConf.save(self.cfg.dataset, self.config_path)

            # print path
            #print(f'dataset_chunks path: {self.raw_paths[self.file_idx]}')
            #print(f'dataset_chunks: {dataset_chunks}')
            #print(f'len dataset chunks: {len(dataset_chunks)}')
            #dataset_chunks = dataset_chunks[:1]
            #print(f'len dataset chunks after: {len(dataset_chunks)}')
            # gettrace = None # getattr(sys, 'gettrace', None) TODO: change
            # log.info("Gettrace: {}".format(gettrace))
            # if gettrace is None:
            #     with Pool(self.cfg.dataset.num_processes) as pool:
            #         # Parallelize the sub_process function across the dataset chunks without collecting any output
            #         async_results = []
            #         for chunk_idx, dataset_chunk in enumerate(dataset_chunks):
            #             log.info(f'processing chunk_idx {chunk_idx}\n')
            #             #print(f'dataset_chunk: {dataset_chunk}')
            #             async_result = pool.apply_async(self.sub_process, args=(dataset_chunk,))
            #             async_results.append(async_result)
            #         pool.close() # Prevents any more tasks from being submitted to the pool
            #         pool.join() # Wait for the worker processes to exit
            #         # Optionally, handle the results or exceptions
            #         for async_result in async_results:
            #             try:
            #                 async_result.get()  # This will raise any exceptions that occurred in the worker
            #             except Exception as e:
            #                 log.info(f'Error processing chunk: {e}')
            # else:
            #     for dataset_chunk in dataset_chunks:
            #         self.sub_process(dataset_chunk)

            # all_graphs = []
            # all_filtered_raw = []
            # for chunk_idx, dataset_chunk in enumerate(dataset_chunks):
            #     log.info(f'merging chunk_idx {chunk_idx}\n')
            #     subprocess_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'graphs_{dataset_chunk[1]}.pickle')
            #     graph = pickle.load(open(subprocess_path, 'rb'))
            #     all_graphs.append(graph)
            #     filtered_raw_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'filtered_raw_{dataset_chunk[1]}.csv')
            #     filtered_raw = open(filtered_raw_path, 'r').readlines()
            #     all_filtered_raw.extend(filtered_raw)
                
            # every_graph = [g for subgraph in all_graphs for g in subgraph]
            # # interim_path = os.path.join('/'.join(self.processed_paths[self.file_idx].split('/')[:-1]), 'nested_list.pickle')
            # # pickle.dump(all_graphs_nested_list, open(interim_path, 'wb'))
            
            # # # 'all_graphs_nested_list' will be a list of lists (each inner list is the result from a single chunk)
            # # all_graphs = [graph for sublist in all_graphs_nested_list for graph in sublist]
            # # list_path = os.path.join(self.processed_paths[self.file_idx].split('/')[:-1], self.stage+'.pickle')
            # # pickle.dump(all_graphs, open(list_path, 'wb'))
            # torch.save(self.collate(every_graph), self.processed_paths[self.file_idx])
            # with open(self.processed_paths[self.file_idx].split('.pt')[0]+'_filtered_raw.csv', 'w') as f:
            #     f.writelines(all_filtered_raw)
            # OmegaConf.save(self.cfg.dataset, self.config_path)

    def turn_reactants_and_product_smiles_into_graphs_OLD(self, reactants, products, data_idx):
        # preprocess: get total number of product nodes
        nb_product_nodes = sum([len(Chem.MolFromSmiles(p.strip()).GetAtoms()) for p in products])
        nb_rct_nodes = sum([len(Chem.MolFromSmiles(r.strip()).GetAtoms()) for r in reactants])

        nb_dummy_toadd_to_rcts = nb_product_nodes + self.cfg.dataset.nb_rct_dummy_nodes - nb_rct_nodes
        if nb_dummy_toadd_to_rcts<0 and self.stage=='train':
            # drop the rxns in the training set which we cannot generate
            return None

        offset = 0
        for j, r in enumerate(reactants):
            # TODO: How to actually do this? 
            # Should probably just always add the stereo-tags, and just not use them if not needed
            # But: We do require some code for handling whether ot not to have the charges as separate features
            # hmm or could actually do that later, it doesn't hurt to process them as well. 
            # NOTE: with_explicit_h is deprecated now

            nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = mol.smiles_to_graph_with_stereochem(smi=r, cfg=self.cfg)

            edge_index += offset
            nodes_rct = torch.cat((nodes_rct, nodes), dim=0) if j > 0 else nodes # already a tensor
            edge_index_rct = torch.cat((edge_index_rct, edge_index), dim=1) if j > 0 else edge_index
            bond_types_rct = torch.cat((bond_types_rct, bond_types), dim=0) if j > 0 else bond_types
            atom_map_reactants = torch.cat((atom_map_reactants, atom_map), dim=0) if j > 0 else atom_map
            atom_charges_rct = torch.cat((atom_charges_rct, atom_charges), dim=0) if j > 0 else atom_charges
            atom_chiral_rct = torch.cat((atom_chiral_rct, atom_chiral), dim=0) if j > 0 else atom_chiral
            bond_dirs_rct = torch.cat((bond_dirs_rct, bond_dirs), dim=0) if j > 0 else bond_dirs
            mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.ones(nodes.shape[0], dtype=torch.long) * j+1], dim=0) if j > 0 else torch.ones(nodes.shape[0], dtype=torch.long) * j+1
            offset += nodes.shape[0]

            # nodes, edge_index, bond_types, atom_map = mol.mol_to_graph(mol=r, atom_types=self.atom_types,
            #                                                                bond_types=self.bond_types,
            #                                                                with_explicit_h=self.with_explicit_h,
            #                                                                with_formal_charge=self.with_formal_charge,
            #                                                                offset=offset, get_atom_mapping=True,
            #                                                                canonicalize_molecule=self.canonicalize_molecule)
            # nodes_rct = torch.cat((nodes_rct, nodes), dim=0) if j > 0 else nodes # already a tensor
            # edge_index_rct = torch.cat((edge_index_rct, edge_index), dim=1) if j > 0 else edge_index
            # bond_types_rct = torch.cat((bond_types_rct, bond_types), dim=0) if j > 0 else bond_types
            # atom_map_reactants = torch.cat((atom_map_reactants, atom_map), dim=0) if j > 0 else atom_map
            # mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.ones(nodes.shape[0], dtype=torch.long) * j+1], dim=0) if j > 0 else torch.ones(nodes.shape[0], dtype=torch.long) * j+1
            # offset += nodes.shape[0]
        print(f'nb_dummy_toadd_to_rcts: {nb_dummy_toadd_to_rcts}')
        if nb_dummy_toadd_to_rcts>0:
            nodes_dummy = torch.ones(nb_dummy_toadd_to_rcts, dtype=torch.long) * self.cfg.dataset.atom_types.index(DUMMY_RCT_NODE_TYPE)
            nodes_dummy = F.one_hot(nodes_dummy, num_classes=len(self.cfg.dataset.atom_types)).float() # This is hardcoded
            edges_idx_dummy = torch.zeros([2, 0], dtype=torch.long)
            bond_types_dummy = torch.zeros([0, len(self.cfg.dataset.bond_types)], dtype=torch.long)
            nodes_rct = torch.cat([nodes_rct, nodes_dummy], dim=0)
            edge_index_rct = torch.cat([edge_index_rct, edges_idx_dummy], dim=1)
            bond_types_rct = torch.cat([bond_types_rct, bond_types_dummy], dim=0)
            atom_charges_rct = torch.cat([atom_charges_rct, F.one_hot(torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long), num_classes=len(self.cfg.dataset.atom_charges))], dim=0)
            atom_chiral_rct = torch.cat([atom_chiral_rct, F.one_hot(torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long), num_classes=len(self.cfg.dataset.atom_chiral_tags))], dim=0)
            bond_dirs_rct = torch.cat([bond_dirs_rct, torch.zeros([0, len(self.cfg.dataset.bond_dirs)], dtype=torch.long)], dim=0)
            atom_map_reactants = torch.cat([atom_map_reactants, torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long)], dim=0)
            mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.zeros(nb_dummy_toadd_to_rcts, dtype=torch.long)], dim=0)

        # Permute the rows here to make sure that the NN can only process topological information
        if self.cfg.dataset.permute_mols:
            data_utils.permute_rows(nodes_rct, atom_map_reactants, mol_assignment_reactants, edge_index_rct)

        offset = 0
        for j, p in enumerate(products):
            # if data_idx == 56:
            #     breakpoint()
            nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = mol.smiles_to_graph_with_stereochem(smi=p, cfg=self.cfg)

            nodes_prod = torch.cat((nodes_prod, nodes), dim=0) if j > 0 else nodes # already a tensor
            edge_index_prod = torch.cat((edge_index_prod, edge_index), dim=1) if j > 0 else edge_index
            bond_types_prod = torch.cat((bond_types_prod, bond_types), dim=0) if j > 0 else bond_types
            atom_charges_prod = torch.cat((atom_charges_prod, atom_charges), dim=0) if j > 0 else atom_charges
            atom_chiral_prod = torch.cat((atom_chiral_prod, atom_chiral), dim=0) if j > 0 else atom_chiral
            bond_dirs_prod = torch.cat((bond_dirs_prod, bond_dirs), dim=0) if j > 0 else bond_dirs
            atom_map_products = torch.cat((atom_map_products, atom_map), dim=0) if j > 0 else atom_map
            mol_assignment_products = torch.cat([mol_assignment_products, torch.ones(nodes.shape[0], dtype=torch.long) * len(reactants)+j+1], dim=0) if j > 0 else torch.ones(nodes.shape[0], dtype=torch.long) * len(reactants)+j+1
            offset += nodes.shape[0]

        y = torch.zeros((1, 0), dtype=torch.float)
        
        # perm = torch.arange(max(atom_map_reactants.max().item(), atom_map_products.max().item())+1)[1:]
        # perm = perm[torch.randperm(len(perm))]
        # perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])
        # atom_map_reactants = perm[atom_map_reactants]
        # atom_map_products = perm[atom_map_products]
        
        # TODO : Clean some of this stuff up, less asserts, and have them somewhere else
        if self.stage == 'train':
            #assert (atom_map_reactants != 0).sum() == (atom_map_products != 0).sum()
            atom_map_reactants, atom_map_products = data_utils.fix_atom_mappings(atom_map_reactants, atom_map_products)
            # # should have equal number of atom mappings on both sides
            # assert (atom_map_reactants != 0).sum() == (atom_map_products != 0).sum(), f'{atom_map_reactants}, {atom_map_products}'
            # # atom maps should be unique
            # assert len(set(atom_map_reactants.tolist()) - set([0])) == (atom_map_reactants != 0).sum().item()
            # assert len(set(atom_map_products.tolist())) == (atom_map_products != 0).sum().item()
            equal_number_of_atom_map_numbers = (atom_map_reactants!=0).sum() == (atom_map_products != 0).sum()
            unique_reactants_atom_map_numbers = len(set(atom_map_reactants.tolist()) - set([0])) == (atom_map_reactants != 0).sum().item()
            unique_product_atom_map_numbers = len(set(atom_map_products.tolist())) == (atom_map_products != 0).sum().item()
            if not equal_number_of_atom_map_numbers or not unique_reactants_atom_map_numbers or \
                not unique_product_atom_map_numbers:
                    print(f'skipping rxn {data_idx} in train\n')
                    print(f'==== equal_number_of_atom_map_numbers={equal_number_of_atom_map_numbers}, unique_reactants_atom_map_numbers={unique_reactants_atom_map_numbers}, unique_product_atom_map_numbers={unique_product_atom_map_numbers}')
                    return None  
            if set(atom_map_reactants.tolist()) - set([0]) != set(atom_map_products.tolist()) - set([0]):
                print("SOMETHING WRONG HERE")
                print(atom_map_reactants)
                print(atom_map_products)
        else:
            if self.cfg.dataset.name != 'uspto-50k':
                atom_map_reactants, atom_map_products = data_utils.fix_atom_mappings(atom_map_reactants, atom_map_products) # Just use the same for now
                # # should have equal number of atom mappings on both sides
                # assert (atom_map_reactants != 0).sum() == (atom_map_products != 0).sum(), f'{atom_map_reactants}, {atom_map_products}'
                # # atom maps should be unique
                # assert len(set(atom_map_reactants.tolist()) - set([0])) == (atom_map_reactants != 0).sum().item()
                # assert len(set(atom_map_products.tolist())) == (atom_map_products != 0).sum().item()
                equal_number_of_atom_map_numbers = (atom_map_reactants != 0).sum() == (atom_map_products != 0).sum()
                unique_reactants_atom_map_numbers = len(set(atom_map_reactants.tolist()) - set([0])) == (atom_map_reactants != 0).sum().item()
                unique_product_atom_map_numbers = len(set(atom_map_products.tolist())) == (atom_map_products != 0).sum().item()
                if not equal_number_of_atom_map_numbers or not unique_reactants_atom_map_numbers or \
                    not unique_product_atom_map_numbers:
                        print(f'skipping rxn {data_idx} in test/val\n')
                        return None 
            assert set(atom_map_reactants.tolist()) - set([0]) == set(atom_map_products.tolist()) - set([0]), f'Atom map numbers not equal for rxn {data_idx} in subset {self.stage}: atom_map_reactants={atom_map_reactants}, atom_map_products={atom_map_products}'
            pass

        # Align the graphs here according to the atom mapping, so that alignment is not necessary afterwards
        if self.cfg.dataset.pre_align_graphs:
            # First permute the atom mappings
            # TODO: SEEMS THAT THERE ARE CASES WHERE max(atom_map_reactants) != max(atom_map_products)...
            perm = torch.cat([torch.tensor([0], dtype=torch.long), 1+torch.randperm(atom_map_reactants.max())])
            atom_map_reactants = perm[atom_map_reactants]
            atom_map_products = perm[atom_map_products]

            atom_map_reactants, indices = torch.sort(atom_map_reactants, dim=0, descending=True) # put the non-atom-mapped stuff as last
            inverse_indices = torch.argsort(indices) #torch.tensor([idx[0].item() for idx in sorted(zip(torch.arange(len(indices)), indices), key=lambda x: x[1])])
            nodes_rct = nodes_rct[indices]
            edge_index_rct = inverse_indices[edge_index_rct]
            mol_assignment_reactants = mol_assignment_reactants[indices]
            atom_chiral_rct = atom_chiral_rct[indices]
            atom_charges_rct = atom_charges_rct[indices]

            atom_map_products, indices = torch.sort(atom_map_products, dim=0, descending=True) # put the non-atom-mapped stuff as last
            inverse_indices = torch.argsort(indices)
            nodes_prod = nodes_prod[indices]
            edge_index_prod = inverse_indices[edge_index_prod]
            mol_assignment_products = mol_assignment_products[indices]
            atom_chiral_prod = atom_chiral_prod[indices]
            atom_charges_prod = atom_charges_prod[indices]

        # Check that the one-hot encodings are correct
        assert all(bond_types_rct.sum(-1) == 1) # Should be one-hot encodings here
        assert all(bond_types_prod.sum(-1) == 1) # TODO: REPLACE WITH .all() for efficiency
        assert all(nodes_rct.sum(-1) == 1)
        assert all(nodes_prod.sum(-1) == 1)

        # Make sure that there are no duplicate edges
        assert len(set([(edge[0].item(), edge[1].item()) for edge in edge_index_rct.T])) == edge_index_rct.shape[1]
        assert len(set([(edge[0].item(), edge[1].item()) for edge in edge_index_prod.T])) == edge_index_prod.shape[1]

        # Make sure that there are no edges pointing to nodes that don't exist
        assert all((edge_index_rct < nodes_rct.shape[0]).flatten())
        assert all((edge_index_prod < nodes_prod.shape[0]).flatten())

        smiles_to_save = ".".join(reactants) + ">>" + ".".join(products) #data_utils.remove_atom_mapping_from_reaction(".".join(reactants) + ">>" + ".".join(products))

        graph = Data(x=torch.cat([nodes_rct, nodes_prod], dim=0), 
                     edge_index=torch.cat([edge_index_rct, edge_index_prod + len(nodes_rct)], dim=1),
                     edge_attr=torch.cat([bond_types_rct, bond_types_prod], dim=0), y=y, idx=data_idx,
                     mol_assignment=torch.cat([mol_assignment_reactants, mol_assignment_products], dim=0),
                     atom_map_numbers=torch.cat([atom_map_reactants, atom_map_products], dim=0),
                     smiles=smiles_to_save,
                     atom_charges=torch.cat([atom_charges_rct, atom_charges_prod], dim=0),
                     atom_chiral=torch.cat([atom_chiral_rct, atom_chiral_prod], dim=0),
                     bond_dirs=torch.cat([bond_dirs_rct, bond_dirs_prod], dim=0))
        
        return graph

    def sub_process_atom_mapping(self, chunk_id, dataset_chunk):
        log.info(f'===== in sub_process_atom_mapping, chunk_id={chunk_id}, using cuda ={torch.cuda.is_available()}')
        import chython
        from chython import smiles
        chython.torch_device = f'cuda:{chunk_id}' if torch.cuda.is_available() else 'cpu'

        results = []
        print(f'len(dataset_chunk[0])={len(dataset_chunk[0])}')
        for i in range(len(dataset_chunk[0])):
            log.info(f'recalculating atom maps for reaction {i}')
            smiles_rxn = dataset_chunk[0][i]
            try:
                r_mol = Chem.MolFromSmiles(smiles_rxn.split('>>')[0])
                [r_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(r_mol.GetNumAtoms())]
                p_mol = Chem.MolFromSmiles(smiles_rxn.split('>>')[1])
                [p_mol.GetAtomWithIdx(a).ClearProp('molAtomMapNumber') for a in range(p_mol.GetNumAtoms())]
                smiles_rxn = Chem.MolToSmiles(r_mol, canonical=True) + ">>" + Chem.MolToSmiles(p_mol, canonical=True)
                chython_smiles = smiles(smiles_rxn)
                chython_smiles.reset_mapping() # assign atom mappings with the chython library
                regular_smiles = format(chython_smiles, 'm')
                #print(f'regular_smiles={regular_smiles}')
                # remove atom mappings from the reactant side for atoms that don't have atom maps
                r_mol = Chem.MolFromSmiles(regular_smiles.split('>>')[0])
                p_mol = Chem.MolFromSmiles(regular_smiles.split('>>')[1])
                p_am_set = set([a.GetAtomMapNum() for a in p_mol.GetAtoms() if a.GetAtomMapNum() != 0])
                for atom in r_mol.GetAtoms():
                    if atom.GetAtomMapNum() not in p_am_set:
                        atom.SetAtomMapNum(0)
                results.append(Chem.MolToSmiles(r_mol, canonical=True) + ">>" + Chem.MolToSmiles(p_mol, canonical=True))
            except Exception as e:
                print(f'Error in sub_process_atom_mapping for reaction {i}: {e}')
                results.append(smiles_rxn)
            log.info(f"{i} atom mapped")
            #print(f"{i} atom mapped")

        return [results, dataset_chunk[1]]

    def sub_process(self, dataset_chunk):
        assert DUMMY_RCT_NODE_TYPE in self.cfg.dataset.atom_types, 'DUMMY_RCT_NODE_TYPE not in atom_types.'
        
        def sub_function():
            graphs = []
            filtered_raw_rxns = []
            for i, rxn_ in enumerate(dataset_chunk[0]):
                log.info(f'Processing reaction {i}')
                
                reactants = [r for r in rxn_.split('>>')[0].split('.')]
                products = [p for p in rxn_.split('>>')[1].split('.')]
                # skip
                if reactants == [''] or products == ['']:
                    continue
                g = graph.turn_reactants_and_product_smiles_into_graphs(self.cfg, reactants, products, i+dataset_chunk[1], self.stage)
                if g is not None:
                    graphs.append(g)
                    filtered_raw_rxns.append(rxn_)
                else:
                    log.info(f'No graphs for reaction {i}')
            
            os.makedirs(self.processed_paths[self.file_idx].split('.pt')[0], exist_ok=True)
            subprocess_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'graphs_{dataset_chunk[1]}.pickle')
            subprocess_filtered_raw_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'filtered_raw_{dataset_chunk[1]}.csv')
            
            if len(graphs) > 0:
                pickle.dump(graphs, open(subprocess_path, 'wb'))
                print(f'saved subprocess graphs in {subprocess_path}\n')
                open(subprocess_filtered_raw_path, 'w').writelines(filtered_raw_rxns)
                print(f'saved filtered raw in {subprocess_filtered_raw_path}\n')
            else:
                print(f'No graphs found for dataset chunk {dataset_chunk[1]}. Skipping this chunk.\n')
            
            return graphs, filtered_raw_rxns

        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is not None:  # In debugger:
            log.info(f'in sub_process, gettrace={gettrace}')
            return sub_function()
        else:
            try:
                log.info(f'in sub_process, gettrace={gettrace}')
                return sub_function()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f'Error in sub_process: {e}', exc_info=True)
                return [], []  # Return empty lists in case of an error

        # def sub_process(self, dataset_chunk):
        #     assert DUMMY_RCT_NODE_TYPE in self.cfg.dataset.atom_types, 'DUMMY_RCT_NODE_TYPE not in atom_types.'
        #     def sub_function():
        #         #print(f'Processing dataset chunk in sub function, len={len(dataset_chunk[0])}')
        #         graphs = []
        #         filtered_raw_rxns = []

        #         for i, rxn_ in enumerate(dataset_chunk[0]):
        #             #print(f'Processing reaction {i}')
        #             reactants = [r for r in rxn_.split('>>')[0].split('.')]
        #             products = [p for p in rxn_.split('>>')[1].split('.')]
                    
        #             # skip 
        #             if reactants==[''] or products==['']:
        #                 continue

        #             g = graph.turn_reactants_and_product_smiles_into_graphs(self.cfg, reactants, products, i+dataset_chunk[1], self.stage)
                    
        #             # self.turn_reactants_and_product_smiles_into_graphs(reactants, products, data_idx=i+dataset_chunk[1])
        #             if g is not None:
        #                 graphs.append(g)
        #                 filtered_raw_rxns.append(rxn_)
                    
        #             #log.info(f"processed reaction {i}")
                    
        #         os.makedirs(self.processed_paths[self.file_idx].split('.pt')[0], exist_ok=True)
        #         subprocess_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'graphs_{dataset_chunk[1]}.pickle')
        #         subprocess_filtered_raw_path = os.path.join(self.processed_paths[self.file_idx].split('.pt')[0], f'filtered_raw_{dataset_chunk[1]}.csv')
                
        #         assert len(graphs)>0, f'No graphs found for dataset {self.processed_paths[self.file_idx]}.'
        #         torch.save(self.collate(graphs), self.processed_paths[self.file_idx])
        #         pickle.dump(graphs, open(subprocess_path, 'wb'))
        #         print(f'saved subprocess graphs in {subprocess_path}\n')
        #         open(subprocess_filtered_raw_path, 'w').writelines(filtered_raw_rxns)
        #         print(f'saved filtered raw in {subprocess_path}\n')
                    
        #     gettrace = getattr(sys, 'gettrace', None)
        #     if gettrace is not None: #In debugger:
        #         sub_function()
        #     else:
        #         try:
        #             sub_function()
        #         except Exception as e:
        #             logger = logging.getLogger(__name__)
        #             logger.error(f'Error in sub_process: {e}', exc_info=True)
                

class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)
    
    def prepare_data(self, shuffle=True, slices={'train':None, 'val':None, 'test':None}) -> None:

        if self.cfg.dataset.process_only=='train': 
            datasets = {'train': Dataset(stage='train', cfg=self.cfg)}
        elif self.cfg.dataset.process_only=='val': 
            datasets = {'val': Dataset(stage='val', cfg=self.cfg)}
        elif self.cfg.dataset.process_only=='test': 
            datasets = {'test': Dataset(stage='test', cfg=self.cfg)}
        else: 
            datasets = {'train': Dataset(stage='train', cfg=self.cfg),
                        'val': Dataset(stage='val', cfg=self.cfg),
                    'test': Dataset(stage='test', cfg=self.cfg)}
        
        for key in slices.keys():
            if slices[key] is not None and key in datasets.keys():
                datasets[key] = datasets[key][slices[key][0]:slices[key][1]]
                
        for key in datasets.keys():
            print(f'len {key} datasets {len(datasets[key])}\n')
        
        # TODO: Remove this super().prepare_data thing, and define a custom DataLoader class. 
        # TODO: Do we want to turn the data into dense format here? Do we want to calculate the Laplacian eigenvectors as well? -> probably. 
        # -> should probably have a legacy method to calculate the Laplacian eigenvector coefficients, as well as using pytorch_geometric to calculate the Laplacian eigenvector coefficients.
        
        super().prepare_data(datasets, shuffle=shuffle)

    def node_counts(self, max_nodes_possible=MAX_ATOMS_RXN):
        '''
            Number of nodes in a reaction.
        '''
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']: # over all datasets?
            for i, data in enumerate(self.dataloaders[split]):
                # batch_without_sn = data.batch[data.mask_sn] # No supernodes anymore
                _, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1

        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts/all_counts.sum()
        
        return all_counts

    def node_types(self):
        data = next(iter(self.dataloaders['train']))
        num_classes = data.x.shape[1] # including supernode 
        d = torch.zeros(num_classes)

        for data in self.dataloaders['train']:
            d += data.x.sum(dim=0) # supernode is at encoder index -1 => discard it
        d = d / d.sum()

        return d

    def edge_types(self):
        num_classes = None
        data = next(iter(self.dataloaders['train']))
        num_classes = data.edge_attr.shape[1]
        d = torch.zeros(num_classes)

        for i, data in enumerate(self.dataloaders['train']):
            # batch_without_sn = data.batch[data.mask_sn]
            unique, counts = torch.unique(data.batch, return_counts=True)
            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)
            non_sn_node_idx = (data.mask_sn==True).nonzero(as_tuple=True)[0]
            non_sn_edge_index, non_sn_edge_attr = subgraph(non_sn_node_idx, data.edge_index, data.edge_attr)

            num_edges = non_sn_edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = non_sn_edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:] 

        d = d/d.sum() 

        return d

    def node_types_unnormalized(self):
        #TODO: Can this be abstracted to the AbstractDataModule class?
        '''
            Return distribution over the of atom types in molecules.

            Output:
                counts: distribution over types of atoms in molecules.
        '''
        data = next(iter(self.dataloaders['train']))
        num_classes = data.x.shape[1] # get number of atom types from node encoding
        counts = torch.zeros(num_classes)

        for data in self.dataloaders['train']:
            counts += data.x.sum(dim=0)
            
        # ignore SuNo atom type 
        # (set frequencies to 0. because it does not appear in the data anyway)
        if 'SuNo' in self.cfg.dataset.atom_types:
            suno_idx = self.cfg.dataset.atom_types.index('SuNo')
            counts[suno_idx] = 0.

        return counts.long()
    
    def edge_types_unnormalized(self):
        #TODO: Can this be abstracted to the AbstractDataModule class?
        data = next(iter(self.dataloaders['train']))
        num_classes = data.edge_attr.shape[1]

        d = torch.zeros(num_classes)

        for i, data in enumerate(self.dataloaders['train']):
            _, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1) # all_pairs does not include edge from the node to itself

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges
            
            edge_types = data.edge_attr.sum(dim=0)
            edge_types += data.edge_attr.sum(dim=0)

            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]   
            
        # ignore SuNo edge types
        for t in ['mol', 'within', 'across']:
            if t in self.cfg.dataset.bond_types:
                suno_idx = self.cfg.dataset.bond_types.index(t)
                d[suno_idx] = 0.

        return d.long()
    
class DatasetInfos:
    def __init__(self, datamodule, cfg, recompute_info=False):
        '''
            zero_bond_order is a temporary fix to a bug in the creation of bond_orders that affects extra_features later.
            The fix is to accommodate models trained with the bug.
        '''
        self.datamodule = datamodule
        self.name = 'supernode_graphs'
        # self.atom_encoder = ['none']+atom_types # takes type (str) get idx (int)
        self.atom_decoder = cfg.dataset.atom_types
        self.bond_decoder = get_rdkit_bond_types(cfg.dataset.bond_types)
        self.atom_charges = cfg.dataset.atom_charges
        self.atom_chiral_tags = get_rdkit_chiral_tags(cfg.dataset.atom_chiral_tags)
        self.bond_dirs = get_rdkit_bond_dirs(cfg.dataset.bond_dirs)
        self.remove_h = cfg.dataset.remove_h
        # self.valencies = [0] + list(abs[0] for atom_type, abs in allowed_bonds.items() if atom_type in atom_types) + [0]
        if cfg.dataset.different_valencies:
            self.valencies = list(self.get_possible_valences(atom_type)[0] for atom_type in self.atom_decoder)
        else:
            self.valencies = [0] + list(abs[0] for atom_type, abs in cfg.dataset.allowed_bonds.items() if atom_type in cfg.dataset.atom_types) + [0]
        periodic_table = Chem.rdchem.GetPeriodicTable()
        atom_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in self.atom_decoder[1:-1]] + [0] # discard charge
        atom_weights = {atom_type: weight for atom_type, weight in zip(self.atom_decoder, atom_weights)}
        self.atom_weights = atom_weights
        self.max_weight = 390
        print(f'zero_bond_order {cfg.dataset.zero_bond_order}\n')
        if cfg.dataset.zero_bond_order: self.bond_orders = get_bond_orders(self.bond_decoder)
        else: self.bond_orders = get_bond_orders_correct(self.bond_decoder)
        print(f'self.bond_orders {self.bond_orders}\n')
        print(f"Base path: {os.path.realpath(__file__)}")
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.dataset_nb!='': self.datadist_dir += '-'+str(cfg.dataset.dataset_nb)
        root_path = os.path.join(base_path, self.datadist_dir, 'processed')
        print(f"Root path: {root_path}")
        node_count_path = os.path.join(root_path, 'n_counts.txt')
        atom_type_path = os.path.join(root_path, 'atom_types.txt')
        edge_type_path = os.path.join(root_path, 'edge_types.txt')
        atom_type_unnorm_path = os.path.join(root_path, 'atom_types_unnorm_mol.txt')
        edge_type_unnorm_path = os.path.join(root_path, 'edge_types_unnorm_mol.txt')
        print(f"Checking if all the paths exits ({node_count_path, atom_type_path, edge_type_path, atom_type_unnorm_path, edge_type_unnorm_path})")
        paths_exist = os.path.exists(node_count_path) and os.path.exists(atom_type_path)\
                      and os.path.exists(edge_type_path) and os.path.exists(atom_type_unnorm_path)\
                      and os.path.exists(edge_type_unnorm_path)
        print(f"{paths_exist}")

        if not recompute_info and paths_exist:
            # use the same distributions for all subsets of the dataset
            print(f"Loading the info files...")
            self.n_nodes = torch.from_numpy(np.loadtxt(node_count_path)).float()
            self.node_types = torch.from_numpy(np.loadtxt(atom_type_path)).float()
            self.edge_types = torch.from_numpy(np.loadtxt(edge_type_path)).float()
            self.node_types_unnormalized = torch.from_numpy(np.loadtxt(atom_type_unnorm_path)).long()
            self.edge_types_unnormalized = torch.from_numpy(np.loadtxt(edge_type_unnorm_path)).long()
            print(f"Loaded the info files")
        else:
            print('Recomputing\n')
            np.set_printoptions(suppress=True, precision=5)

            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(node_count_path, self.n_nodes.cpu().numpy())

            self.node_types_unnormalized = datamodule.node_types_unnormalized()
            print("Counts of node types", self.node_types_unnormalized)
            np.savetxt(atom_type_unnorm_path, self.node_types_unnormalized.cpu().numpy())

            self.edge_types_unnormalized = datamodule.edge_types_unnormalized()
            print("Counts of edge types", self.edge_types_unnormalized)
            np.savetxt(edge_type_unnorm_path, self.edge_types_unnormalized.cpu().numpy())

            self.node_types = self.node_types_unnormalized / self.node_types_unnormalized.sum()
            print("Distribution of node types", self.node_types)
            np.savetxt(atom_type_path, self.node_types.cpu().numpy())

            self.edge_types = self.edge_types_unnormalized / self.edge_types_unnormalized.sum()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(edge_type_path, self.edge_types.cpu().numpy())

        print("Completing infos...")
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        print("Done completing infos!")
        
    def get_possible_valences(self, atom_type):
        # TODO: is this buggy somehow?
        pt = Chem.GetPeriodicTable()
        try:
            valence_list = pt.GetValenceList(atom_type)
        except:
            valence_list = [0]
    
        return list(valence_list)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, dx=None, de=None, dy=None, datamodule=None):
        assert datamodule is not None or dx is not None, f'Got datamodule={datamodule} and dx={dx}. One of the two should be specified.\n'
        
        log.info("Computing input/output dims (in function)")

        if dx is not None and de is not None and dy is not None:
            self.input_dims = {'X': dx, 'E': de, 'y': dy+1}  # + 1 due to time conditioning

            self.output_dims = {'X': dx, # output dim = # of features
                                'E': de,
                                'y': 0}
 
        else:
            log.info("Taking an example batch...")
            log.info(f"datamodule.train_dataloader(): {datamodule.train_dataloader()}")
            log.info(f"len(datamodule.train_dataloader()): {len(datamodule.train_dataloader())}")
            log.info(f"iter(datamodule.train_dataloader()): {iter(datamodule.train_dataloader())}")
            example_batch = next(iter(datamodule.train_dataloader()))
            log.info("Completed taking the example batch!")

            self.input_dims = {'X': example_batch.x.size(1), # n or dx?
                            'E': example_batch.edge_attr.size(1),
                            'y': example_batch.y.size(1) + 1,
                            'atom_charges': example_batch.atom_charges.size(1),
                            'atom_chiral': example_batch.atom_chiral.size(1),
                            'bond_dirs': example_batch.bond_dirs.size(1)}  # + 1 due to time conditioning
            self.output_dims = {'X': example_batch.x.size(1), # output dim = # of features
                                'E': example_batch.edge_attr.size(1),
                                'y': 0,
                                'atom_charges': example_batch.atom_charges.size(1),
                                'atom_chiral': example_batch.atom_chiral.size(1),
                                'bond_dirs': example_batch.bond_dirs.size(1)}
        log.info("Completed compute_input_output_dims! (in function)")

