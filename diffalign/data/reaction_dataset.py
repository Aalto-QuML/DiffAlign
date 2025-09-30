'''
Create a graph reaction dataset to train on. 
Input: reaction data in a smiles format.
Output: batched pyg's graph objects.
'''
import requests
import zipfile
from pathlib import Path
import os
import torch
from torch_geometric.data import InMemoryDataset

from diffalign.helpers import PROJECT_ROOT
from diffalign.data.helpers import get_reactions_from_dataset, reaction_smiles_to_graph

class ReactionDataset(InMemoryDataset):
    def __init__(self, cfg, stage, transform=None, pre_transform=None, pre_filter=None):
        self.root = os.path.join(PROJECT_ROOT, 'dataset', cfg.dataset.data_dir)
        self.cfg = cfg
        self.dataset_information = self.get_dataset_information()
        self.stage = stage
        self.file_idx = ['train', 'test', 'val'].index(self.stage)
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[self.file_idx])

    def get_dataset_information(self):
        '''
            Return information to process the dataset.
        '''
        # upload information
        atom_types = [a.strip() for a in open(os.path.join(self.root, 'processed',
                                            self.cfg.dataset.atom_types_path),
                                            'r', encoding='utf-8').readlines()]
        atom_charges = [a.strip() for a in open(os.path.join(self.root, 'processed',
                                            self.cfg.dataset.atom_charges_path),
                                            'r', encoding='utf-8').readlines()]
        atom_chiral_tags = [a.strip() for a in open(os.path.join(self.root, 'processed',
                                            self.cfg.dataset.atom_chiral_tags_path),
                                            'r', encoding='utf-8').readlines()]
        atom_types_charged = [a.strip() for a in open(os.path.join(self.root, 'processed',
                                            self.cfg.dataset.atom_types_charged_path),
                                            'r', encoding='utf-8').readlines()]     
        bond_types = [a.strip() for a in open(os.path.join(self.root, 'processed',
                                            self.cfg.dataset.bond_types_path),
                                            'r', encoding='utf-8').readlines()]     
        bond_dirs = [a.strip() for a in open(os.path.join(self.root, 'processed',
                                            self.cfg.dataset.bond_dirs_path),
                                            'r', encoding='utf-8').readlines()]
        return {
            'atom_types': atom_types,
            'atom_charges': atom_charges,
            'atom_chiral_tags': atom_chiral_tags,
            'atom_types_charged': atom_types_charged,
            'bond_types': bond_types,
            'bond_dirs': bond_dirs
        }
    
    @property
    def raw_file_names(self):
        '''
            Return the file names of the raw data.
        '''
        return ['train.csv', 'test.csv', 'val.csv']

    @property
    def processed_file_names(self):
        '''
            Return the file names of the processed data.
        '''
        return ['train.pt', 'test.pt', 'val.pt']
    
    def download(self):
        '''
            Download the dataset from gln's dropbox url and extract it to the raw directory.
            Then rename the files from raw_train to train, raw_test to test, and raw_val to val.
        '''
        # Convert to direct download URL
        url = "https://www.dropbox.com/scl/fo/swuggv6qf8ombw914yxh8/AEwUgTxowsq2vrnv0D2xRNg/schneider50k?dl=1&rlkey=1ed5tqauj7udn5n2olvw1looi"

        os.makedirs(self.raw_dir, exist_ok=True)
        download_path = Path(self.raw_dir) / 'dataset.zip'
        
        try:
            print(f'Downloading dataset from {url} to {download_path}')
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Download with progress
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)

            # rename files from raw_train to train, same for test and val
            os.rename(os.path.join(self.raw_dir, 'raw_train.csv'), os.path.join(self.raw_dir, 'train.csv'))
            os.rename(os.path.join(self.raw_dir, 'raw_test.csv'), os.path.join(self.raw_dir, 'test.csv'))
            os.rename(os.path.join(self.raw_dir, 'raw_val.csv'), os.path.join(self.raw_dir, 'val.csv'))
        
            download_path.unlink()  # Remove zip file
            print(f'Dataset downloaded and extracted to {self.raw_dir}')
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            self._manual_download_instructions()
    
    def process(self):
        '''
            Process the dataset and save it to the processed directory.
        '''
        # TODO: replace with function from retrofilter
        reactions = get_reactions_from_dataset(self.cfg, self.raw_paths[self.file_idx])
        # TODO: add code to turn a reaction smiles into a pyg graph
        all_graphs = []
        cnt_dropped_reactions = 0
        for idx, reaction in enumerate(reactions):
            graph = reaction_smiles_to_graph(self.cfg, stage=self.stage,
                                            dataset_information=self.dataset_information,
                                            reaction_smiles_idx=idx,
                                            reaction_smiles=reaction)
            if graph is None:
                cnt_dropped_reactions += 1
            else:
                all_graphs.append(graph)
        print(f'===== Dropped {cnt_dropped_reactions}/{len(reactions)} reactions.')
        print(f'===== Saving {len(all_graphs)} graphs to {self.processed_paths[self.file_idx]}.')
        torch.save(self.collate(all_graphs), self.processed_paths[self.file_idx])

    # def len(self):
    #     pass