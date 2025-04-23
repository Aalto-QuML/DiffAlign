
from diffalign.utils import graph
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader, HGTLoader
import numpy
import random
import logging
from diffalign.utils import data_utils
# A logger for this file
log = logging.getLogger(__name__)

# from memory_profiler import profile
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.type_as(batch_n_nodes)
        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)

        return log_p
    
class AbstractDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataloaders = None
        self.input_dims = None
        self.output_dims = None
        
    def prepare_data(self, datasets, shuffle=True, slices=None, seed=0) -> None:
        train_batch_size = self.cfg.train.batch_size
        test_batch_size = self.cfg.test.batch_size
        assert type(train_batch_size) == int and type(test_batch_size) == int
        batch_sizes = {"train": train_batch_size, "val": test_batch_size, "test": test_batch_size}
        num_workers = self.cfg.dataset.num_workers
        # shuffle_ = 'debug' not in self.cfg.general.name and shuffle
        # g = torch.Generator()
        # g.manual_seed(seed)
        # log.info(batch_sizes)
        # log.info(datasets)
        self.datasets = datasets
        print(f'datasets: {datasets}')
        if 'train' in datasets.keys():
            print(f'datasets[train]: {datasets["train"]}')
            print(f'datasets[train].slices: {datasets["train"].slices}')
        if not self.cfg.train.batch_by_size:
            self.dataloaders = {split: DataLoader(dataset, 
                                                batch_size=batch_sizes[split],
                                                num_workers=num_workers,
                                                shuffle=shuffle)
                                for split, dataset in datasets.items()}
        else:
            self.dataloaders = {}
            if datasets['train'].slices is not None: # dataset requires slicing/indexing, i.e. has multiple graphs => use VariableBatchSampler
                self.dataloaders['train'] = DataLoader(datasets['train'],
                                                    num_workers=num_workers,
                                                    batch_sampler=data_utils.VariableBatchSampler(datasets['train'], self.cfg.dataset.size_bins['train'], self.cfg.dataset.batchsize_bins['train'],
                                                                                                datasets['train'].slices['x']))
            else:
                self.dataloaders['train'] = DataLoader(datasets['train'],
                                                batch_size=batch_sizes['train'],
                                                num_workers=num_workers,
                                                shuffle=shuffle)
            self.dataloaders['val'] = DataLoader(datasets['val'],
                                                batch_size=batch_sizes['val'],
                                                num_workers=num_workers,
                                                shuffle=shuffle)
            self.dataloaders['test'] = DataLoader(datasets['test'],
                                                batch_size=batch_sizes['test'],
                                                num_workers=num_workers,
                                                shuffle=shuffle)
    
    def train_dataloader(self):
        return self.dataloaders["train"] 

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]
        
    def __getitem__(self, idx):
        return self.dataloaders['train'][idx]

    def node_counts(self, max_nodes_possible):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)

                all_pairs = 0
                for count in counts:
                    all_pairs += count * (count - 1)

                num_edges = data.edge_index.shape[1]
                num_non_edges = all_pairs - num_edges

                edge_types = data.edge_attr.sum(dim=0)
                assert num_non_edges >= 0
                d[0] += num_non_edges
                d[1:] += edge_types[1:]

        d = d / d.sum()
        return d
    
    def train_dataloader_bysize(self, dataset_class, root, split_size, stage='train', 
                                shuffle=True, seed=0, batchsize=None):
        
        batchsize = batchsize or self.cfg.train.batch_size
        shuffle_ = 'debug' not in self.cfg.general.name and shuffle
        g = torch.Generator()
        g.manual_seed(seed)

        dataset = dataset_class(stage=stage, root=root, split_size=split_size)
        dataloader = DataLoader(dataset, batch_size=batchsize, 
                                num_workers=self.cfg.dataset.num_workers, shuffle=shuffle_,
                                worker_init_fn=seed_worker, generator=g)
        
        return dataloader
    
class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = graph.to_dense(data=example_batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
        
        if extra_features is not None and domain_features is not None:
            ex_extra_feat = extra_features(example_data)
            self.input_dims['X'] += ex_extra_feat.X.size(-1)
            self.input_dims['E'] += ex_extra_feat.E.size(-1)
            self.input_dims['y'] += ex_extra_feat.y.size(-1)

            ex_extra_molecular_feat = domain_features(example_data)
            self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
            self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
            self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}

class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]

                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        
        return valencies