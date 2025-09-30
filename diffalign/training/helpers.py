from pickletools import optimize
from torch.utils.data import DataLoader
import torch

from diffalign.model.diffusion import DiscreteDenoisingDiffusionRxn
from diffalign.data.reaction_dataset import ReactionDataset

def load_data(cfg):
    '''
        Loads the data from the dataset and returns the train and val loaders.
    '''
    train_dataset = ReactionDataset(cfg, stage='train')
    val_dataset = ReactionDataset(cfg, stage='val')
    # TODO: change this with smarter loaders which group by length
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    #return val_loader
    return train_loader, val_loader

def create_model_and_optimizer(cfg):
    '''
        Creates the model and optimizer.
    '''
    diffusion = DiscreteDenoisingDiffusionRxn(cfg, dataset_infos=None)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)

    return diffusion, optimizer, scheduler

def train_batch(model, optimizer, scheduler, batch, loss_fn):
    pass

def validate(model, val_loader, loss_fn):
    pass

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    pass