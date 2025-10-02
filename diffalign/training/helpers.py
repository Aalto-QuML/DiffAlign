from pickletools import optimize
from torch.utils.data import DataLoader
import torch

from diffalign.model.diffusion import DiscreteDenoisingDiffusionRxn
from diffalign.data.reaction_dataset import ReactionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    if cfg.evaluate.checkpoint_path is not None:
        checkpoint = torch.load(cfg.evaluate.checkpoint_path, map_location=device)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
    
    if cfg.train.resume and cfg.train.resume_checkpoint_path is not None:
        checkpoint = torch.load(cfg.train.resume_checkpoint_path, map_location=device)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return diffusion, optimizer, scheduler

def train_batch(model, optimizer, scheduler, batch, loss_fn):
    pass

def validate(model, val_loader, loss_fn):
    pass

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    pass