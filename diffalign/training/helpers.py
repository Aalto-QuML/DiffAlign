from torch.utils.data import DataLoader

from diffalign.data.dataset import ReactionDataset

def load_data(cfg):
    '''
        Loads the data from the dataset and returns the train and val loaders.
    '''
    train_dataset = ReactionDataset(cfg, stage='train')
    val_dataset = ReactionDataset(cfg, stage='val')
    # TODO: change this with smarter loaders which group by length
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    return train_loader, val_loader

def create_model_and_optimizer(cfg):
    '''
        Creates the model and optimizer.
    '''
    pass

def train_batch(model, optimizer, scheduler, batch, loss_fn):
    pass

def validate(model, val_loader, loss_fn):
    pass

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    pass