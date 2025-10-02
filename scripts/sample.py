import hydra
import torch

from diffalign.helpers import PROJECT_ROOT, device
from diffalign.training.helpers import load_data, create_model_and_optimizer, \
                                        train_batch, validate, save_checkpoint

@hydra.main(config_path='../configs', config_name='config.yaml')
def sample(cfg):
    # load the data
    train_loader, val_loader = load_data(cfg)

    # create the model
    model, optimizer, scheduler = create_model_and_optimizer(cfg)

    # generate and save samples
    print(f'Generating and saving samples...')

if __name__ == '__main__':
    sample()