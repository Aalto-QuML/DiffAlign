import hydra
import torch

from diffalign.helpers import PROJECT_ROOT, device
from diffalign.training.helpers import load_data, create_model_and_optimizer, \
                                        train_batch, validate, save_checkpoint

@hydra.main(config_path='../configs', config_name='config.yaml')
def train(cfg):
    # load the data
    train_loader, val_loader = load_data(cfg)

    # create the model
    model, optimizer, scheduler = create_model_and_optimizer(cfg)

    loss_fn = torch.nn.CrossEntropyLoss()

    # training loop
    for epoch in range(cfg.training.epochs):
        for batch in train_loader:
            loss = train_batch(model, optimizer, scheduler, batch, loss_fn)

        if epoch % cfg.training.val_interval == 0:
            val_loss = validate(model, val_loader, loss_fn)

        if epoch % cfg.training.print_every == 0:
            print(f'Epoch {epoch} loss: {loss.item()}, val_loss: {val_loss.item()}')
        
        if epoch % cfg.training.save_every == 0:
            save_checkpoint(model, optimizer, epoch, cfg.training.checkpoint_dir)

if __name__ == '__main__':
    train()