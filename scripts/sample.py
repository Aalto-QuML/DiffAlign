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
    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = model.sample_n_conditions(dataloader=val_loader, inpaint_node_idx=None, 
                                                                  inpaint_edge_idx=None, epoch_num=760, device_to_use=device)
    print(f'here')


if __name__ == '__main__':
    sample()