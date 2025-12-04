import hydra
import torch

from diffalign.helpers import PROJECT_ROOT, device
from diffalign.training.helpers import load_data, create_model_and_optimizer, \
                                        train_batch, validate, save_checkpoint
from diffalign.data import mol, graph

@hydra.main(config_path='../configs', config_name='config.yaml')
def sample(cfg):
    # load the data
    train_loader, val_loader = load_data(cfg)
    # data_ = next(iter(val_loader))
    # dense_data_dup = graph.to_dense(data_, cfg=cfg).to_device(device)
    # dense_data_dup = dense_data_dup.mask(collapse=True)

    # true_rxns = mol.get_cano_smiles_from_dense_with_stereochem(dense_data=dense_data_dup, cfg=cfg, return_dict=False)
    # create the model
    model, optimizer, scheduler = create_model_and_optimizer(cfg)

    # generate and save samples
    print(f'Generating and saving samples...')
    all_gen_rxn_smiles, all_true_rxn_smiles, all_gen_rxn_pyg, all_true_rxn_pyg = model.sample_n_conditions(
        dataloader=val_loader, 
        device_to_use=device, 
        inpaint_node_idx=None, 
        inpaint_edge_idx=None,
        epoch_num=760)
    print(f'here')


if __name__ == '__main__':
    sample()