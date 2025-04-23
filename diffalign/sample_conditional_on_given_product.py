import hydra
from omegaconf import DictConfig
import logging
import pathlib
import torch
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.data import Data
from torch_geometric.data import Batch

import torch.nn.functional as F
from diffalign.utils import graph, mol, setup
from diffalign.datasets import supernode_dataset
from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import rdChemReactions as Reactions

import warnings
import copy
import wandb
import matplotlib.pyplot as plt
# A logger for this file
log = logging.getLogger(__name__)

# warnings.filterwarnings("ignore", category=PossibleUserWarning)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    print(f'cfg.dataset.atom_types {cfg.dataset.atom_types}\n')
    print(f'started\n')
    orig_cfg = copy.deepcopy(cfg)
    run = None

    entity = cfg.general.wandb.entity
    project = cfg.general.wandb.project
    # Extract only the command-line overrides
    # cfg should contain cfg.general.wandb.run_id here
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')
    run_config = setup.load_wandb_config(cfg)
    cfg = setup.merge_configs(default_cfg=cfg, new_cfg=run_config, cli_overrides=cli_overrides)

    cfg.general.wandb.entity = entity # some kind of check for old runs, I think
    cfg.general.wandb.project = project

    cfg.train.seed = 1
    cfg.diffusion.diffusion_steps_eval = 10

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = cfg.general.wandb.checkpoint_epochs[0]

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], 
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False)

    # ground_truth_reaction = "Cl[CH2:1][c:2]1[n:3][c:4]([CH:5]=[CH:6][c:7]2[cH:8][cH:9][c:10]([S:11](=[O:12])[C:13]([F:14])([F:15])[F:16])[cH:17][cH:18]2)[o:19][cH:20]1.[OH:21][c:22]1[cH:23][cH:24][c:25]([CH2:26][O:27][CH2:28][CH2:29][n:30]2[cH:31][cH:32][n:33][n:34]2)[cH:35][cH:36]1>>[CH2:1]([c:2]1[n:3][c:4](/[CH:5]=[CH:6]/[c:7]2[cH:8][cH:9][c:10]([S:11](=[O:12])[C:13]([F:14])([F:15])[F:16])[cH:17][cH:18]2)[o:19][cH:20]1)[O:21][c:22]1[cH:23][cH:24][c:25]([CH2:26][O:27][CH2:28][CH2:29][n:30]2[cH:31][cH:32][n:33][n:34]2)[cH:35][cH:36]1"
    ground_truth_reaction = "c1ccc(C[O:1][C:2]([c:3]2[cH:4][cH:5][cH:6][c:7]([NH:8][C:9]([NH:10][CH2:11][C:12]([N:13]3[CH:14]([C:15]([O:16][C:17]([CH3:18])([CH3:19])[CH3:20])=[O:21])[CH2:22][CH:23]([S:24]([CH3:25])(=[O:26])=[O:27])[CH:28]3[c:29]3[cH:30][cH:31][cH:32][cH:33][c:34]3[F:35])=[O:36])=[O:37])[cH:38]2)=[O:39])cc1>>[OH:1][C:2]([c:3]1[cH:4][cH:5][cH:6][c:7]([NH:8][C:9]([NH:10][CH2:11][C:12]([N:13]2[CH:14]([C:15]([O:16][C:17]([CH3:18])([CH3:19])[CH3:20])=[O:21])[CH2:22][CH:23]([S:24]([CH3:25])(=[O:26])=[O:27])[CH:28]2[c:29]2[cH:30][cH:31][cH:32][cH:33][c:34]2[F:35])=[O:36])=[O:37])[cH:38]1)=[O:39]"
    products = ground_truth_reaction.split(">>")[1].split(".")
    reactants = ground_truth_reaction.split(">>")[0].split(".")

    g = datamodule.datasets['train'].turn_reactants_and_product_smiles_into_graphs(reactants, products, data_idx=0)
    # graph Data() to DataBatch():
    
    g = datamodule.datasets['train'].transform(g) # add the pos enc
    data = Batch.from_data_list([g])
    dense_data = graph.to_dense(data).to_device(device)
    n_samples = 1
    dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)

    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': False},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=1)
    
    # 4. load the weights to the model
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=1)
    
    # 5. set the model to evaluation mode
    model = model.to(device)
    model.eval()

    # inpaint_edge_idx = [[(i,j) for i in reactant_indices for j in all_indices] + [(j,i) for i in reactant_indices for j in all_indices]] * n_samples
    samples = model.sample_one_batch(data=dense_data, inpaint_node_idx=None, 
                                               inpaint_edge_idx=None, get_true_rxns=False, get_chains=False, device=device)

    # Turn the generated samples to SMILES

    gen_rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(samples.mask(samples.node_mask, collapse=True), cfg)
    original_rxn_smiles = mol.get_cano_smiles_from_dense_with_stereochem(dense_data.mask(dense_data.node_mask, collapse=True), cfg)

    # gen_rxn_smiles = mol.get_cano_smiles_from_dense(X=samples.X.argmax(-1), E=samples.E.argmax(-1), mol_assignment=samples.mol_assignment, atom_types=dataset_infos.atom_decoder,
    #                                                bond_types=dataset_infos.bond_decoder, return_dict=False)

    # original_rxn_smiles = mol.get_cano_smiles_from_dense(X=dense_data.X.argmax(-1), E=dense_data.E.argmax(-1), mol_assignment=samples.mol_assignment, atom_types=dataset_infos.atom_decoder,
    #                                                bond_types=dataset_infos.bond_decoder, return_dict=False)
    
    print(f'original_rxn_smiles {original_rxn_smiles}\n')
    print(f'gen_rxn_smiles {gen_rxn_smiles}\n')
    # Check how many of the SMILES match
    count = 0
    for i, rxn in enumerate(gen_rxn_smiles):
        if rxn == original_rxn_smiles[0]:
            count += 1
            print(f'Found a match at index {i}\n')
    print("matching count: ", count)

    rxn_img = mol.rxn_plot(samples, cfg)
    plt.imshow(rxn_img)
    plt.show()
    
if __name__ == '__main__':
    main()