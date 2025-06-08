from flask import Flask, render_template, request, jsonify

import hydra
from omegaconf import DictConfig
import logging
import pathlib
import torch
import os
import random
import numpy as np

import torch.nn.functional as F
from diffalign_old.utils import graph, mol, setup
from diffalign_old.datasets import supernode_dataset
from diffalign_old.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions

log = logging.getLogger(__name__)

app = Flask(__name__)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
HYDRA_CFG = None 
    
@hydra.main(version_base='1.1', config_path='../configs', config_name='default')
def init_app(cfg: DictConfig):
    global MODEL
    global HYDRA_CFG
    HYDRA_CFG = cfg
    data_class = supernode_dataset
    model_class = DiscreteDenoisingDiffusionRxn
        
    # Extract only the command-line overrides
    cli_overrides = setup.capture_cli_overrides()
    log.info(f'cli_overrides {cli_overrides}\n')

    if cfg.general.wandb.mode=='online': 
        # run, cfg = setup.setup_wandb(cfg, cli_overrides=cli_overrides, job_type='ranking') # This creates a new wandb run or resumes a run given its id
        run, cfg = setup.setup_wandb(cfg, job_type='ranking')

    entity = cfg.general.wandb.entity
    project = cfg.general.wandb.project

    if cfg.general.wandb.load_run_config: 
        run_config = setup.load_wandb_config(cfg)
        cfg = setup.merge_configs(default_cfg=cfg, new_cfg=run_config, cli_overrides=cli_overrides)
    
    cfg.general.wandb.entity = entity
    cfg.general.wandb.project = project

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    log.info(f"Random seed: {cfg.train.seed}")
    log.info(f"Shuffling on: {cfg.dataset.shuffle}")
    
    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device_count: {device_count}, device: {device}\n')
    
    #epoch_num = cfg.general.wandb.checkpoint_epochs[0]
    epoch_num = 380
    sampling_steps = cfg.diffusion.diffusion_steps_eval

    datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'], 
                                                  shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False)
    
    model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                         model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                       'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                       'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                       'use_data_parallel': device_count>1},
                                                                                         parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                         load_weights_bool=False, device=device, device_count=device_count)

    # 4. load the weights to the model
    savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
    MODEL, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=device_count)
    MODEL = MODEL.to(device)

def get_inpainted_samples(product_smi, generated_reactants, bond_indices, atom_indices):
    #print(f'flask-app: generated_reactants {generated_reactants}\n')
    
    final_samples = graph.json_to_graph(generated_reactants, x_classes=len(HYDRA_CFG.dataset.atom_types), 
                                        e_classes=len(supernode_dataset.bond_types))
    
    final_samples = final_samples.to_device(device)
    #print(f'flask-app: final_samples.X.shape {final_samples.X.shape}\n')

    n_samples = final_samples.X.shape[0]
    #print(f'flask-app: final_samples.E.shape {final_samples.E.shape}\n')
    n_samples = HYDRA_CFG.test.n_samples_per_condition
    data = graph.get_graph_data_from_product_smi(product_smi, HYDRA_CFG)
    dense_data = graph.to_dense(data).to_device(device)
    dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)

    if len(atom_indices)>0 and len(bond_indices)>0:
        inpaint_node_idx = [list(set([int(a) for bond in  atom_indices for a in bond]))]*n_samples
        inpaint_edge_idx = [[(int(s), int(e)) for s,e in atom_indices]]*n_samples
    else:
        inpaint_node_idx, inpaint_edge_idx = None, None
    
    inpainted_samples = MODEL.sample_one_batch(data=final_samples, inpaint_node_idx=inpaint_node_idx, 
                                               inpaint_edge_idx=inpaint_edge_idx, get_true_rxns=False, get_chains=False, device=device)
        
    dense_data = dense_data.mask(collapse=True)
    inpainted_samples = inpainted_samples.mask(collapse=True)
    scores, gen_rxns, weighted_prob_sorted_rxns = MODEL.score_one_batch(final_samples=inpainted_samples, true_data=dense_data, bs=1, n_samples=n_samples, 
                                                n=final_samples.X.shape[1], device=device)

    reactants = weighted_prob_sorted_rxns[list(weighted_prob_sorted_rxns.keys())[0]][0]['rcts']
    
    rct_smiles, prod_smiles = mol.get_cano_list_smiles(X=inpainted_samples.X, E=inpainted_samples.E, 
                                                       atom_types=HYDRA_CFG.dataset.atom_types, bond_types=supernode_dataset.bond_types, 
                                                       plot_dummy_nodes=False)
    
    
    # then write a different function for turning that to a list of smiles 
    # and returning it to front end
    #reactants = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    # reactants = rct_smiles[0]
    
    print(f'====== inpainted reactants {reactants}\n')

    return reactants, final_samples

def get_samples(product_smi, plot_dummy_nodes=False):
    # need to turn the product smiles to a graph object
    n_samples = HYDRA_CFG.test.n_samples_per_condition
    
    data = graph.get_graph_data_from_product_smi(product_smi, HYDRA_CFG)
    dense_data = graph.to_dense(data).to_device(device)
    prod_mol = mol.mol_from_graph(node_list=dense_data.X[0,...].argmax(-1), adjacency_matrix=dense_data.E[0,...].argmax(-1), atom_types=HYDRA_CFG.dataset.atom_types, 
                                  bond_types=graph.bond_types, plot_dummy_nodes=plot_dummy_nodes)                     
    prod_smiles_processed = Chem.MolToSmiles(prod_mol, kekuleSmiles=True, isomericSmiles=True)
    
    dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)
    
    # need to get graph object of sampled reactants
    final_samples = MODEL.sample_one_batch(data=dense_data, inpaint_node_idx=None, inpaint_edge_idx=None, 
                                           get_true_rxns=False, get_chains=False, device=device)
    
    print(f'final_samples.X.shape {final_samples.X.argmax(-1)[0,...]}\n')
    
    dense_data = dense_data.mask(collapse=True)
    final_samples = final_samples.mask(collapse=True)
    scores, gen_rxns, weighted_prob_sorted_rxns = MODEL.score_one_batch(final_samples=final_samples, true_data=dense_data, bs=1, n_samples=n_samples, n=data.x.shape[0], device=device)
    
    # then write a different function for turning that to a list of smiles 
    # and returning it to front end
    #reactants = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    reactants = weighted_prob_sorted_rxns[prod_smiles_processed][0]['rcts']
    
    #reactants = ['COC1=CC(OC)=C(CCS(=O)(=O)CC2=CC=C(OC)C(O)=C2)C(OC)=C1']
    
    final_samples_ = final_samples.get_new_object(X=F.one_hot(final_samples.X, num_classes=len(HYDRA_CFG.dataset.atom_types)).to(torch.float32), 
                                                  E=F.one_hot(final_samples.E, num_classes=len(graph.bond_types)).to(torch.float32))
    
    return reactants, final_samples_

@app.route('/getReactants', methods=['POST'])
def get_reactants():
    print(f'generating reactants.')
    # Extract bond/atom indices from request
    data = request.json
    product_smi = data['productSmi']

    # TODO: Process the data with your model
    sampled_reactants_smi, sampled_reactants_dense = get_samples(product_smi=product_smi)
    
    print(f'done with reactants.')

    return jsonify(sampled_reactants_smi=sampled_reactants_smi, product_smi=product_smi, sampled_reactants_dense=sampled_reactants_dense.serialize())

@app.route('/getInpaintedReactants', methods=['POST'])
def get_inpainted_reactants():
    # Extract bond/atom indices from request
    data = request.json
    product_smi = data['productSmi']
    bond_indices = data['bondIndices']
    atom_indices = data['atomIndices']
    generated_reactants = data['generated_reactants']

    # TODO: Process the data with your model
    reactants_smi, reactants_dense = get_inpainted_samples(product_smi, generated_reactants, bond_indices, atom_indices)

    # Return new reactants
    return jsonify(reactants_smi=reactants_smi, product_smi=product_smi, sampled_reactants_dense=reactants_dense.serialize())

def remove_atom_mapping_and_kekulize(rxn_smiles):
    # Parse the reaction SMILES
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles)

    # Function to remove mapping and kekulize a molecule
    def process_molecule(mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    # Process reactants and products
    reactants = [process_molecule(mol) for mol in rxn.GetReactants()]
    products = [process_molecule(mol) for mol in rxn.GetProducts()]

    # Create a new reaction from the processed molecules
    new_rxn = rdChemReactions.ChemicalReaction()
    for mol in reactants:
        new_rxn.AddReactantTemplate(mol)
    for mol in products:
        new_rxn.AddProductTemplate(mol)

    return rdChemReactions.ReactionToSmarts(new_rxn)

@app.route('/plot_molecule', methods=['POST'])
def plot_molecule():
    data = request.json
    smiles = data['smiles']
    # mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # print(f'smiles {smiles}\n')
    #smiles = '[CH3:1][O:2][c:3]1[cH:4][c:5]([O:6][CH3:7])[c:8](/[CH:9]=[CH:10]/[S:11](=[O:12])(=[O:13])[CH2:14][c:15]2[cH:16][cH:17][c:18]([O:19][CH3:20])[c:21]([OH:22])[cH:23]2)[c:24]([O:25][CH3:26])[cH:27]1>>[CH3:1][O:2][c:3]1[cH:4][c:5]([O:6][CH3:7])[c:8]([CH2:9][CH2:10][S:11](=[O:12])(=[O:13])[CH2:14][c:15]2[cH:16][cH:17][c:18]([O:19][CH3:20])[c:21]([OH:22])[cH:23]2)[c:24]([O:25][CH3:26])[cH:27]1'
    #smiles = 'Cl[CH2:1][C:2]([O:3][CH2:4][CH3:5])=[O:6].[CH3:7][O:8][c:9]1[cH:10][cH:11][c:12]2[c:13]([cH:14]1)[CH2:15][CH2:16][NH:17][C:18]2=[O:19]>>[CH2:1]([C:2]([O:3][CH2:4][CH3:5])=[O:6])[N:17]1[CH2:16][CH2:15][c:13]2[c:12]([cH:11][cH:10][c:9]([O:8][CH3:7])[cH:14]2)[C:18]1=[O:19]'
    rxns_smarts = remove_atom_mapping_and_kekulize(smiles)
    rxn = rdChemReactions.ReactionFromSmarts(rxns_smarts)
    drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)  # Adjust size as needed
    # drawer.drawOptions().addAtomIndices = True   # Optional: to add atom indices
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with open("molecule.svg", "w") as file:
        file.write(svg)
    return jsonify({'svg': svg})
    
@app.route('/')
def index():
    init_app()
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
