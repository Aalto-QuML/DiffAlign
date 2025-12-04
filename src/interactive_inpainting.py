from flask import Flask, render_template, request, jsonify

import hydra
from omegaconf import DictConfig
import logging
import pathlib
import torch
import os

import torch.nn.functional as F
from src.utils import graph, mol, setup
from src.datasets import supernode_dataset
from src.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn

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
        
    # need a better way to initialize this stuff
    log.info('Getting dataset infos...')
    datamodule, dataset_infos = setup.get_dataset(HYDRA_CFG, data_class, shuffle=HYDRA_CFG.dataset.shuffle, 
                                                  return_datamodule=True, recompute_info=False)

    log.info('Getting model...')
    MODEL, optimizer, lr_scheduler, scaler, start_epoch = setup.get_model_and_train_objects(HYDRA_CFG, model_class=model_class, 
                                                                                model_kwargs={'dataset_infos': dataset_infos, 
                                                                                            'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                            'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized},
                                                                                parent_path=parent_path)
    MODEL = MODEL.to(device)

def get_inpainted_samples(product_smi, generated_reactants, bond_indices, atom_indices):
    #print(f'flask-app: generated_reactants {generated_reactants}\n')
    
    final_samples = graph.json_to_graph(generated_reactants, x_classes=len(supernode_dataset.atom_types), 
                                        e_classes=len(supernode_dataset.bond_types))
    #print(f'flask-app: final_samples.X.shape {final_samples.X.shape}\n')

    n_samples = final_samples.X.shape[0]
    #print(f'flask-app: final_samples.E.shape {final_samples.E.shape}\n')
    n_samples = 3
    data = graph.get_graph_data_from_product_smi(product_smi, HYDRA_CFG)
    dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)
    if len(atom_indices)>0 and len(bond_indices)>0:
        inpaint_node_idx = list(set([a for bond in  atom_indices for a in bond]))*n_samples
        inpaint_edge_idx = [atom_indices]*n_samples
    else:
        inpaint_node_idx, inpaint_edge_idx = None, None
    
    inpainted_samples = MODEL.sample_one_batch(data=final_samples, inpaint_node_idx=inpaint_node_idx, 
                                               inpaint_edge_idx=inpaint_edge_idx, get_true_rxns=False, get_chains=False)
        
    dense_data = dense_data.mask(collapse=True)
    inpainted_samples = inpainted_samples.mask(collapse=True)
    scores, gen_rxns, _ = MODEL.score_one_batch(final_samples=inpainted_samples, true_data=dense_data, bs=1, n_samples=n_samples, n=final_samples.X.shape[1])
    
    rct_smiles, prod_smiles = mol.get_cano_list_smiles(X=inpainted_samples.X, E=inpainted_samples.E, 
                                                       atom_types=supernode_dataset.atom_types, bond_types=supernode_dataset.bond_types, 
                                                       plot_dummy_nodes=False)
    
    # then write a different function for turning that to a list of smiles 
    # and returning it to front end
    #reactants = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    reactants = rct_smiles[1]
    
    print(f'====== inpainted reactants {reactants}\n')

    return reactants, final_samples

def get_samples(product_smi, plot_dummy_nodes=False):
    # need to turn the product smiles to a graph object
    n_samples = 3
    data = graph.get_graph_data_from_product_smi(product_smi, HYDRA_CFG)
    
    dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)
    
    # need to get graph object of sampled reactants
    final_samples = MODEL.sample_one_batch(data=dense_data, inpaint_node_idx=None, inpaint_edge_idx=None, 
                                           get_true_rxns=False, get_chains=False)
    
    dense_data = dense_data.mask(collapse=True)
    final_samples = final_samples.mask(collapse=True)
    scores, gen_rxns, _ = MODEL.score_one_batch(final_samples=final_samples, true_data=dense_data, bs=1, n_samples=n_samples, n=data.x.shape[0])
    
    rct_smiles, prod_smiles = mol.get_cano_list_smiles(X=final_samples.X, E=final_samples.E, 
                                                       atom_types=supernode_dataset.atom_types, bond_types=supernode_dataset.bond_types, 
                                                       plot_dummy_nodes=plot_dummy_nodes)
    
    # then write a different function for turning that to a list of smiles 
    # and returning it to front end
    #reactants = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    reactants = rct_smiles[1]
    
    #reactants = ['COC1=CC(OC)=C(CCS(=O)(=O)CC2=CC=C(OC)C(O)=C2)C(OC)=C1']
    
    final_samples_ = final_samples.get_new_object(X=F.one_hot(final_samples.X, num_classes=len(supernode_dataset.atom_types)).to(torch.float32), 
                                                  E=F.one_hot(final_samples.E, num_classes=len(supernode_dataset.bond_types)).to(torch.float32))
    
    return reactants, final_samples_

@app.route('/getReactants', methods=['POST'])
def get_reactants():
    print(f'generating reactants.')
    # Extract bond/atom indices from request
    data = request.json
    product_smi = data['productSmi']

    # TODO: Process the data with your model
    sampled_reactants_smi, final_samples = get_samples(product_smi=product_smi)
    
    print(f'done with reactants.')

    return jsonify(sampled_reactants_smi=sampled_reactants_smi, product_smi=product_smi, final_samples=final_samples.serialize())

@app.route('/getInpaintedReactants', methods=['POST'])
def get_inpainted_reactants():
    # Extract bond/atom indices from request
    data = request.json
    product_smi = data['productSmi']
    bond_indices = data['bondIndices']
    atom_indices = data['atomIndices']
    generated_reactants = data['generated_reactants']

    # TODO: Process the data with your model
    reactants_smi, final_samples = get_inpainted_samples(product_smi, generated_reactants, bond_indices, atom_indices)

    # Return new reactants
    return jsonify(reactants_smi=reactants_smi, product_smi=product_smi, final_samples=final_samples.serialize())

@app.route('/')
def index():
    init_app()
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
