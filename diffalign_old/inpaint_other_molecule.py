import hydra
from omegaconf import DictConfig
import logging
import pathlib
import torch
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.data import Batch, Data

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

    cfg.train.seed = 15
    cfg.diffusion.diffusion_steps_eval = 100
    cfg.dataset.canonicalize_molecule=False # needed for identifying the correct atoms!
    cfg.test.return_smiles_with_atom_mapping=True
    n_samples = 100#cfg.test.n_samples_per_condition

    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = cfg.general.wandb.checkpoint_epochs[0]

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
    model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=device_count)
    
    # 5. set the model to evaluation mode
    model = model.to(device)
    model.eval()

    draw_with_atom_mapping = False

    ground_truth_reaction = "I[c:1]1[cH:2][n:3]([C:4]([O:5][C:6]([CH3:7])([CH3:8])[CH3:9])=[O:10])[c:11]2[c:12]1[cH:13][c:14]([Br:15])[cH:16][cH:17]2.OB(O)[c:18]1[cH:19][n:20][c:21]2[cH:22][cH:23][cH:24][cH:25][c:26]2[cH:27]1>>[c:1]1(-[c:18]2[cH:19][n:20][c:21]3[cH:22][cH:23][cH:24][cH:25][c:26]3[cH:27]2)[cH:2][n:3]([C:4]([O:5][C:6]([CH3:7])([CH3:8])[CH3:9])=[O:10])[c:11]2[c:12]1[cH:13][c:14]([Br:15])[cH:16][cH:17]2"
    original_ground_truth_reaction = ground_truth_reaction

    inpaint_node_idx = [[0,1,15,16,21,26]] * n_samples
    inpaint_atom_map_nums = [1,2,16,17,18,20]

    # Ibuprofen
    # step 1: friedel-crafts acylation
    # catalyst: HF
    # ground_truth_reaction = '[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1.[C:13][C:11](=[O:12])OC(=O)C>>[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11](=[O:12])[C:13]'
    # original_ground_truth_reaction = ground_truth_reaction
    # inpaint_node_idx = [[14,15,16]] * n_samples # NOTE: This requires that the atom map numbers are not shuffled!
    # inpaint_atom_map_nums = []

    # step 2: hydrogenation
    # catalyst: raney nickel
    # NOTE: could also try this reaction without mentioning H2 since it's not really modeled with our encoding
    # original_ground_truth_reaction = '[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11](=[O:12])[C:13].[H][H]>>[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11]([O:12])[C:13]'
    # ground_truth_reaction = '[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11](=[O:12])[C:13].[C][O]>>[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11]([O:12])[C:13]'
    # """On this reaction:
    # The reaction seems to convert a ketone (C=O) into an alcohol (COH) group on the complex aromatic compound. This type of reaction is chemically plausible and is known as a reduction reaction, where the ketone is reduced to an alcohol. Such reactions can be accomplished with various reducing agents.
    # However, methanol (CH3OH) by itself is not typically a reducing agent in organic chemistry. For a ketone to be reduced to an alcohol, more powerful reducing agents such as sodium borohydride (NaBH4) or lithium aluminum hydride (LiAlH4) are usually required. Methanol can sometimes be involved 
    # in reductions as a solvent or a part of the reaction medium, especially in transfer hydrogenation processes or in the presence of specific catalysts or reagents that can activate methanol for such a purpose.
    # In summary, while the conversion of a ketone to an alcohol is plausible, the involvement of methanol as the sole reagent in direct reduction seems unlikely without further context, such as the presence of a catalyst or an additional reducing agent.
    # """
    # # ground_truth_reaction = '[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11](=[O:12])[C:13].[O=CO]>>[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11]([O:12])[C:13]'
    # inpaint_node_idx = [[11,13,14]] * n_samples
    # inpaint_atom_map_nums = []

    # # step 3: carbonylationt
    # ground_truth_reaction = '[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11]([O:12])[C:13].[O:14][C:15]>>[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11]([C:13])[C:15](=[O:14])[O:12]'
    # # ground_truth_reaction = "[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1.[C:11][C:12](=[O:13])[O][C](=O)C>>[C:1][C:2]([C:3])[C:4][c:5]1[c:6][c:7][c:8]([c:9][c:10]1)[C:11](=[O:12])[C:13]"
    # original_ground_truth_reaction = ground_truth_reaction
    # inpaint_node_idx = [[7,10,11,12,13]] * n_samples
    # inpaint_atom_map_nums = [8,11,12,13,14]

    # aspirin step 2
    # ground_truth_reaction = "[c:1]1[c:2][c:3]([c:4]([c:5][c:6]1[C:7](=[O:8])[O:9])[O:10])[O:11].[C:12][C:13](=O)OC(=O)C>>[C:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]"
    # ground_truth_reaction = "Cl[CH2:1][c:2]1[n:3][c:4]([CH:5]=[CH:6][c:7]2[cH:8][cH:9][c:10]([S:11](=[O:12])[C:13]([F:14])([F:15])[F:16])[cH:17][cH:18]2)[o:19][cH:20]1.[OH:21][c:22]1[cH:23][cH:24][c:25]([CH2:26][O:27][CH2:28][CH2:29][n:30]2[cH:31][cH:32][n:33][n:34]2)[cH:35][cH:36]1>>[CH2:1]([c:2]1[n:3][c:4](/[CH:5]=[CH:6]/[c:7]2[cH:8][cH:9][c:10]([S:11](=[O:12])[C:13]([F:14])([F:15])[F:16])[cH:17][cH:18]2)[o:19][cH:20]1)[O:21][c:22]1[cH:23][cH:24][c:25]([CH2:26][O:27][CH2:28][CH2:29][n:30]2[cH:31][cH:32][n:33][n:34]2)[cH:35][cH:36]1"
    products = ground_truth_reaction.split(">>")[1].split(".")
    reactants = ground_truth_reaction.split(">>")[0].split(".")
    dataset = datamodule.datasets['train']
    g = turn_reactants_and_product_into_graph(dataset, reactants, products, dataset_infos.bond_decoder, data_idx=0)
    data = Batch.from_data_list([g])

    # Load a batch of data
    # data = next(iter(datamodule.val_dataloader()))

    dense_data = graph.to_dense(data).to_device(device)

    # Filter only data which has at least two reactants
    dense_data = dense_data.select_subset(dense_data.mol_assignment.max(-1).values > 2)
    dense_data = dense_data.select_by_batch_idx(0)
    dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)

    # reactant_to_keep = random.randint(1,2) # OLD WAY OF ENFORCING INPAINTING (THAT ALSO WORKED)
    # reactant_to_keep = 2
    # reactant_indices = (dense_data.mol_assignment[0] == reactant_to_keep).nonzero()
    # reactant_indices = [idx.item() for idx in reactant_indices]
    # rcts_mol = mol.mol_from_graph(node_list=dense_data.X[0,reactant_indices].argmax(-1), adjacency_matrix=dense_data.E[0,reactant_indices][:,reactant_indices].argmax(-1), atom_types=dataset_infos.atom_decoder, 
    #                               bond_types=dataset_infos.bond_decoder, plot_dummy_nodes=False)
    
    # smiles_of_mol_to_keep = Chem.MolToSmiles(rcts_mol) # OLD WAY OF ENFORCING INPAINTING
    # mol.get_cano_smiles_from_dense(X=dense_data.X[0,reactant_indices], E=dense_data.E[0,reactant_indices][:,reactant_indices], atom_types=dataset_infos.atom_decoder,
    #                                                    bond_types=dataset_infos.bond_decoder, return_dict=False)

    all_indices = list(range(dense_data.X.shape[-2]))

    # inpaint_node_idx = [reactant_indices] * n_samples
    
    inpaint_edge_idx = ["NO_ADDITIONAL_CONNECTIONS"] * n_samples
    # inpaint_edge_idx = [[(i,j) for i in reactant_indices for j in all_indices] + [(j,i) for i in reactant_indices for j in all_indices]] * n_samples
    regular_generated_samples = model.sample_one_batch(data=dense_data, inpaint_node_idx=None, 
                                               inpaint_edge_idx=None, get_true_rxns=False, get_chains=False, device=device)

    inpainted_samples = model.sample_one_batch(data=dense_data, inpaint_node_idx=inpaint_node_idx, 
                                               inpaint_edge_idx=inpaint_edge_idx, get_true_rxns=False, get_chains=False, device=device)
    print(inpainted_samples.atom_map_numbers)

    dense_data = dense_data.mask(collapse=True)
    regular_generated_samples = regular_generated_samples.mask(collapse=True)
    inpainted_samples = inpainted_samples.mask(collapse=True)
    
    scores, gen_rxns, regular_weighted_prob_sorted_rxns = model.score_one_batch(final_samples=regular_generated_samples, true_data=dense_data, bs=1, n_samples=n_samples, n=data.x.shape[0], device=device)
    regular_generated_rcts = regular_weighted_prob_sorted_rxns[list(regular_weighted_prob_sorted_rxns.keys())[0]][0]['rcts']
    regular_generated_prods = regular_weighted_prob_sorted_rxns[list(regular_weighted_prob_sorted_rxns.keys())[0]][0]['prod']
    regular_generated_smarts = ".".join(regular_generated_rcts) + ">>" + regular_generated_prods[0]

    scores, gen_rxns, weighted_prob_sorted_rxns = model.score_one_batch(final_samples=inpainted_samples, true_data=dense_data, bs=1, n_samples=n_samples, n=data.x.shape[0], device=device)
    inpainted_rcts = weighted_prob_sorted_rxns[list(weighted_prob_sorted_rxns.keys())[0]][0]['rcts']
    # inpainted_idx = inpainted_rcts.index(smiles_of_mol_to_keep) # OLD METHOD
    inpainted_prods = weighted_prob_sorted_rxns[list(weighted_prob_sorted_rxns.keys())[0]][0]['prod']
    inpainted_smarts = ".".join(inpainted_rcts) + ">>" + inpainted_prods[0]

    # Quick hack to get the original reaction SMARTS
    # dense_data = dense_data.select_by_batch_idx(0)
    # dense_data = dense_data.to_argmaxed()
    # dense_data = dense_data.to_cpu()
    # bond_types = [None, BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
    # orig_rxn_img, orig_rxn_smrts = mol.rxn_plot(dense_data, cfg.dataset.atom_types,bond_types, filename='test.png', return_smarts=True, plot_dummy_nodes=False)

    # inpainted_samples = inpainted_samples.to_argmaxed()
    # inpainted_samples = inpainted_samples.to_cpu()
    # inpainted_rxn_img, inpainted_rxn_smrts = mol.rxn_plot(inpainted_samples, cfg.dataset.atom_types,bond_types, filename='test.png', return_smarts=True, plot_dummy_nodes=False)

    # fig, ax = plt.subplots(2,1,figsize=(10,8))
    # ax[0].imshow(orig_rxn_img)
    # ax[1].imshow(inpainted_rxn_img)
    # plt.show()

    reactant_images, product_image = draw_molecules_from_reaction_with_highlight(original_ground_truth_reaction, highlight_atom_map_nums=[], draw_with_atom_mapping=draw_with_atom_mapping)
    orig_image = draw_reaction(reactant_images, product_image)

    reactant_images, product_image = draw_molecules_from_reaction_with_highlight(regular_generated_smarts, highlight_atom_map_nums=[], draw_with_atom_mapping=draw_with_atom_mapping)
    non_inpainted_image = draw_reaction(reactant_images, product_image)

    reactant_images, product_image = draw_molecules_from_reaction_with_highlight(inpainted_smarts, highlight_atom_map_nums=inpaint_atom_map_nums, draw_with_atom_mapping=draw_with_atom_mapping)
    inpainted_image = draw_reaction(reactant_images, product_image)

    fig, ax = plt.subplots(3,1,figsize=(10,10))
    ax[0].imshow(orig_image)
    ax[1].imshow(non_inpainted_image)
    ax[2].imshow(inpainted_image)
    ax[0].set_title("Original reaction")
    ax[1].set_title("Generated reaction without inpainting")
    ax[2].set_title("Generated reaction with one of the reactants fixed")
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()
    plt.savefig('inpaintings.png')
    print("hei")

    inpainted_smarts_all = []
    for i in range(len(weighted_prob_sorted_rxns[list(weighted_prob_sorted_rxns.keys())[0]])):
        inpainted_rcts = weighted_prob_sorted_rxns[list(weighted_prob_sorted_rxns.keys())[0]][i]['rcts']
        # inpainted_idx = inpainted_rcts.index(smiles_of_mol_to_keep)
        inpainted_prods = weighted_prob_sorted_rxns[list(weighted_prob_sorted_rxns.keys())[0]][i]['prod']
        inpainted_smarts = ".".join(inpainted_rcts) + ">>" + inpainted_prods[0]
        reactant_images, product_image = draw_molecules_from_reaction_with_highlight(inpainted_smarts, highlight_atom_map_nums=inpaint_atom_map_nums, draw_with_atom_mapping=draw_with_atom_mapping)
        inpainted_image = draw_reaction(reactant_images, product_image)
        plt.figure()
        plt.imshow(inpainted_image)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(f'inpaintings{i}.png')
        inpainted_smarts_all.append(inpainted_smarts + "\n")
    
    regular_gen_smarts_all = []
    for i in range(len(regular_weighted_prob_sorted_rxns[list(regular_weighted_prob_sorted_rxns.keys())[0]])):
        # scores, gen_rxns, regular_weighted_prob_sorted_rxns = model.score_one_batch(final_samples=regular_generated_samples, true_data=dense_data, bs=1, n_samples=n_samples, n=data.x.shape[0], device=device)
        regular_generated_rcts = regular_weighted_prob_sorted_rxns[list(regular_weighted_prob_sorted_rxns.keys())[0]][i]['rcts']
        regular_generated_prods = regular_weighted_prob_sorted_rxns[list(regular_weighted_prob_sorted_rxns.keys())[0]][i]['prod']
        regular_generated_smarts = ".".join(regular_generated_rcts) + ">>" + regular_generated_prods[0]
        reactant_images, product_image = draw_molecules_from_reaction_with_highlight(regular_generated_smarts, highlight_atom_map_nums=[], draw_with_atom_mapping=draw_with_atom_mapping)
        generated_image = draw_reaction(reactant_images, product_image)
        plt.figure()
        plt.imshow(generated_image)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(f'regulars{i}.png')
        regular_gen_smarts_all.append(regular_generated_smarts + "\n")

    with open("regular_gen_rxns.txt", "w") as f:
        f.writelines(regular_gen_smarts_all)
    
    with open("inpainted_rxns.txt", "w") as f:
        f.writelines(inpainted_smarts_all)

    pass

def draw_molecules_from_reaction_with_highlight(rxn_smrts, highlight_atom_map_nums, draw_with_atom_mapping=False):
    rxn_obj = Reactions.ReactionFromSmarts(rxn_smrts)

    # Function to highlight and draw a molecule
    def draw_mol_with_highlights(mol, highlight_atoms=[], highlight_color=(0.9,0.0,0)):
        try:
            Chem.SanitizeMol(mol) # just in case rdkit doesn't like the molecule for some reaosn
            smiles = Chem.MolToSmiles(mol) # put mol to standard format for plotting
            mol_ = Chem.MolFromSmiles(smiles)
            mol = mol_ if mol_ is not None else mol
            drawer = Draw.MolDraw2DCairo(int(30*np.sqrt(mol.GetNumAtoms())),int(30*np.sqrt(mol.GetNumAtoms())))  # Specify image size
            highlight_dict = {atom: highlight_color for atom in highlight_atoms}
            Draw.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_dict)
            return drawer.GetDrawingText()
        except:
            return None
            # drawer = Draw.MolDraw2DCairo(int(30*np.sqrt(mol.GetNumAtoms())),int(30*np.sqrt(mol.GetNumAtoms())))  # Specify image size
            # highlight_dict = {atom: highlight_color for atom in highlight_atoms}
            # Draw.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_dict)
            # return drawer.GetDrawingText()
        # drawer = Draw.MolDraw2DCairo(int(30*np.sqrt(mol.GetNumAtoms())),int(30*np.sqrt(mol.GetNumAtoms())))  # Specify image size
        # highlight_dict = {atom: highlight_color for atom in highlight_atoms}
        # Draw.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_dict)
        # return drawer.GetDrawingText()

    from PIL import Image
    from io import BytesIO

    reactant_images = []
    for i in range(len(rxn_obj.GetReactants())):
        # create the atommapnum into an index...?
        highlight_atoms = [idx for idx, a in zip(range(len(rxn_obj.GetReactants()[i].GetAtoms())),rxn_obj.GetReactants()[i].GetAtoms()) if a.GetAtomMapNum() in highlight_atom_map_nums]
        if not draw_with_atom_mapping:
            for atom in rxn_obj.GetReactants()[i].GetAtoms():
                atom.SetAtomMapNum(0)
        # highlight_atoms = list(range(rxn_obj.GetReactants()[highlighted_molecule_idx].GetNumAtoms()))
        img_data = draw_mol_with_highlights(rxn_obj.GetReactants()[i], highlight_atoms=highlight_atoms)
        if img_data is not None:
            img_data = Image.open(BytesIO(img_data))
            reactant_images.append(img_data)

    if not draw_with_atom_mapping:
        for atom in rxn_obj.GetProducts()[0].GetAtoms():
            atom.SetAtomMapNum(0)
    prod_mol = rxn_obj.GetProducts()[0]
    product_image = draw_mol_with_highlights(rxn_obj.GetProducts()[0], highlight_atoms=[])
    product_image = Image.open(BytesIO(product_image))
    return reactant_images, product_image

def draw_molecules_from_reaction_with_highlight_old(rxn_smrts, highlighted_molecule_idx):
    rxn_obj = Reactions.ReactionFromSmarts(rxn_smrts)

    # Function to highlight and draw a molecule
    def draw_mol_with_highlights(mol, highlight_atoms=[], highlight_color=(1,0,0)):
        try:
            Chem.SanitizeMol(mol) # just in case rdkit doesn't like the molecule for some reaosn
        except:
            pass
        drawer = Draw.MolDraw2DCairo(int(30*np.sqrt(mol.GetNumAtoms())),int(30*np.sqrt(mol.GetNumAtoms())))  # Specify image size
        highlight_dict = {atom: highlight_color for atom in highlight_atoms}
        Draw.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_dict)
        return drawer.GetDrawingText()

    from PIL import Image
    from io import BytesIO

    reactant_images = []
    for i in range(len(rxn_obj.GetReactants())):
        if i == highlighted_molecule_idx:
            highlight_atoms = list(range(rxn_obj.GetReactants()[highlighted_molecule_idx].GetNumAtoms()))
            img_data = draw_mol_with_highlights(rxn_obj.GetReactants()[i], highlight_atoms=highlight_atoms)
        else:
            img_data = draw_mol_with_highlights(rxn_obj.GetReactants()[i], highlight_atoms=[])
        img_data = Image.open(BytesIO(img_data))
        reactant_images.append(img_data)

    product_image = draw_mol_with_highlights(rxn_obj.GetProducts()[0], highlight_atoms=[])
    product_image = Image.open(BytesIO(product_image))
    return reactant_images, product_image

def draw_reaction(reactant_images, product_image):
    # Let's assume `images` is a list of your PIL images of the reactants and products in order
    # For demonstration, you would replace these with the images generated as described previously
    images = reactant_images + [product_image]

    # Calculate total width and max height
    total_width = sum(image.width for image in images) + 60 * (len(images) - 1)  # Adding space for arrows or text
    max_height = max(image.height for image in images)

    # Create a new image with enough space
    combined_image = Image.new('RGB', (total_width, max_height), 'white')

    # Paste each image onto the combined image
    x_offset = 0
    for image in images:
        y_offset = int((max_height - image.height)/2)
        combined_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + 60  # Space for arrows or text between molecules

    # Draw the reaction arrow or text
    draw = ImageDraw.Draw(combined_image)
    arrow_start = sum([img.width for img in images[:-1]]) + 60 * (len(images) - 2) + 20
    # arrow_end = 25
    # draw_text(combined_image, position=(arrow_start, max_height // 2), text="->", scale_factor=6)

    # font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", 28)
    font_path = os.path.expanduser("~/.local/share/fonts/roboto/Roboto-Thin.ttf")
    font = ImageFont.truetype(font_path, 28)

    draw.text(((arrow_start), max_height // 2 - 14), "->", fill=(0, 0, 0), font=font)

    plus_starts = [sum([img.width for img in images[:-1-i]]) + 60 * (len(images) - 2 - i) + 20 for i in range(1,len(images)-1)]
    for start in plus_starts:
        draw.text(((start), max_height // 2 - 14), "+", fill=(0, 0, 0), font=font)

    # Display or save the combined reaction image
    return combined_image  # Or save with combined_image.save('reaction_visualization.png')

def draw_text(image, position, text, font=None, fill=None, scale_factor=1):
    # Create a temporary image to get a "bitmap" of the text
    temp_image = Image.new('RGBA', (12,12), 'white')
    temp_draw = ImageDraw.Draw(temp_image)
    if not font:
        font = ImageFont.load_default()  # Load the default font
    temp_draw.text((0, 0), text, font=font, fill=(0,0,0))

    # Scale the text size up
    text_image = temp_image.resize([int(scale * s) for scale, s in zip((scale_factor, scale_factor), temp_image.size)])

    # Paste the scaled-up text image onto the original image
    image.paste(text_image, position)

def turn_reactants_and_product_into_graph(dataset, reactants, products, bond_types, data_idx):
    MAX_ATOMS_RXN = 1000
    DUMMY_RCT_NODE_TYPE = 'U'
    offset = 0 
    cannot_generate = False
    # mask: (n), with n = nb of nodes
    mask_product_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only reactant nodes = True
    mask_reactant_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool) # only product nodes = True
    mask_sn = torch.ones(MAX_ATOMS_RXN, dtype=torch.bool) # only sn = False
    mask_atom_mapping = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)
    mol_assignment = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)

    # preprocess: get total number of product nodes
    nb_product_nodes = sum([len(Chem.MolFromSmiles(p).GetAtoms()) for p in products])
    nb_rct_nodes = sum([len(Chem.MolFromSmiles(r).GetAtoms()) for r in reactants])
    
    # add dummy nodes: (nodes_in_product + max_added) - nodes_in_reactants
    nb_dummy_toadd = nb_product_nodes + dataset.max_nodes_more_than_product - nb_rct_nodes
    if nb_dummy_toadd<0 and dataset.stage=='train':
        # drop the rxns in the training set which we cannot generate
        return None
    if nb_dummy_toadd<0 and (dataset.stage=='test' or dataset.stage=='val'):
        # cut the rct nodes
        nb_dummy_toadd = 0
        cannot_generate = True

    for j, r in enumerate(reactants):
        # NOTE: no supernodes for reactants (treated as one block)
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = mol.mol_to_graph(mol=r, atom_types=dataset.atom_types, 
                                                                        bond_types=bond_types,
                                                                        with_explicit_h=dataset.with_explicit_h,
                                                                        with_formal_charge=dataset.with_formal_charge,
                                                                        offset=offset, get_atom_mapping=True,
                                                                        canonicalize_molecule=dataset.canonicalize_molecule)
        g_nodes_rct = torch.cat((g_nodes_rct, gi_nodes), dim=0) if j > 0 else gi_nodes # already a tensor
        g_edge_index_rct = torch.cat((g_edge_index_rct, gi_edge_index), dim=1) if j > 0 else gi_edge_index
        g_edge_attr_rct = torch.cat((g_edge_attr_rct, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr

        atom_mapped_idx = (atom_map!=0).nonzero()
        mask_atom_mapping[atom_mapped_idx+offset] = atom_map[atom_mapped_idx]
        mol_assignment[offset:offset+gi_nodes.shape[0]] = j+1
        offset += gi_nodes.shape[0] 

    ## remove rct_cut stuff with atom_map 0
    # not_atom_mapped = (mask_atom_mapping[:g_nodes_rct.shape[0]]==0).nonzero()
    # # randomly choose some of these nodes to be removed
    # chosen_idx = torch.randperm(not_atom_mapped.shape[0])[:rct_cut]
    # not_atom_mapped = not_atom_mapped[chosen_idx]
    # g_nodes_rct, g_edge_index_rct, g_edge_attr_rct, other_tensors = mol.filter_out_nodes(out_node_idx=not_atom_mapped, nodes=g_nodes_rct, 
    #                                                                                      edge_index=g_edge_index_rct, edge_attr=g_edge_attr_rct, 
    #                                                                                      atom_map=mask_atom_mapping[:g_nodes_rct.shape[0]])
    # mask_atom_mapping = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long) # reset atom_mapping info
    # mask_atom_mapping[:g_nodes_rct.shape[0]] = other_tensors['atom_map']
    # offset -= rct_cut

    g_nodes_dummy = torch.ones(nb_dummy_toadd, dtype=torch.long) * dataset.atom_types.index(DUMMY_RCT_NODE_TYPE)
    g_nodes_dummy = F.one_hot(g_nodes_dummy, num_classes=len(dataset.atom_types)).float()
    # edges: fully connected to every node in the rct side with edge type 'none'
    g_edges_idx_dummy = torch.zeros([2, 0], dtype=torch.long)
    g_edges_attr_dummy = torch.zeros([0, len(bond_types)], dtype=torch.long)
    mask_product_and_sn[:g_nodes_rct.shape[0]+g_nodes_dummy.shape[0]] = True
    mol_assignment[offset:offset+g_nodes_dummy.shape[0]] = 0
    # mask_atom_mapping[offset:offset+g_nodes_dummy.shape[0]] = MAX_ATOMS_RXN
    offset += g_nodes_dummy.shape[0]
    
    g_nodes = torch.cat([g_nodes_rct, g_nodes_dummy], dim=0)
    g_edge_index = torch.cat([g_edge_index_rct, g_edges_idx_dummy], dim=1)
    g_edge_attr = torch.cat([g_edge_attr_rct, g_edges_attr_dummy], dim=0)

    # Permute the rows here to make sure that the NN can only process topological information
    def permute_rows(nodes, mask_atom_mapping, mol_assignment, edge_index):
        # Permutes the graph specified by nodes, mask_atom_mapping, mol_assignment and edge_index
        # nodes: (n,d_x) node feature tensor
        # mask_atom_mapping (n,) tensor
        # mol_assignment: (n,) tensor
        # edge_index: (2,num_edges) tensor
        # does everything in-place
        rct_section_len = nodes.shape[0]
        perm = torch.randperm(rct_section_len)
        nodes[:] = nodes[perm]
        mask_atom_mapping[:rct_section_len] = mask_atom_mapping[:rct_section_len][perm]
        mol_assignment[:rct_section_len] = mol_assignment[:rct_section_len][perm]
        inv_perm = torch.zeros(rct_section_len, dtype=torch.long)
        inv_perm.scatter_(dim=0, index=perm, src=torch.arange(rct_section_len))
        edge_index[:] = inv_perm[edge_index]

    # if dataset.permute_mols:
    #     permute_rows(g_nodes, mask_atom_mapping, mol_assignment, g_edge_index)

    supernodes_prods = []
    for j, p in enumerate(products):
        # NOTE: still need supernode for product to distinguish it from reactants
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = mol.rxn_to_graph_supernode(mol=p, atom_types=dataset.atom_types, bond_types=bond_types,
                                                                                    with_explicit_h=dataset.with_explicit_h, supernode_nb=offset+1,
                                                                                    with_formal_charge=dataset.with_formal_charge,
                                                                                    add_supernode_edges=dataset.add_supernode_edges, get_atom_mapping=True,
                                                                                    canonicalize_molecule=dataset.canonicalize_molecule)
        
        g_nodes_prod = torch.cat((g_nodes_prod, gi_nodes), dim=0) if j > 0 else gi_nodes # already a tensor
        g_edge_index_prod = torch.cat((g_edge_index_prod, gi_edge_index), dim=1) if j > 0 else gi_edge_index
        g_edge_attr_prod = torch.cat((g_edge_attr_prod, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr
        atom_mapped_idx = (atom_map!=0).nonzero()
        mask_atom_mapping[atom_mapped_idx+offset] = atom_map[atom_mapped_idx]
        mask_reactant_and_sn[offset:gi_nodes.shape[0]+offset] = True
        mol_assignment[offset] = 0 # supernode does not belong to any molecule
        suno_idx = offset # there should only be one supernode and one loop through the products
        mol_assignment[offset+1:offset+1+gi_nodes.shape[0]] = len(reactants)+j+1 # TODO: Is there one too many assigned as a product atom here?
        mask_sn[offset] = False
        mask_reactant_and_sn[offset] = False
        # supernode is always in the first position
        si = 0 # gi_edge_index[0][0].item()
        supernodes_prods.append(si)
        offset += gi_nodes.shape[0]

    # Keep the supernode intact here, others are permuted
    def permute_rows_product(g_nodes_prod, mask_atom_mapping, g_edge_index_prod):
        prod_indices = (suno_idx, suno_idx + g_nodes_prod.shape[0])
        perm = torch.cat([torch.tensor([0], dtype=torch.long), 1 + torch.randperm(g_nodes_prod.shape[0]-1)], 0)
        inv_perm = torch.zeros(len(perm), dtype=torch.long)
        inv_perm.scatter_(dim=0, index=perm, src=torch.arange(len(perm)))
        g_nodes_prod[:] = g_nodes_prod[perm]
        
        # sn_and_prod_selection = (prod_selection | suno_idx == torch.arange(len(prod_selection)))
        mask_atom_mapping[prod_indices[0]:prod_indices[1]] = mask_atom_mapping[prod_indices[0]:prod_indices[1]][perm]
        
        # The following because g_edge_index_prod are counted with their offset in the final graph
        offset_padded_perm = torch.cat([torch.zeros(suno_idx, dtype=torch.long), suno_idx + perm]) # for debugging
        offset_padded_inv_perm = torch.cat([torch.zeros(suno_idx, dtype=torch.long), suno_idx + inv_perm])
        
        g_edge_index_prod[:] = offset_padded_inv_perm[g_edge_index_prod]

    # if dataset.permute_mols:
    #     permute_rows_product(g_nodes_prod, mask_atom_mapping, g_edge_index_prod)

    # concatenate all types of nodes and edges
    g_nodes = torch.cat([g_nodes, g_nodes_prod], dim=0)
    g_edge_index = torch.cat([g_edge_index, g_edge_index_prod], dim=1)
    g_edge_attr = torch.cat([g_edge_attr, g_edge_attr_prod], dim=0)

    y = torch.zeros((1, 0), dtype=torch.float)
    
    # trim masks => one element per node in the rxn graph
    mask_product_and_sn = mask_product_and_sn[:g_nodes.shape[0]] # only reactant nodes = True
    mask_reactant_and_sn = mask_reactant_and_sn[:g_nodes.shape[0]]
    mask_sn = mask_sn[:g_nodes.shape[0]]
    mask_atom_mapping = mask_atom_mapping[:g_nodes.shape[0]]
    mol_assignment = mol_assignment[:g_nodes.shape[0]]
    
    # mask_atom_mapping = mol.sanity_check_and_fix_atom_mapping(mask_atom_mapping, g_nodes)
    
    assert mask_atom_mapping.shape[0]==g_nodes.shape[0] and mask_sn.shape[0]==g_nodes.shape[0] and \
        mask_reactant_and_sn.shape[0]==g_nodes.shape[0] and mask_product_and_sn.shape[0]==g_nodes.shape[0] and \
        mol_assignment.shape[0]==g_nodes.shape[0]

    # erase atom mapping absolute information for good. 
    # perm = torch.arange(mask_atom_mapping.max().item()+1)[1:]
    # perm = perm[torch.randperm(len(perm))]
    # perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])
    # mask_atom_mapping = perm[mask_atom_mapping]

    graph = Data(x=g_nodes, edge_index=g_edge_index, 
                edge_attr=g_edge_attr, y=y, idx=data_idx,
                mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn, 
                mask_product_and_sn=mask_product_and_sn, mask_atom_mapping=mask_atom_mapping,
                mol_assignment=mol_assignment, cannot_generate=cannot_generate)

    return graph

if __name__ == '__main__':
    main()