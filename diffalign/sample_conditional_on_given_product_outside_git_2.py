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

    ground_truth_reaction = "Cl[CH2:1][c:2]1[n:3][c:4]([CH:5]=[CH:6][c:7]2[cH:8][cH:9][c:10]([S:11](=[O:12])[C:13]([F:14])([F:15])[F:16])[cH:17][cH:18]2)[o:19][cH:20]1.[OH:21][c:22]1[cH:23][cH:24][c:25]([CH2:26][O:27][CH2:28][CH2:29][n:30]2[cH:31][cH:32][n:33][n:34]2)[cH:35][cH:36]1>>[CH2:1]([c:2]1[n:3][c:4](/[CH:5]=[CH:6]/[c:7]2[cH:8][cH:9][c:10]([S:11](=[O:12])[C:13]([F:14])([F:15])[F:16])[cH:17][cH:18]2)[o:19][cH:20]1)[O:21][c:22]1[cH:23][cH:24][c:25]([CH2:26][O:27][CH2:28][CH2:29][n:30]2[cH:31][cH:32][n:33][n:34]2)[cH:35][cH:36]1"
    products = ground_truth_reaction.split(">>")[1].split(".")
    reactants = ground_truth_reaction.split(">>")[0].split(".")

    dataset = datamodule.datasets['train']
    g = turn_reactants_and_product_into_graph(dataset, reactants, products, dataset_infos.bond_decoder, data_idx=0)
    # graph Data() to DataBatch():
    
    data = Batch.from_data_list([g])
    dense_data = graph.to_dense(data).to_device(device)
    n_samples = 20
    dense_data = graph.duplicate_data(dense_data, n_samples=n_samples, get_discrete_data=False)

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

    # inpaint_edge_idx = [[(i,j) for i in reactant_indices for j in all_indices] + [(j,i) for i in reactant_indices for j in all_indices]] * n_samples
    samples = model.sample_one_batch(data=dense_data, inpaint_node_idx=None, 
                                               inpaint_edge_idx=None, get_true_rxns=False, get_chains=False, device=device)

    # Turn the generated samples to SMILES
    gen_rxn_smiles = mol.get_cano_smiles_from_dense(X=samples.X.argmax(-1), E=samples.E.argmax(-1), mol_assignment=samples.mol_assignment, atom_types=dataset_infos.atom_decoder,
                                                   bond_types=dataset_infos.bond_decoder, return_dict=False)
    # gen_rxn_smiles[1] = 'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C' + '>>' + products[0]

    original_rxn_smiles = mol.get_cano_smiles_from_dense(X=dense_data.X.argmax(-1), E=dense_data.E.argmax(-1), mol_assignment=samples.mol_assignment, atom_types=dataset_infos.atom_decoder,
                                                   bond_types=dataset_infos.bond_decoder, return_dict=False)
    
    print(f'original_rxn_smiles {original_rxn_smiles}\n')
    print(f'gen_rxn_smiles {gen_rxn_smiles}\n')
    # Check how many of the SMILES match
    count = 0
    for i, rxn in enumerate(gen_rxn_smiles):
        if rxn == original_rxn_smiles[0]:
            count += 1
            print(f'Found a match at index {i}\n')
    print("matching count: ", count)
    pass
    
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
    perm = torch.arange(mask_atom_mapping.max().item()+1)[1:]
    perm = perm[torch.randperm(len(perm))]
    perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])
    mask_atom_mapping = perm[mask_atom_mapping]

    graph = Data(x=g_nodes, edge_index=g_edge_index, 
                edge_attr=g_edge_attr, y=y, idx=data_idx,
                mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn, 
                mask_product_and_sn=mask_product_and_sn, mask_atom_mapping=mask_atom_mapping,
                mol_assignment=mol_assignment, cannot_generate=cannot_generate)

    return graph

if __name__ == '__main__':
    main()