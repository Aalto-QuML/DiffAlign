import re
import pandas as pd
from rdkit import Chem
import torch
import os
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType as BT

from rdkit import RDLogger
# Temporarily disable
RDLogger.DisableLog('rdApp.*')

from diffalign.data.chem_helpers import create_canonicalized_mol, get_atom_symbol, \
                                        get_cip_ranking, switch_between_bond_cw_ccw_label_and_cip_based_label
from diffalign.data.rdkit_helpers import get_rdkit_chiral_tags, get_rdkit_bond_types, get_rdkit_bond_dirs
from diffalign.data import graph

def add_supernodes(cfg, data):
    """In case the data does not have a supernode as the first node of each molecule, we add it here. 
    Need to also update the mol_assignment, atom_map_numbers, and pos_encoding fields. 
    Also to atom_charges, atom_chiral, and edge_index. 
    Should reproduce the behavior of the way that the supernodes used to be added"""
    mol_assignment = data.mol_assignment.clone()
    nodes = data.x.clone()
    atom_map_numbers = data.atom_map_numbers.clone()
    edge_index = data.edge_index.clone()
    pos_encoding = data.pos_encoding.clone() if 'pos_encoding' in data else None
    atom_charges = data.atom_charges.clone()
    atom_chiral = data.atom_chiral.clone()
    # Create a supernode one-hot encoding
    suno = F.one_hot(torch.tensor([cfg.dataset.atom_types.index('SuNo')], dtype=torch.long), num_classes=len(cfg.dataset.atom_types))
    # Only add supernode to the beginning of the products.
    supernode_addition_indices = (mol_assignment == mol_assignment.max()).nonzero()[0]
    # add the supernodes
    for index in reversed(supernode_addition_indices):
        nodes = torch.cat([nodes[:index], suno.clone(), nodes[index:]], dim=0)
        mol_assignment = torch.cat([mol_assignment[:index], torch.tensor([0], dtype=torch.int), mol_assignment[index:]], dim=0) # supernode does not belong to any molecule
        # Update the atom_map_numbers
        atom_map_numbers = torch.cat([atom_map_numbers[:index], torch.tensor([0], dtype=torch.int), atom_map_numbers[index:]], dim=0)
        # Update the pos_encoding, if it exists
        if 'pos_encoding' in data:
            pos_encoding = torch.cat([pos_encoding[:index], torch.zeros(1, pos_encoding.shape[1]), pos_encoding[index:]], dim=0)
        atom_charges = torch.cat([atom_charges[:index], torch.zeros(1, atom_charges.shape[1]), atom_charges[index:]], dim=0)
        atom_chiral = torch.cat([atom_chiral[:index], torch.zeros(1, atom_chiral.shape[1]), atom_chiral[index:]], dim=0)
        # Update the edge_index
        edge_index = edge_index + (edge_index >= index).int()
    # Update the data object
    data.x = nodes
    data.mol_assignment = mol_assignment
    data.atom_map_numbers = atom_map_numbers
    data.edge_index = edge_index
    if 'pos_encoding' in data:
        data.pos_encoding = pos_encoding
    data.atom_charges = atom_charges
    data.atom_chiral = atom_chiral
    return data


def dense_from_pyg_file_data(cfg, reaction):
    data_list = graph.pyg_to_full_precision_expanded(reaction, atom_types=cfg.dataset.atom_types, bond_types=cfg.dataset.bond_types).to_data_list()
    data_batch = Batch.from_data_list(data_list)
    dense_data = graph.to_dense(data_batch)
    return dense_data

def dense_from_pyg_file_data_for_reaction_list(cfg, reactions):
    # Format of reactions:
    # 2-length list [gen_rxns, true_rxns], where gen_rxns and true_rxns are lists of DataBatches that 
    # contain n_samples_per_condition samples each.
    samples_graphs, true_rxn_graphs = [], []
    for i in range(len(reactions['gen'])):
        #print(f"reactions['gen'][i] {reactions['gen'][i]}\n")
        true_rxn_graphs.extend(graph.pyg_to_full_precision_expanded(reactions['true'][i], cfg=cfg).to_data_list())
        samples_graphs.extend(graph.pyg_to_full_precision_expanded(reactions['gen'][i], cfg=cfg).to_data_list())

    if (cfg.neuralnet.pos_encoding_type != 'none' and cfg.neuralnet.pos_encoding_type != 'no_pos_enc'): # recalculate the positional encodings here
        pos_encoding_size = data_utils.get_pos_enc_size(cfg)
        for i in range(len(samples_graphs)):
            samples_graphs[i] = data_utils.positional_encoding_adding_transform(samples_graphs[i], cfg.neuralnet.pos_encoding_type, pos_encoding_size)

    true_pyg_data = Batch.from_data_list(true_rxn_graphs)
    sample_pyg_data = Batch.from_data_list(samples_graphs)
    true_graph_data = graph.to_dense(true_pyg_data)
    sample_graph_data = graph.to_dense(sample_pyg_data)
    
    return true_graph_data, sample_graph_data

    # for i, (true_rxn, samples) in enumerate(reactions):
    #     true_rxn_graphs.extend(true_rxn.to_data_list())
    #     samples_graphs.extend(samples.to_data_list())
    # true_pyg_data = Batch.from_data_list(true_rxn_graphs)
    # sample_pyg_data = Batch.from_data_list(samples_graphs)
    # true_graph_data = graph.to_dense(true_pyg_data)
    # sample_graph_data = graph.to_dense(sample_pyg_data)
    # return true_graph_data, sample_graph_data


def rxn_vs_sample_plot(true_rxns, sampled_rxns, cfg, chain_name='default', rxn_offset_nb=0):
    '''
       Visualize the true rxn vs a rxn being sampled to compare the reactants more easily.
       
       rxn_offset_nb: where to start the count for naming the rxn plot (file).
    '''

    assert true_rxns.X.shape[0]==sampled_rxns[0][1].X.shape[0], 'You need to give as many true_rxns as there are chains.'+\
            f' Currently there are {true_rxns.X.shape[0]} true rxns and {sampled_rxns[0][1].X.shape[0]} chains.'
            
    # initialize the params of the video writer
    nb_of_chains = true_rxns.X.shape[0] # number of graph chains to plot
    imgio_kargs = {'fps': 1, 'quality': 10, 'macro_block_size': None, 'codec': 'h264', 'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    # create a frame for each time step t
    writers = []
    for t, samples_t in sampled_rxns:
        for c in range(nb_of_chains):
            chain_pic_name = f'{chain_name}_t{t}_rxn{c+rxn_offset_nb}.png'
            # get image of the true rxn, to be added to each plot at time t 
            true_rxn = true_rxns.subset_by_idx(start_idx=c, end_idx=c+1)
            # true_rxn = graph.PlaceHolder(X=true_rxns.X[c,...].unsqueeze(0), E=true_rxns.E[c,...].unsqueeze(0), node_mask=true_rxns.node_mask[c,...].unsqueeze(0), y=true_rxns.y,
            #                              mol_assignment=true_rxns.mol_assignment[c,...].unsqueeze(0))
            true_img = mol.rxn_plot(rxn=true_rxn, cfg=cfg)
            # true_img = true_img.resize((600, 300))

            # get image of the sample rxn at time t
            # one_sample_t = graph.PlaceHolder(X=samples_t.X[c,...].unsqueeze(0), E=samples_t.E[c,...].unsqueeze(0), y=samples_t.y, node_mask=samples_t.node_mask[c,...].unsqueeze(0),
            #                                  mol_assignment=samples_t.mol_assignment[c,...].unsqueeze(0))
            one_sample_t = samples_t.subset_by_idx(start_idx=c, end_idx=c+1)
            sampled_img = mol.rxn_plot(rxn=one_sample_t, cfg=cfg)
            # sampled_img = sampled_img.resize((600, 300))
            
            # plot sampled and true rxn in the same fig
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 7)) # x, y
            axes[0].axis('off')
            axes[1].axis('off')
            
            axes[0].set_title('sampled')
            axes[1].set_title('true')
            
            axes[0].imshow(sampled_img)
            axes[1].imshow(true_img)
            fig.suptitle(chain_pic_name.split('.png')[0])
            plt.savefig(chain_pic_name, dpi=199)
            plt.close()
            
            if c >= len(writers):
                writer = imageio.get_writer(f'{chain_name}_rxn{c+rxn_offset_nb}.mp4', **imgio_kargs)
                writers.append(writer)

            img = imageio.v2.imread(os.path.join(os.getcwd(), chain_pic_name), format='PNG')
            img = np.array(img) 
            writers[c].append_data(img)
            
            # repeat the last frame 10 times for a nicer video
            if t==0:
                for _ in range(10):
                    writers[c].append_data(img)
                
    # close previous writers
    for c in range(len(writers)):
        writers[c].close()
                
    return [os.path.join(os.getcwd(), f'{chain_name}_rxn{c+rxn_offset_nb}.mp4') for  c in range(nb_of_chains)]
  

def rxn_diagnostic_chains(chains, atom_types, bond_types, chain_name='default'):
    '''
        Visualize chains of a process as an mp4 video.

        chains: list of PlaceHolder objects representing a batch of graphs at each time step.
        len(chains)==nb_time_steps.

        Returns:
            (str) list of paths of mp4 videos of chains.
    '''
    nb_of_chains = chains[0][1].X.shape[0] # number of graph chains to plot
    imgio_kargs = {'fps': 1, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
                   'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    # init a writer per chain
    writers = {}  
    sampled_mols = {}
    for t, samples_t in chains:
        for c in range(nb_of_chains):
            suno_idx = atom_types.index('SuNo') # offset because index 0 is for no node   
    
            suno_indices = (samples_t.X[c,...].argmax(-1)==suno_idx).nonzero(as_tuple=True)[0].cpu() 
            mols_atoms = torch.tensor_split(samples_t.X[c,...], suno_indices, dim=0)[1:-1] # ignore first set (SuNo) and last set (product)
            mols_edges = torch.tensor_split(samples_t.E[c,...], suno_indices, dim=0)[1:-1]
            node_masks = torch.tensor_split(samples_t.node_mask[c,...], suno_indices, dim=-1)[1:-1]
            
            for m, mol_atoms in enumerate(mols_atoms): # for each mol in sample
                chain_pic_name = f'{chain_name}_sample_t{t}_chain{c}_mol{m}.png'

                if c not in writers.keys():
                    writer = imageio.get_writer(f'{chain_name}_chain{c}_mol{m}.mp4', **imgio_kargs)
                    writers[c] = {m: writer}
                else:
                    if m not in writers[c].keys():
                        writer = imageio.get_writer(f'{chain_name}_chain{c}_mol{m}.mp4', **imgio_kargs)
                        writers[c][m] = writer
                    else:
                        writer = writers[c][m]

                mol_edges_to_all = mols_edges[m] 
                mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=1)[1:] # ignore first because empty SuNo set
                mol_edges = mol_edges_t[m]
                mol_edges = mol_edges[1:,:][:,1:] # (n-1, n-1)
                mol_atoms = mol_atoms[1:] # (n-1)
                node_mask = node_masks[m][1:]
                
                one_sample = PlaceHolder(X=mol_atoms, E=mol_edges, node_mask=node_mask, y=torch.tensor([t], device=device).unsqueeze(-1))
                
                fig, mol = mol_diagnostic_plots(sample=one_sample, atom_types=atom_types, bond_types=bond_types, 
                                                name=chain_pic_name, show=False, return_mol=True)
                
                if c not in sampled_mols.keys():
                    sampled_mols[c] = {m: [mol]}
                else:
                    if m not in sampled_mols[c].keys():
                        sampled_mols[c][m] = [mol]
                    else:
                        sampled_mols[c][m].append(mol) 
                    
                img = imageio.v2.imread(os.path.join(os.getcwd(), chain_pic_name))
                writers[c][m].append_data(img)
            # repeat the last frame 10 times for a nicer video
            if t==0:
                for _ in range(10):
                    writers[c][m].append_data(img)

    # close previous writers
    for c in writers.keys():
        for m in writers[c].keys():
            writers[c][m].close()
    
    # for c in sampled_mols.keys():
    #     for m in sampled_mols[c].keys():
    #         img = Draw.MolToImage(sampled_mols[c][m][-1], size=(300, 300))
    #         plt.imshow(img)
    #         plt.title(f'chain{c}_mol{m}')
    #         plt.axis('off')
    #         plt.savefig(f'chain{c}_mol{m}.png')
        
    # plot the last rxn for each chain as a separate image
    # return [('', os.path.join(os.getcwd(), chain), smi) for chain, smi in zip(img_paths, sampled_smis)]
    return [(os.path.join(os.getcwd(), f'{chain_name}_chain{c}_mol{m}.mp4'), os.path.join(os.getcwd(), f'chain{c}_mol{m}.png'), Chem.MolToSmiles(sampled_mols[c][m][-1])) for c in writers.keys() for m in range(len(writers[c]))]


def get_formal_charge_as_str(formal_charge):
    '''
        Get the formal charge as a string.
    '''
    if formal_charge > 0:
        formal_charge_str = '+' + str(formal_charge)
    elif formal_charge < 0:
        formal_charge_str = '-' + str(abs(formal_charge))
    else:
        formal_charge_str = '0'
    return formal_charge_str

def get_reactant_and_product_from_reaction_smiles(reaction_smiles, return_as_str=False):
    '''
        Get the reactants and products from a reaction smiles.
    '''
    if '>>' in reaction_smiles:
        reactants = reaction_smiles.split('>>')[0]
        products = reaction_smiles.split('>>')[1]
    elif '>' in reaction_smiles:
        # add reagents to reactants
        reactants = reaction_smiles.split('>')[0]+'.'+reaction_smiles.split('>')[1]
        products = reaction_smiles.split('>')[-1]
    else:
        raise ValueError(f'Invalid reaction smiles: {reaction_smiles}')
    if return_as_str:
        return reactants, products
    else:
        return reactants.split(' ')[0].split('.'), products.split(' ')[0].split('.')
    
def get_reaction_smiles(reaction):
    '''
        Get the reaction smiles from the reaction.

        Not sure if checking for mixing '>' and '>>' is necessary. The idea is correct smiles should 
        only have one or the other. Also not that right now we treat reagents as reactants.
    '''
    if '>>' in reaction:
        reactants = reaction.split('>>')[0]
        products = reaction.split('>>')[-1]
        if '>' in reactants or '>' in products:
            raise ValueError(f'Reaction {reaction} contains >')
        return f'{reactants}>>{products}'
    elif '>' in reaction:
        print(f'Reaction {reaction} contains >')
        reactants = reaction.split('>')[0]
        reagents = reaction.split('>')[1]
        products = reaction.split('>')[2]
        if '>>' in reactants or '>>' in reagents or '>>' in products:
            raise ValueError(f'Reaction {reaction} contains >>')
        return f'{reactants}.{reagents}>>{products}'
    else:
        raise ValueError(f'Reaction {reaction} does not contain >> or >')

# TODO: replace with function from retrofilter
def get_reactions_from_dataset(cfg, dataset_path):
    '''
        Get the reactions from the dataset. 
    '''
    if cfg.dataset.data_dir == 'uspto_50k' or cfg.dataset.data_dir == 'uspto_50k_debug':
        df = pd.read_csv(dataset_path)
        reactions = df['reactants>reagents>production'].apply(get_reaction_smiles).tolist()
        # TODO: can add sanity checks here to fix atom-mapping etc.
        return reactions
    else:
        raise ValueError(f'Dataset {cfg.dataset.data_dir} not supported')
    
def is_reaction_smiles_valid(reaction_smiles):
    '''
        Check if a reaction smiles is valid.
    '''
    if '>>' in reaction_smiles and '>' not in reaction_smiles.split('>>')[0] and '>' not in reaction_smiles.split('>>')[1]:
        return True
    return False


def mol_to_graph_with_stereochem_(
    cfg,
    mol_input,
    rdkit_atom_types,
    rdkit_atom_charges,
    rdkit_atom_chiral_tags,
    rdkit_bond_types,
    rdkit_bond_dirs,
    mol_idx,
    with_formal_charge_in_atom_symbols=False,
):
    """
    m: rdkit molecule object
    returns: atom_symbols, atom_charges, atom_chiral, edge_index, bond_type, bond_dir, atom_map
    """
    assert (
        not with_formal_charge_in_atom_symbols
        or len([at for at in rdkit_atom_types if "+" in at or "-" in at]) > 0
    ), f"with_formal_charge_in_atom_symbols={with_formal_charge_in_atom_symbols} but no formal charges in rdkit_atom_types={rdkit_atom_types}.\n"
    assert (
        with_formal_charge_in_atom_symbols
        or len([at for at in rdkit_atom_types if "+" in at or "-" in at]) == 0
    ), f"with_formal_charge_in_atom_symbols={with_formal_charge_in_atom_symbols} but formal charges in rdkit_atom_types={rdkit_atom_types}.\n"

    if cfg.dataset.canonicalize_molecule:
        mol_input = create_canonicalized_mol(mol_input)

    # NOTE: not kekulizing anymore, can't remember why
    Chem.Kekulize(
        mol_input, clearAromaticFlags=True
    )  # we need this because the graph will be represented in kekulized format

    # get atom symbols
    atom_symbols = F.one_hot(
        torch.tensor(
            [
                rdkit_atom_types.index(
                    get_atom_symbol(atom, with_formal_charge_in_atom_symbols)
                )
                for atom in mol_input.GetAtoms()
            ]
        ),
        num_classes=len(rdkit_atom_types),
    ).float()

    # get atom map number
    atom_map = torch.tensor([atom.GetAtomMapNum() for atom in mol_input.GetAtoms()])

    # get atom charges
    atom_charges = F.one_hot(
        torch.tensor(
            [
                rdkit_atom_charges.index(get_formal_charge_as_str(atom.GetFormalCharge()))
                for atom in mol_input.GetAtoms()
            ]
        ),
        num_classes=len(rdkit_atom_charges),
    ).float()

    # get atom chirality
    atom_chiral = F.one_hot(
        torch.tensor(
            [
                rdkit_atom_chiral_tags.index(atom.GetChiralTag())
                for atom in mol_input.GetAtoms()
            ]
        ),
        num_classes=len(rdkit_atom_chiral_tags),
    ).float()

    # get bonds' end indices
    # bond_end_indices = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in m.GetBonds()]
    edge_index = (
        torch.tensor(
            [
                [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                + [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
                for bond in mol_input.GetBonds()
            ]
        )
        .flatten()
        .reshape(-1, 2)
        .t()
        .contiguous()
        .long()
    )

    # get bond types
    bond_types_list = [bond.GetBondType() for bond in mol_input.GetBonds()]

    if len(bond_types_list) > 0:
        bond_types = F.one_hot(
            torch.tensor(
                [
                    rdkit_bond_types.index(bond.GetBondType())
                    for bond in mol_input.GetBonds()
                ]
            ).repeat_interleave(2),
            num_classes=len(rdkit_bond_types),
        ).float()
    else:
        bond_types = torch.zeros((0, len(rdkit_bond_types)))

    bond_dirs_list = [
        rdkit_bond_dirs.index(bond.GetBondDir()) for bond in mol_input.GetBonds()
    ]
    if len(bond_dirs_list) > 0:
        bond_dirs = F.one_hot(
            torch.tensor(
                [
                    rdkit_bond_dirs.index(bond.GetBondDir())
                    for bond in mol_input.GetBonds()
                ]
            ).repeat_interleave(2),
            num_classes=len(rdkit_bond_dirs),
        ).float()
    else:
        bond_dirs = torch.zeros((0, len(rdkit_bond_dirs)))

    # get atom chirality
    cip_ranking = get_cip_ranking(
        mol_input
    )  # Note: this could be slightly slow for large molecules
    chiral_labels = []
    for atom in mol_input.GetAtoms():
        chiral_labels.append(
            switch_between_bond_cw_ccw_label_and_cip_based_label(
                atom, atom.GetChiralTag(), cip_ranking
            )
        )
    atom_chiral = F.one_hot(
        torch.tensor([rdkit_atom_chiral_tags.index(label) for label in chiral_labels]),
        num_classes=len(rdkit_atom_chiral_tags),
    ).float()

    return {'x': atom_symbols,
            'edge_index': edge_index,
            'edge_attr': bond_types,
            'mol_assignment': torch.ones(atom_symbols.shape[0], dtype=torch.long) * mol_idx,
            'atom_map_numbers': atom_map,
            'atom_charges': atom_charges,
            'atom_chiral': atom_chiral,
            'bond_dirs': bond_dirs}

def smiles_to_graph_with_stereochem(smi, cfg, dataset_information, mol_idx):
    '''
        Return data making a pyg graph for a molecule given its smiles string.
    '''
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"Could not get rdkit mol object from mol_input={smi}\n"
    graph_data = mol_to_graph_with_stereochem_(
        cfg,
        m,
        rdkit_atom_types=dataset_information['atom_types'],
        rdkit_atom_charges=dataset_information['atom_charges'],
        rdkit_atom_chiral_tags=get_rdkit_chiral_tags(dataset_information['atom_chiral_tags']),
        rdkit_bond_types=get_rdkit_bond_types(dataset_information['bond_types']),
        rdkit_bond_dirs=get_rdkit_bond_dirs(dataset_information['bond_dirs']),
        with_formal_charge_in_atom_symbols=cfg.dataset.with_formal_charge_in_atom_symbols,
        mol_idx=mol_idx,
    )
    return graph_data

def split_reaction_graph_to_reactants_and_products(graph_data):
    '''
    Split a reaction graph into reactants and products.
    '''
    # Find indices for each molecule type
    reactant_mask = graph_data['mol_assignment'] == 0
    #dummy_mask = graph_data['mol_assignment'] == 1
    product_mask = graph_data['mol_assignment'] == 2
    
    # Get the last index of each type
    end_of_reactant_idx = torch.nonzero(reactant_mask)[-1].item()
    start_of_product_idx = torch.nonzero(product_mask)[0].item()
    #end_of_dummy_idx = torch.nonzero(dummy_mask)[-1].item()
    
    # Find edge splits based on node indices
    end_of_reactant_edge_idx = torch.where((graph_data['edge_index'] > end_of_reactant_idx).any(dim=0))[0]
    if len(end_of_reactant_edge_idx) > 0:
        end_of_reactant_edge_idx = end_of_reactant_edge_idx[0].item()
    else:
        end_of_reactant_edge_idx = graph_data['edge_index'].shape[1]
    
    start_of_product_edge_idx = torch.where((graph_data['edge_index'] >= start_of_product_idx).any(dim=0))[0]
    if len(start_of_product_edge_idx) > 0:
        start_of_product_edge_idx = start_of_product_edge_idx[0].item()
    else:
        start_of_product_edge_idx = graph_data['edge_index'].shape[1]
    
    # Create reactants graph
    reactants_graph = {}
    for key, value in graph_data.items():
        if key == 'edge_index':
            reactants_graph[key] = value[:,:end_of_reactant_edge_idx]
        elif key in ['edge_attr', 'bond_dirs']:  # Edge features
            reactants_graph[key] = value[:end_of_reactant_edge_idx]
        else:  # Node features
            reactants_graph[key] = value[reactant_mask]
    
    # Create products graph
    products_graph = {}
    for key, value in graph_data.items():
        if key == 'edge_index':
            products_graph[key] = value[:,start_of_product_edge_idx:] - start_of_product_idx
        elif key in ['edge_attr', 'bond_dirs']:  # Edge features
            products_graph[key] = value[start_of_product_edge_idx:]
        else:  # Node features
            products_graph[key] = value[product_mask]
    
    return reactants_graph, products_graph

# def split_reaction_graph_to_reactants_and_products(graph_data):
#     '''
#         Split a reaction graph into reactants and products.
#     '''
#     end_of_reactant_idx = torch.nonzero(graph_data['mol_assignment']==0).max() # assumes 0 is for reactant, 1 for dummy nodes, 2 for product
#     end_of_dummy_idx = torch.nonzero(graph_data['mol_assignment']==1).max()
#     end_of_reactant_edge_idx = torch.where((graph_data['edge_index'] > end_of_reactant_idx).any(dim=0))[0][0].item()
#     start_of_product_edge_idx = torch.where((graph_data['edge_index'] > end_of_dummy_idx).any(dim=0))[0][0].item()
#     reactants_graph = {key: value[:end_of_reactant_idx+1] if key!='edge_index' else value[..., :end_of_reactant_edge_idx] for key, value in graph_data.items()}
#     products_graph = {key: value[end_of_dummy_idx+1:] if key!='edge_index' else value[..., start_of_product_edge_idx:] for key, value in graph_data.items()}
#     dummy_nodes = graph_data['x'][end_of_reactant_idx+1:end_of_dummy_idx+1]
#     products_graph['edge_index'] = products_graph['edge_index'] - reactants_graph['x'].shape[0] - dummy_nodes.shape[0]
#     return reactants_graph, products_graph

def graph_to_smiles_with_stereochem(graph_data, cfg, dataset_information):
    atom_types = dataset_information['atom_types']
    atom_charges = dataset_information['atom_charges']
    chiral_tags = get_rdkit_chiral_tags(dataset_information['atom_chiral_tags'])
    bond_types = get_rdkit_bond_types(dataset_information['bond_types'])
    bond_dirs = get_rdkit_bond_dirs(dataset_information['bond_dirs'])

    rw_mol = Chem.RWMol()

    # add atoms 
    for i, atom in enumerate(graph_data['x']):
        atom_symbol = atom_types[atom.argmax().item()]
        atom = Chem.Atom(atom_symbol)
        atom_charge = atom_charges[graph_data['atom_charges'][i].argmax().item()]
        atom.SetFormalCharge(int(atom_charge))
        atom.SetChiralTag(chiral_tags[graph_data['atom_chiral'][i].argmax().item()])
        rw_mol.AddAtom(atom)

    # add bonds between specific atoms
    for i, bond in enumerate(graph_data['edge_index'].T):
        if bond[1] < bond[0]:
            continue
        rw_mol.AddBond(bond[0].item(), bond[1].item(), bond_types[graph_data['edge_attr'][i].argmax().item()])
        # Set bond direction
        rw_mol.GetBondBetweenAtoms(bond[0].item(), bond[1].item()).SetBondDir(
            bond_dirs[graph_data['bond_dirs'][i].argmax().item()]
        )

    # readjust chiral tags with cip ranking
    cip_ranking = get_cip_ranking(
        rw_mol
    )  # Note: this could be slightly slow for large molecules
    for atom in rw_mol.GetAtoms():
        new_tag = switch_between_bond_cw_ccw_label_and_cip_based_label(
                atom, atom.GetChiralTag(), cip_ranking
            )
        atom.SetChiralTag(new_tag)
    
    return Chem.MolToSmiles(rw_mol)

def reaction_graph_to_smiles_with_stereochem(graph_data, cfg, dataset_information):
    '''
        Convert a graph data dictionary to a smiles string.
    '''
    reactants_graph, product_graph = split_reaction_graph_to_reactants_and_products(graph_data)
    reaction_smiles = graph_to_smiles_with_stereochem(reactants_graph, cfg, dataset_information=dataset_information) + '>>' + graph_to_smiles_with_stereochem(product_graph, cfg, dataset_information=dataset_information)
    return reaction_smiles

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

def add_dummy_nodes(reactants, products, mol_idx, cfg, dataset_information):
    '''
        Add dummy nodes to the graph.
    '''
    atom_types = dataset_information['atom_types']
    atom_charges = dataset_information['atom_charges']
    atom_chiral_tags = dataset_information['atom_chiral_tags']
    bond_types = dataset_information['bond_types']
    bond_dirs = dataset_information['bond_dirs']
    num_dummy_nodes_to_add = compute_num_dummy_nodes_to_add(reactants, products, cfg)
     # NOTE: for non training set, we don't want to drop reactions which cannot be generated with the given dummy nodes
    num_dummy_nodes_to_add = max(num_dummy_nodes_to_add, 0)

    return {'x':  F.one_hot(torch.ones(num_dummy_nodes_to_add, dtype=torch.long) * atom_types.index(cfg.dataset.dummy_node_type), num_classes=len(atom_types)).float(),
            'edge_index': torch.zeros([2, 0], dtype=torch.long),
            'edge_attr': torch.zeros([0, len(bond_types)], dtype=torch.long),
            'mol_assignment': torch.ones(num_dummy_nodes_to_add, dtype=torch.long) * mol_idx,
            'atom_map_numbers': torch.zeros(num_dummy_nodes_to_add, dtype=torch.long),
            'atom_charges': F.one_hot(torch.zeros(num_dummy_nodes_to_add, dtype=torch.long), num_classes=len(atom_charges)),
            'atom_chiral': F.one_hot(torch.zeros(num_dummy_nodes_to_add, dtype=torch.long), num_classes=len(atom_chiral_tags)),
            'bond_dirs': torch.zeros([0, len(bond_dirs)], dtype=torch.long)}

def concatenate_data_dicts(data_dicts):
    '''
        Concatenate a list of data dictionaries.
    '''

    # odes_rct = torch.cat([nodes_rct, nodes_dummy], dim=0)
    # edge_index_rct = torch.cat([edge_index_rct, edges_idx_dummy], dim=1)
    # bond_types_rct = torch.cat([bond_types_rct, bond_types_dummy], dim=0)
    # atom_charges_rct = torch.cat([atom_charges_rct, F.one_hot(torch.zeros(num_dummy_nodes_to_add, dtype=torch.long), num_classes=len(cfg.dataset.atom_charges))], dim=0)
    # atom_chiral_rct = torch.cat([atom_chiral_rct, F.one_hot(torch.zeros(num_dummy_nodes_to_add, dtype=torch.long), num_classes=len(cfg.dataset.atom_chiral_tags))], dim=0)
    # bond_dirs_rct = torch.cat([bond_dirs_rct, torch.zeros([0, len(cfg.dataset.bond_dirs)], dtype=torch.long)], dim=0)
    # atom_map_reactants = torch.cat([atom_map_reactants, torch.zeros(num_dummy_nodes_to_add, dtype=torch.long)], dim=0)
    # mol_assignment_reactants = torch.cat([mol_assignment_reactants, torch.zeros(num_dummy_nodes_to_add, dtype=torch.long)], dim=0)
    output_data_dict = {}
    for i, data_dict in enumerate(data_dicts):
        for key in data_dict:
            if key not in output_data_dict:
                output_data_dict[key] = data_dict[key]
            else:
                if key=='edge_index':
                    output_data_dict[key] = torch.cat([output_data_dict[key], data_dict[key]], dim=1)
                else:
                    output_data_dict[key] = torch.cat([output_data_dict[key], data_dict[key]], dim=0)
    return output_data_dict

def is_graph_data_valid(graph_data):
    # Check that the one-hot encodings are correct
    assert (graph_data['edge_attr'].sum(-1) == 1).all(), 'edge_attr should be one-hot encodings' # Should be one-hot encodings here
    assert (graph_data['bond_dirs'].sum(-1) == 1).all(), 'bond_dirs should be one-hot encodings' # TODO: REPLACE WITH .all() for efficiency
    assert (graph_data['x'].sum(-1) == 1).all(), 'x should be one-hot encodings'

    # Make sure that there are no duplicate edges
    assert len(set([(edge[0].item(), edge[1].item()) for edge in graph_data['edge_index'].T])) == graph_data['edge_index'].shape[1],\
          'there are duplicate edges'

    # Make sure that there are no edges pointing to nodes that don't exist
    assert (graph_data['edge_index'] < graph_data['x'].shape[0]).flatten().all(), 'edges point to non-existent nodes'

    return True

def compute_num_dummy_nodes_to_add(reactants, products, cfg):
    nb_product_nodes = len(Chem.MolFromSmiles(products.strip()).GetAtoms())
    nb_rct_nodes = len(Chem.MolFromSmiles(reactants.strip()).GetAtoms())
    num_dummy_nodes_to_add = nb_product_nodes + cfg.dataset.num_dummy_nodes - nb_rct_nodes

    return num_dummy_nodes_to_add

def should_skip_reaction(reactants, products, stage, cfg):
    num_dummy_nodes_to_add = compute_num_dummy_nodes_to_add(reactants, products, cfg)
    if num_dummy_nodes_to_add<0 and stage=='train':
        #print(f'dropping rxn {reactants}>>{products} in {stage} set')
        return True
    return False

def get_canonicalized_molecule_smiles_without_atom_mapping(smi, kekulize=True):
    mol = Chem.MolFromSmiles(smi.strip())
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    return Chem.MolToSmiles(mol)

def get_canonicalized_reaction_smiles_without_atom_mapping(reaction_smiles):
    '''
        Get the canonicalized reaction smiles without atom mapping.
    '''
    reactants, products = reaction_smiles.split('>>')
    return get_canonicalized_molecule_smiles_without_atom_mapping(reactants) + '>>' + get_canonicalized_molecule_smiles_without_atom_mapping(products)

def _check_constructed_graph_valid(smi, graph, cfg, dataset_information, reaction_smiles_idx):
    '''
        Check if the reaction smiles is valid.
    '''
    reconstructed_smi = graph_to_smiles_with_stereochem(graph, cfg, dataset_information=dataset_information)
    cano_smi = get_canonicalized_molecule_smiles_without_atom_mapping(smi)
    if reconstructed_smi != cano_smi:
        print(f'============== Reactants smiles {reconstructed_smi} do not match {cano_smi}, idx {reaction_smiles_idx}')

def _check_constructed_reaction_graph_valid(smi, graph, cfg, dataset_information, reaction_smiles_idx):
    '''
        Check if the reaction smiles is valid.
    '''
    reconstructed_smi = reaction_graph_to_smiles_with_stereochem(graph, cfg, dataset_information=dataset_information)
    cano_smi = get_canonicalized_reaction_smiles_without_atom_mapping(smi)
    if reconstructed_smi != cano_smi:
        print(f'============== Reactants smiles {reconstructed_smi} do not match {cano_smi} for idx {reaction_smiles_idx}')

def _check_if_difference_is_only_explicit_hydrogens(rxn_smi1, rxn_smi2):
    '''
        Check if the difference between the two smiles is only explicit hydrogens.
    '''
    products1 = rxn_smi1.split('>>')[-1]
    products2 = rxn_smi2.split('>>')[-1]
    reactants1 = rxn_smi1.split('>>')[0]
    reactants2 = rxn_smi2.split('>>')[0]

    if products1 != products2:
        return False
    if reactants1 != reactants2:
        return False

def reaction_smiles_to_graph(cfg,
                             stage,
                             dataset_information,
                             reaction_smiles_idx,
                             reaction_smiles):
    '''
        Convert a reaction smiles to a graph.
        
        The processing here is based on the fact that rdkit can handle two molecule smiles as a single disjoint molecule, 
        so no need to process one reactant/product at a time.

        For now forget all supernode stuff, try to get the model to work without it.
    '''
    if not is_reaction_smiles_valid(reaction_smiles):
        raise ValueError(f'Reaction {reaction_smiles} is not valid')
    # if reaction_smiles_idx<3290:
    #     return None
    # if reaction_smiles_idx==3290:
    #     print(f'reaction_smiles={reaction_smiles}')
    reaction_smiles_without_atom_mapping = get_canonicalized_reaction_smiles_without_atom_mapping(reaction_smiles)
    reactants, products = reaction_smiles_without_atom_mapping.split('>>')
    if should_skip_reaction(reactants, products, stage, cfg):
        return None
    reactants_data = smiles_to_graph_with_stereochem(smi=reactants, cfg=cfg, dataset_information=dataset_information, mol_idx=0)
    _check_constructed_graph_valid(reactants, reactants_data, cfg, dataset_information, reaction_smiles_idx)
    dummy_data  = add_dummy_nodes(reactants, products, mol_idx=1, cfg=cfg, dataset_information=dataset_information)
    # Permute the rows here to make sure that the NN can only process topological information
    # if cfg.dataset.permute_mols:
    #     permute_rows(nodes_rct, atom_map_reactants, mol_assignment_reactants, edge_index_rct)
    # get the num of atoms in molecule
    # NOTE: set dataset.max_atoms_rxn_parse to None if you want to parse all reactions regardless of size
    product_data = smiles_to_graph_with_stereochem(smi=products, cfg=cfg, dataset_information=dataset_information, mol_idx=2)
    product_data['edge_index'] = product_data['edge_index'] + reactants_data['x'].shape[0] + dummy_data['x'].shape[0]
    reaction_data = concatenate_data_dicts([reactants_data, dummy_data, product_data])
    _check_constructed_reaction_graph_valid(reaction_smiles_without_atom_mapping, reaction_data, cfg, dataset_information, reaction_smiles_idx)
    y = torch.zeros((1, 0), dtype=torch.float)
    graph = Data(**reaction_data, smiles=reaction_smiles, y=y)
    if not is_graph_data_valid(graph):
        raise ValueError(f'Graph data is not valid for reaction {reaction_smiles}')
    return graph