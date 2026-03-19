import re
import torch
from rdkit import Chem
import torch.nn.functional as F
import os
import pathlib
import copy
from diffalign.utils import graph
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import Draw
import logging
log = logging.getLogger(__name__)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rxn_list_to_str(rcts, prods):
    rxn_str = ''

    for i, m in enumerate(rcts):
        if i==len(rcts)-1: # last molecule is product
            rxn_str += m
        else:
            rxn_str += m + '.'
    
    rxn_str += '>>'
    
    for i, m in enumerate(prods):
        if i==len(prods)-1: # last molecule is product
            rxn_str += m
        else:
            rxn_str += m + '.'
        
    return rxn_str
        
def get_cano_list_smiles(X, E, atom_types, bond_types, plot_dummy_nodes=False):
    '''
        Returns canonical smiles of all the molecules in a reaction
        given a dense matrix representation of said reaction.
        Invidual molecules are identified by splitting their smiles by '.'.
        A set of canonical smiles is returned for each rxn.
        Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
        Handles batched reactions.

        X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
        E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

        return: list of smiles of valid molecules from rxn.
    '''   
    assert X.ndim==2 and E.ndim==3,\
            'Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n).' \
            + f' Got X.shape={X.shape} and E.shape={E.shape} instead.'       

    suno_idx = atom_types.index('SuNo') # offset because index 0 is for no node

    all_rcts = []
    all_prods = []
    for j in range(X.shape[0]): # for each rxn in batch
        suno_indices = (X[j,:]==suno_idx).nonzero(as_tuple=True)[0].cpu() 
        cutoff = 1 if 0 in suno_indices else 0 # relevant in case there's a SuNo node in the first position
        atoms = torch.tensor_split(X[j,:], suno_indices, dim=-1)[cutoff:] # ignore first set (SuNo)
        edges = torch.tensor_split(E[j,:,:], suno_indices, dim=-1)[cutoff:]

        rct_smiles = []
        prod_smiles = []
        for i, mol_atoms in enumerate(atoms): # for each mol in rxn
            mol_edges_to_all = edges[i] 
            cutoff = 1 if 0 in suno_indices else 0 # relevant in case there's a SuNo node in the first position
            mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[cutoff:]
            mol_edges = mol_edges_t[i] # the edges from the molecule to the entire reaction
            cutoff = 1 if suno_idx in mol_atoms else 0
            mol_atoms = mol_atoms[cutoff:] # (n-1)
            mol_edges = mol_edges[cutoff:,:][:,cutoff:] # (n-1, n-1)
            mol = mol_from_graph(node_list=mol_atoms, adjacency_matrix=mol_edges, 
                                 atom_types=atom_types, bond_types=bond_types, plot_dummy_nodes=plot_dummy_nodes)                     
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True)
            set_mols = smiles.split('.')
            if i==len(atoms)-1:
                prod_smiles.extend(set_mols)
            else:
                rct_smiles.extend(set_mols)
        all_rcts.append(rct_smiles)
        all_prods.append(prod_smiles)

    return all_rcts, all_prods

def get_cano_smiles_from_dense(X, E, atom_types, bond_types, return_dict=False):     
    '''
        Returns canonical smiles of all the molecules in a reaction
        given a dense matrix representation of said reaction.
        Dense matrix representation = X (bs*n_samples, n), E (bs*n_samples, n, n).
        Handles batched reactions.

        X: nodes of a reaction in matrix dense format. (bs*n_samples, n)
        E: Edges of a reaction in matrix dense format. (bs*n_samples, n, n)

        return: list of smiles of valid molecules from rxn.
    '''   
    assert X.ndim==2 and E.ndim==3,\
            'Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n).' \
            + f' Got X.shape={X.shape} and E.shape={E.shape} instead.'       

    suno_idx = atom_types.index('SuNo') # offset because index 0 is for no node

    all_smiles = {}
    all_rxn_str = []
    for j in range(X.shape[0]): # for each rxn in batch
        #print(f'j {j}\n')
        suno_indices = (X[j,:]==suno_idx).nonzero(as_tuple=True)[0].cpu() 
        cutoff = 1 if 0 in suno_indices else 0
        atoms = torch.tensor_split(X[j,:], suno_indices, dim=-1)[cutoff:] # ignore first set (SuNo)
        edges = torch.tensor_split(E[j,:,:], suno_indices, dim=-1)[cutoff:]

        rxn_smiles = []
        rxn_str = ''
        for i, mol_atoms in enumerate(atoms): # for each mol in rxn
            cutoff = 1 if 0 in suno_indices else 0
            mol_edges_to_all = edges[i] 
            mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[cutoff:]
            mol_edges = mol_edges_t[i]
            cutoff = 1 if suno_idx in mol_atoms else 0
            mol_atoms = mol_atoms[cutoff:] # (n-1)
            mol_edges = mol_edges[cutoff:,:][:,cutoff:] # (n-1, n-1)
            mol = mol_from_graph(node_list=mol_atoms, adjacency_matrix=mol_edges, 
                                 atom_types=atom_types, bond_types=bond_types)                     
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)
            
            if i<len(atoms)-2: # if the mol is not the last reactant
                rxn_str += smiles + '.' # instead of dot to make it easier to read rxn
            elif i==len(atoms)-2: # if the mol is the last reactant
                rxn_str += smiles + '>>'
            elif i==len(atoms)-1: # if the mol is the product
                rxn_str += smiles
                 
            rxn_smiles.append(smiles)
        all_rxn_str.append(rxn_str)
        all_smiles[j] = rxn_smiles

    return all_smiles if return_dict else all_rxn_str

def get_mol_nodes(mol, atom_types, with_formal_charge=True, get_atom_mapping=False):
    atoms = mol.GetAtoms()
    atom_mapping = torch.zeros(len(atoms), dtype=torch.long)
    
    for i, atom in enumerate(atoms):
        if with_formal_charge: at = atom.GetSymbol() if atom.GetFormalCharge()==0 else atom.GetSymbol()+f'{atom.GetFormalCharge():+}'
        else: at = atom.GetSymbol()
        try:
            atom_type = torch.tensor([atom_types.index(at)], dtype=torch.long) # needs to be int for one hot
        except:
            log.info(f'at {at}\n')
            log.info(f'atom types: {atom_types}')
            # exit()
        atom_types_ = torch.cat((atom_types_, atom_type), dim=0) if i > 0 else atom_type
        atom_mapping[i] = atom.GetAtomMapNum()
    
    atom_feats = F.one_hot(atom_types_, num_classes=len(atom_types)).float()
    
    if get_atom_mapping: 
        return atom_feats, atom_mapping
    
    return atom_feats

def get_mol_edges(mol, bond_types, offset=1):
    '''
        Input:
            offset (optional): default: 1. To account for SuNo added at the beginning of the graph.
    '''
    # print(f'len(mol.GetBonds()) {len(mol.GetBonds())}\n')
    for i, b in enumerate(mol.GetBonds()):
        beg_atom_idx = b.GetBeginAtom().GetIdx()
        end_atom_idx = b.GetEndAtom().GetIdx()
        e_beg = torch.tensor([beg_atom_idx+offset, end_atom_idx+offset], dtype=torch.long).unsqueeze(-1)
        e_end = torch.tensor([end_atom_idx+offset, beg_atom_idx+offset], dtype=torch.long).unsqueeze(-1)
        e_type = torch.tensor([bond_types.index(b.GetBondType()), 
                               bond_types.index(b.GetBondType())], dtype=torch.long) # needs to be int for one hot
        begs = torch.cat((begs, e_beg), dim=0) if i > 0 else e_beg
        ends = torch.cat((ends, e_end), dim=0) if i > 0 else e_end
        edge_type = torch.cat((edge_type, e_type), dim=0) if i > 0 else e_type

    if len(mol.GetBonds())>0:
        edge_index = torch.cat((begs, ends), dim=1).mT.contiguous()
        edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)).float() # add 1 to len of bonds to account for no edge
    else:
        edge_index = torch.tensor([]).long().reshape(2,0)
        edge_attr = torch.tensor([]).float().reshape(0, len(bond_types))

    return edge_index, edge_attr

def create_canonicalized_mol(mol):
    atom_mapping = {}
    for atom in mol.GetAtoms():
        atom_mapping[atom.GetIdx()] = atom.GetAtomMapNum()

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    
    smi = Chem.MolToSmiles(mol)
    mol_cano = Chem.MolFromSmiles(smi)
    matches = mol.GetSubstructMatches(mol_cano) # This maps from the canonical atom order to the original atom order, I think
    if matches: # What if no matches?
        for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
            atom.SetAtomMapNum(atom_mapping[mat]) # Is this how it works?
    
    return mol_cano

def mol_to_graph(mol, atom_types, bond_types, offset=0, with_explicit_h=True, with_formal_charge=True, get_atom_mapping=False,
                 canonicalize_molecule=True):
    if type(mol)==str: mol = Chem.MolFromSmiles(mol)

    # TODO: Could add a flag for this to be turned on or off
    if canonicalize_molecule:
        mol = create_canonicalized_mol(mol)
    
    Chem.RemoveStereochemistry(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    if with_explicit_h: mol = Chem.AddHs(mol, explicitOnly=True)
    
    if not get_atom_mapping:    
        m_nodes = get_mol_nodes(mol=mol, atom_types=atom_types, with_formal_charge=with_formal_charge, get_atom_mapping=get_atom_mapping)
    else:
        m_nodes, atom_map = get_mol_nodes(mol=mol, atom_types=atom_types, 
                                with_formal_charge=with_formal_charge, 
                                get_atom_mapping=get_atom_mapping)
        
    m_edge_index, m_edge_attr = get_mol_edges(mol=mol, bond_types=bond_types, offset=offset)

    if not get_atom_mapping:
        return m_nodes, m_edge_index, m_edge_attr
    else:
        return m_nodes, m_edge_index, m_edge_attr, atom_map

def mol_from_graph(node_list, adjacency_matrix, atom_types, bond_types, plot_dummy_nodes=False):
    """
        Convert graphs to RDKit molecules.

        node_list: the nodes of one molecule (n)
        adjacency_matrix: the adjacency_matrix of the molecule (n, n)

        return: RDKit's editable mol object.
    """
    # fc = node_list[...,-1] # get formal charge of each atom 
    # node_list = torch.argmax(node_list[...,:-1], dim=-1) 
    # adjacency_matrix = torch.argmax(adjacency_matrix, dim=-1) 
    # create empty editable mol object
        
    mol = Chem.RWMol()
    if not plot_dummy_nodes:
        masking_atom = atom_types.index('U') if 'U' in atom_types else 0
    else:
        masking_atom = 0

    node_to_idx = {} # needed because using 0 to mark node paddings 
    # add atoms to mol and keep track of index
    for i in range(len(node_list)):
        # ignore padding nodes
        if node_list[i]==0 or node_list[i]==masking_atom:
            continue
        at = atom_types[int(node_list[i])]
        fc = re.findall('[-+]\d+', at)
        s = re.split('[-+]\d+', at)[0]
        a = Chem.Atom(s)
        if len(fc)!=0: a.SetFormalCharge(int(fc[0]))
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the symmetric matrix
            if iy <= ix:
                continue
            # only consider nodes parsed earlier (ignore empty nodes)
            if (ix not in node_to_idx.keys()) or (iy not in node_to_idx.keys()) :
                continue
            # only consider valid edges types
            
            bond_type = bond_types[bond]
            if bond_type not in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE]:
                continue
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    return mol

def connect_mol_to_supernode(mol, atom_types, bond_types, supernode_nb=1):
    s_nodes = F.one_hot(torch.tensor([atom_types.index('SuNo')], dtype=torch.long), 
                        num_classes=len(atom_types)).float()
    
    # connect all atoms to supernode (idx supernode_nb - 1)
    # print(f'len(mol.GetAtoms()) {len(mol.GetAtoms())}\n')
    for i, a in enumerate(mol.GetAtoms()):
        e_beg = torch.tensor([supernode_nb-1, a.GetIdx()+supernode_nb], dtype=torch.long).unsqueeze(-1)
        e_end = torch.tensor([a.GetIdx()+supernode_nb, supernode_nb-1], dtype=torch.long).unsqueeze(-1)

        begs = torch.cat((begs, e_beg), dim=0) if i > 0 else e_beg
        ends = torch.cat((ends, e_end), dim=0) if i > 0 else e_end

    s_edge_index = torch.cat((begs, ends), dim=1).mT.contiguous()
    edge_type = torch.full(size=(begs.shape[0],), fill_value=bond_types.index('mol'), dtype=torch.long) # needs to be int for one hot
    s_edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)).float() # add 1 to len of bonds to account for no edge

    # print(f's_edge_index.shape {s_edge_index.shape}\n')
        
    return s_nodes, s_edge_index, s_edge_attr

def rxn_plot(rxn, atom_types, bond_types, filename='test.png', return_smarts=False, plot_dummy_nodes=False):
    '''
        Return a plot of a rxn given a rxn graph (with supernodes).
    '''
    rxn_smrts = rxn_from_graph_supernode(data=rxn, atom_types=atom_types, bond_types=bond_types, plot_dummy_nodes=plot_dummy_nodes)
    rxn_obj = Reactions.ReactionFromSmarts(rxn_smrts)
    
    # drawer = rdMolDraw2D.MolDraw2DCairo(800, 200)
    # drawer.SetFontSize(1.0)
    # drawer.DrawReaction(rxn_obj)
    # drawer.FinishDrawing()
    # drawer.WriteDrawingText(filename)
    rxn_img = Draw.ReactionToImage(rxn_obj) # TODO: investigate fancy reaction plotting
    
    if return_smarts:
        return rxn_img, rxn_smrts
    return rxn_img
    
def rxn_to_graph_supernode(mol, atom_types, bond_types, supernode_nb=1, with_explicit_h=True, 
                           with_formal_charge=True, add_supernode_edges=True, get_atom_mapping=False,
                           canonicalize_molecule=True):
    if type(mol)==str: mol = Chem.MolFromSmiles(mol)
    
    if not get_atom_mapping:    
        m_nodes, m_edge_index, m_edge_attr = mol_to_graph(mol=mol, atom_types=atom_types, 
                                                          bond_types=bond_types, offset=supernode_nb, 
                                                          with_explicit_h=with_explicit_h,
                                                          with_formal_charge=with_formal_charge, 
                                                          get_atom_mapping=get_atom_mapping,
                                                          canonicalize_molecule=canonicalize_molecule)
    else:
        m_nodes, m_edge_index, m_edge_attr, atom_map = mol_to_graph(mol=mol, atom_types=atom_types, 
                                                                    bond_types=bond_types, offset=supernode_nb, 
                                                                    with_explicit_h=with_explicit_h,
                                                                    with_formal_charge=with_formal_charge, 
                                                                    get_atom_mapping=get_atom_mapping,
                                                                    canonicalize_molecule=canonicalize_molecule)
        # add 0 for SuNo node
        atom_map = torch.cat((torch.zeros(1, dtype=torch.long), atom_map), dim=0) 
    #     print(f'm_edge_index {m_edge_index.shape}\n')

    # print(f'add_supernode_edges {add_supernode_edges}\n')
    if add_supernode_edges:
        s_nodes, s_edge_index, s_edge_attr = connect_mol_to_supernode(mol=mol, atom_types=atom_types, 
                                                      bond_types=bond_types, supernode_nb=supernode_nb)
        # print(f'm_nodes.shape {m_nodes.shape}\n')
        g_nodes = torch.cat([s_nodes, m_nodes], dim=0)
        g_edge_index = torch.cat([s_edge_index, m_edge_index], dim=1) # s/m_edge_index: (2, n_edges)
        g_edge_attr = torch.cat([s_edge_attr, m_edge_attr], dim=0)
    else:
        s_nodes = F.one_hot(torch.tensor([atom_types.index('SuNo')], dtype=torch.long), 
                        num_classes=len(atom_types)).float()
        g_nodes = torch.cat([s_nodes, m_nodes], dim=0)
        g_edge_index = m_edge_index
        g_edge_attr = m_edge_attr
        
    if not get_atom_mapping:
        return g_nodes, g_edge_index, g_edge_attr
    else:
        return g_nodes, g_edge_index, g_edge_attr, atom_map

def rxn_from_graph_supernode(data, atom_types, bond_types, plot_dummy_nodes=True):
    if type(data)!=graph.PlaceHolder:
        data_ = graph.to_dense(data)
        data_ = data_.mask(data_.node_mask, collapse=True)
    else:
        data_ = copy.deepcopy(data)
    
    assert data_.X.shape[0]==1, 'Function expects a single example, batch given instead.'
    
    X = data_.X.squeeze()
    E = data_.E.squeeze()
    suno_idx = atom_types.index('SuNo') # offset because index 0 is for no node  
    suno_indices = (X.squeeze()==suno_idx).nonzero(as_tuple=True)[0].cpu()
    cutoff = 1 if 0 in suno_indices else 0
    mols_atoms = torch.tensor_split(X.squeeze(), suno_indices, dim=-1)[cutoff:] # ignore first set (SuNo)
    mols_edges = torch.tensor_split(E.squeeze(), suno_indices, dim=-1)[cutoff:]

    smiles = []
    for i, mol_atoms in enumerate(mols_atoms): # for each mol in sample
        mol_edges_to_all = mols_edges[i] 
        mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[cutoff:] # ignore first because empty SuNo set
        mol_edges = mol_edges_t[i]
        cutoff = 1 if suno_idx in mol_atoms else 0
        mol_atoms = mol_atoms[cutoff:] # (n-1)
        mol_edges = mol_edges[cutoff:,:][:,cutoff:] # (n-1, n-1)
        
        mol = mol_from_graph(node_list=mol_atoms, adjacency_matrix=mol_edges, 
                             atom_types=atom_types, bond_types=bond_types, 
                             plot_dummy_nodes=plot_dummy_nodes)
        #Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        smiles.append(smi)
    
    rxn_smrts = '>>' + smiles[-1]
    
    for smi in smiles[1:-1]:
        rxn_smrts = '.' + smi + rxn_smrts
    
    rxn_smrts = smiles[0] + rxn_smrts
    
    return rxn_smrts

def check_valid_product_in_rxn(X, E, true_rxn_smiles, atom_types, bond_types):     
    '''
        Checks if the product given in dense tensor format is valid.

        Input:
            X: nodes of a reaction in (discrete) matrix dense format. (bs*n_samples, n)
            E: Edges of a reaction in (discrete) matrix dense format. (bs*n_samples, n, n)
            n_samples: number of samples generated for each rxn.

        Output: 
            avg_validity: avg validity of each set of precursors generated for each test product. (bs*samples,)
    '''   
    assert X.ndim==2 and E.ndim==3, 'Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n).' \
                                    + f' Got X.shape={X.shape} and E.shape={E.shape} instead.'       

    all_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(device) # all generated precursors are valid
    atleastone_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(device) # at least one generated precursor is valid
    suno_idx = atom_types.index('SuNo')
    gen_rxns = {}
    for j in range(X.shape[0]): # for each rxn in batch    
        log.debug(f'True rxn: {true_rxn_smiles[j]}\n')
        suno_indices = (X[j,:]==suno_idx).nonzero(as_tuple=True)[0].cpu() 
        log.debug(f'\nChecking precursors {j}\n')

        # TODO: refactor to make more generic => ignore whatever is masked
        mol_atoms = torch.tensor_split(X[j,:], suno_indices, dim=-1)[-1] # get only last set (product)
        mol_atoms = mol_atoms[1:] # ignore first atom because it's SuNo, (n-1)
        mol_edges_to_all = torch.tensor_split(E[j,:,:], suno_indices, dim=-1)[-1]
        mol_edges = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[-1] # ignore first because empty SuNo set
        mol_edges = mol_edges[1:,:][:,1:] # ignore edges to/from first atom because SuNo, (n-1,n-1)
        # smi, mol, issue = check_valid_molecule(mol_atoms, mol_edges, atom_types, bond_types)
        ## use the same function as when evaluating single molecules
        # smi, mol, issue = check_valid_molecule_fragment(mol_atoms=mol_atoms, mol_edges=mol_edges, 
        #                                                 atom_types=atom_types, bond_types=bond_types)
        mol = mol_from_graph(node_list=mol_atoms, adjacency_matrix=mol_edges, atom_types=atom_types, 
                                bond_types=bond_types)   
        try:                  
            smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
            issue = 'no_issue'
        except:
            issue = 'invalid'
            
        all_valid_per_sample = 0
        all_mols_in_prods = 0
        log.debug(f'Product #{j}: {smi}\n')
        if issue=='no_issue':
            log.debug(f'valid products!\n')  
            prods = smi.split('.')
            all_mols_in_prods += len(prods)
            for i_p, p in enumerate(prods):
                mol_p = Chem.MolFromSmiles(p)
                try:
                    smi_p = Chem.MolToSmiles(mol_p, kekuleSmiles=True)
                    all_valid_per_sample += 1
                except:
                    log.debug(f'p {i_p} is invalid\n')
        
        rct = true_rxn_smiles[j].split('>>')[0]
        gen_rxn = rct + '>>' + smi

        if rct in gen_rxns.keys():
            gen_rxns[rct].append(gen_rxn)
        else:
            gen_rxns[rct] = [gen_rxn]

        all_valid[j] = float((all_valid_per_sample==all_mols_in_prods) and all_mols_in_prods>0)
        atleastone_valid[j] = float((all_valid_per_sample>0) and all_mols_in_prods>0)

    return all_valid, atleastone_valid, gen_rxns
    
def check_valid_reactants_in_rxn(X, E, true_rxn_smiles, n_samples, atom_types, bond_types):     
    '''
        Checks if the molecules given in dense tensor format are valid.

        Input:
            X: nodes of a reaction in (discrete) matrix dense format. (bs*n_samples, n)
            E: Edges of a reaction in (discrete) matrix dense format. (bs*n_samples, n, n)
            n_samples: number of samples generated for each rxn.

        Output: 
            avg_validity: avg validity of each set of precursors generated for each test product. (bs*samples,)
    '''   
    assert X.ndim==2 and E.ndim==3,\
           'Expects batched X of shape (bs*n_samples, n), and batched E of shape (bs*n_samples, n, n).' \
           + f' Got X.shape={X.shape} and E.shape={E.shape} instead.'       

    all_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(device) # all generated precursors are valid
    atleastone_valid = torch.zeros([X.shape[0]], dtype=torch.float).to(device) # at least one generated precursor is valid
    suno_idx = atom_types.index('SuNo')
    gen_rxns = {}
    for j in range(X.shape[0]): # for each rxn in batch
        log.debug(f'True rxn: {true_rxn_smiles[j]}\n')
        suno_indices = (X[j,:]==suno_idx).nonzero(as_tuple=True)[0].cpu() 
        cutoff = 1 if 0 in suno_indices else 0
        # TODO: refactor to make more generic => ignore whatever is masked
        mols_atoms = torch.tensor_split(X[j,:], suno_indices, dim=-1)[cutoff:-1] # ignore first set (SuNo) and last set (product)
        mols_edges = torch.tensor_split(E[j,:,:], suno_indices, dim=-1)[cutoff:-1]

        log.debug(f'\nChecking precursors {j}, total nb of molecules: {len(mols_atoms)}\n')
        all_valid_per_sample = 0
        all_mols_in_rcts = 0
        gen_rxn = ''
        for i, mol_atoms in enumerate(mols_atoms): # for each mol in sample
            cutoff = 1 if 0 in suno_indices else 0
            mol_edges_to_all = mols_edges[i] 
            mol_edges_t = torch.tensor_split(mol_edges_to_all, suno_indices, dim=0)[cutoff:] # ignore first because empty SuNo set
            mol_edges = mol_edges_t[i]
            cutoff = 1 if suno_idx in mol_atoms else 0
            mol_atoms = mol_atoms[cutoff:] # (n-1)
            mol_edges = mol_edges[cutoff:,:][:,cutoff:] # (n-1, n-1)
            #smi, mol, issue = check_valid_molecule(mol_atoms, mol_edges, atom_types, bond_types)
            ## use the same function as when evaluating single molecules
            # smi, mol, issue = check_valid_molecule_fragment(mol_atoms=mol_atoms, mol_edges=mol_edges, 
            #                                                 atom_types=atom_types, bond_types=bond_types)
            mol = mol_from_graph(node_list=mol_atoms, adjacency_matrix=mol_edges, 
                                 atom_types=atom_types, bond_types=bond_types)   
            try:                  
                smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
                issue = 'no_issue'
            except:
                issue = 'invalid'
            
            log.debug(f'Molecule #{i}: {smi}\n')
            if issue=='no_issue':
                log.debug(f'valid reactants!\n')  
                rcts = smi.split('.')
                all_mols_in_rcts += len(rcts)
                for i_r, r in enumerate(rcts):
                    mol_r = Chem.MolFromSmiles(r)
                    try:
                        smi_r = Chem.MolToSmiles(mol_r, kekuleSmiles=True)
                        all_valid_per_sample += 1
                    except:
                        log.debug(f'r {i_r} is invalid\n')
            
            gen_rxn = smi if gen_rxn=='' else gen_rxn + '.' + smi

        product = true_rxn_smiles[j].split('>>')[-1]
        gen_rxn += '>>' + product + '\n'
        if product in gen_rxns.keys():
            gen_rxns[product].append(gen_rxn)
        else:
            gen_rxns[product] = [gen_rxn]

        all_valid[j] = float((all_valid_per_sample==all_mols_in_rcts) and all_mols_in_rcts>0)
        atleastone_valid[j] = float((all_valid_per_sample>0) and all_mols_in_rcts>0)

    return all_valid, atleastone_valid, gen_rxns

