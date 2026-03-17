
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

# ── CPU inference optimizations ───────────────────────────────────────────────
torch.set_num_threads(4)  # fewer threads = less contention on shared cloud vCPUs
torch.backends.mkldnn.enabled = True
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.utils.setup import dotdict, load_weights
from src.utils.mol import rxn_to_graph_supernode, mol_to_graph, get_cano_smiles_from_dense
from src.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
from src.datasets.abstract_dataset import DistributionNodes
from src.utils.graph import to_dense, get_index_from_states
from src.diffusion.noise_schedule import AbsorbingStateTransitionMaskNoEdge

log = logging.getLogger(__name__)

MAX_ATOMS_RXN = 300
DUMMY_RCT_NODE_TYPE = 'U'
PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
BOND_TYPES = ['none', BT.SINGLE, BT.DOUBLE, BT.TRIPLE, 'mol', 'within', 'across']
BOND_ORDERS = [0, 1, 2, 3, 0, 0, 0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model caching ──────────────────────────────────────────────────────────
_cached_model = None
_cached_cfg = None


def get_model_and_cfg():
    """Load model and config once, return cached copies on subsequent calls."""
    global _cached_model, _cached_cfg
    if _cached_model is not None:
        return _cached_model, _cached_cfg

    log.info("Loading DiffAlign model (one-time)...")
    GlobalHydra.instance().clear()
    with initialize(config_path="../configs"):
        cfg = compose(
            config_name="default",
            overrides=["+experiment=align_absorbing"]
        )
    # Default eval steps — cheap to rebuild later if caller wants different value
    cfg.diffusion.diffusion_steps_eval = 10

    model = load_model(cfg)
    model.eval()
    log.info("DiffAlign model loaded and cached.")

    _cached_model = model
    _cached_cfg = cfg
    return _cached_model, _cached_cfg


# ── Transition-model cache (keyed by diffusion_steps_eval) ─────────────────
_transition_cache = {}


def _get_or_build_transition_model_eval(model, cfg, diffusion_steps_eval):
    """Rebuild transition_model_eval for a different step count. Cheap (just matrices)."""
    if diffusion_steps_eval in _transition_cache:
        return _transition_cache[diffusion_steps_eval]

    abs_state_position_x = model.dataset_info.atom_decoder.index('Au')
    abs_state_position_e = 0
    node_idx_to_mask, edge_idx_to_mask = get_index_from_states(
        atom_decoder=model.dataset_info.atom_decoder,
        bond_decoder=model.dataset_info.bond_decoder,
        node_states_to_mask=cfg.diffusion.node_states_to_mask,
        edge_states_to_mask=cfg.diffusion.edge_states_to_mask,
        device=device,
    )

    tm = AbsorbingStateTransitionMaskNoEdge(
        x_classes=model.Xdim_output,
        e_classes=model.Edim_output,
        y_classes=model.ydim_output,
        timesteps=diffusion_steps_eval,
        diffuse_edges=cfg.diffusion.diffuse_edges,
        abs_state_position_x=abs_state_position_x,
        abs_state_position_e=abs_state_position_e,
        node_idx_to_mask=node_idx_to_mask,
        edge_idx_to_mask=edge_idx_to_mask,
    )
    _transition_cache[diffusion_steps_eval] = tm
    return tm


def smiles_to_dense_data(
    product_smiles: str,
    max_nodes_more_than_product: int,
    with_explicit_h: bool,
    with_formal_charge: bool,
    canonicalize_molecule: bool,
    permute_mols: bool,
    add_supernode_edges: bool,
    atom_types: List[str],
    bond_types: List[BT],
):
    cannot_generate = False
    # Add atom mapping numbers so positional encodings work correctly.
    # Without mappings, all atoms get identical encodings (catastrophically OOD).
    mol = Chem.MolFromSmiles(product_smiles)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    product_smiles_mapped = Chem.MolToSmiles(mol)
    rxn_smiles = f"{product_smiles_mapped}>>{product_smiles_mapped}"
    reactants = [r for r in rxn_smiles.split('>>')[0].split('.')]
    products = [p for p in rxn_smiles.split('>>')[1].split('.')]
    offset = 0 
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
    nb_dummy_toadd = nb_product_nodes + max_nodes_more_than_product - nb_rct_nodes
    if nb_dummy_toadd<0:
        # cut the rct nodes
        nb_dummy_toadd = 0
        cannot_generate = True
    for j, r in enumerate(reactants):
        # NOTE: no supernodes for reactants (treated as one block)
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = mol_to_graph(
            mol=r, 
            atom_types=atom_types, 
            bond_types=bond_types,
            with_explicit_h=with_explicit_h,
            with_formal_charge=with_formal_charge,
            offset=offset, 
            get_atom_mapping=True,
            canonicalize_molecule=canonicalize_molecule
        )
        g_nodes_rct = torch.cat((g_nodes_rct, gi_nodes), dim=0) if j > 0 else gi_nodes # already a tensor
        g_edge_index_rct = torch.cat((g_edge_index_rct, gi_edge_index), dim=1) if j > 0 else gi_edge_index
        g_edge_attr_rct = torch.cat((g_edge_attr_rct, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr
        atom_mapped_idx = (atom_map!=0).nonzero()
        mask_atom_mapping[atom_mapped_idx+offset] = atom_map[atom_mapped_idx]
        mol_assignment[offset:offset+gi_nodes.shape[0]] = j+1
        offset += gi_nodes.shape[0] 

    g_nodes_dummy = torch.ones(nb_dummy_toadd, dtype=torch.long) * atom_types.index(DUMMY_RCT_NODE_TYPE)
    g_nodes_dummy = F.one_hot(g_nodes_dummy, num_classes=len(atom_types)).float()
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

    if permute_mols:
        permute_rows(g_nodes, mask_atom_mapping, mol_assignment, g_edge_index)

    supernodes_prods = []
    for j, p in enumerate(products):
        # NOTE: still need supernode for product to distinguish it from reactants
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = rxn_to_graph_supernode(
            mol=p, 
            atom_types=atom_types, 
            bond_types=bond_types,
            with_explicit_h=with_explicit_h, 
            supernode_nb=offset+1,
            with_formal_charge=with_formal_charge,
            add_supernode_edges=False,  # must match dataset process() which hardcodes False
            get_atom_mapping=True,
            canonicalize_molecule=canonicalize_molecule
        )
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
    if permute_mols:
        permute_rows_product(g_nodes_prod, mask_atom_mapping, g_edge_index_prod)
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
    assert mask_atom_mapping.shape[0]==g_nodes.shape[0] and mask_sn.shape[0]==g_nodes.shape[0] and \
            mask_reactant_and_sn.shape[0]==g_nodes.shape[0] and mask_product_and_sn.shape[0]==g_nodes.shape[0] and \
            mol_assignment.shape[0]==g_nodes.shape[0]
    # erase atom mapping absolute information for good. 
    perm = torch.arange(mask_atom_mapping.max().item()+1)[1:]
    perm = perm[torch.randperm(len(perm))]
    perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])
    mask_atom_mapping = perm[mask_atom_mapping]

    data = Data(x=g_nodes, edge_index=g_edge_index,
                    edge_attr=g_edge_attr, y=y, idx=0,
                    mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn,
                    mask_product_and_sn=mask_product_and_sn, mask_atom_mapping=mask_atom_mapping,
                    mol_assignment=mol_assignment, cannot_generate=cannot_generate)
    # Wrap in a Batch to match the working pipeline (to_dense expects data.batch)
    batch = Batch.from_data_list([data])
    dense_data = to_dense(batch)
    return dense_data
  
def load_model(cfg):
    device_count = 1
    # TODO: get rid of dataset_infos and maybe model_kwargs
    atom_types = cfg.dataset.atom_types
    allowed_bonds = cfg.dataset.allowed_bonds
    valencies = [0] + list(abs[0] for atom_type, abs in allowed_bonds.items() if atom_type in atom_types) + [0]
    periodic_table = Chem.rdchem.GetPeriodicTable()
    atom_weights = [0] + [periodic_table.GetAtomicWeight(re.split(r'\+|\-', atom_type)[0]) for atom_type in atom_types[1:-1]] + [0] # discard charge
    atom_weights = {atom_type: weight for atom_type, weight in zip(atom_types, atom_weights)}

    # Look in checkpoints/ first, fall back to data processed dir
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    data_dir = os.path.join(PROJECT_ROOT, cfg.dataset.datadir + '-' + str(cfg.dataset.dataset_nb), 'processed')
    def _find(name):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
        return None
    node_count_path = _find('n_counts.txt')
    atom_type_path = _find('atom_types.txt')
    edge_type_path = _find('edge_types.txt')
    atom_type_unnorm_path = _find('atom_types_unnorm_mol.txt')
    edge_type_unnorm_path = _find('edge_types_unnorm_mol.txt')
    paths_exist = all(p is not None for p in [node_count_path, atom_type_path, edge_type_path,
                                               atom_type_unnorm_path, edge_type_unnorm_path])

    if paths_exist:
        # use the same distributions for all subsets of the dataset
        n_nodes = torch.from_numpy(np.loadtxt(node_count_path))
        node_types = torch.from_numpy(np.loadtxt(atom_type_path))
        edge_types = torch.from_numpy(np.loadtxt(edge_type_path))
        node_types_unnormalized = torch.from_numpy(np.loadtxt(atom_type_unnorm_path)).long()
        edge_types_unnormalized = torch.from_numpy(np.loadtxt(edge_type_unnorm_path)).long()
    
    dataset_infos = dotdict({
        'node_types_unnormalized': node_types_unnormalized,
        'edge_types_unnormalized': edge_types_unnormalized,
        'num_classes': len(node_types),
        'max_n_nodes': len(n_nodes) - 1,
        'nodes_dist': DistributionNodes(n_nodes),
        'atom_decoder': cfg.dataset.atom_types,
        'bond_decoder': BOND_TYPES,
        'valencies': valencies,
        'atom_weights': atom_weights,
        'max_weight': cfg.dataset.max_atom_weight,
        'bond_orders': BOND_ORDERS,
        'remove_h': cfg.dataset.remove_h,
        'input_dims': dotdict({
            'X': len(cfg.dataset.atom_types),
            'E': len(BOND_TYPES),
            'y': 1
        }),
        'output_dims': dotdict({
            'X': len(cfg.dataset.atom_types),
            'E': len(BOND_TYPES),
            'y': 0
        })
    })
    model_kwargs={
        'dataset_infos': dataset_infos, 
        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
        'use_data_parallel': device_count>1
    }
    model = DiscreteDenoisingDiffusionRxn(cfg=cfg, **model_kwargs)
    model = model.to(device)
    checkpoint_file = os.path.join(PROJECT_ROOT, 'checkpoints', f'epoch760.pt')
    print(f'loading model from {checkpoint_file}')
    state_dict = torch.load(checkpoint_file, map_location=device)['model_state_dict']
    model = load_weights(model, state_dict, device_count=1)
    return model

def predict_precursors_from_diffalign(
    product_smiles: str,
    n_precursors: int = 5,
    diffusion_steps: int = 10
):
    model, cfg = get_model_and_cfg()

    # Validate diffusion_steps divides T=500
    T = cfg.diffusion.diffusion_steps  # 500
    assert T % diffusion_steps == 0, (
        f"diffusion_steps={diffusion_steps} must divide T={T}. "
        f"Valid values: {[d for d in range(1, T+1) if T % d == 0]}"
    )

    # Swap transition_model_eval if steps differ from current
    if diffusion_steps != cfg.diffusion.diffusion_steps_eval:
        model.transition_model_eval = _get_or_build_transition_model_eval(
            model, cfg, diffusion_steps
        )
        cfg.diffusion.diffusion_steps_eval = diffusion_steps
    log.info(f"Running inference with diffusion_steps_eval={diffusion_steps}")
    # create dense data
    dense_data = smiles_to_dense_data(
        max_nodes_more_than_product=cfg.dataset.nb_rct_dummy_nodes,
        product_smiles=product_smiles,
        atom_types=cfg.dataset.atom_types,
        bond_types=BOND_TYPES,
        with_explicit_h=cfg.dataset.with_explicit_h,
        with_formal_charge=cfg.dataset.with_formal_charge,
        add_supernode_edges=cfg.dataset.add_supernode_edges,
        canonicalize_molecule=cfg.dataset.canonicalize_molecule,
        permute_mols=cfg.dataset.permute_mols
    )
    dense_data = dense_data.to_device(device)
    print('done creating dense data.')
    # sample
    with torch.inference_mode():
        final_samples = model.sample_for_condition(
            dense_data=dense_data,
            n_samples=n_precursors,
            inpaint_node_idx=None,
            inpaint_edge_idx=None,
            device=device
        )
    print('done sampling.')

    # Convert dense samples to reaction SMILES strings
    all_rxn_str = get_cano_smiles_from_dense(
        final_samples.X, final_samples.E,
        cfg.dataset.atom_types, BOND_TYPES
    )

    # Extract reactant SMILES (left side of >>), deduplicate, and score by frequency
    reactant_counts = {}
    for rxn_str in all_rxn_str:
        if '>>' in rxn_str:
            reactants = rxn_str.split('>>')[0]
        else:
            reactants = rxn_str
        if reactants:
            reactant_counts[reactants] = reactant_counts.get(reactants, 0) + 1

    total = len(all_rxn_str) if all_rxn_str else 1
    results = []
    for precursors, count in reactant_counts.items():
        results.append({'precursors': precursors, 'score': count / total})
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:n_precursors]


def predict_precursors(
    product_smiles: str,
    n_precursors: int = 5,
    diffusion_steps: int = 100,
    temperature: float = 1.0,
    beam_size: int = 10,
) -> List[Dict[str, Any]]:
    """Predict retrosynthesis precursors for a given product SMILES."""
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return []

    results = predict_precursors_from_diffalign(
        product_smiles=product_smiles,
        n_precursors=n_precursors,
        diffusion_steps=diffusion_steps,
    )

    # Validate SMILES in results
    validated = []
    for result in results:
        all_valid = True
        for smi in result['precursors'].split('.'):
            if Chem.MolFromSmiles(smi) is None:
                all_valid = False
                break
        if all_valid:
            validated.append(result)
    return validated


# ── Eager load at import time (runs once, before first request) ────────────
get_model_and_cfg()