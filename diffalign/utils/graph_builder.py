"""
Shared graph construction module for building reaction graphs from SMILES.

Consolidates the duplicated reaction-graph-building logic from:
- api/predict.py (smiles_to_dense_data)
- diffalign/datasets/supernode_dataset.py (Dataset.process)
- diffalign/utils/graph.py (rxn_smi_to_graph_data, get_graph_data_from_product_smi)
"""

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data

from diffalign.constants import MAX_ATOMS_RXN, DUMMY_RCT_NODE_TYPE, BOND_TYPES
from diffalign.utils.mol import mol_to_graph, rxn_to_graph_supernode


def _permute_rows(nodes, mask_atom_mapping, mol_assignment, edge_index):
    """Permute reactant-side rows in-place so the NN can only process topological information.

    Args:
        nodes: (n, d_x) node feature tensor (reactant + dummy section)
        mask_atom_mapping: (MAX_ATOMS_RXN,) tensor — only [:n] is permuted
        mol_assignment: (MAX_ATOMS_RXN,) tensor — only [:n] is permuted
        edge_index: (2, num_edges) tensor
    """
    rct_section_len = nodes.shape[0]
    perm = torch.randperm(rct_section_len)
    nodes[:] = nodes[perm]
    mask_atom_mapping[:rct_section_len] = mask_atom_mapping[:rct_section_len][perm]
    mol_assignment[:rct_section_len] = mol_assignment[:rct_section_len][perm]
    inv_perm = torch.zeros(rct_section_len, dtype=torch.long)
    inv_perm.scatter_(dim=0, index=perm, src=torch.arange(rct_section_len))
    edge_index[:] = inv_perm[edge_index]


def _permute_rows_product(g_nodes_prod, mask_atom_mapping, g_edge_index_prod, suno_idx):
    """Permute product-side rows in-place, keeping the supernode at position 0.

    Args:
        g_nodes_prod: (n_prod, d_x) product node features
        mask_atom_mapping: (MAX_ATOMS_RXN,) tensor
        g_edge_index_prod: (2, num_prod_edges) tensor
        suno_idx: offset of the supernode in the full graph
    """
    prod_indices = (suno_idx, suno_idx + g_nodes_prod.shape[0])
    perm = torch.cat([torch.tensor([0], dtype=torch.long), 1 + torch.randperm(g_nodes_prod.shape[0] - 1)], 0)
    inv_perm = torch.zeros(len(perm), dtype=torch.long)
    inv_perm.scatter_(dim=0, index=perm, src=torch.arange(len(perm)))
    g_nodes_prod[:] = g_nodes_prod[perm]
    # sn_and_prod_selection = (prod_selection | suno_idx == torch.arange(len(prod_selection)))
    mask_atom_mapping[prod_indices[0]:prod_indices[1]] = mask_atom_mapping[prod_indices[0]:prod_indices[1]][perm]
    # The following because g_edge_index_prod are counted with their offset in the final graph
    offset_padded_perm = torch.cat([torch.zeros(suno_idx, dtype=torch.long), suno_idx + perm])  # for debugging
    offset_padded_inv_perm = torch.cat([torch.zeros(suno_idx, dtype=torch.long), suno_idx + inv_perm])
    g_edge_index_prod[:] = offset_padded_inv_perm[g_edge_index_prod]


def build_rxn_graph(
    reactants,
    products,
    atom_types,
    bond_types,
    max_nodes_more_than_product,
    with_explicit_h=False,
    with_formal_charge=False,
    add_supernode_edges=False,
    canonicalize_molecule=True,
    permute_mols=False,
    scramble_atom_mapping=True,
    idx=0,
):
    """Build a reaction graph from reactant and product SMILES lists.

    This is the shared core logic previously duplicated across the codebase.

    Args:
        reactants: list of reactant SMILES strings
        products: list of product SMILES strings
        atom_types: list of atom type strings (e.g. from cfg.dataset.atom_types)
        bond_types: list of bond types (from diffalign.constants.BOND_TYPES or cfg)
        max_nodes_more_than_product: max extra dummy nodes beyond product node count
        with_explicit_h: whether to include explicit hydrogens
        with_formal_charge: whether to include formal charges
        add_supernode_edges: whether to add supernode edges in product graph
        canonicalize_molecule: whether to canonicalize molecules
        permute_mols: whether to randomly permute reactant and product node order
        scramble_atom_mapping: whether to scramble atom mapping indices (erase absolute info)
        idx: index for the Data object (used in dataset processing)

    Returns:
        tuple: (data, cannot_generate) where data is a torch_geometric Data object
               and cannot_generate is a bool indicating whether the reaction cannot
               be generated (nb_dummy_toadd < 0).
    """
    cannot_generate = False
    offset = 0

    # mask: (n), with n = nb of nodes
    mask_product_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool)  # only reactant nodes = True
    mask_reactant_and_sn = torch.zeros(MAX_ATOMS_RXN, dtype=torch.bool)  # only product nodes = True
    mask_sn = torch.ones(MAX_ATOMS_RXN, dtype=torch.bool)  # only sn = False
    mask_atom_mapping = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)
    mol_assignment = torch.zeros(MAX_ATOMS_RXN, dtype=torch.long)

    # preprocess: get total number of product nodes
    nb_product_nodes = sum([len(Chem.MolFromSmiles(p).GetAtoms()) for p in products])
    nb_rct_nodes = sum([len(Chem.MolFromSmiles(r).GetAtoms()) for r in reactants])

    # add dummy nodes: (nodes_in_product + max_added) - nodes_in_reactants
    nb_dummy_toadd = nb_product_nodes + max_nodes_more_than_product - nb_rct_nodes
    if nb_dummy_toadd < 0:
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
            canonicalize_molecule=canonicalize_molecule,
        )
        g_nodes_rct = torch.cat((g_nodes_rct, gi_nodes), dim=0) if j > 0 else gi_nodes  # already a tensor
        g_edge_index_rct = torch.cat((g_edge_index_rct, gi_edge_index), dim=1) if j > 0 else gi_edge_index
        g_edge_attr_rct = torch.cat((g_edge_attr_rct, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr
        atom_mapped_idx = (atom_map != 0).nonzero()
        mask_atom_mapping[atom_mapped_idx + offset] = atom_map[atom_mapped_idx]
        mol_assignment[offset:offset + gi_nodes.shape[0]] = j + 1
        offset += gi_nodes.shape[0]

    g_nodes_dummy = torch.ones(nb_dummy_toadd, dtype=torch.long) * atom_types.index(DUMMY_RCT_NODE_TYPE)
    g_nodes_dummy = F.one_hot(g_nodes_dummy, num_classes=len(atom_types)).float()
    # edges: fully connected to every node in the rct side with edge type 'none'
    g_edges_idx_dummy = torch.zeros([2, 0], dtype=torch.long)
    g_edges_attr_dummy = torch.zeros([0, len(bond_types)], dtype=torch.long)
    mask_product_and_sn[:g_nodes_rct.shape[0] + g_nodes_dummy.shape[0]] = True
    mol_assignment[offset:offset + g_nodes_dummy.shape[0]] = 0
    # mask_atom_mapping[offset:offset+g_nodes_dummy.shape[0]] = MAX_ATOMS_RXN
    offset += g_nodes_dummy.shape[0]
    g_nodes = torch.cat([g_nodes_rct, g_nodes_dummy], dim=0)
    g_edge_index = torch.cat([g_edge_index_rct, g_edges_idx_dummy], dim=1)
    g_edge_attr = torch.cat([g_edge_attr_rct, g_edges_attr_dummy], dim=0)

    # Permute the rows here to make sure that the NN can only process topological information
    if permute_mols:
        _permute_rows(g_nodes, mask_atom_mapping, mol_assignment, g_edge_index)

    supernodes_prods = []
    for j, p in enumerate(products):
        # NOTE: still need supernode for product to distinguish it from reactants
        gi_nodes, gi_edge_index, gi_edge_attr, atom_map = rxn_to_graph_supernode(
            mol=p,
            atom_types=atom_types,
            bond_types=bond_types,
            with_explicit_h=with_explicit_h,
            supernode_nb=offset + 1,
            with_formal_charge=with_formal_charge,
            add_supernode_edges=add_supernode_edges,
            get_atom_mapping=True,
            canonicalize_molecule=canonicalize_molecule,
        )
        g_nodes_prod = torch.cat((g_nodes_prod, gi_nodes), dim=0) if j > 0 else gi_nodes  # already a tensor
        g_edge_index_prod = torch.cat((g_edge_index_prod, gi_edge_index), dim=1) if j > 0 else gi_edge_index
        g_edge_attr_prod = torch.cat((g_edge_attr_prod, gi_edge_attr), dim=0) if j > 0 else gi_edge_attr
        atom_mapped_idx = (atom_map != 0).nonzero()
        mask_atom_mapping[atom_mapped_idx + offset] = atom_map[atom_mapped_idx]
        mask_reactant_and_sn[offset:gi_nodes.shape[0] + offset] = True
        mol_assignment[offset] = 0  # supernode does not belong to any molecule
        suno_idx = offset  # there should only be one supernode and one loop through the products
        mol_assignment[offset + 1:offset + 1 + gi_nodes.shape[0]] = len(reactants) + j + 1  # TODO: Is there one too many assigned as a product atom here?
        mask_sn[offset] = False
        mask_reactant_and_sn[offset] = False
        # supernode is always in the first position
        si = 0  # gi_edge_index[0][0].item()
        supernodes_prods.append(si)
        offset += gi_nodes.shape[0]

    # Keep the supernode intact here, others are permuted
    if permute_mols:
        _permute_rows_product(g_nodes_prod, mask_atom_mapping, g_edge_index_prod, suno_idx)

    # concatenate all types of nodes and edges
    g_nodes = torch.cat([g_nodes, g_nodes_prod], dim=0)
    g_edge_index = torch.cat([g_edge_index, g_edge_index_prod], dim=1)
    g_edge_attr = torch.cat([g_edge_attr, g_edge_attr_prod], dim=0)

    y = torch.zeros((1, 0), dtype=torch.float)

    # trim masks => one element per node in the rxn graph
    mask_product_and_sn = mask_product_and_sn[:g_nodes.shape[0]]  # only reactant nodes = True
    mask_reactant_and_sn = mask_reactant_and_sn[:g_nodes.shape[0]]
    mask_sn = mask_sn[:g_nodes.shape[0]]
    mask_atom_mapping = mask_atom_mapping[:g_nodes.shape[0]]
    mol_assignment = mol_assignment[:g_nodes.shape[0]]

    assert mask_atom_mapping.shape[0] == g_nodes.shape[0] and mask_sn.shape[0] == g_nodes.shape[0] and \
        mask_reactant_and_sn.shape[0] == g_nodes.shape[0] and mask_product_and_sn.shape[0] == g_nodes.shape[0] and \
        mol_assignment.shape[0] == g_nodes.shape[0]

    # erase atom mapping absolute information for good.
    if scramble_atom_mapping:
        perm = torch.arange(mask_atom_mapping.max().item() + 1)[1:]
        perm = perm[torch.randperm(len(perm))]
        perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])
        mask_atom_mapping = perm[mask_atom_mapping]

    data = Data(
        x=g_nodes, edge_index=g_edge_index,
        edge_attr=g_edge_attr, y=y, idx=idx,
        mask_sn=mask_sn, mask_reactant_and_sn=mask_reactant_and_sn,
        mask_product_and_sn=mask_product_and_sn, mask_atom_mapping=mask_atom_mapping,
        mol_assignment=mol_assignment, cannot_generate=cannot_generate,
    )

    return data, cannot_generate
