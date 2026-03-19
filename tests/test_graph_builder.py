"""
Regression test for graph_builder deduplication.

Verifies that build_rxn_graph() produces valid graph Data objects
with correct shapes, masks, and structure for a set of test molecules.
"""
import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from diffalign.utils.graph_builder import build_rxn_graph
from diffalign.constants import BOND_TYPES

# Test molecules: varying sizes and complexities
TEST_RXNS = [
    # (reactants, products, description)
    (["[CH3:1][OH:2]", "[CH3:3][Cl:4]"], ["[CH3:1][O:2][CH3:3]"], "simple Williamson ether"),
    (["[CH3:1][C:2](=[O:3])[OH:4]"], ["[CH3:1][C:2](=[O:3])[OH:4]"], "identity (product=reactant)"),
    (["[NH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"], ["[NH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"], "aniline identity"),
    (["[CH3:1][NH:2][CH3:3]", "[CH3:4][C:5](=[O:6])[Cl:7]"], ["[CH3:1][N:2]([CH3:3])[C:5](=[O:6])[CH3:4]"], "amide formation"),
]

ATOM_TYPES = ['none', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'SuNo', 'U']
MAX_DUMMY = 35


def test_build_rxn_graph_basic_shapes():
    """Test that build_rxn_graph returns Data objects with correct basic shapes."""
    for reactants, products, desc in TEST_RXNS:
        data, cannot_generate = build_rxn_graph(
            reactants=reactants,
            products=products,
            atom_types=ATOM_TYPES,
            bond_types=BOND_TYPES,
            max_nodes_more_than_product=MAX_DUMMY,
            with_explicit_h=False,
            with_formal_charge=False,
            add_supernode_edges=False,
            canonicalize_molecule=True,
            permute_mols=False,
            scramble_atom_mapping=False,
        )
        n_nodes = data.x.shape[0]
        assert data.x.ndim == 2, f"{desc}: X should be 2D"
        assert data.x.shape[1] == len(ATOM_TYPES), f"{desc}: X feature dim should match atom_types"
        assert data.edge_attr.shape[1] == len(BOND_TYPES), f"{desc}: E feature dim should match bond_types"
        assert data.mask_sn.shape[0] == n_nodes, f"{desc}: mask_sn length mismatch"
        assert data.mask_product_and_sn.shape[0] == n_nodes, f"{desc}: mask_product_and_sn length mismatch"
        assert data.mask_reactant_and_sn.shape[0] == n_nodes, f"{desc}: mask_reactant_and_sn length mismatch"
        assert data.mask_atom_mapping.shape[0] == n_nodes, f"{desc}: mask_atom_mapping length mismatch"
        assert data.mol_assignment.shape[0] == n_nodes, f"{desc}: mol_assignment length mismatch"


def test_build_rxn_graph_deterministic():
    """Test that without permutation/scrambling, output is deterministic."""
    for reactants, products, desc in TEST_RXNS:
        kwargs = dict(
            reactants=reactants,
            products=products,
            atom_types=ATOM_TYPES,
            bond_types=BOND_TYPES,
            max_nodes_more_than_product=MAX_DUMMY,
            with_explicit_h=False,
            with_formal_charge=False,
            add_supernode_edges=False,
            canonicalize_molecule=True,
            permute_mols=False,
            scramble_atom_mapping=False,
        )
        data1, _ = build_rxn_graph(**kwargs)
        data2, _ = build_rxn_graph(**kwargs)
        assert torch.equal(data1.x, data2.x), f"{desc}: X not deterministic"
        assert torch.equal(data1.edge_attr, data2.edge_attr), f"{desc}: edge_attr not deterministic"
        assert torch.equal(data1.mask_atom_mapping, data2.mask_atom_mapping), f"{desc}: mask_atom_mapping not deterministic"


def test_build_rxn_graph_supernode_present():
    """Test that product side has exactly one supernode (SuNo) per product molecule."""
    for reactants, products, desc in TEST_RXNS:
        data, _ = build_rxn_graph(
            reactants=reactants,
            products=products,
            atom_types=ATOM_TYPES,
            bond_types=BOND_TYPES,
            max_nodes_more_than_product=MAX_DUMMY,
            with_explicit_h=False,
            with_formal_charge=False,
            add_supernode_edges=False,
            canonicalize_molecule=True,
            permute_mols=False,
            scramble_atom_mapping=False,
        )
        suno_idx = ATOM_TYPES.index("SuNo")
        node_types = data.x.argmax(dim=-1)
        n_sunos = (node_types == suno_idx).sum().item()
        assert n_sunos == len(products), f"{desc}: expected {len(products)} supernodes, got {n_sunos}"


def test_build_rxn_graph_masks_consistent():
    """Test that masks are mutually consistent."""
    for reactants, products, desc in TEST_RXNS:
        data, _ = build_rxn_graph(
            reactants=reactants,
            products=products,
            atom_types=ATOM_TYPES,
            bond_types=BOND_TYPES,
            max_nodes_more_than_product=MAX_DUMMY,
            with_explicit_h=False,
            with_formal_charge=False,
            add_supernode_edges=False,
            canonicalize_molecule=True,
            permute_mols=False,
            scramble_atom_mapping=False,
        )
        # mask_sn should be False only at supernode positions
        suno_idx = ATOM_TYPES.index("SuNo")
        node_types = data.x.argmax(dim=-1)
        suno_positions = (node_types == suno_idx)
        assert torch.equal(~data.mask_sn, suno_positions), f"{desc}: mask_sn inconsistent with SuNo positions"


def test_build_rxn_graph_cannot_generate():
    """Test that cannot_generate is set when reactants have more nodes than product + max_dummy."""
    # Use a product with few atoms and reactants with many, with max_dummy=0
    reactants = ["[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][OH:6]"]
    products = ["[CH3:1][OH:2]"]
    data, cannot_generate = build_rxn_graph(
        reactants=reactants,
        products=products,
        atom_types=ATOM_TYPES,
        bond_types=BOND_TYPES,
        max_nodes_more_than_product=0,
        with_explicit_h=False,
        with_formal_charge=False,
        add_supernode_edges=False,
        canonicalize_molecule=True,
        permute_mols=False,
        scramble_atom_mapping=False,
    )
    assert cannot_generate is True, "Should flag cannot_generate when reactants > product + max_dummy"


if __name__ == "__main__":
    test_build_rxn_graph_basic_shapes()
    print("PASS: test_build_rxn_graph_basic_shapes")
    test_build_rxn_graph_deterministic()
    print("PASS: test_build_rxn_graph_deterministic")
    test_build_rxn_graph_supernode_present()
    print("PASS: test_build_rxn_graph_supernode_present")
    test_build_rxn_graph_masks_consistent()
    print("PASS: test_build_rxn_graph_masks_consistent")
    test_build_rxn_graph_cannot_generate()
    print("PASS: test_build_rxn_graph_cannot_generate")
    print("\nAll graph builder regression tests passed!")
