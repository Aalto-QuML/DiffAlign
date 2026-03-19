"""
Unit tests for SMILES parsing and graph conversion utilities.

Tests mol_to_graph, mol_from_graph, and rxn_to_graph_supernode from
diffalign/utils/mol.py, using atom_types and bond_types from diffalign/constants.py.
"""
import os, sys, pathlib
PROJECT_ROOT = pathlib.Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import unittest

# Skip entire module if rdkit, torch, or system libs are unavailable.
# Import order matters: graph must be imported before mol to resolve a
# circular dependency (mol -> graph -> graph_builder -> mol).
try:
    import torch
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT
    from diffalign.utils import graph as _graph  # resolves circular import
    from diffalign.utils.mol import mol_to_graph, mol_from_graph, rxn_to_graph_supernode
    from diffalign.constants import BOND_TYPES
    HAS_DEPS = True
except (ImportError, OSError):
    HAS_DEPS = False

# Shared atom types list used by all test classes
_ATOM_TYPES = [
    'none', 'C', 'O', 'H', 'N', 'S', 'F', 'Cl', 'Br',
    'I', 'P', 'Si', 'B', 'Se', 'Sn',
    'C-1', 'N+1', 'N-1', 'O-1', 'O+1', 'S-1', 'S+1',
    'Se+1', 'Se-1', 'Te', 'Ge',
    'U', 'SuNo', 'mask',
]


@unittest.skipUnless(HAS_DEPS, "rdkit/torch/X11 libs not available")
class TestMolToGraph(unittest.TestCase):
    """Test mol_to_graph produces correct node and edge tensors."""

    @classmethod
    def setUpClass(cls):
        cls.bond_types = BOND_TYPES
        cls.atom_types = _ATOM_TYPES

    def test_ethanol_node_count(self):
        """mol_to_graph('CCO') should produce nodes for C, C, O."""
        nodes, edge_idx, edge_attr = mol_to_graph(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            offset=0, with_explicit_h=False, with_formal_charge=True,
            canonicalize_molecule=True,
        )
        self.assertEqual(nodes.shape[0], 3, f"Expected 3 nodes, got {nodes.shape[0]}")
        self.assertEqual(nodes.shape[1], len(self.atom_types))

    def test_ethanol_node_types(self):
        """Verify node types are C, C, O for CCO."""
        nodes, _, _ = mol_to_graph(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            offset=0, with_explicit_h=False, with_formal_charge=True,
            canonicalize_molecule=True,
        )
        node_classes = nodes.argmax(dim=-1).tolist()
        c_idx = self.atom_types.index('C')
        o_idx = self.atom_types.index('O')
        self.assertEqual(node_classes.count(c_idx), 2)
        self.assertEqual(node_classes.count(o_idx), 1)

    def test_ethanol_edges(self):
        """CCO should have 2 bonds (4 directed edges), all SINGLE."""
        _, edge_idx, edge_attr = mol_to_graph(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            offset=0, with_explicit_h=False, with_formal_charge=True,
            canonicalize_molecule=True,
        )
        self.assertEqual(edge_idx.shape[0], 2)
        self.assertEqual(edge_idx.shape[1], 4)
        single_idx = self.bond_types.index(BT.SINGLE)
        for i in range(edge_attr.shape[0]):
            self.assertEqual(edge_attr[i].argmax().item(), single_idx)

    def test_one_hot_encoding(self):
        """Each node tensor row should be a valid one-hot vector."""
        nodes, _, _ = mol_to_graph(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            offset=0, with_explicit_h=False, with_formal_charge=True,
            canonicalize_molecule=True,
        )
        for i in range(nodes.shape[0]):
            self.assertAlmostEqual(nodes[i].sum().item(), 1.0, places=5)

    def test_double_bond_molecule(self):
        """Ethene (C=C) should have a double bond."""
        _, edge_idx, edge_attr = mol_to_graph(
            mol="C=C", atom_types=self.atom_types, bond_types=self.bond_types,
            offset=0, with_explicit_h=False, with_formal_charge=True,
            canonicalize_molecule=True,
        )
        double_idx = self.bond_types.index(BT.DOUBLE)
        bond_types_found = edge_attr.argmax(dim=-1).tolist()
        self.assertIn(double_idx, bond_types_found)


@unittest.skipUnless(HAS_DEPS, "rdkit/torch/X11 libs not available")
class TestMolFromGraph(unittest.TestCase):
    """Test mol_from_graph can reconstruct a molecule from node/edge tensors."""

    @classmethod
    def setUpClass(cls):
        cls.bond_types = BOND_TYPES
        cls.atom_types = _ATOM_TYPES

    def _roundtrip(self, smiles):
        """Convert SMILES -> graph -> RDKit mol -> SMILES and check validity."""
        nodes, edge_idx, edge_attr = mol_to_graph(
            mol=smiles, atom_types=self.atom_types, bond_types=self.bond_types,
            offset=0, with_explicit_h=False, with_formal_charge=True,
            canonicalize_molecule=True,
        )
        n_atoms = nodes.shape[0]
        adj = torch.zeros(n_atoms, n_atoms, dtype=torch.long)
        for i in range(edge_idx.shape[1]):
            src, dst = edge_idx[0, i].item(), edge_idx[1, i].item()
            bond_class = edge_attr[i].argmax().item()
            adj[src, dst] = bond_class

        node_list = nodes.argmax(dim=-1)
        mol = mol_from_graph(
            node_list=node_list, adjacency_matrix=adj,
            atom_types=self.atom_types, bond_types=self.bond_types,
        )
        return mol

    def test_ethanol_roundtrip(self):
        mol = self._roundtrip("CCO")
        self.assertIsNotNone(mol)
        smi = Chem.MolToSmiles(mol)
        self.assertIsInstance(smi, str)
        self.assertGreater(len(smi), 0)

    def test_benzene_roundtrip(self):
        """Benzene (c1ccccc1) should survive a round-trip."""
        mol = self._roundtrip("c1ccccc1")
        self.assertIsNotNone(mol)
        smi = Chem.MolToSmiles(mol)
        self.assertIsInstance(smi, str)
        self.assertGreater(len(smi), 0)

    def test_atom_count_preserved(self):
        """Number of heavy atoms should be preserved in round-trip."""
        smiles = "CCO"
        original_mol = Chem.MolFromSmiles(smiles)
        original_count = original_mol.GetNumHeavyAtoms()
        reconstructed = self._roundtrip(smiles)
        reconstructed_count = reconstructed.GetNumHeavyAtoms()
        self.assertEqual(original_count, reconstructed_count)


@unittest.skipUnless(HAS_DEPS, "rdkit/torch/X11 libs not available")
class TestRxnToGraphSupernode(unittest.TestCase):
    """Test rxn_to_graph_supernode produces a graph with a supernode."""

    @classmethod
    def setUpClass(cls):
        cls.bond_types = BOND_TYPES
        cls.atom_types = _ATOM_TYPES

    def test_supernode_present(self):
        """The first node returned should be a SuNo."""
        suno_idx = self.atom_types.index('SuNo')
        nodes, edge_idx, edge_attr = rxn_to_graph_supernode(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            supernode_nb=1, with_explicit_h=False, with_formal_charge=True,
            add_supernode_edges=True, canonicalize_molecule=True,
        )
        self.assertEqual(nodes[0].argmax().item(), suno_idx)

    def test_extra_node_for_supernode(self):
        """Total nodes = original atoms + 1 (supernode)."""
        nodes_plain, _, _ = mol_to_graph(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            offset=0, with_explicit_h=False, with_formal_charge=True,
            canonicalize_molecule=True,
        )
        nodes_sn, _, _ = rxn_to_graph_supernode(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            supernode_nb=1, with_explicit_h=False, with_formal_charge=True,
            add_supernode_edges=True, canonicalize_molecule=True,
        )
        self.assertEqual(nodes_sn.shape[0], nodes_plain.shape[0] + 1)

    def test_supernode_edges(self):
        """When add_supernode_edges=True, there should be 'mol' type edges."""
        mol_edge_idx = self.bond_types.index('mol')
        _, edge_idx, edge_attr = rxn_to_graph_supernode(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            supernode_nb=1, with_explicit_h=False, with_formal_charge=True,
            add_supernode_edges=True, canonicalize_molecule=True,
        )
        edge_classes = edge_attr.argmax(dim=-1).tolist()
        self.assertIn(mol_edge_idx, edge_classes)

    def test_no_supernode_edges(self):
        """When add_supernode_edges=False, there should be no 'mol' type edges."""
        mol_edge_idx = self.bond_types.index('mol')
        nodes, edge_idx, edge_attr = rxn_to_graph_supernode(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            supernode_nb=1, with_explicit_h=False, with_formal_charge=True,
            add_supernode_edges=False, canonicalize_molecule=True,
        )
        if edge_attr.shape[0] > 0:
            edge_classes = edge_attr.argmax(dim=-1).tolist()
            self.assertNotIn(mol_edge_idx, edge_classes)

    def test_atom_mapping_returned(self):
        """get_atom_mapping=True should return a 4-tuple including atom_map."""
        result = rxn_to_graph_supernode(
            mol="CCO", atom_types=self.atom_types, bond_types=self.bond_types,
            supernode_nb=1, with_explicit_h=False, with_formal_charge=True,
            add_supernode_edges=True, get_atom_mapping=True,
            canonicalize_molecule=True,
        )
        self.assertEqual(len(result), 4)
        nodes, edge_idx, edge_attr, atom_map = result
        self.assertEqual(atom_map.shape[0], nodes.shape[0])


if __name__ == "__main__":
    unittest.main()
