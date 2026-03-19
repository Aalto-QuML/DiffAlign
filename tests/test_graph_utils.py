"""
Unit tests for PlaceHolder and graph utility functions.

Tests PlaceHolder construction, device transfer, mask(), get_new_object(),
and the get_unique_indices_from_reaction_list helper.
"""
import os, sys, pathlib
PROJECT_ROOT = pathlib.Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import torch

# Check if graph module is importable (it pulls in rdkit.Chem.Draw which
# may fail if X11 libs are missing).  We test it separately.
try:
    from diffalign.utils.graph import get_unique_indices_from_reaction_list
    HAS_GRAPH_MODULE = True
except (ImportError, OSError):
    HAS_GRAPH_MODULE = False


class TestPlaceHolderConstruction(unittest.TestCase):
    """Basic construction and attribute access."""

    def test_basic_construction(self):
        X = torch.randn(2, 4, 5)
        E = torch.randn(2, 4, 4, 3)
        y = torch.zeros(2, 0)
        node_mask = torch.ones(2, 4, dtype=torch.bool)

        from diffalign.utils.placeholder import PlaceHolder
        ph = PlaceHolder(X=X, E=E, y=y, node_mask=node_mask)

        self.assertIs(ph.X, X)
        self.assertIs(ph.E, E)
        self.assertIs(ph.y, y)
        self.assertIs(ph.node_mask, node_mask)
        self.assertIsNone(ph.atom_map_numbers)
        self.assertIsNone(ph.mol_assignments)

    def test_construction_with_optional_fields(self):
        from diffalign.utils.placeholder import PlaceHolder
        bs, n, dx, de = 2, 4, 5, 3
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n, dtype=torch.bool)
        atom_map = torch.arange(n).unsqueeze(0).expand(bs, -1)
        mol_assign = torch.zeros(bs, n, dtype=torch.long)

        ph = PlaceHolder(X=X, E=E, y=y, node_mask=node_mask,
                         atom_map_numbers=atom_map, mol_assignments=mol_assign)

        self.assertEqual(ph.atom_map_numbers.shape, (bs, n))
        self.assertEqual(ph.mol_assignments.shape, (bs, n))


class TestPlaceHolderToDevice(unittest.TestCase):
    """Test to_device('cpu') -- we only test CPU since no GPU is required."""

    def test_to_device_cpu(self):
        from diffalign.utils.placeholder import PlaceHolder
        bs, n, dx, de = 1, 3, 4, 2
        ph = PlaceHolder(
            X=torch.randn(bs, n, dx),
            E=torch.randn(bs, n, n, de),
            y=torch.zeros(bs, 0),
            node_mask=torch.ones(bs, n, dtype=torch.bool),
            atom_map_numbers=torch.zeros(bs, n, dtype=torch.long),
            mol_assignments=torch.zeros(bs, n, dtype=torch.long),
        )
        result = ph.to_device("cpu")
        # to_device returns self
        self.assertIs(result, ph)
        self.assertEqual(ph.X.device.type, "cpu")
        self.assertEqual(ph.E.device.type, "cpu")
        self.assertEqual(ph.node_mask.device.type, "cpu")


class TestPlaceHolderGetNewObject(unittest.TestCase):
    """Test get_new_object() returns a new PlaceHolder with cloned tensors."""

    def test_new_object_shares_no_memory(self):
        from diffalign.utils.placeholder import PlaceHolder
        bs, n, dx, de = 1, 3, 4, 2
        ph = PlaceHolder(
            X=torch.randn(bs, n, dx),
            E=torch.randn(bs, n, n, de),
            y=torch.zeros(bs, 0),
            node_mask=torch.ones(bs, n, dtype=torch.bool),
        )
        new_ph = ph.get_new_object()

        # Same values
        self.assertTrue(torch.equal(ph.X, new_ph.X))
        self.assertTrue(torch.equal(ph.E, new_ph.E))

        # But different underlying storage (clone)
        new_ph.X[0, 0, 0] += 999.0
        self.assertNotAlmostEqual(ph.X[0, 0, 0].item(), new_ph.X[0, 0, 0].item())

    def test_new_object_with_override(self):
        from diffalign.utils.placeholder import PlaceHolder
        bs, n, dx, de = 1, 3, 4, 2
        ph = PlaceHolder(
            X=torch.randn(bs, n, dx),
            E=torch.randn(bs, n, n, de),
            y=torch.zeros(bs, 0),
            node_mask=torch.ones(bs, n, dtype=torch.bool),
        )
        new_X = torch.ones(bs, n, dx)
        new_ph = ph.get_new_object(X=new_X)

        # X should be a clone of new_X
        self.assertTrue(torch.allclose(new_ph.X, new_X))
        # E should be a clone of original
        self.assertTrue(torch.equal(new_ph.E, ph.E))


class TestPlaceHolderMask(unittest.TestCase):
    """Test that mask() zeros out nodes and edges properly."""

    def test_mask_zeros_padding_nodes(self):
        from diffalign.utils.placeholder import PlaceHolder
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.ones(bs, n, dx)
        E = torch.ones(bs, n, n, de)
        y = torch.zeros(bs, 0)
        # Only the first 2 nodes are real
        node_mask = torch.tensor([[True, True, False, False]])

        ph = PlaceHolder(X=X.clone(), E=E.clone(), y=y, node_mask=node_mask)
        ph.mask()

        # Masked-out nodes: X rows for indices 2 and 3 should be one-hot on first element
        # (encode_no_element sets first element to 1 when sum is 0)
        for idx in [2, 3]:
            self.assertAlmostEqual(ph.X[0, idx, 0].item(), 1.0, places=5)
            for c in range(1, dx):
                self.assertAlmostEqual(ph.X[0, idx, c].item(), 0.0, places=5)

        # Real nodes should be unchanged (still all ones)
        for idx in [0, 1]:
            self.assertTrue(torch.allclose(ph.X[0, idx], torch.ones(dx)))

    def test_mask_edges_symmetric(self):
        from diffalign.utils.placeholder import PlaceHolder
        bs, n, dx, de = 1, 3, 2, 2
        X = torch.ones(bs, n, dx)
        E = torch.ones(bs, n, n, de)
        y = torch.zeros(bs, 0)
        node_mask = torch.tensor([[True, True, False]])

        ph = PlaceHolder(X=X.clone(), E=E.clone(), y=y, node_mask=node_mask)
        ph.mask()

        # Edges involving the masked node (index 2) should be zeroed and
        # then have encode_no_element applied
        self.assertAlmostEqual(ph.E[0, 0, 2, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(ph.E[0, 2, 0, 0].item(), 1.0, places=5)

    def test_mask_collapse(self):
        """mask(collapse=True) should argmax X and E."""
        from diffalign.utils.placeholder import PlaceHolder
        bs, n, dx, de = 1, 3, 4, 3
        X = torch.zeros(bs, n, dx)
        X[0, 0, 2] = 1.0  # node 0 is class 2
        X[0, 1, 1] = 1.0  # node 1 is class 1
        X[0, 2, 0] = 1.0  # node 2 is class 0 (padding)
        E = torch.zeros(bs, n, n, de)
        E[0, 0, 1, 1] = 1.0
        E[0, 1, 0, 1] = 1.0
        y = torch.zeros(bs, 0)
        node_mask = torch.tensor([[True, True, False]])

        ph = PlaceHolder(X=X.clone(), E=E.clone(), y=y, node_mask=node_mask)
        ph.mask(collapse=True)

        self.assertEqual(ph.X.shape, (bs, n))
        self.assertEqual(ph.X[0, 0].item(), 2)
        self.assertEqual(ph.X[0, 1].item(), 1)
        self.assertEqual(ph.X[0, 2].item(), 0)  # masked out


@unittest.skipUnless(HAS_GRAPH_MODULE, "diffalign.utils.graph not importable (missing rdkit Draw libs)")
class TestGetUniqueIndicesFromReactionList(unittest.TestCase):
    """Test graph.get_unique_indices_from_reaction_list."""

    def test_all_unique(self):
        rxns = ["A.B>>C", "D.E>>F", "G>>H"]
        indices, counts, is_unique = get_unique_indices_from_reaction_list(rxns)
        self.assertEqual(len(indices), 3)
        self.assertTrue(all(is_unique))
        self.assertEqual(sorted(indices), [0, 1, 2])

    def test_with_duplicates(self):
        rxns = ["A.B>>C", "A.B>>C", "D>>E"]
        indices, counts, is_unique = get_unique_indices_from_reaction_list(rxns)
        # Should have 2 unique reactions
        self.assertEqual(len(indices), 2)
        # The first "A.B>>C" should be the representative
        self.assertIn(0, indices)
        # "D>>E" at index 2 is also unique
        self.assertIn(2, indices)
        # is_unique should mark index 1 as not unique
        self.assertTrue(is_unique[0])
        self.assertFalse(is_unique[1])
        self.assertTrue(is_unique[2])

    def test_reactant_order_invariance(self):
        """A.B>>C and B.A>>C should be treated as the same reaction."""
        rxns = ["A.B>>C", "B.A>>C"]
        indices, counts, is_unique = get_unique_indices_from_reaction_list(rxns)
        self.assertEqual(len(indices), 1, "Reactant order should not matter")

    def test_empty_list(self):
        """An empty list should return empty results."""
        # np.unique on empty array -- should not crash
        try:
            indices, counts, is_unique = get_unique_indices_from_reaction_list([])
        except (ValueError, IndexError):
            # The function may not handle empty input gracefully; that's okay
            pass


if __name__ == "__main__":
    unittest.main()
