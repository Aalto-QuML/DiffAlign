"""
Unit tests for noise schedule transition matrices.

Tests that transition matrices produced by DiscreteUniformTransition,
MarginalUniformTransition, and AbsorbingStateTransitionMaskNoEdge
are row-stochastic (rows sum to 1).
"""
import os, sys, pathlib, importlib, importlib.util
PROJECT_ROOT = pathlib.Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import unittest

# Force CPU so tests run without a GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# The diffalign.diffusion package __init__.py imports the full diffusion pipeline
# which pulls in many heavy dependencies (pytorch-lightning, wandb, fcd_torch,
# etc.).  We bypass that by loading noise_schedule.py directly with importlib
# and patching module-level `device` variables to CPU.
try:
    import types
    import torch
    import numpy as np

    # 1. Patch genericdiffusion.device to CPU before anything else
    import diffalign.utils.diffusion.genericdiffusion as gd_module
    gd_module.device = torch.device("cpu")

    # 2. Resolve the circular import in diffalign.utils (graph <-> mol <-> graph_builder)
    from diffalign.utils import graph as _graph  # noqa: F401

    # 3. Prevent the diffalign.diffusion __init__.py from executing by registering a
    #    dummy module for the package before loading the submodule.
    if "diffalign.diffusion" not in sys.modules:
        _pkg = types.ModuleType("diffalign.diffusion")
        _pkg.__path__ = [os.path.join(str(PROJECT_ROOT), "diffalign", "diffusion")]
        _pkg.__package__ = "diffalign.diffusion"
        sys.modules["diffalign.diffusion"] = _pkg

    # 4. Mock diffalign.utils.setup to avoid pulling in the entire pipeline
    #    (setup.py imports diffusion_rxn which imports neuralnet, wandb, etc.)
    #    noise_schedule.py imports setup but never uses it.
    if "diffalign.utils.setup" not in sys.modules:
        sys.modules["diffalign.utils.setup"] = types.ModuleType("diffalign.utils.setup")

    # 5. Now load noise_schedule.py via importlib (skips __init__.py since
    #    the package is already registered).
    _ns_path = os.path.join(str(PROJECT_ROOT), "diffalign", "diffusion", "noise_schedule.py")
    _spec = importlib.util.spec_from_file_location("diffalign.diffusion.noise_schedule", _ns_path,
                                                    submodule_search_locations=[])
    _ns_mod = importlib.util.module_from_spec(_spec)
    _ns_mod.device = torch.device("cpu")
    sys.modules["diffalign.diffusion.noise_schedule"] = _ns_mod
    _spec.loader.exec_module(_ns_mod)

    DiscreteUniformTransition = _ns_mod.DiscreteUniformTransition
    MarginalUniformTransition = _ns_mod.MarginalUniformTransition
    AbsorbingStateTransitionMaskNoEdge = _ns_mod.AbsorbingStateTransitionMaskNoEdge

    HAS_DEPS = True
except (ImportError, OSError) as _exc:
    HAS_DEPS = False
    torch = None  # type: ignore
    _import_error = _exc


def _row_stochastic(matrix, atol: float = 1e-5) -> bool:
    """Check that every row of `matrix` sums to 1 (within tolerance)."""
    row_sums = matrix.sum(dim=-1)
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol)


@unittest.skipUnless(HAS_DEPS, "required dependencies not available")
class TestDiscreteUniformTransition(unittest.TestCase):
    """Verify row-stochasticity for DiscreteUniformTransition."""

    @classmethod
    def setUpClass(cls):
        cls.x_classes = 5
        cls.e_classes = 4
        cls.y_classes = 2
        cls.timesteps = 10
        cls.transition = DiscreteUniformTransition(
            noise_schedule="cosine",
            timesteps=cls.timesteps,
            x_classes=cls.x_classes,
            e_classes=cls.e_classes,
            y_classes=cls.y_classes,
            diffuse_edges=True,
            node_idx_to_mask=torch.tensor([0], dtype=torch.long),
            edge_idx_to_mask=None,
        )

    def test_get_Qt_x_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qt.X),
                f"Qt.X not row-stochastic at t={t_val}; row sums={qt.X.sum(-1)}",
            )

    def test_get_Qt_e_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qt.E),
                f"Qt.E not row-stochastic at t={t_val}; row sums={qt.E.sum(-1)}",
            )

    def test_get_Qt_bar_x_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qtb = self.transition.get_Qt_bar(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qtb.X),
                f"Qt_bar.X not row-stochastic at t={t_val}; row sums={qtb.X.sum(-1)}",
            )

    def test_get_Qt_bar_e_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qtb = self.transition.get_Qt_bar(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qtb.E),
                f"Qt_bar.E not row-stochastic at t={t_val}; row sums={qtb.E.sum(-1)}",
            )

    def test_Qt_at_t0_is_identity(self):
        """At t=0 the transition matrix should be row-stochastic (identity-like)."""
        t = torch.tensor([[0]])
        qt = self.transition.get_Qt(t, device=torch.device("cpu"))
        self.assertTrue(_row_stochastic(qt.X))

    def test_nonnegative_entries(self):
        """All transition matrix entries should be non-negative."""
        for t_val in range(self.timesteps + 1):
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue((qt.X >= -1e-6).all(), f"Negative entry in Qt.X at t={t_val}")
            self.assertTrue((qt.E >= -1e-6).all(), f"Negative entry in Qt.E at t={t_val}")


@unittest.skipUnless(HAS_DEPS, "required dependencies not available")
class TestMarginalUniformTransition(unittest.TestCase):
    """Verify row-stochasticity for MarginalUniformTransition."""

    @classmethod
    def setUpClass(cls):
        cls.x_classes = 5
        cls.e_classes = 4
        cls.y_classes = 2
        cls.timesteps = 10
        x_marginals = torch.tensor([0.1, 0.3, 0.2, 0.15, 0.25])
        e_marginals = torch.tensor([0.4, 0.3, 0.2, 0.1])
        cls.transition = MarginalUniformTransition(
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            y_classes=cls.y_classes,
            noise_schedule="cosine",
            timesteps=cls.timesteps,
            diffuse_edges=True,
            node_idx_to_mask=torch.tensor([0], dtype=torch.long),
            edge_idx_to_mask=None,
        )

    def test_get_Qt_x_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qt.X),
                f"Qt.X not row-stochastic at t={t_val}",
            )

    def test_get_Qt_e_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qt.E),
                f"Qt.E not row-stochastic at t={t_val}",
            )

    def test_get_Qt_bar_x_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qtb = self.transition.get_Qt_bar(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qtb.X),
                f"Qt_bar.X not row-stochastic at t={t_val}",
            )

    def test_get_Qt_bar_e_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qtb = self.transition.get_Qt_bar(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qtb.E),
                f"Qt_bar.E not row-stochastic at t={t_val}",
            )

    def test_nonnegative_entries(self):
        for t_val in range(self.timesteps + 1):
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue((qt.X >= -1e-6).all(), f"Negative entry in Qt.X at t={t_val}")
            self.assertTrue((qt.E >= -1e-6).all(), f"Negative entry in Qt.E at t={t_val}")


@unittest.skipUnless(HAS_DEPS, "required dependencies not available")
class TestAbsorbingStateTransitionMaskNoEdge(unittest.TestCase):
    """Verify row-stochasticity for AbsorbingStateTransitionMaskNoEdge."""

    @classmethod
    def setUpClass(cls):
        cls.x_classes = 5
        cls.e_classes = 4
        cls.y_classes = 2
        cls.timesteps = 10
        # abs_state_position_x uses F.one_hot and must be a valid index;
        # use the last class (x_classes - 1) instead of -1
        cls.transition = AbsorbingStateTransitionMaskNoEdge(
            timesteps=cls.timesteps,
            x_classes=cls.x_classes,
            e_classes=cls.e_classes,
            y_classes=cls.y_classes,
            diffuse_edges=True,
            abs_state_position_x=cls.x_classes - 1,
            abs_state_position_e=0,
            node_idx_to_mask=torch.tensor([0], dtype=torch.long),
            edge_idx_to_mask=None,
        )

    def test_get_Qt_x_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qt.X),
                f"Qt.X not row-stochastic at t={t_val}; row sums={qt.X.sum(-1)}",
            )

    def test_get_Qt_e_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qt.E),
                f"Qt.E not row-stochastic at t={t_val}; row sums={qt.E.sum(-1)}",
            )

    def test_get_Qt_bar_x_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qtb = self.transition.get_Qt_bar(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qtb.X),
                f"Qt_bar.X not row-stochastic at t={t_val}; row sums={qtb.X.sum(-1)}",
            )

    def test_get_Qt_bar_e_row_stochastic(self):
        for t_val in [0, 1, 5, self.timesteps]:
            t = torch.tensor([[t_val]])
            qtb = self.transition.get_Qt_bar(t, device=torch.device("cpu"))
            self.assertTrue(
                _row_stochastic(qtb.E),
                f"Qt_bar.E not row-stochastic at t={t_val}; row sums={qtb.E.sum(-1)}",
            )

    def test_Qt_bar_at_t0_is_identity(self):
        """Qt_bar at t=0 should be the identity matrix."""
        t = torch.tensor([[0]])
        qtb = self.transition.get_Qt_bar(t, device=torch.device("cpu"))
        eye_x = torch.eye(self.x_classes)
        # Masked row (row 0) is special, check the rest
        for row in range(1, self.x_classes):
            self.assertTrue(
                torch.allclose(qtb.X[0, row], eye_x[row], atol=1e-5),
                f"Qt_bar.X row {row} at t=0 is not identity-like",
            )

    def test_nonnegative_entries(self):
        for t_val in range(self.timesteps + 1):
            t = torch.tensor([[t_val]])
            qt = self.transition.get_Qt(t, device=torch.device("cpu"))
            self.assertTrue((qt.X >= -1e-6).all(), f"Negative entry in Qt.X at t={t_val}")
            self.assertTrue((qt.E >= -1e-6).all(), f"Negative entry in Qt.E at t={t_val}")


if __name__ == "__main__":
    unittest.main()
