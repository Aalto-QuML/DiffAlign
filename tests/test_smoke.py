"""
Minimal smoke tests: imports, config loading, task mapping.
"""
import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_key_imports():
    """Verify that the core modules can be imported without error."""
    from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
    from diffalign.datasets import supernode_dataset
    from diffalign.utils import setup, graph, mol
    assert DiscreteDenoisingDiffusionRxn is not None
    assert supernode_dataset is not None


def test_hydra_config_loads():
    """Verify that the default Hydra config loads successfully."""
    from hydra import compose, initialize_config_dir

    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(config_name="default")
        assert cfg is not None
        assert hasattr(cfg, "general")
        assert hasattr(cfg, "train")
        assert hasattr(cfg, "diffusion")
        assert hasattr(cfg, "neuralnet")


def test_task_to_class_and_model_valid():
    """Verify that task_to_class_and_model entries point to real classes."""
    from diffalign.utils import setup

    for task_name, mapping in setup.task_to_class_and_model.items():
        assert "data_class" in mapping, f"Task {task_name} missing data_class"
        assert "model_class" in mapping, f"Task {task_name} missing model_class"
        assert mapping["data_class"] is not None, f"Task {task_name} has None data_class"
        assert mapping["model_class"] is not None, f"Task {task_name} has None model_class"


def test_train_script_no_exit():
    """Verify that the train.py script no longer contains exit() in the training loop."""
    train_path = PROJECT_ROOT / "scripts" / "train.py"
    source = train_path.read_text()
    # Should not have bare exit() in the loop body
    lines = source.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "exit()" and i < len(lines) - 5:  # not at very end of file
            raise AssertionError(f"Found exit() at line {i+1} in train.py — this kills training after one epoch")


if __name__ == "__main__":
    test_key_imports()
    print("PASS: test_key_imports")
    test_hydra_config_loads()
    print("PASS: test_hydra_config_loads")
    test_task_to_class_and_model_valid()
    print("PASS: test_task_to_class_and_model_valid")
    test_train_script_no_exit()
    print("PASS: test_train_script_no_exit")
    print("\nAll smoke tests passed!")
