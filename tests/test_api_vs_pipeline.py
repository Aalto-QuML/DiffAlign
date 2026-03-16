"""
Diagnostic comparison: API-generated dense_data vs dataset-pipeline dense_data.

Loads the same product SMILES through both paths and compares tensor shapes,
values, and distributions to identify divergences that cause garbage samples.
"""
import os
import sys
import pathlib

# Ensure project root is on the path
PROJECT_ROOT = pathlib.Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Batch
from hydra import compose, initialize_config_dir

from src.utils import graph, mol, setup
from src.utils.graph import to_dense
from src.datasets.uspto_rxn_dataset import Dataset as USPTODataset
from api.predict import smiles_to_dense_data, load_model, BOND_TYPES

# ---------- constants ----------
TEST_PRODUCT = "O=S(C1=CC=C(C=CC2=NC(COC3=CC=C(COCCN4C=CN=N4)C=C3)=CO2)C=C1)C(F)(F)F"
DATASET_BOND_TYPES = ['none', BT.SINGLE, BT.DOUBLE, BT.TRIPLE]  # 4 types used by dataset
GRAPH_BOND_TYPES = graph.bond_types  # 7 types from graph.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg():
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="default",
            overrides=["+experiment=align_absorbing"],
        )
    return cfg


def get_dataset_dense_data(cfg, index=0):
    """Load processed test data from disk and convert to dense via the training pipeline."""
    root_path = os.path.join(PROJECT_ROOT, cfg.dataset.datadir + "-" + str(cfg.dataset.dataset_nb))
    dataset = USPTODataset(
        stage="test",
        root=root_path,
        with_explicit_h=cfg.dataset.with_explicit_h,
        with_formal_charge=cfg.dataset.with_formal_charge,
        max_nodes_more_than_product=cfg.dataset.nb_rct_dummy_nodes,
        canonicalize_molecule=cfg.dataset.canonicalize_molecule,
        add_supernode_edges=cfg.dataset.add_supernode_edges,
    )
    data_item = dataset[index]
    batch = Batch.from_data_list([data_item])
    dense_data = to_dense(batch)
    return dense_data, data_item


def get_api_dense_data(cfg, product_smiles):
    """Create dense_data through the API path (smiles_to_dense_data)."""
    dense_data = smiles_to_dense_data(
        product_smiles=product_smiles,
        max_nodes_more_than_product=cfg.dataset.nb_rct_dummy_nodes,
        atom_types=cfg.dataset.atom_types,
        bond_types=BOND_TYPES,
        with_explicit_h=cfg.dataset.with_explicit_h,
        with_formal_charge=cfg.dataset.with_formal_charge,
        add_supernode_edges=cfg.dataset.add_supernode_edges,
        canonicalize_molecule=cfg.dataset.canonicalize_molecule,
        permute_mols=cfg.dataset.permute_mols,
    )
    return dense_data


def extract_product_smiles(dense_data, atom_types):
    """Extract the product SMILES from a dense_data object using get_cano_smiles_from_dense."""
    X_idx = dense_data.X.argmax(dim=-1)
    E_idx = dense_data.E.argmax(dim=-1)
    rxn_strs = mol.get_cano_smiles_from_dense(
        X_idx, E_idx, atom_types, GRAPH_BOND_TYPES
    )
    if rxn_strs:
        rxn = rxn_strs[0]
        if ">>" in rxn:
            return rxn.split(">>")[1]
        return rxn
    return None


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def compare_shapes(label, t1, t2):
    match = "MATCH" if t1.shape == t2.shape else "MISMATCH"
    print(f"  {label:30s}  dataset={str(t1.shape):20s}  api={str(t2.shape):20s}  [{match}]")
    return t1.shape == t2.shape


def compare_dtypes(label, t1, t2):
    match = "MATCH" if t1.dtype == t2.dtype else "MISMATCH"
    print(f"  {label:30s}  dataset={str(t1.dtype):20s}  api={str(t2.dtype):20s}  [{match}]")
    return t1.dtype == t2.dtype


def compare_values(label, t1, t2):
    if t1.shape != t2.shape:
        print(f"  {label:30s}  Cannot compare values (shapes differ)")
        return False
    if t1.dtype == torch.bool:
        match_count = (t1 == t2).sum().item()
        total = t1.numel()
        print(f"  {label:30s}  matching={match_count}/{total} ({100*match_count/total:.1f}%)")
        return match_count == total
    elif t1.dtype in (torch.float32, torch.float64):
        close = torch.allclose(t1, t2, atol=1e-5)
        max_diff = (t1 - t2).abs().max().item()
        print(f"  {label:30s}  allclose={close}  max_diff={max_diff:.6f}")
        return close
    else:
        match_count = (t1 == t2).sum().item()
        total = t1.numel()
        print(f"  {label:30s}  matching={match_count}/{total} ({100*match_count/total:.1f}%)")
        return match_count == total


def node_type_distribution(X, atom_types):
    """Print distribution of atom types in X (argmax of one-hot)."""
    types = X.argmax(dim=-1).flatten()
    counts = torch.bincount(types, minlength=len(atom_types))
    print("    Node type distribution:")
    for i, (at, c) in enumerate(zip(atom_types, counts)):
        if c > 0:
            print(f"      [{i:2d}] {at:6s}: {c.item()}")


def edge_type_distribution(E, bond_type_labels):
    """Print distribution of edge types in E (argmax of one-hot)."""
    types = E.argmax(dim=-1).flatten()
    counts = torch.bincount(types, minlength=len(bond_type_labels))
    print("    Edge type distribution:")
    for i, (bt, c) in enumerate(zip(bond_type_labels, counts)):
        if c > 0:
            print(f"      [{i:2d}] {str(bt):10s}: {c.item()}")


def suno_position(X, atom_types):
    """Find position of SuNo node(s)."""
    suno_idx = atom_types.index("SuNo")
    types = X.argmax(dim=-1)
    for b in range(types.shape[0]):
        positions = (types[b] == suno_idx).nonzero(as_tuple=True)[0]
        print(f"    Batch {b}: SuNo at positions {positions.tolist()}")


def main():
    print_separator("Loading config")
    cfg = load_cfg()
    atom_types = list(cfg.dataset.atom_types)
    print(f"  atom_types: {len(atom_types)} types")
    print(f"  BOND_TYPES (API / graph.py): {len(BOND_TYPES)} types: {BOND_TYPES}")
    print(f"  DATASET_BOND_TYPES: {len(DATASET_BOND_TYPES)} types: {DATASET_BOND_TYPES}")

    # ---- 1. Load dataset dense_data ----
    print_separator("1. Loading dataset dense_data (test[0])")
    ds_dense, ds_item = get_dataset_dense_data(cfg, index=0)
    print(f"  Raw PyG data: x.shape={ds_item.x.shape}, edge_index.shape={ds_item.edge_index.shape}, "
          f"edge_attr.shape={ds_item.edge_attr.shape}")

    # Extract product SMILES from dataset
    # get_cano_smiles_from_dense expects argmax'd X (bs, n) and E (bs, n, n)
    ds_X_idx = ds_dense.X.argmax(dim=-1)  # (1, n)
    ds_E_idx = ds_dense.E.argmax(dim=-1)  # (1, n, n)
    ds_rxn_7 = mol.get_cano_smiles_from_dense(ds_X_idx, ds_E_idx, atom_types, GRAPH_BOND_TYPES)
    print(f"  Dataset rxn (decoded with 7 bond types): {ds_rxn_7}")

    # Also read the raw CSV to get the original SMILES
    raw_test_path = os.path.join(
        PROJECT_ROOT, cfg.dataset.datadir + "-" + str(cfg.dataset.dataset_nb), "raw", "test.csv"
    )
    with open(raw_test_path, "r") as f:
        first_line = f.readline().strip()
    print(f"  Raw test.csv first line: {first_line}")
    raw_product = first_line.split(">>")[1].split(".")[0] if ">>" in first_line else first_line

    # ---- 2. Create API dense_data from same product ----
    # Use the test product or the raw product from dataset
    product_to_use = TEST_PRODUCT
    print_separator(f"2. Creating API dense_data for product")
    print(f"  Product SMILES: {product_to_use}")
    api_dense = get_api_dense_data(cfg, product_to_use)

    # Also create API dense_data for the raw dataset product for direct comparison
    print(f"\n  Also creating API dense_data for dataset product: {raw_product}")
    api_dense_from_ds_product = get_api_dense_data(cfg, raw_product)

    # ---- 3. Shape comparisons ----
    print_separator("3. Shape comparison (dataset vs API from dataset product)")
    ds = ds_dense
    api = api_dense_from_ds_product
    compare_shapes("X (nodes)", ds.X, api.X)
    compare_shapes("E (edges)", ds.E, api.E)
    compare_shapes("y", ds.y, api.y)
    if ds.node_mask is not None and api.node_mask is not None:
        compare_shapes("node_mask", ds.node_mask, api.node_mask)
    if ds.atom_map_numbers is not None and api.atom_map_numbers is not None:
        compare_shapes("atom_map_numbers", ds.atom_map_numbers, api.atom_map_numbers)
    if ds.mol_assignments is not None and api.mol_assignments is not None:
        compare_shapes("mol_assignments", ds.mol_assignments, api.mol_assignments)

    # ---- 4. Dtype comparisons ----
    print_separator("4. Dtype comparison")
    compare_dtypes("X", ds.X, api.X)
    compare_dtypes("E", ds.E, api.E)
    if ds.node_mask is not None and api.node_mask is not None:
        compare_dtypes("node_mask", ds.node_mask, api.node_mask)

    # ---- 5. Edge dimension analysis (CRITICAL) ----
    print_separator("5. Edge dimension analysis (CRITICAL)")
    print(f"  Dataset edge_attr dim (raw PyG): {ds_item.edge_attr.shape[-1]}")
    print(f"  Dataset E dim (dense):           {ds.E.shape[-1]}")
    print(f"  API E dim (dense):               {api.E.shape[-1]}")
    print(f"  Model expects input_dims.E:      {len(BOND_TYPES)} (from BOND_TYPES)")
    if ds.E.shape[-1] != api.E.shape[-1]:
        print(f"  *** CRITICAL MISMATCH: Dataset E has {ds.E.shape[-1]} edge types, "
              f"API E has {api.E.shape[-1]} edge types! ***")

    # ---- 6. Node type distributions ----
    print_separator("6. Node type distributions")
    print("  Dataset:")
    node_type_distribution(ds.X, atom_types)
    print("  API (from dataset product):")
    node_type_distribution(api.X, atom_types)

    # ---- 7. Edge type distributions ----
    print_separator("7. Edge type distributions")
    print("  Dataset (interpreting with 4 bond types):")
    edge_type_distribution(ds.E, DATASET_BOND_TYPES)
    print("  API (interpreting with 7 bond types):")
    edge_type_distribution(api.E, BOND_TYPES)

    # ---- 8. SuNo positions ----
    print_separator("8. SuNo positions")
    print("  Dataset:")
    suno_position(ds.X, atom_types)
    print("  API (from dataset product):")
    suno_position(api.X, atom_types)

    # ---- 9. Node mask comparison ----
    print_separator("9. Node mask analysis")
    if ds.node_mask is not None:
        print(f"  Dataset: {ds.node_mask.sum().item()} active nodes out of {ds.node_mask.numel()}")
    if api.node_mask is not None:
        print(f"  API:     {api.node_mask.sum().item()} active nodes out of {api.node_mask.numel()}")

    # ---- 10. Atom mapping analysis ----
    print_separator("10. Atom mapping analysis")
    if ds.atom_map_numbers is not None:
        ds_am = ds.atom_map_numbers[ds.node_mask] if ds.node_mask is not None else ds.atom_map_numbers.flatten()
        print(f"  Dataset atom_map: unique values={torch.unique(ds_am).tolist()[:20]}...")
        print(f"  Dataset atom_map: nonzero count={((ds_am != 0).sum().item())}")
    if api.atom_map_numbers is not None:
        api_am = api.atom_map_numbers[api.node_mask] if api.node_mask is not None else api.atom_map_numbers.flatten()
        print(f"  API atom_map: unique values={torch.unique(api_am).tolist()[:20]}...")
        print(f"  API atom_map: nonzero count={((api_am != 0).sum().item())}")

    # ---- 11. Mol assignment analysis ----
    print_separator("11. Mol assignment analysis")
    if ds.mol_assignments is not None:
        print(f"  Dataset mol_assignments: unique={torch.unique(ds.mol_assignments).tolist()}")
    if api.mol_assignments is not None:
        print(f"  API mol_assignments: unique={torch.unique(api.mol_assignments).tolist()}")

    # ---- 12. Value comparison where shapes match ----
    print_separator("12. Value comparison (dataset vs API, same product)")
    if ds.X.shape == api.X.shape:
        compare_values("X values", ds.X, api.X)
    if ds.E.shape == api.E.shape:
        compare_values("E values", ds.E, api.E)
    if ds.node_mask is not None and api.node_mask is not None and ds.node_mask.shape == api.node_mask.shape:
        compare_values("node_mask values", ds.node_mask, api.node_mask)

    # ---- 13. Reactant side analysis ----
    print_separator("13. Reactant side analysis")
    print("  Dataset (real reactants exist on LHS):")
    # In the dataset, the reactant side has actual atom types
    ds_X_types = ds.X.argmax(dim=-1)[0]
    none_idx = atom_types.index("none")
    u_idx = atom_types.index("U")
    suno_idx = atom_types.index("SuNo")
    ds_active = ds_X_types[ds.node_mask[0]] if ds.node_mask is not None else ds_X_types
    print(f"    Active nodes: {len(ds_active)}")
    print(f"    'none' nodes: {(ds_active == none_idx).sum().item()}")
    print(f"    'U' (dummy) nodes: {(ds_active == u_idx).sum().item()}")
    print(f"    'SuNo' nodes: {(ds_active == suno_idx).sum().item()}")
    print(f"    Real atom nodes: {((ds_active != none_idx) & (ds_active != u_idx) & (ds_active != suno_idx)).sum().item()}")

    print("  API (product copied to both sides):")
    api_X_types = api.X.argmax(dim=-1)[0]
    api_active = api_X_types[api.node_mask[0]] if api.node_mask is not None else api_X_types
    print(f"    Active nodes: {len(api_active)}")
    print(f"    'none' nodes: {(api_active == none_idx).sum().item()}")
    print(f"    'U' (dummy) nodes: {(api_active == u_idx).sum().item()}")
    print(f"    'SuNo' nodes: {(api_active == suno_idx).sum().item()}")
    print(f"    Real atom nodes: {((api_active != none_idx) & (api_active != u_idx) & (api_active != suno_idx)).sum().item()}")

    # ---- 14. Model sampling comparison (optional, needs checkpoint) ----
    print_separator("14. Model sampling comparison (SAME product, both paths)")
    checkpoint_file = os.path.join(PROJECT_ROOT, "checkpoints", "epoch760.pt")
    if not os.path.exists(checkpoint_file):
        print("  Skipping model sampling: no checkpoint found at", checkpoint_file)
        print("  To run this section, place the checkpoint at the expected path.")
    else:
        print("  Loading model...")
        cfg.diffusion.diffusion_steps_eval = 100
        model = load_model(cfg)
        model.eval()
        print("  Model loaded.")

        # Use the SAME product for both paths — extract from the dataset's raw CSV
        # Strip atom mapping to get a clean product SMILES
        raw_product_clean = Chem.MolToSmiles(Chem.MolFromSmiles(raw_product))
        print(f"\n  Shared product SMILES: {raw_product_clean}")

        n_samples = 5
        seed = 42

        # --- A. Dataset path ---
        print(f"\n  [A] Dataset path: sampling {n_samples} from dataset dense_data (test[0])...")
        ds_for_model = ds_dense.to_device(device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            ds_samples = model.sample_for_condition(
                dense_data=ds_for_model, n_samples=n_samples,
                inpaint_node_idx=None, inpaint_edge_idx=None, device=device
            )
        ds_rxns = mol.get_cano_smiles_from_dense(
            ds_samples.X, ds_samples.E, atom_types, GRAPH_BOND_TYPES
        )
        print(f"      Dataset samples X shape: {ds_samples.X.shape}")
        print(f"      Dataset decoded rxns ({len(ds_rxns)}):")
        for i, rxn in enumerate(ds_rxns):
            rcts = rxn.split(">>")[0] if ">>" in rxn else rxn
            print(f"        [{i}] reactants: {rcts}")

        # --- B. API path (same product) ---
        print(f"\n  [B] API path: sampling {n_samples} from API dense_data (same product)...")
        api_same = get_api_dense_data(cfg, raw_product_clean).to_device(device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            api_samples = model.sample_for_condition(
                dense_data=api_same, n_samples=n_samples,
                inpaint_node_idx=None, inpaint_edge_idx=None, device=device
            )
        api_rxns = mol.get_cano_smiles_from_dense(
            api_samples.X, api_samples.E, atom_types, BOND_TYPES
        )
        print(f"      API samples X shape: {api_samples.X.shape}")
        print(f"      API decoded rxns ({len(api_rxns)}):")
        for i, rxn in enumerate(api_rxns):
            rcts = rxn.split(">>")[0] if ">>" in rxn else rxn
            print(f"        [{i}] reactants: {rcts}")

        # --- C. Compare ---
        print_separator("15. Reactant comparison (dataset vs API, same product)")
        ds_rct_set = set()
        for rxn in ds_rxns:
            if ">>" in rxn:
                ds_rct_set.add(rxn.split(">>")[0])
        api_rct_set = set()
        for rxn in api_rxns:
            if ">>" in rxn:
                api_rct_set.add(rxn.split(">>")[0])
        common = ds_rct_set & api_rct_set
        print(f"  Dataset unique reactant sets: {len(ds_rct_set)}")
        for r in sorted(ds_rct_set):
            print(f"    {r}")
        print(f"  API unique reactant sets:     {len(api_rct_set)}")
        for r in sorted(api_rct_set):
            print(f"    {r}")
        print(f"  Common reactant sets:         {len(common)}")
        for r in sorted(common):
            print(f"    {r}")
        if not common:
            print("  *** WARNING: No overlap between dataset and API predictions! ***")
            print("  This suggests the two paths are conditioning the model differently.")

    # ---- Summary ----
    print_separator("SUMMARY OF FINDINGS")
    issues = []

    # Dense data structure checks (same product, both paths)
    if ds.E.shape[-1] != api.E.shape[-1]:
        issues.append(
            f"CRITICAL: Edge feature dim mismatch. Dataset E has {ds.E.shape[-1]} dims, "
            f"API E has {api.E.shape[-1]} dims."
        )

    # Check for edge type distribution divergence
    ds_E_types = ds.E.argmax(dim=-1).flatten()
    api_E_types = api.E.argmax(dim=-1).flatten()
    ds_counts = torch.bincount(ds_E_types, minlength=max(ds.E.shape[-1], api.E.shape[-1]))
    api_counts = torch.bincount(api_E_types, minlength=max(ds.E.shape[-1], api.E.shape[-1]))
    for i in range(len(BOND_TYPES)):
        ds_c = ds_counts[i].item() if i < len(ds_counts) else 0
        api_c = api_counts[i].item() if i < len(api_counts) else 0
        if ds_c != api_c and (ds_c == 0 or api_c == 0) and i > 0:
            issues.append(
                f"Edge type '{BOND_TYPES[i]}' (idx={i}): dataset has {ds_c}, API has {api_c}. "
                f"One path produces this edge type while the other does not."
            )

    if not issues:
        print("  No structural issues detected in dense_data construction.")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")


if __name__ == "__main__":
    main()
