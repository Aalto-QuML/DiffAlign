"""Aggregate per-chunk evaluation pickle files into summary metrics."""
import argparse
import glob
import os
import pickle
import numpy as np


def collect_scores(exp_dir):
    """Load and concatenate all score dicts from pickle files in exp_dir."""
    pattern = os.path.join(exp_dir, "scores_*.pickle")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No scores_*.pickle files found in {exp_dir}")

    all_scores = []
    for f in files:
        with open(f, "rb") as fh:
            all_scores.extend(pickle.load(fh))
    return all_scores, files


def aggregate(all_scores):
    """Compute mean and std for each metric across all conditions."""
    keys = sorted(all_scores[0].keys())
    results = {}
    for k in keys:
        vals = np.array([s[k] for s in all_scores], dtype=np.float64)
        results[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return results


def print_results(results, n_conditions, n_files):
    """Pretty-print aggregated metrics grouped by category."""
    topk = {k: v for k, v in results.items() if k.startswith("top-") and "weighted" not in k}
    topk_w = {k: v for k, v in results.items() if k.startswith("top-") and "weighted" in k}
    mrr = {k: v for k, v in results.items() if k.startswith("mrr") and "weighted" not in k}
    mrr_w = {k: v for k, v in results.items() if k.startswith("mrr") and "weighted" in k}
    other = {k: v for k, v in results.items() if not k.startswith("top-") and not k.startswith("mrr")}

    def sort_key(name):
        if "top-" in name:
            return int(name.split("-")[1].split("_")[0])
        return name

    print(f"Aggregated over {n_conditions} conditions from {n_files} files\n")

    for header, metrics in [
        ("Validity & Matching", other),
        ("Top-K Accuracy (ELBO-sorted)", topk),
        ("Top-K Accuracy (weighted)", topk_w),
        ("MRR (ELBO-sorted)", mrr),
        ("MRR (weighted)", mrr_w),
    ]:
        print(f"=== {header} ===")
        for k in sorted(metrics, key=sort_key):
            m, s = metrics[k]["mean"], metrics[k]["std"]
            print(f"  {k:40s}  mean={m:.4f}  std={s:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation scores from pickle files.")
    parser.add_argument("exp_dir", type=str, help="Path to the experiment directory containing scores_*.pickle files.")
    args = parser.parse_args()

    all_scores, files = collect_scores(args.exp_dir)
    results = aggregate(all_scores)
    print_results(results, n_conditions=len(all_scores), n_files=len(files))


if __name__ == "__main__":
    main()
