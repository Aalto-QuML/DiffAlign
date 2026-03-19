# DiffAlign Verification & Reproduction Guide

Run the levels below in order. Each level builds on the previous one.

## Level 1 — Smoke Tests (seconds, no data/GPU)

```bash
pytest tests/test_smoke.py -v
```

Verifies that core modules import correctly and Hydra configs load.

## Level 2 — Unit Tests (seconds, no data/GPU)

```bash
pytest tests/test_noise_schedule.py tests/test_graph_utils.py tests/test_mol_utils.py tests/test_graph_builder.py -v
```

Tests noise schedule transition matrices (row-stochasticity), PlaceHolder utilities,
SMILES parsing/graph conversion, and shared graph construction.

## Level 3 — Training Verification (minutes, needs data, GPU recommended)

```bash
python scripts/train.py +experiment=align_absorbing train.epochs=2 general.wandb.mode=offline
```

Runs 2 epochs on USPTO-50k to verify the training loop completes without errors.

## Level 4 — Sampling (minutes, needs checkpoint + data)

```bash
python scripts/sample.py +experiment=align_absorbing \
    general.wandb.mode=offline \
    general.wandb.run_id=YOUR_RUN_ID \
    test.n_conditions=10 \
    test.n_samples_per_condition=10 \
    diffusion.diffusion_steps_eval=1
```

Generates samples from a trained checkpoint. Adjust `n_conditions` and
`n_samples_per_condition` for a quick sanity check.

## Level 5 — Evaluation (minutes, needs samples from Level 4)

```bash
python scripts/evaluate.py +experiment=align_absorbing \
    general.wandb.mode=offline \
    general.wandb.run_id=YOUR_RUN_ID
```

Computes top-k accuracy and other metrics on the sampled results.

## Level 6 — Full Paper Reproduction (hours/days)

To reproduce the ICLR 2025 paper results:

1. Train for 760+ epochs on USPTO-50k
2. Sample 5000 conditions × 100 samples with `diffusion_steps_eval=100`
3. Evaluate with ELBO ranking

```bash
python scripts/train.py +experiment=align_absorbing train.epochs=760

python scripts/sample.py +experiment=align_absorbing \
    general.wandb.run_id=YOUR_RUN_ID \
    test.n_conditions=5000 \
    test.n_samples_per_condition=100 \
    diffusion.diffusion_steps_eval=100

python scripts/evaluate.py +experiment=align_absorbing \
    general.wandb.run_id=YOUR_RUN_ID
```

## Level 7 — Syntheseus Integration

```python
from diffalign import DiffAlignModel

model = DiffAlignModel(model_dir="checkpoints", device="cpu", diffusion_steps=1)
```

Or use the inference functions directly:

```python
from diffalign import predict_precursors

results = predict_precursors("CC(=O)Oc1ccccc1C(=O)O", n_precursors=5, diffusion_steps=1)
for r in results:
    print(f"{r['precursors']}  (score: {r['score']:.2f})")
```
