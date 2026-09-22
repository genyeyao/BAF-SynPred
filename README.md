# BAF-SynPred

BAF-SynPred is a cell-conditioned, dual-view model for drug-combination
synergy prediction. This repository is the minimal reproducible release of the
final CrossTalk-retained model used for the reported five-fold result.

## Included

- the exact final model and training implementation;
- the 10,183-sample O'Neil interaction table and 977-dimensional cell features;
- the fixed five-fold partition used in the study;
- five canonical best checkpoints;
- validation, preprocessing, training, checkpoint evaluation, and tests;
- a SHA-256 release manifest covering source, data, splits, and weights.

Historical models, hyperparameter searches, ablations, multi-omics pilots,
figures, manuscripts, and intermediate runs are intentionally excluded.

## Canonical configuration

The final model uses `embed_dim=128`, `bond_k=3`, `cell_noise_std=0.05`,
`cell_drug_dropout=0.2`, `adaptive_kl_beta=0.5`, `batch_size=64`,
`lr=2e-4`, `weight_decay=0`, `recon_weight=0.01`, and
`grad_clip_norm=1.0`. The atom and bond branches each contain one
HypergraphConv block, and bidirectional gated atom-bond CrossTalk is retained.

The bundled checkpoint family is the canonical run
`20260908_225152_bafsynpred`, with mean five-fold ROC-AUC
`0.981064 +/- 0.003102`. These are within-dataset out-of-fold results, not an
independent external test set.

## Windows quick start

Open PowerShell in the repository directory. If local scripts are blocked,
run `Set-ExecutionPolicy -Scope Process Bypass` once in that terminal.

```powershell
.\install_windows.ps1
.\run_preflight.ps1
.\run_evaluation.ps1
```

To reproduce five-fold training instead of only evaluating the supplied
weights:

```powershell
.\run_training.ps1 -HighPriority
```

The evaluation summary is written to `runs/canonical_5fold_evaluation.json`.
New training runs are written below `runs/`.

## Manual commands

```bash
python scripts/verify_release.py
python scripts/validate.py
python scripts/preprocess.py
python scripts/evaluate.py --run-dir weights/canonical_5fold --data-root data
python scripts/train.py --data-root data --output-root runs --device cuda
```

## Repository layout

```text
data/                         included inputs and fixed split
scripts/                      validation, preprocessing, training, evaluation
src/bafsynpred/               final model implementation
tests/                        deterministic regression tests
weights/canonical_5fold/      five supplied checkpoints and run metadata
release_manifest.json         immutable SHA-256 inventory
```

## Environment

The canonical run used Python 3.7.12, PyTorch 1.13.0, CUDA 11.6,
PyTorch Geometric 2.3.0, `torch-scatter` 2.1.1, and `torch-cluster` 1.6.1.
The supplied Windows installer recreates this environment with Conda.

## Data and licensing note

The included files are the processed inputs used for reproducibility. Before
publishing a public fork, repository owners should confirm that redistribution
of every dataset file is permitted by its original data-use terms and should
add the software license selected by the authors. No license is inferred by
this packaging step.

