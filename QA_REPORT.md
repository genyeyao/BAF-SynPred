# Release quality-assurance report

The portable repository was validated on 2026-09-22 with the existing
`py37_pytorch` environment (Python 3.7.12, PyTorch 1.13.0, CUDA 11.6) and an
NVIDIA GeForce RTX 4060 Laptop GPU.

## Passed checks

- SHA-256 verification passed for all 25 protected source, data, split, test,
  and checkpoint files.
- Input audit passed: 10,183 samples, 37 cell lines, 977 cell features, and the
  expected five fold sizes.
- All six deterministic unit/regression tests passed.
- The final model contains 4,705,756 parameters.
- Fold-0 checkpoint loading and single-batch CPU inference passed.
- All five canonical checkpoints loaded and completed GPU evaluation.

## Reproduced five-fold metrics

| Metric | Mean | Standard deviation |
|---|---:|---:|
| ROC-AUC | 0.981064 | 0.003102 |
| PR-AUC | 0.937478 | 0.007313 |
| Accuracy | 0.952372 | 0.004367 |
| F1 | 0.869579 | 0.012660 |
| Kappa | 0.840466 | 0.015246 |

The reproduced metrics match the bundled canonical experiment summary.
Generated caches and local evaluation outputs were removed after validation;
the public run scripts recreate them automatically.
