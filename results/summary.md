# Hi-C Transformer-MAE Imputation Demo — Summary (8-run mini-sweep)

## Best run (selected)
**Goal:** best trade-off between **masked accuracy** and **structure preservation**.

### Configuration
| Item | Value |
|---|---|
| mask_mode | `dist` (distance-biased missingness) |
| dist_k | 3.0 |
| patch | 4 |
| mask_ratio | 0.4 |
| lambda_insul | 0.02 |
| beta_l1 | 0.1 |
| dist_gamma | 0.5 |

## Results
### Primary metrics
| Metric | Value | Interpretation |
|---|---:|---|
| masked_RMSE | **0.3603** | lower is better |
| insulation_Pearson | **0.9923** | higher is better (structure preserved) |
| coverage95 | **0.9645** | close to 0.95 indicates good calibration |
| spurious_hotspot_control (topk\|resid\|) | **1.0072** | ~1.0 indicates residual hotspots are controlled |

### Baselines (for reference)
| Baseline | masked_RMSE |
|---|---:|
| no-enhance | 2.0346 |
| smooth | 2.0393 |

## Notes
- `coverage95` computed using Gaussian interval: **μ ± 1.96σ**.
- `spurious_hotspot_control` is reported as a **ratio** (closer to 1.0 is better).

