# Hi-C Transformer-MAE Imputation Demo — Summary (8-run mini-sweep)

## Best run (selected)
**Selection rule (8 runs):** prioritize **structure preservation** first, then minimize **masked error**.  
Concretely: choose runs with high `insulation_Pearson`, then pick the one with the lowest `masked_RMSE` among them.

**Takeaway:** distance-biased masking (`mask_mode=dist`) + light insulation regularization yields **high structural fidelity** while keeping **masked error low**, with **reasonable uncertainty calibration**.

### Configuration
| Item | Value | Note |
|---|---|---|
| mask_mode | `dist` | distance-biased missingness |
| dist_k | 3.0 | strength/scale of distance bias |
| dist_gamma | 0.5 | distance decay exponent |
| patch | 4 | patch size |
| mask_ratio | 0.4 | fraction masked |
| lambda_insul | 0.02 | insulation proxy regularization |
| beta_l1 | 0.1 | sparsity / denoising term |

## Results

### Primary metrics
| Metric | Value | Interpretation |
|---|---:|---|
| masked_RMSE | **0.3603** | lower is better |
| insulation_Pearson | **0.9923** | higher is better (structure preserved) |
| coverage95 | **0.9645** | close to 0.95 indicates good calibration |
| spurious_hotspot_control (topk\|resid\|) | **1.0072** | ~1.0 indicates residual hotspots are controlled |

### Baselines (for reference)
| Baseline | masked_RMSE | Note |
|---|---:|---|
| no-enhance | 2.0346 | baseline only (no reconstruction) |
| smooth | 2.0393 | simple smoothing baseline |

## Notes
- `coverage95` uses a Gaussian interval: **μ ± 1.96σ** (target coverage = 0.95).
- `spurious_hotspot_control` is a **ratio** (closer to 1.0 is better).
- Baselines report `masked_RMSE` only; other metrics are model-specific in this demo.
