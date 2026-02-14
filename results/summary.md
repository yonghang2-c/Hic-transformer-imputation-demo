# Demo summary (8-run mini-sweep)

## Selected configuration (best accuracy/structure)
- mask_mode: dist (distance-biased missingness)
- dist_k: 3.0
- patch: 4, mask_ratio: 0.4
- lambda_insul: 0.02, beta_l1: 0.1, dist_gamma: 0.5

### Metrics
- masked_RMSE: 0.3603
- insulation_Pearson: 0.9923
- coverage95: 0.9645
- spurious_hotspot_control(topk|resid|): 1.0072
- baseline(no-enhance) masked_RMSE: 2.0346
- baseline(smooth) masked_RMSE: 2.0393
