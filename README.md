# Hi-C contact map imputation with Mechanism-aware Transformer-MAE (PyTorch)

This repository provides a **reproducible demo / portfolio artifact** for **Hi-C contact map imputation** using a **Transformer-based Masked Autoencoder (MAE)** implemented in **PyTorch**.

The key idea is to turn Hi-C enhancement into a true **imputation (inpainting)** problem: we explicitly **mask** (treat as missing) parts of the contact map and reconstruct only the missing region, while preserving biologically meaningful 3D-genome structure.

---

## What is Hi-C (in one paragraph)

Hi-C measures the **3D proximity** between genomic regions by sequencing ligated DNA fragments that were physically close in the nucleus. The output is a **contact map** (a symmetric matrix), where entry *(i, j)* reflects how frequently genomic bins *i* and *j* are observed to contact each other. Hi-C maps show characteristic patterns such as **distance decay**, **A/B compartments**, **TADs**, and **loops**.

---

## Why imputation for Hi-C?

Hi-C contact maps are often **sparse** at limited sequencing depth (and extremely sparse in single-cell Hi-C).  
Recovering high-quality maps from sparse observations is a natural **missing-data imputation** problem, but it is also risky: models can improve pixel metrics while **hallucinating** biological structures (false loops/TADs). This demo therefore emphasizes:

- **Mechanism-aware missingness** (Hi-C missingness is distance-dependent)
- **Structure preservation** (insulation proxy)
- **Uncertainty + calibration** (coverage of prediction intervals)
- **False-positive control** (spurious hotspot test on “no loop/no TAD” maps)

---

## Key contributions / novelty (demo)

### 1) Mechanism-aware masking (missingness mechanism)
Hi-C sparsity is **not MCAR**: long-range contacts are typically rarer.  
We implement **distance-biased masking** (farther from diagonal → higher missing probability) and allow:
- `mask_mode=random` (standard MAE)
- `mask_mode=dist` (distance-biased / MNAR-ish)
- `mask_mode=mixed` (mixture of random + distance-biased)

### 2) Transformer-MAE inpainting + Hi-C-aware attention
We use a MAE-style Transformer (ViT-like) with **2D relative position bias** in self-attention, which is better aligned with contact-map structure.

### 3) Structure-preserving objective
We add a light regularizer on an **insulation-profile proxy**, encouraging reconstructed maps to preserve TAD-boundary-related structure.

### 4) Uncertainty-aware imputation with calibration
The model outputs **mean + log-variance**, and evaluation reports **95% prediction-interval coverage** (`coverage95`) overall and by distance bins.

### 5) False-positive control
We test “hallucination risk” on synthetic maps generated **without loops/TADs**, reporting a **spurious hotspot** score (top-k absolute residual magnitude off-diagonal; lower is better).

---

## Project structure

- `src/simulate_hic.py`  
  Synthetic Hi-C generator (distance decay + compartments + optional TADs + optional loops).
- `src/dataset.py`  
  Downsampling (binomial thinning), distance channel, distance bins/weights, insulation proxy.
- `src/model_mae_vit2d.py`  
  Transformer-MAE with 2D relative position bias + mechanism-aware masking; outputs mean + logvar.
- `src/train.py`  
  Training with distance-weighted NLL (+ optional L1), insulation regularization; saves `model.pt` + `cfg.json`.
- `src/eval.py`  
  Metrics + calibration + imputation visualization (4-panel figure), plus false-positive control.

---

## Installation (macOS)

### Option A: conda (recommended)

```bash
conda create -n hicdemo python=3.10 -y
conda activate hicdemo
pip install -r requirements.txt
```

### Option B: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

### 1) Train a single run (recommended default: mechanism-aware mixed masking)

```bash
python -m src.train \
  --patch 4 --mask_ratio 0.4 \
  --mask_mode mixed --mixed_prob 0.7 --dist_k 3.0 \
  --lambda_insul 0.02 --beta_l1 0.1 \
  --dist_gamma 0.5 --epochs 30 --seed 104
```

The training script prints a run directory, e.g.:

```
Done. Run directory: runs/<run_name>
```

### 2) Evaluate

```bash
python -m src.eval --run_dir runs/<run_name>
```

Outputs in the run directory:
- `metrics.txt` / `metrics.json`
- `imputation_4panel.png` (true / input-with-holes / pred / masked-error)
- `insulation_profile.png`

---

## Lightweight hyperparameter tuning (demo)

This repository is a **demo / portfolio artifact**, not a full benchmark paper.
To improve qualitative outputs and stability, we performed a **small 8-run sweep** over a few key hyperparameters:

- masking mechanism: `random` vs `distance-biased` vs `mixed`
- mechanism strength: `dist_k`
- structure regularization: `lambda_insul`
- peak-preserving term: `beta_l1`

Each run outputs `metrics.txt/metrics.json` and the imputation visualization (`imputation_4panel.png`).
Selection is based primarily on **masked RMSE**, with secondary checks on **insulation correlation**, **95% coverage calibration**, and **false-positive control**.

A short summary of the 8-run mini-sweep and the selected configuration is provided in:
- `results/summary.md`

To reproduce the 8-run mini-sweep:

```bash
chmod +x sweep8.sh
./sweep8.sh
```

---

## Selected configurations (example)

### Best accuracy/structure (distance-biased masking)

A strong `mask_mode=dist` configuration typically achieves:
- masked_RMSE around **0.36** (baseline no-enhance around **2.0**)
- insulation_Pearson around **0.99**
- coverage95 around **0.96**

### Conservative variant (lower hallucination risk)

A “conservative” run is chosen by a lower **spurious hotspot control score**, indicating reduced tendency to produce extreme off-diagonal artifacts on “no loop/no TAD” control maps.

---

## Example outputs (place in `figures/` for GitHub display)

Then the images can be displayed on GitHub as:

[Best imputation]![imputation_best.png](figures%20/imputation_best.png)

[Best insulation]![insulation_best.png](figures%20/insulation_best.png)

[Conservative imputation]![imputation_conservative.png](figures%20/imputation_conservative.png)

---

## Notes on evaluation metrics

- **masked_RMSE / masked_Pearson**: computed only on masked pixels, consistent with an imputation setting.
- **distance_binned_RMSE**: RMSE stratified by genomic distance bins.
- **coverage95**: fraction of masked pixels where the true value lies within `mu ± 1.96 * sigma`.
- **spurious hotspot control**: top-k absolute residual magnitude off-diagonal on maps generated without loops/TADs.

---

## Disclaimer

This codebase is intended as a demonstration of a method-development workflow (simulation → mechanism-aware masking → imputation model → structure & calibration evaluation). It is not a full production pipeline and does not claim to outperform specialized tools on all real datasets.

---

## License

MIT License. See `LICENSE`.

---

## Contact

Yonghang Lai / lai.yonghang@nies.go.jp
