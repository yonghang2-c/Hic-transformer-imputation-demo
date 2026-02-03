# Hi-C contact map imputation with Transformer-MAE (PyTorch)

This repository provides a **reproducible, minimal-yet-research-oriented demo** for **Hi-C contact map imputation / enhancement** using a **Transformer-based Masked Autoencoder (MAE)** implemented in **PyTorch**.

The demo is designed as a portfolio-style artifact for method-development roles: it includes **clear synthetic ground truth**, **baselines**, **distance-aware evaluation**, and a **false-positive control** (spurious hotspot test).

---

## Why Hi-C contact imputation?

Hi-C contact maps are often **sparse** due to limited sequencing depth (or extremely sparse in single-cell Hi-C).  
Recovering high-quality contact maps from low coverage is a natural **imputation / denoising** problem, but it is also tricky:
models can improve pixel metrics while **hallucinating biological structures** (false loops/TADs).  
This demo explicitly addresses that risk with **structure-aware regularization** and **false-positive control**.

---

## Key ideas / contributions

### 1) Masked inpainting (MAE-style) for imputation
Instead of only learning a mapping from low-coverage → high-coverage, we train a Transformer to **reconstruct randomly masked patches** (inpainting), which directly matches the concept of **missing-data imputation**.

### 2) Hi-C-aware Transformer via 2D relative position bias
We add a **2D relative position bias** in self-attention, making the model more suitable for contact maps where patterns depend strongly on **relative genomic distance and direction**.

### 3) Distance-binned weighting (avoid only learning the diagonal)
Hi-C maps are dominated by the diagonal (short-range contacts).  
We apply **distance-dependent weights** in the loss to upweight longer-range pixels, and report **distance-binned RMSE**.

### 4) Structure-aware regularization (insulation proxy)
We add a light regularizer on a simple **insulation-profile proxy**, encouraging reconstructed maps to preserve TAD-boundary-related structure and reduce spurious structures.

### 5) Uncertainty (heteroscedastic Gaussian NLL)
The model outputs **mean + log-variance**, enabling uncertainty-aware training and potential downstream usage (e.g., “where the model is confident vs uncertain”).

### 6) Baselines + false-positive control
We include two baselines and one explicit control experiment:
- Baseline A: **no enhancement** (use low-coverage map as prediction)
- Baseline B: **smoothing** (simple box-filter on the low-coverage map)
- Control: generate maps with **no loops and no TADs** and quantify **spurious hotspot score** (hallucination tendency)

---

## Project structure

- `src/simulate_hic.py`  
  Synthetic Hi-C generator with distance decay + compartments + (optional) TADs + (optional) loops.
- `src/dataset.py`  
  Downsampling (binomial thinning), distance channel, distance weights/bins, insulation proxy.
- `src/model_mae_vit2d.py`  
  Transformer-MAE with 2D relative position bias; outputs mean + logvar.
- `src/train.py`  
  MAE inpainting training + distance-weighted NLL + insulation regularizer; saves `runs/demo/best.pt`.
- `src/eval.py`  
  Metrics + plots + baselines + false-positive control; writes `runs/demo/metrics.txt`.

---

## Quickstart (macOS)

### 1) Create an environment (recommended: conda)
```bash
conda create -n hicdemo python=3.10 -y
conda activate hicdemo
pip install -r requirements.txt
(Alternative: venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2) Train
python -m src.train
3) Evaluate
python -m src.eval
Outputs
After running, check runs/demo/:
best.pt
Best checkpoint (by validation loss)
metrics.txt
model: masked RMSE/Pearson, insulation correlation, distance-binned RMSE
baselines: RMSE/Pearson
control: spurious hotspot scores (lower is better)
example_true_low_pred.png
Heatmaps: true high-coverage vs low-coverage input vs model prediction
insulation_profile.png
Insulation proxy curves: true vs predicted
Tip: for GitHub, you may copy key images to figures/ and exclude runs/ via .gitignore.
Notes on the synthetic setup
We generate:
a high-depth "true" contact map (Poisson sampling from an intensity surface), then
a low-coverage map by binomial thinning.
For the MAE objective, the model additionally sees randomly masked patches and learns to reconstruct the masked region.
This design enables strict evaluation because the ground-truth map is known.

How this aligns with method-development job requirements
This demo emphasizes:
PyTorch deep learning for imputation
genomics / epigenomics / 3D genome data modality (Hi-C / Micro-C style contact maps)
a reproducible pipeline (simulation → masking → training → evaluation)
attention to biological realism and false-positive control
License
Choose your preferred license (MIT/Apache-2.0). If unsure, MIT is a common default.
Contact
(Your name / email / affiliation)
::contentReference[oaicite:0]{index=0}