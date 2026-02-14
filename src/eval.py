import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .dataset import HiCDemoDataset, insulation_profile, distance_weight_map
from .model_mae_vit2d import MAEViT2D
from .train import patch_mask_to_pixel

def masked_rmse(a, b, m):
    diff = (a - b) * m
    num = m.sum().clip(min=1.0)
    return float(np.sqrt((diff**2).sum() / num))

def masked_pearson(a, b, m):
    x = (a[m == 1]).reshape(-1)
    y = (b[m == 1]).reshape(-1)
    if x.size < 5:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()) + 1e-12)
    return float((x * y).sum() / denom)

def binned_stat(values, mask, bin_id, n_bins, fn="mean"):
    out = []
    for k in range(n_bins):
        mk = mask * (bin_id == k).astype(np.float32)
        if mk.sum() < 10:
            out.append(float("nan"))
            continue
        v = values[mk == 1]
        if fn == "mean":
            out.append(float(np.mean(v)))
        elif fn == "rmse":
            out.append(float(np.sqrt(np.mean(v * v))))
        else:
            raise ValueError
    return out

def smooth_baseline(x_low_log, k=5):
    assert k % 2 == 1
    pad = k // 2
    t = torch.from_numpy(x_low_log)[None, None]
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    w = torch.ones(1, 1, k, k) / (k * k)
    y = F.conv2d(t, w)
    return y[0, 0].numpy()

def spurious_hotspot_score(pred, truth, diag_band=3, topk=50):
    H = pred.shape[0]
    ii = np.arange(H)
    d = np.abs(ii[:, None] - ii[None, :])
    off = (d > diag_band)
    resid = pred - truth
    vals = np.abs(resid[off].reshape(-1))
    if vals.size == 0:
        return float("nan")
    vals = np.sort(vals)[::-1][:min(topk, vals.size)]
    return float(np.mean(vals))

def coverage_95(mu, var, y, m):
    sd = np.sqrt(np.maximum(var, 1e-8))
    lo = mu - 1.96 * sd
    hi = mu + 1.96 * sd
    ok = ((y >= lo) & (y <= hi) & (m == 1))
    denom = np.sum(m == 1)
    return float(np.sum(ok) / max(1, denom))

@torch.no_grad()
def run_model(model, x, y, device):
    mu, logvar, mpatch = model(x)
    mpix = patch_mask_to_pixel(mpatch, H=y.shape[-2], W=y.shape[-1], patch=model.patch).to(device)
    return mu, logvar, mpix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    def pick_device(d):
        if d != "auto":
            return d
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = pick_device(args.device)

    with open(os.path.join(args.run_dir, "cfg.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    n = int(cfg["n"])
    model = MAEViT2D(
        img_size=n,
        patch=int(cfg["patch"]),
        mask_ratio=float(cfg["mask_ratio"]),
        enc_dim=int(cfg["model"]["enc_dim"]),
        enc_depth=int(cfg["model"]["enc_depth"]),
        enc_heads=int(cfg["model"]["enc_heads"]),
        dec_dim=int(cfg["model"]["dec_dim"]),
        dec_depth=int(cfg["model"]["dec_depth"]),
        dec_heads=int(cfg["model"]["dec_heads"]),
        mask_mode=str(cfg["mask_mode"]),
        mixed_prob=float(cfg["mixed_prob"]),
        dist_k=float(cfg["dist_k"]),
    ).to(device)

    state = torch.load(os.path.join(args.run_dir, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    ds = HiCDemoDataset(
        n_maps=10, n=n,
        depth_true=float(cfg["depth_true"]),
        keep_rate=float(cfg["keep_rate"]),
        seed=123,
        enable_tad=True,
        enable_loops=True,
    )
    batch = ds[args.idx]
    x = batch["x"][None].to(device).float()
    y = batch["y"][None].to(device).float()

    torch.manual_seed(args.seed)
    mu, logvar, mpix = run_model(model, x, y, device)

    y_np = y.cpu().numpy()[0, 0]
    mu_np = mu.cpu().numpy()[0, 0]
    var_np = np.exp(logvar.cpu().numpy()[0, 0])
    x_low = x.cpu().numpy()[0, 0]
    m_np = mpix.cpu().numpy()[0, 0].astype(np.float32)

    rmse = masked_rmse(mu_np, y_np, m_np)
    pcc = masked_pearson(mu_np, y_np, m_np)

    ins_true = insulation_profile(y[0, 0], w=6).cpu().numpy()
    ins_pred = insulation_profile(mu[0, 0], w=6).cpu().numpy()
    ins_pcc = float(np.corrcoef(ins_true, ins_pred)[0, 1])

    _, bin_id, edges, n_bins = distance_weight_map(n, n_bins=int(cfg["dist_bins"]), gamma=float(cfg["dist_gamma"]))
    resid = (mu_np - y_np)
    b_rmse = binned_stat(resid, m_np, bin_id, n_bins, fn="rmse")

    cov95 = coverage_95(mu_np, var_np, y_np, m_np)
    sd = np.sqrt(np.maximum(var_np, 1e-8))
    lo = mu_np - 1.96 * sd
    hi = mu_np + 1.96 * sd
    ok_map = ((y_np >= lo) & (y_np <= hi)).astype(np.float32)
    b_cov = binned_stat(ok_map, m_np, bin_id, n_bins, fn="mean")

    base1_rmse = masked_rmse(x_low, y_np, m_np)
    base2_rmse = masked_rmse(smooth_baseline(x_low, k=5), y_np, m_np)

    ctrl = HiCDemoDataset(
        n_maps=1, n=n,
        depth_true=float(cfg["depth_true"]),
        keep_rate=float(cfg["keep_rate"]),
        seed=999,
        enable_tad=False,
        enable_loops=False,
    )[0]
    x_c = ctrl["x"][None].to(device).float()
    y_c = ctrl["y"][None].to(device).float()
    torch.manual_seed(args.seed)
    mu_c, _, _ = run_model(model, x_c, y_c, device)
    spurious = spurious_hotspot_score(mu_c.cpu().numpy()[0,0], y_c.cpu().numpy()[0,0])

    # 4-panel imputation visualization
    x_hole = x_low.copy()
    hole_value = np.nanmin(x_low) - 0.5
    x_hole[m_np == 1] = hole_value
    err_mask = np.abs(mu_np - y_np) * m_np

    plt.figure(figsize=(14, 4))
    panels = [
        (y_np, "true log1p(high)"),
        (x_hole, "input log1p(low) with masked holes"),
        (mu_np, "pred log1p(high)"),
        (err_mask, "|pred-true| on masked region"),
    ]
    for i, (arr, title) in enumerate(panels, start=1):
        plt.subplot(1, 4, i)
        plt.imshow(arr, aspect="auto")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "imputation_best.png"), dpi=200)

    plt.figure()
    plt.plot(ins_true, label="true insulation proxy")
    plt.plot(ins_pred, label="pred insulation proxy")
    plt.legend()
    plt.title("Insulation profile proxy")
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "insulation_best.png"), dpi=200)

    metrics = {
        "masked_RMSE": rmse,
        "masked_Pearson": pcc,
        "insulation_Pearson": ins_pcc,
        "coverage95": cov95,
        "distance_binned_RMSE": [{"bin": k, "upper": int(edges[k]) if k < len(edges) else int(edges[-1]), "rmse": float(v)} for k, v in enumerate(b_rmse)],
        "distance_binned_coverage95": [{"bin": k, "upper": int(edges[k]) if k < len(edges) else int(edges[-1]), "coverage": float(v)} for k, v in enumerate(b_cov)],
        "spurious_hotspot_control": spurious,
        "baseline_no_enhance_masked_RMSE": base1_rmse,
        "baseline_smooth_masked_RMSE": base2_rmse,
        "mask_mode": cfg["mask_mode"],
        "mixed_prob": cfg["mixed_prob"],
        "dist_k": cfg["dist_k"],
    }

    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.run_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("=== Model (Mechanism-aware MAE Imputation) ===\n")
        f.write(f"mask_mode={cfg['mask_mode']} mixed_prob={cfg['mixed_prob']} dist_k={cfg['dist_k']}\n")
        f.write(f"masked_RMSE={rmse:.4f}\n")
        f.write(f"masked_Pearson={pcc:.4f}\n")
        f.write(f"insulation_Pearson={ins_pcc:.4f}\n")
        f.write(f"coverage95={cov95:.4f}\n")
        f.write("distance_binned_RMSE:\n")
        for k, v in enumerate(b_rmse):
            upper = edges[k] if k < len(edges) else edges[-1]
            f.write(f"  bin{k}_upper{upper}: {v}\n")
        f.write("distance_binned_coverage95:\n")
        for k, v in enumerate(b_cov):
            upper = edges[k] if k < len(edges) else edges[-1]
            f.write(f"  bin{k}_upper{upper}: {v}\n")
        f.write(f"\nspurious_hotspot_control(topk|resid|)={spurious:.4f}\n")
        f.write("\n=== Baselines (masked RMSE) ===\n")
        f.write(f"no_enhance={base1_rmse:.4f}\n")
        f.write(f"smooth={base2_rmse:.4f}\n")

    print("Saved metrics and plots to:", args.run_dir)

if __name__ == "__main__":
    main()
