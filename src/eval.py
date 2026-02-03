import os
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

def binned_rmse(a, b, m, bin_id, n_bins):
    out = []
    for k in range(n_bins):
        mk = m * (bin_id == k).astype(np.float32)
        if mk.sum() < 10:
            out.append(float("nan"))
        else:
            out.append(masked_rmse(a, b, mk))
    return out

def smooth_baseline(x_low_log, k=5):
    """
    Simple box filter smoothing baseline on (H,W) numpy array.
    """
    assert k % 2 == 1
    pad = k // 2
    t = torch.from_numpy(x_low_log)[None, None]  # (1,1,H,W)
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    w = torch.ones(1, 1, k, k) / (k * k)
    y = F.conv2d(t, w)
    return y[0, 0].numpy()

def spurious_hotspot_score(pred, truth, diag_band=3, topk=50):
    """
    False-positive proxy: on maps that should NOT have loops/TADs,
    measure how large the top-k positive residuals are away from the diagonal.
    Larger => more spurious hotspots.
    """
    H = pred.shape[0]
    ii = np.arange(H)
    d = np.abs(ii[:, None] - ii[None, :])
    off = (d > diag_band)

    resid = (pred - truth)
    vals = resid[off].reshape(-1)
    if vals.size == 0:
        return float("nan")
    vals = np.sort(vals)[::-1]
    vals = vals[:min(topk, vals.size)]
    return float(np.mean(vals))

@torch.no_grad()
def run_model(model, x, y, device):
    mu, logvar, mpatch = model(x)
    mpix = patch_mask_to_pixel(mpatch, H=y.shape[-2], W=y.shape[-1], patch=model.patch).to(device)
    return mu, logvar, mpix

def main(outdir="runs/demo", idx=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(outdir, "best.pt"), map_location="cpu")
    cfg = ckpt["cfg"]
    n = cfg["n"]

    model = MAEViT2D(
        img_size=n,
        patch=cfg["patch"],
        mask_ratio=cfg["mask_ratio"],
        enc_dim=cfg["enc_dim"],
        enc_depth=cfg["enc_depth"],
        enc_heads=cfg["enc_heads"],
        dec_dim=cfg["dec_dim"],
        dec_depth=cfg["dec_depth"],
        dec_heads=cfg["dec_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # -------- main evaluation sample (with loops + TAD) --------
    ds = HiCDemoDataset(
        n_maps=10, n=n,
        depth_true=cfg["depth_true"],
        keep_rate=cfg["keep_rate"],
        seed=123,
        enable_tad=True,
        enable_loops=True,
    )
    batch = ds[idx]
    x = batch["x"][None].to(device).float()
    y = batch["y"][None].to(device).float()

    mu, logvar, mpix = run_model(model, x, y, device)

    y_np  = y.cpu().numpy()[0, 0]
    mu_np = mu.cpu().numpy()[0, 0]
    x_low = x.cpu().numpy()[0, 0]  # channel0: log1p(low)
    m_np  = mpix.cpu().numpy()[0, 0].astype(np.float32)

    # metrics (model)
    rmse = masked_rmse(mu_np, y_np, m_np)
    pcc  = masked_pearson(mu_np, y_np, m_np)
    ins_true = insulation_profile(y[0, 0], w=6).cpu().numpy()
    ins_pred = insulation_profile(mu[0, 0], w=6).cpu().numpy()
    ins_pcc = float(np.corrcoef(ins_true, ins_pred)[0, 1])

    # distance bins
    _, bin_id, edges, n_bins = distance_weight_map(n, n_bins=cfg["dist_bins"], gamma=cfg["dist_gamma"])
    b_rmse = binned_rmse(mu_np, y_np, m_np, bin_id, n_bins)

    # -------- baselines --------
    base1 = x_low.copy()
    base1_rmse = masked_rmse(base1, y_np, m_np)
    base1_pcc  = masked_pearson(base1, y_np, m_np)

    base2 = smooth_baseline(x_low, k=5)
    base2_rmse = masked_rmse(base2, y_np, m_np)
    base2_pcc  = masked_pearson(base2, y_np, m_np)

    # -------- false-positive control (no loops, no TAD) --------
    ctrl = HiCDemoDataset(
        n_maps=1, n=n,
        depth_true=cfg["depth_true"],
        keep_rate=cfg["keep_rate"],
        seed=999,
        enable_tad=False,
        enable_loops=False,
    )[0]
    x_c = ctrl["x"][None].to(device)
    y_c = ctrl["y"][None].to(device)

    mu_c, _, _ = run_model(model, x_c, y_c, device)
    y_c_np  = y_c.cpu().numpy()[0, 0]
    mu_c_np = mu_c.cpu().numpy()[0, 0]
    x_c_low = x_c.cpu().numpy()[0, 0]

    spurious_model = spurious_hotspot_score(mu_c_np, y_c_np, diag_band=3, topk=50)
    spurious_base1 = spurious_hotspot_score(x_c_low, y_c_np, diag_band=3, topk=50)
    spurious_base2 = spurious_hotspot_score(smooth_baseline(x_c_low, k=5), y_c_np, diag_band=3, topk=50)

    # -------- write report --------
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("=== Model (MAE+RPB+uncertainty+insulation+distance-weight) ===\n")
        f.write(f"masked_RMSE={rmse:.4f}\n")
        f.write(f"masked_Pearson={pcc:.4f}\n")
        f.write(f"insulation_Pearson={ins_pcc:.4f}\n")
        f.write("distance_binned_RMSE:\n")
        for k, val in enumerate(b_rmse):
            upper = edges[k] if k < len(edges) else edges[-1]
            f.write(f"  bin{k}_upper{upper}: {val}\n")

        f.write("\n=== Baselines (on same masked region) ===\n")
        f.write(f"baseline_no_enhance_RMSE={base1_rmse:.4f}  Pearson={base1_pcc:.4f}\n")
        f.write(f"baseline_smooth_RMSE={base2_rmse:.4f}     Pearson={base2_pcc:.4f}\n")

        f.write("\n=== False-positive control (no loops/no TAD) ===\n")
        f.write("spurious_hotspot_score (mean top-k positive residuals off-diagonal; lower is better)\n")
        f.write(f"model={spurious_model:.4f}\n")
        f.write(f"baseline_no_enhance={spurious_base1:.4f}\n")
        f.write(f"baseline_smooth={spurious_base2:.4f}\n")

    # -------- plots --------
    plt.figure(figsize=(12, 4))
    for i, (arr, title) in enumerate(
        [(y_np, "true log1p(high)"),
         (x_low, "input log1p(low)"),
         (mu_np, "pred log1p(high)")],
        start=1,
    ):
        plt.subplot(1, 3, i)
        plt.imshow(arr, aspect="auto")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "example_true_low_pred.png"), dpi=200)

    plt.figure()
    plt.plot(ins_true, label="true insulation proxy")
    plt.plot(ins_pred, label="pred insulation proxy")
    plt.legend()
    plt.title("Insulation profile proxy")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "insulation_profile.png"), dpi=200)

    print("Saved metrics and plots to:", outdir)

if __name__ == "__main__":
    main()
