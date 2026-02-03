import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import make_loaders, insulation_profile, distance_weight_map
from .model_mae_vit2d import MAEViT2D

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def patch_mask_to_pixel(mask_patch, H, W, patch):
    B, L = mask_patch.shape
    h = H // patch
    w = W // patch
    m = mask_patch.reshape(B, h, w, 1, 1).repeat(1, 1, 1, patch, patch)
    return m.reshape(B, 1, H, W)

def gaussian_nll_weighted(mu, logvar, y, mask_pix, wdist):
    """
    mu, logvar, y: (B,1,H,W)
    mask_pix: (B,1,H,W) in {0,1}
    wdist: (1,1,H,W) positive weights (mean ~1)
    """
    var = torch.exp(logvar) + 1e-6
    nll = 0.5 * (logvar + (y - mu) ** 2 / var)

    w = wdist * mask_pix
    denom = w.sum().clamp_min(1.0)
    return (nll * w).sum() / denom

def main(
    outdir="runs/demo",
    seed=42,
    n_maps=400,
    n=128,
    depth_true=3e6,
    keep_rate=0.08,
    batch_size=8,
    epochs=25,
    lr=2e-4,
    mask_ratio=0.5,
    lambda_insul=0.05,
    dist_bins=6,
    dist_gamma=0.5,
):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_loader, va_loader = make_loaders(
        batch_size=batch_size,
        n_maps=n_maps,
        n=n,
        depth_true=depth_true,
        keep_rate=keep_rate,
        seed=seed,
        enable_tad=True,
        enable_loops=True,
    )

    w_np, _, edges, _ = distance_weight_map(n, n_bins=dist_bins, gamma=dist_gamma)
    wdist = torch.from_numpy(w_np)[None, None, :, :]  # (1,1,n,n)

    device = pick_device()
    wdist = wdist.to(device)

    model = MAEViT2D(
        in_ch=2, out_ch=1, img_size=n, patch=8,
        enc_dim=256, enc_depth=6, enc_heads=8,
        dec_dim=192, dec_depth=4, dec_heads=6,
        mask_ratio=mask_ratio, drop=0.0
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best = 1e18
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in tqdm(tr_loader, desc=f"epoch {ep} train"):
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()

            opt.zero_grad()
            mu, logvar, mpatch = model(x)
            mpix = patch_mask_to_pixel(mpatch, H=n, W=n, patch=model.patch).to(device)

            loss_pix = gaussian_nll_weighted(mu, logvar, y, mpix, wdist)

            # structure-aware: insulation proxy
            ins_mu = torch.stack([insulation_profile(mu[b, 0], w=6) for b in range(mu.shape[0])], dim=0)
            ins_y  = torch.stack([insulation_profile(y[b, 0],  w=6) for b in range(y.shape[0])], dim=0)
            loss_ins = F.l1_loss(ins_mu, ins_y)

            loss = loss_pix + lambda_insul * loss_ins
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item())

        tr_loss /= max(1, len(tr_loader))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(va_loader, desc=f"epoch {ep} val"):
                x = batch["x"].to(device).float()
                y = batch["y"].to(device).float()
                mu, logvar, mpatch = model(x)
                mpix = patch_mask_to_pixel(mpatch, H=n, W=n, patch=model.patch).to(device)
                va_loss += float(gaussian_nll_weighted(mu, logvar, y, mpix, wdist).item())
        va_loss /= max(1, len(va_loader))

        print(f"[epoch {ep}] train={tr_loss:.6f} val_weighted_nll={va_loss:.6f}")

        if va_loss < best:
            best = va_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": {
                        "n": n,
                        "patch": model.patch,
                        "mask_ratio": mask_ratio,
                        "keep_rate": keep_rate,
                        "depth_true": depth_true,
                        "enc_dim": 256,
                        "enc_depth": 6,
                        "enc_heads": 8,
                        "dec_dim": 192,
                        "dec_depth": 4,
                        "dec_heads": 6,
                        "dist_bins": dist_bins,
                        "dist_gamma": dist_gamma,
                        "dist_edges": edges,
                    },
                },
                os.path.join(outdir, "best.pt"),
            )
            print("  saved best.pt")

if __name__ == "__main__":
    main()
