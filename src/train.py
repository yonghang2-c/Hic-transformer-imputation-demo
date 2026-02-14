import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import make_loaders, insulation_profile, distance_weight_map
from .model_mae_vit2d import MAEViT2D
from .utils import set_seed, save_json, format_run_name

def pick_device(device: str):
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

def patch_mask_to_pixel(mask_patch, H, W, patch):
    B, L = mask_patch.shape
    h = H // patch
    w = W // patch
    m = mask_patch.reshape(B, h, w, 1, 1).repeat(1, 1, 1, patch, patch)
    return m.reshape(B, 1, H, W)

def gaussian_nll_weighted(mu, logvar, y, mask_pix, wdist):
    var = torch.exp(logvar) + 1e-6
    nll = 0.5 * (logvar + (y - mu) ** 2 / var)
    w = wdist * mask_pix
    denom = w.sum().clamp_min(1.0)
    return (nll * w).sum() / denom

def l1_weighted(mu, y, mask_pix, wdist):
    w = wdist * mask_pix
    denom = w.sum().clamp_min(1.0)
    return (torch.abs(mu - y) * w).sum() / denom

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)

    # data
    ap.add_argument("--n_maps", type=int, default=400)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--depth_true", type=float, default=3e6)
    ap.add_argument("--keep_rate", type=float, default=0.08)

    # train
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_insul", type=float, default=0.02)
    ap.add_argument("--beta_l1", type=float, default=0.1)

    # model
    ap.add_argument("--patch", type=int, default=4, choices=[2, 4, 8, 16])
    ap.add_argument("--mask_ratio", type=float, default=0.4)

    # mechanism-aware masking
    ap.add_argument("--mask_mode", default="mixed", choices=["random", "dist", "mixed"])
    ap.add_argument("--mixed_prob", type=float, default=0.7)
    ap.add_argument("--dist_k", type=float, default=3.0)

    # distance weighting
    ap.add_argument("--dist_bins", type=int, default=6)
    ap.add_argument("--dist_gamma", type=float, default=0.5)

    # output
    ap.add_argument("--runs_dir", default="runs")

    args = ap.parse_args()

    cfg = {
        "seed": args.seed,
        "n_maps": args.n_maps,
        "n": args.n,
        "depth_true": args.depth_true,
        "keep_rate": args.keep_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_insul": args.lambda_insul,
        "beta_l1": args.beta_l1,
        "patch": args.patch,
        "mask_ratio": args.mask_ratio,
        "mask_mode": args.mask_mode,
        "mixed_prob": args.mixed_prob,
        "dist_k": args.dist_k,
        "dist_bins": args.dist_bins,
        "dist_gamma": args.dist_gamma,
    }

    run_name = format_run_name(cfg)
    outdir = os.path.join(args.runs_dir, run_name)
    os.makedirs(outdir, exist_ok=True)

    set_seed(args.seed)
    device = pick_device(args.device)

    tr_loader, va_loader = make_loaders(
        batch_size=args.batch_size,
        n_maps=args.n_maps,
        n=args.n,
        depth_true=args.depth_true,
        keep_rate=args.keep_rate,
        seed=args.seed,
        enable_tad=True,
        enable_loops=True,
    )

    w_np, _, edges, _ = distance_weight_map(args.n, n_bins=args.dist_bins, gamma=args.dist_gamma)
    wdist = torch.from_numpy(w_np)[None, None, :, :].to(device)

    model = MAEViT2D(
        in_ch=2, out_ch=1, img_size=args.n, patch=args.patch,
        enc_dim=256, enc_depth=6, enc_heads=8,
        dec_dim=192, dec_depth=4, dec_heads=6,
        mask_ratio=args.mask_ratio, drop=0.0,
        mask_mode=args.mask_mode, mixed_prob=args.mixed_prob, dist_k=args.dist_k
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 1e18
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in tqdm(tr_loader, desc=f"epoch {ep} train"):
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()

            opt.zero_grad()
            mu, logvar, mpatch = model(x)
            mpix = patch_mask_to_pixel(mpatch, H=args.n, W=args.n, patch=model.patch).to(device)

            loss = gaussian_nll_weighted(mu, logvar, y, mpix, wdist)
            if args.beta_l1 > 0:
                loss = loss + args.beta_l1 * l1_weighted(mu, y, mpix, wdist)

            ins_mu = torch.stack([insulation_profile(mu[b, 0], w=6) for b in range(mu.shape[0])], dim=0)
            ins_y  = torch.stack([insulation_profile(y[b, 0],  w=6) for b in range(y.shape[0])], dim=0)
            loss = loss + args.lambda_insul * F.l1_loss(ins_mu, ins_y)

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
                mpix = patch_mask_to_pixel(mpatch, H=args.n, W=args.n, patch=model.patch).to(device)
                loss = gaussian_nll_weighted(mu, logvar, y, mpix, wdist)
                if args.beta_l1 > 0:
                    loss = loss + args.beta_l1 * l1_weighted(mu, y, mpix, wdist)
                va_loss += float(loss.item())

        va_loss /= max(1, len(va_loader))
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"[epoch {ep}] train={tr_loss:.6f} val={va_loss:.6f}")

        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))

            cfg_out = dict(cfg)
            cfg_out.update({
                "device": device,
                "dist_edges": edges,
                "model": {"enc_dim": 256, "enc_depth": 6, "enc_heads": 8,
                          "dec_dim": 192, "dec_depth": 4, "dec_heads": 6}
            })
            save_json(cfg_out, os.path.join(outdir, "cfg.json"))
            save_json({"best_val_loss": best, "history": history}, os.path.join(outdir, "train_log.json"))
            print("  saved model.pt + cfg.json")

    print("Done. Run directory:", outdir)

if __name__ == "__main__":
    main()
