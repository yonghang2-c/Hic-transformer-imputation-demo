import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .simulate_hic import simulate_hic_map, poisson_counts

def log1p_norm(C):
    return np.log1p(C).astype(np.float32)

def make_low_coverage(C_true, keep_rate=0.1, seed=42):
    rng = np.random.default_rng(seed)
    C_low = rng.binomial(C_true.astype(np.int64), keep_rate).astype(np.float32)
    C_low = (C_low + C_low.T) / 2.0
    return C_low

def insulation_profile(M, w=5):
    n = M.shape[0]
    vals = []
    for i in range(n):
        a0 = max(0, i - w); a1 = i
        b0 = i; b1 = min(n, i + w)
        if a1 - a0 < 1 or b1 - b0 < 1:
            vals.append(torch.tensor(0.0, device=M.device))
        else:
            vals.append(M[a0:a1, b0:b1].mean())
    return torch.stack(vals)

def make_distance_bins(n: int, n_bins: int = 6):
    maxd = n - 1
    raw = np.logspace(0, np.log10(maxd), n_bins + 1)
    edges = np.unique(np.round(raw).astype(int))
    edges[0] = 1
    edges[-1] = maxd
    bin_uppers = [0] + edges.tolist()
    bin_uppers = sorted(set(bin_uppers))
    if bin_uppers[-1] != maxd:
        bin_uppers.append(maxd)
    return bin_uppers

def distance_weight_map(n: int, n_bins: int = 6, gamma: float = 0.5):
    ii = np.arange(n)
    d = np.abs(ii[:, None] - ii[None, :]).astype(np.int32)

    w = (d.astype(np.float32) + 1.0) ** float(gamma)
    w = w / (w.mean() + 1e-12)

    uppers = make_distance_bins(n, n_bins=n_bins)
    bin_id = np.zeros_like(d, dtype=np.int32)
    prev = -1
    b = 0
    for up in uppers:
        sel = (d > prev) & (d <= up)
        bin_id[sel] = b
        prev = up
        b += 1
    return w.astype(np.float32), bin_id, uppers, b

class HiCDemoDataset(Dataset):
    def __init__(self, n_maps=200, n=128, depth_true=3e6, keep_rate=0.1, seed=42,
                 enable_tad=True, enable_loops=True):
        self.n_maps = int(n_maps)
        self.n = int(n)
        self.depth_true = float(depth_true)
        self.keep_rate = float(keep_rate)
        self.seed = int(seed)
        self.enable_tad = bool(enable_tad)
        self.enable_loops = bool(enable_loops)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        seed = self.seed + idx * 17
        X_int, _ = simulate_hic_map(
            n=self.n, seed=seed,
            enable_tad=self.enable_tad,
            enable_loops=self.enable_loops
        )
        C_true = poisson_counts(X_int, depth=self.depth_true, seed=seed + 1)
        C_low  = make_low_coverage(C_true, keep_rate=self.keep_rate, seed=seed + 2)

        n = self.n
        ii = np.arange(n)
        dist = np.abs(ii[:, None] - ii[None, :]).astype(np.float32) + 1.0
        dist_ch = (np.log(dist) / np.log(n + 1.0)).astype(np.float32)

        x_in = np.stack([log1p_norm(C_low), dist_ch], axis=0).astype(np.float32)
        y_true = log1p_norm(C_true)[None, ...].astype(np.float32)

        return {"x": torch.from_numpy(x_in).float(),
                "y": torch.from_numpy(y_true).float()}

def make_loaders(batch_size=8, **kwargs):
    ds = HiCDemoDataset(**kwargs)
    n = len(ds)
    n_tr = int(n * 0.8)
    n_va = n - n_tr
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_va])
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0)
    return tr_loader, va_loader
