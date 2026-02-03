import numpy as np

def _smooth1d(x, k=9):
    k = max(3, int(k) | 1)
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, w, mode="same")

def simulate_hic_map(n=128, seed=42, enable_tad=True, enable_loops=True):
    """
    Simulate a symmetric Hi-C intensity map with:
    - distance decay
    - A/B compartments
    - (optional) TAD blocks
    - (optional) loops (hotspots)

    Returns:
      X_int: (n,n) float32 nonnegative intensity
      meta: dict with ab, cuts, loop_pairs
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float32) + 1.0

    # distance decay baseline
    alpha = 1.2 + 0.3 * rng.random()
    base = 2.5 / (dist ** alpha)

    # compartments (A/B): low-frequency sign pattern
    ab = rng.normal(size=n).astype(np.float32)
    ab = _smooth1d(ab, k=max(9, n // 8))
    ab = np.sign(ab + 1e-6).astype(np.float32)  # +/-1
    comp = 1.0 + 0.35 * (ab[:, None] * ab[None, :])

    # TAD blocks
    cuts = [0, n]
    tad = np.ones((n, n), dtype=np.float32)
    if enable_tad:
        n_domains = int(rng.integers(4, 9))
        cuts = sorted(rng.choice(np.arange(10, n - 10), size=n_domains - 1, replace=False).tolist())
        cuts = [0] + cuts + [n]
        for a, b in zip(cuts[:-1], cuts[1:]):
            tad[a:b, a:b] += 0.8 + 0.5 * rng.random()

    # loops: sparse hotspots
    loops = np.zeros((n, n), dtype=np.float32)
    loop_pairs = []
    if enable_loops:
        n_loops = int(rng.integers(8, 20))
        for _ in range(n_loops):
            i = int(rng.integers(0, n))
            j = int(np.clip(i + rng.integers(-n // 3, n // 3), 0, n - 1))
            if i == j:
                continue
            amp = 3.0 + 4.0 * rng.random()
            loops[i, j] += amp
            loops[j, i] += amp
            loop_pairs.append((min(i, j), max(i, j)))

    X = base * comp * tad + loops
    X = (X + X.T) / 2.0
    X = np.clip(X, 0, None).astype(np.float32)

    meta = {"ab": ab, "cuts": cuts, "loop_pairs": loop_pairs}
    return X, meta

def poisson_counts(X_intensity, depth=3e6, seed=42):
    """
    Convert intensity -> counts via Poisson sampling.
    """
    rng = np.random.default_rng(seed)
    P = X_intensity / (X_intensity.sum() + 1e-12)
    lam = P * float(depth)
    C = rng.poisson(lam).astype(np.float32)
    C = (C + C.T) / 2.0
    return C
