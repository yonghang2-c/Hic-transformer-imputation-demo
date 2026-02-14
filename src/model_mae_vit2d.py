import torch
import torch.nn as nn

def patchify(x, p):
    B, C, H, W = x.shape
    assert H % p == 0 and W % p == 0
    h = H // p
    w = W // p
    x = x.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
    return x

def unpatchify(tokens, p, H, W, out_c):
    B, L, D = tokens.shape
    h = H // p
    w = W // p
    x = tokens.reshape(B, h, w, p, p, out_c)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, out_c, H, W)
    return x

class RelPosBias2D(nn.Module):
    def __init__(self, Hp: int, Wp: int, n_heads: int):
        super().__init__()
        self.num_rel = (2 * Hp - 1) * (2 * Wp - 1)
        self.bias_table = nn.Parameter(torch.zeros(self.num_rel, n_heads))

        coords_h = torch.arange(Hp)
        coords_w = torch.arange(Wp)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = coords.reshape(2, -1).T  # (L,2)

        rel = coords[:, None, :] - coords[None, :, :]  # (L,L,2)
        rel[..., 0] += Hp - 1
        rel[..., 1] += Wp - 1
        rel_index = rel[..., 0] * (2 * Wp - 1) + rel[..., 1]
        self.register_buffer("rel_index", rel_index, persistent=False)

        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, idx=None):
        if idx is None:
            rel = self.rel_index
        else:
            rel = self.rel_index[idx][:, idx]
        Lx = rel.shape[0]
        bias = self.bias_table[rel.reshape(-1)].view(Lx, Lx, -1)
        return bias.permute(2, 0, 1)

class MHSA_RPB(nn.Module):
    def __init__(self, dim, n_heads, Hp, Wp, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.rpb = RelPosBias2D(Hp, Wp, n_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, idx=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rpb(idx)[None, :, :, :]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class ViTBlockRPB(nn.Module):
    def __init__(self, dim, n_heads, Hp, Wp, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA_RPB(dim, n_heads, Hp, Wp, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, idx=None):
        x = x + self.attn(self.norm1(x), idx=idx)
        x = x + self.mlp(self.norm2(x))
        return x

class MAEViT2D(nn.Module):
    """
    Mechanism-aware MAE imputation:
    - mask_mode: random | dist | mixed
    - outputs mean + logvar
    - same mask across batch for stable subset RPB
    """
    def __init__(
        self,
        in_ch=2,
        out_ch=1,
        img_size=128,
        patch=4,
        enc_dim=256,
        enc_depth=6,
        enc_heads=8,
        dec_dim=192,
        dec_depth=4,
        dec_heads=6,
        mask_ratio=0.4,
        drop=0.0,
        mask_mode="mixed",
        mixed_prob=0.7,
        dist_k=3.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = mask_mode
        self.mixed_prob = float(mixed_prob)
        self.dist_k = float(dist_k)

        Hp = img_size // patch
        Wp = img_size // patch
        self.Hp, self.Wp = Hp, Wp
        self.L = Hp * Wp

        rr = torch.arange(Hp)[:, None].repeat(1, Wp)
        cc = torch.arange(Wp)[None, :].repeat(Hp, 1)
        d = (rr - cc).abs().reshape(-1).float()  # (L,)
        self.register_buffer("patch_dist", d, persistent=False)

        token_in_dim = patch * patch * in_ch
        token_out_dim = patch * patch * out_ch

        self.enc_embed = nn.Linear(token_in_dim, enc_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.L, enc_dim))
        self.enc_blocks = nn.ModuleList([ViTBlockRPB(enc_dim, enc_heads, Hp, Wp, drop=drop) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.dec_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.pos_dec = nn.Parameter(torch.zeros(1, self.L, dec_dim))
        self.dec_blocks = nn.ModuleList([ViTBlockRPB(dec_dim, dec_heads, Hp, Wp, drop=drop) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)

        self.head_mu = nn.Linear(dec_dim, token_out_dim)
        self.head_logvar = nn.Linear(dec_dim, token_out_dim)

        nn.init.normal_(self.pos_enc, std=0.02)
        nn.init.normal_(self.pos_dec, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def _mask_random(self, device):
        num_mask = int(self.L * self.mask_ratio)
        mask = torch.zeros(self.L, device=device)
        idx = torch.randperm(self.L, device=device)[:num_mask]
        mask[idx] = 1.0
        return mask

    def _mask_dist(self, device):
        d = self.patch_dist.to(device)
        d_norm = d / (d.max().clamp_min(1.0))
        logits = self.dist_k * (d_norm - 0.5)
        p = torch.sigmoid(logits)
        p = p * (self.mask_ratio / (p.mean().clamp_min(1e-6)))
        p = p.clamp(0.0, 1.0)
        u = torch.rand(self.L, device=device)
        return (u < p).float()

    def _make_mask(self, B, device):
        mode = self.mask_mode
        if mode == "mixed":
            mode = "random" if torch.rand(1, device=device).item() < self.mixed_prob else "dist"
        if mode == "random":
            m = self._mask_random(device)
        elif mode == "dist":
            m = self._mask_dist(device)
        else:
            raise ValueError("mask_mode must be random|dist|mixed")
        return m[None, :].repeat(B, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "img_size mismatch"

        tokens = patchify(x, self.patch)
        device = tokens.device

        mask = self._make_mask(B, device)
        visible = (mask == 0)
        visible_idx = torch.nonzero(visible[0], as_tuple=False).squeeze(1)

        enc = self.enc_embed(tokens) + self.pos_enc
        enc_vis = enc[:, visible_idx, :]
        for blk in self.enc_blocks:
            enc_vis = blk(enc_vis, idx=visible_idx)
        enc_vis = self.enc_norm(enc_vis)

        dec_vis = self.dec_embed(enc_vis)
        dec_full = self.mask_token.expand(B, self.L, -1).clone()
        dec_full[:, visible_idx, :] = dec_vis
        dec_full = dec_full + self.pos_dec
        for blk in self.dec_blocks:
            dec_full = blk(dec_full, idx=None)
        dec_full = self.dec_norm(dec_full)

        mu_tok = self.head_mu(dec_full)
        logvar_tok = self.head_logvar(dec_full).clamp(-6.0, 6.0)

        mu_img = unpatchify(mu_tok, self.patch, H, W, out_c=1)
        logvar_img = unpatchify(logvar_tok, self.patch, H, W, out_c=1)
        return mu_img, logvar_img, mask

