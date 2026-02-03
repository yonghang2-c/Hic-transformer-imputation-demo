import torch
import torch.nn as nn

def patchify(x, p):
    B, C, H, W = x.shape
    assert H % p == 0 and W % p == 0
    h = H // p
    w = W // p
    x = x.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
    return x  # (B, L, p*p*C)

def unpatchify(tokens, p, H, W, out_c):
    B, L, D = tokens.shape
    h = H // p
    w = W // p
    x = tokens.reshape(B, h, w, p, p, out_c)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, out_c, H, W)
    return x

class RelPosBias2D(nn.Module):
    """
    2D relative position bias for patch grid (Hp x Wp).
    Supports subset selection (idx) for MAE encoder visible tokens.
    """
    def __init__(self, Hp: int, Wp: int, n_heads: int):
        super().__init__()
        self.num_rel = (2 * Hp - 1) * (2 * Wp - 1)
        self.bias_table = nn.Parameter(torch.zeros(self.num_rel, n_heads))

        coords_h = torch.arange(Hp)
        coords_w = torch.arange(Wp)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2,Hp,Wp)
        coords = coords.reshape(2, -1).T  # (L,2)

        rel = coords[:, None, :] - coords[None, :, :]  # (L,L,2)
        rel[..., 0] += Hp - 1
        rel[..., 1] += Wp - 1
        rel_index = rel[..., 0] * (2 * Wp - 1) + rel[..., 1]  # (L,L)
        self.register_buffer("rel_index", rel_index, persistent=False)

        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, idx=None):
        """
        idx: optional 1D LongTensor of shape (L_sub,)
             If provided, return bias for subset (L_sub x L_sub).
        Returns: (H, Lx, Lx)
        """
        if idx is None:
            rel = self.rel_index  # (L,L)
        else:
            rel = self.rel_index[idx][:, idx]  # (L_sub, L_sub)

        Lx = rel.shape[0]
        bias = self.bias_table[rel.reshape(-1)].view(Lx, Lx, -1)  # (Lx,Lx,H)
        return bias.permute(2, 0, 1)  # (H,Lx,Lx)

class MHSA_RPB(nn.Module):
    """
    Multi-head self-attention with 2D relative position bias.
    """
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
        # x: (B, L, dim), idx: subset indices for RPB (optional)
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,H,L,D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,L,L)
        attn = attn + self.rpb(idx)[None, :, :, :]     # (1,H,L,L)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,H,L,D)
        out = out.transpose(1, 2).reshape(B, L, C)  # (B,L,dim)

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
    Masked Autoencoder (2D) with Hi-C-aware 2D relative position bias.
    Outputs mean + log-variance (uncertainty).
    Note: encoder uses the same mask across the batch for simplicity and stability.
    """
    def __init__(
        self,
        in_ch=2,
        out_ch=1,
        img_size=128,
        patch=8,
        enc_dim=256,
        enc_depth=6,
        enc_heads=8,
        dec_dim=192,
        dec_depth=4,
        dec_heads=6,
        mask_ratio=0.5,
        drop=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.mask_ratio = mask_ratio

        Hp = img_size // patch
        Wp = img_size // patch
        self.Hp, self.Wp = Hp, Wp
        self.L = Hp * Wp

        token_in_dim = patch * patch * in_ch
        token_out_dim = patch * patch * out_ch

        # encoder
        self.enc_embed = nn.Linear(token_in_dim, enc_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.L, enc_dim))
        self.enc_blocks = nn.ModuleList([ViTBlockRPB(enc_dim, enc_heads, Hp, Wp, drop=drop) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(enc_dim)

        # decoder
        self.dec_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.pos_dec = nn.Parameter(torch.zeros(1, self.L, dec_dim))
        self.dec_blocks = nn.ModuleList([ViTBlockRPB(dec_dim, dec_heads, Hp, Wp, drop=drop) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)

        # output heads: mean and logvar per patch token
        self.head_mu = nn.Linear(dec_dim, token_out_dim)
        self.head_logvar = nn.Linear(dec_dim, token_out_dim)

        nn.init.normal_(self.pos_enc, std=0.02)
        nn.init.normal_(self.pos_dec, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def _random_mask(self, B, device):
        """
        Make the SAME random mask for all samples in the batch.
        mask: (B, L) where 1=masked, 0=visible
        """
        num_mask = int(self.L * self.mask_ratio)
        mask = torch.zeros(1, self.L, device=device)
        idx = torch.randperm(self.L, device=device)[:num_mask]
        mask[0, idx] = 1.0
        return mask.repeat(B, 1)

    def forward(self, x):
        """
        x: (B, in_ch, H, W)
        Returns:
          mu_img, logvar_img: (B, 1, H, W)
          mask_patch: (B, L) 1=masked
        """
        B, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "img_size mismatch"

        p = self.patch
        tokens = patchify(x, p)  # (B,L,token_in_dim)
        device = tokens.device

        mask = self._random_mask(B, device)
        visible = (mask == 0)

        # Use indices from the first sample (same mask across batch)
        visible_idx = torch.nonzero(visible[0], as_tuple=False).squeeze(1)  # (L_vis,)

        # encoder: only visible tokens
        enc = self.enc_embed(tokens) + self.pos_enc  # (B,L,enc_dim)
        enc_vis = enc[:, visible_idx, :]             # (B,L_vis,enc_dim)

        for blk in self.enc_blocks:
            enc_vis = blk(enc_vis, idx=visible_idx)
        enc_vis = self.enc_norm(enc_vis)

        # decoder: fill full length with mask tokens, then scatter visible tokens
        dec_vis = self.dec_embed(enc_vis)  # (B,L_vis,dec_dim)
        dec_full = self.mask_token.expand(B, self.L, -1).clone()
        dec_full[:, visible_idx, :] = dec_vis
        dec_full = dec_full + self.pos_dec

        for blk in self.dec_blocks:
            # decoder sees full length; use full RPB (idx=None)
            dec_full = blk(dec_full, idx=None)
        dec_full = self.dec_norm(dec_full)

        mu_tok = self.head_mu(dec_full)
        logvar_tok = self.head_logvar(dec_full).clamp(-6.0, 6.0)

        mu_img = unpatchify(mu_tok, p, H, W, out_c=1)
        logvar_img = unpatchify(logvar_tok, p, H, W, out_c=1)
        return mu_img, logvar_img, mask
