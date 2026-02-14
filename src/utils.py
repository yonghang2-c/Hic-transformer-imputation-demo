import json
import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def format_run_name(cfg: dict) -> str:
    return (
        f"n{cfg['n']}_p{cfg['patch']}_mr{cfg['mask_ratio']}"
        f"_mask{cfg['mask_mode']}_mix{cfg['mixed_prob']}_k{cfg['dist_k']}"
        f"_lr{cfg['lr']}_ep{cfg['epochs']}"
        f"_keep{cfg['keep_rate']}_ins{cfg['lambda_insul']}"
        f"_dg{cfg['dist_gamma']}_l1{cfg['beta_l1']}_seed{cfg['seed']}"
    )
