"""
Drifting Models for UNSW-NB15. Mirrors 05_drifting.py.
Generator input dim D is loaded automatically (D=176).
"""

import json, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

DATA_DIR = Path("data/processed_unsw")
OUT_DIR  = Path("outputs/drifting_unsw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTS = {"backdoor": 1746, "shellcode": 1133, "worms": 130}
DEVICE    = torch.device("cpu")
ALPHA     = 1.0
BETA      = 1.0
HIDDEN    = 256
N_LAYERS  = 3
NUM_EPOCHS = 3000
BATCH_GEN  = 256
BATCH_REAL = 256
LR         = 1e-3
SEED       = 42
BW_SUB     = 500

feat_cols = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()
D = len(feat_cols)
print(f"Feature dim D = {D}")


class Generator(nn.Module):
    def __init__(self, d, hidden=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(d, hidden), nn.SiLU()]
        for _ in range(n_layers-1): layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, d))
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

def rbf_kernel(X, Y, h):
    return torch.exp(-(X.unsqueeze(1)-Y.unsqueeze(0)).pow(2).sum(-1) / h)

def attraction(y, x_real, h):
    K = rbf_kernel(y, x_real, h)
    w = K / (K.sum(1, keepdim=True) + 1e-10)
    return (w.unsqueeze(2)*x_real.unsqueeze(0)).sum(1) - y

def repulsion(y, h):
    n = y.shape[0]
    K = rbf_kernel(y, y, h) * (1 - torch.eye(n, device=y.device))
    w = K / (K.sum(1, keepdim=True) + 1e-10)
    return y - (w.unsqueeze(2)*y.unsqueeze(0)).sum(1)

def drifting_field(y, x_real, h):
    return ALPHA * attraction(y, x_real, h) - BETA * repulsion(y, h)

def estimate_bandwidth(X, sub=BW_SUB):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(sub, len(X)), replace=False)
    Xs  = X[idx].astype(np.float64)
    sq  = ((Xs[:,None]-Xs[None,:])**2).sum(-1)
    upper = sq[np.triu_indices(len(Xs), k=1)]
    nz    = upper[upper > 1e-8]
    h     = float(np.percentile(nz, 75)) if len(nz) else float(Xs.shape[1]*0.1)
    return max(h, Xs.shape[1]*0.01)

def train(gen, X_real, h, epochs, batch_gen, lr):
    gen.train()
    opt   = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_real = len(X_real)
    log_every = max(epochs//10, 1)
    losses = []
    for ep in range(1, epochs+1):
        idx    = torch.randperm(n_real)[:BATCH_REAL]
        x_batch= X_real[idx]
        z      = torch.randn(batch_gen, D, device=DEVICE)
        y      = gen(z)
        y_sg   = y.detach()
        with torch.no_grad():
            V = drifting_field(y_sg, x_batch, h)
        loss = F.mse_loss(y, y_sg + V)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if ep % log_every == 0 or ep == epochs:
            print(f"  epoch {ep:>5}/{epochs}  loss={loss.item():.6f}  "
                  f"|V|={V.norm(dim=1).mean():.4f}")
    return losses

all_results = []

for cls, n in TARGET_COUNTS.items():
    print(f"\n{'='*55}")
    print(f"  Drifting | class={cls}  n={n}  D={D}")
    print(f"{'='*55}")
    torch.manual_seed(SEED)

    X_np   = np.load(DATA_DIR / f"X_train_standard_{cls}.npy").astype(np.float32)
    X_real = torch.tensor(X_np, device=DEVICE)

    h = estimate_bandwidth(X_np)
    print(f"  bandwidth h = {h:.4f}")

    gen = Generator(D, HIDDEN, N_LAYERS).to(DEVICE)
    print(f"  params: {sum(p.numel() for p in gen.parameters()):,}")

    t0_tr  = time.perf_counter()
    losses = train(gen, X_real, h, NUM_EPOCHS, BATCH_GEN, LR)
    train_time = time.perf_counter() - t0_tr
    torch.save(gen.state_dict(), OUT_DIR / f"drifting_model_standard_{cls}.pt")

    gen.eval()
    t0_g = time.perf_counter()
    with torch.no_grad():
        X_synth = gen(torch.randn(n, D, device=DEVICE)).cpu().numpy().astype(np.float32)
    gen_time = time.perf_counter() - t0_g
    np.save(OUT_DIR / f"X_synth_drifting_standard_{cls}.npy", X_synth)

    res = {"class": cls, "n_train": len(X_np), "n_generated": n,
           "bandwidth_h": round(h,4),
           "train_time_sec": round(train_time,2), "gen_time_sec": round(gen_time,4),
           "final_loss": round(float(losses[-1]),6)}
    all_results.append(res)
    print(f"  train={train_time:.1f}s  gen={gen_time:.4f}s  loss={losses[-1]:.6f}")

    with open(OUT_DIR / "timing_log.json", "w") as f: json.dump(all_results, f, indent=2)

pd.DataFrame(all_results).to_csv(OUT_DIR / "timing_summary.csv", index=False)
print("\nDrifting UNSW-NB15 complete.")
