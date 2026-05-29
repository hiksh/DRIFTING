"""
Drifting Models for NIDS tabular data generation.
Implements arXiv:2602.04770 — "Generative Modeling via Drifting"

Drifting field:
    V(x) = alpha * V+(x) - beta * V-(x)

    V+(x) = kernel-weighted mean-shift toward real data
          = (sum_i K(x, x_i) * x_i) / (sum_i K(x, x_i))  -  x

    V-(x) = kernel-weighted mean-shift repulsion from generated batch
          = x  -  (sum_j K(x, y_j) * y_j) / (sum_j K(x, y_j))
            (diagonal masked so a point does not repel itself)

    RBF kernel: K(x, y) = exp(-||x - y||^2 / h)
    Bandwidth h: 75th-percentile of nonzero pairwise squared distances
                 (median is unstable for tight clusters in class-specific data)

Training objective:
    L(theta) = E_eps[ || f(eps) - sg(f(eps) + V(f(eps))) ||^2 ]

    Gradient w.r.t. theta pushes f(eps) in direction V(f(eps)).
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
OUT_DIR  = Path("outputs/drifting")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTS = {"bot": 1360, "brute_force": 1062, "xss": 436}
SCALERS       = ["standard", "robust"]
DEVICE        = torch.device("cpu")

# Drifting field
ALPHA        = 1.0   # attraction strength
BETA         = 1.0   # repulsion strength
BW_SUBSAMPLE = 500   # points used to estimate bandwidth

# Generator
HIDDEN_DIM   = 256
N_LAYERS     = 3

# Training
NUM_EPOCHS    = 3000
BATCH_GEN     = 256   # generated samples per step
BATCH_REAL    = 256   # real data samples used for V+ per step (mini-batch)
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
SEED          = 42


# ── generator ──────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, d: int, hidden: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, d))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── drifting field ─────────────────────────────────────────────────────────────
def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, h: float) -> torch.Tensor:
    """
    X: (n, d), Y: (m, d) -> (n, m) kernel matrix K_ij = exp(-||x_i - y_j||^2 / h)
    Computed in chunks to avoid O(n*m*d) peak memory.
    """
    diff = X.unsqueeze(1) - Y.unsqueeze(0)   # (n, m, d)
    sq   = diff.pow(2).sum(-1)               # (n, m)
    return torch.exp(-sq / h)


def attraction(y: torch.Tensor, x_real: torch.Tensor, h: float) -> torch.Tensor:
    """V+(y) = kernel-weighted mean of real data - y."""
    K   = rbf_kernel(y, x_real, h)                          # (n, m)
    w   = K / (K.sum(1, keepdim=True) + 1e-10)              # normalise rows
    mu  = (w.unsqueeze(2) * x_real.unsqueeze(0)).sum(1)     # (n, d)
    return mu - y


def repulsion(y: torch.Tensor, h: float) -> torch.Tensor:
    """V-(y_i) = y_i - kernel-weighted mean of other generated points."""
    n   = y.shape[0]
    K   = rbf_kernel(y, y, h)                               # (n, n)
    K   = K * (1.0 - torch.eye(n, device=y.device))         # zero diagonal
    w   = K / (K.sum(1, keepdim=True) + 1e-10)
    mu  = (w.unsqueeze(2) * y.unsqueeze(0)).sum(1)          # (n, d)
    return y - mu


def drifting_field(
    y:      torch.Tensor,
    x_real: torch.Tensor,
    h:      float,
    alpha:  float = 1.0,
    beta:   float = 1.0,
) -> torch.Tensor:
    """V(y) = alpha * V+(y) - beta * V-(y)."""
    return alpha * attraction(y, x_real, h) - beta * repulsion(y, h)


# ── bandwidth estimation ───────────────────────────────────────────────────────
def estimate_bandwidth(X: np.ndarray, subsample: int = BW_SUBSAMPLE, seed: int = 0) -> float:
    """
    75th-percentile of nonzero pairwise squared L2 distances.
    More stable than median when the class has many near-duplicate rows or
    tight sub-clusters (brute_force, xss have median pairwise dist ~ 0.01).
    Lower-bounded at 0.01 * d to ensure non-degenerate kernel support.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(subsample, len(X)), replace=False)
    Xs  = X[idx].astype(np.float64)
    d   = Xs.shape[1]

    diff   = Xs[:, None, :] - Xs[None, :, :]     # (n, n, d)
    sq     = (diff ** 2).sum(-1)                  # (n, n)
    upper  = sq[np.triu_indices(len(Xs), k=1)]    # exclude diagonal & lower tri
    nonzero = upper[upper > 1e-8]

    if len(nonzero) == 0:
        return float(d * 0.1)

    h = float(np.percentile(nonzero, 75))
    h = max(h, d * 0.01)         # lower bound: 1% of dimensionality
    return h


# ── training loop ──────────────────────────────────────────────────────────────
def train_drifting(
    gen:        Generator,
    X_real:     torch.Tensor,
    h:          float,
    num_epochs: int,
    batch_gen:  int,
    lr:         float,
    weight_decay: float,
) -> list[float]:
    gen.train()
    opt       = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    d         = X_real.shape[1]
    n_real    = len(X_real)

    log_every = max(num_epochs // 10, 1)
    losses    = []

    for epoch in range(1, num_epochs + 1):
        # mini-batch of real data for V+ (uniform cost regardless of class size)
        idx      = torch.randperm(n_real, device=DEVICE)[:BATCH_REAL]
        x_batch  = X_real[idx]

        # sample noise and generate
        z = torch.randn(batch_gen, d, device=DEVICE)
        y = gen(z)                       # (batch_gen, d)  — has gradient

        # compute drift with stop_grad
        y_sg = y.detach()
        with torch.no_grad():
            V = drifting_field(y_sg, x_batch, h, ALPHA, BETA)

        target = y_sg + V                # sg(f(eps) + V(f(eps)))

        loss = F.mse_loss(y, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % log_every == 0 or epoch == num_epochs:
            print(f"    epoch {epoch:>5}/{num_epochs}  loss={loss.item():.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"|V|={V.norm(dim=1).mean().item():.4f}")

    return losses


# ── per-run entry point ────────────────────────────────────────────────────────
def run_drifting(scaler: str, cls: str, n_samples: int) -> dict:
    tag = f"{scaler}_{cls}"
    print(f"\n{'='*60}")
    print(f"  Drifting  |  scaler={scaler}  class={cls}  n={n_samples}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    X_np   = np.load(DATA_DIR / f"X_train_{scaler}_{cls}.npy").astype(np.float32)
    X_real = torch.tensor(X_np, device=DEVICE)
    d      = X_real.shape[1]

    # bandwidth
    h = estimate_bandwidth(X_np)
    print(f"  Bandwidth h = {h:.4f}  (75th-pct nonzero pairwise sq-dist)")

    gen    = Generator(d, HIDDEN_DIM, N_LAYERS).to(DEVICE)
    n_par  = sum(p.numel() for p in gen.parameters())
    print(f"  Generator params : {n_par:,}")
    print(f"  Train samples    : {len(X_np)}")

    # ── train ──────────────────────────────────────────────────────────────────
    t0_train = time.perf_counter()
    losses   = train_drifting(gen, X_real, h, NUM_EPOCHS, BATCH_GEN, LR, WEIGHT_DECAY)
    train_time = time.perf_counter() - t0_train

    torch.save(gen.state_dict(), OUT_DIR / f"drifting_model_{tag}.pt")
    np.save(OUT_DIR / f"drifting_loss_{tag}.npy", np.array(losses, dtype=np.float32))

    # ── generate ───────────────────────────────────────────────────────────────
    gen.eval()
    t0_gen = time.perf_counter()
    with torch.no_grad():
        z_all   = torch.randn(n_samples, d, device=DEVICE)
        X_synth = gen(z_all).cpu().numpy().astype(np.float32)
    gen_time = time.perf_counter() - t0_gen

    np.save(OUT_DIR / f"X_synth_drifting_{tag}.npy", X_synth)

    result = {
        "scaler":          scaler,
        "class":           cls,
        "n_train":         int(len(X_np)),
        "n_generated":     int(n_samples),
        "bandwidth_h":     round(h, 4),
        "train_time_sec":  round(train_time, 2),
        "gen_time_sec":    round(gen_time, 4),
        "final_loss":      round(float(losses[-1]), 6),
        "output_shape":    list(X_synth.shape),
        "generator_params": n_par,
    }

    print(f"  Train time  : {train_time:.1f}s")
    print(f"  Gen time    : {gen_time:.4f}s")
    print(f"  Final loss  : {losses[-1]:.6f}")
    print(f"  Saved: X_synth_drifting_{tag}.npy  {X_synth.shape}")

    return result


# ── main ───────────────────────────────────────────────────────────────────────
all_results = []

for scaler in SCALERS:
    for cls, n in TARGET_COUNTS.items():
        res = run_drifting(scaler, cls, n)
        all_results.append(res)

        with open(OUT_DIR / "timing_log.json", "w") as f:
            json.dump(all_results, f, indent=2)

print("\n\n── Summary ──────────────────────────────────────────────────────────")
df = pd.DataFrame(all_results)
print(df[["scaler", "class", "n_train", "bandwidth_h",
          "train_time_sec", "gen_time_sec", "final_loss"]].to_string(index=False))
df.to_csv(OUT_DIR / "timing_summary.csv", index=False)
print(f"\nAll outputs in: {OUT_DIR.resolve()}")
