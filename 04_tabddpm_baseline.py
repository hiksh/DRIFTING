"""
TabDDPM baseline for NIDS minority-class generation.
Implements Gaussian DDPM (Kotelnikov et al., 2023) from scratch using PyTorch.

All features are continuous numerical — no categorical diffusion needed.

Architecture:
  - Linear noise schedule (beta 1e-4 → 0.02, T=1000)
  - MLPDiffusion: sinusoidal time embedding + residual MLP blocks
  - Epsilon (noise) parameterization
  - DDPM ancestral sampling
"""

import json
import math
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
OUT_DIR  = Path("outputs/tabddpm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTS = {"bot": 1360, "brute_force": 1062, "xss": 436}
SCALERS       = ["standard", "robust"]
DEVICE        = torch.device("cpu")

# Diffusion hyperparameters
T           = 1000
BETA_START  = 1e-4
BETA_END    = 0.02

# Network hyperparameters
HIDDEN_DIM  = 512
N_LAYERS    = 4
T_EMB_DIM   = 128

# Training hyperparameters
BATCH_SIZE  = 256
NUM_EPOCHS  = 5000   # ~10k–30k gradient steps depending on class size
LR          = 3e-4
WEIGHT_DECAY= 1e-5
SEED        = 42


# ── model ──────────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class MLPDiffusion(nn.Module):
    """Denoising network: predicts epsilon given (x_t, t)."""

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, t_emb_dim: int):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.t_embed(t)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


# ── diffusion process ──────────────────────────────────────────────────────────
class GaussianDiffusion:
    def __init__(self, T: int, beta_start: float, beta_end: float, device: torch.device):
        betas      = torch.linspace(beta_start, beta_end, T, device=device)
        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.T          = T
        self.betas      = betas
        self.alphas     = alphas
        self.alpha_bars = alpha_bars

        # precompute for q_sample
        self.sqrt_ab     = alpha_bars.sqrt()
        self.sqrt_one_ab = (1.0 - alpha_bars).sqrt()

        # precompute for p_sample
        alpha_bars_prev      = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        self.posterior_var   = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.posterior_log_var_clipped = torch.log(self.posterior_var.clamp(min=1e-20))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x0 at timesteps t. Returns (x_t, noise)."""
        noise = torch.randn_like(x0)
        x_t = (
            self.sqrt_ab[t, None] * x0
            + self.sqrt_one_ab[t, None] * noise
        )
        return x_t, noise

    @torch.no_grad()
    def p_sample_step(self, model: MLPDiffusion, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One DDPM reverse step: x_t → x_{t-1}."""
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_pred = model(x_t, t_batch)

        # reconstruct x0 estimate
        sqrt_ab_t     = self.sqrt_ab[t]
        sqrt_one_ab_t = self.sqrt_one_ab[t]
        x0_pred = (x_t - sqrt_one_ab_t * eps_pred) / sqrt_ab_t

        # posterior mean
        alpha_bar     = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=x_t.device)
        beta_t        = self.betas[t]

        coef1 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar)
        coef2 = (self.alphas[t].sqrt() * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)
        mean  = coef1 * x0_pred + coef2 * x_t

        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        std   = self.posterior_log_var_clipped[t].exp().sqrt()
        return mean + std * noise

    @torch.no_grad()
    def sample(self, model: MLPDiffusion, n: int, d: int, verbose: bool = True) -> torch.Tensor:
        """Full ancestral sampling: x_T ~ N(0,I) → x_0."""
        device = next(model.parameters()).device
        x = torch.randn(n, d, device=device)
        steps = range(self.T - 1, -1, -1)
        if verbose:
            try:
                from tqdm import tqdm
                steps = tqdm(steps, desc="  sampling", leave=False)
            except ImportError:
                pass
        for t in steps:
            x = self.p_sample_step(model, x, t)
        return x


# ── training ───────────────────────────────────────────────────────────────────
def train(
    model: MLPDiffusion,
    diffusion: GaussianDiffusion,
    X: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> list[float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    log_interval = max(num_epochs // 10, 1)
    losses = []

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for (x0,) in loader:
            t = torch.randint(0, diffusion.T, (x0.shape[0],), device=x0.device)
            x_t, noise = diffusion.q_sample(x0, t)
            eps_pred   = model(x_t, t)
            loss       = F.mse_loss(eps_pred, noise)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * x0.shape[0]

        scheduler.step()
        avg_loss = epoch_loss / len(X)
        losses.append(avg_loss)

        if epoch % log_interval == 0 or epoch == num_epochs:
            print(f"    epoch {epoch:>5}/{num_epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return losses


# ── per-run entry point ────────────────────────────────────────────────────────
def run_tabddpm(scaler: str, cls: str, n_samples: int) -> dict:
    tag = f"{scaler}_{cls}"
    print(f"\n{'='*60}")
    print(f"  TabDDPM  |  scaler={scaler}  class={cls}  n={n_samples}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    X_np = np.load(DATA_DIR / f"X_train_{scaler}_{cls}.npy").astype(np.float32)
    X    = torch.tensor(X_np, device=DEVICE)
    d    = X.shape[1]

    diffusion = GaussianDiffusion(T, BETA_START, BETA_END, DEVICE)
    model     = MLPDiffusion(d, HIDDEN_DIM, N_LAYERS, T_EMB_DIM).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params : {n_params:,}")
    print(f"  Train samples: {len(X_np)}")

    # ── train ──────────────────────────────────────────────────────────────────
    t0_train = time.perf_counter()
    losses   = train(model, diffusion, X, NUM_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY)
    train_time = time.perf_counter() - t0_train

    torch.save(model.state_dict(), OUT_DIR / f"tabddpm_model_{tag}.pt")
    np.save(OUT_DIR / f"tabddpm_loss_{tag}.npy", np.array(losses, dtype=np.float32))

    # ── generate ───────────────────────────────────────────────────────────────
    model.eval()
    t0_gen = time.perf_counter()
    X_synth = diffusion.sample(model, n_samples, d, verbose=True)
    gen_time = time.perf_counter() - t0_gen

    X_synth_np = X_synth.cpu().numpy().astype(np.float32)
    np.save(OUT_DIR / f"X_synth_tabddpm_{tag}.npy", X_synth_np)

    result = {
        "scaler":         scaler,
        "class":          cls,
        "n_train":        int(len(X_np)),
        "n_generated":    int(n_samples),
        "train_time_sec": round(train_time, 2),
        "gen_time_sec":   round(gen_time, 2),
        "final_loss":     round(float(losses[-1]), 6),
        "output_shape":   list(X_synth_np.shape),
        "model_params":   n_params,
    }

    print(f"  Train time : {train_time:.1f}s")
    print(f"  Gen time   : {gen_time:.1f}s")
    print(f"  Final loss : {losses[-1]:.6f}")
    print(f"  Saved: X_synth_tabddpm_{tag}.npy  {X_synth_np.shape}")

    return result


# ── main ───────────────────────────────────────────────────────────────────────
all_results = []

for scaler in SCALERS:
    for cls, n in TARGET_COUNTS.items():
        res = run_tabddpm(scaler, cls, n)
        all_results.append(res)

        with open(OUT_DIR / "timing_log.json", "w") as f:
            json.dump(all_results, f, indent=2)

print("\n\n── Summary ──────────────────────────────────────────────────────────")
df_summary = pd.DataFrame(all_results)
print(df_summary[["scaler", "class", "n_train", "train_time_sec", "gen_time_sec", "final_loss"]].to_string(index=False))
df_summary.to_csv(OUT_DIR / "timing_summary.csv", index=False)
print(f"\nAll outputs in: {OUT_DIR.resolve()}")
