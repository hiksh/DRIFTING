"""
TabDDPM baseline for UNSW-NB15. Mirrors 04_tabddpm_baseline.py.
Input dim D is loaded automatically from feature_names.csv (D=176).
"""

import json, math, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

DATA_DIR = Path("data/processed_unsw")
OUT_DIR  = Path("outputs/tabddpm_unsw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTS = {"backdoor": 1746, "shellcode": 1133, "worms": 130}
DEVICE = torch.device("cpu")
T, BETA_START, BETA_END = 1000, 1e-4, 0.02
HIDDEN_DIM, N_LAYERS, T_EMB_DIM = 512, 4, 128
BATCH_SIZE, NUM_EPOCHS, LR = 256, 5000, 3e-4
SEED = 42

feat_cols = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()
D = len(feat_cols)
print(f"Feature dim D = {D}")


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / (half-1))
        a = t[:,None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim,dim), nn.SiLU(), nn.Linear(dim,dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))

class MLPDiffusion(nn.Module):
    def __init__(self, d, hidden=512, n_layers=4, t_emb=128):
        super().__init__()
        self.t_embed    = nn.Sequential(SinusoidalEmbedding(t_emb),
                                        nn.Linear(t_emb,hidden), nn.SiLU(),
                                        nn.Linear(hidden,hidden))
        self.input_proj = nn.Linear(d, hidden)
        self.blocks     = nn.ModuleList([ResBlock(hidden) for _ in range(n_layers)])
        self.out_proj   = nn.Linear(hidden, d)
    def forward(self, x, t):
        h = self.input_proj(x) + self.t_embed(t)
        for b in self.blocks: h = b(h)
        return self.out_proj(h)

class GaussianDiffusion:
    def __init__(self, T, b0, b1, device):
        betas  = torch.linspace(b0, b1, T, device=device)
        alphas = 1 - betas
        ab     = torch.cumprod(alphas, 0)
        ab_prev = F.pad(ab[:-1], (1,0), value=1.0)
        self.T, self.betas, self.alphas, self.alpha_bars = T, betas, alphas, ab
        self.sqrt_ab     = ab.sqrt()
        self.sqrt_one_ab = (1-ab).sqrt()
        self.post_log_var = torch.log((betas*(1-ab_prev)/(1-ab)).clamp(min=1e-20))

    def q_sample(self, x0, t):
        n = torch.randn_like(x0)
        return self.sqrt_ab[t,None]*x0 + self.sqrt_one_ab[t,None]*n, n

    @torch.no_grad()
    def p_step(self, model, x, t):
        tb = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        ep = model(x, tb)
        x0 = (x - self.sqrt_one_ab[t]*ep) / self.sqrt_ab[t]
        ab_prev = self.alpha_bars[t-1] if t>0 else torch.tensor(1., device=x.device)
        c1 = ab_prev.sqrt()*self.betas[t]/(1-self.alpha_bars[t])
        c2 = self.alphas[t].sqrt()*(1-ab_prev)/(1-self.alpha_bars[t])
        mean = c1*x0 + c2*x
        return mean if t==0 else mean + self.post_log_var[t].exp().sqrt()*torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, n, d):
        x = torch.randn(n, d, device=next(model.parameters()).device)
        for t in range(self.T-1, -1, -1): x = self.p_step(model, x, t)
        return x

def train(model, diffusion, X, epochs, batch_size, lr):
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
    log_every = max(epochs//10, 1)
    losses = []
    for ep in range(1, epochs+1):
        ep_loss = 0
        for (x0,) in loader:
            t  = torch.randint(0, diffusion.T, (x0.shape[0],), device=x0.device)
            xt, noise = diffusion.q_sample(x0, t)
            loss = F.mse_loss(model(xt, t), noise)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss.item()*len(x0)
        sched.step()
        avg = ep_loss/len(X); losses.append(avg)
        if ep % log_every == 0 or ep == epochs:
            print(f"  epoch {ep:>5}/{epochs}  loss={avg:.6f}  lr={sched.get_last_lr()[0]:.2e}")
    return losses

all_results = []
diffusion   = GaussianDiffusion(T, BETA_START, BETA_END, DEVICE)

for cls, n in TARGET_COUNTS.items():
    print(f"\n{'='*55}")
    print(f"  TabDDPM | class={cls}  n={n}  D={D}")
    print(f"{'='*55}")
    torch.manual_seed(SEED)

    X_np = np.load(DATA_DIR / f"X_train_standard_{cls}.npy").astype(np.float32)
    X    = torch.tensor(X_np, device=DEVICE)

    model = MLPDiffusion(D, HIDDEN_DIM, N_LAYERS, T_EMB_DIM).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    t0_tr = time.perf_counter()
    losses = train(model, diffusion, X, NUM_EPOCHS, BATCH_SIZE, LR)
    train_time = time.perf_counter() - t0_tr
    torch.save(model.state_dict(), OUT_DIR / f"tabddpm_model_standard_{cls}.pt")

    model.eval()
    t0_g = time.perf_counter()
    X_synth = diffusion.sample(model, n, D).cpu().numpy().astype(np.float32)
    gen_time = time.perf_counter() - t0_g
    np.save(OUT_DIR / f"X_synth_tabddpm_standard_{cls}.npy", X_synth)

    res = {"class": cls, "n_train": len(X_np), "n_generated": n,
           "train_time_sec": round(train_time,2), "gen_time_sec": round(gen_time,2),
           "final_loss": round(float(losses[-1]),6)}
    all_results.append(res)
    print(f"  train={train_time:.1f}s  gen={gen_time:.1f}s  loss={losses[-1]:.6f}")

    with open(OUT_DIR / "timing_log.json", "w") as f: json.dump(all_results, f, indent=2)

pd.DataFrame(all_results).to_csv(OUT_DIR / "timing_summary.csv", index=False)
print("\nTabDDPM UNSW-NB15 complete.")
