"""
Augmentation ratio experiment: x1 / x3 / x5 synthetic samples.
Models: CTGAN, TabDDPM, Drifting (standard scaler only).

Step 1 — Generate x3, x5 samples from saved models.
Step 2 — 5-seed RF experiment for each ratio.
Step 3 — Comparison table: ratio x model x class.
"""

import math
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/processed")
CTGAN_DIR   = Path("outputs/ctgan")
TABDDPM_DIR = Path("outputs/tabddpm")
DRIFT_DIR   = Path("outputs/drifting")
OUT_DIR     = Path("outputs/ratio_experiment")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MINORITY_CLS = ["bot", "brute_force", "xss"]
LABEL_MAP    = {"bot": 1, "brute_force": 2, "xss": 3}
MINORITY_IDX = [1, 2, 3]
BASE_COUNTS  = {"bot": 1360, "brute_force": 1062, "xss": 436}
RATIOS       = [1, 3, 5]
MODELS       = ["CTGAN", "TabDDPM", "Drifting"]

BENIGN_CAP = 50_000
RF_SEEDS   = [42, 123, 456, 789, 1024]
RF_PARAMS  = dict(n_estimators=100, n_jobs=-1, class_weight="balanced",
                  max_features="sqrt")

feat_cols = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()
D = len(feat_cols)

COLORS = {"CTGAN": "steelblue", "TabDDPM": "darkorange", "Drifting": "seagreen"}


# ── model classes ──────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """Drifting generator — must match 05_drifting.py."""
    def __init__(self, d: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, d))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SinusoidalEmbedding(nn.Module):
    """Must match 04_tabddpm_baseline.py."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class MLPDiffusion(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, n_layers: int = 4, t_emb: int = 128):
        super().__init__()
        self.t_embed    = nn.Sequential(SinusoidalEmbedding(t_emb),
                                        nn.Linear(t_emb, hidden), nn.SiLU(),
                                        nn.Linear(hidden, hidden))
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks     = nn.ModuleList([ResBlock(hidden) for _ in range(n_layers)])
        self.out_proj   = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.t_embed(t)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


class GaussianDiffusion:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: torch.device = torch.device("cpu")):
        betas      = torch.linspace(beta_start, beta_end, T, device=device)
        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev          = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        self.T                   = T
        self.betas               = betas
        self.alphas              = alphas
        self.alpha_bars          = alpha_bars
        self.sqrt_ab             = alpha_bars.sqrt()
        self.sqrt_one_ab         = (1.0 - alpha_bars).sqrt()
        self.posterior_log_var   = torch.log(
            (betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)).clamp(min=1e-20))

    @torch.no_grad()
    def p_sample_step(self, model: MLPDiffusion, x_t: torch.Tensor, t: int) -> torch.Tensor:
        t_b      = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_pred = model(x_t, t_b)
        x0_pred  = (x_t - self.sqrt_one_ab[t] * eps_pred) / self.sqrt_ab[t]
        ab_prev  = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=x_t.device)
        coef1    = (ab_prev.sqrt() * self.betas[t]) / (1.0 - self.alpha_bars[t])
        coef2    = (self.alphas[t].sqrt() * (1.0 - ab_prev)) / (1.0 - self.alpha_bars[t])
        mean     = coef1 * x0_pred + coef2 * x_t
        if t == 0:
            return mean
        return mean + self.posterior_log_var[t].exp().sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model: MLPDiffusion, n: int, d: int) -> torch.Tensor:
        device = next(model.parameters()).device
        x = torch.randn(n, d, device=device)
        for t in range(self.T - 1, -1, -1):
            x = self.p_sample_step(model, x, t)
        return x


# ── canonical synth file path ──────────────────────────────────────────────────
def synth_path(model: str, cls: str, ratio: int) -> Path:
    dirs  = {"CTGAN": CTGAN_DIR, "TabDDPM": TABDDPM_DIR, "Drifting": DRIFT_DIR}
    suffix = "" if ratio == 1 else f"_x{ratio}"
    return dirs[model] / f"X_synth_{model.lower()}_standard_{cls}{suffix}.npy"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — generate x3 and x5 samples from saved models
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Generating x3 and x5 samples")
print("=" * 60)

# ── CTGAN ──────────────────────────────────────────────────────────────────────
for cls in MINORITY_CLS:
    ctgan = CTGANSynthesizer.load(str(CTGAN_DIR / f"ctgan_model_standard_{cls}.pkl"))
    for ratio in [3, 5]:
        out = synth_path("CTGAN", cls, ratio)
        if out.exists():
            print(f"  [skip] {out.name}")
            continue
        n  = BASE_COUNTS[cls] * ratio
        t0 = time.perf_counter()
        X_s = ctgan.sample(num_rows=n)[feat_cols].values.astype(np.float32)
        np.save(out, X_s)
        print(f"  CTGAN    {cls:<14} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.1f}s")

# ── TabDDPM ────────────────────────────────────────────────────────────────────
diffusion = GaussianDiffusion()
for cls in MINORITY_CLS:
    net = MLPDiffusion(D)
    net.load_state_dict(torch.load(
        TABDDPM_DIR / f"tabddpm_model_standard_{cls}.pt", weights_only=True))
    net.eval()
    for ratio in [3, 5]:
        out = synth_path("TabDDPM", cls, ratio)
        if out.exists():
            print(f"  [skip] {out.name}")
            continue
        n  = BASE_COUNTS[cls] * ratio
        t0 = time.perf_counter()
        X_s = diffusion.sample(net, n, D).numpy().astype(np.float32)
        np.save(out, X_s)
        print(f"  TabDDPM  {cls:<14} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.1f}s")

# ── Drifting ───────────────────────────────────────────────────────────────────
for cls in MINORITY_CLS:
    gen = Generator(D)
    gen.load_state_dict(torch.load(
        DRIFT_DIR / f"drifting_model_standard_{cls}.pt", weights_only=True))
    gen.eval()
    for ratio in [3, 5]:
        out = synth_path("Drifting", cls, ratio)
        if out.exists():
            print(f"  [skip] {out.name}")
            continue
        n  = BASE_COUNTS[cls] * ratio
        t0 = time.perf_counter()
        with torch.no_grad():
            X_s = gen(torch.randn(n, D)).numpy().astype(np.float32)
        np.save(out, X_s)
        print(f"  Drifting {cls:<14} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.3f}s")

# verify
print("\nAll synth files:")
for model in MODELS:
    for cls in MINORITY_CLS:
        for ratio in RATIOS:
            p   = synth_path(model, cls, ratio)
            arr = np.load(p)
            print(f"  {p.name:<58} {arr.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 5-seed RF experiment per ratio
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — 5-seed RF experiment")
print("=" * 60)
print(f"  {len(RATIOS)} ratios x {len(RF_SEEDS)} seeds x "
      f"(1 baseline + {len(MODELS)} models) = "
      f"{len(RATIOS) * len(RF_SEEDS) * (1 + len(MODELS))} RF trainings\n")

X_tr_full = np.load(DATA_DIR / "X_train_standard.npy")
y_tr_full = np.load(DATA_DIR / "y_train.npy")
X_te      = np.load(DATA_DIR / "X_test_standard.npy")
y_te      = np.load(DATA_DIR / "y_test.npy")

benign_idx   = np.where(y_tr_full == 0)[0]
minority_idx = np.where(y_tr_full != 0)[0]


def build_base(seed: int):
    rng = np.random.default_rng(seed)
    b   = rng.choice(benign_idx, size=min(BENIGN_CAP, len(benign_idx)), replace=False)
    return X_tr_full[np.concatenate([b, minority_idx])], y_tr_full[np.concatenate([b, minority_idx])]


def build_augmented(X_base, y_base, model: str, ratio: int):
    xs, ys = [X_base], [y_base]
    for cls in MINORITY_CLS:
        X_s = np.load(synth_path(model, cls, ratio))
        xs.append(X_s)
        ys.append(np.full(len(X_s), LABEL_MAP[cls], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def rf_eval(X_tr, y_tr, seed: int) -> dict:
    rf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    _, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=MINORITY_IDX, zero_division=0)
    return {c: {"f1": float(f1[k]), "recall": float(rec[k])}
            for k, c in enumerate(MINORITY_CLS)}


raw_rows = []

for seed in RF_SEEDS:
    print(f"  seed={seed}", end="  ", flush=True)
    X_base, y_base = build_base(seed)
    base_m = rf_eval(X_base, y_base, seed)
    print(f"base done", end="  ", flush=True)

    for ratio in RATIOS:
        for model in MODELS:
            X_aug, y_aug = build_augmented(X_base, y_base, model, ratio)
            aug_m = rf_eval(X_aug, y_aug, seed)
            for cls in MINORITY_CLS:
                raw_rows.append({
                    "seed":             seed,
                    "model":            model,
                    "ratio":            ratio,
                    "class":            cls,
                    "n_synth":          BASE_COUNTS[cls] * ratio,
                    "f1_baseline":      base_m[cls]["f1"],
                    "f1_augmented":     aug_m[cls]["f1"],
                    "f1_delta":         aug_m[cls]["f1"]     - base_m[cls]["f1"],
                    "recall_baseline":  base_m[cls]["recall"],
                    "recall_augmented": aug_m[cls]["recall"],
                    "recall_delta":     aug_m[cls]["recall"] - base_m[cls]["recall"],
                })
    print(flush=True)

df_raw = pd.DataFrame(raw_rows)
df_raw.to_csv(OUT_DIR / "raw_results.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — aggregate and format
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Results")
print("=" * 60)

agg = (df_raw
       .groupby(["model", "ratio", "class"])
       .agg(
           f1_delta_mean     =("f1_delta",     "mean"),
           f1_delta_std      =("f1_delta",     "std"),
           recall_delta_mean =("recall_delta", "mean"),
           recall_delta_std  =("recall_delta", "std"),
           f1_aug_mean       =("f1_augmented", "mean"),
           f1_aug_std        =("f1_augmented", "std"),
       )
       .round(4)
       .reset_index())

agg.to_csv(OUT_DIR / "summary_aggregated.csv", index=False)

def fmt(mean, std): return f"{mean:+.4f} +- {std:.4f}"

pivot_rows = []
for cls in MINORITY_CLS:
    for model in MODELS:
        row = {"class": cls, "model": model}
        for ratio in RATIOS:
            sub = agg[(agg.model==model) & (agg.ratio==ratio) & (agg["class"]==cls)]
            if len(sub):
                row[f"f1_delta_x{ratio}"]    = fmt(sub.f1_delta_mean.iloc[0],     sub.f1_delta_std.iloc[0])
                row[f"recall_delta_x{ratio}"] = fmt(sub.recall_delta_mean.iloc[0], sub.recall_delta_std.iloc[0])
            else:
                row[f"f1_delta_x{ratio}"] = row[f"recall_delta_x{ratio}"] = "-"
        pivot_rows.append(row)

df_pivot = pd.DataFrame(pivot_rows)
df_pivot.to_csv(OUT_DIR / "comparison_table.csv", index=False)

print("\n-- F1 delta (mean +- std, 5 seeds) --------------------------------------")
print(df_pivot[["class","model"] + [f"f1_delta_x{r}" for r in RATIOS]].to_string(index=False))
print("\n-- Recall delta (mean +- std, 5 seeds) ----------------------------------")
print(df_pivot[["class","model"] + [f"recall_delta_x{r}" for r in RATIOS]].to_string(index=False))


# ── plots ──────────────────────────────────────────────────────────────────────
# 1. F1 delta vs ratio — line + shaded std, per class
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, cls in zip(axes, MINORITY_CLS):
    for model in MODELS:
        sub   = agg[(agg.model==model) & (agg["class"]==cls)].sort_values("ratio")
        means = sub.f1_delta_mean.values
        stds  = sub.f1_delta_std.values
        ax.plot(sub.ratio.values, means, marker="o", label=model,
                color=COLORS[model], linewidth=2, markersize=7)
        ax.fill_between(sub.ratio.values, means - stds, means + stds,
                        alpha=0.15, color=COLORS[model])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(RATIOS); ax.set_xlabel("Augmentation ratio")
    ax.set_ylabel("F1 delta"); ax.set_title(cls); ax.legend(fontsize=8)
plt.suptitle("F1 delta vs augmentation ratio (mean +- 1 std, 5 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "f1_delta_vs_ratio.png", dpi=150); plt.close()

# 2. Recall delta vs ratio
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, cls in zip(axes, MINORITY_CLS):
    for model in MODELS:
        sub   = agg[(agg.model==model) & (agg["class"]==cls)].sort_values("ratio")
        means = sub.recall_delta_mean.values
        stds  = sub.recall_delta_std.values
        ax.plot(sub.ratio.values, means, marker="o", label=model,
                color=COLORS[model], linewidth=2, markersize=7)
        ax.fill_between(sub.ratio.values, means - stds, means + stds,
                        alpha=0.15, color=COLORS[model])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(RATIOS); ax.set_xlabel("Augmentation ratio")
    ax.set_ylabel("Recall delta"); ax.set_title(cls); ax.legend(fontsize=8)
plt.suptitle("Recall delta vs augmentation ratio (mean +- 1 std, 5 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "recall_delta_vs_ratio.png", dpi=150); plt.close()

# 3. Absolute F1 grouped bar per class
n_models = len(MODELS)
w = 0.22
offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * w

for cls in MINORITY_CLS:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(RATIOS))
    for i, model in enumerate(MODELS):
        sub  = agg[(agg.model==model) & (agg["class"]==cls)].sort_values("ratio")
        aug  = sub.f1_aug_mean.values
        stds = sub.f1_aug_std.values
        bars = ax.bar(x + offsets[i], aug, w, label=model,
                      color=COLORS[model], alpha=0.85)
        ax.errorbar(x + offsets[i], aug, yerr=stds, fmt="none",
                    color="black", capsize=3, linewidth=1)
        for bar, val in zip(bars, aug):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    base_mean = df_raw[df_raw["class"]==cls]["f1_baseline"].mean()
    base_std  = df_raw[df_raw["class"]==cls]["f1_baseline"].std()
    ax.axhline(base_mean, color="gray", linewidth=1.4, linestyle="--",
               label=f"Baseline {base_mean:.3f}")
    ax.fill_between([-0.5, len(RATIOS)-0.5], base_mean-base_std, base_mean+base_std,
                    color="gray", alpha=0.1)
    ax.set_xticks(x); ax.set_xticklabels([f"x{r}" for r in RATIOS], fontsize=11)
    ax.set_xlabel("Augmentation ratio"); ax.set_ylabel("F1 score (mean +- std)")
    ax.set_ylim(0, 1); ax.set_title(f"Absolute F1 -- {cls}"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"absolute_f1_{cls}.png", dpi=150); plt.close()

print(f"\nSaved to: {OUT_DIR.resolve()}")
print("Ratio experiment complete.")
