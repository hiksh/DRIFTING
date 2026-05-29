"""
Augmentation ratio experiment: x1 / x3 / x5 synthetic samples.
Models: CTGAN, Drifting (standard scaler only).

Step 1 — Generate x3, x5 samples from saved models.
Step 2 — 5-seed RF experiment for each ratio.
Step 3 — Comparison table: ratio × model × class.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/processed")
CTGAN_DIR    = Path("outputs/ctgan")
DRIFT_DIR    = Path("outputs/drifting")
OUT_DIR      = Path("outputs/ratio_experiment")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MINORITY_CLS = ["bot", "brute_force", "xss"]
LABEL_MAP    = {"bot": 1, "brute_force": 2, "xss": 3}
MINORITY_IDX = [1, 2, 3]
BASE_COUNTS  = {"bot": 1360, "brute_force": 1062, "xss": 436}
RATIOS       = [1, 3, 5]
MODELS       = ["CTGAN", "Drifting"]

BENIGN_CAP = 50_000
RF_SEEDS   = [42, 123, 456, 789, 1024]
RF_PARAMS  = dict(n_estimators=100, n_jobs=-1, class_weight="balanced",
                  max_features="sqrt")

feat_cols = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()
D = len(feat_cols)

COLORS = {"CTGAN": "steelblue", "Drifting": "seagreen"}
MARKERS = {1: "o", 3: "s", 5: "^"}


# ── Generator (must match 05_drifting.py) ─────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, d: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, d))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — generate x3 and x5 samples from saved models
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Generating x3 and x5 samples")
print("=" * 60)

def synth_path(model: str, cls: str, ratio: int) -> Path:
    """Canonical path for a synthetic array."""
    d = CTGAN_DIR if model == "CTGAN" else DRIFT_DIR
    suffix = "" if ratio == 1 else f"_x{ratio}"
    return d / f"X_synth_{model.lower()}_standard_{cls}{suffix}.npy"


# ── CTGAN ──────────────────────────────────────────────────────────────────────
for cls in MINORITY_CLS:
    model_path = CTGAN_DIR / f"ctgan_model_standard_{cls}.pkl"
    ctgan = CTGANSynthesizer.load(str(model_path))

    for ratio in [3, 5]:
        out = synth_path("CTGAN", cls, ratio)
        if out.exists():
            print(f"  [skip] {out.name} already exists")
            continue
        n = BASE_COUNTS[cls] * ratio
        t0 = time.perf_counter()
        df_s = ctgan.sample(num_rows=n)
        X_s  = df_s[feat_cols].values.astype(np.float32)
        np.save(out, X_s)
        print(f"  CTGAN  {cls:<14} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.2f}s  -> {out.name}")

# ── Drifting ───────────────────────────────────────────────────────────────────
for cls in MINORITY_CLS:
    model_path = DRIFT_DIR / f"drifting_model_standard_{cls}.pt"
    gen = Generator(D)
    gen.load_state_dict(torch.load(model_path, weights_only=True))
    gen.eval()

    for ratio in [3, 5]:
        out = synth_path("Drifting", cls, ratio)
        if out.exists():
            print(f"  [skip] {out.name} already exists")
            continue
        n = BASE_COUNTS[cls] * ratio
        t0 = time.perf_counter()
        with torch.no_grad():
            X_s = gen(torch.randn(n, D)).numpy().astype(np.float32)
        np.save(out, X_s)
        print(f"  Drifting {cls:<14} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.4f}s  -> {out.name}")

# verify all files
print("\nGenerated files:")
for model in MODELS:
    for cls in MINORITY_CLS:
        for ratio in RATIOS:
            p = synth_path(model, cls, ratio)
            arr = np.load(p)
            print(f"  {p.name:<55} {arr.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 5-seed RF experiment per ratio
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — 5-seed RF experiment")
print("=" * 60)
print(f"  {len(RATIOS)} ratios × {len(RF_SEEDS)} seeds × "
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
    idx = np.concatenate([b, minority_idx])
    return X_tr_full[idx], y_tr_full[idx]


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
        y_te, y_pred, labels=MINORITY_IDX, zero_division=0
    )
    return {c: {"f1": float(f1[k]), "recall": float(rec[k])}
            for k, c in enumerate(MINORITY_CLS)}


raw_rows = []

for seed in RF_SEEDS:
    print(f"  seed={seed}", end="  ", flush=True)
    X_base, y_base = build_base(seed)

    # baseline (computed once per seed, shared across ratios)
    base_m = rf_eval(X_base, y_base, seed)
    print(f"base={sum(y_base==c for c in range(4))[0] if False else len(y_base):,}", end="  ", flush=True)

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
# STEP 3 — aggregate and format comparison tables
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

# ── display table: model | class | x1 | x3 | x5 (F1 delta mean±std) ──────────
def fmt(mean, std): return f"{mean:+.4f} ± {std:.4f}"

pivot_rows = []
for cls in MINORITY_CLS:
    for model in MODELS:
        row = {"class": cls, "model": model}
        for ratio in RATIOS:
            sub = agg[(agg.model==model) & (agg.ratio==ratio) & (agg["class"]==cls)]
            if len(sub):
                row[f"f1_delta_x{ratio}"]     = fmt(sub.f1_delta_mean.iloc[0],     sub.f1_delta_std.iloc[0])
                row[f"recall_delta_x{ratio}"]  = fmt(sub.recall_delta_mean.iloc[0], sub.recall_delta_std.iloc[0])
            else:
                row[f"f1_delta_x{ratio}"] = row[f"recall_delta_x{ratio}"] = "—"
        pivot_rows.append(row)

df_pivot = pd.DataFrame(pivot_rows)
df_pivot.to_csv(OUT_DIR / "comparison_table.csv", index=False)

print("\n── F1 delta  (mean ± std, 5 seeds) ─────────────────────────────────────")
f1_cols = ["class", "model"] + [f"f1_delta_x{r}" for r in RATIOS]
print(df_pivot[f1_cols].to_string(index=False))

print("\n── Recall delta  (mean ± std, 5 seeds) ─────────────────────────────────")
rec_cols = ["class", "model"] + [f"recall_delta_x{r}" for r in RATIOS]
print(df_pivot[rec_cols].to_string(index=False))


# ── plots ──────────────────────────────────────────────────────────────────────
x_pos = np.array(RATIOS, dtype=float)

# 1. F1 delta vs ratio — line plots per class
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, cls in zip(axes, MINORITY_CLS):
    for model in MODELS:
        sub = agg[(agg.model==model) & (agg["class"]==cls)].sort_values("ratio")
        means = sub.f1_delta_mean.values
        stds  = sub.f1_delta_std.values
        ax.plot(sub.ratio.values, means, marker="o", label=model,
                color=COLORS[model], linewidth=2, markersize=7)
        ax.fill_between(sub.ratio.values, means - stds, means + stds,
                        alpha=0.18, color=COLORS[model])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(RATIOS); ax.set_xlabel("Augmentation ratio")
    ax.set_ylabel("F1 delta"); ax.set_title(f"{cls}")
    ax.legend(fontsize=9)

plt.suptitle("F1 delta vs augmentation ratio (mean ± 1 std, 5 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "f1_delta_vs_ratio.png", dpi=150); plt.close()

# 2. Recall delta vs ratio
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, cls in zip(axes, MINORITY_CLS):
    for model in MODELS:
        sub = agg[(agg.model==model) & (agg["class"]==cls)].sort_values("ratio")
        means = sub.recall_delta_mean.values
        stds  = sub.recall_delta_std.values
        ax.plot(sub.ratio.values, means, marker="o", label=model,
                color=COLORS[model], linewidth=2, markersize=7)
        ax.fill_between(sub.ratio.values, means - stds, means + stds,
                        alpha=0.18, color=COLORS[model])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(RATIOS); ax.set_xlabel("Augmentation ratio")
    ax.set_ylabel("Recall delta"); ax.set_title(f"{cls}")
    ax.legend(fontsize=9)

plt.suptitle("Recall delta vs augmentation ratio (mean ± 1 std, 5 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "recall_delta_vs_ratio.png", dpi=150); plt.close()

# 3. Grouped bar: absolute F1 at each ratio — per class
for cls in MINORITY_CLS:
    fig, ax = plt.subplots(figsize=(9, 5))
    n_groups = len(RATIOS)
    x  = np.arange(n_groups)
    w  = 0.35

    for i, model in enumerate(MODELS):
        sub  = agg[(agg.model==model) & (agg["class"]==cls)].sort_values("ratio")
        base = sub.f1_aug_mean.values - sub.f1_delta_mean.values   # baseline
        aug  = sub.f1_aug_mean.values
        stds = sub.f1_aug_std.values
        bars = ax.bar(x + (i - 0.5) * w, aug, w, label=model,
                      color=COLORS[model], alpha=0.85)
        ax.errorbar(x + (i - 0.5) * w, aug, yerr=stds, fmt="none",
                    color="black", capsize=4, linewidth=1)
        for bar, val in zip(bars, aug):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # baseline (ratio-independent mean across seeds)
    base_mean = (df_raw[df_raw["class"]==cls]["f1_baseline"].mean())
    base_std  = df_raw[df_raw["class"]==cls]["f1_baseline"].std()
    ax.axhline(base_mean, color="gray", linewidth=1.4, linestyle="--",
               label=f"Baseline {base_mean:.3f}")
    ax.fill_between([-0.5, n_groups - 0.5],
                    base_mean - base_std, base_mean + base_std,
                    color="gray", alpha=0.1)

    ax.set_xticks(x); ax.set_xticklabels([f"x{r}" for r in RATIOS], fontsize=11)
    ax.set_xlabel("Augmentation ratio"); ax.set_ylabel("F1 score (mean ± std)")
    ax.set_ylim(0, 1); ax.set_title(f"Absolute F1 — {cls}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"absolute_f1_{cls}.png", dpi=150); plt.close()

print(f"\nSaved outputs to: {OUT_DIR.resolve()}")
print("Ratio experiment complete.")
