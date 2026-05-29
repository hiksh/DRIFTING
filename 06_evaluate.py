"""
Evaluation pipeline — StandardScaler results only.

Fidelity (per model × class, computed once — deterministic):
  - KS statistic: ks_2samp per feature (real vs synthetic) → mean / max / p90
  - Correlation fidelity: Frobenius norm of (corr_real - corr_synth)

Utility (5-seed repeated experiment):
  - Each seed controls both benign subsampling and RF random state.
  - Baseline RF  : original data (benign capped, all minority)
  - Augmented RF : original + synthetic minority for all 3 classes
  - Reported as F1 delta mean ± std and Recall delta mean ± std.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
OUT_DIR  = Path("outputs/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTH_DIRS = {
    "CTGAN":    Path("outputs/ctgan"),
    "TabDDPM":  Path("outputs/tabddpm"),
    "Drifting": Path("outputs/drifting"),
}

MODELS       = list(SYNTH_DIRS.keys())
MINORITY_CLS = ["bot", "brute_force", "xss"]
LABEL_MAP    = {"bot": 1, "brute_force": 2, "xss": 3}
MINORITY_IDX = [1, 2, 3]
BENIGN_CAP   = 50_000

RF_SEEDS  = [42, 123, 456, 789, 1024]
RF_PARAMS = dict(n_estimators=100, n_jobs=-1, class_weight="balanced",
                 max_features="sqrt")

COLORS = {"CTGAN": "steelblue", "TabDDPM": "darkorange", "Drifting": "seagreen"}

# ── load static data ───────────────────────────────────────────────────────────
print("Loading data …")
X_tr_full = np.load(DATA_DIR / "X_train_standard.npy")
y_tr_full = np.load(DATA_DIR / "y_train.npy")
X_te      = np.load(DATA_DIR / "X_test_standard.npy")
y_te      = np.load(DATA_DIR / "y_test.npy")

benign_idx    = np.where(y_tr_full == 0)[0]
minority_idx  = np.where(y_tr_full != 0)[0]   # fixed across seeds

real_by_cls = {
    c: np.load(DATA_DIR / f"X_train_standard_{c}.npy") for c in MINORITY_CLS
}

# preload synthetic arrays once (same across seeds)
synth_by_model: dict[str, dict[str, np.ndarray]] = {}
for model_name, synth_dir in SYNTH_DIRS.items():
    synth_by_model[model_name] = {}
    for cls in MINORITY_CLS:
        hits = list(synth_dir.glob(f"X_synth_*standard_{cls}.npy"))
        if hits:
            synth_by_model[model_name][cls] = np.load(hits[0])

print(f"  Full train : {len(X_tr_full):,}  |  Test: {len(X_te):,}")
print(f"  Benign pool: {len(benign_idx):,}  (cap={BENIGN_CAP:,} per seed)")


# ── helpers ────────────────────────────────────────────────────────────────────
def build_base(seed: int):
    """Subsample benign + keep all minority. Both vary by seed."""
    rng = np.random.default_rng(seed)
    b   = rng.choice(benign_idx, size=min(BENIGN_CAP, len(benign_idx)), replace=False)
    idx = np.concatenate([b, minority_idx])
    return X_tr_full[idx], y_tr_full[idx]


def build_augmented(X_base, y_base, model_name: str):
    """Append synthetic minority rows to base training set."""
    xs, ys = [X_base], [y_base]
    for cls in MINORITY_CLS:
        if cls in synth_by_model[model_name]:
            X_s = synth_by_model[model_name][cls]
            xs.append(X_s)
            ys.append(np.full(len(X_s), LABEL_MAP[cls], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def rf_eval(X_tr, y_tr, seed: int) -> dict[str, dict]:
    """Train RF and return {class: {f1, recall}} for minority classes."""
    rf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    _, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=MINORITY_IDX, zero_division=0
    )
    cls_names = ["bot", "brute_force", "xss"]
    return {c: {"f1": float(f1[k]), "recall": float(rec[k])}
            for k, c in enumerate(cls_names)}


def ks_stats(X_real, X_synth) -> dict:
    vals = [ks_2samp(X_real[:, j], X_synth[:, j]).statistic
            for j in range(X_real.shape[1])]
    a = np.array(vals)
    return {"ks_mean": float(a.mean()), "ks_max": float(a.max()),
            "ks_p90": float(np.percentile(a, 90)), "_arr": a}


def corr_frob(X_real, X_synth) -> float:
    active = (X_real.std(0) > 1e-8) & (X_synth.std(0) > 1e-8)
    if active.sum() < 2:
        return float("nan")
    cr = np.corrcoef(X_real[:, active].astype(np.float64).T)
    cs = np.corrcoef(X_synth[:, active].astype(np.float64).T)
    np.nan_to_num(cr, copy=False, nan=0.0)
    np.nan_to_num(cs, copy=False, nan=0.0)
    return float(np.linalg.norm(cr - cs, "fro"))


# ── fidelity (once, deterministic) ────────────────────────────────────────────
print("\nComputing fidelity metrics …")
fidelity_rows = []
for model_name in MODELS:
    for cls in MINORITY_CLS:
        if cls not in synth_by_model[model_name]:
            continue
        X_r = real_by_cls[cls]
        X_s = synth_by_model[model_name][cls]
        ks  = ks_stats(X_r, X_s)
        frob = corr_frob(X_r, X_s)
        fidelity_rows.append({
            "model": model_name, "class": cls,
            "ks_mean":   round(ks["ks_mean"], 4),
            "ks_max":    round(ks["ks_max"],  4),
            "ks_p90":    round(ks["ks_p90"],  4),
            "corr_frob": round(frob, 4) if not np.isnan(frob) else None,
        })
        print(f"  {model_name:<10} {cls:<14}  "
              f"KS_mean={ks['ks_mean']:.4f}  KS_max={ks['ks_max']:.4f}  "
              f"Corr_Frob={frob:.4f}")

df_fid = pd.DataFrame(fidelity_rows)
df_fid.to_csv(OUT_DIR / "fidelity_summary.csv", index=False)


# ── utility: 5-seed repeated experiment ───────────────────────────────────────
print(f"\nUtility: {len(RF_SEEDS)} seeds × "
      f"(1 baseline + {len(MODELS)} models) = "
      f"{len(RF_SEEDS)*(1+len(MODELS))} RF trainings")

raw_rows = []   # one row per (seed, model, class)

for seed in RF_SEEDS:
    print(f"\n  seed={seed} ", end="", flush=True)

    X_base, y_base = build_base(seed)

    # baseline
    t0 = time.perf_counter()
    base_m = rf_eval(X_base, y_base, seed)
    print(f"base={time.perf_counter()-t0:.1f}s", end="  ", flush=True)

    # augmented per model
    for model_name in MODELS:
        X_aug, y_aug = build_augmented(X_base, y_base, model_name)
        t0 = time.perf_counter()
        aug_m = rf_eval(X_aug, y_aug, seed)
        print(f"{model_name}={time.perf_counter()-t0:.1f}s", end="  ", flush=True)

        for cls in MINORITY_CLS:
            raw_rows.append({
                "seed":             seed,
                "model":            model_name,
                "class":            cls,
                "f1_baseline":      base_m[cls]["f1"],
                "f1_augmented":     aug_m[cls]["f1"],
                "f1_delta":         aug_m[cls]["f1"]     - base_m[cls]["f1"],
                "recall_baseline":  base_m[cls]["recall"],
                "recall_augmented": aug_m[cls]["recall"],
                "recall_delta":     aug_m[cls]["recall"] - base_m[cls]["recall"],
            })

print()

df_raw = pd.DataFrame(raw_rows)
df_raw.to_csv(OUT_DIR / "utility_raw.csv", index=False)


# ── aggregate mean ± std ───────────────────────────────────────────────────────
agg = (df_raw
       .groupby(["model", "class"])
       .agg(
           f1_base_mean      =("f1_baseline",  "mean"),
           f1_base_std       =("f1_baseline",  "std"),
           f1_aug_mean       =("f1_augmented", "mean"),
           f1_aug_std        =("f1_augmented", "std"),
           f1_delta_mean     =("f1_delta",     "mean"),
           f1_delta_std      =("f1_delta",     "std"),
           recall_base_mean  =("recall_baseline",  "mean"),
           recall_aug_mean   =("recall_augmented", "mean"),
           recall_delta_mean =("recall_delta",     "mean"),
           recall_delta_std  =("recall_delta",     "std"),
       )
       .round(4)
       .reset_index())

agg.to_csv(OUT_DIR / "utility_summary.csv", index=False)

# human-readable version: "mean ± std"
def fmt(mean_col, std_col, df=agg):
    return df.apply(lambda r: f"{r[mean_col]:+.4f} ± {r[std_col]:.4f}", axis=1)

display = agg[["model", "class"]].copy()
display["f1_base"]        = agg.apply(lambda r: f"{r.f1_base_mean:.4f} ± {r.f1_base_std:.4f}", axis=1)
display["f1_aug"]         = agg.apply(lambda r: f"{r.f1_aug_mean:.4f} ± {r.f1_aug_std:.4f}",  axis=1)
display["f1_delta"]       = fmt("f1_delta_mean",     "f1_delta_std")
display["recall_delta"]   = fmt("recall_delta_mean", "recall_delta_std")
display.to_csv(OUT_DIR / "utility_display.csv", index=False)

combined = agg.merge(df_fid, on=["model", "class"], how="left")
combined.to_csv(OUT_DIR / "evaluation_combined.csv", index=False)

print("\n\n── Fidelity ─────────────────────────────────────────────────────────")
print(df_fid.to_string(index=False))

print("\n\n── Utility (mean ± std over 5 seeds) ────────────────────────────────")
print(display.to_string(index=False))


# ── plots ──────────────────────────────────────────────────────────────────────
x = np.arange(len(MINORITY_CLS))
w = 0.25

# helper: extract (means, stds) for a column per model
def get_vals(df, model, mean_col, std_col=None):
    rows = [df[(df.model == model) & (df["class"] == c)] for c in MINORITY_CLS]
    means = [float(r[mean_col].iloc[0]) if len(r) else np.nan for r in rows]
    stds  = ([float(r[std_col].iloc[0])  if len(r) else np.nan for r in rows]
             if std_col else None)
    return means, stds


# 1. Fidelity — KS mean
fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(MODELS):
    vals, _ = get_vals(df_fid, model, "ks_mean")
    ax.bar(x + (i - 1) * w, vals, w, label=model, color=COLORS[model], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(MINORITY_CLS, fontsize=11)
ax.set_ylabel("Mean KS statistic (lower = better)"); ax.set_ylim(0, 1)
ax.set_title("Fidelity — Mean KS statistic per model × class")
ax.legend(); plt.tight_layout()
fig.savefig(OUT_DIR / "fidelity_ks_mean.png", dpi=150); plt.close()

# 2. Fidelity — Correlation Frobenius
fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(MODELS):
    vals, _ = get_vals(df_fid, model, "corr_frob")
    ax.bar(x + (i - 1) * w, vals, w, label=model, color=COLORS[model], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(MINORITY_CLS, fontsize=11)
ax.set_ylabel("Correlation Frobenius norm (lower = better)")
ax.set_title("Fidelity — Correlation matrix distance per model × class")
ax.legend(); plt.tight_layout()
fig.savefig(OUT_DIR / "fidelity_corr_frob.png", dpi=150); plt.close()

# 3. Utility — F1 delta with error bars
fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(MODELS):
    means, stds = get_vals(agg, model, "f1_delta_mean", "f1_delta_std")
    bars = ax.bar(x + (i-1)*w, means, w, label=model,
                  color=COLORS[model], alpha=0.85)
    ax.errorbar(x + (i-1)*w, means, yerr=stds, fmt="none",
                color="black", capsize=4, linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(MINORITY_CLS, fontsize=11)
ax.set_ylabel("F1 delta (mean ± std, 5 seeds)")
ax.set_title("Utility — F1 improvement from synthetic augmentation")
ax.legend(); plt.tight_layout()
fig.savefig(OUT_DIR / "utility_f1_delta.png", dpi=150); plt.close()

# 4. Utility — Recall delta with error bars
fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(MODELS):
    means, stds = get_vals(agg, model, "recall_delta_mean", "recall_delta_std")
    ax.bar(x + (i-1)*w, means, w, label=model, color=COLORS[model], alpha=0.85)
    ax.errorbar(x + (i-1)*w, means, yerr=stds, fmt="none",
                color="black", capsize=4, linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(MINORITY_CLS, fontsize=11)
ax.set_ylabel("Recall delta (mean ± std, 5 seeds)")
ax.set_title("Utility — Recall improvement from synthetic augmentation")
ax.legend(); plt.tight_layout()
fig.savefig(OUT_DIR / "utility_recall_delta.png", dpi=150); plt.close()

# 5. Absolute F1: baseline vs augmented, per class — mean ± std error bars
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, cls in zip(axes, MINORITY_CLS):
    sub = agg[agg["class"] == cls].set_index("model")
    base_mean = sub["f1_base_mean"].iloc[0]
    base_std  = sub["f1_base_std"].iloc[0]
    labels = ["Baseline"] + MODELS
    colors = ["gray"] + [COLORS[m] for m in MODELS]
    means  = [base_mean] + [sub.loc[m, "f1_aug_mean"] for m in MODELS]
    stds   = [base_std]  + [sub.loc[m, "f1_aug_std"]  for m in MODELS]
    bars   = ax.bar(labels, means, color=colors, alpha=0.85)
    ax.errorbar(range(len(labels)), means, yerr=stds, fmt="none",
                color="black", capsize=5, linewidth=1.2)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(cls, fontsize=12); ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score (mean ± std)"); ax.tick_params(axis="x", rotation=20)
plt.suptitle("Utility — F1 per class: Baseline vs Augmented (5 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "utility_f1_absolute.png", dpi=150); plt.close()

print(f"\nSaved to {OUT_DIR.resolve()}")
print("Evaluation complete.")
