"""
CICIDS2017 EDA — class distribution, feature stats, imbalance ratio.
Used as the starting point for Drifting Models NIDS data generation experiments.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("cicids2017")
OUT_DIR   = Path("outputs/eda")
TRAIN_CSV = DATA_DIR / "training-flow.csv"
TEST_CSV  = DATA_DIR / "test-flow.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "attack_name"
FLAG_COL  = "attack_flag"
STEP_COL  = "attack_step"
META_COLS = [LABEL_COL, FLAG_COL, STEP_COL]

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading data …")
train = pd.read_csv(TRAIN_CSV, encoding="utf-8", low_memory=False)
test  = pd.read_csv(TEST_CSV,  encoding="utf-8", low_memory=False)

# Normalise label strings (test set has garbled "web attack …" entries)
def clean_label(s: str) -> str:
    s = str(s).strip().lower()
    if "brute force" in s:  return "brute force"
    if "xss"         in s:  return "xss"
    if "sql"         in s:  return "sql injection"
    return s

train[LABEL_COL] = train[LABEL_COL].map(clean_label)
test[LABEL_COL]  = test[LABEL_COL].map(clean_label)

feat_cols = [c for c in train.columns if c not in META_COLS]

print(f"Train : {train.shape[0]:>9,} rows × {len(feat_cols)} features")
print(f"Test  : {test.shape[0]:>9,} rows × {len(feat_cols)} features")

# ── 1. class distribution ──────────────────────────────────────────────────────
def class_stats(df: pd.DataFrame, split: str) -> pd.DataFrame:
    counts = df[LABEL_COL].value_counts()
    pct    = counts / counts.sum() * 100
    stats  = pd.DataFrame({"count": counts, "pct": pct.round(3)})
    stats.index.name = "class"
    stats["split"] = split
    return stats

train_dist = class_stats(train, "train")
test_dist  = class_stats(test,  "test")

print("\n── Train class distribution ──────────────────────────────────────────")
print(train_dist[["count", "pct"]].to_string())

print("\n── Test class distribution ───────────────────────────────────────────")
print(test_dist[["count", "pct"]].to_string())

# imbalance ratio (majority / minority, excluding benign from minority calc)
attack_counts = train_dist.loc[train_dist.index != "benign", "count"]
print(f"\nImbalance ratio (benign / rarest attack): "
      f"{train_dist.loc['benign','count'] / attack_counts.min():.1f}×")
print(f"Imbalance ratio (most common attack / rarest attack): "
      f"{attack_counts.max() / attack_counts.min():.1f}×")

# ── 2. attack_step distribution ────────────────────────────────────────────────
print("\n── Train attack_step distribution ────────────────────────────────────")
print(train[STEP_COL].value_counts().to_string())

# ── 3. feature statistics ──────────────────────────────────────────────────────
print("\n── Feature statistics (train) ────────────────────────────────────────")
desc = train[feat_cols].describe().T
desc["cv"] = (desc["std"] / desc["mean"].replace(0, np.nan)).round(3)
with pd.option_context("display.max_rows", 10, "display.float_format", "{:.3f}".format):
    print(desc[["mean", "std", "min", "max", "cv"]].head(10))

zero_var = (desc["std"] == 0).sum()
print(f"\nZero-variance features: {zero_var}")

# ── 4. plots ───────────────────────────────────────────────────────────────────
ATTACK_ORDER = train_dist.index.tolist()   # sorted by frequency

# 4a. bar chart — train class counts (log scale)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(ATTACK_ORDER)),
              train_dist.loc[ATTACK_ORDER, "count"].values,
              color=["steelblue" if c == "benign" else "tomato" for c in ATTACK_ORDER])
ax.set_xticks(range(len(ATTACK_ORDER)))
ax.set_xticklabels(ATTACK_ORDER, rotation=35, ha="right", fontsize=9)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylabel("Sample count (log scale)")
ax.set_title("CICIDS2017 — Train class distribution")

for bar, val in zip(bars, train_dist.loc[ATTACK_ORDER, "count"].values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
            f"{val:,}", ha="center", va="bottom", fontsize=7, rotation=45)

plt.tight_layout()
fig.savefig(OUT_DIR / "class_distribution_train.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR / 'class_distribution_train.png'}")

# 4b. train vs test proportion comparison
merged = train_dist[["pct"]].rename(columns={"pct":"train"}).join(
         test_dist[["pct"]].rename(columns={"pct":"test"}), how="outer").fillna(0)
merged = merged.loc[ATTACK_ORDER]

x = np.arange(len(ATTACK_ORDER))
w = 0.38
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, merged["train"].values, w, label="Train", color="steelblue")
ax.bar(x + w/2, merged["test"].values,  w, label="Test",  color="darkorange", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(ATTACK_ORDER, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Proportion (%)")
ax.set_title("CICIDS2017 — Train vs Test class proportion")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "class_proportion_train_vs_test.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'class_proportion_train_vs_test.png'}")

# 4c. feature correlation heatmap (sampled, top-variance features)
sample = train[feat_cols].sample(n=min(50_000, len(train)), random_state=42)
var_rank = sample.var().sort_values(ascending=False)
top_feats = var_rank.head(20).index.tolist()
corr = sample[top_feats].corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(top_feats)))
ax.set_xticklabels([f.replace(" ", "\n") for f in top_feats], fontsize=7, rotation=45, ha="right")
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats, fontsize=7)
ax.set_title("Top-20 variance features — Pearson correlation (train, 50k sample)")
plt.tight_layout()
fig.savefig(OUT_DIR / "feature_correlation_top20.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'feature_correlation_top20.png'}")

# 4d. per-class feature mean heatmap (normalised)
sample_full = train.sample(n=min(100_000, len(train)), random_state=42)
top10 = var_rank.head(10).index.tolist()
class_means = (sample_full
               .groupby(LABEL_COL)[top10]
               .mean()
               .reindex(ATTACK_ORDER))
normed = (class_means - class_means.min()) / (class_means.max() - class_means.min() + 1e-9)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(normed.values, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(top10)))
ax.set_xticklabels([f.replace(" ", "\n") for f in top10], fontsize=8, rotation=45, ha="right")
ax.set_yticks(range(len(ATTACK_ORDER)))
ax.set_yticklabels(ATTACK_ORDER, fontsize=8)
ax.set_title("Per-class normalised feature means (top-10 variance features)")
plt.tight_layout()
fig.savefig(OUT_DIR / "class_feature_means.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'class_feature_means.png'}")

# ── 5. save summary CSV ────────────────────────────────────────────────────────
summary = train_dist[["count", "pct"]].copy()
summary.columns = ["train_count", "train_pct"]
summary = summary.join(test_dist[["count", "pct"]].rename(
    columns={"count": "test_count", "pct": "test_pct"}), how="outer").fillna(0)
summary.to_csv(OUT_DIR / "class_distribution_summary.csv")
print(f"Saved: {OUT_DIR / 'class_distribution_summary.csv'}")

print("\nEDA complete.")
