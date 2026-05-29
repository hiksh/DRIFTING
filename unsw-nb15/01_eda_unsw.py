"""
EDA for UNSW-NB15 — class distribution, feature stats, imbalance analysis.
Outputs to outputs/eda_unsw/.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA_DIR = Path("unsw-nb15")
OUT_DIR  = Path("outputs/eda_unsw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "attack_name"
FLAG_COL  = "attack_flag"
STEP_COL  = "attack_step"
META_COLS = [LABEL_COL, FLAG_COL, STEP_COL]
CAT_COLS  = ["protocol", "service"]

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading data ...")
train = pd.read_csv(DATA_DIR / "training-flow.csv", low_memory=False)
test  = pd.read_csv(DATA_DIR / "test-flow.csv",  low_memory=False)

num_cols = [c for c in train.columns if c not in META_COLS and c not in CAT_COLS]

print(f"Train : {len(train):>8,} rows x {len(train.columns)} cols")
print(f"Test  : {len(test):>8,} rows x {len(test.columns)} cols")
print(f"Numerical features : {len(num_cols)}")
print(f"Categorical features: {CAT_COLS}")

# ── 1. class distribution ──────────────────────────────────────────────────────
def class_stats(df, split):
    counts = df[LABEL_COL].value_counts()
    return pd.DataFrame({
        "count": counts,
        "pct":   (counts / counts.sum() * 100).round(3),
        "split": split,
    })

train_dist = class_stats(train, "train")
test_dist  = class_stats(test,  "test")
CLASS_ORDER = train_dist.index.tolist()

print("\n── Train class distribution ──────────────────────────────────────────")
print(f"  {'Class':<22} {'Count':>8}  {'%':>7}")
print("  " + "-"*42)
for cls in CLASS_ORDER:
    n = train_dist.loc[cls, "count"]
    p = train_dist.loc[cls, "pct"]
    print(f"  {cls:<22} {n:>8,}  {p:>6.3f}%")

print("\n── Test class distribution ───────────────────────────────────────────")
print(f"  {'Class':<22} {'Count':>8}  {'%':>7}")
print("  " + "-"*42)
for cls in CLASS_ORDER:
    if cls not in test_dist.index:
        continue
    n = test_dist.loc[cls, "count"]
    p = test_dist.loc[cls, "pct"]
    print(f"  {cls:<22} {n:>8,}  {p:>6.3f}%")

# imbalance ratios
normal_n    = train_dist.loc["Normal", "count"]
attack_cnts = train_dist.loc[train_dist.index != "Normal", "count"]
print(f"\nNormal / rarest attack  : {normal_n / attack_cnts.min():.1f}x  ({attack_cnts.idxmin()}, n={attack_cnts.min()})")
print(f"Most common / rarest attack: {attack_cnts.max() / attack_cnts.min():.1f}x")

# ── 2. attack_flag / attack_step ───────────────────────────────────────────────
print("\n── attack_flag ───────────────────────────────────────────────────────")
print(train[FLAG_COL].value_counts().to_string())

print("\n── attack_step ───────────────────────────────────────────────────────")
print(train[STEP_COL].value_counts().to_string())

# ── 3. feature statistics ──────────────────────────────────────────────────────
print("\n── Numerical feature stats (train) ───────────────────────────────────")
desc = train[num_cols].describe().T
desc["cv"] = (desc["std"] / desc["mean"].replace(0, np.nan)).round(3)
zero_var = (desc["std"] == 0).sum()
print(desc[["mean","std","min","max","cv"]].head(10).to_string())
print(f"\nZero-variance features: {zero_var}")
if zero_var:
    zv_feats = desc[desc["std"] == 0].index.tolist()
    print(f"  {zv_feats}")

print("\n── Categorical feature cardinality ───────────────────────────────────")
for col in CAT_COLS:
    print(f"  {col}: {train[col].nunique()} unique values")
    if train[col].nunique() <= 15:
        vc = train[col].value_counts().head(10)
        for v, c in vc.items():
            print(f"    {v:<20} {c:>7,}  ({c/len(train)*100:.2f}%)")

# ── 4. plots ───────────────────────────────────────────────────────────────────
# 4a. class distribution bar (log scale)
fig, ax = plt.subplots(figsize=(11, 5))
colors = ["steelblue" if c == "Normal" else "tomato" for c in CLASS_ORDER]
bars = ax.bar(range(len(CLASS_ORDER)),
              train_dist.loc[CLASS_ORDER, "count"].values,
              color=colors)
ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right", fontsize=9)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylabel("Sample count (log scale)")
ax.set_title("UNSW-NB15 — Train class distribution")
for bar, val in zip(bars, train_dist.loc[CLASS_ORDER, "count"].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2,
            f"{val:,}", ha="center", va="bottom", fontsize=7.5, rotation=40)
plt.tight_layout()
fig.savefig(OUT_DIR / "class_distribution_train.png", dpi=150); plt.close()

# 4b. train vs test proportion
merged = (train_dist[["pct"]].rename(columns={"pct":"train"})
          .join(test_dist[["pct"]].rename(columns={"pct":"test"}), how="outer")
          .fillna(0).loc[CLASS_ORDER])
x = np.arange(len(CLASS_ORDER)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, merged["train"].values, w, label="Train", color="steelblue")
ax.bar(x + w/2, merged["test"].values,  w, label="Test",  color="darkorange", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Proportion (%)"); ax.set_title("UNSW-NB15 — Train vs Test class proportion")
ax.legend(); plt.tight_layout()
fig.savefig(OUT_DIR / "class_proportion_train_vs_test.png", dpi=150); plt.close()

# 4c. imbalance bar chart (sorted count, linear)
fig, ax = plt.subplots(figsize=(11, 5))
sorted_cls   = train_dist["count"].sort_values(ascending=False)
sorted_clrs  = ["steelblue" if c == "Normal" else "tomato" for c in sorted_cls.index]
bars = ax.bar(range(len(sorted_cls)), sorted_cls.values, color=sorted_clrs)
ax.set_xticks(range(len(sorted_cls)))
ax.set_xticklabels(sorted_cls.index, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Sample count")
ax.set_title("UNSW-NB15 — Train class counts (linear scale)")
for bar, val in zip(bars, sorted_cls.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 300,
            f"{val:,}", ha="center", va="bottom", fontsize=8, rotation=40)
plt.tight_layout()
fig.savefig(OUT_DIR / "class_counts_linear.png", dpi=150); plt.close()

# 4d. correlation heatmap (numerical, top-variance features, sampled)
sample = train[num_cols].sample(n=min(50_000, len(train)), random_state=42)
var_rank = sample.var().sort_values(ascending=False)
top20 = var_rank.head(20).index.tolist()
corr  = sample[top20].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(top20))); ax.set_yticks(range(len(top20)))
ax.set_xticklabels([f.replace("_","\n") for f in top20], fontsize=7, rotation=45, ha="right")
ax.set_yticklabels(top20, fontsize=7)
ax.set_title("Top-20 variance features — Pearson correlation (train, 50k sample)")
plt.tight_layout()
fig.savefig(OUT_DIR / "feature_correlation_top20.png", dpi=150); plt.close()

# ── 5. save summary CSV ────────────────────────────────────────────────────────
summary = train_dist[["count","pct"]].copy()
summary.columns = ["train_count","train_pct"]
summary = summary.join(
    test_dist[["count","pct"]].rename(columns={"count":"test_count","pct":"test_pct"}),
    how="outer").fillna(0)
summary.to_csv(OUT_DIR / "class_distribution_summary.csv")

print(f"\nSaved plots and CSV to: {OUT_DIR.resolve()}")
print("EDA complete.")
