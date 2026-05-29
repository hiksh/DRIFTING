"""
Preprocessing pipeline for CICIDS2017 NIDS data.

Steps:
  1. Load train/test CSVs
  2. Filter to target classes: benign, bot, brute force, xss
  3. Drop 8 zero-variance features
  4. Fit StandardScaler and RobustScaler on train, transform both splits
  5. Save to data/processed/

Both scalers are saved because Drifting Models uses an L2 kernel that is
sensitive to inter-feature scale differences; RobustScaler is more appropriate
when outlier-heavy network traffic skews the standard deviation.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("cicids2017")
OUT_DIR   = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "training-flow.csv"
TEST_CSV  = DATA_DIR / "test-flow.csv"

META_COLS    = ["attack_name", "attack_flag", "attack_step"]
TARGET_CLASSES = ["benign", "bot", "brute force", "xss"]

ZERO_VAR_FEATS = [
    "bwd psh flags",
    "bwd urg flags",
    "fwd avg bytes/bulk",
    "fwd avg packets/bulk",
    "fwd avg bulk rate",
    "bwd avg bytes/bulk",
    "bwd avg packets/bulk",
    "bwd avg bulk rate",
]

# ── helpers ────────────────────────────────────────────────────────────────────
def clean_label(s: str) -> str:
    s = str(s).strip().lower()
    if "brute force" in s: return "brute force"
    if "xss"         in s: return "xss"
    if "sql"         in s: return "sql injection"
    return s

# ── 1. load ────────────────────────────────────────────────────────────────────
print("Loading CSVs …")
train_raw = pd.read_csv(TRAIN_CSV, encoding="utf-8", low_memory=False)
test_raw  = pd.read_csv(TEST_CSV,  encoding="utf-8", low_memory=False)

train_raw["attack_name"] = train_raw["attack_name"].map(clean_label)
test_raw["attack_name"]  = test_raw["attack_name"].map(clean_label)

print(f"  Raw train : {len(train_raw):>9,}")
print(f"  Raw test  : {len(test_raw):>9,}")

# ── 2. filter to target classes ────────────────────────────────────────────────
train = train_raw[train_raw["attack_name"].isin(TARGET_CLASSES)].copy()
test  = test_raw[test_raw["attack_name"].isin(TARGET_CLASSES)].copy()

print(f"\nAfter filtering to {TARGET_CLASSES}:")
print(f"  Train : {len(train):>7,}")
print(f"  Test  : {len(test):>7,}")
print("\n  Train class counts:")
for cls, cnt in train["attack_name"].value_counts().items():
    print(f"    {cls:<15} {cnt:>6,}  ({cnt/len(train)*100:.2f}%)")
print("\n  Test class counts:")
for cls, cnt in test["attack_name"].value_counts().items():
    print(f"    {cls:<15} {cnt:>6,}  ({cnt/len(test)*100:.2f}%)")

# ── 3. separate features / labels ─────────────────────────────────────────────
feat_cols = [c for c in train.columns
             if c not in META_COLS and c not in ZERO_VAR_FEATS]

print(f"\nFeatures after dropping zero-variance: {len(feat_cols)}  (removed {len(ZERO_VAR_FEATS)})")

X_train = train[feat_cols].values.astype(np.float64)
X_test  = test[feat_cols].values.astype(np.float64)

# encode labels: benign=0, bot=1, brute force=2, xss=3  (alphabetical → deterministic)
le = LabelEncoder()
le.fit(sorted(TARGET_CLASSES))
y_train = le.transform(train["attack_name"])
y_test  = le.transform(test["attack_name"])

print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 4. fit & apply scalers ─────────────────────────────────────────────────────
scalers = {
    "standard": StandardScaler(),
    "robust":   RobustScaler(quantile_range=(5.0, 95.0)),
}

for name, scaler in scalers.items():
    print(f"\nFitting {name} scaler on train …")
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    # save arrays
    np.save(OUT_DIR / f"X_train_{name}.npy", X_tr_scaled.astype(np.float32))
    np.save(OUT_DIR / f"X_test_{name}.npy",  X_te_scaled.astype(np.float32))

    # save scaler object
    with open(OUT_DIR / f"scaler_{name}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"  Saved X_train_{name}.npy  {X_tr_scaled.shape}")
    print(f"  Saved X_test_{name}.npy   {X_te_scaled.shape}")
    print(f"  Saved scaler_{name}.pkl")

# ── 5. save labels and metadata ───────────────────────────────────────────────
np.save(OUT_DIR / "y_train.npy", y_train)
np.save(OUT_DIR / "y_test.npy",  y_test)

with open(OUT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# feature names (needed to reconstruct DataFrames later)
feat_series = pd.Series(feat_cols, name="feature")
feat_series.to_csv(OUT_DIR / "feature_names.csv", index=False)

# zero-variance list (for reference)
pd.Series(ZERO_VAR_FEATS, name="feature").to_csv(
    OUT_DIR / "zero_variance_features.csv", index=False)

print(f"\nSaved y_train.npy          {y_train.shape}")
print(f"Saved y_test.npy           {y_test.shape}")
print(f"Saved label_encoder.pkl")
print(f"Saved feature_names.csv    ({len(feat_cols)} features)")
print(f"Saved zero_variance_features.csv")

# ── 6. quick sanity check ──────────────────────────────────────────────────────
print("\n── Sanity check (standard scaler) ───────────────────────────────────")
X_check = np.load(OUT_DIR / "X_train_standard.npy")
print(f"  mean  (should ~0): {X_check.mean():.6f}")
print(f"  std   (should ~1): {X_check.std():.6f}")
print(f"  min  : {X_check.min():.3f}")
print(f"  max  : {X_check.max():.3f}")

print("\n── Sanity check (robust scaler) ─────────────────────────────────────")
X_check_r = np.load(OUT_DIR / "X_train_robust.npy")
print(f"  median (should ~0): {np.median(X_check_r):.6f}")
print(f"  IQR-normalised std : {X_check_r.std():.3f}")
print(f"  min  : {X_check_r.min():.3f}")
print(f"  max  : {X_check_r.max():.3f}")

# ── 7. per-class arrays for generative model training (benign excluded) ────────
print("\n── Per-class split (bot / brute force / xss) ────────────────────────")
minority_classes = {"bot": 1, "brute_force": 2, "xss": 3}

for scaler_name in ("standard", "robust"):
    X_full = np.load(OUT_DIR / f"X_train_{scaler_name}.npy")
    for slug, label_idx in minority_classes.items():
        mask  = (y_train == label_idx)
        X_cls = X_full[mask]
        path  = OUT_DIR / f"X_train_{scaler_name}_{slug}.npy"
        np.save(path, X_cls)
        print(f"  Saved {path.name}  {X_cls.shape}")

print("\nPreprocessing complete.")
print(f"Outputs in: {OUT_DIR.resolve()}")
