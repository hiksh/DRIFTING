"""
Preprocessing pipeline for UNSW-NB15.

Steps:
  1. Load train/test CSVs
  2. Filter to target classes: Normal, Backdoor, Shellcode, Worms
  3. One-hot encode protocol (133 unique) and service (13 unique)
  4. StandardScaler on 32 numerical features
  5. Concatenate [scaled_num | OHE_protocol | OHE_service]
  6. Drop any zero-variance columns in the encoded space
  7. Save to data/processed_unsw/
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

DATA_DIR = Path("unsw-nb15")
OUT_DIR  = Path("data/processed_unsw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS      = ["attack_name", "attack_flag", "attack_step"]
CAT_COLS       = ["protocol", "service"]
TARGET_CLASSES = ["Normal", "Backdoor", "Shellcode", "Worms"]
LABEL_MAP      = {"Normal": 0, "Backdoor": 1, "Shellcode": 2, "Worms": 3}

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading CSVs ...")
train_raw = pd.read_csv(DATA_DIR / "training-flow.csv", low_memory=False)
test_raw  = pd.read_csv(DATA_DIR / "test-flow.csv",  low_memory=False)

print(f"  Raw train: {len(train_raw):,}  |  Raw test: {len(test_raw):,}")

# ── filter ─────────────────────────────────────────────────────────────────────
train = train_raw[train_raw["attack_name"].isin(TARGET_CLASSES)].copy()
test  = test_raw[test_raw["attack_name"].isin(TARGET_CLASSES)].copy()

print(f"\nAfter filtering to {TARGET_CLASSES}:")
print(f"  Train: {len(train):,}  |  Test: {len(test):,}")
print("\n  Train class counts:")
for cls, cnt in train["attack_name"].value_counts().items():
    print(f"    {cls:<12} {cnt:>6,}  ({cnt/len(train)*100:.2f}%)")
print("\n  Test class counts:")
for cls, cnt in test["attack_name"].value_counts().items():
    print(f"    {cls:<12} {cnt:>6,}  ({cnt/len(test)*100:.2f}%)")

# ── labels ─────────────────────────────────────────────────────────────────────
y_train = np.array([LABEL_MAP[v] for v in train["attack_name"]], dtype=np.int64)
y_test  = np.array([LABEL_MAP[v] for v in test["attack_name"]],  dtype=np.int64)

le = LabelEncoder()
le.classes_ = np.array(["Normal", "Backdoor", "Shellcode", "Worms"])
print(f"\nLabel map: {LABEL_MAP}")

# ── numerical features ─────────────────────────────────────────────────────────
num_cols = [c for c in train.columns if c not in META_COLS and c not in CAT_COLS]
print(f"\nNumerical features: {len(num_cols)}")

scaler = StandardScaler()
X_num_train = scaler.fit_transform(train[num_cols].values.astype(np.float64))
X_num_test  = scaler.transform(test[num_cols].values.astype(np.float64))

# ── one-hot encode categorical features ───────────────────────────────────────
# fit on train, handle unknowns in test with ignore
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
X_cat_train = ohe.fit_transform(train[CAT_COLS])
X_cat_test  = ohe.transform(test[CAT_COLS])

ohe_feat_names = ohe.get_feature_names_out(CAT_COLS).tolist()
print(f"OHE features: {len(ohe_feat_names)}  "
      f"(protocol: {sum(1 for f in ohe_feat_names if f.startswith('protocol'))}  "
      f"service: {sum(1 for f in ohe_feat_names if f.startswith('service'))})")

# ── concatenate: [scaled_num | OHE] ───────────────────────────────────────────
X_train = np.concatenate([X_num_train.astype(np.float32), X_cat_train], axis=1)
X_test  = np.concatenate([X_num_test.astype(np.float32),  X_cat_test],  axis=1)

all_feat_names = num_cols + ohe_feat_names
print(f"\nCombined feature dim before zero-var drop: {X_train.shape[1]}")

# ── drop zero-variance columns (in encoded train space) ───────────────────────
stds = X_train.std(axis=0)
active = stds > 1e-8
n_dropped = (~active).sum()
print(f"Zero-variance columns dropped: {n_dropped}")
if n_dropped:
    dropped = [all_feat_names[i] for i in np.where(~active)[0]]
    print(f"  {dropped}")

X_train = X_train[:, active].astype(np.float32)
X_test  = X_test[:,  active].astype(np.float32)
feat_names_final = [f for f, a in zip(all_feat_names, active) if a]

D = X_train.shape[1]
print(f"Final feature dim D = {D}")

# ── per-class minority arrays ──────────────────────────────────────────────────
minority_cls = {"Backdoor": 1, "Shellcode": 2, "Worms": 3}
print("\nPer-class minority arrays:")
for cls, idx in minority_cls.items():
    mask = (y_train == idx)
    X_cls = X_train[mask]
    slug  = cls.lower()
    np.save(OUT_DIR / f"X_train_standard_{slug}.npy", X_cls)
    print(f"  X_train_standard_{slug}.npy  {X_cls.shape}")

# ── save full arrays ───────────────────────────────────────────────────────────
np.save(OUT_DIR / "X_train_standard.npy", X_train)
np.save(OUT_DIR / "X_test_standard.npy",  X_test)
np.save(OUT_DIR / "y_train.npy", y_train)
np.save(OUT_DIR / "y_test.npy",  y_test)

with open(OUT_DIR / "scaler_standard.pkl", "wb") as f: pickle.dump(scaler, f)
with open(OUT_DIR / "ohe.pkl",             "wb") as f: pickle.dump(ohe, f)
with open(OUT_DIR / "label_encoder.pkl",   "wb") as f: pickle.dump(le, f)

pd.Series(feat_names_final, name="feature").to_csv(
    OUT_DIR / "feature_names.csv", index=False)
pd.DataFrame({"dropped": [f for f, a in zip(all_feat_names, active) if not a]}).to_csv(
    OUT_DIR / "zero_variance_features.csv", index=False)
pd.Series(num_cols, name="feature").to_csv(
    OUT_DIR / "numerical_cols.csv", index=False)

# ── sanity check ───────────────────────────────────────────────────────────────
print(f"\nSanity (scaled num block, should be ~0/1): "
      f"mean={X_train[:, :len(num_cols)].mean():.4f}  "
      f"std={X_train[:, :len(num_cols)].std():.4f}")
print(f"Shapes: X_train={X_train.shape}  X_test={X_test.shape}")
print(f"\nSaved to {OUT_DIR.resolve()}")
print("Preprocessing complete.")
