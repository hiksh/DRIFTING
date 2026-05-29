"""
Evaluation pipeline for UNSW-NB15. Mirrors 06_evaluate.py.

Fidelity: KS statistic + Correlation Frobenius norm
Utility : 5-seed RF (baseline vs augmented), F1 delta mean +- std
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

DATA_DIR = Path("data/processed_unsw")
OUT_DIR  = Path("outputs/evaluation_unsw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTH_DIRS = {
    "CTGAN":    Path("outputs/ctgan_unsw"),
    "TabDDPM":  Path("outputs/tabddpm_unsw"),
    "Drifting": Path("outputs/drifting_unsw"),
}
MODELS       = list(SYNTH_DIRS.keys())
MINORITY_CLS = ["backdoor", "shellcode", "worms"]
LABEL_MAP    = {"backdoor": 1, "shellcode": 2, "worms": 3}
MINORITY_IDX = [1, 2, 3]
RF_SEEDS     = [42, 123, 456, 789, 1024]
RF_PARAMS    = dict(n_estimators=100, n_jobs=-1, class_weight="balanced",
                    max_features="sqrt")
COLORS       = {"CTGAN": "steelblue", "TabDDPM": "darkorange", "Drifting": "seagreen"}

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading data ...")
X_tr = np.load(DATA_DIR / "X_train_standard.npy")
y_tr = np.load(DATA_DIR / "y_train.npy")
X_te = np.load(DATA_DIR / "X_test_standard.npy")
y_te = np.load(DATA_DIR / "y_test.npy")

# UNSW-NB15: Normal=56k, no benign_cap needed
print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")
for lbl, name in [(0,"Normal"),(1,"Backdoor"),(2,"Shellcode"),(3,"Worms")]:
    print(f"    {name}: train={(y_tr==lbl).sum():,}  test={(y_te==lbl).sum():,}")

real_by_cls = {c: np.load(DATA_DIR/f"X_train_standard_{c}.npy") for c in MINORITY_CLS}
synth_by_model = {}
for m, d in SYNTH_DIRS.items():
    synth_by_model[m] = {}
    for c in MINORITY_CLS:
        hits = list(d.glob(f"X_synth_*standard_{c}.npy"))
        if hits: synth_by_model[m][c] = np.load(hits[0])


# ── helpers ────────────────────────────────────────────────────────────────────
def ks_stats(Xr, Xs):
    vals = [ks_2samp(Xr[:,j], Xs[:,j]).statistic for j in range(Xr.shape[1])]
    a = np.array(vals)
    return {"ks_mean": float(a.mean()), "ks_max": float(a.max()),
            "ks_p90": float(np.percentile(a, 90))}

def corr_frob(Xr, Xs):
    active = (Xr.std(0) > 1e-8) & (Xs.std(0) > 1e-8)
    if active.sum() < 2: return float("nan")
    cr = np.corrcoef(Xr[:,active].astype(np.float64).T)
    cs = np.corrcoef(Xs[:,active].astype(np.float64).T)
    np.nan_to_num(cr, copy=False, nan=0.); np.nan_to_num(cs, copy=False, nan=0.)
    return float(np.linalg.norm(cr - cs, "fro"))

def rf_eval(X_tr, y_tr, seed):
    rf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    _, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=MINORITY_IDX, zero_division=0)
    return {c: {"f1": float(f1[k]), "recall": float(rec[k])}
            for k, c in enumerate(MINORITY_CLS)}


# ── fidelity ───────────────────────────────────────────────────────────────────
print("\nComputing fidelity ...")
fid_rows = []
for m in MODELS:
    for c in MINORITY_CLS:
        if c not in synth_by_model[m]: continue
        ks = ks_stats(real_by_cls[c], synth_by_model[m][c])
        fr = corr_frob(real_by_cls[c], synth_by_model[m][c])
        fid_rows.append({"model": m, "class": c,
                         "ks_mean": round(ks["ks_mean"],4), "ks_max": round(ks["ks_max"],4),
                         "ks_p90": round(ks["ks_p90"],4),
                         "corr_frob": round(fr,4) if not np.isnan(fr) else None})
        print(f"  {m:<10} {c:<12} KS_mean={ks['ks_mean']:.4f}  Corr_Frob={fr:.4f}")

df_fid = pd.DataFrame(fid_rows)
df_fid.to_csv(OUT_DIR / "fidelity_summary.csv", index=False)


# ── utility: 5-seed RF ─────────────────────────────────────────────────────────
print(f"\nUtility: {len(RF_SEEDS)} seeds x (1 baseline + {len(MODELS)} models)")

raw_rows = []
for seed in RF_SEEDS:
    print(f"  seed={seed}", end="  ", flush=True)
    base_m = rf_eval(X_tr, y_tr, seed)
    print(f"base done", end="  ", flush=True)

    for m in MODELS:
        xs, ys = [X_tr], [y_tr]
        for c in MINORITY_CLS:
            if c in synth_by_model[m]:
                xs.append(synth_by_model[m][c])
                ys.append(np.full(len(synth_by_model[m][c]), LABEL_MAP[c], dtype=np.int64))
        X_aug = np.concatenate(xs); y_aug = np.concatenate(ys)
        aug_m = rf_eval(X_aug, y_aug, seed)
        print(f"{m} done", end="  ", flush=True)

        for c in MINORITY_CLS:
            raw_rows.append({"seed": seed, "model": m, "class": c,
                             "f1_baseline": base_m[c]["f1"],
                             "f1_augmented": aug_m[c]["f1"],
                             "f1_delta": aug_m[c]["f1"] - base_m[c]["f1"],
                             "recall_baseline": base_m[c]["recall"],
                             "recall_augmented": aug_m[c]["recall"],
                             "recall_delta": aug_m[c]["recall"] - base_m[c]["recall"]})
    print(flush=True)

df_raw = pd.DataFrame(raw_rows)
df_raw.to_csv(OUT_DIR / "utility_raw.csv", index=False)

agg = (df_raw.groupby(["model","class"])
       .agg(f1_base_mean=("f1_baseline","mean"), f1_base_std=("f1_baseline","std"),
            f1_aug_mean=("f1_augmented","mean"), f1_aug_std=("f1_augmented","std"),
            f1_delta_mean=("f1_delta","mean"), f1_delta_std=("f1_delta","std"),
            recall_delta_mean=("recall_delta","mean"), recall_delta_std=("recall_delta","std"))
       .round(4).reset_index())
agg.to_csv(OUT_DIR / "utility_summary.csv", index=False)

def fmt(r, mc, sc): return f"{r[mc]:+.4f} +- {r[sc]:.4f}"
disp = agg[["model","class"]].copy()
disp["f1_base"]      = agg.apply(lambda r: f"{r.f1_base_mean:.4f} +- {r.f1_base_std:.4f}", axis=1)
disp["f1_aug"]       = agg.apply(lambda r: f"{r.f1_aug_mean:.4f} +- {r.f1_aug_std:.4f}",  axis=1)
disp["f1_delta"]     = agg.apply(lambda r: fmt(r,"f1_delta_mean","f1_delta_std"),          axis=1)
disp["recall_delta"] = agg.apply(lambda r: fmt(r,"recall_delta_mean","recall_delta_std"),  axis=1)
disp.to_csv(OUT_DIR / "utility_display.csv", index=False)
df_fid.merge(agg, on=["model","class"], how="left").to_csv(
    OUT_DIR / "evaluation_combined.csv", index=False)

print("\n\n-- Fidelity -----------------------------------------------------------")
print(df_fid.to_string(index=False))
print("\n\n-- Utility (mean +- std, 5 seeds) ------------------------------------")
print(disp.to_string(index=False))


# ── plots ──────────────────────────────────────────────────────────────────────
x = np.arange(len(MINORITY_CLS)); w = 0.25

def get_vals(df, model, mc, sc=None):
    rows = [df[(df.model==model)&(df["class"]==c)] for c in MINORITY_CLS]
    m_   = [float(r[mc].iloc[0]) if len(r) else np.nan for r in rows]
    s_   = [float(r[sc].iloc[0]) if len(r) else np.nan for r in rows] if sc else None
    return m_, s_

for fname, mc, sc, title, ylabel in [
    ("fidelity_ks_mean.png",    "ks_mean",        None,               "Fidelity KS mean",        "KS mean (lower=better)"),
    ("fidelity_corr_frob.png",  "corr_frob",      None,               "Fidelity Corr Frobenius",  "Corr Frob (lower=better)"),
    ("utility_f1_delta.png",    "f1_delta_mean",  "f1_delta_std",     "Utility F1 delta",         "F1 delta (mean +- std)"),
    ("utility_recall_delta.png","recall_delta_mean","recall_delta_std","Utility Recall delta",     "Recall delta (mean +- std)"),
]:
    fig, ax = plt.subplots(figsize=(9,5))
    for i, model in enumerate(MODELS):
        src = df_fid if "fidelity" in fname else agg
        m_, s_ = get_vals(src, model, mc, sc)
        ax.bar(x+(i-1)*w, m_, w, label=model, color=COLORS[model], alpha=0.85)
        if s_:
            ax.errorbar(x+(i-1)*w, m_, yerr=s_, fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(MINORITY_CLS, fontsize=11)
    ax.set_ylabel(ylabel); ax.set_title(f"UNSW-NB15 -- {title}")
    ax.legend(); plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150); plt.close()

print(f"\nSaved to: {OUT_DIR.resolve()}")
print("Evaluation complete.")
