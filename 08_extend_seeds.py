"""
Extend both datasets from 5 seeds to 10 seeds.

Strategy: load existing 5-seed raw results, run 5 new seeds only,
combine, re-aggregate. No re-running the original 5 seeds.

New seeds: [2048, 3141, 5000, 7777, 9999]
All results saved with _seed10 suffix; originals untouched.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

NEW_SEEDS = [2048, 3141, 5000, 7777, 9999]
RF_PARAMS = dict(n_estimators=100, n_jobs=-1, class_weight="balanced",
                 max_features="sqrt")


# ═══════════════════════════════════════════════════════════════════════════════
# Generic helpers
# ═══════════════════════════════════════════════════════════════════════════════

def rf_eval(X_tr, y_tr, X_te, y_te, minority_idx, minority_cls, seed):
    rf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    _, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=minority_idx, zero_division=0)
    return {c: {"f1": float(f1[k]), "recall": float(rec[k])}
            for k, c in enumerate(minority_cls)}


def synth_path(model, cls, ratio, dirs):
    d   = dirs[model]
    suf = "" if ratio == 1 else f"_x{ratio}"
    return d / f"X_synth_{model.lower()}_standard_{cls}{suf}.npy"


def run_new_seeds(cfg, new_seeds):
    """Run new seeds and return raw rows."""
    X_tr_full = np.load(cfg["data_dir"] / "X_train_standard.npy")
    y_tr_full = np.load(cfg["data_dir"] / "y_train.npy")
    X_te      = np.load(cfg["data_dir"] / "X_test_standard.npy")
    y_te      = np.load(cfg["data_dir"] / "y_test.npy")

    benign_idx   = np.where(y_tr_full == 0)[0]
    minority_idx_arr = np.where(y_tr_full != 0)[0]
    minority_idx = cfg["minority_idx"]
    minority_cls = cfg["minority_cls"]
    label_map    = cfg["label_map"]
    benign_cap   = cfg.get("benign_cap", None)

    def build_base(seed):
        rng = np.random.default_rng(seed)
        if benign_cap is not None:
            b = rng.choice(benign_idx, size=min(benign_cap, len(benign_idx)), replace=False)
        else:
            b = benign_idx
        idx = np.concatenate([b, minority_idx_arr])
        return X_tr_full[idx], y_tr_full[idx]

    raw_rows = []
    for seed in new_seeds:
        print(f"  seed={seed}", end="  ", flush=True)
        X_base, y_base = build_base(seed)
        base_m = rf_eval(X_base, y_base, X_te, y_te, minority_idx, minority_cls, seed)
        print("base done", end="  ", flush=True)

        for ratio in cfg["ratios"]:
            for model in cfg["models"]:
                xs, ys = [X_base], [y_base]
                for c in minority_cls:
                    p = synth_path(model, c, ratio, cfg["synth_dirs"])
                    X_s = np.load(p)
                    xs.append(X_s)
                    ys.append(np.full(len(X_s), label_map[c], dtype=np.int64))
                X_aug = np.concatenate(xs)
                y_aug = np.concatenate(ys)
                aug_m = rf_eval(X_aug, y_aug, X_te, y_te, minority_idx, minority_cls, seed)

                for c in minority_cls:
                    raw_rows.append({
                        "seed": seed, "model": model, "ratio": ratio, "class": c,
                        "n_synth": cfg["base_counts"][c] * ratio,
                        "f1_baseline":      base_m[c]["f1"],
                        "f1_augmented":     aug_m[c]["f1"],
                        "f1_delta":         aug_m[c]["f1"]     - base_m[c]["f1"],
                        "recall_baseline":  base_m[c]["recall"],
                        "recall_augmented": aug_m[c]["recall"],
                        "recall_delta":     aug_m[c]["recall"] - base_m[c]["recall"],
                    })
        print(flush=True)

    return pd.DataFrame(raw_rows)


def aggregate_and_save(df_combined, out_dir, label):
    """Aggregate + save display table and comparison table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(out_dir / f"raw_results_seed10.csv", index=False)

    agg = (df_combined
           .groupby(["model", "ratio", "class"])
           .agg(f1_delta_mean=("f1_delta", "mean"),
                f1_delta_std =("f1_delta", "std"),
                recall_delta_mean=("recall_delta","mean"),
                recall_delta_std =("recall_delta","std"),
                f1_aug_mean=("f1_augmented","mean"),
                f1_aug_std =("f1_augmented","std"))
           .round(4).reset_index())
    agg.to_csv(out_dir / "summary_aggregated_seed10.csv", index=False)

    # utility display (x1 only, mirrors 06_evaluate)
    df_x1  = df_combined[df_combined.ratio == 1].copy()
    agg_x1 = (df_x1
              .groupby(["model","class"])
              .agg(f1_base_mean=("f1_baseline","mean"), f1_base_std=("f1_baseline","std"),
                   f1_aug_mean=("f1_augmented","mean"), f1_aug_std=("f1_augmented","std"),
                   f1_delta_mean=("f1_delta","mean"),   f1_delta_std=("f1_delta","std"),
                   recall_delta_mean=("recall_delta","mean"), recall_delta_std=("recall_delta","std"))
              .round(4).reset_index())
    agg_x1.to_csv(out_dir / "utility_summary_seed10.csv", index=False)

    disp = agg_x1[["model","class"]].copy()
    disp["f1_base"]      = agg_x1.apply(lambda r: f"{r.f1_base_mean:.4f} +- {r.f1_base_std:.4f}", axis=1)
    disp["f1_aug"]       = agg_x1.apply(lambda r: f"{r.f1_aug_mean:.4f} +- {r.f1_aug_std:.4f}",  axis=1)
    disp["f1_delta"]     = agg_x1.apply(lambda r: f"{r.f1_delta_mean:+.4f} +- {r.f1_delta_std:.4f}", axis=1)
    disp["recall_delta"] = agg_x1.apply(lambda r: f"{r.recall_delta_mean:+.4f} +- {r.recall_delta_std:.4f}", axis=1)
    disp.to_csv(out_dir / "utility_display_seed10.csv", index=False)

    # ratio comparison table
    ratios = sorted(df_combined.ratio.unique())
    models = sorted(df_combined.model.unique())
    classes= df_combined["class"].unique()

    def fmt(m, s): return f"{m:+.4f} +- {s:.4f}"
    pivot_rows = []
    for c in classes:
        for model in models:
            row = {"class": c, "model": model}
            for ratio in ratios:
                sub = agg[(agg.model==model)&(agg.ratio==ratio)&(agg["class"]==c)]
                row[f"f1_delta_x{ratio}"] = (fmt(sub.f1_delta_mean.iloc[0], sub.f1_delta_std.iloc[0])
                                             if len(sub) else "-")
            pivot_rows.append(row)

    df_pivot = pd.DataFrame(pivot_rows)
    df_pivot.to_csv(out_dir / "comparison_table_seed10.csv", index=False)

    return agg_x1, df_pivot, agg


# ═══════════════════════════════════════════════════════════════════════════════
# CICIDS2017
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CICIDS2017 — extending to 10 seeds")
print("=" * 60)

cfg_cic = {
    "data_dir": Path("data/processed"),
    "minority_cls": ["bot", "brute_force", "xss"],
    "minority_idx": [1, 2, 3],
    "label_map":    {"bot": 1, "brute_force": 2, "xss": 3},
    "base_counts":  {"bot": 1360, "brute_force": 1062, "xss": 436},
    "models":  ["CTGAN", "TabDDPM", "Drifting"],
    "ratios":  [1, 3, 5],
    "benign_cap": 50_000,
    "synth_dirs": {
        "CTGAN":    Path("outputs/ctgan"),
        "TabDDPM":  Path("outputs/tabddpm"),
        "Drifting": Path("outputs/drifting"),
    },
}

df_cic_old  = pd.read_csv("outputs/ratio_experiment/raw_results.csv")
df_cic_new  = run_new_seeds(cfg_cic, NEW_SEEDS)
df_cic_all  = pd.concat([df_cic_old, df_cic_new], ignore_index=True)
print(f"\n  Combined: {len(df_cic_all)} rows  seeds={sorted(df_cic_all.seed.unique())}")

agg_cic_x1, pivot_cic, agg_cic = aggregate_and_save(
    df_cic_all, Path("outputs/ratio_experiment"), "CICIDS2017")

# also save utility_raw_seed10 in evaluation folder
df_cic_all[df_cic_all.ratio==1].to_csv(
    "outputs/evaluation/utility_raw_seed10.csv", index=False)
agg_cic_x1.to_csv("outputs/evaluation/utility_summary_seed10.csv", index=False)
disp_cic_path = Path("outputs/evaluation/utility_display_seed10.csv")
pd.read_csv("outputs/ratio_experiment/utility_display_seed10.csv").to_csv(
    disp_cic_path, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# UNSW-NB15
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("UNSW-NB15 — extending to 10 seeds")
print("=" * 60)

cfg_unsw = {
    "data_dir": Path("data/processed_unsw"),
    "minority_cls": ["backdoor", "shellcode", "worms"],
    "minority_idx": [1, 2, 3],
    "label_map":    {"backdoor": 1, "shellcode": 2, "worms": 3},
    "base_counts":  {"backdoor": 1746, "shellcode": 1133, "worms": 130},
    "models":  ["CTGAN", "TabDDPM", "Drifting"],
    "ratios":  [1, 3, 5],
    "benign_cap": None,   # no cap for UNSW (Normal=56k, manageable)
    "synth_dirs": {
        "CTGAN":    Path("outputs/ctgan_unsw"),
        "TabDDPM":  Path("outputs/tabddpm_unsw"),
        "Drifting": Path("outputs/drifting_unsw"),
    },
}

df_unsw_old = pd.read_csv("outputs/ratio_experiment_unsw/raw_results.csv")
df_unsw_new = run_new_seeds(cfg_unsw, NEW_SEEDS)
df_unsw_all = pd.concat([df_unsw_old, df_unsw_new], ignore_index=True)
print(f"\n  Combined: {len(df_unsw_all)} rows  seeds={sorted(df_unsw_all.seed.unique())}")

agg_unsw_x1, pivot_unsw, agg_unsw = aggregate_and_save(
    df_unsw_all, Path("outputs/ratio_experiment_unsw"), "UNSW-NB15")

df_unsw_all[df_unsw_all.ratio==1].to_csv(
    "outputs/evaluation_unsw/utility_raw_seed10.csv", index=False)
agg_unsw_x1.to_csv("outputs/evaluation_unsw/utility_summary_seed10.csv", index=False)
pd.read_csv("outputs/ratio_experiment_unsw/utility_display_seed10.csv").to_csv(
    "outputs/evaluation_unsw/utility_display_seed10.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Print final tables
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("RESULTS (10 seeds)")
print("=" * 60)

for label, agg_x1, pivot, path in [
    ("CICIDS2017", agg_cic_x1,  pivot_cic,  "outputs/ratio_experiment"),
    ("UNSW-NB15",  agg_unsw_x1, pivot_unsw, "outputs/ratio_experiment_unsw"),
]:
    print(f"\n-- {label} Baseline F1 --")
    base = (agg_x1.groupby("class")
            .apply(lambda g: f"{g.f1_base_mean.iloc[0]:.4f} +- {g.f1_base_std.iloc[0]:.4f}")
            .to_string())
    print(base)

    print(f"\n-- {label} Utility x1 (F1 delta) --")
    disp = pd.read_csv(f"{path}/utility_display_seed10.csv")
    print(disp[["model","class","f1_delta","recall_delta"]].to_string(index=False))

    print(f"\n-- {label} Ratio F1 delta --")
    comp = pd.read_csv(f"{path}/comparison_table_seed10.csv")
    cols = ["class","model"] + [c for c in comp.columns if c.startswith("f1_delta_x")]
    print(comp[cols].to_string(index=False))

print("\nDone. All _seed10 files saved.")
