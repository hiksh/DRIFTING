"""
CTGAN baseline for NIDS minority-class generation.

For each (scaler, class) pair:
  - Load pre-scaled class-specific train array
  - Fit CTGANSynthesizer
  - Sample N rows (= original class count)
  - Save synthetic array + timing metadata
"""

import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
OUT_DIR  = Path("outputs/ctgan")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# original counts drive how many samples to generate
TARGET_COUNTS = {
    "bot":         1360,
    "brute_force": 1062,
    "xss":          436,
}
SCALERS = ["standard", "robust"]

CTGAN_PARAMS = dict(
    epochs             = 300,
    batch_size         = 500,
    generator_dim      = (256, 256),
    discriminator_dim  = (256, 256),
    embedding_dim      = 128,
    generator_lr       = 2e-4,
    discriminator_lr   = 2e-4,
    discriminator_steps= 1,
    verbose            = True,
    enable_gpu         = False,   # CPU only
)

# ── load feature names once ────────────────────────────────────────────────────
feat_cols = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()

# ── helpers ────────────────────────────────────────────────────────────────────
def make_metadata(df: pd.DataFrame) -> Metadata:
    """All columns are continuous numerical — mark them explicitly."""
    meta = Metadata.detect_from_dataframe(df)
    for col in df.columns:
        meta.update_column(col, sdtype="numerical")
    return meta


def run_ctgan(scaler: str, cls: str, n_samples: int) -> dict:
    tag = f"{scaler}_{cls}"
    print(f"\n{'='*60}")
    print(f"  CTGAN  |  scaler={scaler}  class={cls}  n={n_samples}")
    print(f"{'='*60}")

    # load scaled train data for this class
    X = np.load(DATA_DIR / f"X_train_{scaler}_{cls}.npy").astype(np.float64)
    df_train = pd.DataFrame(X, columns=feat_cols)

    meta = make_metadata(df_train)

    # ── train ──────────────────────────────────────────────────────────────────
    model = CTGANSynthesizer(meta, **CTGAN_PARAMS)

    t0_train = time.perf_counter()
    model.fit(df_train)
    train_time = time.perf_counter() - t0_train

    # save model
    model.save(str(OUT_DIR / f"ctgan_model_{tag}.pkl"))

    # ── generate ───────────────────────────────────────────────────────────────
    t0_gen = time.perf_counter()
    df_synth = model.sample(num_rows=n_samples)
    gen_time = time.perf_counter() - t0_gen

    X_synth = df_synth[feat_cols].values.astype(np.float32)
    np.save(OUT_DIR / f"X_synth_ctgan_{tag}.npy", X_synth)

    result = {
        "scaler":         scaler,
        "class":          cls,
        "n_train":        len(df_train),
        "n_generated":    n_samples,
        "train_time_sec": round(train_time, 2),
        "gen_time_sec":   round(gen_time, 4),
        "output_shape":   list(X_synth.shape),
    }

    print(f"  Train time : {train_time:.1f}s")
    print(f"  Gen time   : {gen_time:.3f}s")
    print(f"  Saved: X_synth_ctgan_{tag}.npy  {X_synth.shape}")

    return result


# ── main loop ──────────────────────────────────────────────────────────────────
all_results = []

for scaler in SCALERS:
    for cls, n in TARGET_COUNTS.items():
        res = run_ctgan(scaler, cls, n)
        all_results.append(res)

        # save timing log incrementally (safe against mid-run interruption)
        with open(OUT_DIR / "timing_log.json", "w") as f:
            json.dump(all_results, f, indent=2)

# ── summary table ──────────────────────────────────────────────────────────────
print("\n\n── Summary ──────────────────────────────────────────────────────────")
df_summary = pd.DataFrame(all_results)
print(df_summary[["scaler", "class", "n_train", "train_time_sec", "gen_time_sec"]].to_string(index=False))
df_summary.to_csv(OUT_DIR / "timing_summary.csv", index=False)
print(f"\nAll outputs in: {OUT_DIR.resolve()}")
