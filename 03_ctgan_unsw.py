"""
CTGAN baseline for UNSW-NB15 minority-class generation.
Mirrors 03_ctgan_baseline.py; standard scaler only.
"""

import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("data/processed_unsw")
OUT_DIR  = Path("outputs/ctgan_unsw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTS = {"backdoor": 1746, "shellcode": 1133, "worms": 130}
feat_cols = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()

CTGAN_PARAMS = dict(
    epochs=300, batch_size=500,
    generator_dim=(256,256), discriminator_dim=(256,256),
    embedding_dim=128, generator_lr=2e-4, discriminator_lr=2e-4,
    discriminator_steps=1, verbose=True, enable_gpu=False,
)

all_results = []

for cls, n in TARGET_COUNTS.items():
    print(f"\n{'='*55}")
    print(f"  CTGAN | class={cls}  n={n}")
    print(f"{'='*55}")

    X = np.load(DATA_DIR / f"X_train_standard_{cls}.npy").astype(np.float64)
    df_train = pd.DataFrame(X, columns=feat_cols)

    meta = Metadata.detect_from_dataframe(df_train)
    for col in feat_cols:
        meta.update_column(col, sdtype="numerical")

    model = CTGANSynthesizer(meta, **CTGAN_PARAMS)
    t0 = time.perf_counter()
    model.fit(df_train)
    train_time = time.perf_counter() - t0
    model.save(str(OUT_DIR / f"ctgan_model_standard_{cls}.pkl"))

    t0 = time.perf_counter()
    X_synth = model.sample(num_rows=n)[feat_cols].values.astype(np.float32)
    gen_time = time.perf_counter() - t0
    np.save(OUT_DIR / f"X_synth_ctgan_standard_{cls}.npy", X_synth)

    res = {"class": cls, "n_train": len(X), "n_generated": n,
           "train_time_sec": round(train_time, 2), "gen_time_sec": round(gen_time, 3)}
    all_results.append(res)
    print(f"  train={train_time:.1f}s  gen={gen_time:.3f}s  saved {X_synth.shape}")

    with open(OUT_DIR / "timing_log.json", "w") as f:
        json.dump(all_results, f, indent=2)

pd.DataFrame(all_results).to_csv(OUT_DIR / "timing_summary.csv", index=False)
print("\nCTGAN UNSW-NB15 complete.")
