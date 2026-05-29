# Generative Modeling via Drifting for Tabular NIDS Data

Applying the **Drifting Models** framework (arXiv:2602.04770) to the class-imbalance problem in Network Intrusion Detection Systems (NIDS), evaluated on CICIDS2017.

---

## 1. Background and Motivation

Network intrusion detection datasets suffer from severe class imbalance: benign traffic dominates by orders of magnitude, while rare but critical attack classes provide too few samples for reliable classifier training. In CICIDS2017, the three target minority classes account for fewer than 0.2% of all flows:

| Class | Train samples | Train % |
|---|---:|---:|
| benign | 1,589,610 | 80.32% |
| bot | 1,360 | 0.069% |
| brute force | 1,062 | 0.054% |
| xss | 436 | 0.022% |

Standard oversampling (SMOTE, random duplication) generates interpolated or repeated samples that do not reflect the true underlying distribution, limiting downstream classifier generalisation.
Generative models offer a principled alternative, but their effectiveness on tabular network flow data — with extreme imbalance, near-constant features, and heavy-tailed distributions — remains underexplored.

---

## 2. Proposed Approach: Drifting Models on Tabular NIDS Data

### 2.1 Drifting Field

Following arXiv:2602.04770, a vector field **V** is defined over the generated sample space:

```
V(x) = α · V⁺(x)  −  β · V⁻(x)
```

| Component | Role | Formula |
|---|---|---|
| **V⁺** (attraction) | Pulls generated samples toward real data | Kernel-weighted mean-shift: `Σ K(x, xᵢ)·xᵢ / Σ K(x, xᵢ)  − x` |
| **V⁻** (repulsion) | Pushes generated samples away from each other | `x  −  Σ K(x, yⱼ)·yⱼ / Σ K(x, yⱼ)` (diagonal masked) |

**Kernel:** RBF — `K(x, y) = exp(−‖x − y‖² / h)`

**Bandwidth:** 75th percentile of nonzero pairwise squared distances on training data (class-specific, more robust than median for tight clusters found in brute_force and xss).

### 2.2 Training Objective

```
L(θ) = E_ε [ ‖ fθ(ε)  −  sg( fθ(ε) + V(fθ(ε)) ) ‖² ]
```

where `sg(·)` denotes stop-gradient. The gradient pushes `fθ(ε)` in the direction of **V**, encouraging generated samples to settle at the intersection of real-data density modes (via **V⁺**) and diversity-preserving repulsion (via **V⁻**).

### 2.3 Generator Architecture

A lightweight MLP mapping noise to feature space:

```
z ~ N(0, I₇₀)  →  Linear(70→256) → SiLU → ×2 → Linear(256→70)
```

| Hyperparameter | Value |
|---|---|
| Hidden dim | 256 |
| Layers | 3 |
| Parameters | 167,750 |
| Epochs | 3,000 |
| Optimizer | Adam + CosineAnnealingLR (lr=1e-3) |
| Attraction α / Repulsion β | 1.0 / 1.0 |

---

## 3. Experimental Setup

### 3.1 Dataset — CICIDS2017

| Split | Rows | Features |
|---|---:|---:|
| Train | 1,979,229 | 70 (after removing 8 zero-variance features) |
| Test | 848,647 | 70 |

Preprocessing: StandardScaler fitted on train; test transformed with the same scaler. 8 zero-variance features (`bwd psh flags`, bulk-rate features) removed prior to scaling. Within the three target classes, an additional 20 features have zero within-class variance.

### 3.2 Baseline Generative Models

| Model | Library / Implementation | Epochs | Parameters | Train time (3 classes) |
|---|---|---:|---:|---:|
| **CTGAN** | SDV 1.36 (`CTGANSynthesizer`) | 300 | — | ~9 min |
| **TabDDPM** | Custom PyTorch (Gaussian DDPM) | 5,000 | 2,506,310 | ~76 min |
| **Drifting** | Custom PyTorch (this work) | 3,000 | 167,750 | ~3 min |

TabDDPM architecture: sinusoidal time embedding + 4 residual MLP blocks (hidden=512), T=1000 linear noise schedule, DDPM ancestral sampling.

### 3.3 Evaluation Protocol

**Fidelity** (distribution quality, computed once per model):
- Per-feature KS statistic: `ks_2samp(X_real[:, j], X_synth[:, j])` for each of 70 features → report mean, max, 90th percentile
- Correlation fidelity: Frobenius norm `‖corr_real − corr_synth‖_F` (active features only)

**Utility** (downstream classification, 5-seed repeated experiment):
- Random Forest classifier (`n_estimators=100`, `class_weight=balanced`)
- Train set: 50,000 benign (subsampled) + all minority class samples ± synthetic augmentation
- Test set: full held-out split (682,967 samples)
- Metric: F1 delta = F1(augmented) − F1(baseline), reported as mean ± std over 5 seeds

**Augmentation ratio experiment** (CTGAN and Drifting only):
- Ratios x1, x3, x5 relative to original class count
- Same 5-seed protocol as above

---

## 4. Results

### 4.1 Baseline F1 (No Augmentation)

| Class | Baseline F1 (mean ± std, 5 seeds) |
|---|---|
| bot | 0.668 ± 0.032 |
| brute_force | 0.721 ± 0.009 |
| xss | 0.429 ± 0.010 |

### 4.2 Fidelity

Lower is better. Computed on standard-scaled data (x1 synthetic samples).

| Model | Class | KS mean ↓ | KS max ↓ | KS p90 ↓ | Corr Frob ↓ |
|---|---|---:|---:|---:|---:|
| CTGAN | bot | **0.360** | 0.841 | 0.591 | 29.67 |
| CTGAN | brute_force | **0.312** | 0.791 | 0.666 | 30.49 |
| CTGAN | xss | **0.366** | 0.872 | 0.780 | 29.50 |
| TabDDPM | bot | 0.350 | **0.556** | **0.537** | **29.16** |
| TabDDPM | brute_force | 0.481 | 0.719 | 0.591 | **18.71** |
| TabDDPM | xss | 0.481 | **0.601** | 0.571 | **26.24** |
| Drifting | bot | 0.701 | 0.949 | 0.918 | 61.05 |
| Drifting | brute_force | 0.740 | 0.907 | 0.906 | 41.11 |
| Drifting | xss | 0.632 | 0.968 | 0.959 | 54.21 |

### 4.3 Utility — F1 Delta (x1 augmentation, mean ± std over 5 seeds)

| Model | bot | brute_force | xss |
|---|---|---|---|
| CTGAN | +0.0499 ± 0.0406 | +0.0210 ± 0.0088 | +0.0221 ± 0.0201 |
| TabDDPM | +0.0212 ± 0.0496 | +0.0176 ± 0.0069 | −0.0037 ± 0.0172 |
| **Drifting** | +0.0257 ± 0.0395 | +0.0144 ± 0.0074 | **+0.0295 ± 0.0222** |

Recall delta (x1):

| Model | bot | brute_force | xss |
|---|---|---|---|
| CTGAN | −0.003 ± 0.003 | +0.021 ± 0.015 | −0.053 ± 0.025 |
| TabDDPM | −0.003 ± 0.003 | +0.036 ± 0.015 | −0.093 ± 0.035 |
| **Drifting** | −0.002 ± 0.003 | +0.003 ± 0.009 | −0.014 ± 0.035 |

### 4.4 Augmentation Ratio Experiment — F1 Delta (CTGAN vs Drifting, mean ± std, 5 seeds)

| Class | Model | x1 | x3 | x5 |
|---|---|---|---|---|
| bot | CTGAN | +0.050 ± 0.041 | +0.078 ± 0.045 | +0.085 ± 0.036 |
| bot | **Drifting** | +0.026 ± 0.040 | +0.051 ± 0.031 | **+0.092 ± 0.036** |
| brute_force | **CTGAN** | **+0.021 ± 0.009** | **+0.023 ± 0.007** | **+0.026 ± 0.014** |
| brute_force | Drifting | +0.014 ± 0.007 | +0.011 ± 0.017 | +0.013 ± 0.011 |
| xss | CTGAN | +0.022 ± 0.020 | +0.039 ± 0.005 | +0.047 ± 0.018 |
| xss | **Drifting** | +0.030 ± 0.022 | **+0.066 ± 0.018** | **+0.067 ± 0.018** |

---

## 5. Key Findings

### Finding 1 — Fidelity-Utility Inversion

The model with the **worst distributional fidelity** (Drifting; KS mean 0.63–0.74) achieves the **strongest utility** for the hardest class (xss, +0.030 F1 at x1, +0.067 at x3/x5). Conversely, CTGAN produces samples closest to the real marginal distributions yet yields smaller improvements for xss. This suggests that for extreme minority classes, the ability to generate *diverse, boundary-informative* samples matters more than faithfully replicating the empirical feature distributions. The Drifting repulsion term **V⁻** actively discourages mode collapse, which likely drives this diversity.

### Finding 2 — Drifting Scales Better with More Samples (xss)

For xss (436 training samples, the most data-scarce class), Drifting's F1 delta increases sharply from +0.030 (x1) to +0.066 (x3) and plateaus at +0.067 (x5), indicating **saturation near x3**. CTGAN also improves with ratio but reaches only +0.047 at x5. At x5, Drifting achieves an absolute F1 of **0.496** vs. CTGAN's **0.476** and a baseline of **0.429**, representing a 15.8% relative improvement. The xss recall delta also turns positive for Drifting at x3/x5 (+0.019/+0.020) while CTGAN's xss recall remains negative at all ratios.

### Finding 3 — brute_force Resists Augmentation

Neither model substantially improves brute_force F1 beyond +0.026 at x5, and both show near-zero sensitivity to ratio (CTGAN: +0.021/+0.023/+0.026; Drifting: +0.014/+0.011/+0.013). Brute_force has 20 within-class zero-variance features after scaling, suggesting a highly constrained feature subspace. Synthetic samples from both models may faithfully reproduce this constrained geometry without adding information orthogonal to what the classifier already learns from 1,062 real samples.

---

## 6. Computational Cost

| Model | Train per class (std) | Total (3 classes) | Generation (x1) |
|---|---:|---:|---:|
| CTGAN | ~82 s | ~9 min | ~0.7 s |
| TabDDPM | ~774 s | ~39 min | ~29 s |
| **Drifting** | **~31 s** | **~3 min** | **<0.01 s** |

Drifting is **~25× faster** to train than TabDDPM and **~2.6× faster** than CTGAN, while achieving competitive or superior downstream utility.

---

## 7. Next Steps

- **Multi-dataset validation** on UNSW-NB15, NSL-KDD, and CICIoT2023 to assess generalisation across different traffic distributions and feature spaces.
- **Bandwidth sensitivity analysis** — the 75th-percentile heuristic works well for standard-scaled data but degrades under RobustScaler (robust/bot: h = 107,721, loss = 0.13). Learnable or cross-validated bandwidth selection is warranted.
- **Conditional generation** — extend the Drifting field to support class-conditional generation, enabling joint training across all minority classes rather than one model per class.
- **Evaluation beyond RF** — test augmented training with gradient-boosted trees (XGBoost/LightGBM) and deep NIDS classifiers to verify that utility gains are architecture-agnostic.
- **Comparison with SMOTE variants** (BorderlineSMOTE, ADASYN) as additional baselines.

---

## 8. Repository Structure

```
.
├── cicids2017/              # Raw CICIDS2017 CSV files
├── data/processed/          # Preprocessed NumPy arrays (scaled, class-split)
├── outputs/
│   ├── ctgan/               # CTGAN models + synthetic samples
│   ├── tabddpm/             # TabDDPM models + synthetic samples
│   ├── drifting/            # Drifting models + synthetic samples
│   ├── evaluation/          # Fidelity + utility results (5-seed)
│   └── ratio_experiment/    # x1/x3/x5 augmentation ratio comparison
├── 01_eda.py                # Class distribution + feature analysis
├── 02_preprocess.py         # Feature selection, scaling, class filtering
├── 03_ctgan_baseline.py     # CTGAN training + generation
├── 04_tabddpm_baseline.py   # TabDDPM (custom PyTorch) training + generation
├── 05_drifting.py           # Drifting Models training + generation
├── 06_evaluate.py           # Fidelity + utility pipeline (5-seed RF)
└── 07_ratio_experiment.py   # x3/x5 generation + ratio comparison
```

---

## References

- Kotelnikov, A., et al. "TabDDPM: Modelling Tabular Data with Diffusion Models." *ICML 2023*.
- Xu, L., et al. "Modeling Tabular Data using Conditional GAN." *NeurIPS 2019*.
- Sharafaldin, I., et al. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *ICISSP 2018*.
- arXiv:2602.04770 — "Generative Modeling via Drifting."
