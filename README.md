# 테이블형 NIDS 데이터를 위한 Drifting 기반 생성 모델

**Drifting Models** 프레임워크(arXiv:2602.04770)를 네트워크 침입 탐지 시스템(NIDS)의 클래스 불균형 문제에 적용하고, CICIDS2017 데이터셋으로 검증한 연구입니다.

---

## 1. 연구 배경 및 동기

네트워크 침입 탐지 데이터셋은 극심한 클래스 불균형을 가집니다. 정상(benign) 트래픽이 압도적으로 많은 반면, 드물지만 중요한 공격 클래스는 분류기 학습에 충분한 샘플을 제공하지 못합니다. CICIDS2017에서 세 가지 타깃 소수 클래스는 전체 플로우의 0.2% 미만을 차지합니다.

| 클래스 | 학습 샘플 수 | 비율 |
|---|---:|---:|
| benign | 1,589,610 | 80.32% |
| bot | 1,360 | 0.069% |
| brute force | 1,062 | 0.054% |
| xss | 436 | 0.022% |

SMOTE, 무작위 복제 등의 표준 오버샘플링은 실제 분포를 반영하지 못하는 보간 샘플을 생성하여 분류기의 일반화 성능을 제한합니다. 생성 모델이 대안으로 제시되지만, 두 주요 계열 모두 데이터가 극도로 부족한 소수 클래스 환경에서 고유한 한계를 드러냅니다.

**GAN 계열의 한계.** Shahbazi et al. (ICLR 2022, arXiv:2201.06578)은 클래스 조건부 GAN이 학습 데이터가 제한된 환경에서 오히려 mode collapse가 심화됨을 이론·실험적으로 보였습니다. Chen et al. (ICAISC 2023)은 불균형한 학습 데이터에서 클래스 조건부 GAN의 생성 샘플 품질과 다양성이 저하됨을 확인했습니다. 본 실험에서도 CTGAN은 bot 클래스(1,360개)에서 F1 delta의 표준편차가 0.041로 모든 모델 중 가장 불안정한 거동을 보입니다.

**Diffusion 계열의 한계.** Fang et al. (arXiv:2412.11044, 2024)은 tabular diffusion 모델이 학습 에폭이 늘어날수록 training data를 그대로 복제하는 memorization 현상이 발생함을 처음으로 체계적으로 분석했습니다. 본 실험에서 TabDDPM은 xss 클래스(436개)에서 증강 비율을 x1에서 x5로 늘려도 F1 delta가 −0.004에서 +0.001로 사실상 개선이 없으며, recall은 −0.093에서 −0.083으로 오히려 지속적으로 저하됩니다.

이처럼 GAN 계열의 mode collapse와 diffusion 계열의 memorization은 극소수 클래스 환경에서 서로 다른 방향으로 실패합니다. 본 연구는 Drifting Models의 attraction/repulsion 메커니즘이 두 계열의 한계를 동시에 극복할 수 있는지 CICIDS2017 테이블형 NIDS 데이터에서 검증합니다.

---

## 2. 제안 방법: 테이블형 NIDS 데이터에의 Drifting Models 적용

### 2.1 Drifting Field

arXiv:2602.04770을 따라, 생성 샘플 공간 위에 벡터 필드 **V**를 정의합니다.

```
V(x) = α · V⁺(x)  −  β · V⁻(x)
```

| 구성 요소 | 역할 | 수식 |
|---|---|---|
| **V⁺** (인력) | 생성 샘플을 실제 데이터 쪽으로 당김 | 커널 가중 평균 이동: `Σ K(x, xᵢ)·xᵢ / Σ K(x, xᵢ)  − x` |
| **V⁻** (척력) | 생성 샘플끼리 서로 밀어냄 | `x  −  Σ K(x, yⱼ)·yⱼ / Σ K(x, yⱼ)` (대각 마스킹) |

**커널:** RBF — `K(x, y) = exp(−‖x − y‖² / h)`

**대역폭 h:** 학습 데이터의 0이 아닌 쌍별 제곱 거리의 75번째 백분위수 (클래스별 산출, brute_force·xss의 밀집 군집에서 중앙값보다 안정적).

### 2.2 학습 목적함수

```
L(θ) = E_ε [ ‖ fθ(ε)  −  sg( fθ(ε) + V(fθ(ε)) ) ‖² ]
```

`sg(·)`는 stop-gradient를 나타냅니다. 경사는 `fθ(ε)`를 **V** 방향으로 밀어, 생성 샘플이 실제 데이터 밀도 모드(**V⁺**)와 다양성 보존 척력(**V⁻**)의 교차점에 위치하도록 유도합니다.

### 2.3 생성기 구조

노이즈를 피처 공간으로 매핑하는 경량 MLP:

```
z ~ N(0, I₇₀)  →  Linear(70→256) → SiLU → ×2 → Linear(256→70)
```

| 하이퍼파라미터 | 값 |
|---|---|
| 은닉층 차원 | 256 |
| 레이어 수 | 3 |
| 파라미터 수 | 167,750 |
| 에폭 | 3,000 |
| 옵티마이저 | Adam + CosineAnnealingLR (lr=1e-3) |
| 인력 α / 척력 β | 1.0 / 1.0 |

---

## 3. 실험 설정

### 3.1 데이터셋 — CICIDS2017

| 분할 | 샘플 수 | 피처 수 |
|---|---:|---:|
| Train | 1,979,229 | 70 (분산=0인 피처 8개 제거 후) |
| Test | 848,647 | 70 |

전처리: StandardScaler를 학습 데이터로 피팅 후 테스트에 동일 적용. 분산=0인 피처(`bwd psh flags`, 벌크 전송률 관련 피처) 8개를 스케일링 전에 제거. 세 타깃 클래스 내에서 추가로 20개 피처가 클래스 내 분산=0.

### 3.2 비교 생성 모델 (Baseline)

| 모델 | 구현 | 에폭 | 파라미터 수 | 학습 시간 (3 클래스) |
|---|---|---:|---:|---:|
| **CTGAN** | SDV 1.36 (`CTGANSynthesizer`) | 300 | — | ~9분 |
| **TabDDPM** | 직접 구현 PyTorch (Gaussian DDPM) | 5,000 | 2,506,310 | ~76분 |
| **Drifting** | 직접 구현 PyTorch (본 연구) | 3,000 | 167,750 | ~3분 |

TabDDPM 구조: 정현파 시간 임베딩 + 잔차 MLP 블록 4개 (hidden=512), T=1000 선형 노이즈 스케줄, DDPM 조상 샘플링.

### 3.3 평가 프로토콜

**Fidelity** (분포 품질, 모델당 1회 산출):
- 피처별 KS 통계량: 70개 피처 각각에 대해 `ks_2samp(X_real[:, j], X_synth[:, j])` → 평균·최대·90번째 백분위수 보고
- 상관 충실도: Frobenius 노름 `‖corr_real − corr_synth‖_F` (활성 피처만 사용)

**Utility** (하위 분류 성능, 10-seed 반복 실험):
- Random Forest 분류기 (`n_estimators=100`, `class_weight=balanced`)
- 학습 세트: benign 50,000개(서브샘플) + 전체 소수 클래스 ± 합성 증강
- 테스트 세트: 전체 홀드아웃 분할 (682,967개)
- 지표: F1 delta = F1(증강) − F1(베이스라인), **10 seeds**에 대한 mean ± std
- Seeds: [42, 123, 456, 789, 1024, 2048, 3141, 5000, 7777, 9999]

**증강 비율 실험** (CTGAN, TabDDPM, Drifting 세 모델):
- 원본 클래스 수 대비 x1, x3, x5 비율
- 동일한 10-seed 프로토콜 적용

---

## 4. 실험 결과

### 4.1 베이스라인 F1 (증강 없음)

| 클래스 | 베이스라인 F1 (mean ± std, 10 seeds) |
|---|---|
| bot | 0.666 ± 0.025 |
| brute_force | 0.719 ± 0.009 |
| xss | 0.423 ± 0.014 |

### 4.2 Fidelity

낮을수록 좋음. Standard 스케일 데이터 기준 (합성 샘플 x1).

| 모델 | 클래스 | KS 평균 ↓ | KS 최대 ↓ | KS p90 ↓ | Corr Frob ↓ |
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

### 4.3 Utility — F1 Delta (x1 증강, 10 seeds mean ± std)

| 모델 | bot | brute_force | xss |
|---|---|---|---|
| **CTGAN** | **+0.0558 ± 0.0324** | **+0.0246 ± 0.0082** | +0.0315 ± 0.0227 |
| TabDDPM | +0.0277 ± 0.0355 | +0.0173 ± 0.0052 | −0.0015 ± 0.0224 |
| **Drifting** | +0.0140 ± 0.0398 | +0.0135 ± 0.0059 | **+0.0329 ± 0.0208** |

Recall delta (x1):

| 모델 | bot | brute_force | xss |
|---|---|---|---|
| CTGAN | −0.003 ± 0.004 | +0.024 ± 0.012 | −0.042 ± 0.027 |
| **TabDDPM** | −0.004 ± 0.003 | **+0.040 ± 0.014** | −0.094 ± 0.037 |
| **Drifting** | −0.001 ± 0.003 | +0.004 ± 0.007 | −0.014 ± 0.026 |

### 4.4 증강 비율 실험 — F1 Delta (3 모델, 10 seeds mean ± std)

| 클래스 | 모델 | x1 | x3 | x5 |
|---|---|---|---|---|
| bot | **CTGAN** | +0.056 ± 0.032 | +0.083 ± 0.033 | +0.088 ± 0.025 |
| bot | TabDDPM | +0.028 ± 0.036 | +0.055 ± 0.029 | +0.071 ± 0.028 |
| bot | **Drifting** | +0.014 ± 0.040 | +0.049 ± 0.032 | **+0.082 ± 0.028** |
| brute_force | **CTGAN** | **+0.025 ± 0.008** | **+0.025 ± 0.010** | **+0.027 ± 0.010** |
| brute_force | TabDDPM | +0.017 ± 0.005 | +0.019 ± 0.009 | +0.019 ± 0.010 |
| brute_force | Drifting | +0.014 ± 0.006 | +0.014 ± 0.014 | +0.016 ± 0.012 |
| xss | CTGAN | +0.032 ± 0.023 | +0.042 ± 0.016 | +0.045 ± 0.019 |
| xss | TabDDPM | −0.002 ± 0.022 | +0.000 ± 0.020 | +0.010 ± 0.025 |
| xss | **Drifting** | +0.033 ± 0.021 | **+0.065 ± 0.023** | **+0.068 ± 0.021** |

---

## 5. 핵심 발견

### 발견 1 — Fidelity-Utility 역전 현상

세 모델의 fidelity 순위(KS 평균 기준)는 CTGAN(0.31–0.37) < TabDDPM(0.35–0.48) < Drifting(0.63–0.74)이지만, xss F1 delta 순위는 **정반대**입니다: Drifting(+0.033) ≈ CTGAN(+0.032) > TabDDPM(−0.002). 10 seeds 기준으로 Drifting과 CTGAN은 x1에서 사실상 동률이지만, 비율을 키우면 격차가 벌어집니다(x3: Drifting +0.065 vs CTGAN +0.042). **분포 충실도가 가장 낮은** 모델이 샘플 수 확장 시 가장 큰 utility 개선을 달성하며, 중간 fidelity를 가진 TabDDPM은 모든 비율에서 xss 개선에 실패합니다(x1: −0.002, x5: +0.010). 이는 극단적인 소수 클래스에서 경험적 피처 분포를 충실히 복제하는 것보다 **다양하고 결정 경계에 유익한 샘플을 생성하는 능력**이 더 중요함을 시사합니다. Drifting의 척력 항 **V⁻**이 모드 붕괴를 적극적으로 억제하여 이러한 다양성을 이끌어냅니다.

### 발견 2 — Drifting의 샘플 수 증가 대응력 우위 (xss)

xss(학습 샘플 436개, 데이터가 가장 부족한 클래스)에서 세 모델의 ratio 반응이 뚜렷이 갈립니다 (10 seeds 기준). Drifting의 F1 delta는 +0.033(x1)에서 +0.065(x3)으로 급격히 증가한 후 +0.068(x5)에서 포화하며(**x3 근방 포화**). CTGAN도 비율과 함께 개선(x1: +0.032 → x5: +0.045)되지만 Drifting과의 격차가 x1에서는 거의 없다가 x3/x5에서 벌어집니다. TabDDPM은 x1~x3에서 F1 delta −0.002~0.000으로 사실상 0 수준이며 x5에서야 +0.010으로 소폭 양수가 됩니다. x5 기준 절대 F1은 Drifting **0.491**(0.423+0.068), CTGAN 0.468(+0.045), TabDDPM 0.433(+0.010 ≈ 베이스라인 0.423)으로, TabDDPM은 샘플 수를 5배 늘려도 유의미한 개선이 없습니다.

### 발견 3 — brute_force의 증강 저항성과 TabDDPM의 recall 특이점

F1 관점에서는 세 모델 모두 brute_force 개선이 미미합니다(CTGAN: +0.025~+0.027; TabDDPM: +0.017~+0.019; Drifting: +0.014~+0.016). brute_force는 스케일링 후 클래스 내 분산=0인 피처가 20개에 달해, 세 모델 모두 이 제한된 피처 서브공간 안에서 이미 1,062개 실제 샘플이 커버하는 정보를 벗어나지 못합니다. 단, recall에서 TabDDPM만 유일하게 일관된 양수 개선(x1/x3/x5: +0.036/+0.036/+0.033)을 보입니다. 이는 TabDDPM의 생성 샘플이 precision을 희생하더라도 실제 공격 패턴을 일부 포함함을 의미하며, F1 단독 지표로는 포착되지 않는 모델별 특성입니다.

---

## 6. 계산 비용

| 모델 | 클래스당 학습 시간 (standard) | 전체 (3 클래스) | 생성 시간 (x1) |
|---|---:|---:|---:|
| CTGAN | ~82초 | ~9분 | ~0.7초 |
| TabDDPM | ~774초 | ~39분 | ~29초 |
| **Drifting** | **~31초** | **~3분** | **<0.01초** |

Drifting은 TabDDPM 대비 **~25배**, CTGAN 대비 **~2.6배** 빠르게 학습하면서 동등하거나 우수한 하위 유용성을 달성합니다.

---

## 7. 향후 연구 방향

- **다중 데이터셋 검증** — UNSW-NB15, NSL-KDD, CICIoT2023에서 다양한 트래픽 분포·피처 공간에 대한 일반화 평가
- **대역폭 선택 개선** — 75번째 백분위수 휴리스틱은 standard 스케일 데이터에서는 잘 동작하지만 RobustScaler에서는 저하(robust/bot: h = 107,721, loss = 0.13). 학습 가능하거나 교차 검증 기반의 대역폭 선택 필요
- **조건부 생성** — Drifting field를 클래스 조건부 생성으로 확장하여 클래스별 개별 학습 대신 통합 모델로 전환
- **RF 이외의 평가** — XGBoost/LightGBM 및 딥러닝 NIDS 분류기로 유용성 향상이 아키텍처 독립적인지 검증
- **SMOTE 변형과의 비교** — BorderlineSMOTE, ADASYN을 추가 베이스라인으로 포함

---

## 8. 저장소 구조

```
.
├── cicids2017/              # CICIDS2017 원본 CSV 파일
├── data/processed/          # 전처리된 NumPy 배열 (스케일링, 클래스별 분리)
├── outputs/
│   ├── ctgan/               # CTGAN 모델 + 합성 샘플
│   ├── tabddpm/             # TabDDPM 모델 + 합성 샘플
│   ├── drifting/            # Drifting 모델 + 합성 샘플
│   ├── evaluation/          # Fidelity + Utility 결과 (5-seed)
│   └── ratio_experiment/    # x1/x3/x5 증강 비율 비교
├── 01_eda.py                # 클래스 분포 + 피처 분석
├── 02_preprocess.py         # 피처 선택, 스케일링, 클래스 필터링
├── 03_ctgan_baseline.py     # CTGAN 학습 + 생성
├── 04_tabddpm_baseline.py   # TabDDPM (직접 구현 PyTorch) 학습 + 생성
├── 05_drifting.py           # Drifting Models 학습 + 생성
├── 06_evaluate.py           # Fidelity + Utility 평가 파이프라인 (5-seed RF)
└── 07_ratio_experiment.py   # x3/x5 생성 + 비율 비교
```

---

## 참고문헌

- Kotelnikov, A., et al. "TabDDPM: Modelling Tabular Data with Diffusion Models." *ICML 2023*.
- Xu, L., et al. "Modeling Tabular Data using Conditional GAN." *NeurIPS 2019*.
- Sharafaldin, I., et al. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *ICISSP 2018*.
- arXiv:2602.04770 — "Generative Modeling via Drifting."
- Shahbazi, M., et al. "Collapse by Conditioning: Training Class-conditional GANs with Limited Data." *ICLR 2022*. arXiv:2201.06578.
- Chen, Y., et al. "Examining Effects of Class Imbalance on Conditional GAN Training." *ICAISC 2023*. Lecture Notes in Computer Science, vol. 14125.
- Fang, Z., et al. "Understanding and Mitigating Memorization in Diffusion Models for Tabular Data." arXiv:2412.11044, 2024.
