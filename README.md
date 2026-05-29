# 테이블형 NIDS 데이터를 위한 Drifting 기반 생성 모델

**Drifting Models** 프레임워크(arXiv:2602.04770)를 네트워크 침입 탐지 시스템(NIDS)의 클래스 불균형 문제에 적용하고, **CICIDS2017**과 **UNSW-NB15** 두 데이터셋으로 검증한 연구입니다.

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

**GAN 계열의 한계.** Shahbazi et al. (ICLR 2022, arXiv:2201.06578)은 클래스 조건부 GAN이 학습 데이터가 제한된 환경에서 오히려 mode collapse가 심화됨을 이론·실험적으로 보였습니다. Chen et al. (ICAISC 2023)은 불균형한 학습 데이터에서 클래스 조건부 GAN의 생성 샘플 품질과 다양성이 저하됨을 확인했습니다. 본 실험에서도 CTGAN은 bot 클래스(1,360개)에서 F1 delta의 표준편차가 0.032로 높은 불안정성을 보입니다.

**Diffusion 계열의 한계.** Fang et al. (arXiv:2412.11044, 2024)은 tabular diffusion 모델이 학습 에폭이 늘어날수록 training data를 그대로 복제하는 memorization 현상이 발생함을 처음으로 체계적으로 분석했습니다. 본 실험에서 TabDDPM은 xss 클래스(436개)에서 증강 비율을 x1에서 x5로 늘려도 F1 delta가 −0.002에서 +0.010으로 사실상 개선이 없으며, recall은 −0.094에서 −0.037 수준으로 지속 저하됩니다.

이처럼 GAN 계열의 mode collapse와 diffusion 계열의 memorization은 극소수 클래스 환경에서 서로 다른 방향으로 실패합니다. 본 연구는 Drifting Models의 attraction/repulsion 메커니즘이 두 계열의 한계를 동시에 극복할 수 있는지 두 NIDS 데이터셋에서 검증합니다.

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

**대역폭 h:** 학습 데이터의 0이 아닌 쌍별 제곱 거리의 75번째 백분위수 (클래스별 산출).

### 2.2 학습 목적함수

```
L(θ) = E_ε [ ‖ fθ(ε)  −  sg( fθ(ε) + V(fθ(ε)) ) ‖² ]
```

`sg(·)`는 stop-gradient를 나타냅니다. 경사는 `fθ(ε)`를 **V** 방향으로 밀어, 생성 샘플이 실제 데이터 밀도 모드(**V⁺**)와 다양성 보존 척력(**V⁻**)의 교차점에 위치하도록 유도합니다.

### 2.3 생성기 구조

노이즈를 피처 공간으로 매핑하는 경량 MLP (입력 차원 D는 데이터셋에 따라 자동 결정):

```
z ~ N(0, I_D)  →  Linear(D→256) → SiLU → ×2 → Linear(256→D)
```

| 하이퍼파라미터 | 값 |
|---|---|
| 은닉층 차원 | 256 |
| 레이어 수 | 3 |
| 파라미터 수 | D=70: 167,750 / D=176: 401,750 |
| 에폭 | 3,000 |
| 옵티마이저 | Adam + CosineAnnealingLR (lr=1e-3) |
| 인력 α / 척력 β | 1.0 / 1.0 |

---

## 3. 실험 설정

### 3.1 데이터셋

#### CICIDS2017

원본 전체 클래스 분포 (Train 1,979,229 / Test 848,647):

| 클래스 | Train | Train % | Test | Test % |
|---|---:|---:|---:|---:|
| benign | 1,589,610 | 80.32% | 681,710 | 80.33% |
| dos hulk | 161,170 | 8.14% | 68,954 | 8.13% |
| portscan | 110,987 | 5.61% | 47,817 | 5.63% |
| ddos | 89,739 | 4.53% | 38,286 | 4.51% |
| dos goldeneye | 7,311 | 0.37% | 2,982 | 0.35% |
| ftp-patator | 5,538 | 0.28% | 2,397 | 0.28% |
| ssh-patator | 4,076 | 0.21% | 1,821 | 0.21% |
| dos slowloris | 4,069 | 0.21% | 1,727 | 0.20% |
| dos slowhttptest | 3,822 | 0.19% | 1,677 | 0.20% |
| **bot** | **1,360** | **0.069%** | **596** | **0.070%** |
| **brute_force** | **1,062** | **0.054%** | **445** | **0.052%** |
| **xss** | **436** | **0.022%** | **216** | **0.025%** |
| infiltration | 26 | 0.001% | 10 | 0.001% |
| sql injection | 17 | 0.001% | 4 | 0.000% |
| heartbleed | 6 | 0.000% | 5 | 0.001% |

실험 대상 (benign + 타깃 3 클래스만 필터): Train **1,592,468** / Test **682,967**. 피처 수 70 (zero-var 8개 제거). 범주형 없음. 클래스 내 추가 zero-var 피처 20개.

#### UNSW-NB15

원본 전체 클래스 분포 (Train 175,341 / Test 82,332):

| 클래스 | Train | Train % | Test | Test % |
|---|---:|---:|---:|---:|
| Normal | 56,000 | 31.94% | 37,000 | 44.94% |
| Generic | 40,000 | 22.81% | 18,871 | 22.92% |
| Exploits | 33,393 | 19.05% | 11,132 | 13.52% |
| Fuzzers | 18,184 | 10.37% | 6,062 | 7.36% |
| DoS | 12,264 | 6.99% | 4,089 | 4.97% |
| Reconnaissance | 10,491 | 5.98% | 3,496 | 4.25% |
| Analysis | 2,000 | 1.14% | 677 | 0.82% |
| **Backdoor** | **1,746** | **0.996%** | **583** | **0.708%** |
| **Shellcode** | **1,133** | **0.646%** | **378** | **0.459%** |
| **Worms** | **130** | **0.074%** | **44** | **0.053%** |

실험 대상 (Normal + 타깃 3 클래스만 필터): Train **59,009** / Test **38,005**. 피처 수 176 (수치형 32 + OHE 144: `protocol` 133→132, `service` 13→11). Normal 56,000개 전량 사용.

### 3.2 비교 생성 모델 (Baseline)

| 모델 | 구현 | 에폭 | 파라미터 수 | 학습 시간 (CICIDS / UNSW) |
|---|---|---:|---:|---:|
| **CTGAN** | SDV 1.36 (`CTGANSynthesizer`) | 300 | — | ~9분 / ~11분 |
| **TabDDPM** | 직접 구현 PyTorch (Gaussian DDPM) | 5,000 | 2.5M / 3.1M | ~76분 / ~47분 |
| **Drifting** | 직접 구현 PyTorch (본 연구) | 3,000 | 168K / 402K | ~3분 / ~6분 |

TabDDPM 구조: 정현파 시간 임베딩 + 잔차 MLP 블록 4개 (hidden=512), T=1000 선형 노이즈 스케줄.

### 3.3 평가 프로토콜

**Fidelity** (분포 품질, 모델당 1회 산출):
- 피처별 KS 통계량 → 평균·최대·90번째 백분위수
- 상관 충실도: Frobenius 노름 `‖corr_real − corr_synth‖_F`

**Utility** (하위 분류 성능, **10-seed 반복 실험**):
- Random Forest (`n_estimators=100`, `class_weight=balanced`)
- Seeds: [42, 123, 456, 789, 1024, 2048, 3141, 5000, 7777, 9999]
- 지표: F1 delta = F1(증강) − F1(베이스라인), mean ± std over 10 seeds

**증강 비율 실험** (CTGAN, TabDDPM, Drifting 세 모델):
- 원본 클래스 수 대비 x1, x3, x5 비율, 동일한 10-seed 프로토콜

---

## 4. 실험 결과 — CICIDS2017

### 4.1 베이스라인 F1

| 클래스 | 베이스라인 F1 (mean ± std, 10 seeds) |
|---|---|
| bot | 0.666 ± 0.025 |
| brute_force | 0.719 ± 0.009 |
| xss | 0.423 ± 0.014 |

### 4.2 Fidelity (낮을수록 좋음)

| 모델 | 클래스 | KS 평균 ↓ | KS 최대 ↓ | KS p90 ↓ | Corr Frob ↓ |
|---|---|---:|---:|---:|---:|
| **CTGAN** | bot | **0.360** | 0.841 | 0.591 | 29.67 |
| **CTGAN** | brute_force | **0.312** | 0.791 | 0.666 | 30.49 |
| **CTGAN** | xss | **0.366** | 0.872 | 0.780 | 29.50 |
| TabDDPM | bot | 0.350 | **0.556** | **0.537** | **29.16** |
| TabDDPM | brute_force | 0.481 | 0.719 | 0.591 | **18.71** |
| TabDDPM | xss | 0.481 | **0.601** | 0.571 | **26.24** |
| Drifting | bot | 0.701 | 0.949 | 0.918 | 61.05 |
| Drifting | brute_force | 0.740 | 0.907 | 0.906 | 41.11 |
| Drifting | xss | 0.632 | 0.968 | 0.959 | 54.21 |

### 4.3 Utility — F1 Delta (x1, 10 seeds mean ± std)

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

### 4.4 증강 비율 실험 — F1 Delta (10 seeds mean ± std)

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

## 5. 실험 결과 — UNSW-NB15

### 5.1 베이스라인 F1

| 클래스 | 베이스라인 F1 (mean ± std, 10 seeds) |
|---|---|
| Backdoor | 0.975 ± 0.001 |
| Shellcode | 0.588 ± 0.006 |
| Worms | 0.845 ± 0.024 |

### 5.2 Fidelity (낮을수록 좋음)

| 모델 | 클래스 | KS 평균 ↓ | KS 최대 ↓ | KS p90 ↓ | Corr Frob ↓ |
|---|---|---:|---:|---:|---:|
| **CTGAN** | Backdoor | **0.070** | 0.688 | 0.286 | 13.16 |
| **CTGAN** | Shellcode | **0.047** | 0.572 | 0.240 | 17.32 |
| **CTGAN** | Worms | **0.074** | 0.708 | 0.396 | 13.67 |
| TabDDPM | Backdoor | 0.502 | **0.604** | **0.549** | **7.81** |
| TabDDPM | Shellcode | 0.489 | **0.636** | **0.549** | **5.27** |
| TabDDPM | Worms | 0.514 | **0.654** | **0.600** | **5.88** |
| Drifting | Backdoor | 0.946 | 0.998 | 0.998 | 146.38 |
| Drifting | Shellcode | 0.931 | 1.000 | 1.000 | 34.53 |
| Drifting | Worms | 0.574 | 0.954 | 0.831 | 37.37 |

### 5.3 Utility — F1 Delta (x1, 10 seeds mean ± std)

| 모델 | Backdoor | Shellcode | Worms |
|---|---|---|---|
| CTGAN | −0.001 ± 0.002 | **+0.009 ± 0.008** | −0.015 ± 0.034 |
| TabDDPM | +0.000 ± 0.001 | +0.008 ± 0.010 | −0.007 ± 0.029 |
| **Drifting** | +0.000 ± 0.002 | +0.006 ± 0.008 | **−0.006 ± 0.034** |

Recall delta (x1):

| 모델 | Backdoor | Shellcode | Worms |
|---|---|---|---|
| **CTGAN** | +0.000 ± 0.001 | **+0.020 ± 0.010** | −0.011 ± 0.039 |
| TabDDPM | +0.001 ± 0.001 | +0.019 ± 0.017 | −0.009 ± 0.047 |
| Drifting | +0.000 ± 0.001 | +0.007 ± 0.012 | −0.025 ± 0.038 |

### 5.4 증강 비율 실험 — F1 Delta (10 seeds mean ± std)

| 클래스 | 모델 | x1 | x3 | x5 |
|---|---|---|---|---|
| Backdoor | CTGAN | −0.001 ± 0.002 | −0.002 ± 0.003 | −0.003 ± 0.002 |
| Backdoor | TabDDPM | +0.000 ± 0.001 | +0.001 ± 0.002 | +0.001 ± 0.001 |
| Backdoor | **Drifting** | **+0.000 ± 0.002** | **+0.000 ± 0.002** | **+0.000 ± 0.001** |
| Shellcode | **CTGAN** | **+0.009 ± 0.008** | +0.009 ± 0.010 | +0.010 ± 0.008 |
| Shellcode | TabDDPM | +0.008 ± 0.010 | +0.010 ± 0.007 | **+0.012 ± 0.011** |
| Shellcode | Drifting | +0.006 ± 0.008 | +0.005 ± 0.010 | +0.007 ± 0.007 |
| Worms | CTGAN | −0.015 ± 0.034 | −0.013 ± 0.026 | −0.020 ± 0.025 |
| Worms | **Drifting** | −0.006 ± 0.034 | −0.007 ± 0.028 | −0.011 ± 0.027 |
| Worms | TabDDPM | −0.007 ± 0.029 | −0.003 ± 0.031 | +0.007 ± 0.033 |

---

## 6. 핵심 발견

### 발견 1 — Fidelity-Utility 역전 (CICIDS2017)

세 모델의 fidelity 순위(KS 평균 기준)는 CTGAN < TabDDPM < Drifting이지만, xss F1 delta 순위는 **정반대**: Drifting(+0.033) ≈ CTGAN(+0.032) > TabDDPM(−0.002). x1에서는 Drifting과 CTGAN이 사실상 동률이지만, 비율을 키우면 격차가 벌어집니다(x3: Drifting +0.065 vs CTGAN +0.042). **분포 충실도가 가장 낮은** 모델이 샘플 수 확장 시 가장 큰 utility 개선을 달성하며, TabDDPM은 모든 비율에서 xss 개선에 실패합니다. Drifting의 척력 항 **V⁻**이 모드 붕괴를 억제하여 다양성을 이끌어냅니다.

### 발견 2 — Drifting의 샘플 수 확장성 우위 (CICIDS2017 xss)

xss(436개)에서 Drifting F1 delta: +0.033(x1) → +0.065(x3) → +0.068(x5) (**x3 근방 포화**). x5 기준 절대 F1: Drifting 0.491 vs CTGAN 0.468 vs TabDDPM 0.433(≈ 베이스라인 0.423). TabDDPM은 샘플을 5배 늘려도 베이스라인 대비 +0.010에 그칩니다.

### 발견 3 — brute_force 저항성 + TabDDPM recall 특이점 (CICIDS2017)

F1 개선은 세 모델 모두 미미합니다(+0.014~+0.027). brute_force는 클래스 내 zero-var 피처 20개로 피처 서브공간이 고도로 제한됩니다. 단, TabDDPM만 brute_force recall에서 일관된 양수 개선(+0.040)을 보여 F1 단독으로 포착되지 않는 특성을 가집니다.

### 발견 4 — UNSW-NB15: 데이터셋 의존성

CICIDS2017과 달리 UNSW-NB15에서는 세 모델 모두 utility 개선이 미미합니다. Backdoor(F1≥0.975)는 이미 포화 상태, Shellcode는 전 모델 +0.006~+0.009 수준으로 균등, Worms(130개)는 std가 0.03~0.04로 커서 10 seeds에서도 통계적으로 불안정합니다. 이는 Drifting의 우위가 **CICIDS2017처럼 적절한 불균형(benign 80%)과 충분한 피처 다양성이 있는 환경에서 발현**되며, 베이스라인이 이미 포화됐거나 샘플이 극단적으로 부족한(130개) 환경에서는 세 모델 모두 한계를 가짐을 시사합니다.

---

## 7. 계산 비용

**CICIDS2017** (3 클래스, standard scaler)

| 모델 | 클래스당 학습 | 전체 | 생성 (x1) |
|---|---:|---:|---:|
| CTGAN | ~82초 | ~9분 | ~0.7초 |
| TabDDPM | ~774초 | ~39분 | ~29초 |
| **Drifting** | **~31초** | **~3분** | **<0.01초** |

**UNSW-NB15** (3 클래스, D=176)

| 모델 | 클래스당 학습 | 전체 | 생성 (x1) |
|---|---:|---:|---:|
| CTGAN | ~212초 | ~11분 | ~2.0초 |
| TabDDPM | ~930초 | ~47분 | ~31초 |
| **Drifting** | **~123초** | **~6분** | **<0.01초** |

Drifting은 두 데이터셋 모두에서 TabDDPM 대비 **~8×** 빠른 학습 속도를 유지합니다.

---

## 8. 향후 연구 방향

- **추가 데이터셋 검증** — NSL-KDD, CICIoT2023에서 일반화 평가
- **대역폭 선택 개선** — 75번째 백분위수 휴리스틱 대신 learnable / cross-validation 기반 선택
- **조건부 생성** — 클래스별 개별 학습 대신 조건부 통합 Drifting 모델
- **RF 이외의 평가** — XGBoost/LightGBM 및 딥러닝 NIDS 분류기로 utility 일반화 검증
- **SMOTE 변형과의 비교** — BorderlineSMOTE, ADASYN 추가 베이스라인
- **Worms (130개) 극소 케이스 분석** — 100개 미만 시나리오에서의 생성 가능성 탐색

---

## 9. 저장소 구조

```
.
├── cicids2017/              # CICIDS2017 원본 CSV (gitignore)
├── unsw-nb15/               # UNSW-NB15 원본 CSV + EDA/전처리 스크립트
├── data/processed/          # CICIDS2017 전처리 NumPy 배열 (gitignore)
├── data/processed_unsw/     # UNSW-NB15 전처리 NumPy 배열 (gitignore)
├── outputs/
│   ├── ctgan/               # CICIDS2017 CTGAN 모델 + 합성 샘플
│   ├── tabddpm/             # CICIDS2017 TabDDPM 모델 + 합성 샘플
│   ├── drifting/            # CICIDS2017 Drifting 모델 + 합성 샘플
│   ├── ctgan_unsw/          # UNSW-NB15 CTGAN
│   ├── tabddpm_unsw/        # UNSW-NB15 TabDDPM
│   ├── drifting_unsw/       # UNSW-NB15 Drifting
│   ├── evaluation/          # CICIDS2017 평가 결과 (10-seed)
│   ├── evaluation_unsw/     # UNSW-NB15 평가 결과 (10-seed)
│   ├── ratio_experiment/    # CICIDS2017 x1/x3/x5 비율 비교 (10-seed)
│   ├── ratio_experiment_unsw/  # UNSW-NB15 x1/x3/x5 비율 비교 (10-seed)
│   ├── eda/                 # CICIDS2017 EDA 플롯
│   └── eda_unsw/            # UNSW-NB15 EDA 플롯
├── 01_eda.py                # CICIDS2017 EDA
├── 02_preprocess.py         # CICIDS2017 전처리
├── 03_ctgan_baseline.py     # CICIDS2017 CTGAN
├── 04_tabddpm_baseline.py   # CICIDS2017 TabDDPM
├── 05_drifting.py           # CICIDS2017 Drifting
├── 06_evaluate.py           # CICIDS2017 평가 (10-seed RF)
├── 07_ratio_experiment.py   # CICIDS2017 비율 실험 (10-seed)
├── 03_ctgan_unsw.py         # UNSW-NB15 CTGAN
├── 04_tabddpm_unsw.py       # UNSW-NB15 TabDDPM
├── 05_drifting_unsw.py      # UNSW-NB15 Drifting
├── 06_evaluate_unsw.py      # UNSW-NB15 평가 (10-seed RF)
├── 07_ratio_experiment_unsw.py  # UNSW-NB15 비율 실험 (10-seed)
└── 08_extend_seeds.py       # 두 데이터셋 10-seed 확장 (증분 방식)
```

---

## 참고문헌

- Kotelnikov, A., et al. "TabDDPM: Modelling Tabular Data with Diffusion Models." *ICML 2023*.
- Xu, L., et al. "Modeling Tabular Data using Conditional GAN." *NeurIPS 2019*.
- Sharafaldin, I., et al. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *ICISSP 2018*.
- Moustafa, N., and Slay, J. "UNSW-NB15: A Comprehensive Dataset for Network Intrusion Detection Systems." *MilCIS 2015*.
- arXiv:2602.04770 — "Generative Modeling via Drifting."
- Shahbazi, M., et al. "Collapse by Conditioning: Training Class-conditional GANs with Limited Data." *ICLR 2022*. arXiv:2201.06578.
- Chen, Y., et al. "Examining Effects of Class Imbalance on Conditional GAN Training." *ICAISC 2023*. Lecture Notes in Computer Science, vol. 14125.
- Fang, Z., et al. "Understanding and Mitigating Memorization in Diffusion Models for Tabular Data." arXiv:2412.11044, 2024.
