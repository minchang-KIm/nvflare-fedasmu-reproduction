# FedASMU 논문의 실험 조건 전수 정리 (공개 / 비공개 / 우리 값)

## 출처와 표기

**출처**
- [AAAI] AAAI-24 본문 (9쪽)
- [arXiv] arXiv 2312.05770v1 (부록 포함 18쪽). 본문의 "Please see details in Appendix"가 가리키는 부록은 arXiv 판에만 있습니다.

**두 판에서 확인한 범위**
- 두 판 모두 끝까지 읽고 다음 키워드를 전부 검색했습니다: straggler, buffer, tune, smooth, batch, epoch, Dirichlet, bandwidth, µ, K, seed, runs.
- 실험 조건이 적힌 곳은 다음뿐입니다.
  - [AAAI] 본문 "Experimental Setup", Table 1–2
  - [arXiv] 부록 "Experiment details", Table 3–6, "Visualization of Experimental Results", "Communication Overhead Analysis", "Hyper-parameter Fine-tuning", "Comparison with Other Baselines", "Ablation Study for Request Time Slot Selection"

**공개 칸 표기**
- ✅ 논문에 값이 있음
- ⚠️ 언급은 있으나 값 · 형태가 없음, 또는 논문 안에서 서로 모순됨
- ❌ 언급 없음

## 1. 시스템 · 실험 환경

| 항목 | 논문 내용 (원문 위치) | 공개 | 우리 값 | 근거 |
|---|---|---|---|---|
| 기기 수 m | 100 (본문 Setup, 표 5·6) | ✅ | 100 | 논문 |
| 동시 학습 수 m′ | 10 = "maximum parallelism 10 % of devices" (부록, 표 5·6) | ✅ | 10 | 논문 |
| 트리거 방식 | 표 5·6: 주기 T = 10. 부록 본문: "triggers one idle task every 5 seconds" | ⚠️ 표와 본문이 다름 (10 s에 m′대 vs 5 s마다 1대) | 10 s마다 빈 슬롯을 채움 (최대 10대) | 표 값 사용 |
| 최대 staleness τ | 99 (표 5·6) | ✅ | 99 | 논문 |
| 학습 예산 | 표 5·6: "T = 500 (maximum number of global rounds)", 모든 워크로드. 부록 본문: "500 epochs for synchronous, 5000 for asynchronous" | ⚠️ 단위("epoch")가 모호하고, 표에서는 FedASMU도 500 | 비동기 5,000 업데이트, FedAvg 500 라운드 | 본문 해석 |
| 기기 속도 이질성 (straggler) | 부록 Experiment details: "slowest device five times longer than the fastest, sampled independently and randomly within this range". 같은 부록의 Figure 7 설명: "augmenting the difference (from 110 times faster to 440 times faster) between the fastest device and the lowest device" | ⚠️ 5배와 110배 · 440배가 함께 적혀 있음. 분포 형태(균등 여부)도 없음 | 기본 5배 균등. 진단 실험으로 110배 | 아래 3절 |
| 통신 대역폭 | "normal bandwidth"(값 없음), "modest" = 50배 · 100배 낮음 (부록 Fig. 7, Communication Overhead) | ⚠️ 기준 값 없음 | 기기별 1–5 MB/s | 가정 |
| 시간 모델 | 시뮬레이션 (V100 44장에서 FL 환경 시뮬레이션) | ⚠️ 에폭당 시간 · 통신 시간 계산법 없음 | 에폭당 10 s (작은 모델) / 30 s (큰 모델) × 기기별 1–5배 | 가정 |
| 데이터 분할 | 비IID, Dirichlet (Li et al. 2021 인용) | ⚠️ α 값 없음 | α = 0.5 | 인용한 NIID-Bench의 기본값 |
| 로컬 최적화 | SGD, Formula 4: $w_{o,l}=w_{o,l-1}-\eta_i\nabla F_i(w_{o,l-1},\zeta_{l-1})$, $\zeta$ = mini-batch, $l$ = "local epoch" | ⚠️ 한 "epoch"가 미니배치 1스텝인지 데이터 한 바퀴인지 모호 | 데이터 한 바퀴 = 1 epoch | 해석 |
| 최대 로컬 epoch L | 기호만 있음 (표 4) | ❌ | 5 | 가정 |
| 배치 크기 | — | ❌ | 50 | 가정 |
| 로컬 학습률 ηᵢ | 표 5·6 (워크로드별) | ✅ (FedASMU 기준인지, 모든 기법 공통인지는 불명) | 표 값, 모든 기법 공통 | 논문 |
| 학습률 감쇠 | "We exploit a learning rate decay" (부록) | ⚠️ 형태 · 계수 없음 | 업데이트마다 ×0.9998 | 가정 |
| 모멘텀 · weight decay | — | ❌ | 없음 | 가정 |
| 평가 | "convergence accuracy of the global model" (표 1·2) | ⚠️ 정의(마지막 값 · 최고값 · 평균), 평가 주기, 테스트셋 구성 없음 | 50 업데이트마다, 최종 = 마지막 5회 평균 | 가정 |
| 목표 정확도 | 표 1·2 캡션 (예: LeNet / CIFAR-10 0.30) | ✅ | 같음 | 논문 |
| 반복 횟수 · seed | — | ❌ | seed 1개 (0) | — |
| 하이퍼파라미터 튜닝 | "We fine-tune the hyper-parameters for each approach and report the best one in the paper" (부록) | ⚠️ 튜닝했다는 사실만 있음. 대상 · 범위 · 검증 데이터(테스트셋 사용 여부) 없음 | 튜닝 안 함 (테스트셋 튜닝 금지) | 프로젝트 규칙 |
| 곡선 평활화 | — | ❌ 언급 없음 | 평활화 안 함 | — |

## 2. 모델

| 모델 | 논문 내용 | 공개 | 우리 구현 |
|---|---|---|---|
| CNN | 표 3: conv(64, k = 2) → act → conv(32, k = 2) → act → flatten → dense(10) → softmax | ✅ (활성화 종류, 풀링 유무는 없음) | 표 3 그대로 (ReLU) |
| LeNet5 | 이름 + LeCun et al. 1989 인용 | ⚠️ 구조 없음 | conv6-5 / conv16-5, ReLU + max-pool, FC 120-84 (현대식 변형) |
| ResNet20 | 이름 + He et al. 2016 | ⚠️ | CIFAR식 ResNet-20, BatchNorm 포함 |
| AlexNet | 이름 + Krizhevsky 2012 | ⚠️ (32×32 입력용 변형 방법 없음) | CIFAR용 5 conv + FC 4096×2, dropout |
| VGG-11 | 이름 + Simonyan 2015 | ⚠️ | 설정 A, BN 없음, FC 512-512 |
| TextCNN | 이름 + Zhou et al. 2021b | ⚠️ (임베딩 · 필터 · 어휘 수 없음) | 임베딩 100, 필터 3/4/5 × 100, 어휘 2만 |
| 데이터 전처리 · 증강 | — | ❌ | 정규화만, 증강 없음 |

## 3. FedASMU 고유 파라미터

| 파라미터 | 논문 | 공개 | 우리 값 |
|---|---|---|---|
| ηλ, ησ, ηι (서버 계수 학습률) | 표 5·6 | ✅ | 표 값 |
| ηγ, ηυ (기기 계수 학습률) | 표 5·6 | ✅ | 표 값 |
| η_RL (LSTM 학습률) | 표 5·6 (모두 0.001) | ✅ | 0.001 |
| μα, μβ (α, β 식의 계수) | "hyper-parameter" (Eq. 2, Formula 8) | ❌ 값 없음 | 10, 10 |
| λ₀, σ₀, ι₀ (서버 계수 초기값) | — | ❌ | 1, 0.5, 0.01 |
| γ₀, υ₀ (기기 계수 초기값) | — | ❌ | 1, 0.5 |
| α_min (수렴 분석의 하한) | 부록 수렴 분석에 "αmin ≤ α" 가정 | ⚠️ 실험에서 썼는지, 값이 무엇인지 없음 | 하한 없음 (ξ ≥ 0만 보장) |
| 계수 범위 제한 (clip) | — | ❌ | λ, σ ≥ 0 |
| ρ (보상 기준값 b의 이동평균 계수) | "ρ being a hyper-parameter" | ❌ | 0.1 |
| Q-러닝 φ, ψ, ε | 기호와 ε-greedy만 | ❌ | 0.1, 0.9, 0.1 |
| LSTM 구조 · 입력 | "LSTM + FC" | ❌ | 은닉 32, 입력 = (기기 속도, 데이터 크기) |
| LSTM 사전학습 | "pre-trained with some historical data" | ⚠️ 데이터 · 방법 없음 | 같은 워크로드의 이전 FedASMU 실행 기록 |
| 부록 미분식 | 부록 A | ⚠️ σ 미분식이 ln σ로 인쇄됨 (Eq. 2를 미분하면 ln τ). 의사기울기 부호도 모호 | ln τ 사용, SGD와 맞는 부호 (인쇄된 부호는 진단 실험으로 비교) |

## 4. 비교 기법 파라미터

| 기법 | 필요한 주요 파라미터 | 논문 공개 | 우리 |
|---|---|---|---|
| FedBuff | 버퍼 크기 K, 서버 학습률, staleness 가중 | ❌ 하나도 없음 ("fine-tune … report the best"만 있음) | NVFLARE `BuffModelManager`, K = 10 (FedBuff 논문 권장), 서버 학습률 1 |
| FedAvg | 라운드당 기기 수, 로컬 epoch, 집계 가중치 | ⚠️ "500 epochs"만 있음 | 라운드당 10대, 데이터 크기 가중 평균 |
| FedProx | μ | ❌ | 미구현 |
| MOON | μ, 온도 τ | ❌ | 미구현 |
| FedDyn | α | ❌ | 미구현 |
| FedAsync | α, staleness 함수(poly a) | ❌ | 미구현 (FedASMU-0 ≈ FedAsync + staleness 상한) |
| PORT, ASO-Fed, FedSA | 각 기법의 고유 파라미터 | ❌ | 미구현 |

## 5. 논문 안의 불일치 (확인된 것)

1. **기기 속도 차이:** 부록 본문은 5배라고 하고, Figure 7 설명은 "110배 → 440배"라고 합니다.
   - Figure 7(a) "Moderately Heterogeneous" 막대의 정확도는 표 2의 LeNet / FMNIST 값과 같습니다: FedASMU 0.84 vs 0.829, FedAvg 0.70 vs 0.706, FedBuff 0.76 vs 0.767.
   - 막대의 목표 도달 시간은 Figure 6(c) 곡선에서 잰 값의 약 0.5–0.65배로, 같은 실행은 아닙니다.
2. **Figure 7 패널 순서:** 본문은 (a) = 대역폭, (b) = 이질성이라고 하지만, 그림 캡션은 (a) = 이질성, (b) = 대역폭입니다.
3. **트리거:** 표의 T = 10과 부록 본문의 "5초마다 1개"가 다릅니다.
4. **학습 예산:** 표의 "T = 500 global rounds"(모든 기법 공통처럼 읽힘)와 부록 본문의 "동기 500 / 비동기 5000"이 다릅니다.
5. **VGG / CIFAR-100 FedBuff 목표 도달 시간:** 표 2는 167,053 s, Figure 6(b) 곡선은 약 249,000 s입니다.
6. **Q-러닝 stay 행동:** 본문은 stay를 "l* + 1"로 적었는데, add와 같아서 오타로 보입니다.

## 6. 가설 점검 (2026-11-12)

### 6-1. "straggler 때문에 FedAvg가 느리다"
- 논문 Figure 7 설명의 110배를 그대로 넣어 LeNet / CIFAR-10을 다시 돌렸습니다 (`diagnose` profile, 기기 속도 U[1, 110], 나머지는 동일).

| 설정 | FedASMU 최종 | FedBuff 최종 | FedAvg 최종 | FedAvg / FedASMU 시간 (목표 0.30) | (목표 0.40) |
|---|---|---|---|---|---|
| 논문 (곡선에서 추출) | 0.483 | 0.362 | 0.428 | 14.1 | 12.3 |
| 우리, 속도 차이 5배 | 0.448 | 0.540 | 0.504 | 1.16 | 1.08 |
| 우리, 속도 차이 110배 | 0.466 | 0.535 | 0.508 | 1.38 | 1.25 |

- straggler를 넣으면 FedAvg 전체 시간은 비동기 대비 1.55배에서 1.82배로 늘어납니다.
- 그러나 목표 도달 시간 비율은 1.2–1.4배에 그쳐 논문의 12–14배와 거리가 멉니다.
- 이유: 기기 속도가 균등 분포면 "10대 중 가장 느린 기기 / 평균"이 속도 차이를 아무리 키워도 1.82배를 넘지 못합니다. 로그 균등 분포에서도 110배일 때 3.3배, 440배일 때 3.9배입니다.
- 결론: 논문에 적힌 straggler 설정만으로는 논문의 FedAvg 곡선이 나오지 않습니다. 기기 속도의 분포 형태 또는 동기 기법의 다른 설정(라운드당 기기 수, 로컬 학습량 등)이 논문과 달랐을 가능성이 큽니다. 이 설정들은 공개되지 않았습니다.

### 6-2. "튜닝과 곡선 평활화를 했다"
- **튜닝:** 논문에 명시돼 있습니다 ("fine-tune the hyper-parameters for each approach and report the best"). 다만 어떤 데이터로 골랐는지는 없습니다.
- **평활화:** 논문에 언급이 없습니다. 근거는 곡선 모양뿐입니다.
  - 논문 비동기 곡선의 거칠기는 0.0001–0.0006입니다. 우리 곡선은 0.002–0.03입니다.
  - 논문 FedAvg 곡선은 0.0003–0.003으로, 모델이 10배 더 자주 바뀌는 비동기 곡선보다 거칩니다.
  - 따라서 평활화했거나 여러 번 돌려 평균을 낸 곡선으로 보는 것이 자연스럽습니다.

### 6-3. "기본 모델이 아니었을 수 있다"
- 우리 FedAvg 최종 정확도가 논문보다 높습니다 (LeNet / CIFAR-10 0.504 vs 0.431, VGG / CIFAR-100 0.29 vs 0.0975).
- 구조가 공개된 것은 CNN(표 3)뿐이고, 우리 CNN은 표 3과 같습니다. 나머지 모델은 이름과 인용만 있습니다.
- LeNet은 LeCun et al. **1989**를 인용합니다. 1989 원형은 tanh 활성화와 평균 풀링을 쓰고, 우리 구현은 ReLU와 max-pool입니다. 이런 차이는 정확도를 바꿀 수 있습니다.
- 정확도에 영향을 주는 미공개 값도 많습니다: Dirichlet α, 로컬 epoch 수, 배치 크기, Formula 4의 "epoch" 정의, 감쇠 형태, 전처리.
- 이 가설은 아직 실험하지 않았습니다.
