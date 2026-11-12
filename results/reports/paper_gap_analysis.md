# Paper vs ours: why the appendix curves differ

Generated tables below come from `scripts/analyze_paper_gap.py`. The paper curves are digitised from the
vector graphics of the paper PDF (`scripts/digitize_paper_curves.py`, arXiv 2312.05770v1, CC BY 4.0).
Check of the digitisation: the last point of every FedASMU / FedBuff / FedAvg curve is within 0.005 of the
final accuracy in the paper's Tables 1-2. The time-to-target read from the curves is within 5 % of the
tables, except VGG-11 / CIFAR-100 FedBuff, where the paper's own figure (249,017 s) and table (167,053 s) disagree.

## 1. Was the stated setting reproduced?

Yes. `scripts/check_paper_settings.py` (report: `paper_settings_check.md`) compares every appendix run with
the paper's stated setting, transcribed again from the paper independently of the config files. It checks
both the recorded configuration and what the logs show actually happened. All completed runs pass every check:
- 100 devices and 10 concurrent: at most 10 running in the logs.
- τ = 99: the largest accepted staleness was ≤ 99.
- T = 10.
- Budgets: 5,000 updates for asynchronous methods, 500 rounds for FedAvg.
- Speeds: slowest device ≤ 5× the fastest; measured spread 4.8×.
- Every learning rate of Tables 5–6.

The three methods of a workload use the same data split, initial weights and device profiles. The algorithm
itself passes 27/27 formula checks on the run logs (`results/verification/`).

What the paper does not state, and we had to assume, is listed in `docs/PAPER_REPRODUCTION.md`:
- μα, μβ, λ₀, σ₀, ι₀, γ₀, υ₀;
- the number of local epochs L, the batch size and the Dirichlet α;
- the time per epoch and the bandwidth;
- FedBuff's K and server learning rate.

The differences below come from these unstated values and from how the paper's figures were produced, not
from a deviation from the stated setting.

## 2. The four visible differences

### (a) Our runs end earlier on the time axis
- **Ours:** runs end at 77 k–390 k virtual seconds.
- **Paper:** all 10 methods end at exactly 450 k s (100 k s for FMNIST and IMDb), including the synchronous
  methods with a 500-round budget and the asynchronous methods with a 5,000-update budget.

The paper therefore plots a fixed time window. Our absolute time scale follows from the assumed time per
local epoch (10 s or 30 s per epoch, slowed down by up to 5× per device). This changes only the scale of the
x axis, not the order of the methods.

### (b) FedAvg is not slower than FedASMU in our runs, but is 2–14× slower in the paper
- **Paper:** FedAvg needs 2.3–14× FedASMU's time to reach the target (table below).
- **Ours:** 0.9–1.5×.

The paper's own time model cannot produce its ratios:
- With m′ = 10 devices per round and device times spread at most 5×, a synchronous round of 10 trainings costs
  the time of its slowest device.
- For speeds uniform in [1, 5], the expected slowest of 10 is 1.55× the mean device time. That is exactly what
  our runs show: the 500 FedAvg rounds take 1.55× as long as the 5,000 asynchronous updates.
- Even in the worst case (every round waits for a device 5× slower than every other), the time model can slow
  FedAvg down by at most 5×.

The paper reports 7–14× on 8 of 11 workloads. In the paper, FedAvg must therefore also need several times more
local trainings than FedASMU to reach the target.

In our runs, FedAvg needs the fewest local trainings of the three methods (e.g. LeNet / CIFAR-10: FedAvg 150,
FedASMU 200, FedBuff 300). This is expected, because FedAvg averages models that all started from the same
global model and so has no staleness.

The paper's synchronous baselines (FedAvg, FedProx, MOON, FedDyn) all rise equally slowly. This suggests a
setting for synchronous methods that the paper does not describe, so we cannot reproduce it from the text.

- **Straggler test (diagnose profile, LeNet / CIFAR-10, device speeds U[1, 110]).** This uses the "110 times"
  of the paper's Figure 7 text.
  - FedAvg now takes 1.82× the asynchronous time for its whole budget, up from 1.55×.
  - FedAvg reaches the targets only 1.38× (0.30) and 1.25× (0.40) later than FedASMU. The paper reports 14.1×
    and 12.3×.
  - Final accuracies barely change: FedASMU 0.466, FedBuff 0.535, FedAvg 0.508.
  - Conclusion: the stated straggler setting does not produce the paper's slow FedAvg curves.

### (c) The paper's curves are 10–100× smoother
- **Roughness** (table below) is the mean |Δ accuracy| between neighbouring points in the second half of the
  run. Paper: 0.0001–0.0006. Ours: 0.002–0.03.
- **Paper's asynchronous curves are smoother than its FedAvg curves.** This is so even though the asynchronous
  methods change the global model 10× more often than FedAvg does (5,000 updates vs 500 rounds).
- **Likely cause:** the plotted curves were smoothed or averaged over runs. The paper does not say so.
- **Our curves:** single-seed test accuracies evaluated every 50 updates, not smoothed.

### (d) FedASMU is not better than FedBuff in our runs
- **α is large.** FedASMU's server weight α (Eq. 1–2) averages 0.17–0.27, and 0.80 on LeNet / CIFAR-10.
  - In FedBuff (K = 10), one update weighs 0.1 and is averaged with 9 others.
  - With a larger single-update weight, the global model follows individual non-IID devices more closely.
  - Our FedASMU curve is rougher than our FedBuff curve on 9 of 10 workloads.
  - The scale of α is set by μα, λ₀ and ι₀, which the paper does not give (assumed: 10, 1, 0.01).
- **The control parameters drift in one direction.** Eq. 3 estimates ∂F/∂ι from gᵢ(w_o)ᵀ(w_{o′}ⁱ − w_{o−1}),
  the product of a device's current update and its own previous update.
  - Under non-IID data, a device's consecutive updates point the same way.
  - This product had the same sign in 99.8–100 % of updates on every image workload (90 % on IMDb).
  - So the estimate is not a noisy, zero-mean gradient but a constant push, and ι moves monotonically at a
    speed set by η_ι.
  - On LeNet / CIFAR-10, the paper's η_ι = 0.1 (Table 5) takes ι from 0.01 to 0.67 and α to 0.87.
  - This is the paper's estimator applied as printed. Recomputation from the logs matches it exactly
    (checks S6–S8), so it is not a coding error.
  - The direction of the drift depends on the sign convention of the pseudo-gradient (see section 3).
- **FedBuff.** The paper's FedBuff stops rising at 0.36–0.37 on LeNet and CNN / CIFAR-10. The NVFLARE FedBuff
  (K = 10, server learning rate 1) keeps improving to 0.54–0.56. The paper does not give its FedBuff K or
  server learning rate.

## 3. Diagnostic runs (`configs/experiments/diagnose.yaml`, seed 0, one change each)

These runs are not tuning and do not replace any appendix result; all of them are reported.
Figure: `results/figures/appendix/gap_diagnose.png`.

| run | change | final acc. | alpha mean (last 500) | iota end | checks |
|---|---|---|---|---|---|
| LeNet / CIFAR-10 FedASMU (appendix) | none (SGD-consistent sign) | 0.448 | 0.87 | +0.67 | 27/27 |
| LeNet / CIFAR-10 FedASMU, printed sign | Eq. 3 sign exactly as printed in the appendix | 0.362 | 0.00 | -9.75 | 28/28 |
| LeNet / CIFAR-10 FedASMU-DA | Eq. 3 off (lambda, sigma, iota fixed) | 0.523 | 0.13 | 0.01 | - |
| LeNet / CIFAR-10 FedBuff / FedAvg (appendix) | - | 0.540 / 0.504 | - | - | - |
| CNN / CIFAR-10 FedASMU (appendix) | none | 0.537 | 0.19 | +0.019 | 27/27 |
| CNN / CIFAR-10 FedASMU, printed sign | Eq. 3 sign as printed | 0.550 | 0.04 | -0.001 | 28/28 |
| CNN / CIFAR-10 FedBuff (appendix) | - | 0.564 | - | - | - |

Findings:
- With the sign printed in the paper, iota runs away the other way. On LeNet / CIFAR-10 (eta_iota = 0.1) it
  reaches -9.7 and alpha drops to 0 after about 400 updates. From then on, the global model no longer changes:
  the accuracy stays at 0.362.
- With the SGD-consistent sign, iota grows and alpha rises to 0.87 (accuracy 0.448).
- With Eq. 3 switched off, alpha stays at 0.13-0.16 and the accuracy is 0.523. This is our best FedASMU
  variant on this workload and is above the paper's FedASMU (0.486), but still below FedBuff (0.540).
- On CNN / CIFAR-10, where eta_iota is 1000x smaller (1e-4), the drift is slow and both signs end within
  1.3 pp (0.537 / 0.550). Both are still below FedBuff (0.564).
- Conclusion: the Eq. 3 control-parameter update, as specified, drifts monotonically in either sign
  convention, because its gradient estimate is not zero-mean under non-IID data. With the paper's large
  eta_iota (LeNet / CIFAR-10) this dominates the result. Neither convention reproduces the paper's smooth
  0.486 curve, and none of the variants beats FedBuff. The paper's result therefore depends on something the
  paper does not state: the initial values or mu_alpha, a clipping of the control parameters, or the
  smoothing of the plotted curves.

<!-- tables -->

## Time to target (virtual s) and local trainings to target (n)

Paper times are read from the digitised paper curves; n = received local updates (FedAvg: device trainings, 10 per round).

| workload | target | paper FedASMU t | ours FedASMU t (n) | paper FedBuff t | ours FedBuff t (n) | paper FedAvg t | ours FedAvg t (n) | paper FedAvg/FedASMU | ours FedAvg/FedASMU |
|---|---|---|---|---|---|---|---|---|---|
| lenet5-cifar10 | 0.3 | 8,984 | 3,158 (200) | 11,967 | 4,570 (300) | 126,433 | 3,656 (150) | 14.1 | 1.16 |
| cnn-cifar10 | 0.4 | 10,771 | 2,431 (150) | - | 2,409 (150) | 121,669 | 2,478 (100) | 11.3 | 1.02 |
| resnet20-tinyimagenet | 0.12 | 23,151 | - (-) | 46,552 | - (-) | 160,786 | - (-) | 6.9 | - |
| lenet5-cifar100 | 0.13 | 21,264 | 7,749 (500) | 26,251 | 11,379 (750) | 97,810 | 8,272 (350) | 4.6 | 1.07 |
| cnn-cifar100 | 0.15 | 31,675 | 2,577 (150) | 68,139 | 2,583 (150) | 73,704 | 3,828 (150) | 2.3 | 1.49 |
| resnet20-cifar100 | 0.25 | 16,641 | 56,544 (1,250) | 28,350 | 77,762 (1,700) | 118,533 | 54,676 (800) | 7.1 | 0.97 |
| alexnet-cifar10 | 0.4 | 13,810 | 39,920 (700) | 29,402 | 61,679 (1,150) | 162,234 | 34,801 (450) | 11.7 | 0.87 |
| vgg11-cifar10 | 0.45 | 44,059 | 32,684 (650) | 116,204 | 53,433 (1,100) | 341,245 | 32,335 (450) | 7.7 | 0.99 |
| textcnn-imdb | 0.85 | 3,608 | - (-) | 7,872 | - (-) | 13,960 | - (-) | 3.9 | - |
| alexnet-cifar100 | 0.12 | 12,783 | 102,912 (1,800) | 28,351 | - (-) | 93,991 | - (-) | 7.4 | - |
| vgg11-cifar100 | 0.12 | 84,226 | 65,550 (1,300) | 249,017 | 99,668 (2,050) | - | 68,322 (950) | - | 1.04 |
| lenet5-fmnist | 0.7 | 8,601 | 10,744 (700) | 30,015 | 14,474 (950) | 66,605 | 11,710 (500) | 7.7 | 1.09 |

## Curve roughness and FedASMU control parameters

Roughness = mean |Δ accuracy| between 50 equally spaced points in the second half of the run.

| workload | paper FedASMU | ours FedASMU | paper FedBuff | ours FedBuff | paper FedAvg | ours FedAvg | ours alpha mean | ours iota end | share of g·Δprev < 0 |
|---|---|---|---|---|---|---|---|---|---|
| lenet5-cifar10 | 0.0004 | 0.0276 | 0.0001 | 0.0068 | 0.0025 | 0.0089 | 0.804 | 0.674 | 0.998 |
| cnn-cifar10 | 0.0003 | 0.0085 | 0.0004 | 0.0052 | 0.0021 | 0.0070 | 0.196 | 0.019 | 1.000 |
| resnet20-tinyimagenet | 0.0002 | - | 0.0001 | - | 0.0003 | - | - | - | - |
| lenet5-cifar100 | 0.0001 | 0.0066 | 0.0002 | 0.0027 | 0.0003 | 0.0026 | 0.266 | 0.045 | 1.000 |
| cnn-cifar100 | 0.0002 | 0.0017 | 0.0002 | 0.0014 | 0.0003 | 0.0015 | 0.170 | 0.012 | 1.000 |
| resnet20-cifar100 | 0.0001 | 0.0099 | 0.0001 | 0.0029 | 0.0004 | 0.0048 | 0.246 | 0.035 | 1.000 |
| alexnet-cifar10 | 0.0005 | 0.0077 | 0.0006 | 0.0064 | 0.0018 | 0.0059 | 0.192 | 0.019 | 1.000 |
| vgg11-cifar10 | 0.0004 | 0.0042 | 0.0010 | 0.0053 | 0.0031 | 0.0040 | 0.186 | 0.016 | 1.000 |
| textcnn-imdb | 0.0001 | 0.0015 | 0.0001 | 0.0105 | 0.0001 | 0.0157 | 0.181 | 0.016 | 0.905 |
| alexnet-cifar100 | 0.0002 | 0.0115 | 0.0002 | - | 0.0006 | - | 0.218 | 0.032 | 1.000 |
| vgg11-cifar100 | 0.0005 | 0.0081 | 0.0027 | 0.0039 | 0.0014 | 0.0046 | 0.229 | 0.032 | 0.998 |
| lenet5-fmnist | 0.0006 | 0.0105 | 0.0008 | 0.0065 | 0.0015 | 0.0120 | 0.168 | 0.012 | 0.999 |
