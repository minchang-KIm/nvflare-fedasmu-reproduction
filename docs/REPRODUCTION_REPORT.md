# FedASMU vs FedBuff on NVFLARE: reproduction report

Paper: Liu et al., *FedASMU: Efficient Asynchronous Federated Learning with Dynamic Staleness-aware Model Update*, AAAI-24 (extended version arXiv:2312.05770).
Baseline: FedBuff (Nguyen et al., AISTATS 2022), using NVFLARE's official `BuffModelManager`.
All numbers below come from the CSV files under `results/`. Nothing was estimated or copied in from elsewhere.

## 1. Summary

| Item | Outcome |
|---|---|
| Runtime | NVFLARE 2.9.0 `SimEnv`. FedBuff, FedASMU, ablations and FedAvg all run on the same custom Controller/Executor. No NVFLARE core file was modified. |
| FedBuff | The official NVFLARE aggregation (`BuffModelManager`) is used unchanged. The paper's staleness weight is available as the variant `fedbuff_paper`. |
| FedASMU | Full algorithm: server dynamic aggregation (Eq. 1–3), device fresh-model request and merge (Eq. 7–9), Q-learning time slot (Eq. 6) and LSTM meta model (Eq. 5). |
| Priority experiments | All 6 runs done (seed 0). The first VGG-11+C100 FedASMU run diverged; the reported number is from the re-run with the non-finite safeguard (see §5.3). |
| Ablations | All 8 runs done (CNN+C10). The paper's ordering is **not** reproduced. The differences are within single-seed noise. |
| Multi-seed paper profile, other models/datasets | **NOT RUN** (compute budget). |

Headline (seed 0, final accuracy = mean of the last 5 evaluations):

| Model + data | FedBuff (ours / paper) | FedASMU (ours / paper) | FedASMU − FedBuff (ours / paper) |
|---|---|---|---|
| CNN + CIFAR-10 | 0.564 / 0.364 | 0.540 / 0.603 | **−2.4 pp** / +23.9 pp |
| VGG-11 + CIFAR-10 | 0.662 / 0.620 | 0.673 / 0.653 | **+1.1 pp** / +3.3 pp |
| VGG-11 + CIFAR-100 | 0.265 / 0.238 | 0.292 / 0.264 (re-run) | **+2.7 pp** / +2.6 pp |

- **Accuracy:** with VGG-11, FedASMU is better than FedBuff on both datasets, in the same direction as the paper. The CIFAR-100 margin matches the paper's (+2.7 vs +2.6 pp); the CIFAR-10 margin is smaller. With the small CNN, FedASMU is not better.
- **Speed:** FedASMU reaches the VGG-11 targets 34 % (CIFAR-10) and 35 % (CIFAR-100) sooner in simulated time. The paper reports 60 % and 50 %.
- **Caveats:** one seed per run, so significance is unknown. The CIFAR-100 FedASMU number comes from a re-run after the first attempt diverged (§5.3).
- **CNN gap:** the paper's large CNN gap is not reproduced. Our FedBuff reaches 0.564, not the paper's 0.364 (see §6).

## 2. Environment

The full details are in `docs/ENVIRONMENT.md`.

- **Experiments:** all ran on gpu-host: 2× RTX 3080 10 GB, i9-14900KF, 62 GB RAM, Ubuntu.
- **Development:** done on dev-host. Its GPUs are shared with another user's job, so GPU runs were moved off it.
- **Software:** Python 3.12, NVFLARE 2.9.0, PyTorch 2.11.0+cu128. All versions are pinned in `requirements.lock`.
- **Process layout:** 10 NVFLARE site processes, placed round-robin on the two GPUs. The server (aggregation and evaluation) runs on CPU.
- **Memory:** glibc malloc settings (`scripts/run_experiment.py`) are needed to keep RAM flat for VGG-sized models.

## 3. What was implemented

Details are in `docs/ARCHITECTURE.md` and `docs/PAPER_REPRODUCTION.md`.

**Runtime and simulation**
- There are 100 logical devices multiplexed onto 10 NVFLARE sites, where 10 = max concurrency m′ (10 % of devices).
- Each device keeps its own data shard, its own RL state (γ, υ, Q-table, baseline) and its own RNG stream, stored in its own state file.
- The simulation runs in virtual time. Each device gets epoch time = base × U[1,5] and bandwidth U[1,5] MB/s.
- The server processes uploads in simulated completion order (a conservative discrete-event loop). Different devices really train in parallel processes.
- FedASMU's mid-training model request is an NVFLARE aux message. The server answers it at the virtual request time, with the global model that existed at that time.

**FedBuff**
- Uses `nvflare.edge.assessors.buff_model_manager.BuffModelManager` with K = 10 and global lr 1.0, without changes.
- Differences from the paper are listed in `docs/ARCHITECTURE.md` §5. The main one is the staleness weight: 1/(1+√(τ+1)) in NVFLARE versus 1/√(1+τ) in the paper.

**FedASMU**
- Implemented in `src/algorithms/fedasmu/`.
- Interpretation choices for points the paper leaves unclear:
  - pseudo-gradient sign;
  - the ∂ξ/∂σ typo in the appendix;
  - what "previous upload" means under asynchrony;
  - inputs to the meta model.
- Each choice is marked *Assumption* or *Interpretation* in `docs/PAPER_REPRODUCTION.md`.

**Tests**
- 32 unit tests: finite-difference checks of the Eq. 3 and Eq. 9 gradients, Q-learning, the LSTM meta model, FedBuff buffer semantics and staleness weights, and Dirichlet partitioning.
- 3 integration tests, running real SimEnv jobs: requests and merges actually happen, completion order, and checkpoint resume without duplicates.
- Result: `pytest tests` → 35 passed.

## 4. Experimental conditions (identical for all algorithms)

| Item | Value | Status |
|---|---|---|
| Devices / concurrency / trigger period | 100 / 10 / 10 s | Paper |
| Data split | Dirichlet label skew, α = 0.5, seed 0, same split for every algorithm | Method from paper; α assumed |
| Budget | 5000 received local updates (async); 500 rounds (FedAvg) | Paper |
| Local training | 5 epochs, batch 50, plain SGD, lr per Tables 5–6, decay 0.9998 per received update | lr from paper; the rest assumed |
| FedASMU hyper-parameters | η_λ, η_σ, η_ι, η_γ, η_υ, η_RL per Tables 5–6; µ = 10, λ₀ = 1, σ₀ = 0.5, ι₀ = 0.01, γ₀ = 1, υ₀ = 0.5 | η from paper; initial values assumed |
| Staleness bound | 99 | Paper |
| Initial weights | Same file per (model, dataset, seed) for all algorithms | – |
| Tuning | None. No test-set tuning and no validation tuning. | – |

**How results are measured**
- The test set (official CIFAR test, 10 000 images) is evaluated every 50 received updates (every 5 rounds for FedAvg).
- **Final** accuracy = mean of the last 5 evaluations.
- **Best** = the maximum, reported for information only.
- **Time to target** = the virtual time of the first evaluation at or above the paper's target. It is quantised to the 50-update evaluation grid.
- **Staleness:** for FedASMU it counts global versions (each update creates a version). For FedBuff it counts buffer aggregations, i.e. NVFLARE model versions, so the two are not directly comparable.

## 5. Results

### 5.1 Main comparison (`results/reports/paper_comparison.csv`)

| Run | Status | Final | Best | Paper | Δ to paper | Time to target, virtual s (paper) | Updates to target | Wall-clock |
|---|---|---|---|---|---|---|---|---|
| CNN+C10 FedBuff | FULL / ASSUMPTION-BASED | 0.564 | 0.572 | 0.364 | +20.0 pp | 2 409 (–) | 150 | 23 min |
| CNN+C10 FedASMU | FULL / ASSUMPTION-BASED | 0.540 | 0.565 | 0.603 | −6.3 pp | 2 431 (10 109) | 150 | 41 min |
| VGG-11+C10 FedBuff | FULL / ASSUMPTION-BASED | 0.662 | 0.668 | 0.620 | +4.2 pp | 53 433 (109 082) | 1 100 | 116 min |
| VGG-11+C10 FedASMU | FULL / ASSUMPTION-BASED | 0.673 | 0.681 | 0.653 | +2.0 pp | 35 157 (43 093) | 700 | 88 min |
| VGG-11+C100 FedBuff | FULL / ASSUMPTION-BASED | 0.265 | 0.268 | 0.238 | +2.7 pp | 99 668 (167 053) | 2 050 | 116 min |
| VGG-11+C100 FedASMU (re-run) | FULL / ASSUMPTION-BASED (1 update rejected as non-finite) | 0.292 | 0.300 | 0.264 | +2.8 pp | 64 978 (83 226) | 1 300 | 88 min |
| VGG-11+C100 FedASMU (1st run) | FAILED (diverged at update 2 990) | 0.010 | 0.264 | 0.264 | – | 65 034 (83 226) | – | 72 min |

- **Status labels:** "FULL" means the complete algorithm at the full budget. "ASSUMPTION-BASED" means the result depends on the unpublished settings in §4.
- **Virtual times:** the absolute values are not comparable with the paper, because the paper's time model is unpublished. Only the ratio between the two algorithms on the same simulated system is meaningful.

Figures:
- Accuracy curves for each model and dataset: `results/figures/pilot_<model>_<data>_acc_vs_{updates,vtime,wall}.png`
- Staleness histograms: `results/figures/pilot_<model>_<data>_staleness.png`
- Our results next to the paper's: `results/figures/pilot_paper_vs_reproduced.png`
- Time to target: `results/figures/pilot_time_to_target.png`

### 5.2 FedASMU internals (from `updates.csv`)

| | CNN+C10 | VGG-11+C10 |
|---|---|---|
| Staleness mean / max (versions) | 9.7 / 20 | 9.9 / 21 |
| Updates discarded by bound 99 | 0 | 0 |
| Mean α (server weight) | 0.196 | 0.186 |
| Fresh-model merges (g > o) | 87 % | 93 % |
| Mean β (device weight) | 0.153 | 0.160 |
| Mean reward | +0.006 | +0.021 |
| Request slot l* chosen (1 / 2 / 3 / 4) | 3136 / 733 / 524 / 607 | 1957 / 1272 / 1017 / 754 |
| λ, σ, ι at the end (start 1, 0.5, 0.01) | 1.0008, 0.498, 0.017 | 1.00005, 0.4999, 0.016 |

- **λ, σ, ι barely move.** With the paper's learning rates (1e-3 or 1e-4) and our gradient scale, they stay almost fixed. The "dynamic" aggregation therefore behaves almost like a static staleness polynomial. That is consistent with FedASMU-DA performing the same as FedASMU (§5.4).
- **The reward is small.** Its mean is +0.006 on CNN and +0.021 on VGG. Merging the fresh model helps only marginally on average.

### 5.3 VGG-11 + CIFAR-100 FedASMU: first run diverged, re-run completed

What happened:
- Accuracy rose to 0.264 at about 2 900 updates, which equals the paper's 0.264.
- The test loss had been rising since about 2 400 updates, from 3.49 to 4.61.
- At update 2 990, one device's local training produced NaN weights. The server aggregated them, so α, λ, σ, ι and the global model all became NaN.
- From then on, accuracy was 0.01, and devices that trained from the NaN model failed.

Response:
- The run is archived in `results/failed_runs/pilot_vgg11_cifar100_fedasmu_s0_diverged/`.
- A safeguard was added for **all** algorithms: the server rejects any update with non-finite weights and treats it as a failed device. Rejections are counted as `rejected_nonfinite` in `server_summary.json`.
- The run was restarted from scratch at 06:31 on 2026-11-06.

What is unresolved:
- The underlying instability (no-BN VGG-11, lr 0.03, 100 classes) was **not** changed. Changing it would be tuning.
- The VGG-11/CIFAR-10 FedASMU curve also shows a single sharp dip, from 0.65 to 0.41 at about 135 000 virtual s, which it recovered from. So instability can occur with this configuration.

**Re-run** (2026-11-06 06:31–08:00, same seed and config, plus the safeguard; `results/metrics/pilot_vgg11_cifar100_fedasmu_s0/`):
- COMPLETED: 5000 received, 4999 aggregated.
- At update 2 375 the safeguard rejected one update with non-finite weights (device 25). That update was dropped, and training continued.
- Final accuracy 0.292, best 0.300, against the paper's 0.264 and our FedBuff's 0.265.
- Time to the 0.12 target: 64 978 virtual s. FedBuff took 99 668 s.
- The test loss still rises strongly, from 3.31 at about 1 900 updates to 5.5 at the end, while accuracy keeps improving. The model becomes over-confident without collapsing.

Determinism:
- The re-run's trajectory differs from the first run's from the first evaluations on, although the seed is the same.
- The runs are not bitwise reproducible: GPU kernels are non-deterministic, and the wall-clock order in which sites pick up tasks varies.
- The data split, initial weights, device profiles and virtual-time schedule are identical between the two runs.
- Whether a given seed diverges is therefore partly chance. Without the safeguard, the re-run would very likely also have been destroyed by the update at 2 375.

### 5.4 Ablation (CNN + CIFAR-10, seed 0; `results/reports/ablation.csv`, `results/figures/ablation_*.png`)

| Variant | Definition | Final | Best | Time to 0.40 (virtual s) | Status |
|---|---|---|---|---|---|
| FedASMU | full | 0.540 | 0.565 | 2 431 | FULL |
| FedASMU-DA | λ, σ, ι frozen | 0.546 | 0.566 | 2 431 | ABLATION |
| FedASMU-FA | no fresh-model request | 0.551 | 0.566 | 1 668 | ABLATION |
| FedASMU-0 | neither (static polynomial + bound) | 0.555 | 0.566 | 1 668 | ABLATION |
| FedAvg (sync, 500 rounds) | – | 0.541 | 0.559 | 2 478 | ABLATION |
| FedBuff, NVFLARE weight | – | 0.564 | 0.572 | 2 409 | FULL |
| FedBuff, paper weight 1/√(1+τ) | – | 0.542 | 0.563 | 2 409 | ABLATION |
| FedASMU, H1 (request after epoch 1) | fixed slot | 0.540 | 0.565 | 1 689 | ABLATION |
| FedASMU, H2 (after epoch 2) | fixed slot | 0.539 | 0.568 | 2 431 | ABLATION |
| FedASMU, H3 (after epoch 4) | fixed slot | 0.547 | 0.567 | 2 431 | ABLATION |

How to read this:
- **All variants land within 0.539–0.564** in final accuracy, and within 0.559–0.572 in best accuracy. With a single seed, run-to-run noise is about ±1 pp, so none of the differences can be called significant.
- **The paper's Fig. 2 ordering is not reproduced on CNN+C10.** The paper has FedASMU above -DA and -FA, which are above -0. Here the variants without the request mechanism (-FA, -0) reach the target one evaluation earlier (100 versus 150 updates). The reason is that the request adds latency and an extra download to each training.
- **Fixed-slot heuristics:** H1, H2 and H3 are the same as RL slot selection within noise.

### 5.5 Throughput (wall-clock, gpu-host)

| | Wall-clock for 5000 updates |
|---|---|
| CNN | 23 min (FedBuff / FedASMU-FA / -0) to 41 min (FedASMU with requests) |
| VGG-11 | 88–116 min |

FedASMU with requests is slower in wall-clock on CNN because devices wait in real time for the virtual-time-ordered answer to their request. Paper-style time comparisons should use virtual time, not wall-clock.

## 6. Interpretation and discrepancies

1. **FedBuff on CNN+C10 is much stronger than in the paper** (0.564 versus 0.364).
   - The paper's FedBuff number (0.364) is far below what plain buffered aggregation reaches here.
   - Our FedBuff uses the official NVFLARE aggregation with K = 10 and η_g = 1. We did no tuning in its favour.
   - Possible causes, none of which can be checked from the paper: a different K, server lr or momentum in the paper's FedBuff; their training budget unit; their time model.
   - The reversal of the FedASMU/FedBuff order on CNN is therefore the main **non-reproduced** result.
2. **On VGG-11 the paper's direction is reproduced on both datasets.**
   - CIFAR-10: FedASMU is +1.1 pp better (paper: +3.3 pp) and 34 % faster in virtual time to 0.45 (paper: 60 %).
   - CIFAR-100: FedASMU is +2.7 pp better (paper: +2.6 pp) and 35 % faster to 0.12 (paper: 50 %). This is from the re-run, after one rejected NaN update.
   - Single seed, so the significance is unknown.
3. **The dynamic parts of FedASMU are weak in this setting.**
   - With the published η values, λ, σ and ι change by less than 0.1 %, and the mean reward of merging is around +0.01.
   - Most of FedASMU's behaviour is therefore its staleness-polynomial weighting (α ≈ 0.19 on average) plus fresh-model merging with β ≈ 0.15.
   - This matches the ablation result that the variants are indistinguishable.
4. **Stability.** FedASMU applies every update immediately, with α up to about 0.9 early on, so a single bad update has a large effect.
   - Together with a no-BN VGG-11 at lr 0.03, this produced NaN updates in both C100 attempts (one destroyed the first run) and one large transient dip (C10).
   - FedBuff averages 10 updates before applying them, and neither of its VGG runs showed this.
5. **Assumptions that could change the numbers.**
   - Unknown in the paper: α_Dirichlet, local epochs and batch size, initial control parameters, µ, time and bandwidth model, and how "final accuracy" is measured.
   - The VGG and CNN architecture details are also partly assumed.
   - See `docs/PAPER_REPRODUCTION.md`.

## 7. Status of every planned result

| Result | Status |
|---|---|
| CNN+C10 FedBuff / FedASMU | FULL (seed 0) |
| VGG-11+C10 FedBuff / FedASMU | FULL (seed 0) |
| VGG-11+C100 FedBuff | FULL (seed 0) |
| VGG-11+C100 FedASMU | FULL (re-run, seed 0); first attempt FAILED (diverged), archived |
| Ablations DA / FA / 0 / FedAvg / FedBuff-paper / H1–H3 (CNN+C10) | ABLATION, done (seed 0) |
| Seeds 1, 2 for all of the above (`paper` profile) | NOT RUN |
| CNN+C100, LeNet-5 C10/C100, ResNet-20 C100, AlexNet C10/C100 | NOT RUN (models implemented and configured) |
| TextCNN/IMDb, ResNet-20/Tiny-ImageNet, LeNet/FMNIST | NOT RUN (not implemented) |
| Ablations on models other than CNN+C10 | NOT RUN |

## 8. How to re-run

```bash
bash setup.sh                                   # venv, pinned packages, CIFAR download, environment check
bash run_all.sh --profile smoke                 # unit + integration tests, 4 tiny runs (~5 min, CPU)
bash run_all.sh --profile pilot                 # 6 runs, seed 0 (~9 h on 2x RTX 3080; run one at a time, ~25 GB RAM)
bash run_all.sh --profile ablation              # 8 CNN runs (~4 h)
bash run_all.sh --profile paper                 # 18 combos x 3 seeds (not executed here)
bash run_all.sh --profile pilot --only vgg11_cifar10_fedasmu   # a single run
.venv/bin/python report.py --profiles pilot ablation          # tables and figures only
```

- Runs that are already COMPLETED are skipped.
- A run that is interrupted resumes from its last checkpoint (every 500 updates).
- The per-run config, git hash, versions and data digests are in `results/metrics/<run_id>/manifest.json`.
