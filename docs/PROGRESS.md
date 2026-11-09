# Progress log

Commit hashes refer to the commit that closes the step (the hash of the commit that
records a hash is necessarily the *next* one; each entry lists the code commit).

## Step 1 — project, Git, NVFLARE install — DONE
- Implemented: project folder, git repo (`main`), `.gitignore`, isolated `.venv`,
  NVFLARE 2.9.0 + PyTorch 2.11 (cu128), dataset preparation (`scripts/prepare_data.py`),
  `setup.sh`, `scripts/validate_environment.py`, sanity example `examples/sanity_fedavg`.
- Tests: official `FedAvgRecipe` in `SimEnv` (2 sites, 3 rounds, real SGD on GPU):
  global accuracy 0.1035 → 0.3230 → 0.3720 (weights really updated).
- Errors fixed: torchvision CIFAR download at ~30 KB/s → switched to HF parquet mirror.
- Open issues: remote GitHub repository does not exist and cannot be created (no gh/token).
- Remote push: see per-step entries below.
- Next: Step 2 paper analysis.

## Step 2 — paper analysis and settings — DONE
- `docs/PAPER_REPRODUCTION.md`: every setting is tagged as Paper, Assumption, Interpretation or Not run.
  Hyperparameters come from the arXiv appendix Tables 5/6, the CNN structure from Table 3,
  and the heterogeneity (5×), m′ = 10 %, τ = 99 and 5000/500 budgets from the appendix text.
- Executable configs: `configs/common/{base,paper_settings}.yaml`, `configs/experiments/*.yaml`.
- The partition, initial weights and device profiles are generated once per (dataset, seed) and shared
  by all algorithms (sha256 stored in `results/manifests`).
- Commit 7bb3d64. Push to GitHub: FAILED (repository does not exist; no gh/token).

## Step 3 — NVFLARE execution structure — DONE
- Chose a custom Controller + Executor + aux channel. EdgeFedBuff's `BuffModelManager` is reused unchanged
  for FedBuff. The Collab API was rejected (reasons in `docs/ARCHITECTURE.md`).
- The mid-training fresh-model request is verified in real SimEnv runs: `merged=True`, `fresh_version > o`.
- Commit ec1cce1. Push to GitHub: FAILED (same reason).

## Step 4 — FedBuff — DONE
- Official `BuffModelManager` (`fedbuff`) and the paper staleness variant (`fedbuff_paper`).
  A line-by-line comparison is in ARCHITECTURE.md §5. The difference: the official staleness weight is 1/(1+√(τ+1)).
- Tests: buffer waits for K, the same device counted twice, staleness weights for both variants, global_lr.
  Real training: smoke 24 updates → 2 aggregations, accuracy 0.092 → 0.240.
- Commit be02887. Push to GitHub: FAILED.

## Step 5 — FedASMU — DONE (code + tests); full-scale accuracy in step 7
- Server Eq. 1–3 (including appendix derivatives, with corrected sign and ln-term, documented); device Eq. 7–9;
  LSTM meta model (Eq. 5, REINFORCE); Q-learning (Eq. 6); request after l* epochs over NVFLARE aux; merge; reward.
- Tests: 32 unit tests (hand computations, finite differences vs autograd, Eq. 7 merge with lr = 0,
  no merge if g ≤ o, Q update by hand, REINFORCE direction, checkpoint round trips) plus 3 NVFLARE
  integration tests (async order, real merges, FedBuff K, resume without duplicates): **35 passed**.
- Errors fixed during this step:
  - The server's RAM out-of-memory with VGG was fixed by a disk-backed previous-delta store.
  - GPU contention: one CUDA context per site via `set_device`, an OOM retry, and a CPU server.
  - Request/poll is idempotent for retried tasks.
- Commit 65e5332 (+ docs 839f9fc). Push to GitHub: FAILED; push to node2: OK.

## Step 6 — common experiment environment — DONE
- Same partition, initial weights, device profiles, local optimizer/lr/batch/epochs, budget and evaluation code for
  every algorithm. Only the `algorithm` key differs.
- Smoke profile re-run on gpu-host (CPU): fedbuff, fedbuff_paper, fedasmu and fedavg all COMPLETED.

## Step 7 — pilot runs — IN PROGRESS (started 2026-11-05 ~18:45 on gpu-host)
- CNN+CIFAR-10 pilot pair COMPLETED (5000 updates each): FedASMU final 0.540 / best 0.565; FedBuff final 0.564 / best 0.572.
- Errors fixed (19:10–19:50):
  1. Running two streams in parallel → kernel OOM-killed both VGG FedBuff servers (RAM). Decision: one run at a time.
  2. A single VGG run was still OOM-killed (FedASMU VGG-C10 at 700 updates). Cause: the RSS of all processes grew
     linearly (~150 MB per 20 s) through glibc heap fragmentation from model-sized buffers. Fix: MALLOC_MMAP_THRESHOLD_ = 1 MiB,
     MALLOC_TRIM_THRESHOLD_ = 1 MiB, MALLOC_ARENA_MAX = 4 (scripts/run_experiment.py). Verified: total site RSS flat at
     ≈16 GB and the server oscillates 4–7 GB over 370 updates.
  3. A crashed run restarted without a checkpoint appended to its old CSV → partial logs are now moved aside.
- The failed VGG runs were deleted and are re-run from scratch by `scripts/queue_node2.sh` (sequential queue).
- 21:18 VGG-11+CIFAR-10 FedASMU COMPLETED (5000 updates, 88 min wall, no OOM kill): final 0.673 / best 0.681
  (paper 0.653, +2.0 pp); time to target 0.45 = 35 157 virtual s (paper 43 093 s, different time model).
  Its LSTM history was saved as results/metrics/metahist_vgg11_cifar10.json for later pre-training. Next in queue: VGG C10 FedBuff.
- 23:14 VGG-11+CIFAR-10 FedBuff (official NVFLARE manager) COMPLETED (5000 updates / 500 aggregations, 116 min wall):
  final 0.662 / best 0.668 (paper 0.620, +4.2 pp); time to 0.45 = 53 433 virtual s vs FedASMU 35 157 s
  (FedASMU 34 % faster to target; paper: 43 093 vs 109 082 s). Next in queue: VGG-11+CIFAR-100 FedASMU.
- Git: node2's checkout is written by the running queue, so pushes go to branch `node1-sync` on node2.
- 00:27 VGG-11+CIFAR-100 FedASMU **diverged** (run itself finished, status COMPLETED, but the result is invalid):
  accuracy rose to best 0.2637 (paper 0.264) at ~2900 updates, but test loss had been rising since ~2400.
  At update 2990 one device's local training (device 96, lr 0.0165) produced NaN weights.
  The server aggregated them, so α, λ, σ, ι and the global model became NaN, and accuracy fell to 0.01.
  The run is archived under results/failed_runs/pilot_vgg11_cifar100_fedasmu_s0_diverged (with its LSTM history).
  - Fix (all algorithms): the controller now rejects an update whose weights are not finite, counts it as a
    failed device (`slot_src = rejected:nonfinite`, `rejected_nonfinite` in server_summary.json), and does not
    aggregate it. Tests: 35 passed.
  - The run is re-queued from scratch (no checkpoint, no meta pre-training, same as the VGG-C10 pilot)
    after the current queue. The underlying instability (lr 0.03 on VGG-11 without BN, 100 classes) is not
    changed. If the re-run degrades again, it is reported as such.
- 02:23 VGG-11+CIFAR-100 FedBuff COMPLETED (5000 updates / 500 aggregations, 116 min wall): final 0.265 / best 0.268
  (paper 0.238, +2.7 pp); time to 0.12 = 99 668 virtual s (paper 167 053 s). Test loss rises after ~3400 updates
  (3.22 → 3.82) while accuracy still improves (overfitting/over-confidence, no NaN). Next: ablation profile (8 CNN runs).
- 06:31 Ablation profile COMPLETED (8 CNN+CIFAR-10 runs, seed 0, 5000 updates each, no failures):
  final accuracy FedASMU 0.540 (pilot), -DA 0.546, -FA 0.551, -0 0.555, FedAvg 0.541 (500 rounds),
  FedBuff-paper-weight 0.542 (official 0.564), H1 0.540, H2 0.539, H3 0.547. All within ~1.5 pp of each other.
  Single seed, so the differences are not significant. The paper's ablation ordering (FedASMU > -DA/-FA > -0) is **not** reproduced on CNN.
  Next: VGG-11+CIFAR-100 FedASMU re-run (started 06:31), REPRODUCTION_REPORT.md.
- 08:02 VGG-11+CIFAR-100 FedASMU re-run COMPLETED (88 min wall): final 0.292 / best 0.300 (paper 0.264, +2.8 pp;
  FedBuff 0.265). Time to 0.12 = 64 978 virtual s (FedBuff 99 668 s). The safeguard rejected 1 non-finite update
  (update 2375, device 25). Same seed, but the trajectory differs from the first run (GPU non-determinism and
  site scheduling), so the runs are not bitwise reproducible.
- Step 7 — DONE. All 6 pilot runs and 8 ablation runs completed.

## Step 8 — report — DONE
- docs/REPRODUCTION_REPORT.md: environment, implementation, conditions, results vs paper, internals, ablation,
  divergence analysis, interpretation, status of every planned result (FULL / ABLATION / FAILED / NOT RUN), and re-run commands.
- results/reports/{summary,paper_comparison,ablation}.csv and results/figures/*.png were regenerated by report.py.

## Final checklist
| Item | State |
|---|---|
| NVFLARE official FedBuff (BuffModelManager) + paper comparison | done (ARCHITECTURE.md §5, unit tests) |
| FedASMU Eq. 1–3, 5–9 incl. LSTM + Q-learning | done (unit + integration tests, 35 passed) |
| Same runtime for all algorithms | done (AsyncSimController / DeviceSimExecutor on SimEnv) |
| CNN-C10, VGG-C10, VGG-C100 × {FedBuff, FedASMU} | done, seed 0 |
| Ablations DA / FA / 0 / FedAvg (+ FedBuff paper weight, H1–H3) | done, CNN-C10, seed 0 |
| run_all.sh smoke/pilot/paper, setup.sh, report.py, checkpoint/resume | done |
| Multi-seed paper profile, other model/dataset combos | NOT RUN |
| Push to GitHub origin | FAILED every step: the repository does not exist and there is no token to create it. Synced to node2 (branches main and node1-sync) instead. |

## Step 9 — remaining paper combinations — IN PROGRESS (started 2026-11-09 on gpu-host)
- Profile `extended` (seed 0, same conditions as the pilot): CNN-C100, LeNet-C10/C100, AlexNet-C10/C100, ResNet-20-C100 × {FedASMU, FedBuff}, then LeNet-FMNIST, TextCNN-IMDb, ResNet-20-Tiny-ImageNet.
- Added: datasets FMNIST, Tiny-ImageNet (val split as test), IMDb (Hugging Face parquet exports), model TextCNN, and the paper's Table 1/2/5/6 values for the three new combinations.
- Assumptions for the new parts: IMDb vocabulary 20 000 (train split only), length 400, TextCNN embedding 100 and 100 filters of width 3/4/5; trainable random embedding; base epoch time 10 s (LeNet, TextCNN), 30 s (ResNet-20).
- Tests: 38 unit + 3 integration passed. CPU functional checks of the new runs: FMNIST and IMDb completed (24 updates); Tiny-ImageNet reached 16/24 updates and was stopped by the time limit (CPU only, no error).
