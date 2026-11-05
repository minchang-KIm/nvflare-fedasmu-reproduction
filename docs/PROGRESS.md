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
