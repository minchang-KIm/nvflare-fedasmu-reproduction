# FedASMU vs FedBuff on NVIDIA FLARE: a reproduction

This repository reproduces the CIFAR experiments of **FedASMU** (Liu et al., *Efficient Asynchronous Federated Learning with Dynamic Staleness-aware Model Update*, AAAI-24; [arXiv:2312.05770](https://arxiv.org/abs/2312.05770)) and compares FedASMU with **FedBuff** (Nguyen et al., AISTATS 2022).

All algorithms run on the same [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) 2.9.0 runtime:
- FedBuff uses NVFLARE's official `BuffModelManager` without modification.
- FedASMU implements the complete algorithm:
  - server-side dynamic staleness-aware aggregation (Eq. 1–3);
  - device-side mid-training fresh-model request and merge (Eq. 7–9);
  - Q-learning (Eq. 6) and an LSTM meta model (Eq. 5) for choosing the request time slot.

No official FedASMU code exists. Every unpublished detail is listed as an assumption in [docs/PAPER_REPRODUCTION.md](docs/PAPER_REPRODUCTION.md).

## Results (seed 0, 100 devices, 10 concurrent, 5000 updates)

Final accuracy is the mean of the last 5 test evaluations.

| Model + data | FedBuff (ours / paper) | FedASMU (ours / paper) | FedASMU − FedBuff (ours / paper) |
|---|---|---|---|
| CNN + CIFAR-10 | 0.564 / 0.364 | 0.540 / 0.603 | −2.4 pp / +23.9 pp |
| VGG-11 + CIFAR-10 | 0.662 / 0.620 | 0.673 / 0.653 | +1.1 pp / +3.3 pp |
| VGG-11 + CIFAR-100 | 0.265 / 0.238 | 0.292 / 0.264 | +2.7 pp / +2.6 pp |

**What reproduces**
- On VGG-11, FedASMU beats FedBuff on both datasets.
- It also reaches the target accuracy about 35 % sooner in simulated time. The paper reports 50–60 %.

**What does not**
- On the small CNN, FedBuff is much stronger than the paper reports (0.564 vs 0.364), and FedASMU is not better.
- The ablations (FedASMU-DA / -FA / -0, FedAvg, fixed request slots) all land within about 1.5 pp of each other on CNN + CIFAR-10. The paper's ordering is not reproduced.

**Caveats**
- Each result comes from a single seed, so differences are not shown to be significant.
- The first VGG-11 + CIFAR-100 FedASMU run diverged (NaN). The run reported above is a re-run with a safeguard that drops non-finite updates.

The full analysis is in **[docs/REPRODUCTION_REPORT.md](docs/REPRODUCTION_REPORT.md)**. Tables are in [results/reports/](results/reports/) and figures in [results/figures/](results/figures/).

## Quick start

Requirements: Linux, Python 3.12, a CUDA GPU (CPU works for the smoke profile).

```bash
bash setup.sh                         # venv, pinned packages, CIFAR-10/100 download, environment check
bash run_all.sh --profile smoke       # unit + integration tests and 4 tiny runs (~5 min, CPU)
bash run_all.sh --profile pilot       # 3 model/data combos x {FedBuff, FedASMU}, seed 0
bash run_all.sh --profile ablation    # FedASMU-DA/-FA/-0, FedAvg, FedBuff paper weight, H1-H3
bash run_all.sh --profile paper       # all CIFAR combos x 3 seeds (not executed here)
.venv/bin/python report.py --profiles pilot ablation   # rebuild tables and figures
```

- Runs that are already completed are skipped.
- An interrupted run resumes from its last checkpoint.
- For VGG-sized models, run one experiment at a time (about 25 GB RAM).

## Repository layout

| Path | Contents |
|---|---|
| `src/algorithms/fedasmu/` | Server aggregation, device merge, Q-learning, LSTM meta model |
| `src/algorithms/fedbuff/` | Wrapper around NVFLARE `BuffModelManager` (plus the paper's staleness-weight variant) |
| `src/nvflare_integration/` | Custom NVFLARE Controller (virtual-time asynchronous event loop) and Executor |
| `src/models/`, `src/datasets/`, `src/scheduling/` | Models (CNN, VGG-11, LeNet-5, ResNet-20, AlexNet), Dirichlet partitioning, device heterogeneity |
| `configs/` | Common settings, per-task paper hyper-parameters, experiment profiles |
| `scripts/run_experiment.py`, `run_all.sh`, `report.py` | Experiment runner, profiles, report generation |
| `tests/` | 38 unit tests and 3 NVFLARE integration tests |
| `results/` | Per-run metrics (CSV), manifests, summary tables, figures |
| `docs/` | Report, architecture, paper settings, environment, progress log |

## Documentation

- [docs/REPRODUCTION_REPORT.md](docs/REPRODUCTION_REPORT.md): results, interpretation, and the status of every planned experiment
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): NVFLARE design, virtual-time simulation, and FedBuff (NVFLARE) vs the FedBuff paper
- [docs/PAPER_REPRODUCTION.md](docs/PAPER_REPRODUCTION.md): every setting, marked Paper / Assumption / Interpretation
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md): hardware and software used
- [docs/PROGRESS.md](docs/PROGRESS.md): step-by-step development log, including errors and fixes
