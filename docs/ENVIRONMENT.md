# Execution environment (measured 2026-11-05)

| Item | Value | How checked |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, x86_64 | `uname -a`, `/etc/os-release` |
| CPU | Intel Core i9-14900KF, 32 logical CPUs | `lscpu` |
| RAM | 62 GiB (≈56 GiB available at start), swap 1 GiB | `free -g` |
| GPU | 2 × NVIDIA GeForce RTX 3080, 10 GiB VRAM each | `nvidia-smi` |
| Driver / CUDA | driver 570.211.01, driver CUDA 12.8 | `nvidia-smi` |
| Disk | 215 GB free on `/` | `df -h` |
| Python | 3.12.7 (Anaconda interpreter used only to create an isolated venv `.venv`) | `python3 --version` |
| NVFLARE | **2.9.0** (latest stable on PyPI at setup time; 2.9.1 only as rc) | `pip index versions nvflare` |
| PyTorch | 2.11.0+cu128, torchvision 0.26.0+cu128, `torch.cuda.is_available()==True`, 2 devices | import check |
| Other | numpy, pandas, matplotlib, pyarrow, pillow, pytest, pyyaml (see `requirements.lock`) | `pip freeze` |
| Git | git user `minchang-KIm`; SSH key authenticates to GitHub (`ssh -T git@github.com` OK) | |
| GitHub CLI | **not installed**, no `GH_TOKEN`/`GITHUB_TOKEN` | `which gh` |

## Decisions

* **Isolated venv** (`.venv`, git-ignored) instead of the Anaconda base env, to avoid
  conflicts with packages already in base. `pip check` → *No broken requirements found*.
* **NVFLARE source of truth = installed wheel 2.9.0** (`.venv/lib/python3.12/site-packages/nvflare`).
  All API usage (Controller/Executor, aux channel, `BuffModelManager`, `SimEnv`, `FedJob`)
  was read from this installed code, not from GitHub `main`.
* **Datasets**: CIFAR-10/100 are taken from the Hugging Face mirror of the original
  archives (`uoft-cs/cifar10`, `uoft-cs/cifar100`) because the cs.toronto.edu mirror
  delivered ~30 KB/s on this host. 50 000/10 000 train/test images, identical content.
* **Parallelism**: the NVFLARE simulator starts one OS process per simulated site
  (`simulator_worker`). With 10 sites (= max concurrency of the paper, 10 % of 100
  devices) and 2 GPUs, sites are placed round-robin on `cuda:0`/`cuda:1`.
  100 logical devices are multiplexed onto the 10 site processes (see ARCHITECTURE.md).

## Remote Git

The SSH key works, but the remote repository `git@github.com:minchang-KIm/nvflare-fedasmu-reproduction.git`
does not exist and cannot be created here (no `gh`, no API token). `origin` is configured
to that URL; every step attempts `git push` and records the actual outcome in PROGRESS.md.
To enable sync, create an empty private repo with that name and run `git push -u origin main`.

## Step-1 sanity run (official FedAvgRecipe + SimEnv + Client API)

`python examples/sanity_fedavg/run.py` — 2 sites, 3 rounds, paper CNN, 2 000 CIFAR-10
samples/site, real SGD on GPU. Global model accuracy evaluated on the received global
model at each round (log `logs/step1_sanity_fedavg.log`, git-ignored):

| round | accuracy of received global model |
|---|---|
| 0 (initial) | 0.1035 |
| 1 | 0.3230 |
| 2 | 0.3720 |

→ Server/clients run, weights are aggregated and the global model changes.
