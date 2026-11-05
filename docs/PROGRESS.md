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
