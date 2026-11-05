#!/usr/bin/env bash
# Create the isolated environment and download datasets.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install nvflare==2.9.0
.venv/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install numpy pandas matplotlib pyarrow pillow pytest pyyaml
.venv/bin/pip check
.venv/bin/python scripts/prepare_data.py --datasets cifar10 cifar100
.venv/bin/python scripts/validate_environment.py
