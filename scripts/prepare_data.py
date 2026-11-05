"""Download datasets used by the reproduction into ./data (git-ignored).

Usage: python scripts/prepare_data.py [--datasets cifar10 cifar100]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.cifar import DATA_ROOT, load_tensor_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    args = ap.parse_args()
    os.makedirs(DATA_ROOT, exist_ok=True)
    for name in args.datasets:
        x, y, xt, yt, n_cls = load_tensor_dataset(name, download=True)
        print(f"{name}: train={tuple(x.shape)} test={tuple(xt.shape)} classes={n_cls}")


if __name__ == "__main__":
    main()
