"""Print versions and verify CUDA / NVFLARE / datasets are usable."""
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    import nvflare
    import torch

    print("python", platform.python_version(), "| nvflare", nvflare.__version__, "| torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB")
    from src.datasets.cifar import load_tensor_dataset

    for d in ("cifar10", "cifar100"):
        x, y, xt, yt, n = load_tensor_dataset(d)
        print(f"{d}: train {tuple(x.shape)} test {tuple(xt.shape)} classes {n}")
    try:
        print("git:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    except Exception:
        print("git: (no commit yet)")


if __name__ == "__main__":
    main()
