"""CIFAR-10/100 loading as uint8 tensors (cached), normalisation constants.

Data are kept as uint8 NCHW tensors so they can be put on the GPU cheaply and
normalised per mini-batch.  No data augmentation is used (the FedASMU paper does
not mention any; documented as a reproduction assumption).
"""
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("FEDASMU_DATA_ROOT", os.path.join(ROOT, "data"))

MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def load_tensor_dataset(name: str, download: bool = False):
    """Return (x_train uint8 NCHW, y_train int64, x_test, y_test, num_classes)."""
    name = name.lower()
    cache = os.path.join(DATA_ROOT, f"{name}_uint8.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["y"], d["xt"], d["yt"], NUM_CLASSES[name]
    # Source: Hugging Face mirror of the original CIFAR archives (uoft-cs/cifar10,
    # uoft-cs/cifar100); the cs.toronto.edu mirror was ~30 KB/s on this host.
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    hf = {"cifar10": ("c10", "label"), "cifar100": ("c100", "fine_label")}[name]
    url = {"cifar10": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{s}-00000-of-00001.parquet",
           "cifar100": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{s}-00000-of-00001.parquet"}[name]

    def _read(split):
        f = os.path.join(DATA_ROOT, "hf", f"{hf[0]}_{split}.parquet")
        if not os.path.exists(f):
            if not download:
                raise FileNotFoundError(f)
            import urllib.request
            os.makedirs(os.path.dirname(f), exist_ok=True)
            urllib.request.urlretrieve(url.format(s=split), f)
        t = pq.read_table(f).to_pydict()
        imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
        return torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))), torch.tensor(t[hf[1]], dtype=torch.int64)

    x, y = _read("train")
    xt, yt = _read("test")
    os.makedirs(DATA_ROOT, exist_ok=True)
    torch.save({"x": x, "y": y, "xt": xt, "yt": yt}, cache)
    return x, y, xt, yt, NUM_CLASSES[name]


class Normalizer:
    """Converts uint8 batches to normalised float tensors on the target device."""

    def __init__(self, name: str, device):
        m, s = MEAN_STD[name.lower()]
        self.mean = torch.tensor(m, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(s, device=device).view(1, 3, 1, 1)

    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        return (xb.float().div_(255.0) - self.mean) / self.std
