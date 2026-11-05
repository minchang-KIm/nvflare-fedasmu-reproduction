"""Label-skew Dirichlet partition (Li et al. 2021, NIID-Bench), as cited by FedASMU.

For each class k a proportion vector p_k ~ Dir(alpha * 1_m) is drawn and the
samples of class k are split among the m devices accordingly.  Re-sampled until
every device owns at least ``min_size`` samples (NIID-Bench procedure).
"""
import hashlib
import json
import os

import numpy as np


def dirichlet_partition(labels, n_devices: int, alpha: float, seed: int, min_size: int = 10):
    labels = np.asarray(labels)
    n_cls = int(labels.max()) + 1
    rng = np.random.RandomState(seed)
    n = len(labels)
    while True:
        idx_batch = [[] for _ in range(n_devices)]
        for k in range(n_cls):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            p = rng.dirichlet(np.repeat(alpha, n_devices))
            # NIID-Bench balancing: devices already holding >= n/m samples get no more
            p = np.array([pi * (len(b) < n / n_devices) for pi, b in zip(p, idx_batch)])
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx_k)).astype(int)[:-1]
            for b, part in zip(idx_batch, np.split(idx_k, cuts)):
                b.extend(part.tolist())
        if min(len(b) for b in idx_batch) >= min_size:
            break
    return [np.array(sorted(b), dtype=np.int64) for b in idx_batch]


def partition_digest(parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(np.asarray(p, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def save_partition(path: str, parts, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{f"d{i}": p for i, p in enumerate(parts)})
    meta = dict(meta, sha256=partition_digest(parts), sizes=[int(len(p)) for p in parts])
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_partition(path: str):
    d = np.load(path)
    return [d[f"d{i}"] for i in range(len(d.files))]
