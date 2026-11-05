"""Shared helpers for the NVFLARE controller / executor."""
import hashlib
import json

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.shareable import Shareable

TASK_TRAIN = "train"
TOPIC_REQ = "fedasmu.fresh_model_request"
TOPIC_POLL = "fedasmu.fresh_model_poll"

ASYNC_FEDASMU = ("fedasmu", "fedasmu_da", "fedasmu_fa", "fedasmu_0")
FEDBUFF = ("fedbuff", "fedbuff_paper")


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def algo_flags(algorithm: str) -> dict:
    """Ablation semantics (paper Fig. 2):  DA = no dynamic server aggregation,
    FA = no fresh-global-model aggregation on devices, 0 = neither."""
    return {
        "fedasmu": dict(dynamic=True, fresh=True),
        "fedasmu_da": dict(dynamic=False, fresh=True),
        "fedasmu_fa": dict(dynamic=True, fresh=False),
        "fedasmu_0": dict(dynamic=False, fresh=False),
    }.get(algorithm, dict(dynamic=False, fresh=False))


def sd_to_numpy(sd: dict) -> dict:
    return {k: v.detach().to("cpu").numpy() for k, v in sd.items()}


def numpy_to_sd(d: dict, device) -> dict:
    return {k: torch.as_tensor(np.asarray(v)).to(device) for k, v in d.items()}


def weights_shareable(np_weights: dict, kind=DataKind.WEIGHTS, headers: dict | None = None) -> Shareable:
    s = DXO(data_kind=kind, data=np_weights).to_shareable()
    for k, v in (headers or {}).items():
        s.set_header(k, v)
    return s


def shareable_weights(s: Shareable) -> dict:
    return from_shareable(s).data


def flat_params(sd: dict, names) -> torch.Tensor:
    return torch.cat([sd[n].reshape(-1).float() for n in names])


def sd_digest(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()
