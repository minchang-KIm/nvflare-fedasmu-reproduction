"""System-heterogeneity model (virtual time).

Paper (arXiv appendix): "the training time of the slowest device is five times
longer than that of the fastest device, and the training time of each device is
independently and randomly sampled within this range."  We therefore draw a
per-device slowdown factor u_i ~ U[1, hetero] (hetero = 5) and set the virtual
time of one local epoch to ``base_epoch_time * u_i``.  Bandwidth is not given in
the paper; each device gets bw_i ~ U[bw_min, bw_max] MB/s (assumption) and a
model transfer takes ``model_MB / bw_i`` seconds.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DeviceProfile:
    device_id: int
    epoch_time: float  # virtual seconds per local epoch
    bandwidth: float  # MB / virtual second
    slowdown: float

    def transfer_time(self, model_mb: float) -> float:
        return model_mb / self.bandwidth


def make_profiles(n: int, seed: int, base_epoch_time: float, hetero: float = 5.0,
                  bw_min: float = 1.0, bw_max: float = 5.0):
    rng = np.random.RandomState(seed + 7919)
    u = rng.uniform(1.0, hetero, size=n)
    bw = rng.uniform(bw_min, bw_max, size=n)
    return [DeviceProfile(i, float(base_epoch_time * u[i]), float(bw[i]), float(u[i])) for i in range(n)]


def profiles_to_json(profiles):
    return [asdict(p) for p in profiles]
