import numpy as np

from src.datasets.partition import dirichlet_partition, partition_digest
from src.scheduling.device_profiles import make_profiles


def test_partition_is_disjoint_complete_and_reproducible():
    y = np.repeat(np.arange(10), 100)
    p1 = dirichlet_partition(y, 20, 0.5, seed=3)
    p2 = dirichlet_partition(y, 20, 0.5, seed=3)
    allidx = np.concatenate(p1)
    assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)
    assert partition_digest(p1) == partition_digest(p2)
    assert min(len(p) for p in p1) >= 10


def test_profiles_heterogeneity_bound():
    pr = make_profiles(100, 0, base_epoch_time=10.0, hetero=5.0)
    et = np.array([p.epoch_time for p in pr])
    assert et.min() >= 10.0 and et.max() <= 50.0 and et.max() / et.min() > 3
