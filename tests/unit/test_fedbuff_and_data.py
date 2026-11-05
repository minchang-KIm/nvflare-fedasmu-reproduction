"""FedBuff buffering with NVFLARE's official BuffModelManager vs. hand computation,
the paper-staleness variant, and the Dirichlet partition."""
import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext

from src.algorithms.fedbuff.manager import current_weights, make_manager, submit_update
from src.datasets.partition import dirichlet_partition


def _mgr(variant, K, sw, lr=1.0):
    m = make_manager(variant, K, lr, sw)
    ctx = FLContext()
    m.initialize_model(DXO(data_kind=DataKind.WEIGHTS, data={"w": np.zeros(2, np.float32)}), ctx)
    m.generate_new_model(ctx)  # version 1 = initial
    return m, ctx


def test_official_buffer_waits_for_K_and_averages():
    m, ctx = _mgr("official", K=3, sw=False)
    assert not submit_update(m, {"w": np.array([3.0, 0.0])}, 1, 0, 0.0, ctx)
    assert not submit_update(m, {"w": np.array([0.0, 3.0])}, 1, 1, 0.0, ctx)
    assert submit_update(m, {"w": np.array([3.0, 3.0])}, 1, 0, 0.0, ctx)  # same device twice counts twice
    assert m.current_model_version == 2
    np.testing.assert_allclose(current_weights(m)["w"], [2.0, 2.0])


def test_official_staleness_weight_formula():
    """NVFLARE 2.9.0: s = 1/(1 + (V_new - v)^0.5), V_new = version being generated."""
    m, ctx = _mgr("official", K=1, sw=True)
    submit_update(m, {"w": np.array([1.0, 1.0])}, 1, 0, 0.0, ctx)  # v=1, V_new=2 -> 1/(1+1)
    np.testing.assert_allclose(current_weights(m)["w"], [0.5, 0.5])
    submit_update(m, {"w": np.array([1.0, 1.0])}, 1, 0, 0.0, ctx)  # v=1, V_new=3 -> 1/(1+sqrt 2)
    np.testing.assert_allclose(current_weights(m)["w"], [0.5 + 1 / (1 + np.sqrt(2))] * 2, rtol=1e-6)


def test_paper_staleness_weight_formula():
    """FedBuff paper: s(tau) = 1/sqrt(1 + tau), tau = V_cur - v."""
    m, ctx = _mgr("paper", K=2, sw=True)
    submit_update(m, {"w": np.array([2.0, 0.0])}, 1, 0, 0.0, ctx)  # tau 0 -> 1
    assert submit_update(m, {"w": np.array([0.0, 2.0])}, 1, 1, 0.0, ctx)
    np.testing.assert_allclose(current_weights(m)["w"], [1.0, 1.0])
    submit_update(m, {"w": np.array([2.0, 2.0])}, 1, 0, 0.0, ctx)  # stale: tau = 2-1 = 1 -> 1/sqrt2
    submit_update(m, {"w": np.array([2.0, 2.0])}, 2, 1, 0.0, ctx)  # fresh -> 1
    np.testing.assert_allclose(current_weights(m)["w"], [1.0 + (2 / np.sqrt(2) + 2) / 2] * 2, rtol=1e-6)


def test_global_lr_scaling():
    m, ctx = _mgr("official", K=1, sw=False, lr=0.5)
    submit_update(m, {"w": np.array([2.0, 4.0])}, 1, 0, 0.0, ctx)
    np.testing.assert_allclose(current_weights(m)["w"], [1.0, 2.0])


def test_dirichlet_partition_is_a_partition_and_reproducible():
    y = np.repeat(np.arange(10), 100)
    p1 = dirichlet_partition(y, 20, 0.5, seed=3)
    p2 = dirichlet_partition(y, 20, 0.5, seed=3)
    allidx = np.concatenate(p1)
    assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    assert min(len(p) for p in p1) >= 10
    # label skew: devices do not all have the global label distribution
    fr = [np.bincount(y[p], minlength=10) / len(p) for p in p1]
    assert np.mean([np.abs(f - 0.1).sum() for f in fr]) > 0.3
