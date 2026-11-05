"""FedBuff on NVFLARE's official BuffModelManager vs the FedBuff paper rule."""
import numpy as np
import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.dxo import DXO, DataKind

from src.algorithms.fedbuff.manager import current_weights, make_manager, submit_update


def _mgr(variant, K, stale=True, lr=1.0):
    m = make_manager(variant, K, lr, stale)
    ctx = FLContext()
    m.initialize_model(DXO(data_kind=DataKind.WEIGHTS, data={"w": np.zeros(2, np.float32)}), ctx)
    m.generate_new_model(ctx)  # version 1 = initial weights
    return m, ctx


@pytest.mark.parametrize("variant", ["official", "paper"])
def test_buffer_waits_for_K_updates(variant):
    m, ctx = _mgr(variant, K=3, stale=False)
    assert m.current_model_version == 1
    assert not submit_update(m, {"w": np.array([1.0, 0.0])}, 1, 0, 0.0, ctx)
    assert not submit_update(m, {"w": np.array([2.0, 0.0])}, 1, 1, 0.0, ctx)
    assert submit_update(m, {"w": np.array([3.0, 3.0])}, 1, 2, 0.0, ctx)  # K-th update -> new version
    assert m.current_model_version == 2
    assert np.allclose(current_weights(m)["w"], [2.0, 1.0])  # w + mean(delta)


def test_same_device_multiple_updates_counted_separately():
    m, ctx = _mgr("official", K=2, stale=False)
    submit_update(m, {"w": np.array([1.0, 1.0])}, 1, 7, 0.0, ctx)
    assert submit_update(m, {"w": np.array([3.0, 3.0])}, 1, 7, 1.0, ctx)
    assert np.allclose(current_weights(m)["w"], [2.0, 2.0])


def test_staleness_weights_official_vs_paper():
    for variant in ("official", "paper"):
        m, ctx = _mgr(variant, K=1, stale=True)
        submit_update(m, {"w": np.zeros(2)}, 1, 0, 0.0, ctx)  # -> version 2
        submit_update(m, {"w": np.zeros(2)}, 2, 0, 0.0, ctx)  # -> version 3
        w_before = current_weights(m)["w"].copy()
        submit_update(m, {"w": np.array([1.0, 1.0])}, 1, 1, 0.0, ctx)  # trained on v1 while server at v3: tau=2
        got = current_weights(m)["w"] - w_before
        if variant == "official":
            expect = 1.0 / (1.0 + (4 - 1) ** 0.5)  # NVFLARE: uses the *new* version number 4
        else:
            expect = 1.0 / np.sqrt(1.0 + 2)  # paper: s(tau) = (1+tau)^-0.5
        assert np.allclose(got, expect), (variant, got, expect)


def test_global_lr_scales_update():
    m, ctx = _mgr("paper", K=1, stale=False, lr=0.5)
    submit_update(m, {"w": np.array([2.0, 4.0])}, 1, 0, 0.0, ctx)
    assert np.allclose(current_weights(m)["w"], [1.0, 2.0])
