"""Server-side FedASMU formulas (Eq. 1-3, Algorithm 1) checked against hand computation
and finite differences."""
import math

import pytest
import torch

from src.algorithms.fedasmu.aggregation import (DeltaStore, DeviceServerState, FedASMUServer, ServerHyper,
                                                alpha_value, control_gradients, xi_value)


def hp(**kw):
    d = dict(mu_alpha=10.0, lam0=1.0, sig0=0.5, iota0=0.01, eta_lam=0.1, eta_sig=0.1, eta_iota=0.1, tau_max=5)
    d.update(kw)
    return ServerHyper(**d)


def test_xi_alpha_hand_values():
    # t=4, o=2 -> staleness 3 ; xi = 1/(sqrt(4)*3^0.5) + 0.01
    xi = xi_value(1.0, 0.5, 0.01, 4, 2)
    assert xi == pytest.approx(1 / (2 * math.sqrt(3)) + 0.01)
    assert alpha_value(xi, 10.0) == pytest.approx(10 * xi / (1 + 10 * xi))
    assert alpha_value(-1.0, 10.0) == 0.0  # xi clipped at 0


def test_aggregate_formula1_and_versioning():
    s = FedASMUServer(hp(), 3)
    w_t = {"w": torch.zeros(4)}
    w_i = {"w": torch.ones(4)}
    new, info = s.aggregate(w_t, w_i, torch.zeros(4), torch.ones(4), torch.zeros(4), device=0, o=0, lr=0.1, steps=1)
    a = alpha_value(xi_value(1.0, 0.5, 0.01, 0, 0), 10.0)  # t=0 -> sqrt(max(t,1))
    assert torch.allclose(new["w"], torch.full((4,), a))
    assert info["staleness"] == 1 and s.t == 1 and info["accepted"]


def test_staleness_bound_discards():
    s = FedASMUServer(hp(tau_max=3), 1)
    s.t = 10
    w = {"w": torch.zeros(2)}
    new, info = s.aggregate(w, w, torch.zeros(2), torch.zeros(2), torch.zeros(2), 0, o=7, lr=0.1, steps=1)
    assert new is None and info["staleness"] == 4 and s.t == 10
    new, info = s.aggregate(w, w, torch.zeros(2), torch.zeros(2), torch.zeros(2), 0, o=8, lr=0.1, steps=1)
    assert new is not None and info["staleness"] == 3


@pytest.mark.parametrize("which", ["lam", "sig", "iota"])
def test_control_gradient_matches_finite_difference(which):
    """dF/dx for F(w_o) = g^T w_o with w_o = w_prev + alpha(x) (w' - w_prev)."""
    torch.manual_seed(0)
    g = torch.randn(5, dtype=torch.float64)
    delta = torch.randn(5, dtype=torch.float64)
    st = DeviceServerState(lam=0.8, sig=0.6, iota=0.02, prev_t=9, prev_tau=4)
    mu = 3.0
    st.prev_xi = xi_value(st.lam, st.sig, st.iota, st.prev_t, st.prev_t - st.prev_tau + 1)

    def F(x):
        p = dict(lam=st.lam, sig=st.sig, iota=st.iota)
        p[which] = x
        xi = xi_value(p["lam"], p["sig"], p["iota"], st.prev_t, st.prev_t - st.prev_tau + 1)
        return float(torch.dot(g, alpha_value(xi, mu) * delta))

    x0 = getattr(st, which)
    eps = 1e-6
    fd = (F(x0 + eps) - F(x0 - eps)) / (2 * eps)
    gl, gs, gi = control_gradients(st, float(torch.dot(g, delta)), mu)
    assert {"lam": gl, "sig": gs, "iota": gi}[which] == pytest.approx(fd, rel=1e-4, abs=1e-8)


def test_dynamic_update_direction_and_static_mode():
    for dynamic in (True, False):
        s = FedASMUServer(hp(dynamic=dynamic, eta_lam=1.0), 1)
        z = torch.zeros(3)
        w_t = {"w": z.clone()}
        # first upload of device 0 moves along +e
        e = torch.tensor([1.0, 0.0, 0.0])
        new, _ = s.aggregate(w_t, {"w": e}, z, e, z, 0, o=0, lr=1.0, steps=1)
        lam_before = s.dev[0].lam
        # second upload again moves along +e from its start point -> aligned with previous direction:
        # pseudo-gradient g = (w_o - w_i)/(lr*steps) = -e, g.delta_prev < 0 -> lambda increases
        w_o = new["w"].clone()
        _, info = s.aggregate(new, {"w": w_o + e}, w_o, w_o + e, new["w"], 0, o=1, lr=1.0, steps=1)
        if dynamic:
            assert info["g_dot_prev"] < 0 and s.dev[0].lam > lam_before
        else:
            assert s.dev[0].lam == lam_before and "g_dot_prev" not in info


def test_state_roundtrip_with_memmap(tmp_path):
    s = FedASMUServer(hp(), 2, delta_path=str(tmp_path / "d.npy"))
    z = torch.zeros(3)
    s.aggregate({"w": z}, {"w": torch.ones(3)}, z, torch.ones(3), z, 1, o=0, lr=1.0, steps=1)
    sd = s.state_dict()
    s2 = FedASMUServer(hp(), 2, delta_path=str(tmp_path / "d.npy"))
    s2.load_state_dict(sd)
    assert s2.t == 1 and torch.allclose(s2.deltas.get(1), torch.ones(3)) and s2.deltas.get(0) is None


def test_delta_store_ram():
    d = DeltaStore(3)
    d.put(2, torch.arange(4.0))
    assert d.get(1) is None and torch.equal(d.get(2), torch.arange(4.0))
