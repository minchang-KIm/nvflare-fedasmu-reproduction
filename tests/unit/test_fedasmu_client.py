"""Device-side FedASMU (Algorithm 2, Eq. 7-9): merge, beta, control gradients, reward,
real mid-training request inside local SGD."""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algorithms.fedasmu.client import (DeviceHyper, LocalConfig, beta_value, control_gradients_device,
                                           local_train, phi_value)

DHP = DeviceHyper(mu_beta=2.0, gamma0=1.0, upsilon0=0.5, eta_gamma=0.1, eta_upsilon=0.1, rho=0.5)


def test_phi_beta_hand_values():
    phi = phi_value(1.0, 0.5, g=9, o=5)  # 1/3 * (1 - 0.5/sqrt(5))
    assert phi == pytest.approx((1 / 3) * (1 - 0.5 / math.sqrt(5)))
    assert beta_value(phi, 2.0) == pytest.approx(2 * phi / (1 + 2 * phi))
    assert beta_value(-0.3, 2.0) == 0.0


@pytest.mark.parametrize("which", ["gamma", "upsilon"])
def test_device_control_gradient_vs_autograd(which):
    torch.manual_seed(1)
    lin = nn.Linear(4, 3)
    xb, yb = torch.randn(8, 4), torch.randint(0, 3, (8,))
    w_loc = torch.cat([p.detach().reshape(-1) for p in lin.parameters()])
    w_g = w_loc + 0.3 * torch.randn_like(w_loc)
    g_ver, o, mu = 7, 3, 2.0
    gamma = torch.tensor(1.2, requires_grad=which == "gamma")
    ups = torch.tensor(0.4, requires_grad=which == "upsilon")
    phi = gamma / math.sqrt(g_ver) * (1 - ups / math.sqrt(g_ver - o + 1))
    beta = mu * phi / (1 + mu * phi)
    w_m = w_loc + beta * (w_g - w_loc)
    W, b = w_m[:12].view(3, 4), w_m[12:]
    loss = F.cross_entropy(xb @ W.T + b, yb)
    loss.backward()
    auto = (gamma if which == "gamma" else ups).grad.item()
    # analytic: gradient at merged model dotted with (w_g - w_loc)
    w_m2 = w_m.detach().clone().requires_grad_(True)
    F.cross_entropy(xb @ w_m2[:12].view(3, 4).T + w_m2[12:], yb).backward()
    gdot = float(torch.dot(w_m2.grad, w_g - w_loc))
    dg, du = control_gradients_device(1.2, 0.4, g_ver, o, mu, gdot)
    assert {"gamma": dg, "upsilon": du}[which] == pytest.approx(auto, rel=1e-5)


def _data():
    torch.manual_seed(0)
    x = torch.randn(40, 4)
    y = (x[:, 0] > 0).long()
    return x, y


def test_mid_training_request_merges_fresh_model_formula7():
    x, y = _data()
    m = nn.Linear(4, 2)
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    w_g = {k: v + 1.0 for k, v in w0.items()}
    calls = []

    def req(epoch):
        calls.append(epoch)
        return w_g, 6

    st = {"gamma": 1.0, "upsilon": 0.5, "b": 0.0}
    # lr = 0 -> SGD leaves weights unchanged, so the final model is exactly the Eq. 7 merge
    stats = local_train(m, x, y, lambda t: t, LocalConfig(epochs=3, batch_size=10, lr=0.0), torch.Generator().manual_seed(0),
                        request_after=1, request_fn=req, dev_state=st, dhp=DHP, o=2)
    beta = beta_value(phi_value(1.0, 0.5, 6, 2), DHP.mu_beta)
    assert calls == [1] and stats["merged"] and stats["beta"] == pytest.approx(beta)
    for k in w0:
        assert torch.allclose(m.state_dict()[k], (1 - beta) * w0[k] + beta * w_g[k], atol=1e-6)
    assert stats["reward"] == pytest.approx(stats["loss_before"] - stats["loss_after"])
    assert st["b"] == pytest.approx(DHP.rho * stats["reward"])
    assert st["gamma"] != 1.0 or st["upsilon"] != 0.5  # Formula 9 applied
    assert stats["steps"] == 3 * 4  # training continued after the merge


def test_request_without_newer_version_does_not_merge():
    x, y = _data()
    m = nn.Linear(4, 2)
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    st = {"gamma": 1.0, "upsilon": 0.5, "b": 0.0}
    stats = local_train(m, x, y, lambda t: t, LocalConfig(epochs=2, batch_size=10, lr=0.0), torch.Generator().manual_seed(0),
                        request_after=1, request_fn=lambda e: ({k: v + 5 for k, v in w0.items()}, 2), dev_state=st,
                        dhp=DHP, o=2)
    assert stats["requested"] and not stats["merged"]
    assert all(torch.equal(m.state_dict()[k], w0[k]) for k in w0)


def test_local_sgd_reduces_loss():
    x, y = _data()
    m = nn.Linear(4, 2)
    l0 = F.cross_entropy(m(x), y).item()
    local_train(m, x, y, lambda t: t, LocalConfig(epochs=5, batch_size=10, lr=0.5), torch.Generator().manual_seed(0))
    assert F.cross_entropy(m(x), y).item() < l0
