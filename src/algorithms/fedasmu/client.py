"""FedASMU device-side algorithm (Algorithm 2, Eq. 4 and 7-9) + shared local SGD.

Formula 7 : w^a = (1 - beta) w^b + beta w_g          (merge fresh global model)
Formula 8 : phi = gamma/sqrt(g) * (1 - upsilon/sqrt(g-o+1)),  beta = mu_b*phi/(1+mu_b*phi)
Formula 9 : gamma/upsilon <- x - eta_x * dF_i(w_merged, zeta)/dx  with (appendix)
   dF/dgamma   = g(w_m)^T (w_g - w_loc) * mu_b/(sqrt(g)(1+mu_b phi)^2) * (1 - upsilon/sqrt(g-o+1))
   dF/dupsilon = g(w_m)^T (w_loc - w_g) * mu_b*gamma/(sqrt(g) sqrt(g-o+1) (1+mu_b phi)^2)
Reward    : R = loss(w before merge, zeta) - loss(w after merge, zeta) (same mini-batch)
Baseline  : b <- (1 - rho) b + rho R                     (Algorithm 2 line 13)
In the paper's Formula 4 one "local epoch" l is one SGD step on a mini-batch;
in our implementation a local epoch is a full pass over the local data and the
request slot l* is counted in such epochs (assumption, documented).
phi is clipped at >= 0 so that beta in [0, 1).
"""
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


def phi_value(gamma: float, upsilon: float, g: int, o: int) -> float:
    return gamma / math.sqrt(max(g, 1)) * (1.0 - upsilon / math.sqrt(g - o + 1))


def beta_value(phi: float, mu: float) -> float:
    phi = max(phi, 0.0)
    return mu * phi / (1.0 + mu * phi)


def control_gradients_device(gamma, upsilon, g, o, mu, grad_dot_wg_minus_wloc):
    """Exact partial derivatives of Formula 9 given  grad^T (w_g - w_loc)."""
    s = math.sqrt(g - o + 1)
    phi = phi_value(gamma, upsilon, g, o)
    c = mu / (math.sqrt(max(g, 1)) * (1.0 + mu * max(phi, 0.0)) ** 2)
    d_gamma = grad_dot_wg_minus_wloc * c * (1.0 - upsilon / s)
    d_upsilon = -grad_dot_wg_minus_wloc * c * gamma / s
    return d_gamma, d_upsilon


@dataclass
class DeviceHyper:
    mu_beta: float
    gamma0: float
    upsilon0: float
    eta_gamma: float
    eta_upsilon: float
    rho: float


@dataclass
class LocalConfig:
    epochs: int
    batch_size: int
    lr: float
    momentum: float = 0.0
    weight_decay: float = 0.0


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _flat_grads(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()])


@torch.no_grad()
def _batch_loss(model, xb, yb):
    was = model.training
    model.eval()
    out = F.cross_entropy(model(xb), yb).item()
    model.train(was)
    return out


def local_train(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, norm, cfg: LocalConfig,
                gen: torch.Generator, request_after: Optional[int] = None,
                request_fn: Optional[Callable[[int], Optional[tuple]]] = None,
                dev_state: Optional[dict] = None, dhp: Optional[DeviceHyper] = None, o: int = 0,
                stop_fn: Optional[Callable[[], bool]] = None):
    """Run local SGD (Formula 4).  If ``request_after`` is set, after that many local
    epochs ``request_fn(l)`` is called; it returns (w_g state_dict, g) or None.  When a
    fresher model is returned it is merged with Formula 7/8, gamma/upsilon updated
    (Formula 9), and the reward R computed.  Returns a dict of statistics.
    ``dev_state`` holds gamma/upsilon/b and is updated in place."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    n = len(x)
    steps, losses = 0, []
    stats = {"requested": False, "merged": False}
    dev = x.device
    for ep in range(1, cfg.epochs + 1):
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, cfg.batch_size):
            if stop_fn is not None and stop_fn():
                stats["aborted"] = True
                return dict(stats, steps=steps, loss=float(sum(losses) / max(len(losses), 1)))
            b = perm[i:i + cfg.batch_size]
            loss = F.cross_entropy(model(norm(x[b])), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            losses.append(loss.item())
        if request_after is not None and ep == request_after and ep < cfg.epochs and request_fn is not None:
            stats["requested"] = True
            stats["request_epoch"] = ep
            resp = request_fn(ep)
            if resp is not None:
                w_g, g = resp
                stats["fresh_version"] = g
                if g > o:
                    stats.update(_merge_fresh(model, w_g, g, o, x, y, norm, cfg, gen, dev_state, dhp))
                    # optimiser state is plain SGD (no momentum by default) -> nothing to reset
    stats.update(steps=steps, loss=float(sum(losses) / max(len(losses), 1)), last_loss=losses[-1] if losses else float("nan"))
    return stats


def _merge_fresh(model, w_g, g, o, x, y, norm, cfg, gen, st, dhp):
    """Algorithm 2 lines 8-13 (Formula 7-9 + reward)."""
    idx = torch.randperm(len(x), generator=gen)[: cfg.batch_size].to(x.device)
    xb, yb = norm(x[idx]), y[idx]
    loss_b = _batch_loss(model, xb, yb)
    gamma, ups = st["gamma"], st["upsilon"]
    phi = phi_value(gamma, ups, g, o)
    beta = beta_value(phi, dhp.mu_beta)
    sd = model.state_dict()
    w_loc_params = _flat_params(model).clone()
    merged = {}
    for k, v in sd.items():
        wg = w_g[k].to(v.device, v.dtype)
        merged[k] = (1 - beta) * v + beta * wg if v.is_floating_point() else wg
    model.load_state_dict(merged)
    # flattened fresh-model params in parameter order
    name_to_param = dict(model.named_parameters())
    w_g_params = torch.cat([w_g[k].to(w_loc_params.device, torch.float32).reshape(-1) for k in name_to_param])
    # loss after merge and gradient at the merged model on the same mini-batch (Formula 9)
    model.zero_grad(set_to_none=True)
    was = model.training
    model.eval()
    loss_a_t = F.cross_entropy(model(xb), yb)
    loss_a_t.backward()
    model.train(was)
    grad = _flat_grads(model)
    model.zero_grad(set_to_none=True)
    loss_a = loss_a_t.item()
    gdot = float(torch.dot(grad, w_g_params - w_loc_params))
    d_gamma, d_ups = control_gradients_device(gamma, ups, g, o, dhp.mu_beta, gdot)
    st["gamma"] = gamma - dhp.eta_gamma * d_gamma
    st["upsilon"] = ups - dhp.eta_upsilon * d_ups
    R = loss_b - loss_a
    st["b"] = (1 - dhp.rho) * st.get("b", 0.0) + dhp.rho * R
    return {"merged": True, "beta": beta, "phi": phi, "loss_before": loss_b, "loss_after": loss_a, "reward": R,
            "gamma": st["gamma"], "upsilon": st["upsilon"], "b": st["b"], "grad_dot": gdot}
