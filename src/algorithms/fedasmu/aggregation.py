"""FedASMU server-side dynamic staleness-aware aggregation (Eq. 1-3, Algorithm 1).

Notation (paper):  xi_t^i(o) = lambda / (sqrt(t) * (t-o+1)^sigma) + iota      (Eq. 2)
                   alpha     = mu_a * xi / (1 + mu_a * xi)                      (Eq. 2)
                   w_{t+1}   = (1-alpha) w_t + alpha w_o^i                      (Eq. 1)
                   lambda/sigma/iota <- x - eta_x * dF/dx                        (Eq. 3)
The partial derivatives follow the arXiv appendix (A, "Details on the Server"):
   dF/dx ~= g_i(w_o)^T * (w_{o'}^i - w_{o-1}) * mu_a/(1+mu_a*xi_prev)^2 * dxi/dx
with w_{o'}^i the device's previous upload, w_{o-1} the global model it was
merged into (version t_prev), and the pseudo-gradient g_i(w_o) approximated from
the current upload.  Implementation choices (see docs/PAPER_REPRODUCTION.md):
 * pseudo-gradient sign: the appendix prints g ~= (w_o^i - w_o)/(eta L), which is
   the *negative* gradient of an SGD trajectory.  Default ``paper_sign=False``
   uses the mathematically consistent g = (w_o - w_o^i)/(eta * steps);
   ``paper_sign=True`` reproduces the printed sign.
 * d xi/d sigma: the appendix prints ln(sigma); the exact derivative of Eq. 2 is
   -lambda * ln(tau) / (sqrt(t) tau^sigma).  We use the exact derivative.
 * sqrt(t) with t=0 is replaced by sqrt(max(t,1)); xi is clipped at >= 0 so that
   alpha in [0,1); lambda, sigma projected to >= 0 after each update.
"""
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch


def xi_value(lam: float, sig: float, iota: float, t: int, o: int) -> float:
    tau = t - o + 1
    return lam / (math.sqrt(max(t, 1)) * tau ** sig) + iota


def alpha_value(xi: float, mu: float) -> float:
    xi = max(xi, 0.0)
    return mu * xi / (1.0 + mu * xi)


@dataclass
class DeviceServerState:
    lam: float
    sig: float
    iota: float
    # information about the device's previous accepted upload
    prev_t: int = 0  # global version the previous upload was merged into
    prev_tau: int = 1  # staleness of the previous upload
    prev_xi: float = 0.0
    history: list = field(default_factory=list)


@dataclass
class ServerHyper:
    mu_alpha: float
    lam0: float
    sig0: float
    iota0: float
    eta_lam: float
    eta_sig: float
    eta_iota: float
    tau_max: int
    dynamic: bool = True  # False -> FedASMU-DA (static control parameters)
    paper_sign: bool = False


def control_gradients(st: DeviceServerState, g_dot_prev: float, mu: float):
    """Return (dF/dlambda, dF/dsigma, dF/diota) given g^T (w_{o'}^i - w_{o-1})."""
    c = mu / (1.0 + mu * st.prev_xi) ** 2
    denom = math.sqrt(max(st.prev_t, 1)) * st.prev_tau ** st.sig
    d_lam = 1.0 / denom
    d_sig = -st.lam * math.log(st.prev_tau) / denom
    d_iota = 1.0
    return g_dot_prev * c * d_lam, g_dot_prev * c * d_sig, g_dot_prev * c * d_iota


class DeltaStore:
    """Storage of the previous update direction (w_{o'}^i - w_{o-1}) of every device.
    In RAM by default; a float32 disk memmap (``path``) for large models so that the
    server process does not need n_devices x |w| bytes of RAM."""

    def __init__(self, n_devices: int, path: Optional[str] = None):
        self.n, self.path, self.mm, self.mem = n_devices, path, None, {}
        self.has = set()

    def _ensure(self, numel):
        if self.path and self.mm is None:
            import numpy as np
            self.mm = np.lib.format.open_memmap(self.path, mode="w+", dtype="float32", shape=(self.n, numel))

    def put(self, i: int, v: torch.Tensor):
        v = v.detach().float().cpu()
        if self.path:
            self._ensure(v.numel())
            self.mm[i] = v.numpy()
        else:
            self.mem[i] = v.clone()
        self.has.add(i)

    def get(self, i: int) -> Optional[torch.Tensor]:
        if i not in self.has:
            return None
        return torch.from_numpy(self.mm[i].copy()) if self.path else self.mem[i]

    def state_dict(self):
        if self.path:
            self.mm.flush() if self.mm is not None else None
            return {"has": sorted(self.has), "path": self.path}
        return {"has": sorted(self.has), "mem": self.mem}

    def load_state_dict(self, d):
        self.has = set(d["has"])
        if self.path:
            import numpy as np
            if os.path.exists(self.path):
                self.mm = np.load(self.path, mmap_mode="r+")
        else:
            self.mem = d.get("mem", {})


class FedASMUServer:
    """Holds w_t, version t and per-device control parameters."""

    def __init__(self, hp: ServerHyper, n_devices: int, delta_path: Optional[str] = None):
        self.hp = hp
        self.t = 0
        self.dev = {i: DeviceServerState(hp.lam0, hp.sig0, hp.iota0) for i in range(n_devices)}
        self.deltas = DeltaStore(n_devices, delta_path)

    def aggregate(self, w_t: dict, w_i: dict, w_o_params: torch.Tensor, w_i_params: torch.Tensor,
                  w_t_params: torch.Tensor, device: int, o: int, lr: float, steps: int):
        """Process one upload (Algorithm 1 lines 8-15).

        w_t, w_i: state dicts (float tensors, same device) of global and uploaded model.
        *_params: flattened trainable parameters of w_o (model the device started from),
        the upload, and the current global model.
        Returns (new_state_dict or None if discarded, info dict).
        """
        hp = self.hp
        t = self.t
        tau = t - o + 1
        info = {"t": t, "o": o, "staleness": tau, "accepted": False}
        if tau > hp.tau_max:  # line 9-10
            return None, info
        st = self.dev[device]
        # values before Eq. 3 (verification trace)
        info.update(lam_before=st.lam, sig_before=st.sig, iota_before=st.iota, prev_t=st.prev_t,
                    prev_tau=st.prev_tau, prev_xi=st.prev_xi)
        prev_delta = self.deltas.get(device) if hp.dynamic else None
        if prev_delta is not None:  # line 12: Eq. 3
            g = (w_i_params - w_o_params) if hp.paper_sign else (w_o_params - w_i_params)
            g = g / (lr * max(steps, 1))
            gdot = float(torch.dot(g, prev_delta.to(g.device)))
            gl, gs, gi = control_gradients(st, gdot, hp.mu_alpha)
            st.lam = max(st.lam - hp.eta_lam * gl, 0.0)
            st.sig = max(st.sig - hp.eta_sig * gs, 0.0)
            st.iota = st.iota - hp.eta_iota * gi
            info.update(g_dot_prev=gdot, grad_lam=gl, grad_sig=gs, grad_iota=gi)
        xi = xi_value(st.lam, st.sig, st.iota, t, o)  # line 13: Eq. 2
        a = alpha_value(xi, hp.mu_alpha)
        new = {k: (1 - a) * w_t[k] + a * w_i[k] if w_t[k].is_floating_point() else w_i[k] for k in w_t}  # Eq. 1
        if hp.dynamic:
            self.deltas.put(device, w_i_params - w_t_params)
        st.prev_t, st.prev_tau, st.prev_xi = max(t, 1), tau, xi
        self.t = t + 1
        info.update(accepted=True, alpha=a, xi=xi, lam=st.lam, sig=st.sig, iota=st.iota)
        return new, info

    # ---- checkpointing -------------------------------------------------
    def state_dict(self):
        return {"t": self.t, "dev": {i: dict(vars(s)) for i, s in self.dev.items()}, "deltas": self.deltas.state_dict()}

    def load_state_dict(self, d):
        self.t = d["t"]
        self.dev = {int(i): DeviceServerState(**s) for i, s in d["dev"].items()}
        self.deltas.load_state_dict(d["deltas"])
