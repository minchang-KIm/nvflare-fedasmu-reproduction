"""NVFLARE client-side Executor that runs the local training of *logical devices*.

Each NVFLARE site (one OS process in the simulator) executes, one at a time, the
local training of whichever logical device the server assigns to it.  The logical
device's private state (data partition index, FedASMU gamma/upsilon/b, Q-table,
RNG counter) lives in ``<run_dir>/device_store/device_<id>.pt`` which emulates the
device's local storage (only the executing site reads/writes it; the server never
does).  Mid-training fresh-global-model retrieval uses NVFLARE's aux channel
(``engine.send_aux_request`` -> server ``register_aux_message_handler``).
"""
import os
import time
import traceback

import numpy as np
import torch

from nvflare.apis.dxo import DataKind
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from src.algorithms.fedasmu.client import DeviceHyper, LocalConfig, local_train
from src.algorithms.fedasmu.q_learning import QTimeSlotSelector
from src.datasets.cifar import Normalizer, load_tensor_dataset
from src.datasets.partition import load_partition
from src.models import build_model

from .common import TOPIC_POLL, TOPIC_REQ, algo_flags, load_cfg, numpy_to_sd, shareable_weights, weights_shareable


class DeviceSimExecutor(Executor):
    def __init__(self, run_config: str):
        super().__init__()
        self.run_config = run_config
        self.ready = False

    def _setup(self, fl_ctx: FLContext):
        if self.ready:
            return
        cfg = self.cfg = load_cfg(self.run_config)
        site = fl_ctx.get_identity_name()
        self.site_idx = int(site.split("-")[-1]) - 1
        gpus = cfg.get("gpus", [0, 1])
        if cfg.get("use_cuda", True) and gpus:
            gpu = gpus[self.site_idx % len(gpus)]
            # select the GPU before any CUDA call -> a single CUDA context per site process
            torch.cuda.set_device(gpu)
            self.dev = torch.device(f"cuda:{gpu}")
        else:
            self.dev = torch.device("cpu")
        x, y, _, _, ncls = load_tensor_dataset(cfg["dataset"])
        # keep the full training set in host memory; only the executing device's shard
        # is copied to the GPU per task (GPU memory is shared with other jobs on this host)
        self.x, self.y = x, y
        self.parts = load_partition(cfg["partition_file"])
        self.norm = Normalizer(cfg["dataset"], self.dev)
        self.model = build_model(cfg["model"], ncls).to(self.dev)
        self.store = os.path.join(cfg["run_dir"], "device_store")
        os.makedirs(self.store, exist_ok=True)
        f = cfg.get("fedasmu", {})
        self.dhp = DeviceHyper(mu_beta=f.get("mu_beta", 1.0), gamma0=f.get("gamma0", 1.0), upsilon0=f.get("upsilon0", 0.5),
                               eta_gamma=f.get("eta_gamma", 1e-4), eta_upsilon=f.get("eta_upsilon", 1e-4), rho=f.get("rho", 0.1))
        self.ready = True
        self.log_info(fl_ctx, f"site {site} ready on {self.dev}; {len(self.parts)} logical devices")

    # ---------------- device local storage ----------------
    def _load_state(self, d: int):
        p = os.path.join(self.store, f"device_{d}.pt")
        if os.path.exists(p):
            return torch.load(p, weights_only=False)
        f = self.cfg.get("fedasmu", {})
        q = QTimeSlotSelector(self.cfg["local"]["epochs"] - 1, f.get("q_phi", 0.1), f.get("q_psi", 0.9),
                              f.get("epsilon", 0.1), seed=self.cfg["seed"] * 1000 + d)
        return {"gamma": self.dhp.gamma0, "upsilon": self.dhp.upsilon0, "b": 0.0, "n_train": 0,
                "q": q.state_dict(), "q_initialised": False}

    def _save_state(self, d: int, st: dict):
        p = os.path.join(self.store, f"device_{d}.pt")
        torch.save(st, p + ".tmp")
        os.replace(p + ".tmp", p)

    # ---------------- task ----------------
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # GPU memory on this host is shared with other jobs: retry a task that hit a CUDA
        # OOM (same inputs, same device RNG state -> same computation) instead of losing it.
        for attempt in range(self.cfg.get("oom_retries", 20) if self.ready else 20):
            try:
                self._setup(fl_ctx)
                return self._train(shareable, fl_ctx, abort_signal)
            except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and "CUBLAS_STATUS_ALLOC_FAILED" not in str(e):
                    self.log_error(fl_ctx, traceback.format_exc())
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                self.log_warning(fl_ctx, f"CUDA OOM (attempt {attempt + 1}), retrying in 10s")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                time.sleep(10)
            except Exception:
                self.log_error(fl_ctx, traceback.format_exc())
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _train(self, s: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        cfg = self.cfg
        d = s.get_header("device")
        o = s.get_header("version")
        task_id = s.get_header("task_id")
        algorithm = cfg["algorithm"]
        flags = algo_flags(algorithm)
        w0 = numpy_to_sd(shareable_weights(s), self.dev)
        self.model.load_state_dict(w0)
        idx = torch.as_tensor(self.parts[d])
        x, y = self.x[idx].to(self.dev), self.y[idx].to(self.dev)
        st = self._load_state(d)
        gen = torch.Generator().manual_seed(cfg["seed"] * 1_000_003 + d * 10_007 + st["n_train"])
        lc = LocalConfig(epochs=cfg["local"]["epochs"], batch_size=cfg["local"]["batch_size"], lr=s.get_header("lr"))

        request_after, slot_src = None, None
        f = cfg.get("fedasmu", {})
        q = QTimeSlotSelector(lc.epochs - 1, f.get("q_phi", 0.1), f.get("q_psi", 0.9), f.get("epsilon", 0.1), seed=0)
        q.load_state_dict(st["q"])
        if flags["fresh"]:
            fixed = f.get("fixed_slot")  # ablation only: 'H1' | 'H2' | 'H3' | int
            if fixed is not None:
                request_after = {"H1": 1, "H2": max(1, lc.epochs // 2), "H3": lc.epochs - 1}.get(fixed, fixed)
                slot_src = f"fixed:{fixed}"
            elif s.get_header("meta_slot") is not None:  # first training: LSTM meta model (server) decides
                request_after, slot_src = int(s.get_header("meta_slot")), "meta"
            else:  # later trainings: per-device Q-learning
                request_after, slot_src = q.select(), "q"

        engine = fl_ctx.get_engine()
        t_req = {}

        def request_fn(epoch):
            req = Shareable()
            req.set_header("device", d)
            req.set_header("task_id", task_id)
            req.set_header("epoch", epoch)
            t_req["sent"] = time.time()
            engine.send_aux_request(targets=None, topic=TOPIC_REQ, request=req, timeout=600.0, fl_ctx=fl_ctx)
            while True:  # non-blocking server: poll until the virtual-time event is processed
                if abort_signal.triggered:
                    return None
                poll = Shareable()
                poll.set_header("device", d)
                poll.set_header("task_id", task_id)
                rep = engine.send_aux_request(targets=None, topic=TOPIC_POLL, request=poll, timeout=600.0, fl_ctx=fl_ctx)
                r = next(iter(rep.values())) if rep else None
                status = r.get_header("status") if r is not None else None
                if status == "ready":
                    t_req["wait"] = time.time() - t_req["sent"]
                    if r.get_header("has_model"):
                        return numpy_to_sd(shareable_weights(r), self.dev), r.get_header("g")
                    return None
                if status == "abort":
                    return None
                time.sleep(0.02)

        stats = local_train(self.model, x, y, self.norm, lc, gen, request_after=request_after,
                            request_fn=request_fn if request_after else None, dev_state=st, dhp=self.dhp, o=o,
                            stop_fn=lambda: abort_signal.triggered)
        if stats.get("aborted"):
            return make_reply(ReturnCode.TASK_ABORTED)
        # ----- RL bookkeeping (device side) -----
        if slot_src == "meta":
            q.init_state(request_after)
            st["q_initialised"] = True
        elif slot_src == "q" and stats.get("merged"):
            stats["q_value"] = q.update(stats["reward"])
        st["q"] = q.state_dict()
        st["n_train"] += 1
        self._save_state(d, st)

        w = {k: v.detach() for k, v in self.model.state_dict().items()}
        if algorithm.startswith("fedbuff"):
            payload = {k: (w[k] - w0[k]).cpu().numpy() if w[k].is_floating_point() else np.zeros_like(w[k].cpu().numpy()) for k in w}
            kind = DataKind.WEIGHT_DIFF
        else:
            payload = {k: v.cpu().numpy() for k, v in w.items()}
            kind = DataKind.WEIGHTS
        clean = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in stats.items()
                 if isinstance(v, (int, float, bool, str, np.floating))}
        clean.update(slot_src=slot_src, request_after=request_after, n_samples=int(len(x)), site=self.site_idx,
                     req_wall_wait=t_req.get("wait"))
        return weights_shareable(payload, kind, headers={"device": d, "task_id": task_id, "version": o, "stats": clean})
