"""NVFLARE server-side Controller: virtual-time asynchronous FL over logical devices.

One controller implements the *system model* shared by every algorithm (paper
Fig. 1 / Algorithm 1 lines 1-6): every ``trigger_period`` virtual seconds the
server triggers idle logical devices (random, up to ``max_concurrency`` training
at the same time) and sends them the current global model as an NVFLARE task.
Local training is really executed (PyTorch, GPU) by the NVFLARE sites; the
*virtual* completion time of each device follows its heterogeneity profile.

Causality is enforced with a conservative discrete-event loop: events
(TRIGGER, REQUEST, ARRIVAL) are processed in virtual-time order; an ARRIVAL is
processed only after the real result has been received, and the loop never
advances past the earliest possible fresh-model request of a running device
whose request has not reached the server yet.  Because every concurrently
training device occupies its own site (#sites == max_concurrency), this is
deadlock free.  Real wall-clock time and virtual time are logged separately.

Algorithms (selected by run-config ``algorithm``):
  fedasmu, fedasmu_da, fedasmu_fa, fedasmu_0   (FedASMU and its ablations)
  fedbuff (NVFLARE official BuffModelManager), fedbuff_paper (paper staleness fn)
  fedavg (synchronous rounds of m' devices; ablation reference)
"""
import csv
import heapq
import json
import os
import pickle
import random
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from src.algorithms.fedasmu.aggregation import FedASMUServer, ServerHyper
from src.algorithms.fedasmu.meta_model import MetaController
from src.algorithms.fedbuff.manager import current_weights, make_manager, submit_update
from src.datasets.cifar import Normalizer, load_tensor_dataset
from src.datasets.partition import load_partition
from src.models import build_model
from src.scheduling.device_profiles import DeviceProfile

from .common import (ASYNC_FEDASMU, FEDBUFF, TASK_TRAIN, TOPIC_POLL, TOPIC_REQ, algo_flags, flat_params, load_cfg,
                     numpy_to_sd, sd_to_numpy, shareable_weights, weights_shareable)

TRIGGER, REQUEST, ARRIVAL = 0, 1, 2  # tie order at equal virtual time

# Verification trace (metrics_dir/trace.csv): one row per received update with the inputs and
# outputs of every FedASMU formula, so that scripts/verify_fedasmu.py can recompute each formula
# from the logged inputs and compare it with what the run actually did.
TRACE_FIELDS = [
    "received", "vtime", "device", "algorithm", "t", "o", "staleness", "accepted",
    # server, Eq. 2-3 and Eq. 1
    "lam_before", "sig_before", "iota_before", "prev_t", "prev_tau", "prev_xi", "g_dot_prev",
    "grad_lam", "grad_sig", "grad_iota", "lam", "sig", "iota", "xi", "alpha", "eq1_ratio", "eq1_resid", "eq1_bound",
    # device, Formula 7-9 and reward
    "slot_src", "request_after", "fresh_version", "merged", "gamma_before", "upsilon_before", "phi", "beta",
    "grad_dot", "d_gamma", "d_upsilon", "gamma", "upsilon", "eq7_ratio", "eq7_resid", "eq7_bound", "loss_before",
    "loss_after", "reward", "b",
    # device Q-learning, Eq. 6
    "q_prev_state", "q_action", "q_next_state", "q_greedy", "q_h_before", "q_max_next", "q_h_after",
    # server LSTM meta model, Eq. 5
    "meta_feat", "meta_probs", "meta_decisions", "meta_slot", "meta_reward", "meta_baseline_before", "meta_adv",
    "meta_lp_before", "meta_lp_after",
    # FedBuff (NVFLARE BuffModelManager)
    "fb_generated", "fb_model_version"]


class _Run:
    __slots__ = ("device", "site", "task_id", "t0", "o", "down", "w_o_params", "req_epoch", "req_time",
                 "reply", "result", "arrival_pushed", "meta", "lr", "wall_start")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class AsyncSimController(Controller):
    def __init__(self, run_config: str):
        super().__init__()
        self.run_config = run_config

    # ------------------------------------------------------------------ setup
    def start_controller(self, fl_ctx: FLContext):
        cfg = self.cfg = load_cfg(self.run_config)
        self.algorithm = cfg["algorithm"]
        self.flags = algo_flags(self.algorithm)
        # The server (aggregation + evaluation) runs on CPU by default: GPU memory on this
        # host is shared with other jobs and is reserved for the 10 training sites.
        sg = cfg.get("server_gpu")
        self.gdev = torch.device(f"cuda:{sg}" if sg is not None and torch.cuda.is_available() else "cpu")
        if self.gdev.type == "cuda":
            torch.cuda.set_device(self.gdev)
        else:
            torch.set_num_threads(cfg.get("server_threads", 12))
        _, _, xt, yt, ncls = load_tensor_dataset(cfg["dataset"])
        self.xt, self.yt = xt.to(self.gdev), yt.to(self.gdev)
        self.norm = Normalizer(cfg["dataset"], self.gdev)
        self.model = build_model(cfg["model"], ncls, cfg["dataset"]).to(self.gdev)
        self.param_names = [n for n, _ in self.model.named_parameters()]
        init = torch.load(cfg["init_weights"], map_location=self.gdev)
        self.global_sd = {k: v.to(self.gdev) for k, v in init.items()}
        self.model_mb = sum(v.numel() * 4 for v in init.values()) / 2**20
        self.parts_sizes = [len(p) for p in load_partition(cfg["partition_file"])]
        self.profiles = [DeviceProfile(**p) for p in cfg["profiles"]]
        self.n_dev = len(self.profiles)
        self.metrics_dir = cfg["metrics_dir"]
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.ckpt_dir = cfg["checkpoint_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.rng = random.Random(cfg["seed"])
        self.lock = threading.Condition()
        self.heap, self.seq = [], 0
        self.running: dict[int, _Run] = {}
        self.free_sites = []
        self.received = self.accepted = self.discarded = self.failed = 0
        self.nonfinite = 0
        self.total_steps = self.total_samples = 0
        self.vtime = 0.0
        self.done = False
        self.task_counter = 0
        self.wall0 = time.time()
        self.next_trigger = 0.0
        # algorithm state
        f = cfg.get("fedasmu", {})
        if self.algorithm in ASYNC_FEDASMU:
            self.server = FedASMUServer(ServerHyper(
                mu_alpha=f["mu_alpha"], lam0=f["lambda0"], sig0=f["sigma0"], iota0=f["iota0"], eta_lam=f["eta_lambda"],
                eta_sig=f["eta_sigma"], eta_iota=f["eta_iota"], tau_max=cfg["staleness_bound"],
                dynamic=self.flags["dynamic"], paper_sign=f.get("paper_sign", False)), self.n_dev,
                delta_path=os.path.join(self.ckpt_dir, "prev_deltas.npy") if self.model_mb > 8 else None)
            self.meta = MetaController(cfg["local"]["epochs"] - 1, f["eta_rl"], f.get("rho", 0.1), cfg["seed"],
                                       f.get("lstm_hidden", 32))
            self.meta_done = set()
            self.meta_history = []
            if f.get("meta_pretrain_file") and os.path.exists(f["meta_pretrain_file"]):
                with open(f["meta_pretrain_file"]) as fh:
                    hist = json.load(fh)
                self.meta_pretrained = self.meta.pretrain(hist)
            else:
                self.meta_pretrained = 0
        elif self.algorithm in FEDBUFF:
            b = cfg["fedbuff"]
            self.mgr = make_manager("official" if self.algorithm == "fedbuff" else "paper", b["K"],
                                    b.get("global_lr", 1.0), b.get("staleness_weight", True))
            from nvflare.apis.dxo import DXO, DataKind
            self.mgr.initialize_model(DXO(data_kind=DataKind.WEIGHTS, data=sd_to_numpy(self.global_sd)), fl_ctx)
            self.mgr.generate_new_model(fl_ctx)  # version 1 == initial weights (NVFLARE convention)
        self.version = 0  # number of global model updates
        self.aggregations = 0
        engine = fl_ctx.get_engine()
        engine.register_aux_message_handler(TOPIC_REQ, self._on_request)
        engine.register_aux_message_handler(TOPIC_POLL, self._on_poll)
        if not (cfg.get("resume", True) and os.path.exists(os.path.join(self.ckpt_dir, "server_state.pkl"))):
            # fresh start (e.g. re-run after a crash before the first checkpoint): discard partial logs
            for name in ("updates.csv", "eval.csv", "trace.csv"):
                p = os.path.join(self.metrics_dir, name)
                if os.path.exists(p):
                    os.replace(p, p + ".discarded")
        self._open_logs()
        self.resumed = self._maybe_resume(fl_ctx)

    def _open_logs(self):
        mode = "a" if os.path.exists(os.path.join(self.metrics_dir, "updates.csv")) else "w"
        self.f_upd = open(os.path.join(self.metrics_dir, "updates.csv"), mode, newline="")
        self.w_upd = csv.DictWriter(self.f_upd, fieldnames=[
            "received", "vtime", "wall", "device", "site", "o", "global_version", "aggregations", "staleness",
            "accepted", "alpha", "beta", "request_after", "slot_src", "fresh_version", "merged", "reward",
            "loss_before", "loss_after", "local_loss", "steps", "lr", "lam", "sig", "iota", "gamma", "upsilon",
            "req_wall_wait", "train_wall"], extrasaction="ignore")
        if mode == "w":
            self.w_upd.writeheader()
        mode = "a" if os.path.exists(os.path.join(self.metrics_dir, "eval.csv")) else "w"
        self.f_ev = open(os.path.join(self.metrics_dir, "eval.csv"), mode, newline="")
        self.w_ev = csv.DictWriter(self.f_ev, fieldnames=[
            "received", "accepted", "global_version", "aggregations", "vtime", "wall", "test_acc", "test_loss",
            "total_steps", "total_samples", "running"])
        if mode == "w":
            self.w_ev.writeheader()
        mode = "a" if os.path.exists(os.path.join(self.metrics_dir, "trace.csv")) else "w"
        self.f_tr = open(os.path.join(self.metrics_dir, "trace.csv"), mode, newline="")
        self.w_tr = csv.DictWriter(self.f_tr, fieldnames=TRACE_FIELDS, extrasaction="ignore")
        if mode == "w":
            self.w_tr.writeheader()

    # -------------------------------------------------------------- helpers
    def _push(self, t, kind, payload):
        heapq.heappush(self.heap, (t, kind, self.seq, payload))
        self.seq += 1

    def _lr(self):
        lc = self.cfg["local"]
        return lc["lr"] * lc.get("lr_decay", 1.0) ** self.received

    def _global_numpy(self):
        if self.algorithm in FEDBUFF:
            return current_weights(self.mgr)
        return sd_to_numpy(self.global_sd)

    def _current_version(self):
        if self.algorithm in FEDBUFF:
            return self.mgr.current_model_version
        return self.server.t if self.algorithm in ASYNC_FEDASMU else self.version

    @torch.no_grad()
    def _evaluate(self):
        sd = numpy_to_sd(current_weights(self.mgr), self.gdev) if self.algorithm in FEDBUFF else self.global_sd
        self.model.load_state_dict(sd)
        self.model.eval()
        correct, loss = 0, 0.0
        bs = 250
        for i in range(0, len(self.xt), bs):
            out = self.model(self.norm(self.xt[i:i + bs]))
            loss += F.cross_entropy(out, self.yt[i:i + bs], reduction="sum").item()
            correct += (out.argmax(1) == self.yt[i:i + bs]).sum().item()
        return correct / len(self.xt), loss / len(self.xt)

    def _log_eval(self):
        acc, loss = self._evaluate()
        row = dict(received=self.received, accepted=self.accepted, global_version=self.version,
                   aggregations=self.aggregations, vtime=round(self.vtime, 3), wall=round(time.time() - self.wall0, 2),
                   test_acc=acc, test_loss=loss, total_steps=self.total_steps, total_samples=self.total_samples,
                   running=len(self.running))
        self.w_ev.writerow(row)
        self.f_ev.flush()
        return acc

    # ---------------------------------------------------------- aux handlers
    def _on_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Device asks for a fresh global model (Algorithm 2 line 5).  The request
        becomes a virtual-time REQUEST event; the reply is delivered via polling."""
        d, tid, ep = request.get_header("device"), request.get_header("task_id"), request.get_header("epoch")
        rep = Shareable()
        with self.lock:
            r = self.running.get(d)
            if r is None or r.task_id != tid or self.done:
                rep.set_header("status", "abort")
                return rep
            if r.req_epoch is None:  # a retried task (after CUDA OOM) re-sends the same request
                p = self.profiles[d]
                r.req_epoch = ep
                r.req_time = r.t0 + r.down + ep * p.epoch_time
                self._push(r.req_time, REQUEST, d)
                self.lock.notify_all()
        rep.set_header("status", "pending")
        return rep

    def _on_poll(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        d, tid = request.get_header("device"), request.get_header("task_id")
        with self.lock:
            r = self.running.get(d)
            if r is None or r.task_id != tid or self.done:
                rep = Shareable()
                rep.set_header("status", "abort")
                return rep
            if r.reply is not None:  # kept until the arrival (idempotent for retried tasks)
                return r.reply
        rep = Shareable()
        rep.set_header("status", "pending")
        return rep

    def _on_result(self, client_task: ClientTask, fl_ctx: FLContext):
        res = client_task.result
        d = client_task.task.props["device"]
        tid = client_task.task.props["task_id"]
        with self.lock:
            r = self.running.get(d)
            if r is not None and r.task_id == tid:
                r.result = res if res is not None else Shareable()
                self.lock.notify_all()
        client_task.result = None  # free memory held by NVFLARE task bookkeeping

    # -------------------------------------------------------------- dispatch
    def _dispatch(self, d: int, t: float, fl_ctx: FLContext):
        site = self.free_sites.pop(0)
        self.task_counter += 1
        tid = f"t{self.task_counter}"
        p = self.profiles[d]
        down = p.transfer_time(self.model_mb)
        o = self._current_version()
        lr = self._lr()
        headers = {"device": d, "version": o, "task_id": tid, "lr": lr}
        meta = None
        if self.algorithm in ASYNC_FEDASMU and self.flags["fresh"] and d not in self.meta_done \
                and self.cfg["fedasmu"].get("fixed_slot") is None:
            feat = [p.slowdown / self.cfg["hetero"], self.parts_sizes[d] / (sum(self.parts_sizes) / self.n_dev) / 4.0]
            probs = self.meta.model.probs(feat)  # distribution over l* before sampling (trace)
            slot, decisions = self.meta.decide(feat)
            headers["meta_slot"] = slot
            meta = (feat, decisions, slot, probs)
        data = weights_shareable(self._global_numpy(), headers=headers)
        w_o_params = flat_params(self.global_sd, self.param_names).clone().cpu() if self.algorithm in ASYNC_FEDASMU else None
        run = _Run(device=d, site=site, task_id=tid, t0=t, o=o, down=down, w_o_params=w_o_params, meta=meta, lr=lr,
                   arrival_pushed=False, wall_start=time.time())
        self.running[d] = run
        if not (self.algorithm in ASYNC_FEDASMU and self.flags["fresh"]):
            E = self.cfg["local"]["epochs"]
            self._push(t + down + E * p.epoch_time + down, ARRIVAL, (d, tid))
            run.arrival_pushed = True
        task = Task(name=TASK_TRAIN, data=data, props={"device": d, "task_id": tid}, timeout=0,
                    result_received_cb=self._on_result)
        self.send(task=task, fl_ctx=fl_ctx, targets=[site])

    # ------------------------------------------------------------ main loop
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            self.free_sites = sorted(c.name for c in fl_ctx.get_engine().get_clients())
            assert len(self.free_sites) >= self.cfg["max_concurrency"], "need one site per concurrent device"
            self.free_sites = self.free_sites[: self.cfg["max_concurrency"]]
            if not self.resumed:
                self._log_eval()
            if self.algorithm == "fedavg":
                self._fedavg_loop(abort_signal, fl_ctx)
            else:
                self._async_loop(abort_signal, fl_ctx)
            self._finish("COMPLETED")
        except Exception:
            self.log_error(fl_ctx, traceback.format_exc())
            self._finish("FAILED", traceback.format_exc())
            raise
        finally:
            with self.lock:
                self.done = True
                self.lock.notify_all()
            self.cancel_all_tasks(fl_ctx=fl_ctx)

    def _next_event(self, abort_signal):
        """Return the next event that is safe to process (blocking)."""
        with self.lock:
            while True:
                if abort_signal.triggered:
                    return None
                # earliest possible not-yet-announced fresh-model request
                lb = None
                if self.algorithm in ASYNC_FEDASMU and self.flags["fresh"]:
                    for r in self.running.values():
                        if r.req_epoch is None and r.result is None:
                            t_lb = r.t0 + r.down + self.profiles[r.device].epoch_time
                            lb = t_lb if lb is None else min(lb, t_lb)
                        elif r.req_epoch is None and r.result is not None and not r.arrival_pushed:
                            # finished without requesting (e.g. aborted request) -> plain arrival
                            p = self.profiles[r.device]
                            self._push(r.t0 + r.down + self.cfg["local"]["epochs"] * p.epoch_time + r.down,
                                       ARRIVAL, (r.device, r.task_id))
                            r.arrival_pushed = True
                if self.heap and (lb is None or self.heap[0][0] < lb):
                    t, kind, _, payload = self.heap[0]
                    if kind == ARRIVAL:
                        d, tid = payload
                        r = self.running.get(d)
                        if r is not None and r.task_id == tid and r.result is None:
                            self.lock.wait(timeout=1.0)  # wait for real training to finish
                            continue
                    heapq.heappop(self.heap)
                    return t, kind, payload
                self.lock.wait(timeout=1.0)

    def _async_loop(self, abort_signal, fl_ctx):
        budget = self.cfg["budget_updates"]
        self._push(self.next_trigger, TRIGGER, None)
        while self.received < budget:
            ev = self._next_event(abort_signal)
            if ev is None:
                break
            t, kind, payload = ev
            self.vtime = max(self.vtime, t)
            if kind == TRIGGER:
                with self.lock:
                    idle = [d for d in range(self.n_dev) if d not in self.running]
                    n_new = min(self.cfg["max_concurrency"] - len(self.running), len(idle),
                                self.cfg.get("devices_per_trigger", 10**9))
                    for d in self.rng.sample(idle, n_new) if n_new > 0 else []:
                        self._dispatch(d, t, fl_ctx)
                    self.next_trigger = t + self.cfg["trigger_period"]
                    self._push(self.next_trigger, TRIGGER, None)
            elif kind == REQUEST:
                self._process_request(payload, t)
            else:
                self._process_arrival(payload, t, fl_ctx)

    def _process_request(self, d, t):
        """Server side of Algorithm 2 lines 5-6 / Fig. 1 steps 3-4 at virtual time t."""
        with self.lock:
            r = self.running.get(d)
            if r is None:
                return
            p = self.profiles[d]
            g = self._current_version()
            rep = Shareable()
            if g > r.o:  # send w_g only if newer than w_o
                rep = weights_shareable(self._global_numpy())
                rep.set_header("has_model", True)
                extra = self.cfg.get("request_latency", 0.1) + r.down
            else:
                rep.set_header("has_model", False)
                extra = self.cfg.get("request_latency", 0.1)
            rep.set_header("status", "ready")
            rep.set_header("g", g)
            r.reply = rep
            E = self.cfg["local"]["epochs"]
            self._push(t + extra + (E - r.req_epoch) * p.epoch_time + r.down, ARRIVAL, (d, r.task_id))
            r.arrival_pushed = True
            self.lock.notify_all()

    def _process_arrival(self, payload, t, fl_ctx):
        d, tid = payload
        with self.lock:
            r = self.running.get(d)
            if r is None or r.task_id != tid:
                return
            del self.running[d]
            self.free_sites.append(r.site)
        res = r.result
        rc = res.get_return_code(ReturnCode.OK)
        row = dict(received=self.received + 1, vtime=round(t, 3), wall=round(time.time() - self.wall0, 2), device=d,
                   site=r.site, o=r.o, lr=r.lr, train_wall=round(time.time() - r.wall_start, 2))
        self.received += 1
        if rc != ReturnCode.OK:
            self.failed += 1
            row.update(accepted=False, slot_src=f"failed:{rc}")
            self.w_upd.writerow(row)
            return
        stats = res.get_header("stats") or {}
        row.update({k: stats.get(k) for k in ("beta", "request_after", "slot_src", "fresh_version", "merged",
                                               "reward", "loss_before", "loss_after", "gamma", "upsilon",
                                               "req_wall_wait")})
        row.update(local_loss=stats.get("loss"), steps=stats.get("steps"))
        self.total_steps += int(stats.get("steps", 0))
        self.total_samples += int(stats.get("n_samples", 0)) * self.cfg["local"]["epochs"]
        w_i = shareable_weights(res)
        if not all(np.isfinite(v).all() for v in w_i.values() if np.issubdtype(v.dtype, np.floating)):
            # diverged local training (NaN/Inf weights): dropped like a failed device, for every algorithm,
            # so that one diverged device cannot destroy the global model (pilot VGG-11/CIFAR-100 FedASMU).
            self.failed += 1
            self.nonfinite += 1
            row.update(accepted=False, slot_src="rejected:nonfinite", global_version=self.version,
                       aggregations=self.aggregations)
            self.w_upd.writerow(row)
            if self.cfg.get("trace", True):  # the device state still changed during this training
                tr = dict(received=row["received"], vtime=row["vtime"], device=d, algorithm=self.algorithm, o=r.o)
                tr.update({k: stats.get(k) for k in TRACE_FIELDS if k in stats})
                tr.update(accepted=False, slot_src="rejected:nonfinite")
                self.w_tr.writerow(tr)
            return
        tr = dict(received=row["received"], vtime=row["vtime"], device=d, algorithm=self.algorithm, o=r.o)
        tr.update({k: stats.get(k) for k in TRACE_FIELDS if k in stats})
        if self.algorithm in ASYNC_FEDASMU:
            w_i_sd = numpy_to_sd(w_i, self.gdev)
            wi_flat = flat_params(w_i_sd, self.param_names)
            wt_flat = flat_params(self.global_sd, self.param_names)
            new, info = self.server.aggregate(self.global_sd, w_i_sd, r.w_o_params, wi_flat, wt_flat, d, r.o, r.lr,
                                              int(stats.get("steps", 1)))
            tr.update({k: v for k, v in info.items() if k in TRACE_FIELDS})
            if new is not None:
                # realised mixing ratio of Eq. 1 measured on the actual tensors (must equal alpha)
                nw = flat_params(new, self.param_names)
                # norms in float64 of the float32 tensors; the stored model itself is float32, so the realised
                # ratio can differ from alpha by at most the rounding of w_{t+1}: 2^-24 |w_{t+1}| / |w_i - w_t|
                den = float(torch.linalg.vector_norm(wi_flat.double() - wt_flat.double()))
                tr["eq1_ratio"] = float(torch.linalg.vector_norm(nw.double() - wt_flat.double())) / den if den > 0 else float("nan")
                tr["eq1_bound"] = 2.0 ** -24 * float(torch.linalg.vector_norm(nw.double())) / den if den > 0 else float("nan")
                a = info["alpha"]
                tr["eq1_resid"] = float(torch.linalg.vector_norm(nw - ((1 - a) * wt_flat + a * wi_flat))) / \
                    max(float(torch.linalg.vector_norm(wt_flat)), 1e-12)
                del nw
                self.global_sd = new
                self.version += 1
                self.aggregations += 1
                self.accepted += 1
            else:
                self.discarded += 1
            row.update(staleness=info["staleness"], accepted=info["accepted"], alpha=info.get("alpha"),
                       lam=info.get("lam"), sig=info.get("sig"), iota=info.get("iota"))
            # LSTM meta model update with the reward of the device's first training (Eq. 5)
            if r.meta is not None:
                feat, decisions, slot, probs = r.meta
                tr.update(meta_feat=json.dumps([round(x, 6) for x in feat]), meta_probs=json.dumps([round(x, 6) for x in probs]),
                          meta_decisions=json.dumps(decisions), meta_slot=slot)
                if stats.get("merged"):
                    with torch.no_grad():
                        lp0 = float(self.meta.model.log_prob(feat, decisions))
                    b0 = self.meta.baseline
                    adv = self.meta.update(feat, decisions, float(stats["reward"]))
                    with torch.no_grad():
                        lp1 = float(self.meta.model.log_prob(feat, decisions))
                    tr.update(meta_reward=float(stats["reward"]), meta_baseline_before=b0, meta_adv=adv,
                              meta_lp_before=lp0, meta_lp_after=lp1)
                    self.meta_history.append({"feat": feat, "decisions": decisions, "slot": slot,
                                              "reward": float(stats["reward"])})
                self.meta_done.add(d)
        elif self.algorithm in FEDBUFF:
            stal = self.mgr.current_model_version - r.o
            generated = submit_update(self.mgr, w_i, r.o, d, t, fl_ctx)
            tr.update(staleness=stal, accepted=True, fb_generated=bool(generated),
                      fb_model_version=self.mgr.current_model_version)
            self.accepted += 1
            if generated:
                self.version += 1
                self.aggregations += 1
            row.update(staleness=stal, accepted=True)
        row.update(global_version=self.version, aggregations=self.aggregations)
        self.w_upd.writerow(row)
        if self.cfg.get("trace", True):
            self.w_tr.writerow(tr)
        if self.received % 10 == 0:
            self.f_upd.flush()
            self.f_tr.flush()
        if self.received % self.cfg["eval_every"] == 0:
            acc = self._log_eval()
            self.log_info(fl_ctx, f"[{self.algorithm}] received={self.received} version={self.version} "
                                  f"vtime={t:.0f}s wall={time.time() - self.wall0:.0f}s acc={acc:.4f}")
        if self.received % self.cfg.get("checkpoint_every", 500) == 0:
            self._checkpoint()

    # ---------------------------------------------------------------- FedAvg
    def _fedavg_loop(self, abort_signal, fl_ctx):
        """Synchronous FedAvg reference (Fig. 2): each round m' random devices train from
        the same global model; the round lasts until the slowest one uploads."""
        rounds = self.cfg["budget_rounds"]
        E = self.cfg["local"]["epochs"]
        t = self.vtime
        while self.version < rounds and not abort_signal.triggered:
            with self.lock:
                chosen = self.rng.sample(range(self.n_dev), self.cfg["max_concurrency"])
                for d in chosen:
                    self._dispatch(d, t, fl_ctx)
            with self.lock:
                while any(r.result is None for r in self.running.values()):
                    if abort_signal.triggered:
                        return
                    self.lock.wait(timeout=1.0)
                runs = list(self.running.values())
                self.running.clear()
                self.free_sites.extend(r.site for r in runs)
            acc_sd, tot = None, 0
            t_end = t
            for r in runs:
                p = self.profiles[r.device]
                t_end = max(t_end, t + 2 * r.down + E * p.epoch_time)
                self.received += 1
                if r.result.get_return_code(ReturnCode.OK) != ReturnCode.OK:
                    self.failed += 1
                    continue
                stats = r.result.get_header("stats") or {}
                n = stats.get("n_samples", 1)
                w_np = shareable_weights(r.result)
                if not all(np.isfinite(v).all() for v in w_np.values() if np.issubdtype(v.dtype, np.floating)):
                    self.failed += 1
                    self.nonfinite += 1
                    continue
                w = numpy_to_sd(w_np, self.gdev)
                acc_sd = {k: n * v.float() for k, v in w.items()} if acc_sd is None else \
                    {k: acc_sd[k] + n * w[k].float() for k in w}
                tot += n
                self.total_steps += int(stats.get("steps", 0))
                self.total_samples += int(n) * E
                self.w_upd.writerow(dict(received=self.received, vtime=round(t_end, 3), device=r.device, site=r.site,
                                         o=r.o, accepted=True, local_loss=stats.get("loss"), steps=stats.get("steps"),
                                         lr=r.lr, global_version=self.version + 1, aggregations=self.version + 1))
            if acc_sd is not None:
                self.global_sd = {k: (acc_sd[k] / tot).to(self.global_sd[k].dtype) for k in acc_sd}
            self.version += 1
            self.aggregations += 1
            self.accepted = self.version
            t = t_end + self.cfg.get("trigger_period", 0)
            self.vtime = t
            if self.version % max(1, self.cfg["eval_every"] // self.cfg["max_concurrency"]) == 0:
                acc = self._log_eval()
                self.log_info(fl_ctx, f"[fedavg] round={self.version} vtime={t:.0f}s acc={acc:.4f}")

    # ------------------------------------------------------- checkpointing
    def _checkpoint(self):
        """Saves everything needed to resume *between* updates.  In-flight local
        trainings are not saved: on resume they are dropped (never aggregated) and
        the devices become idle again (documented limitation)."""
        st = {"global_sd": {k: v.cpu() for k, v in self.global_sd.items()}, "version": self.version,
              "aggregations": self.aggregations, "received": self.received, "accepted": self.accepted,
              "discarded": self.discarded, "failed": self.failed, "nonfinite": self.nonfinite, "total_steps": self.total_steps,
              "total_samples": self.total_samples, "vtime": self.vtime, "next_trigger": self.next_trigger,
              "rng": self.rng.getstate(), "task_counter": self.task_counter,
              "wall_elapsed": time.time() - self.wall0}
        if self.algorithm in ASYNC_FEDASMU:
            st["server"] = self.server.state_dict()
            st["meta"] = self.meta.state_dict()
            st["meta_done"] = sorted(self.meta_done)
            st["meta_history"] = self.meta_history
        if self.algorithm in FEDBUFF:
            st["mgr"] = {"current_model": self.mgr.current_model, "version": self.mgr.current_model_version,
                         "counter": self.mgr.num_updates_counter,
                         "buffers": {v: (ms.aggregator.dict, ms.aggregator.count) for v, ms in self.mgr.updates.items()}}
        path = os.path.join(self.ckpt_dir, "server_state.pkl")
        with open(path + ".tmp", "wb") as f:
            pickle.dump(st, f, protocol=4)
        os.replace(path + ".tmp", path)
        torch.save(st["global_sd"], os.path.join(self.ckpt_dir, "global_model.pt"))

    def _maybe_resume(self, fl_ctx) -> bool:
        path = os.path.join(self.ckpt_dir, "server_state.pkl")
        if not (self.cfg.get("resume", True) and os.path.exists(path)):
            return False
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.global_sd = {k: v.to(self.gdev) for k, v in st["global_sd"].items()}
        for k in ("version", "aggregations", "received", "accepted", "discarded", "failed", "total_steps",
                  "total_samples", "vtime", "next_trigger", "task_counter"):
            setattr(self, k, st[k])
        self.nonfinite = st.get("nonfinite", 0)
        self.rng.setstate(st["rng"])
        self.wall0 = time.time() - st["wall_elapsed"]
        if self.algorithm in ASYNC_FEDASMU:
            self.server.load_state_dict(st["server"])
            self.meta.load_state_dict(st["meta"])
            self.meta_done = set(st["meta_done"])
            self.meta_history = st["meta_history"]
        if self.algorithm in FEDBUFF:
            from src.algorithms.fedbuff.manager import _ModelState  # noqa
            from nvflare.edge.aggregators.model_update_dxo import ModelUpdateDXOAggregator
            m = st["mgr"]
            self.mgr.current_model = m["current_model"]
            self.mgr.current_model_version = m["version"]
            self.mgr.num_updates_counter = m["counter"]
            self.mgr.updates = {}
            for v, (dct, cnt) in m["buffers"].items():
                ms = _ModelState(ModelUpdateDXOAggregator())
                ms.aggregator.dict, ms.aggregator.count = dct, cnt
                self.mgr.updates[v] = ms
        # rows logged after the checkpoint belong to dropped work -> truncate logs
        for name, key in (("updates.csv", "received"), ("eval.csv", "received"), ("trace.csv", "received")):
            p = os.path.join(self.metrics_dir, name)
            self._truncate_csv(p, key, self.received)
        self.f_upd.close()
        self.f_ev.close()
        self.f_tr.close()
        self._open_logs()
        self.log_info(fl_ctx, f"resumed from checkpoint at received={self.received} vtime={self.vtime:.1f}")
        with open(os.path.join(self.metrics_dir, "resume_log.txt"), "a") as f:
            f.write(f"{time.ctime()} resumed at received={self.received} vtime={self.vtime}\n")
        return True

    @staticmethod
    def _truncate_csv(path, key, max_val):
        if not os.path.exists(path):
            return
        with open(path) as f:
            rows = list(csv.DictReader(f))
            fields = rows[0].keys() if rows else None
        if fields is None:
            return
        keep = [r for r in rows if int(r[key]) <= max_val]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fields))
            w.writeheader()
            w.writerows(keep)

    def _finish(self, status, err=None):
        try:
            if status == "COMPLETED":
                self._log_eval()
                self._checkpoint()
        except Exception:
            err = (err or "") + traceback.format_exc()
            status = "FAILED"
        summary = {"status": status, "algorithm": self.algorithm, "received": self.received,
                   "accepted": self.accepted, "discarded": self.discarded, "failed": self.failed,
                   "rejected_nonfinite": self.nonfinite, "global_version": self.version, "aggregations": self.aggregations, "vtime": self.vtime,
                   "wall": time.time() - self.wall0, "total_steps": self.total_steps,
                   "total_samples": self.total_samples, "error": err}
        if self.algorithm in ASYNC_FEDASMU:
            summary["meta_updates"] = self.meta.n_updates
            summary["meta_pretrained_records"] = self.meta_pretrained
            with open(os.path.join(self.metrics_dir, "meta_history.json"), "w") as f:
                json.dump(self.meta_history, f)
            ctrl = {i: {"lam": s.lam, "sig": s.sig, "iota": s.iota} for i, s in self.server.dev.items()}
            with open(os.path.join(self.metrics_dir, "server_control_params.json"), "w") as f:
                json.dump(ctrl, f)
        with open(os.path.join(self.metrics_dir, "server_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        for fh in (self.f_upd, self.f_ev, self.f_tr):
            try:
                fh.flush()
            except Exception:
                pass

    def stop_controller(self, fl_ctx: FLContext):
        for fh in (getattr(self, "f_upd", None), getattr(self, "f_ev", None), getattr(self, "f_tr", None)):
            if fh:
                fh.close()

    def process_result_of_unknown_task(self, client: Client, task_name: str, client_task_id: str,
                                       result: Shareable, fl_ctx: FLContext):
        self.log_warning(fl_ctx, f"ignored late result for {task_name} from {client.name}")
