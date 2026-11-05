# NVFLARE execution architecture

All API usage below was read from the **installed NVFLARE 2.9.0 wheel**, not from GitHub `main`.

## 1. Options considered

| Option | Verdict | Reason |
|---|---|---|
| A. Collaboration API (`nvflare.collab`, `CollabRecipe`) | Rejected | Technical preview. The client→server direction for a request mid-way through a remote call is not part of its documented contract. The client list is fixed during a run, and there are no automatic retries. Our design needs the server to delay its answer until a virtual-time point, which is not expressible there without undocumented internals. |
| B. Custom `Controller` + `Executor` + aux channel | **Chosen** | Official, stable extension points. `Controller.send(Task, targets=[site], result_received_cb=...)` is non-blocking per device. The client→server request uses `engine.send_aux_request(targets=None, topic, ...)` on the client and `engine.register_aux_message_handler(topic, fn)` on the server. Both are public APIs in `apis/server_engine_spec.py` and `private/fed/client/client_engine_executor_spec.py`. |
| C. `EdgeFedBuffRecipe` / edge hierarchy | Partly reused | The edge stack (device simulator, `BuffDeviceManager`, hierarchical leaf clients) only supports *get task / report result*, with no mid-training model retrieval. FedASMU cannot be expressed in it without modifying the core. We reuse its **aggregation component `BuffModelManager` unchanged** for FedBuff, so FedBuff aggregation is NVFLARE's official code. FedASMU and FedBuff then share one runtime (option B). |

No NVFLARE core file was modified.

## 2. Components

```text
SimEnv (official) ── FedJob ──┬─ server: AsyncSimController (src/nvflare_integration/controller.py)
                              │     ├─ virtual-time event loop (TRIGGER / REQUEST / ARRIVAL)
                              │     ├─ FedASMUServer (Eq.1-3)  | BuffModelManager (official) | FedAvg
                              │     ├─ MetaController (LSTM, Eq.5)
                              │     └─ aux handlers: fedasmu.fresh_model_request / _poll
                              └─ sites site-1..site-10: DeviceSimExecutor (src/nvflare_integration/executor.py)
                                    ├─ runs local SGD of the logical device named in the task
                                    ├─ FedASMU device logic (Eq.7-9, Q-learning Eq.6)
                                    └─ device-private state: <run>/device_store/device_<i>.pt
```

The single entry point is `scripts/run_experiment.py`. It builds the shared artefacts, writes `run_config.json`, creates the `FedJob`, and calls `SimEnv(clients=[site-1..site-10], num_threads=10).deploy(job)`.

The NVFLARE simulator starts one OS process (`simulator_worker`) per site. Sites are pinned round-robin to the two GPUs with `torch.cuda.set_device`, giving one CUDA context per process. The server (aggregation and test evaluation) runs on CPU.

## 3. Logical devices vs processes

- There are 100 logical devices and 10 site processes. 10 = max concurrency m′, so every concurrently training device owns a site for the duration of its local training.
- A task carries `device`, `version o`, `task_id`, `lr`, the global weights, and (on the first training) the meta-model slot.
- The executor loads only that device's data shard and its private state (γ, υ, b, Q-table, RNG counter), trains, saves the state atomically, and returns the result.
- Different devices therefore never share data or RL state, even though site processes are reused.

## 4. Virtual time and asynchrony

Each device i has `epoch_time_i = base × U[1,5]` and `bw_i ~ U[1,5] MB/s`. For a local training starting at virtual time t₀:

- request time = t₀ + down + l*·epoch_time
- arrival time = request time + latency (+ down, if a newer model is sent) + (L−l*)·epoch_time + up
- for algorithms without a request: arrival = t₀ + down + L·epoch_time + up

The server processes events in virtual-time order (conservative discrete-event simulation):
- An ARRIVAL is processed only after the real result has arrived, so the loop waits in real time while the computation runs.
- The loop never passes the earliest *possible* request time of a running device whose request has not reached the server yet.
- Because #sites = m′, every pending computation is always making progress, so this cannot deadlock.

Consequences:
- **Fast and slow devices:** updates are aggregated in completion order. A slow device trained from an old version arrives later, with staleness > 1. The integration tests check that `vtime` is monotonic and that staleness > 1 occurs.
- **Concurrent training:** other devices keep training (really, in parallel processes) while the server aggregates.
- **Fresh-model requests:** a FedASMU device that requests at virtual time t_r receives exactly the global model after all arrivals with time < t_r. This is the version it would have seen on a real system with these delays.
- **Two clocks:** real wall-clock (`wall`) and virtual time (`vtime`) are logged separately. Paper-style "time to target" uses `vtime`.

The fresh-model request is non-blocking on the server. `fresh_model_request` registers a REQUEST event and returns `pending`; the device then polls `fresh_model_poll` until the event has been processed. This avoids parking NVFLARE message threads. The reply is cached until the arrival, so a task retried after a CUDA out-of-memory error gets the same answer and is never double-scheduled.

## 5. FedBuff: NVFLARE `BuffModelManager` vs the FedBuff paper

Checked against the installed source `nvflare/edge/assessors/buff_model_manager.py`. Unit tests are in `tests/unit/test_fedbuff.py`.

| # | Aspect | FedBuff paper (Alg. 1, §5) | NVFLARE 2.9.0 | Same? |
|---|---|---|---|---|
| 1 | Trigger | Update when k == K | `num_updates_counter >= num_updates_for_model`, counted per device | Yes |
| 2 | Same client twice | Each update counts | Each `ModelUpdate` counts once per device entry | Yes (test) |
| 3 | Payload | Δ = model difference | `ModelUpdateDXOAggregator` requires `DataKind.WEIGHT_DIFF` | Yes |
| 4 | Server rule | w ← w − η_g · Δ̄/K (sign convention: Δ = y₀ − y_Q) | w ← w + global_lr · Σ(s·Δ)/N, with Δ = w_local − w_recv | Equivalent |
| 5 | Staleness weight | s(τ) = 1/√(1+τ), τ = current − start version | `1/(1+(V_new − v)^0.5)` with V_new = current+1 | **Different** (form and off-by-one) |
| 6 | Buffer not full | Wait | Wait | Yes |
| 7 | Timeout-forced aggregation | None | None in `BuffModelManager` (time-based logic lives in device managers) | Yes |
| 8 | Duplicates | Not addressed | No deduplication | Yes |
| – | Server momentum / optimizer | Tuned per task (FedAvgM-style option) | Not available | Not used by either |

Runs:
- `fedbuff` = official manager, unchanged (`staleness_weight=True`).
- `fedbuff_paper` = `PaperFedBuffModelManager`, a subclass that only replaces the weight with 1/√(1+τ).
- The FedASMU-vs-FedBuff priority comparison uses the official manager. The paper variant is run as an additional check.

## 6. Verification (step 3)

- `tests/integration/test_nvflare_runs.py` runs real SimEnv jobs:
  - FedASMU devices request mid-training, receive version g > o and merge it (`merged=True`, `fresh_version > o`).
  - Arrivals are processed in virtual completion order.
  - FedBuff produces ⌊24/10⌋ = 2 aggregations from 24 updates.
  - Checkpoint/resume produces the update ids 1..24 exactly once.
- Run on 2026-11-05 on CPU (GPUs on host 175 were occupied by another job): 35 passed (32 unit + 3 integration).

## 7. Failure and timeout handling

- **CUDA OOM:** the executor retries the same task (same inputs and RNG state) up to 20 times. A task that still fails returns `EXECUTION_EXCEPTION`, and the server logs it as `failed` (the update is lost, like a device dropout).
- **Task timeout:** `Task.timeout = 0`, so tasks never time out on the server (the simulator has no real network losses). End of budget → `cancel_all_tasks`, and pending polls receive `abort`.
- **Checkpoints:** every 500 updates the run saves the global model, counters, virtual clock, RNG state, FedASMU control parameters (and the previous-direction store, a memmap for large models), LSTM state and history, and the FedBuff buffers.
- **Resume:**
  - In-flight trainings are dropped (never aggregated), and their devices become idle.
  - CSV rows beyond the checkpoint are truncated, so nothing is aggregated twice.
  - Limitation: device-local state files may already contain the effect of a dropped training (Q-update, γ/υ). This is recorded in `resume_log.txt`.
