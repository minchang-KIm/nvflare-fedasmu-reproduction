"""Explanatory figures and animations of the device scheduling, drawn from the real logs of
the appendix runs (LeNet / CIFAR-10, seed 0), not from a mock-up.

Every training's dispatch time is reconstructed exactly from its arrival time and the
device profile used by the controller (src/nvflare_integration/controller.py):
  arrival = t0 + down + E * epoch_time + up                      (FedBuff, FedAvg, FedASMU w/o reply model)
          + request_latency (+ down if the server sent a fresh model)   (FedASMU)
with down = up = model_MB / bandwidth.  The reconstructed t0 fall exactly on trigger
times (multiples of T = 10 s), which checks the reconstruction.

Output (results/figures/scheduling/):
  async_fedasmu.gif   asynchronous scheduling: trigger, idle-device sampling, predicted
                      arrival, mid-training fresh-model request, aggregation on arrival
  async_fedbuff.gif   same scheduling with FedBuff's buffer (aggregate every K = 10)
  sync_fedavg.gif     synchronous rounds: 10 random devices, wait for the slowest
  async_vs_sync.png   the same time window side by side, with idle (waiting) time
  one_training.png    timeline of one device training and how its arrival time is computed
  budget.png          why 500 synchronous rounds = 5000 asynchronous updates
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.models import build_model  # noqa: E402

OUT = os.path.join(ROOT, "results", "figures", "scheduling")
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False
C_TRAIN, C_REQ, C_ARR, C_IDLE, C_WAIT = "#4c78a8", "#2ca02c", "#d62728", "#eeeeee", "#f4a3a3"


def load(algo, wl=("lenet5", "cifar10")):
    rid = f"appendix_{wl[0]}_{wl[1]}_{algo}_s0"
    cfg = json.load(open(os.path.join(ROOT, "workspaces", "runs", rid, "run_config.json")))
    prof = {p["device_id"]: p for p in cfg["profiles"]}
    m = build_model(wl[0], 10, wl[1])
    mb = sum(v.numel() * 4 for v in m.state_dict().values()) / 2 ** 20
    u = pd.read_csv(os.path.join(ROOT, "results", "metrics", rid, "updates.csv"))
    E, lat = cfg["local"]["epochs"], cfg.get("request_latency", 0.1)
    rows = []
    for _, r in u.iterrows():
        p = prof[int(r.device)]
        down = mb / p["bandwidth"]
        fresh = algo == "fedasmu" and r.merged is True or (algo == "fedasmu" and str(r.merged) == "True")
        dur = 2 * down + E * p["epoch_time"] + (lat + (down if fresh else 0) if algo == "fedasmu" else 0)
        t0 = r.vtime - dur
        req = None
        if algo == "fedasmu" and pd.notna(r.request_after):
            req = t0 + down + int(r.request_after) * p["epoch_time"]
        rows.append(dict(dev=int(r.device), t0=round(t0, 3), t1=r.vtime, req=req, fresh=fresh, version=r.global_version,
                         o=r.o, received=int(r.received), slow=p["slowdown"], down=down, epoch=p["epoch_time"]))
    df = pd.DataFrame(rows)
    if algo == "fedavg":
        # updates.csv logs the round's aggregation time for all 10 devices; in a synchronous
        # round every device starts together (previous round end + T) and finishes after its
        # own duration.  Check: the slowest device's finish equals the logged round time.
        start, err = 0.0, 0.0
        for v in sorted(df.version.unique()):
            idx = df.version == v
            dur = df.loc[idx, "t1"] - df.loc[idx, "t0"]
            end_logged = df.loc[idx, "t1"].max()
            df.loc[idx, "t0"] = start
            df.loc[idx, "t1"] = start + dur
            err = max(err, abs(df.loc[idx, "t1"].max() - end_logged))
            start = end_logged + cfg["trigger_period"]
        print("fedavg reconstruction: max |slowest finish - logged round time| =", round(err, 6))
    return df, cfg


def assign_slots(df, n=10):
    """Give each training a display row (one of the 10 concurrent slots)."""
    free_at = [-1.0] * n
    slots = []
    for _, r in df.sort_values(["t0", "t1"]).iterrows():
        s = next(i for i in range(n) if free_at[i] <= r.t0 + 1e-6)
        free_at[s] = r.t1
        slots.append((r.name, s))
    df = df.copy()
    for idx, s in slots:
        df.loc[idx, "slot"] = s
    return df


def gantt_async(ax, df, now, T=10, show_req=True, buffer_k=None):
    ax.clear()
    for _, r in df[df.t0 <= now].iterrows():
        y = r.slot
        end = min(r.t1, now)
        done = r.t1 <= now
        ax.add_patch(Rectangle((r.t0, y - 0.35), end - r.t0, 0.7, color=C_TRAIN, alpha=0.35 if done else 0.85, lw=0))
        known = r.req is None or not show_req or r.req <= now  # FedASMU: fixed once the request is answered
        if not done and known:  # predicted arrival (virtual time) already scheduled by the server
            ax.plot([r.t1, r.t1], [y - 0.42, y + 0.42], color=C_ARR, lw=1.2, ls=":")
        ax.text(r.t0 + 2, y, f"기기 {r.dev}", va="center", fontsize=7, color="white" if not done else "#333")
        if show_req and r.req is not None and r.req <= now:
            ax.plot(r.req, y, marker="D", ms=5, color=C_REQ if r.fresh else "#999999")
        if done:
            ax.plot(r.t1, y, marker="v", ms=6, color=C_ARR)
    for t in np.arange(0, now + 1e-6, T):
        ax.axvline(t, color="#bbbbbb", lw=0.5, zorder=0)
    ax.axvline(now, color="k", lw=1.2)
    ax.set_ylim(-0.7, 9.7)
    ax.set_yticks(range(10), [f"슬롯 {i + 1}" for i in range(10)], fontsize=7)
    ax.set_xlabel("가상 시간 (초)")


def anim_async(df, path, title, window=420, step=3, buffer_k=None, show_req=True):
    df = assign_slots(df[df.t0 < window])
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_axes([0.08, 0.30, 0.9, 0.6])
    info = fig.add_axes([0.08, 0.03, 0.9, 0.17])
    frames = np.arange(0, window + step, step)

    def draw(now):
        gantt_async(ax, df, now, show_req=show_req)
        ax.set_xlim(0, window)
        ax.set_title(title, fontsize=11)
        info.clear()
        info.axis("off")
        arrived = df[df.t1 <= now]
        running = df[(df.t0 <= now) & (df.t1 > now)]
        txt = (f"t = {now:5.0f} s   |   학습 중 {len(running)}/10대   |   도착한 업데이트 {len(arrived)}개   |   ")
        if buffer_k:
            nbuf = len(arrived) % buffer_k
            txt += f"버퍼 {nbuf}/{buffer_k}  →  글로벌 버전 {len(arrived) // buffer_k}"
            for i in range(buffer_k):
                info.add_patch(Rectangle((0.55 + i * 0.035, 0.05), 0.03, 0.3, color=C_ARR if i < nbuf else "#dddddd",
                                         transform=info.transAxes))
        else:
            txt += f"글로벌 버전 {len(arrived)} (도착할 때마다 바로 집계)"
        info.text(0, 0.75, txt, fontsize=10, transform=info.transAxes)
        leg = "■ 학습 중   :  빨간 점선 = 서버가 미리 계산해 둔 도착 예정 시각   ▼ 도착(업로드)   │ 회색 세로선 = 트리거(10초마다)"
        if show_req:
            leg += ("\n◆ 초록 = 학습 도중 최신 모델 요청 → 받아서 병합   ◆ 회색 = 요청했지만 더 새로운 모델이 없음"
                    "   (FedASMU의 도착 예정 시각은 요청을 처리할 때 확정)")
        info.text(0, 0.0 if buffer_k is None else 0.45, leg, fontsize=8, color="#444", transform=info.transAxes)

    ani = animation.FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames))
    ani.save(path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)


def fedavg_rounds(df):
    rounds = []
    for v, g in df.groupby("version"):
        rounds.append(dict(v=int(v), t0=g.t0.min(), t1=g.t1.max(), rows=g))
    return rounds


def anim_sync(df, path, window=900, step=6):
    rounds = [r for r in fedavg_rounds(df) if r["t0"] < window]
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_axes([0.08, 0.30, 0.9, 0.6])
    info = fig.add_axes([0.08, 0.03, 0.9, 0.17])
    frames = np.arange(0, window + step, step)

    def draw(now):
        ax.clear()
        idle_total = busy_total = 0.0
        for rd in rounds:
            if rd["t0"] > now:
                continue
            g = rd["rows"].sort_values("t1").reset_index(drop=True)
            for i, r in g.iterrows():
                end = min(r.t1, now)
                ax.add_patch(Rectangle((r.t0, i - 0.35), end - r.t0, 0.7, color=C_TRAIN, alpha=0.85, lw=0))
                ax.text(r.t0 + 2, i, f"기기 {r.dev}", va="center", fontsize=7, color="white")
                if r.t1 < now:  # finished, waiting for the slowest device
                    w_end = min(rd["t1"], now)
                    ax.add_patch(Rectangle((r.t1, i - 0.35), max(0, w_end - r.t1), 0.7, color=C_WAIT, lw=0))
                    idle_total += max(0, w_end - r.t1)
                busy_total += end - r.t0
            if rd["t1"] <= now:
                ax.axvline(rd["t1"], color=C_ARR, lw=1.5)
                ax.text(rd["t1"], 9.9, f"라운드 {rd['v']} 집계", color=C_ARR, fontsize=7, ha="center")
        ax.axvline(now, color="k", lw=1.2)
        ax.set_xlim(0, window)
        ax.set_ylim(-0.7, 10.3)
        ax.set_yticks(range(10), [f"{i + 1}번째로 끝난 기기" for i in range(10)], fontsize=7)
        ax.set_xlabel("가상 시간 (초)")
        ax.set_title("동기식 FedAvg: 라운드마다 무작위 10대 → 가장 느린 기기까지 기다린 뒤 한 번에 평균", fontsize=11)
        info.clear()
        info.axis("off")
        done = sum(1 for rd in rounds if rd["t1"] <= now)
        frac = idle_total / (idle_total + busy_total) if busy_total else 0
        info.text(0, 0.7, f"t = {now:5.0f} s   |   끝난 라운드 {done}   |   로컬 학습 {done * 10}회   |   "
                          f"기다린 시간 비율 {frac:.0%}", fontsize=10, transform=info.transAxes)
        info.text(0, 0.1, "■ 학습 중   ■ 분홍 = 먼저 끝났지만 가장 느린 기기를 기다리는 시간(유휴)   "
                          "│ 빨간 세로선 = 라운드 집계 → 10초 뒤 다음 라운드 시작", fontsize=8, color="#444",
                  transform=info.transAxes)

    ani = animation.FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames))
    ani.save(path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)


def static_compare(da, ds, path, window=900):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    da = assign_slots(da[da.t0 < window])
    gantt_async(ax[0], da, window, show_req=False)
    ax[0].set_xlabel("")
    busy = sum(min(r.t1, window) - r.t0 for _, r in da.iterrows())
    n_a = int((da.t1 <= window).sum())
    ax[0].set_title(f"비동기식 (FedBuff · FedASMU): 끝난 기기 자리에 다음 트리거(≤10초) 때 바로 새 기기 투입 → "
                    f"{window}초 동안 로컬 학습 {n_a}회, 슬롯 사용률 {busy / (10 * window):.0%}", fontsize=10)
    rounds = [r for r in fedavg_rounds(ds) if r["t0"] < window]
    busy = idle = 0
    n_s = 0
    for rd in rounds:
        g = rd["rows"].sort_values("t1").reset_index(drop=True)
        for i, r in g.iterrows():
            e = min(r.t1, window)
            ax[1].add_patch(Rectangle((r.t0, i - 0.35), e - r.t0, 0.7, color=C_TRAIN, alpha=0.85, lw=0))
            busy += e - r.t0
            if r.t1 < window:
                w = min(rd["t1"], window) - r.t1
                ax[1].add_patch(Rectangle((r.t1, i - 0.35), w, 0.7, color=C_WAIT, lw=0))
                idle += w
        if rd["t1"] <= window:
            ax[1].axvline(rd["t1"], color=C_ARR, lw=1.2)
            n_s += 10
    ax[1].set_ylim(-0.7, 9.7)
    ax[1].set_yticks(range(10), [f"{i + 1}번째로 끝난 기기" for i in range(10)], fontsize=7)
    ax[1].set_xlim(0, window)
    ax[1].set_xlabel("가상 시간 (초)")
    ax[1].set_title(f"동기식 (FedAvg): 가장 느린 기기를 기다림(분홍) → {window}초 동안 로컬 학습 {n_s}회, "
                    f"기다린 시간 비율 {idle / (busy + idle):.0%}", fontsize=10)
    fig.suptitle("같은 기기 100대 · 같은 속도 설정(LeNet / CIFAR-10 실제 실행 로그)에서 처음 "
                 f"{window}초", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def one_training(da, path):
    r = da[(da.fresh) & (da.req.notna())].iloc[3]
    E = 5
    fig, ax = plt.subplots(figsize=(12, 3.6))
    t = r.t0
    segs = [("모델 다운로드", r.down, "#9ecae1")]
    l = int(round((r.req - r.t0 - r.down) / r.epoch))
    segs += [(f"에폭 {i + 1}", r.epoch, C_TRAIN) for i in range(l)]
    segs += [("요청+최신 모델 받기·병합", 0.1 + r.down, C_REQ)]
    segs += [(f"에폭 {i + 1}", r.epoch, C_TRAIN) for i in range(l, E)]
    segs += [("업로드", r.down, "#9ecae1")]
    small = 0
    for name, d, c in segs:
        w = max(d, 1.5)
        ax.add_patch(Rectangle((t + 0.15, 0.3), w - 0.3, 0.4, color=c, lw=0, zorder=3 if d < 8 else 2))
        if d >= 8:
            ax.text(t + d / 2, 0.5, name, ha="center", va="center", fontsize=8, color="white")
        else:  # short segment: label outside the bar
            y = 0.95 if small % 2 == 0 else 0.05
            ax.annotate(f"{name} ({d:.1f}초)", (t + w / 2, 0.7 if y > 0.5 else 0.3), (t + w / 2, y), ha="center",
                        fontsize=7, color="#333", arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
            small += 1
        t += d
    ax.annotate("서버가 보내는 순간(t0)\n= 트리거 시각", (r.t0, 0.3), (r.t0 - 8, -0.35), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"l* = {l} 에폭 뒤 요청\n(LSTM / Q-러닝이 결정)", (r.req, 0.7), (r.req - 25, 1.3), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C_REQ))
    ax.annotate("도착 = 서버가 집계하는 시각", (r.t1, 0.3), (r.t1 + 5, -0.35), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C_ARR))
    ax.set_xlim(r.t0 - 15, r.t1 + 25)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("가상 시간 (초)")
    ax.set_title(f"기기 {r.dev} 한 번의 학습 (속도 배수 {r.slow:.2f}, 에폭당 {r.epoch:.1f}초): "
                 f"도착 시각 = t0 + 다운로드 + 5 × 에폭 시간 + (요청 지연 0.1초 + 최신 모델 다운로드) + 업로드 "
                 f"= {r.t0:.0f} + … = {r.t1:.1f}초", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def budget(path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, (title, per, n, col) in enumerate([("동기식: 500 라운드", 10, 500, "#ff7f0e"),
                                               ("비동기식: 5,000 업데이트", 1, 5000, C_TRAIN)]):
        a = ax[k]
        for i in range(3):
            for j in range(per if per > 1 else 1):
                a.add_patch(Rectangle((i * 3.6 + (j % 5) * 0.6, (j // 5) * 0.6 if per > 1 else 0), 0.5, 0.5, color=col))
            a.annotate("", (i * 3.6 + 3.2, 0.5), (i * 3.6 + 2.9, 0.5), arrowprops=dict(arrowstyle="->"))
            a.text(i * 3.6 + 1.4, -0.6, "글로벌 버전 +1", ha="center", fontsize=8)
        a.text(11.2, 0.5, "…", fontsize=16)
        a.set_xlim(-0.5, 13)
        a.set_ylim(-1.2, 1.8)
        a.axis("off")
        a.set_title(f"{title}\n집계 1번 = 로컬 학습 {per}회  →  {n} × {per} = 로컬 학습 5,000회", fontsize=10)
    fig.suptitle("두 방식이 같은 횟수(5,000회)의 로컬 학습을 하도록 맞춘 예산 (m′ = 10)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    da, _ = load("fedasmu")
    db, _ = load("fedbuff")
    ds, _ = load("fedavg")
    off = da.t0.mod(10).round(6)
    print("dispatch times on trigger grid:", float(((off < 1e-3) | (off > 10 - 1e-3)).mean()))
    anim_async(da, os.path.join(OUT, "async_fedasmu.gif"),
               "비동기식 FedASMU: 10초마다 트리거 → 빈 슬롯에 대기 기기 무작위 투입 → 도착할 때마다 집계")
    anim_async(db, os.path.join(OUT, "async_fedbuff.gif"),
               "비동기식 FedBuff: 스케줄은 같고, 도착한 업데이트를 K = 10개 모아 한 번에 집계",
               buffer_k=10, show_req=False)
    anim_sync(ds, os.path.join(OUT, "sync_fedavg.gif"))
    static_compare(db, ds, os.path.join(OUT, "async_vs_sync.png"))
    one_training(da, os.path.join(OUT, "one_training.png"))
    budget(os.path.join(OUT, "budget.png"))


if __name__ == "__main__":
    main()
