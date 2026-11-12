"""Why do our appendix curves differ from the paper's?  Quantitative comparison of the
appendix runs with the paper curves digitised from the PDF (scripts/digitize_paper_curves.py).

Output: results/reports/paper_gap_analysis.md, results/figures/appendix/gap_analysis.png
        results/figures/appendix/overlay_<fig>.png (paper curves dashed, ours solid, paper axes)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
PA = os.path.join(ROOT, "results", "figures", "paper_appendix")
FIG = os.path.join(ROOT, "results", "figures", "appendix")
REP = os.path.join(ROOT, "results", "reports", "paper_gap_analysis.md")
WLS = ["lenet5-cifar10", "cnn-cifar10", "resnet20-tinyimagenet", "lenet5-cifar100", "cnn-cifar100",
       "resnet20-cifar100", "alexnet-cifar10", "vgg11-cifar10", "textcnn-imdb", "alexnet-cifar100",
       "vgg11-cifar100", "lenet5-fmnist"]
M = {"fedasmu": "FedASMU", "fedbuff": "FedBuff", "fedavg": "FedAvg"}
COL = {"FedASMU": "#ff0000", "FedAvg": "#ff5b2e", "FedBuff": "#4c50f1"}


def ours(wl, algo):
    m, d = wl.split("-")
    p = os.path.join(MET, f"appendix_{m}_{d}_{algo}_s0")
    try:
        ev = pd.read_csv(os.path.join(p, "eval.csv")).drop_duplicates(["received", "global_version"], keep="last")
        st = json.load(open(os.path.join(p, "status.json")))["status"]
    except Exception:
        return None, None
    return ev, p if st == "COMPLETED" else None


def first_hit(t, a, target):
    i = np.where(np.asarray(a) >= target)[0]
    return float(np.asarray(t)[i[0]]) if len(i) else None


def jitter(t, a):
    t, a = np.asarray(t), np.asarray(a)
    xs = np.linspace(t.max() / 2, t.max(), 50)
    return float(np.mean(np.abs(np.diff(np.interp(xs, t, a)))))


def diagnose(pc, axes):
    """Diagnostic runs (configs/experiments/diagnose.yaml) next to the appendix runs."""
    sets = {"lenet5-cifar10": [("appendix_lenet5_cifar10_fedasmu_s0", "FedASMU (default sign)", "#ff0000"),
                               ("diagnose_lenet5_cifar10_fedasmu-psign_s0", "FedASMU (printed sign)", "#8b0000"),
                               ("diagnose_lenet5_cifar10_fedasmu_da_s0", "FedASMU-DA (Eq. 3 off)", "#ff9900"),
                               ("appendix_lenet5_cifar10_fedbuff_s0", "FedBuff", "#4c50f1"),
                               ("appendix_lenet5_cifar10_fedavg_s0", "FedAvg", "#ff5b2e")],
            "cnn-cifar10": [("appendix_cnn_cifar10_fedasmu_s0", "FedASMU (default sign)", "#ff0000"),
                            ("diagnose_cnn_cifar10_fedasmu-psign_s0", "FedASMU (printed sign)", "#8b0000"),
                            ("appendix_cnn_cifar10_fedbuff_s0", "FedBuff", "#4c50f1"),
                            ("appendix_cnn_cifar10_fedavg_s0", "FedAvg", "#ff5b2e")]}
    if not all(os.path.exists(os.path.join(MET, r, "eval.csv")) for v in sets.values() for r, _, _ in v):
        return
    f, ax = plt.subplots(2, 2, figsize=(13, 9))
    for c, (wl, runs) in enumerate(sets.items()):
        d = pc[(pc.workload == wl) & (pc.method == "FedASMU")].sort_values("time_s")
        ax[0, c].plot(d.time_s, d.accuracy, ls="--", color="k", lw=1.5, label="FedASMU paper (digitised)")
        for r, lab, col in runs:
            ev = pd.read_csv(os.path.join(MET, r, "eval.csv")).drop_duplicates(["received", "global_version"], keep="last")
            ax[0, c].plot(ev.vtime, ev.test_acc, color=col, lw=1, label=lab)
            u = os.path.join(MET, r, "updates.csv")
            if "fedasmu" in r:
                u = pd.read_csv(u)
                u = u[u.accepted == True]  # noqa: E712
                ax[1, c].plot(u.received, u.alpha.rolling(100).mean(), color=col, label=lab)
        A = axes[wl]
        ax[0, c].set_xlim(A["xlim"][0], 130000)
        ax[0, c].set_ylim(A["ylim"][0], max(A["ylim"][1], 0.6))
        ax[0, c].set_xlabel("Time (s)")
        ax[0, c].set_ylabel("Accuracy")
        ax[0, c].set_title(f"{wl}: accuracy (paper x axis cut at 130 k s)", fontsize=10)
        ax[0, c].legend(fontsize=7, loc="lower right")
        ax[1, c].set_xlabel("received updates")
        ax[1, c].set_ylabel("alpha (100-update mean)")
        ax[1, c].set_title(f"{wl}: server mixing weight alpha", fontsize=10)
        ax[1, c].legend(fontsize=7)
    f.suptitle("Diagnostic runs: the Eq. 3 control-parameter drift (seed 0, everything else as the appendix runs)",
               fontsize=11)
    f.tight_layout()
    f.savefig(os.path.join(FIG, "gap_diagnose.png"), dpi=100)
    plt.close(f)


def main():
    S = yaml.safe_load(open(os.path.join(ROOT, "configs/common/paper_settings.yaml")))
    pc = pd.read_csv(os.path.join(PA, "curves.csv"))
    axes = json.load(open(os.path.join(PA, "axes.json")))
    rows = []
    for wl in WLS:
        tgt = S[wl]["target"]
        r = {"workload": wl, "target": tgt}
        for algo, name in M.items():
            d = pc[(pc.workload == wl) & (pc.method == name)].sort_values("time_s")
            r[f"paper_{algo}_t"] = first_hit(d.time_s, d.accuracy, tgt)
            r[f"paper_{algo}_jit"] = jitter(d.time_s, d.accuracy)
            ev, done = ours(wl, algo)
            if done is None:
                continue
            r[f"our_{algo}_t"] = first_hit(ev.vtime, ev.test_acc, tgt)
            r[f"our_{algo}_n"] = first_hit(ev.received, ev.test_acc, tgt)  # local trainings to target
            r[f"our_{algo}_jit"] = jitter(ev.vtime, ev.test_acc)
            r[f"our_{algo}_tend"] = float(ev.vtime.max())
            if algo == "fedasmu":
                u = pd.read_csv(os.path.join(done, "updates.csv"))
                u = u[u.accepted == True]  # noqa: E712
                r["alpha_mean"] = float(u.alpha.mean())
                r["iota_end"] = float(u[u.received > 4500].iota.mean())
                tr = os.path.join(done, "trace.csv")
                if os.path.exists(tr):
                    g = pd.read_csv(tr, usecols=["g_dot_prev"]).g_dot_prev.dropna()
                    r["gdot_neg"] = float((g < 0).mean())
        rows.append(r)
    df = pd.DataFrame(rows).set_index("workload")

    # ---- figure: three diagnostics ----------------------------------------
    f, ax = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(df))
    pr = df.paper_fedavg_t / df.paper_fedasmu_t
    orr = df.get("our_fedavg_t") / df.get("our_fedasmu_t")
    ax[0].bar(x - 0.2, pr, 0.4, label="paper", color="0.6")
    ax[0].bar(x + 0.2, orr, 0.4, label="ours", color="#4c50f1")
    ax[0].axhline(1.55, ls="--", color="k", lw=1)
    ax[0].text(len(df) - 0.5, 1.7, "time-model bound 1.55x\n(5x device spread, 10 per round)", ha="right", fontsize=7)
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, df.index, rotation=70, fontsize=7)
    ax[0].set_ylabel("FedAvg time-to-target / FedASMU time-to-target")
    ax[0].set_title("(a) FedAvg slowdown vs FedASMU", fontsize=10)
    ax[0].legend(fontsize=8)
    for i, algo in enumerate(M):
        ax[1].bar(x - 0.3 + 0.2 * i, df[f"paper_{algo}_jit"], 0.2, color=COL[M[algo]], alpha=0.4,
                  label=f"paper {M[algo]}")
        if f"our_{algo}_jit" in df:
            ax[1].scatter(x - 0.3 + 0.2 * i, df[f"our_{algo}_jit"], color=COL[M[algo]], marker="o", s=18,
                          label=f"ours {M[algo]}")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x, df.index, rotation=70, fontsize=7)
    ax[1].set_ylabel("mean |Δ accuracy| between neighbouring points, 2nd half")
    ax[1].set_title("(b) curve roughness: paper (bars) vs ours (dots)", fontsize=10)
    ax[1].legend(fontsize=6, ncol=2)
    for wl in ["lenet5-cifar10", "lenet5-cifar100", "resnet20-cifar100", "cnn-cifar10"]:
        m, d = wl.split("-")
        p = os.path.join(MET, f"appendix_{m}_{d}_fedasmu_s0", "updates.csv")
        if os.path.exists(p):
            u = pd.read_csv(p)
            u = u[u.accepted == True]  # noqa: E712
            ax[2].plot(u.received, u.alpha.rolling(100).mean(), label=f"{wl}: alpha")
    ax[2].set_xlabel("received updates")
    ax[2].set_ylabel("FedASMU alpha (100-update mean)")
    ax[2].axhline(0.1, ls=":", color="#4c50f1")
    ax[2].text(5000, 0.11, "FedBuff: each update weighs 1/K = 0.1", ha="right", fontsize=7, color="#4c50f1")
    ax[2].set_title("(c) FedASMU server mixing weight alpha (Eq. 1-2)", fontsize=10)
    ax[2].legend(fontsize=7)
    f.tight_layout()
    f.savefig(os.path.join(FIG, "gap_analysis.png"), dpi=100)
    plt.close(f)

    # ---- overlay figures: paper (dashed) and ours (solid) on the paper axes --------
    groups = {"fig3": WLS[0:3], "fig4": WLS[3:6], "fig5": WLS[6:9], "fig6": WLS[9:12]}
    for g, wls in groups.items():
        f, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        for a_, wl in zip(axs, wls):
            A = axes[wl]
            for algo, name in M.items():
                d = pc[(pc.workload == wl) & (pc.method == name)].sort_values("time_s")
                a_.plot(d.time_s, d.accuracy, ls="--", lw=1.6, color=COL[name], label=f"{name} paper")
                ev, _ = ours(wl, algo)
                if ev is not None and len(ev):
                    a_.plot(ev.vtime, ev.test_acc, lw=1.0, color=COL[name], alpha=0.85, label=f"{name} ours")
            a_.set_xlim(*A["xlim"])
            a_.set_ylim(*A["ylim"])
            a_.set_xticks(A["xticks"])
            a_.set_yticks(A["yticks"])
            a_.ticklabel_format(axis="x", style="plain")
            a_.tick_params(labelsize=7)
            a_.set_xlabel("Time (s)")
            a_.set_ylabel("Accuracy")
            a_.set_title(wl, fontsize=10)
            a_.legend(fontsize=6, loc="lower right", ncol=2)
        f.suptitle(f"Paper appendix {g.replace('fig', 'Figure ')}: paper curves (dashed, digitised from the PDF) "
                   "and ours (solid), paper axes", fontsize=10)
        f.tight_layout()
        f.savefig(os.path.join(FIG, f"overlay_{g}.png"), dpi=100)
        plt.close(f)

    diagnose(pc, axes)

    # ---- report table --------------------------------------------------------
    def fmt(v, p=0):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{p}f}")
    L = ["| workload | target | paper FedASMU t | ours FedASMU t (n) | paper FedBuff t | ours FedBuff t (n) | "
         "paper FedAvg t | ours FedAvg t (n) | paper FedAvg/FedASMU | ours FedAvg/FedASMU |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for wl, r in df.iterrows():
        def g(k):
            return r.get(k) if k in r and pd.notna(r.get(k)) else None
        ratio_p = g("paper_fedavg_t") / g("paper_fedasmu_t") if g("paper_fedavg_t") and g("paper_fedasmu_t") else None
        ratio_o = g("our_fedavg_t") / g("our_fedasmu_t") if g("our_fedavg_t") and g("our_fedasmu_t") else None
        L.append(f"| {wl} | {r.target} | {fmt(g('paper_fedasmu_t'))} | {fmt(g('our_fedasmu_t'))} ({fmt(g('our_fedasmu_n'))}) | "
                 f"{fmt(g('paper_fedbuff_t'))} | {fmt(g('our_fedbuff_t'))} ({fmt(g('our_fedbuff_n'))}) | "
                 f"{fmt(g('paper_fedavg_t'))} | {fmt(g('our_fedavg_t'))} ({fmt(g('our_fedavg_n'))}) | "
                 f"{fmt(ratio_p, 1)} | {fmt(ratio_o, 2)} |")
    J = ["| workload | paper FedASMU | ours FedASMU | paper FedBuff | ours FedBuff | paper FedAvg | ours FedAvg | "
         "ours alpha mean | ours iota end | share of g·Δprev < 0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for wl, r in df.iterrows():
        J.append(f"| {wl} | " + " | ".join(fmt(r.get(f"{s}_{a}_jit"), 4) for a in M for s in ("paper", "our")) +
                 f" | {fmt(r.get('alpha_mean'), 3)} | {fmt(r.get('iota_end'), 3)} | {fmt(r.get('gdot_neg'), 3)} |")
    txt = open(REP).read() if os.path.exists(REP) else ""
    head = txt.split("<!-- tables -->")[0] if "<!-- tables -->" in txt else "# Paper vs ours: gap analysis\n\n"
    body = ["<!-- tables -->", "", "## Time to target (virtual s) and local trainings to target (n)", "",
            "Paper times are read from the digitised paper curves; n = received local updates (FedAvg: device "
            "trainings, 10 per round).", ""] + L + \
           ["", "## Curve roughness and FedASMU control parameters", "",
            "Roughness = mean |Δ accuracy| between 50 equally spaced points in the second half of the run.", ""] + J
    open(REP, "w").write(head + "\n".join(body) + "\n")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
