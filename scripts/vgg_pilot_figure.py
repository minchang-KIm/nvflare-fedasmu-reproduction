"""VGG-11 / CIFAR-10 and CIFAR-100: the runs behind the bar chart "final accuracy (%)"
(FedBuff 66.2 / FedASMU 67.3 on CIFAR-10, 26.5 / 29.2 on CIFAR-100) = profile 'pilot'
(FedASMU, FedBuff) and profile 'fedavg' (FedAvg), seed 0, compared with the paper's panels
(arXiv 2312.05770v1 appendix Figures 5b and 6b, CC BY 4.0).

Columns: paper panel (unchanged crop) | our curves on the paper's x axis, y axis widened only
as far as needed so that nothing is cut | paper curves (dashed, read from the PDF vector paths)
and ours (solid) on one axis.  Raw curves, no smoothing; time = our virtual time, unscaled.
Final accuracy = mean of the last 5 evaluations (as in the bar chart)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET, PAPER = os.path.join(ROOT, "results/metrics"), os.path.join(ROOT, "results/figures/paper_appendix")
OUT = os.path.join(ROOT, "results/figures/vgg_pilot")
RUNS = [("FedASMU", "pilot", "fedasmu", "#ff0000", "."), ("FedAvg", "fedavg", "fedavg", "#ff5b2e", "o"),
        ("FedBuff", "pilot", "fedbuff", "#4c50f1", "<")]
WL = [("cifar10", "VGG-11 & CIFAR-10", "Figure 5(b)"), ("cifar100", "VGG-11 & CIFAR-100", "Figure 6(b)")]


def ev_of(prof, ds, algo):
    ev = pd.read_csv(os.path.join(MET, f"{prof}_vgg11_{ds}_{algo}_s0", "eval.csv"))
    return ev.drop_duplicates(subset=["received", "global_version"], keep="last")


def main():
    os.makedirs(OUT, exist_ok=True)
    axes = json.load(open(os.path.join(PAPER, "axes.json")))
    pc = pd.read_csv(os.path.join(PAPER, "curves.csv"))
    f, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    rows = []
    for r, (ds, title, pfig) in enumerate(WL):
        key = f"vgg11-{ds}"
        a = axes[key]
        ax[r, 0].imshow(mpimg.imread(os.path.join(PAPER, f"{key}.png")))
        ax[r, 0].axis("off")
        ax[r, 0].set_title(f"{title} - paper ({pfig}, 10 methods)", fontsize=10)
        data = {n: ev_of(p, ds, al) for n, p, al, _, _ in RUNS}
        ymax = max(a["ylim"][1], max(d.test_acc.max() for d in data.values()) + 0.02)
        xmax = max(a["xlim"][1], max(d.vtime.max() for d in data.values()) * 1.03)
        dy = a["yticks"][1] - a["yticks"][0]
        yt = [round(a["yticks"][0] + i * dy, 6) for i in range(int((ymax - a["yticks"][0]) / dy) + 1)]
        for c, overlay in ((1, False), (2, True)):
            x = ax[r, c]
            for n, p, al, col, mk in RUNS:
                d = data[n]
                x.plot(d.vtime, d.test_acc, color=col, lw=1.5, marker=mk, ms=4, markevery=max(len(d) // 6, 1),
                       label=f"{n} ours ({d.tail(5).test_acc.mean() * 100:.1f}%)")
                if overlay:
                    P = pc[(pc.workload == key) & (pc.method == n)].sort_values("time_s")
                    x.plot(P.time_s, P.accuracy, color=col, lw=1.5, ls="--",
                           label=f"{n} paper ({P.accuracy.iloc[-1] * 100:.1f}%)")
            x.set_xlim(a["xlim"][0], xmax)
            x.set_ylim(a["ylim"][0], ymax)
            x.set_xticks(a["xticks"])
            x.set_yticks(yt)
            x.ticklabel_format(axis="x", style="plain")
            x.set_xlabel("Time (s)")
            x.set_ylabel("Accuracy")
            x.grid(alpha=0.3)
            x.legend(fontsize=7.5, loc="lower right", ncol=2 if overlay else 1)
            x.set_title(f"{title} - " + ("paper (dashed) vs ours (solid)" if overlay else
                                         "ours (paper's x axis; y up to %.2f so nothing is cut)" % ymax), fontsize=10)
        for n, p, al, _, _ in RUNS:
            rows.append(dict(workload=key, method=n, run=f"{p}_vgg11_{ds}_{al}_s0",
                             final_acc_pct=round(data[n].tail(5).test_acc.mean() * 100, 2),
                             last_vtime_s=round(data[n].vtime.iloc[-1])))
    f.suptitle("VGG-11: runs behind the bar chart (FedASMU / FedBuff = profile 'pilot', FedAvg = profile 'fedavg', "
               "seed 0) vs the paper's appendix panels\nraw curves, virtual time not rescaled; final accuracy = mean "
               "of the last 5 evaluations", fontsize=10)
    f.tight_layout(rect=(0, 0, 1, 0.94))
    f.savefig(os.path.join(OUT, "vgg11_pilot_vs_paper.png"), dpi=110)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "final_acc.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
