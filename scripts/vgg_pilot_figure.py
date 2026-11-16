"""VGG-11 / CIFAR-10 and CIFAR-100: the runs behind the bar chart "final accuracy (%)"
(FedBuff 66.2 / FedASMU 67.3 on CIFAR-10, 26.5 / 29.2 on CIFAR-100) = profile 'pilot'
(FedASMU, FedBuff) and profile 'fedavg' (FedAvg), seed 0, compared with the paper's panels
(arXiv 2312.05770v1 appendix Figures 5b and 6b, CC BY 4.0).

Columns: paper panel (unchanged crop) | our FedASMU and FedBuff curves (the two methods of the
bar chart) on the paper's x axis, y axis widened only as far as needed so that nothing is cut.
The four runs were continued from their 5,000-update checkpoints to 9,500 updates (profile
'pilotext', same config, only the budget raised) so that the curves cover the paper's time axis
with real data; until a continuation exists the 5,000-update run is drawn.  The dotted vertical
line marks 5,000 updates, where the bar-chart values were taken; the legend gives those values
(mean of the last 5 evaluations up to 5,000 updates).
Smoothing (all curves alike; tables use raw values): a centred running median over 5 evaluations
(removes single-evaluation dips such as 0.41 at 2,700 updates on CIFAR-10) followed by a centred
moving average over 9 evaluations.  Time = our virtual time, unscaled."""
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
RUNS = [("FedASMU", "pilot", "fedasmu", "#e8000b", "o"), ("FedBuff", "pilot", "fedbuff", "#3a3fd9", "<")]
SMOOTH = 9
WL = [("cifar10", "VGG-11 & CIFAR-10", "Figure 5(b)"), ("cifar100", "VGG-11 & CIFAR-100", "Figure 6(b)")]


def ev_of(prof, ds, algo):
    ext = os.path.join(MET, f"{prof}ext_vgg11_{ds}_{algo}_s0", "eval.csv")
    path = ext if os.path.exists(ext) else os.path.join(MET, f"{prof}_vgg11_{ds}_{algo}_s0", "eval.csv")
    ev = pd.read_csv(path)
    ev = ev.drop_duplicates(subset=["received", "global_version"], keep="last").reset_index(drop=True)
    ev.attrs["run"] = os.path.basename(os.path.dirname(path))
    return ev


def main():
    os.makedirs(OUT, exist_ok=True)
    axes = json.load(open(os.path.join(PAPER, "axes.json")))
    f, ax = plt.subplots(2, 2, figsize=(12, 9.5))
    rows = []
    for r, (ds, title, pfig) in enumerate(WL):
        key = f"vgg11-{ds}"
        a = axes[key]
        ax[r, 0].imshow(mpimg.imread(os.path.join(PAPER, f"{key}.png")))
        ax[r, 0].axis("off")
        ax[r, 0].set_title(f"{title} - paper ({pfig})", fontsize=10)
        data = {n: ev_of(p, ds, al) for n, p, al, _, _ in RUNS}
        sm = {n: d.test_acc.rolling(5, center=True, min_periods=1).median()
                    .rolling(SMOOTH, center=True, min_periods=1).mean() for n, d in data.items()}
        bar = {n: d[d.received <= 5000].tail(5).test_acc.mean() for n, d in data.items()}
        ymax = max(a["ylim"][1], max(v.max() for v in sm.values()) + 0.02)
        xmax = max(a["xlim"][1], max(d.vtime.max() for d in data.values()) * 1.03)
        dy = a["yticks"][1] - a["yticks"][0]
        yt = [round(a["yticks"][0] + i * dy, 6) for i in range(int((ymax - a["yticks"][0]) / dy) + 1)]
        x = ax[r, 1]
        for n, p, al, col, mk in RUNS:
            d = data[n]
            x.plot(d.vtime, sm[n], color=col, lw=2.4, marker=mk, ms=6, markevery=max(len(d) // 6, 1),
                   markeredgecolor="white", markeredgewidth=0.6, solid_capstyle="round",
                   label=f"{n} ({bar[n] * 100:.1f}%)")
            t5 = d[d.received <= 5000].vtime.iloc[-1]
            x.axvline(t5, color=col, lw=0.8, ls=":", alpha=0.6)
        x.set_xlim(a["xlim"][0], xmax)
        x.set_ylim(a["ylim"][0], ymax)
        x.set_xticks(a["xticks"])
        x.set_yticks(yt)
        x.ticklabel_format(axis="x", style="plain")
        x.set_xlabel("Time (s)")
        x.set_ylabel("Accuracy")
        x.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        x.legend(fontsize=9, loc="lower right", frameon=False)
        x.set_title(f"{title} - ours", fontsize=10)
        for n, p, al, _, _ in RUNS:
            rows.append(dict(workload=key, method=n, run=data[n].attrs["run"],
                             final_acc_pct_at_5000=round(bar[n] * 100, 2),
                             final_acc_pct_at_end=round(data[n].tail(5).test_acc.mean() * 100, 2),
                             last_received=int(data[n].received.iloc[-1]),
                             last_vtime_s=round(data[n].vtime.iloc[-1])))
    f.suptitle("VGG-11: paper (left) vs our runs behind the bar chart (right)\n"
               "legend = final accuracy at 5,000 updates (dotted line)", fontsize=10)
    f.tight_layout(rect=(0, 0, 1, 0.94))
    f.savefig(os.path.join(OUT, "vgg11_pilot_vs_paper.png"), dpi=110)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "final_acc.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
