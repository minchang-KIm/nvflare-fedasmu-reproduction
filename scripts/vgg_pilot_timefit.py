"""Edited copy of results/figures/vgg_pilot/vgg11_pilot_vs_paper.png (which is kept unchanged):
paper panel (left) | our VGG-11 pilot runs behind the bar chart (right), FedASMU and FedBuff.

The paper gives no time per epoch, so our virtual time axis is multiplied by one factor per
workload (same for both methods), fitted the same way as scripts/appendix_figures.py: geometric mean
of t_paper / t_ours at 50 %, 80 % and 90 % of each curve's own rise, FedASMU and FedBuff.
Smoothing (both curves alike; the legend values are raw = mean of the last 5 evaluations): centred
running median over 5 evaluations, then centred moving average over 9 evaluations.
Output: results/figures/vgg_pilot/vgg11_pilot_vs_paper_timefit.png (+ timefit_scale.json)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from appendix_figures import PAPER, ROOT, rise_time  # noqa: E402

MET, OUT = os.path.join(ROOT, "results/metrics"), os.path.join(ROOT, "results/figures/vgg_pilot")
RUNS = [("FedASMU", "fedasmu", "#e8000b", "o"), ("FedBuff", "fedbuff", "#3a3fd9", "<")]
WL = [("cifar10", "VGG-11 & CIFAR-10", "Figure 5(b)"), ("cifar100", "VGG-11 & CIFAR-100", "Figure 6(b)")]


def main():
    axes = json.load(open(os.path.join(PAPER, "axes.json")))
    pc = pd.read_csv(os.path.join(PAPER, "curves.csv"))
    f, ax = plt.subplots(2, 2, figsize=(12, 9.5))
    scales = {}
    for r, (ds, title, pfig) in enumerate(WL):
        key, a = f"vgg11-{ds}", axes[f"vgg11-{ds}"]
        data = {n: pd.read_csv(os.path.join(MET, f"pilot_vgg11_{ds}_{al}_s0", "eval.csv"))
                .drop_duplicates(subset=["received", "global_version"], keep="last").reset_index(drop=True)
                for n, al, _, _ in RUNS}
        ratios = []
        for n, d in data.items():
            P = pc[(pc.workload == key) & (pc.method == n)].sort_values("time_s")
            for fr in (0.5, 0.8, 0.9):
                ratios.append(rise_time(P.time_s, P.accuracy, fr) / rise_time(d.vtime, d.test_acc, fr))
        s = float(np.exp(np.mean(np.log(ratios))))
        scales[key] = {"scale": s, "ratios": [round(v, 3) for v in ratios]}
        ax[r, 0].imshow(mpimg.imread(os.path.join(PAPER, f"{key}.png")))
        ax[r, 0].axis("off")
        ax[r, 0].set_title(f"{title} - paper ({pfig})", fontsize=10)
        x, ymax, xmax = ax[r, 1], a["ylim"][1], a["xlim"][1]
        for n, al, col, mk in RUNS:
            d = data[n]
            y = d.test_acc.rolling(5, center=True, min_periods=1).median().rolling(9, center=True, min_periods=1).mean()
            t = d.vtime * s
            x.plot(t, y, color=col, lw=2.4, marker=mk, ms=6, markevery=max(len(d) // 6, 1), markeredgecolor="white",
                   markeredgewidth=0.6, solid_capstyle="round", label=f"{n} ({d.tail(5).test_acc.mean() * 100:.1f}%)")
            ymax, xmax = max(ymax, y.max() + 0.02), max(xmax, t.max() * 1.03)
        dy = a["yticks"][1] - a["yticks"][0]
        x.set_yticks([round(a["yticks"][0] + i * dy, 6) for i in range(int((ymax - a["yticks"][0]) / dy) + 1)])
        x.set_xticks(a["xticks"])
        x.set_xlim(a["xlim"][0], xmax)
        x.set_ylim(a["ylim"][0], ymax)
        x.ticklabel_format(axis="x", style="plain")
        x.set_xlabel("Time (s)")
        x.set_ylabel("Accuracy")
        x.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        x.legend(fontsize=9, loc="lower right", frameon=False)
        x.set_title(f"{title} - ours (time x{s:.2f})", fontsize=10)
    f.suptitle("VGG-11: paper (left) vs our runs behind the bar chart (right)\n"
               "our time axis scaled to the paper's (no time per epoch in the paper)", fontsize=10)
    f.tight_layout(rect=(0, 0, 1, 0.94))
    f.savefig(os.path.join(OUT, "vgg11_pilot_vs_paper_timefit.png"), dpi=110)
    json.dump(scales, open(os.path.join(OUT, "timefit_scale.json"), "w"), indent=1)
    print(scales)


if __name__ == "__main__":
    main()
