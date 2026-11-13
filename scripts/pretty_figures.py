"""Presentation versions: paper panel (left) | our curves (right), FedASMU and FedBuff only
(FedAvg not drawn), no overlay, no cut curves.  Our curves are smoothed with a centred moving
average over 9 evaluations (the same for every curve, shown in the title); tables and reports use
the raw values.
  tuned_cifar10.png : LeNet / CNN & CIFAR-10, FedASMU with the validation-selected constants
                      (mu_alpha 1, mu_beta 10), FedASMU default (thin dashed), FedBuff;
                      time factor of results/figures/appendix/time_scale.json
  vgg11_pilot.png   : VGG-11 & CIFAR-10 / CIFAR-100, the pilot runs behind the bar chart,
                      virtual time unscaled"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from appendix_figures import NAMES, PAPER, ROOT, load  # noqa: E402

OUT = os.path.join(ROOT, "results", "figures", "pretty")
W = 9
RED, BLUE = "#e8000b", "#3a3fd9"


def panel(ax, curves, axes, scale, title):
    xmax, ymax = axes["xlim"][1], axes["ylim"][1]
    for prof, algo, lab, col, mk, lw, ls, alpha in curves:
        ev, st = load(prof, *algo)
        if ev is None or not len(ev):
            continue
        y = ev.test_acc.rolling(W, center=True, min_periods=1).mean()
        t = ev.vtime * scale
        ax.plot(t, y, color=col, lw=lw, ls=ls, alpha=alpha, marker=mk, ms=6, markevery=max(len(ev) // 6, 1),
                markeredgecolor="white", markeredgewidth=0.6, label=lab, solid_capstyle="round")
        xmax, ymax = max(xmax, t.max() * 1.03), max(ymax, y.max() + 0.02)
    step = axes["xticks"][1] - axes["xticks"][0]
    while (xmax - axes["xlim"][0]) / step > 7:
        step *= 2
    ax.set_xticks([axes["xticks"][0] + i * step for i in range(int((xmax - axes["xticks"][0]) / step) + 1)])
    dy = axes["yticks"][1] - axes["yticks"][0]
    ax.set_yticks([round(axes["yticks"][0] + i * dy, 6) for i in range(int((ymax - axes["yticks"][0]) / dy) + 1)])
    ax.set_xlim(axes["xlim"][0], xmax)
    ax.set_ylim(axes["ylim"][0], ymax)
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title(title, fontsize=10)


def figure(rows, fname, suptitle):
    axes_all = json.load(open(os.path.join(PAPER, "axes.json")))
    f, ax = plt.subplots(len(rows), 2, figsize=(12, 4.6 * len(rows)), squeeze=False)
    for r, (key, m, d, curves, scale, note) in enumerate(rows):
        ax[r, 0].imshow(mpimg.imread(os.path.join(PAPER, f"{key}.png")))
        ax[r, 0].axis("off")
        ax[r, 0].set_title(f"{NAMES[m]} & {NAMES[d]} - paper", fontsize=10)
        panel(ax[r, 1], curves, axes_all[key], scale, f"{NAMES[m]} & {NAMES[d]} - ours{note}")
    f.suptitle(suptitle, fontsize=10)
    f.tight_layout(rect=(0, 0, 1, 0.95))
    f.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = json.load(open(os.path.join(ROOT, "results/figures/appendix/time_scale.json")))
    rows = []
    for m in ("lenet5", "cnn"):
        k = f"{m}-cifar10"
        curves = [("tuned", (m, "cifar10", "fedasmu-m1b10", 0), "FedASMU", RED, "o", 2.4, "-", 1.0),
                  ("appendix", (m, "cifar10", "fedbuff", 0), "FedBuff", BLUE, "<", 2.4, "-", 1.0),
                  ("appendix", (m, "cifar10", "fedasmu", 0), "FedASMU (default constants)", RED, None, 1.2, "--",
                   0.55)]
        rows.append((k, m, "cifar10", curves, sc[k]["scale"], f", time x{sc[k]['scale']:.2f}"))
    figure(rows, "tuned_cifar10.png", "FedASMU with constants chosen on a validation split (mu_alpha 1, mu_beta 10) "
           f"vs FedBuff; {W}-point moving average on all curves")
    rows = []
    for d in ("cifar10", "cifar100"):
        curves = [("pilot", ("vgg11", d, "fedasmu", 0), "FedASMU", RED, "o", 2.4, "-", 1.0),
                  ("pilot", ("vgg11", d, "fedbuff", 0), "FedBuff", BLUE, "<", 2.4, "-", 1.0)]
        rows.append((f"vgg11-{d}", "vgg11", d, curves, 1.0, ""))
    figure(rows, "vgg11_pilot.png", f"VGG-11 pilot runs (bar chart values) vs paper; {W}-point moving average")


if __name__ == "__main__":
    main()
