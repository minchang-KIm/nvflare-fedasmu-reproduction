"""Side-by-side comparison with the FedASMU paper appendix (arXiv 2312.05770v1,
Figures 3-6): for every workload, the paper's own panel on the left and our run on the
right, drawn on the paper's axes (same x/y limits, ticks and units: "Time (s)" vs
"Accuracy").  Runs of the ``appendix`` profile are read while they are still running,
so the figures can be refreshed in real time (scripts/live_sync.sh).

- Left: the panel cropped unchanged from the paper PDF (CC BY 4.0) by
  scripts/extract_paper_panels.py, which also measured its axes (axes.json).
- Right: FedASMU, FedAvg and FedBuff of our runs, in the paper's colours and markers.
  Time is the virtual (simulated) time of our discrete-event simulation; the paper's
  axis is also simulated time of its own system, whose time model is not published.
  The paper plots 10 methods; only these 3 are implemented here.
- A curve that leaves the paper's axes is cut there, and the panel says so.
  appendix_figN_fullrange.png repeats the figure with the right panel's limits widened
  just enough to show all of our data (ticks keep the paper's spacing).
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
OUT = os.path.join(ROOT, "results", "figures", "appendix")
PAPER = os.path.join(ROOT, "results", "figures", "paper_appendix")

FIGS = {
    "fig3": ("CIFAR-10 and Tiny-ImageNet", [("lenet5", "cifar10"), ("cnn", "cifar10"), ("resnet20", "tinyimagenet")]),
    "fig4": ("CIFAR-100", [("lenet5", "cifar100"), ("cnn", "cifar100"), ("resnet20", "cifar100")]),
    "fig5": ("CIFAR-10 and IMDb", [("alexnet", "cifar10"), ("vgg11", "cifar10"), ("textcnn", "imdb")]),
    "fig6": ("CIFAR-100 and FMNIST", [("alexnet", "cifar100"), ("vgg11", "cifar100"), ("lenet5", "fmnist")]),
}
NAMES = {"lenet5": "LeNet", "cnn": "CNN", "resnet20": "ResNet", "alexnet": "AlexNet", "vgg11": "VGG",
         "textcnn": "TextCNN", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tinyimagenet": "Tiny-ImageNet",
         "imdb": "IMDb", "fmnist": "FMNIST"}
# colours and markers as in the paper's legend (rainbow colormap: red, orange-red, blue)
ALGOS = [("fedasmu", "FedASMU", "#ff0000", "."), ("fedavg", "FedAvg", "#ff5b2e", "o"),
         ("fedbuff", "FedBuff", "#4c50f1", "<")]


def load(profile, model, ds, algo, seed):
    d = os.path.join(MET, f"{profile}_{model}_{ds}_{algo}_s{seed}")
    try:
        ev = pd.read_csv(os.path.join(d, "eval.csv")).drop_duplicates(subset=["received", "global_version"], keep="last")
    except Exception:
        return None, None
    try:
        st = json.load(open(os.path.join(d, "status.json"))).get("status")
    except Exception:
        st = "?"
    return ev, st


def ours(ax, profile, model, ds, seed, axes, full=False):
    """Our curves on the paper panel's exact axes (full=True: limits widened just enough
    to show all of our data, ticks kept at the paper's spacing). Returns status text."""
    x0, x1 = axes["xlim"]
    y0, y1 = axes["ylim"]
    status, notes, xs, ys = [], [], [], []
    for algo, label, color, mk in ALGOS:
        ev, st = load(profile, model, ds, algo, seed)
        if ev is None or not len(ev):
            status.append(f"{label}: not started")
            continue
        lab = label if st == "COMPLETED" else f"{label} (running)"
        n = len(ev)
        ax.plot(ev.vtime, ev.test_acc, color=color, lw=1.6, marker=mk, ms=4, markevery=max(n // 5, 1), label=lab)
        status.append(f"{label}: {st}")
        xs += [ev.vtime.min(), ev.vtime.max()]
        ys += [ev.test_acc.min(), ev.test_acc.max()]
        if full:
            continue
        if ev.vtime.max() > x1:
            notes.append(f"{label} runs to {ev.vtime.max() / 1000:.0f}k s (cut)")
        if ev.test_acc.max() > y1:
            notes.append(f"{label} above y-max (max {ev.test_acc.max():.3f})")
        if (ev.test_acc < y0).any():
            notes.append(f"{label} below y-min (min {ev.test_acc.min():.3f})")
    xt, yt = list(axes["xticks"]), list(axes["yticks"])
    if full and xs:
        dx, dy = xt[1] - xt[0], yt[1] - yt[0]
        while max(xs) > xt[-1] + dx / 2:
            xt.append(xt[-1] + dx)
        while max(ys) > yt[-1]:
            yt.append(round(yt[-1] + dy, 6))
        while min(ys) < yt[0]:
            yt.insert(0, round(yt[0] - dy, 6))
        x1 = max(x1, max(xs) + 0.05 * (max(xs) - x0))
        y1 = max(y1, max(ys) + 0.05 * (max(ys) - y0))
        y0 = min(y0, min(ys) - 0.05 * (y1 - min(ys)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([v for v in xt if x0 <= v <= x1])
    ax.set_yticks([v for v in yt if y0 <= v <= y1])
    ax.ticklabel_format(axis="x", style="plain")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_box_aspect((axes["frame_px_300dpi"][3] - axes["frame_px_300dpi"][1]) /
                      (axes["frame_px_300dpi"][2] - axes["frame_px_300dpi"][0]))
    ax.legend(fontsize=6.5, loc="lower right")
    if notes:
        ax.text(0.01, 0.99, "\n".join(notes), transform=ax.transAxes, fontsize=6, va="top", color="0.35")
    return status


def pair(axl, axr, profile, model, ds, seed, letter, axes_all, full=False):
    key = f"{model}-{ds}"
    axl.imshow(mpimg.imread(os.path.join(PAPER, f"{key}.png")))
    axl.axis("off")
    axl.set_title(f"({letter}) {NAMES[model]} & {NAMES[ds]} - paper", fontsize=9)
    st = ours(axr, profile, model, ds, seed, axes_all[key], full)
    axr.set_title(f"({letter}) {NAMES[model]} & {NAMES[ds]} - ours (NVFLARE)" + (", widened axes" if full else ""),
                  fontsize=9)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="appendix")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    axes_all = json.load(open(os.path.join(PAPER, "axes.json")))
    src = ("Left: paper panel, Liu et al. AAAI-24, arXiv 2312.05770v1 (CC BY 4.0), 10 methods. "
           f"Right: our run on the same axes, FedASMU / FedAvg / FedBuff, virtual time, seed {a.seed}.")
    big, bax = plt.subplots(6, 4, figsize=(17, 22))
    lines, i = [], 0
    for fig, (title, panels) in FIGS.items():
        for full in (False, True):
            f, ax = plt.subplots(3, 2, figsize=(11, 12.5))
            for r, (m, d) in enumerate(panels):
                st = pair(ax[r, 0], ax[r, 1], a.profile, m, d, a.seed, "abc"[r], axes_all, full)
                if not full:
                    pair(bax[i // 2, 2 * (i % 2)], bax[i // 2, 2 * (i % 2) + 1], a.profile, m, d, a.seed, "abc"[r],
                         axes_all)
                    lines.append(f"{m}-{d}: " + "; ".join(st))
                    i += 1
            how = ("right panels widened only where our data leave the paper's axes" if full
                   else "right panels on the paper's exact axes")
            f.suptitle(f"Paper appendix {fig.replace('fig', 'Figure ')} ({title}): paper vs ours, {how}\n{src}",
                       fontsize=9)
            f.tight_layout(rect=(0, 0, 1, 0.97))
            f.savefig(os.path.join(OUT, f"appendix_{fig}{'_fullrange' if full else ''}.png"), dpi=110)
            plt.close(f)
    big.suptitle("FedASMU paper appendix Figures 3-6 vs our NVFLARE reproduction (all 12 workloads)\n" + src, fontsize=11)
    big.tight_layout(rect=(0, 0, 1, 0.98))
    big.savefig(os.path.join(OUT, "appendix_all.png"), dpi=80)
    plt.close(big)
    with open(os.path.join(OUT, "status.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
