"""Paper panel (left) vs our run with the validation-selected FedASMU setting (right).

Right panel: FedASMU (setting chosen on the validation split by scripts/select_tuned.py, test
accuracy), the unchanged appendix FedBuff and FedAvg runs, and the default FedASMU run as a thin
dashed line.  Paper axes, colours and markers as in scripts/appendix_figures.py.  The time axis
uses the factor already fitted for the appendix runs (results/figures/appendix/time_scale.json);
it is NOT refitted on the tuned runs.  Two versions are written:
  tuned_figN.png         raw curves
  tuned_figN_smooth.png  every curve (all methods alike) smoothed with a centred moving
                         average over 5 evaluations, because the paper's curves look smoothed
                         (results/reports/paper_gap_analysis.md); accuracies in tables are raw.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from appendix_figures import ALGOS, FIGS, NAMES, PAPER, ROOT, load  # noqa: E402

OUT = os.path.join(ROOT, "results", "figures", "tuned")


def draw(ax, key, model, ds, variant, axes, scale, smooth, paper=None):
    """paper: digitized paper curves (dashed overlay) or None.  The y axis is widened (ticks at
    the paper's spacing) so that no curve is cut; x keeps the paper's range unless ours is longer."""
    x0, x1 = axes["xlim"]
    y0, y1 = axes["ylim"]
    yticks = list(axes["yticks"])
    curves = [("tuned", f"fedasmu-{variant}", f"FedASMU (tuned {variant})", ALGOS[0][2], ALGOS[0][3], 1.8, "-")]
    curves += [("appendix", a, lab, c, mk, 1.6, "-") for a, lab, c, mk in ALGOS[1:]]
    curves += [("appendix", "fedasmu", "FedASMU (default)", "#ff0000", None, 0.8, "--")]
    notes, ymax, xmax = [], y1, x1
    for prof, algo, lab, c, mk, lw, ls in curves:
        ev, st = load(prof, model, ds, algo, 0)
        if ev is None or not len(ev):
            notes.append(f"{lab}: not run yet")
            continue
        y = ev.test_acc.rolling(5, center=True, min_periods=1).mean() if smooth else ev.test_acc
        t = ev.vtime * scale
        ax.plot(t, y, color=c, lw=lw, ls=ls, marker=mk, ms=4, markevery=max(len(ev) // 5, 1),
                label=lab + ("" if st == "COMPLETED" else " (running)"))
        xmax, ymax = max(xmax, t.max() * 1.02), max(ymax, y.max() + 0.02)
    if paper is not None:
        for name, c in (("FedASMU", "#ff0000"), ("FedAvg", "#ff5b2e"), ("FedBuff", "#4c50f1")):
            P = paper[paper.method == name].sort_values("time_s")
            ax.plot(P.time_s, P.accuracy, color=c, lw=1.3, ls=":", label=f"{name} paper")
    dy = yticks[1] - yticks[0]
    while yticks[-1] + dy <= ymax:
        yticks.append(round(yticks[-1] + dy, 6))
    step = axes["xticks"][1] - axes["xticks"][0]
    while (xmax - x0) / step > 7:
        step *= 2
    xt = [axes["xticks"][0] + i * step for i in range(int((xmax - axes["xticks"][0]) / step) + 1)]
    ax.set_xlim(x0, xmax)
    ax.set_ylim(y0, ymax)
    ax.set_xticks(xt)
    ax.set_yticks(yticks)
    ax.ticklabel_format(axis="x", style="plain")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    fp = axes["frame_px_300dpi"]
    ax.set_box_aspect((fp[3] - fp[1]) / (fp[2] - fp[0]))
    ax.legend(fontsize=6.3, loc="lower right", ncol=2 if paper is not None else 1)
    if notes:
        ax.text(0.01, 0.99, "\n".join(notes), transform=ax.transAxes, fontsize=6, va="top", color="0.35")


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = yaml.safe_load(open(os.path.join(ROOT, "configs/experiments/tuned.yaml")))["runs"]
    tuned = {f"{r['model']}-{r['dataset']}": r["variant"] for r in runs}
    axes_all = json.load(open(os.path.join(PAPER, "axes.json")))
    import pandas as pd
    pc = pd.read_csv(os.path.join(PAPER, "curves.csv"))
    scales = json.load(open(os.path.join(ROOT, "results/figures/appendix/time_scale.json")))
    for smooth in (False, True):
        for fig, (title, panels) in FIGS.items():
            panels = [p for p in panels if f"{p[0]}-{p[1]}" in tuned]
            if not panels:
                continue
            f, ax = plt.subplots(len(panels), 3, figsize=(17, 5.0 * len(panels)), squeeze=False)
            for r, (m, d) in enumerate(panels):
                key = f"{m}-{d}"
                s = scales.get(key, {}).get("scale", 1.0)
                ax[r, 0].imshow(mpimg.imread(os.path.join(PAPER, f"{key}.png")))
                ax[r, 0].axis("off")
                ax[r, 0].set_title(f"{NAMES[m]} & {NAMES[d]} - paper", fontsize=9)
                sm = ", 5-point moving average" if smooth else ""
                draw(ax[r, 1], key, m, d, tuned[key], axes_all[key], s, smooth)
                ax[r, 1].set_title(f"{NAMES[m]} & {NAMES[d]} - ours, time x{s:.2f}{sm}", fontsize=9)
                draw(ax[r, 2], key, m, d, tuned[key], axes_all[key], s, smooth, pc[pc.workload == key])
                ax[r, 2].set_title(f"paper (dotted) vs ours (solid), time x{s:.2f}{sm}", fontsize=9)
            f.suptitle(f"Paper {fig.replace('fig', 'Figure ')} ({title}) vs ours with FedASMU constants chosen on a "
                       "validation split\n(FedBuff / FedAvg unchanged; time factor from the default runs, not refitted)",
                       fontsize=9)
            f.tight_layout(rect=(0, 0, 1, 0.95))
            f.savefig(os.path.join(OUT, f"tuned_{fig}{'_smooth' if smooth else ''}.png"), dpi=110)
            plt.close(f)


if __name__ == "__main__":
    main()
