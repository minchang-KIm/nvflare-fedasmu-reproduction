"""Accuracy vs. time figures in the layout of the FedASMU paper appendix (arXiv
2312.05770, Figures 3-6): one panel per workload, test accuracy of the global model
against training time.  Runs of the ``appendix`` profile are read while they are
still running, so the figures can be refreshed in real time (scripts/live_sync.sh).

Time axis: virtual (simulated system) seconds of our discrete-event simulation; the
paper's axis is "Time (s)" of its own simulated system.  The paper's curves cover
10 methods; only FedASMU, FedBuff and FedAvg are implemented here.  Dotted lines mark
the paper's final accuracies (Tables 1-2) of FedASMU and FedBuff.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
OUT = os.path.join(ROOT, "results", "figures", "appendix")

# paper appendix figure -> panels (model, dataset), in the paper's order
FIGS = {
    "fig3": ("CIFAR-10 and Tiny-ImageNet", [("lenet5", "cifar10"), ("cnn", "cifar10"), ("resnet20", "tinyimagenet")]),
    "fig4": ("CIFAR-100", [("lenet5", "cifar100"), ("cnn", "cifar100"), ("resnet20", "cifar100")]),
    "fig5": ("CIFAR-10 and IMDb", [("alexnet", "cifar10"), ("vgg11", "cifar10"), ("textcnn", "imdb")]),
    "fig6": ("CIFAR-100 and FMNIST", [("alexnet", "cifar100"), ("vgg11", "cifar100"), ("lenet5", "fmnist")]),
}
NAMES = {"lenet5": "LeNet", "cnn": "CNN", "resnet20": "ResNet", "alexnet": "AlexNet", "vgg11": "VGG",
         "textcnn": "TextCNN", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tinyimagenet": "Tiny-ImageNet",
         "imdb": "IMDb", "fmnist": "FMNIST"}
ALGOS = [("fedasmu", "FedASMU", "#d62728"), ("fedavg", "FedAvg", "#1f77b4"), ("fedbuff", "FedBuff", "#2ca02c")]


def load(profile, model, ds, algo, seed):
    d = os.path.join(MET, f"{profile}_{model}_{ds}_{algo}_s{seed}")
    p = os.path.join(d, "eval.csv")
    if not os.path.exists(p):
        return None, None
    try:
        ev = pd.read_csv(p).drop_duplicates(subset=["received", "global_version"], keep="last")
    except Exception:
        return None, None
    try:
        st = json.load(open(os.path.join(d, "status.json"))).get("status")
    except Exception:
        st = "?"
    return ev, st


def panel(ax, profile, model, ds, seed, settings, letter):
    ps = settings.get(f"{model}-{ds}", {})
    paper = ps.get("paper", {})
    status = []
    for algo, label, color in ALGOS:
        ev, st = load(profile, model, ds, algo, seed)
        if ev is None or not len(ev):
            status.append(f"{label}: not started")
            continue
        lab = label if st == "COMPLETED" else f"{label} (running, {int(ev.received.iloc[-1])} upd.)"
        ax.plot(ev.vtime, ev.test_acc, color=color, lw=1.3, label=lab)
        status.append(f"{label}: {st}")
    for key, label, color in (("fedasmu", "FedASMU paper final", "#d62728"), ("fedbuff", "FedBuff paper final", "#2ca02c")):
        if paper.get(key) is not None:
            ax.axhline(paper[key], color=color, ls=":", lw=0.9, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"({letter}) {NAMES[model]} & {NAMES[ds]}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="appendix")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    settings = yaml.safe_load(open(os.path.join(ROOT, "configs/common/paper_settings.yaml")))
    big, bax = plt.subplots(4, 3, figsize=(15, 16))
    lines = []
    for r, (fig, (title, panels)) in enumerate(FIGS.items()):
        f, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for c, (m, d) in enumerate(panels):
            st = panel(ax[c], a.profile, m, d, a.seed, settings, "abc"[c])
            panel(bax[r, c], a.profile, m, d, a.seed, settings, "abc"[c])
            lines.append(f"{m}-{d}: " + "; ".join(st))
        f.suptitle(f"Accuracy and training time, {title} (paper appendix {fig.replace('fig', 'Figure ')} layout; "
                   f"virtual time; seed {a.seed})", fontsize=10)
        f.tight_layout()
        f.savefig(os.path.join(OUT, f"appendix_{fig}.png"), dpi=110)
        plt.close(f)
    big.suptitle(f"FedASMU vs FedBuff vs FedAvg on NVFLARE: accuracy vs virtual time, all 12 workloads (seed {a.seed})",
                 fontsize=12)
    big.tight_layout(rect=(0, 0, 1, 0.98))
    big.savefig(os.path.join(OUT, "appendix_all.png"), dpi=90)
    plt.close(big)
    with open(os.path.join(OUT, "status.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
