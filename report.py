"""Aggregate experiment results into CSV tables, figures and a markdown comparison.

python report.py [--profiles pilot paper ablation]

Only measured values from results/metrics/<run_id>/ are used.  A run without
status COMPLETED is reported with its actual status (RUNNING/FAILED/NOT RUN).
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
MET = os.path.join(ROOT, "results/metrics")
REP = os.path.join(ROOT, "results/reports")
FIG = os.path.join(ROOT, "results/figures")
COLORS = {"fedasmu": "#1f77b4", "fedbuff": "#d62728", "fedbuff_paper": "#ff7f0e", "fedasmu_da": "#2ca02c",
          "fedasmu_fa": "#9467bd", "fedasmu_0": "#8c564b", "fedavg": "#7f7f7f"}
LABEL = {"fedasmu": "FedASMU", "fedbuff": "FedBuff (NVFLARE official)", "fedbuff_paper": "FedBuff (paper staleness)",
         "fedasmu_da": "FedASMU-DA", "fedasmu_fa": "FedASMU-FA", "fedasmu_0": "FedASMU-0", "fedavg": "FedAvg"}


def load_runs(profiles):
    runs = []
    for d in sorted(glob.glob(os.path.join(MET, "*"))):
        if not os.path.isdir(d):
            continue
        rid = os.path.basename(d)
        prof = rid.split("_")[0]
        if profiles and prof not in profiles:
            continue
        man = os.path.join(d, "manifest.json")
        if not os.path.exists(man):
            continue
        m = json.load(open(man))
        c = m["config"]
        st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
        ev = pd.read_csv(os.path.join(d, "eval.csv")) if os.path.exists(os.path.join(d, "eval.csv")) else None
        up = pd.read_csv(os.path.join(d, "updates.csv")) if os.path.exists(os.path.join(d, "updates.csv")) else None
        summ = json.load(open(os.path.join(d, "server_summary.json"))) if os.path.exists(
            os.path.join(d, "server_summary.json")) else {}
        runs.append(dict(run_id=rid, profile=prof, model=c["model"], dataset=c["dataset"], algorithm=c["algorithm"],
                         seed=c["seed"], cfg=c, manifest=m, status=st.get("status", "UNKNOWN"), eval=ev, upd=up,
                         summary=summ, dir=d))
    return runs


def summarize(r):
    ev, up, c = r["eval"], r["upd"], r["cfg"]
    row = dict(run_id=r["run_id"], profile=r["profile"], model=r["model"], dataset=r["dataset"],
               algorithm=r["algorithm"], seed=r["seed"], status=r["status"], git=r["manifest"].get("git"),
               nvflare=r["manifest"].get("nvflare"), clients=c["num_devices"], concurrency=c["max_concurrency"],
               budget_updates=c.get("budget_updates"), target=c.get("target_acc"))
    if ev is not None and len(ev):
        ev = ev.drop_duplicates(subset=["received", "global_version"], keep="last")
        last = ev.tail(5)
        row.update(final_acc=last.test_acc.mean(), final_acc_last=ev.test_acc.iloc[-1], best_acc=ev.test_acc.max(),
                   final_loss=ev.test_loss.iloc[-1], received=int(ev.received.iloc[-1]),
                   global_updates=int(ev.global_version.iloc[-1]), aggregations=int(ev.aggregations.iloc[-1]),
                   total_steps=int(ev.total_steps.iloc[-1]), total_samples=int(ev.total_samples.iloc[-1]),
                   vtime=float(ev.vtime.iloc[-1]), wall_s=float(ev.wall.iloc[-1]))
        hit = ev[ev.test_acc >= c.get("target_acc", 1.1)]
        row["time_to_target_v"] = float(hit.vtime.iloc[0]) if len(hit) else np.nan
        row["updates_to_target"] = int(hit.received.iloc[0]) if len(hit) else np.nan
        row["wall_to_target_s"] = float(hit.wall.iloc[0]) if len(hit) else np.nan
    if up is not None and len(up) and "staleness" in up:
        s = pd.to_numeric(up.staleness, errors="coerce").dropna()
        if len(s):
            row.update(staleness_mean=s.mean(), staleness_median=s.median(), staleness_max=s.max())
        if "merged" in up:
            row["merge_rate"] = float((up.merged.astype(str) == "True").mean())
            rew = pd.to_numeric(up.reward, errors="coerce").dropna()
            row["reward_mean"] = rew.mean() if len(rew) else np.nan
            b = pd.to_numeric(up.beta, errors="coerce").dropna()
            row["beta_mean"] = b.mean() if len(b) else np.nan
        a = pd.to_numeric(up.get("alpha"), errors="coerce").dropna() if "alpha" in up else []
        if len(a):
            row["alpha_mean"] = a.mean()
        row["failed_updates"] = int((up.slot_src.astype(str).str.startswith("failed")).sum()) if "slot_src" in up else 0
    p = c.get("paper", {})
    pk = r["algorithm"] if r["algorithm"] in ("fedasmu",) else ("fedbuff" if r["algorithm"].startswith("fedbuff") else None)
    if pk:
        row["paper_acc"] = p.get(pk)
        row["paper_time"] = p.get(f"{pk}_time")
    return row


def fig_curves(group, key, title, fname, xlabel, xcol):
    plt.figure(figsize=(6.4, 4.2))
    for r in group:
        ev = r["eval"]
        if ev is None or not len(ev):
            continue
        plt.plot(ev[xcol], ev.test_acc, label=f"{LABEL.get(r['algorithm'], r['algorithm'])} (s{r['seed']})",
                 color=COLORS.get(r["algorithm"]), lw=1.4, alpha=0.9 if r["seed"] == 0 else 0.5)
    tgt = group[0]["cfg"].get("target_acc")
    if tgt:
        plt.axhline(tgt, ls=":", color="k", lw=0.8, label=f"target {tgt}")
    plt.xlabel(xlabel)
    plt.ylabel("test accuracy (global model)")
    plt.title(title, fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="*", default=["pilot", "paper", "ablation", "smoke"])
    args = ap.parse_args()
    os.makedirs(REP, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    runs = load_runs(set(args.profiles))
    if not runs:
        print("no runs found")
        return
    summ = pd.DataFrame([summarize(r) for r in runs])
    summ.to_csv(os.path.join(REP, "summary.csv"), index=False)

    # ---- per (profile, model, dataset) figures ----
    keys = sorted({(r["profile"], r["model"], r["dataset"]) for r in runs})
    for prof, model, ds in keys:
        g = [r for r in runs if (r["profile"], r["model"], r["dataset"]) == (prof, model, ds)]
        base = f"{prof}_{model}_{ds}"
        t = f"{model.upper()} + {ds.upper()} [{prof}]"
        fig_curves(g, "received", t + ": accuracy vs received local updates", base + "_acc_vs_updates.png",
                   "received local updates (= FedASMU global updates; FedBuff: K per aggregation)", "received")
        fig_curves(g, "vtime", t + ": accuracy vs simulated time", base + "_acc_vs_vtime.png",
                   "virtual (simulated system) time [s]", "vtime")
        fig_curves(g, "wall", t + ": accuracy vs wall-clock", base + "_acc_vs_wall.png", "wall-clock time [s]", "wall")
        # staleness distribution
        plt.figure(figsize=(6.4, 4.0))
        for r in g:
            up = r["upd"]
            if up is None or "staleness" not in up:
                continue
            s = pd.to_numeric(up.staleness, errors="coerce").dropna()
            if len(s):
                plt.hist(s, bins=np.arange(0, s.max() + 2) - 0.5, alpha=0.5, density=True,
                         label=LABEL.get(r["algorithm"], r["algorithm"]), color=COLORS.get(r["algorithm"]))
        plt.xlabel("staleness at arrival (FedASMU: t-o+1; FedBuff: version diff)")
        plt.ylabel("fraction of updates")
        plt.title(t + ": staleness distribution", fontsize=10)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, base + "_staleness.png"), dpi=130)
        plt.close()

    # ---- paper vs reproduced ----
    main_algos = summ[summ.algorithm.isin(["fedasmu", "fedbuff"]) & summ.profile.isin(["pilot", "paper"])
                      & (summ.status == "COMPLETED")]
    if len(main_algos):
        agg = main_algos.groupby(["profile", "model", "dataset", "algorithm"]).agg(
            final_acc=("final_acc", "mean"), final_acc_std=("final_acc", "std"), best_acc=("best_acc", "mean"),
            n_seeds=("seed", "count"), paper_acc=("paper_acc", "first"), time_to_target_v=("time_to_target_v", "mean"),
            paper_time=("paper_time", "first"), wall_s=("wall_s", "mean"), received=("received", "mean"),
            global_updates=("global_updates", "mean"), staleness_mean=("staleness_mean", "mean")).reset_index()
        agg["diff_pp"] = (agg.final_acc - agg.paper_acc) * 100
        agg.to_csv(os.path.join(REP, "paper_comparison.csv"), index=False)
        for prof in agg.profile.unique():
            a = agg[agg.profile == prof]
            labels = [f"{m}\n{d}\n{al}" for m, d, al in zip(a.model, a.dataset, a.algorithm)]
            x = np.arange(len(a))
            plt.figure(figsize=(max(6, 1.1 * len(a)), 4.2))
            plt.bar(x - 0.2, a.paper_acc, 0.4, label="paper (Table 1/2)", color="#bbbbbb")
            plt.bar(x + 0.2, a.final_acc, 0.4, yerr=a.final_acc_std.fillna(0), label="reproduced (final)",
                    color=[COLORS.get(al) for al in a.algorithm])
            plt.xticks(x, labels, fontsize=7)
            plt.ylabel("accuracy")
            plt.title(f"Paper vs reproduced accuracy [{prof}]", fontsize=10)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG, f"{prof}_paper_vs_reproduced.png"), dpi=130)
            plt.close()
            # time to target
            plt.figure(figsize=(max(6, 1.1 * len(a)), 4.0))
            plt.bar(x, a.time_to_target_v.fillna(0), color=[COLORS.get(al) for al in a.algorithm])
            for xi, v in zip(x, a.time_to_target_v):
                if np.isnan(v):
                    plt.text(xi, 0, "not\nreached", ha="center", va="bottom", fontsize=7)
            plt.xticks(x, labels, fontsize=7)
            plt.ylabel("virtual time to target accuracy [s]")
            plt.title(f"Time to target accuracy (simulated system time) [{prof}]", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG, f"{prof}_time_to_target.png"), dpi=130)
            plt.close()
    print(summ[["run_id", "status", "final_acc", "best_acc", "time_to_target_v", "received", "global_updates",
                "wall_s"]].to_string())


if __name__ == "__main__":
    main()
