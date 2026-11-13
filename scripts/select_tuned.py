"""Pick the FedASMU setting per workload from the validation runs (profile 'tune') with the
rule fixed in configs/experiments/tune.yaml, and write configs/experiments/tuned.yaml.

Rule: highest mean validation accuracy of the last 5 evaluations; if within 0.002 of the best,
the earlier time to the workload's target accuracy wins.  Only validation numbers are read."""
import glob
import json
import os
import re

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings = yaml.safe_load(open(os.path.join(ROOT, "configs/common/paper_settings.yaml")))
rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "results/metrics/tune_*_s0"))):
    rid = os.path.basename(d)
    m = re.match(r"tune_(\w+?)_(\w+)_fedasmu-m(\d+)b(\d+)_s0", rid)
    st = json.load(open(os.path.join(d, "status.json"))).get("status")
    if not m or st != "COMPLETED":
        continue
    model, ds, ma, mb = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
    ev = pd.read_csv(os.path.join(d, "eval.csv")).drop_duplicates(subset=["received", "global_version"], keep="last")
    tgt = settings[f"{model}-{ds}"]["target"]
    hit = ev[ev.test_acc >= tgt]
    rows.append(dict(workload=f"{model}-{ds}", model=model, dataset=ds, mu_alpha=ma, mu_beta=mb,
                     val_final=ev.tail(5).test_acc.mean(), val_time=float(hit.vtime.iloc[0]) if len(hit) else float("inf")))
df = pd.DataFrame(rows)
out = os.path.join(ROOT, "results/reports/tuning_validation.csv")
df.to_csv(out, index=False)
runs = []
for w, g in df.groupby("workload"):
    best = g.val_final.max()
    pick = g[g.val_final >= best - 0.002].sort_values("val_time").iloc[0]
    print(w, "->", f"mu_alpha={pick.mu_alpha:g} mu_beta={pick.mu_beta:g}", f"val={pick.val_final:.4f}", f"(n={len(g)})")
    runs.append({"model": pick.model, "dataset": pick.dataset, "algorithm": "fedasmu",
                 "variant": f"m{int(pick.mu_alpha)}b{int(pick.mu_beta)}",
                 "fedasmu": {"mu_alpha": float(pick.mu_alpha), "mu_beta": float(pick.mu_beta)}})
hdr = ("# Written by scripts/select_tuned.py from the validation runs (results/reports/tuning_validation.csv).\n"
       "# Normal appendix split and test evaluation; compared with the appendix FedBuff / FedAvg runs.\n")
with open(os.path.join(ROOT, "configs/experiments/tuned.yaml"), "w") as f:
    f.write(hdr + yaml.safe_dump({"overrides": {"trace": True}, "runs": runs}, sort_keys=False))
