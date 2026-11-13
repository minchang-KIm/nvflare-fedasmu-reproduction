#!/usr/bin/env bash
# FedASMU validation tuning on dev-host (3 runs at a time; 6 caused NVFLARE communication
# time-outs at start-up), then the selected setting per workload on the normal split with test
# evaluation, then formula verification of those runs.  $1 = regex of workloads (default: all).
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
W="${1:-.}"
.venv/bin/python - "$W" <<'P' > logs/tune_ids.txt
import re, sys, yaml
e = yaml.safe_load(open("configs/experiments/tune.yaml"))
for r in e["runs"]:
    rid = f"tune_{r['model']}_{r['dataset']}_fedasmu-{r['variant']}_s0"
    if re.search(sys.argv[1], rid):
        print(rid)
P
xargs -P 3 -I{} sh -c '.venv/bin/python scripts/run_experiment.py --profile tune --only "^{}$" > logs/{}.out 2>&1' < logs/tune_ids.txt
.venv/bin/python scripts/select_tuned.py > logs/select_tuned.out 2>&1
.venv/bin/python - "$W" <<'P' > logs/tuned_ids.txt
import re, sys, yaml
e = yaml.safe_load(open("configs/experiments/tuned.yaml"))
for r in e["runs"]:
    rid = f"tuned_{r['model']}_{r['dataset']}_fedasmu-{r['variant']}_s0"
    if re.search(sys.argv[1], rid):
        print(rid)
P
xargs -P 3 -I{} sh -c '.venv/bin/python scripts/run_experiment.py --profile tuned --only "^{}$" > logs/{}.out 2>&1 && .venv/bin/python scripts/verify_fedasmu.py {} > logs/verify_{}.out 2>&1' < logs/tuned_ids.txt
touch "logs/queue_tune_done_$(echo "$W" | tr -c 'a-z0-9' '_')"
