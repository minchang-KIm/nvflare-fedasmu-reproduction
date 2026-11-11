#!/usr/bin/env bash
# Runs the appendix profile for the given workloads on this host, one run at a time
# (FedASMU, FedBuff, FedAvg per workload).  Before every run it waits until every GPU has
# at least MIN_FREE_MB free memory, so that it does not start on top of another user's job.
# Usage: bash scripts/queue_appendix.sh NAME model_dataset [model_dataset ...]
cd "$(dirname "$0")/.." || exit 1
NAME=$1; shift
MIN_FREE_MB=${MIN_FREE_MB:-8000}
mkdir -p logs
for w in "$@"; do
  for a in fedasmu fedbuff fedavg; do
    rid="appendix_${w}_${a}_s0"
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
      [ "${free:-0}" -ge "$MIN_FREE_MB" ] && break
      echo "$(date '+%F %T') $rid waiting: GPU free ${free} MiB < ${MIN_FREE_MB} MiB" >> "logs/queue_appendix_${NAME}.log"
      sleep 120
    done
    echo "$(date '+%F %T') start $rid" >> "logs/queue_appendix_${NAME}.log"
    .venv/bin/python scripts/run_experiment.py --profile appendix --only "^${rid}$" > "logs/${rid}.out" 2>&1
    echo "$(date '+%F %T') end $rid rc=$? $(grep -o '"status": "[A-Z]*"' "results/metrics/${rid}/status.json")" >> "logs/queue_appendix_${NAME}.log"
  done
done
touch "logs/queue_appendix_${NAME}_done"
