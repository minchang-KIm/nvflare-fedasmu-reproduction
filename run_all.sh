#!/usr/bin/env bash
# Usage: bash run_all.sh --profile smoke|pilot|paper|ablation [--only REGEX] [extra run_experiment args]
# Completed runs are skipped automatically (status.json == COMPLETED); interrupted runs resume
# from their last checkpoint.
set -uo pipefail
cd "$(dirname "$0")"
PROFILE=smoke
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    *) ARGS+=("$1"); shift;;
  esac
done
PY=.venv/bin/python
mkdir -p logs
if [ "$PROFILE" = "smoke" ]; then
  $PY -m pytest -q tests/unit || exit 1
fi
# refuse to start GPU work when the GPUs are busy with foreign jobs (shared hosts)
if [ "$PROFILE" != "smoke" ] && command -v nvidia-smi >/dev/null; then
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  if [ "${FREE:-0}" -lt "${MIN_FREE_MB:-4000}" ]; then
    echo "GPU free memory ${FREE} MiB < ${MIN_FREE_MB:-4000} MiB - another job is using the GPUs; aborting." >&2
    exit 3
  fi
fi
$PY scripts/run_experiment.py --profile "$PROFILE" "${ARGS[@]}" 2>&1 | tee -a "logs/run_all_${PROFILE}.log" | grep -E "^\[|COMPLETED|FAILED|SKIPPED|Traceback"
STATUS=${PIPESTATUS[0]}
$PY report.py
exit $STATUS
