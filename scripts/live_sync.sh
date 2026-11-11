#!/usr/bin/env bash
# Live results loop for the appendix profile (runs on the development host):
#  every INTERVAL s : copy the runs of the GPU host (git remote "node2"), redraw the appendix
#                     figures (scripts/appendix_figures.py), verify newly completed runs
#                     (scripts/verify_fedasmu.py);
#  every SYNC_MIN min, or when a run has completed: commit figures, verification reports and
#                     run metrics (running runs: eval.csv/status only), push to node2 and publish
#                     to GitHub with .git/publish_sanitized.sh.
# Stops after both queues have finished and a last sync has been made.
cd "$(dirname "$0")/.." || exit 1
INTERVAL=${INTERVAL:-300}
SYNC_MIN=${SYNC_MIN:-60}
REMOTE=$(git remote get-url node2)          # host:path of the GPU host checkout
RHOST=${REMOTE%%:*}; RPATH=${REMOTE#*:}
LOG=logs/live_sync.log
PY=.venv/bin/python
last_sync=0; last_done=-1
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
while true; do
  rsync -a --include='appendix_*/***' --exclude='*' "$RHOST:$RPATH/results/metrics/" results/metrics/ 2>>"$LOG"
  $PY scripts/appendix_figures.py > /dev/null 2>>"$LOG"
  done_runs=$(grep -l '"status": "COMPLETED"' results/metrics/appendix_*/status.json 2>/dev/null | wc -l)
  for s in $(grep -l '"status": "COMPLETED"' results/metrics/appendix_*/status.json 2>/dev/null); do
    rid=$(basename "$(dirname "$s")")
    [ -f "results/verification/$rid/checks.json" ] || $PY scripts/verify_fedasmu.py "$rid" >> "$LOG" 2>&1
  done
  finished=0
  [ -f logs/queue_appendix_dev_done ] && ssh -n "$RHOST" "[ -f $RPATH/logs/queue_appendix_gpu_done ]" && finished=1
  now=$(date +%s)
  if [ $((now - last_sync)) -ge $((SYNC_MIN * 60)) ] || [ "$done_runs" != "$last_done" ] || [ $finished = 1 ]; then
    paths=(results/figures/appendix results/verification)
    for d in results/metrics/appendix_*; do
      [ -d "$d" ] || continue
      if grep -q '"status": "COMPLETED"' "$d/status.json" 2>/dev/null; then paths+=("$d")
      else for f in eval.csv status.json manifest.json; do [ -f "$d/$f" ] && paths+=("$d/$f"); done; fi
    done
    git add -- "${paths[@]}" 2>>"$LOG"
    if ! git diff --cached --quiet -- "${paths[@]}"; then
      git commit -q -m "live: appendix figures and runs ($done_runs/36 completed)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "${paths[@]}" >>"$LOG" 2>&1
      git push -q node2 HEAD:refs/heads/node1-sync >>"$LOG" 2>&1; log "commit $(git rev-parse --short HEAD) node2_push=$?"
      bash .git/publish_sanitized.sh --push > logs/publish_last.log 2>&1; log "github_publish=$? $(grep -o 'sanitized_head=[0-9a-f]*' logs/publish_last.log)"
    fi
    last_sync=$now; last_done=$done_runs
  fi
  [ $finished = 1 ] && { log "both queues finished; stopping"; break; }
  sleep "$INTERVAL"
done
