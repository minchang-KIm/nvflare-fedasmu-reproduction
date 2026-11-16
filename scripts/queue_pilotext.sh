#!/usr/bin/env bash
# Continue the four VGG-11 pilot runs (the bar-chart runs) from their 5,000-update checkpoints
# (copied from node2 as pilotext_*) to 9,500 received updates, so that their curves cover the
# paper's 450,000 s time axis with real data.  Same config; only the budget is raised.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "profile tuned" > /dev/null; do sleep 60; done
for r in vgg11_cifar10_fedasmu vgg11_cifar10_fedbuff vgg11_cifar100_fedasmu vgg11_cifar100_fedbuff; do
  echo "pilotext_${r}_s0"
done | xargs -P 4 -I{} sh -c '.venv/bin/python scripts/run_experiment.py --profile pilot --tag ext --only "^{}$" --set budget_updates=9500 > logs/{}.out 2>&1'
touch logs/pilotext_done
