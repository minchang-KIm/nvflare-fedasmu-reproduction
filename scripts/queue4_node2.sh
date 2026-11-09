#!/usr/bin/env bash
# Datasets added after queue3: wait for it, take the new code, run one at a time.
cd ~/nvflare-fedasmu-reproduction
while [ ! -f logs/queue3_done ]; do sleep 30; done
git update-ref refs/heads/main node1-sync && git reset -q && git checkout HEAD -- src scripts configs report.py tests
for r in lenet5_fmnist textcnn_imdb resnet20_tinyimagenet; do
  for a in fedasmu fedbuff; do
    .venv/bin/python scripts/run_experiment.py --profile extended --only "extended_${r}_${a}_s0" > logs/ext_${r}_${a}.out 2>&1
  done
done
.venv/bin/python report.py --profiles pilot extended ablation > logs/ext_report.out 2>&1
touch logs/queue4_done
