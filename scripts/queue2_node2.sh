#!/usr/bin/env bash
cd ~/nvflare-fedasmu-reproduction
while [ ! -f logs/queue_done ]; do sleep 30; done
.venv/bin/python scripts/run_experiment.py --profile pilot --only vgg11_cifar100_fedasmu > logs/q_6.out 2>&1
.venv/bin/python report.py --profiles pilot ablation > logs/q_6_report.out 2>&1
touch logs/queue2_done
