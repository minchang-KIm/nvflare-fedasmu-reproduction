#!/usr/bin/env bash
# Sequential experiment queue on gpu-host (one run at a time: RAM).
cd ~/nvflare-fedasmu-reproduction
while ps -eo args | grep -q "[p]rofile pilot --tag=-memtest"; do sleep 10; done
rm -rf results/metrics/pilot-memtest_* checkpoints/pilot-memtest_* workspaces/runs/pilot-memtest_*
for r in pilot_vgg11_cifar10_fedasmu_s0 pilot_vgg11_cifar100_fedasmu_s0 pilot_vgg11_cifar10_fedbuff_s0 pilot_vgg11_cifar100_fedbuff_s0; do
  rm -rf results/metrics/$r checkpoints/$r workspaces/runs/$r workspaces/sim/$r
done
bash run_all.sh --profile pilot --only "vgg11_cifar10_fedasmu" > logs/q_1.out 2>&1
bash run_all.sh --profile pilot --only "vgg11_cifar10_fedbuff" > logs/q_2.out 2>&1
bash run_all.sh --profile pilot --only "vgg11_cifar100_fedasmu" > logs/q_3.out 2>&1
bash run_all.sh --profile pilot --only "vgg11_cifar100_fedbuff" > logs/q_4.out 2>&1
bash run_all.sh --profile ablation > logs/q_5.out 2>&1
echo QUEUE_DONE > logs/queue_done
