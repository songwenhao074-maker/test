#!/bin/bash
# 3-seed QoS comparison (seeds 1,2,6) with GAN warmup, per-seed processes.
# Each seed process runs all 9 methods with its own GAN files (_s{seed}).
cd "$(dirname "$0")"
STEPS=60
for seed in 1 2 6; do
  (
    for mode in gobi pcft eclb dftm v0 v1b v2c2 v3b2 v4c1; do
      echo "===== $mode s$seed ====="
      GAN_FRESH=1 GAN_WARMUP=40 timeout 1800 python run_qos_chain.py "$mode" "$STEPS" "$seed" 2>&1 \
        | grep -E "QOS_RESULT|QOS_MIGRATIONS" | sed "s/^/$mode s$seed /"
    done
  ) > "logs/qos_3seed_${seed}.log" 2>&1 &
done
wait
echo "ALL 3 SEEDS DONE"
