#!/bin/bash
# QoS comparison with GAN warmup: 40 warmup steps + 60 measured steps.
cd "$(dirname "$0")"
STEPS=60
SEED=42
for mode in gobi pcft eclb dftm v0 v1b v2c2 v3b2 v4c1; do
  echo "===== $mode ====="
  GAN_FRESH=1 GAN_WARMUP=40 timeout 1800 python run_qos_chain.py "$mode" "$STEPS" "$SEED" 2>&1 \
    | grep -E "QOS_RESULT|QOS_MIGRATIONS|warmup" | tail -3
done
echo "ALL DONE"
