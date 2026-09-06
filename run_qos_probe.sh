#!/bin/bash
# Multi-seed QoS stability probe: 3 seeds x 30 steps for the key methods.
# GAN_FRESH=1 per run for reproducibility (fresh random GAN each run).
cd "$(dirname "$0")"
STEPS=30
for mode in gobi v0 v1b v2c2 v3b2 v4c1; do
  for seed in 1 2 6; do
    GAN_FRESH=1 timeout 900 python run_qos_chain.py "$mode" "$STEPS" "$seed" 2>&1 \
      | grep -E "QOS_RESULT" | sed "s/^/$mode s$seed /"
  done
done
echo "PROBE DONE"
