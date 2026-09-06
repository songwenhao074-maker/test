#!/bin/bash
# Run the full 60-step QoS comparison: gobi + 3 heuristics + final chain v0-v4c1.
# Each GAN-using method starts from a fresh GAN (GAN_FRESH=1) so runs are
# reproducible and no method inherits another's online-tuned state.
cd "$(dirname "$0")"
STEPS=60
SEED=42
for mode in gobi pcft eclb dftm v0 v1b v2c2 v3b2 v4c1; do
  echo "===== $mode ====="
  GAN_FRESH=1 timeout 1800 python run_qos_chain.py "$mode" "$STEPS" "$SEED" 2>&1 \
    | grep -E "QOS_RESULT|QOS_MIGRATIONS|Error|Traceback" | tail -3
done
echo "ALL DONE"
