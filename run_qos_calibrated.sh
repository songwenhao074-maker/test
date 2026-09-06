#!/bin/bash
# QoS 阈值校准实验: 每 (variant, seed) 在 warmup 段 [100:140] 校准 τ (触发率=60%),
# 测量段 [140:200] (60 步, 43 个 p98 真异常 host-step, 35 步有异常)。
# 全部方法串行跑 (并发会丢结果)。3 种子。
cd "$(dirname "$0")"
STEPS=60
TAU_MAP="v0_s1:0.752,v0_s2:0.925,v0_s6:0.986,v1b_s1:1.487,v1b_s2:0.506,v1b_s6:0.992,v2c2_s1:1.458,v2c2_s2:1.129,v2c2_s6:0.994,v3b2_s1:0.737,v3b2_s2:0.868,v3b2_s6:1.093,v4c1_s1:0.994,v4c1_s2:1.501,v4c1_s6:1.010"
for seed in 1 2 6; do
  for mode in gobi pcft eclb dftm v0 v1b v2c2 v3b2 v4c1; do
    echo "===== $mode s$seed ====="
    GAN_FRESH=1 GAN_WARMUP=40 DETECT_TAU_MAP="$TAU_MAP" \
      timeout 1800 python run_qos_chain.py "$mode" "$STEPS" "$seed" 2>&1 \
      | grep -E "QOS_RESULT|QOS_MIGRATIONS" | sed "s/^/[$mode s$seed] /"
  done
done
echo "ALL DONE"
