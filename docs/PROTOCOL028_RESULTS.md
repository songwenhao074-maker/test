# Protocol-028 Results

GitHub Actions run: 35705211072.
Data revision: protocol027_data_revision_002; source frozen artifact is Protocol-027 run 35682811782 / artifact 10675401651.
Completed: True.
Completed comparators: ['C_fixed5', 'D_memory_protected'].
Failed comparators: [].

## Registered D/C answer

- Nine-window valid count: 9/9.
- Equal-weight recurrence-first100 AP delta (D-C): -0.0028698398694849197.
- Positive recurrence windows: 3/9.
- Pooled normal-FPR delta (D-C): 8.20546483958351e-05.
- Development signal: False.
- Full-stream AP delta (D-C): -0.008682992553504132.

## Memory-protection mechanism

- Full-capacity birth skips preserving accepted memory: 5.
- Experts that the old oldest-dormant purge rule would have selected, but were preserved: ['9'].
- Actual purges: 0.
- Actual reactivations: 0.
- Final resident memory IDs: ['9', '15', '17', '18'].

## Six late recurrence windows

| 后期回归窗口 | 匹配进入验收 | 接受复用 | 拒绝复用 | 无记忆 | 无匹配 | 满容量保护跳过birth | 专家ID |
|---|---:|---:|---:|---:|---:|---:|---|
| S4_rec1 | 0 | 0 | 0 | 0 | 6 | 1 | - |
| S2_rec2 | 0 | 0 | 0 | 0 | 5 | 1 | - |
| S6_rec1 | 3 | 0 | 3 | 0 | 3 | 1 | 17 |
| S1_rec2 | 1 | 0 | 1 | 0 | 5 | 0 | 17 |
| S5_rec1 | 0 | 0 | 0 | 0 | 5 | 1 | - |
| S3_rec2 | 4 | 0 | 3 | 0 | 2 | 1 | 9, 17 |

## Reuse decisions

- #1 phase=S6_first, expert=9, accepted=False, loss_improvement=0.014079, normal_ok=True, validation_FPR_delta=0.000000, guard_loss_delta=0.026805, guard_FPR_delta=0.000000, reason=old_knowledge_guard_loss_regression.
- #2 phase=S6_first, expert=17, accepted=False, loss_improvement=-0.024474, normal_ok=False, validation_FPR_delta=0.009479, guard_loss_delta=0.218280, guard_FPR_delta=0.000000, reason=relative_loss_improvement_below_1pct,mean_normal_probability_guard_failed_or_unavailable,old_knowledge_guard_loss_regression.
- #3 phase=S6_rec1, expert=17, accepted=False, loss_improvement=-0.058528, normal_ok=True, validation_FPR_delta=0.000000, guard_loss_delta=0.204709, guard_FPR_delta=0.002083, reason=relative_loss_improvement_below_1pct,old_knowledge_guard_loss_regression.
- #4 phase=S6_rec1, expert=17, accepted=False, loss_improvement=-0.101021, normal_ok=True, validation_FPR_delta=0.000000, guard_loss_delta=0.200660, guard_FPR_delta=0.002083, reason=relative_loss_improvement_below_1pct,old_knowledge_guard_loss_regression.
- #5 phase=S6_rec1, expert=17, accepted=False, loss_improvement=-0.002765, normal_ok=True, validation_FPR_delta=0.009346, guard_loss_delta=0.200437, guard_FPR_delta=0.002083, reason=relative_loss_improvement_below_1pct,old_knowledge_guard_loss_regression.
- #6 phase=S1_rec2, expert=17, accepted=False, loss_improvement=0.000041, normal_ok=True, validation_FPR_delta=0.000000, guard_loss_delta=0.201135, guard_FPR_delta=0.002083, reason=relative_loss_improvement_below_1pct,old_knowledge_guard_loss_regression.
- #7 phase=S3_rec2, expert=9, accepted=False, loss_improvement=-0.165179, normal_ok=True, validation_FPR_delta=0.004292, guard_loss_delta=0.346357, guard_FPR_delta=0.004167, reason=relative_loss_improvement_below_1pct,old_knowledge_guard_loss_regression.
- #8 phase=S3_rec2, expert=17, accepted=False, loss_improvement=-0.044255, normal_ok=True, validation_FPR_delta=0.000000, guard_loss_delta=0.183622, guard_FPR_delta=0.000000, reason=relative_loss_improvement_below_1pct,old_knowledge_guard_loss_regression.
- #9 phase=S3_rec2, expert=9, accepted=False, loss_improvement=-0.094220, normal_ok=True, validation_FPR_delta=0.004049, guard_loss_delta=0.321017, guard_FPR_delta=0.001042, reason=relative_loss_improvement_below_1pct,old_knowledge_guard_loss_regression.

## Measured cost

| 指标 | C_fixed5 | D_memory_protected |
|---|---:|---:|
| 模型循环墙钟秒 | 72.685207 | 752.228295 |
| 模型循环CPU秒 | 145.345026 | 940.563729 |
| 峰值RSS字节 | 406822912 | 393592832 |
| 预测p95秒 | 0.004424 | 0.005927 |
| 候选测得额外秒 | NA | 37.446998 |
| 复用测得额外秒 | NA | 0.473211 |

Timing included: step loop, online updates, lifecycle diagnostics/validation/guards executed in-loop, and finish().
Timing excluded: session.save(), parent comparison/finalization, artifact upload and git operations.

## Main limitation

This is one development trajectory (replay seed700/model1), not a statistical confirmation. Protecting a full specialist memory can also block learning a genuinely new service; D retains extra background-compute and resident-memory privileges, so this is not an equal-total-cost comparison.

Increased reuse or absence of purge is not by itself evidence of improved AP or a causal advantage.

## Evidence

- Artifact: https://github.com/songwenhao074-maker/test/actions/runs/35705211072/artifacts/10684617851
- Artifact SHA-256 digest: e3be1c97d0f629c9ed40f61cfd4561b6b04e37b68357a2c92a23ac8f14e2e27e
- Frozen source artifact SHA-256 digest: sha256:111a4c5fc5c508a9e169fdbffe823ee36dc66c037c1775a6aa1abdbaba31c0a1
