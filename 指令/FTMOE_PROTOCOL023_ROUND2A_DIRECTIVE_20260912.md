# FT-MoE Protocol 023 — Round 2A 实验执行指示

日期：2026-09-12  
分支：`protocol-023`  
当前审查 HEAD：`d7bbc8c75221ed8707bee80ab30fc83adbecc500`

## 1. 本轮目标

本轮仍然**禁止实现 D**。目标是先回答：Fixed-C 在多 regime、有限在线预算下，是否真实存在“能学新知识，但会损失旧知识，旧机制复现时还需要重新学习”的 stability–plasticity conflict。

只有这个冲突在严格 prequential 实验和成熟梯度上同时成立，下一轮才允许实现动态增删专家 D。

当前状态：

- H0 generator correctness：PASS
- H1 Data Gate：FAIL（`valid onset/follow-up` 定义歧义）
- H2 specialization：PASS（2/3）
- H3 gradient interference：INADMISSIBLE（fresh residual bank 的隐藏层梯度退化）
- H4 forgetting：目前只有 probe-level 证据
- D：未实现

## 2. H1 原结果永久保留

Round 1 的严格 full-window 定义下：

- A full-window follow-up = 24
- B = 36
- C = 31
- 阈值 >= 50

因此 `H1-original = FAIL` 必须永久保留，禁止改写成 PASS。

原因是 task median life 只有 5–6 intervals，而注册 response horizon 是 16–18 intervals，full-window 要求结构性过严。

## 3. 新增 H1-v2，并在任何模型运行前冻结

定义 `usable_primary_followup`：

一个 onset event 只有在**第一个注册下游资源 response window**至少有 1 个有效 task-level observation 时才计入。

- A: CPU onset -> RAM window `[t0+4, t0+14)`
- B: RAM onset -> Disk window `[t0+3, t0+13)`
- C: Disk onset -> CPU window `[t0+3, t0+13)`

必须同时报告：

- `n_full_window_followup`
- `n_primary_window_followup`
- `n_any_window_followup`

H1-v2 Gate：

- primary-window follow-up >= 80 / regime
- independent fault events >= 80 / regime
- prevalence 3%–12%
- deployment rejection <= 25%
- migration rejection <= 40%
- worst event share < 10%
- regime prevalence spread <= 4 percentage points

若任一 regime primary-window follow-up < 80，则 `STOP-DATA-v2`。

## 4. 先做 S2.5：Data-only marginal calibration

当前 A/B/C 的边缘统计差异偏大，会产生“专家只是按幅值分工”的替代解释。

A compute-first 完全冻结，因为它必须继续等同 Protocol022 cascade_v2。

只允许调 B/C，且只能看数据统计，禁止看任何模型指标。

允许调：

### B memory-first
- RAM target floor/upper
- RAM ramp multiplier
- Disk response peak
- response duration
- cascade probability（最后手段）

禁止改变：
- `RAM -> Disk -> CPU`
- onset resource = RAM

### C io-first
- Disk retained peak/cap
- CPU response magnitude
- response duration
- cascade probability（最后手段）

禁止改变：
- `Disk -> CPU -> RAM`
- onset resource = Disk

以 A 为参考，建议数据目标：

- event count relative diff <= 20%
- duration relative diff <= 25%
- median peak-ratio relative diff <= 30%
- prevalence absolute diff vs A <= 3 percentage points
- 同时必须满足 H1-v2

所有 calibration 仅可使用 seed700。701/702/703 继续冻结为 confirmation seeds。

## 5. 解决 io_first 样本稀疏

当前 io_first h=1 within-regime test 只有 5 个 positive，AP=0.0136，不能承担最终第三 specialist 的证据。

校准后要求：

- compute_first h=1 positives >= 30
- memory_first h=1 positives >= 30
- io_first h=1 positives >= 30
- 建议目标 >= 50

如果 io_first 仍不足，可以把 C 作为 auxiliary stress regime，但主动态专家 claim 只能依赖 A/B；不能强行写成三 specialist。

## 6. 重新收集 development stream

不要覆盖旧数据，新建：

`dev_seed700_steps2880_calibrated`

manifest 必须记录：

- generator version
- parameter diff
- old H1 result
- H1-v2 definition
- calibration reason
- source parent hash

旧 Round 1 stream 不删除。

## 7. 重跑 H2-v2 specialization

重新运行：

- train A -> test A/B/C
- train B -> test A/B/C
- train C -> test A/B/C

Primary horizon = h=1。

继续使用：

`within AP - mean cross AP >= 0.05`

增加 reliability 条件：

`within-regime positives >= 30`

至少 2/3 regime 同时满足 gap 和 positive count。理想为 3/3。

## 8. 进入 S4 前必须满足

- H0 PASS
- H1-original 保留 FAIL
- H1-v2 PASS
- H2-v2 PASS
- stream integrity PASS
- confirmation seeds untouched

否则不运行 S4。

## 9. S4 只先跑 A + C，不实现 D

A：Frozen baseline  
C：Fixed 4-expert residual

S4 目标不是把 C 调到最高，而是测：

- current-regime adaptation
- previous-regime forgetting
- recurrence relearning

暂时不跑 D、C-wide、C-budget。

## 10. 在线预算必须先冻结

Protocol022 已证明训练更多会持续提高 C，所以必须限制为部署合理的 online budget。

建议默认：

- update every 4 scored intervals
- batch size = 32
- 1 gradient step / opportunity

因此：

- 420-step first exposure ≈ 105 updates
- 360-step recurrence ≈ 90 updates

先做 runtime benchmark，把最终配置写入 `online_budget.json`。正式 S4 开始后禁止改 budget。

## 11. 严格 Prequential 顺序

每个 interval：

1. 用当前模型预测
2. 保存 prediction、model_version、learner_hash
3. environment 前进
4. label mature
5. settlement 历史 prediction
6. replay/anchor selection
7. 若当前 step 有 update opportunity 才训练
8. model_version + 1

禁止当前标签先出现再预测当前样本。

## 12. C 必须保存成熟 checkpoint

至少保存：

- `C_after_A1.pt`
- `C_after_B1.pt`
- `C_after_C1.pt`
- `C_after_F1.pt`
- `C_after_A2.pt`
- `C_after_C2.pt`
- `C_after_B2.pt`

每个 checkpoint 记录：

- base hash
- learner hash
- optimizer hash
- replay hash
- update count
- phase
- time index

## 13. 固定 probe matrix 测真实 forgetting

从校准后的三个 single-regime seed700 streams 中各固定一个 held-out evaluation slice：

- probe_A
- probe_B
- probe_C

这些 slice：

- 不参与 training
- 不进入 replay
- 不参与 threshold tuning

每个 phase boundary 用当前 C checkpoint 评估全部 probe，形成：

| checkpoint | Probe A | Probe B | Probe C |
|---|---:|---:|---:|
| start | | | |
| after A1 | | | |
| after B1 | | | |
| after C1 | | | |
| after F1 | | | |
| after A2 | | | |
| after C2 | | | |
| after B2 | | | |

Primary metric = PR-AUC；secondary = resource macro-F1。

## 14. H4-v2：真实 Fixed-C forgetting Gate

必须先证明 C 学会当前 regime：

至少 2/3 first-exposure regime 满足：

`late-phase PR-AUC(C) >= A + 0.03`

再判定 forgetting：

至少一个 transition 满足：

`previous-regime probe PR-AUC drop >= 0.03`

最好 >=2 transitions。

只有“学新 + 忘旧”同时成立，才叫 stability–plasticity conflict。

如果只忘旧但学不会新，则 `STOP-NO-LEARN`，不能上 D。

## 15. Recurrence relearning Gate

对 A2/C2/B2：

比较：

`previous exposure end score - recurrence first-100 score`

定义 `relearning_gap`。

进入 D 的推荐条件：

`relearning_gap >= 0.03`

至少一个 recurrence 成立，最好两个。

同时要观察 first-100 -> late-recurrence 是否随着在线更新重新恢复。

## 16. H3-v2：必须在成熟 C 上重测梯度冲突

Round 1 H3 是 `INADMISSIBLE`，不是 FAIL。

至少使用：

- C_after_A1
- C_after_B1
- C_after_C1

每个 checkpoint 从 A/B/C 各抽相同数量 mature independent events：

- 建议 64 / regime
- 不足则取 common N
- 但 N >= 30

计算：

- full learner gradient cosine
- router gradient cosine
- per-expert cosine
- per-layer cosine
- negative-pair fraction
- degenerate groups

如果 preregistered parameter groups 仍有大量 undefined gradient，则继续 `INADMISSIBLE`。

## 17. H3-v2 数值门槛不改

继续使用原 Protocol023 Gate：

至少一个 regime pair：

`mean cosine <= -0.05`

或：

`negative pair fraction >= 30%`

不能因为 Round 1 的 cosine 为正就降低门槛。

必须输出 `checkpoint × regime-pair` 矩阵，不能只看最终 checkpoint。

如果成熟 C 仍全部 strongly co-directional，则 `STOP-GI`，当前场景不允许进入 D。

## 18. D 的最终准入条件

下一轮只有以下 8 项全部 PASS 才允许实现 D：

1. H1-v2 PASS
2. H2-v2 PASS
3. C 能学习至少 2 个 regime
4. H3-v2 mature gradient conflict PASS
5. H4-v2 actual forgetting PASS
6. recurrence relearning PASS
7. online budget frozen
8. prequential integrity PASS

必须写：

`D_eligible = true/false`

不能写 “probably” 或 “promising”。

## 19. 本轮禁止事项

禁止：

- 实现 D
- expert birth / retire / reactivate
- 调 D trigger
- 调 D ramp
- 跑 C-wide
- 跑 C-budget
- 使用 seeds 701/702/703
- 修改 offline v4 / P19 / P20 checkpoints

本轮唯一目标：证明动态专家是否真的有科学必要性。

## 20. 建议输出目录

```text
artifacts/ftmoe_online/protocol_023/round2a/
  amendments/
    h1_v2_definition.json
    online_budget.json
  data_calibration/
    candidate_table.json
    selected_generator.json
    marginal_match_v2.json
    data_gate_v2.json
  specialization_v2/
    cross_regime_probe_v2.json
  fixed_c_prequential/
    predictions.npz
    settlements.jsonl
    update_log.jsonl
    phase_metrics.json
    probe_matrix.json
    recurrence_metrics.json
  mature_gradient/
    after_A1.json
    after_B1.json
    after_C1.json
    summary.json
  checkpoints/
    C_after_A1.pt
    C_after_B1.pt
    C_after_C1.pt
    C_after_F1.pt
    C_after_A2.pt
    C_after_C2.pt
    C_after_B2.pt
  gate_status_round2a.json
  problem_log_round2a.jsonl
```

## 21. 下一次 GitHub 反馈重点

下一次优先审查：

### Data
- A/B/C primary-followup
- prevalence
- duration
- peak ratio
- rejection
- h=1 positives

### Specialization
- within AP
- cross AP
- gap
- positive count

### Fixed-C
- current-regime gain vs A
- probe forgetting matrix
- recurrence early/late performance

### Gradient
- mature mean cosine
- negative fraction
- degenerate groups

### Integrity
- prediction-before-update audit
- update counts
- learner/checkpoint hashes

只有这些结果共同证明 Fixed-C 存在真实 stability–plasticity conflict，下一轮才进入 D 的 additive birth/retire/reactivate 实现。
