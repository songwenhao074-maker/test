# Protocol-024 continuation：实验结果、修复与剩余问题（GPT-5.6，2026-09-14）

> **后续审阅补充：** [独立审阅与1140步验证](PROTOCOL024_REVIEW_20260914.md) 发现实际指标输出、保存路径和候选拒绝后的恢复仍有缺口。
> 本文保留原执行记录；下一轮以 [最新实验指示](PROTOCOL024_NEXT_DIRECTIVE_20260914.md) 为准。

> 本文件用于后续模型/研究者接手。**不修改 Protocol-023 已注册结论，也不把工程门禁结果解释为 D 优于 C。**

## 1. 本轮读取到的实验指示

Protocol-024 的既定顺序是：先修复 evaluator/integrity，再把动态 residual-bank D 接入与 C 完全相同的 strict-prequential session；随后完成 topology-aware save/resume、causal shadow qualification、新 business-response-law 场景、raw/history 与 frozen 64-D `z` learnability probe、预算/阈值冻结，最后才能跑最小 A/C/D seed-700 pilot。

旧 Protocol-020 `OnlineEAGateV3` 不能直接作为 D 主对照，因为它会同时改变专家生命周期机制和路由/模型容器。D 必须围绕 Protocol-023 C 的同一 residual bank 实现。

## 2. 本轮实际完成的工作

工作分支：`protocol-024-continuation-gpt56`

### 2.1 Protocol024Session：公平 A/C/D 接线

新增 `ftmoe_protocol024_session.py`：

- `Protocol024Session` 直接继承 `run_ftmoe_protocol023_s4.PrequentialS4`，复用同一 prediction-before-label-before-update 顺序、replay、anchor、supervised loss、optimizer family 和 update opportunity。
- D 不重新构造另一套不同初始化的模型，而是在 C session 已创建但尚未评分任何 interval 时，将**同一个 C learner residual bank 原地转换**为 `DynamicResidualBank`。
- 修复了一个实际发现的公平性 bug：Protocol-023 的 `step()` 把在线更新门禁写死为 `self.arm == "C"`。若直接把新臂标成 `D`，D 会预测但不会执行 C 的在线更新。本轮让 D 执行 inherited step 时通过同一 C update gate，同时对外仍保持 D 身份。

### 2.2 evaluator / finalization 修复

新增并验证：

1. **尾部两条预测结算**：Protocol-024 `finish()` 同时成熟 `steps-2` 与 `steps-1`，不再留下旧 Protocol-023 中的 penultimate row 未结算问题。
2. **same-host onset**：保持 `[time, host]` 二维结构，沿 time 轴同 host 构造未来 onset，禁止 flatten 后跨 host shift。
3. **positive-only resource macro-F1**：CPU/RAM/Disk diagnosis 只在 fault-positive rows 上计算，不把 normal row 强制映射进资源类别。

### 2.3 动态 residual bank 的 topology/tensor snapshot round-trip

新增 `export_dynamic_bank()` / `restore_dynamic_bank()`，当前可以保存并恢复：

- active / dormant / shadow IDs；
- active routing order；
- expert tensors；
- router weight / bias；
- ramp；
- next_id；
- max_experts / ramp_updates。

单元测试覆盖了同时存在 active、dormant、partial-ramp active child、live shadow 的情况，并验证恢复前后 live output 与 routing 完全一致。

**注意：这还不是完整 session resume。** optimizer Adam moments、age/activation EMA、trigger state、candidate validation ledger 等仍未纳入完整 checkpoint。

## 3. GitHub Actions 实际执行结果

最终成功 CI：

- Run ID：`34800910318`
- Job ID：`103843227377`
- Head SHA：`1f243c138dedf4fc4a1c30fd1100e5037c8ff384`

### 3.1 Integrity gates

`test_ftmoe_protocol024_gates.py`：**6/6 PASS**

覆盖：

- dynamic topology/tensor checkpoint round-trip；
- ramp=0 activation output continuity；
- same-host onset 不跨 host；
- positive-only resource macro-F1；
- finish 同时结算两条 tail records；
- Protocol024Session 确实继承 Protocol-023 strict-prequential session。

### 3.2 真实 seed-700 development stream 的 integrated C/D smoke

脚本：`run_ftmoe_protocol024_session_smoke.py`

条件：lifecycle 完全关闭，只验证最终 D runner 与 C 的公平起点；24 intervals，6 次在线更新。

结果：

| 项目 | C | D |
|---|---:|---:|
| intervals | 24 | 24 |
| online updates | 6 | 6 |
| detection probability 最大绝对差 | 0.0 | 0.0 |
| diagnosis probability 最大绝对差 | 0.0 | 0.0 |
| frozen base unchanged | PASS | PASS |

D topology：

- active IDs = `0,1,2,3`
- dormant = none
- shadow = none
- ramp 全部 = 1.0
- max experts = 8

**结论仅限于：Protocol024Session 中，D 在 lifecycle 关闭时与 C 在真实 strict-prequential online updates 后保持完全等价。不能据此写成 D 优于 C。**

## 4. 本轮暴露出的两个具体问题

### 4.1 onset 测试期望值写错

第一次 CI 中实现的 same-host onset 逻辑本身正确，但测试把 eligible current-normal rows 数量误写成 3，实际应为 2。修正测试后对应门禁通过。

这说明后续 temporal metric 测试应显式列出 `(time, host)` eligibility，而不是靠 flatten 后人工计数。

### 4.2 D 被历史 `arm == "C"` 条件静默跳过在线更新

这是本轮最重要的工程问题。修复前，6/6 evaluator/unit gates 已通过，但 integrated smoke 在 interval 7 出现 C/D prediction divergence。追踪后发现 D 没有进入 Protocol-023 的在线 update gate。

这种 bug 很危险：如果直接开启 lifecycle 并跑 A/C/D，它会把“D 不更新”和“动态专家机制效果”混在一起，产生无效结论。当前已经修复，并由最终 integrated smoke 的 `updates_C=6, updates_D=6`、probability diff=0 证实。

## 5. 现在仍不能启动正式 A/C/D pilot 的阻塞项

### P0：生命周期策略尚未冻结

`DynamicResidualBank` 现在只是可靠容器；birth / retire / reactivate 的**数值 trigger** 仍不能照搬旧 EAGate V3。dense residual router 的 novelty 分布不同，需要先在 seed-700 stationary/baseline 上校准并冻结：

- router entropy；
- top1-top2 margin；
- matured supervised-loss EMA；
- consecutive novelty windows；
- candidate minimum matured samples。

严禁使用 regime/mode/event/phase ID 或未来 label 做 trigger。

### P0：shadow candidate 训练与因果资格审查尚未完成

需要：

- shadow optimizer 与 live learner optimizer 分离；
- shadow 不参与 live forward；
- qualification 只能使用创建 candidate 以后、已经 physically matured 的未来窗口；
- activation/rejection 需要可审计 validation ledger。

### P0：完整 save/resume 尚未完成

当前只完成 bank topology/tensor round-trip。还必须保存/恢复：

- live optimizer moments；
- shadow optimizer moments；
- ramp / expert age；
- activation EMA；
- novelty/loss trigger state；
- candidate validation ledger；
- replay/session cursor/RNG 状态（若现有 session checkpoint 未完整覆盖）。

随后必须加 resume-equivalence gate：checkpoint 前后不中断运行与 resume 后的**下一次预测、下一次更新**一致。

### P0：新的 response-law 场景生成器尚不存在

仓库目前只有设计要求，没有可执行的新场景 generator。不能继续用 Protocol-023 仅调整 CPU/RAM/Disk 顺序的场景，因为 mature gradient 证据显示这些 regime 大多同向，不能证明 fixed-C expert interference。

下一场景应改变未来 response law，例如：

- short compute pulse + fast release；
- persistent inference / working-set growth；
- write-back backlog；
- paging/recovery；
- negative-trend batch drain；
- periodic checkpoint/upload。

不同历史趋势可以在相似当前负载下产生不同未来风险；mode/event id 只能 audit，不能输入模型。

### P0：需要确认 frozen 64-D `z` 是否保留可分信息

正式 D 前必须同时做：

- allowed raw/history probe；
- fixed C/D 实际接收的 frozen 64-D `z` probe。

如果 raw/history 可学而 `z` 不可学，应给 C 与 D 同时增加完全相同的轻量 temporal feature，不能只给 D。

### P1：预算和主指标仍需冻结

seed-700 上先做 learnability / budget diagnosis，再冻结第一次驻留和 recurrence。现有文档建议初始参考：first exposure 800–1600 intervals，recurrence 120–240，但不能直接视作最终注册值。

主指标建议在运行前冻结为：每个注册 mode switch 后前 `W=100` intervals 的 detection AP 平均值，并报告 paired D-C、worst-mode AP、full-stream AP；若论文主张提前预警，则 corrected same-host raw onset AP / event recall 必须一起作为对应指标。

## 6. 推荐给下一模型的严格执行顺序

1. 在本分支基础上完成 full session save/resume 与 resume-equivalence test。
2. 加 shadow-only optimizer 和 causal qualification ledger；先做 toy/CI gate，不跑性能结论。
3. 实现 response-law generator，并做 future-label leakage audit。
4. seed-700：raw/history probe + frozen `z` probe。
5. seed-700 stationary/baseline：冻结 novelty/loss trigger、reactivation matching 与 residency/update budget。
6. 先跑最小 A/C/D seed-700 pilot。
7. 只有 D-C 出现有意义开发集差异后，再跑 fixed sparse-8、C-budget、D-no-birth、D-no-reactivation、D-no-retirement。
8. 最后冻结所有规则，使用 701/702/703 confirmation；confirmation 期间禁止调参。

## 7. 本轮状态结论

本轮已经把 Protocol-024 从“动态容器原型”推进到“**D 已接入与 C 一致的 strict-prequential runner，且 lifecycle-off 的真实在线更新等价性通过**”。

但实验仍停留在**工程/评估完整性阶段**。没有新的 response-law stream，没有启用动态 birth/retirement/reactivation，也没有产生正式 Protocol-024 A/C/D 性能结果。因此当前最重要的下一步不是写论文中的性能结论，而是完成 P0 因果生命周期和场景门禁。

机器可读状态：`artifacts/ftmoe_online/protocol_024/continuation_20260914/gpt56_status.json`
