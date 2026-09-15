# Protocol-024 seed700 恢复续跑：结果、阻塞与交接

日期：2026-09-15  
工作分支：`protocol-024-resume-gpt56-20260915`  
确认实验：**否**（仍为 replay seed700 / model seed1 开发证据；701–703 未使用）

## 1. 本轮做了什么

本轮没有重新生成 response-law 流。最新 generation-only run `34918496058` 已经完成 4981 行 `stream.npz`，但在模拟器循环结束、写入最终 audit/manifest 时因变量名拼写错误退出：

```text
NameError: name 'SCORED_STEPS' is not defined
```

源码注册常量实际为 `SCORDED_STEPS=4980`。异常发生在 `stream.npz` 写盘之后，因此将该失败解释为**工程 finalization 失败，而不是数据生成失败或模型结果**。

为避免再消耗约三小时重复同一 seed700 模拟，新增：

- `maintenance/recover_protocol024_generated_stream.py`：从不可变 `stream.npz` 重新计算可由数组严格推出的 audit/manifest；不修改任何科学数组。
- `.github/workflows/protocol024-resume-valid.yml`：从 run `34918496058` 的 artifact 继续执行恢复 → frozen-z learnability → C budget → lifecycle calibration → lifecycle-on A/C/D pilot。
- `artifacts/ftmoe_online/protocol_024/next_round_v1/recovery_review_20260915.json`：本轮机器可读审计与问题清单。

恢复 workflow run：`34948819437`。

## 2. 已确认的 stream 完整性

失败 artifact 中 `stream.npz` SHA-256：

`468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`

完整性检查：

- 4981 行 = 4980 scored intervals + 1 guard row；16 hosts。
- `host_features`: `(4981,16,7)`；`raw_labels`: `(4981,16)`。
- 所有检查的数值数组均为 finite，无 NaN/Inf。
- 从 `overload_ratio` 按注册物理规则重算 `raw_labels` 与存储标签逐元素完全一致。
- scored host-step 类别计数：Normal 72,759；CPU 5,223；RAM 536；Disk 1,162。
- fault-positive host-step rate = 0.08685994。
- audit event ID 连续为 0–1953，共 1,954 个事件；按机制 R1/R2/R3 分别 650/666/638。

因此这份流可以作为注册 seed700 新场景的开发流继续使用；恢复逻辑不重新采样、不重跑 scheduler，也不修改数组内容。

## 3. Response-law 行为与调度压力

三类 response law 在物理标签上形成清楚差异：

| Law | Positive rate | 主要故障类 | Event-related positive 比例 | Deploy reject | Migrate reject |
|---|---:|---|---:|---:|---:|
| R1 | 0.06162 | CPU | 0.98895 | 0.10790 | 0.30675 |
| R2 | 0.16466 | CPU/RAM | 0.99830 | 0.22994 | 0.48668 |
| R3 | 0.05096 | Disk | 0.98270 | 0.10504 | 0.22517 |

全程 deployment rejection = 1174/8000 = **14.675%**；migration rejection = 581/1766 = **32.899%**。

### 风险

R2 的部署拒绝率 22.99%、迁移拒绝率 48.67% 明显偏高。当前结果虽然显示 >98% 的 law 正例与 response event 同期，且 R1/R2/R3 的资源类别符合预期，但 Protocol-024 指示明确要求类别差异不能主要由大规模拒绝制造。因此正式解释 D−C 之前必须审计：

1. R2 的可学习优势是否仍存在于成功部署/迁移子集；
2. response event → 可观察历史 → future fault 的关系是否在控制 rejection 后仍成立；
3. 是否需要把 rejection 作为协变量/分层审计项，而不是模型输入中的未来信息。

**不要因为后续 A/C/D 指标好看而忽略这一项。**

## 4. 已复现的 learnability 子门禁

使用仓库预注册划分：

- train prediction intervals `[0,3887)`；
- isolation gap `[3887,3900)`，共 13 intervals；
- validation `[3900,4980)`，即六次 recurrence；
- 目标为 same-host `raw_label[t+1] > 0`。

在不可变 stream 上按注册 sklearn LogisticRegression 设定复现得到：

| 方法 | Validation AP |
|---|---:|
| prevalence | 0.0882523 |
| persistence/current fault | 0.6029245 |
| current pressure | 0.8079055 |
| current raw 7-D logistic | 0.7447214 |
| raw history 12×7 logistic | **0.8665578** |

raw-history AP 显著高于 `prevalence + 0.05 = 0.1382523`，因此 `raw_history_learnable=true`。按协议，**不应修改 response_law_v1，也不应扫描 701–703 寻找更有利流**。

尚未在本地复现 frozen 64-D z，因为它需要仓库完整 checkpoint/model 路径；该门禁由 resumed GitHub Actions workflow 执行。只有 `z_ap >= history_ap - 0.05` 才进入 budget/pilot。

## 5. 工程问题

### P0：生成器 finalization 拼写错误

`prepare_ftmoe_protocol024_stream.py` 注册的是 `SCORDED_STEPS`，但 per-law finalization 中出现一次 `SCORED_STEPS`。这导致长生成完成后才失败。

建议后续直接修为同一个注册常量，并加入一个不跑模拟器的 finalization regression test。当前恢复分支优先复用已生成流，避免重复昂贵长跑；原 base experiment branch 仍需补该单行修复。

### P0：valid workflow 指向过期失败 run

现有 `.github/workflows/protocol024-next-round-valid.yml` 固定下载 generation run `34839589954`。该 run 在 RAM guard 阶段失败，artifact 没有 `stream.npz`，因此 valid workflow 无法消费最新已生成数据。

本分支新增独立 resume workflow，固定来源为 `34918496058` / artifact `10381312155`，并校验 stream SHA 后才允许继续。

### P1：失败后的可恢复性不足

原 collector 把 `stream.npz` 与 finalization 放在同一个异常边界里；任何 audit/manifest 失败都会让整个 generation workflow 失败。建议后续把过程分成：

1. simulator → immutable stream；
2. independent finalizer → audit/manifest；
3. verifier → downstream-ready marker。

这样 finalizer 可以幂等重跑且无需重新模拟。

### P1：live-only 元数据无法从 stream 完全恢复

完整 `response_events` payload、`short_trace_skips`、精确 generation elapsed seconds 只存在于 generation 进程内。恢复脚本不会伪造这些字段；只从 audit arrays 恢复 event index/机制/观察区间，并在 manifest 明确标记不可恢复字段。

## 6. 当前续跑状态

恢复 workflow：`34948819437`。

执行顺序：

`immutable stream recovery → unit/integrity gates → official frozen-z learnability → C update_every {4,1} budget → fixed lifecycle threshold → lifecycle-on A/C/D pilot → status/artifact`

截至本报告首次写入，workflow 已进入执行阶段；最终结论必须以该 workflow 的 artifact 为准，不把本地 raw-history probe 当作完整 A/C/D 性能结果。

## 7. 下一位模型应如何判断结果

1. 若 frozen-z 门禁失败：不要跑 A/C/D，也不要改 response law；按注册指示给 C/D **同时**增加同一轻量时间特征，然后重新做 lifecycle-off 等价性。
2. 若 frozen-z 通过但 lifecycle 全程未触发：修 trigger/qualification 的具体因果缺口，保留本次失败结果。
3. 若 lifecycle 被实际 exercise：先报告 birth/candidate/accept/reject 等 ledger，再看 recurrence 六窗口 D−C；开发参考是 mean(D−C)≥0.03、至少 4/6 为正、normal FPR 不恶化超过 0.01。
4. 无论 D−C 是否为正，都单列 R2 rejection sensitivity；当前流不能在不审计该问题的情况下直接升级为确认实验。
5. 701–703 仍保持未使用，直到方法、目标、生成器与超参数全部冻结。

## 8. 状态标签

```text
engineering_passed: partial / recovery path implemented
new_stream_generated: true (source run 34918496058; finalization originally failed)
stream_integrity_verified: true
raw_history_learnable: true
frozen_z_gate: pending resumed workflow
budget_comparison_completed: pending resumed workflow
lifecycle_on_run_completed: pending resumed workflow
lifecycle_exercised: pending resumed workflow
development_signal: pending resumed workflow
confirmation_run: false
```
