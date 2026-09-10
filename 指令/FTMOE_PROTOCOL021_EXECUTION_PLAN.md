# Protocol 021 执行计划（本轮实际执行范围）

> 日期：2026-09-10
> 方案来源（**不可改写**）：[`FTMOE_PROTOCOL021_UNSEEN_REGIME_EXECUTION_PLAN_FINAL.md`](FTMOE_PROTOCOL021_UNSEEN_REGIME_EXECUTION_PLAN_FINAL.md)
> 协议说明：[`docs/FTMOE_ONLINE_PROTOCOL_021.md`](../docs/FTMOE_ONLINE_PROTOCOL_021.md) · 问题日志：[`FTMOE_PROTOCOL021_PROBLEM_LOG.md`](FTMOE_PROTOCOL021_PROBLEM_LOG.md)
> 分支：`protocol-021` @ 起点 `b56d63f901a79cf9bd1c2d1a8e39e88f23fc63ac`（= protocol-020 的 4 个提交之后）

## 0. 为什么只执行 64 步里的 01–19

方案 §41 明确：如果下一模型只能先做一件事，就做

> **Exposure Ledger + Temporal Cascade data-only pilot + U3/U4/causal-learnability audit**

并明确**暂不做**：继续调 EAGate 阈值、调 expert max、跑 5×3、用 unseen 数据重新离线适配、直接跑 D。

因此本轮执行 64 步 checklist 的 **01–19 步**（数据层与审计层），并在 §39 的 STOP 条件下决定是否进入 §10/§12 的模型阶段。第 20 步之后的模型阶段（3400 步开发流、A vs 固定 C、梯度诊断、Additive D、第二 regime、独立确认）**未执行**。

## 1. 执行清单与状态

| 步 | 内容 | 状态 | 产物 / 证据 |
|---|---|---|---|
| 01–02 | 登记 HEAD/status，新建 `protocol-021`（不覆盖 020） | 完成 | 分支表、本文件 |
| 03 | 建 protocol.json / problem log / source hash | 完成 | `protocol_021/protocol.json`、`problem_log.jsonl`、`source_sha256_initial.json` |
| 04–05 | 枚举在线起点完整学习链，建 exposure ledger | 完成 | `offline_coverage_audit/exposure_ledger.{json,md}`（43 个数据源，全部 verified） |
| 06 | 区分 gradient/selection/normalization/generator-fit/anchor/dev exposure | 完成 | 同上（8 类 exposure type） |
| 07 | 建 exclusion registry | 完成 | `offline_coverage_audit/exclusion_registry.json` |
| 08 | 核查 unseen cohort 与 P20 S6 train/dev 的 VM 重叠 | 完成 | 三组两两交集为空；`online`(94) 与 train(277)/dev(86) 交集为空 |
| 09–10 | 实现 Temporal Cascade 生成器（不跑模型）+ 确定性/lag/无未来泄漏测试 | 完成 | `simulator/workload/BitbrainWorkloadProtocol021.py`；`test_ftmoe_protocol021_unseen.py` 11/11 通过 |
| 11 | 注册 `cascade-v1` | 完成 | `unseen_registry/cascade_v1.json`（由 `register_ftmoe_protocol021_unseen.py` 从生成器派生） |
| 12–13 | data-only 跑 probability=.15/.25/.35，仅按事件/拒绝/物理选候选 | 完成 | `pilot_streams/p0NN_seed600_steps1200/`；`unseen_data_audit/selected.json` |
| 14 | 输出 `unseen_data_audit` | 完成 | 每个流目录一份 + `candidate_p0NN.json` |
| 15–16 | U3 生成器排除审计 + U4 跨资源/事件统计审计 | 完成 | exposure ledger `mechanism_exclusion=not_found`；`analyze_ftmoe_protocol021_unseen.py` |
| 17 | U3/U4 不过则 STOP | 依结果 | 见 `selected.json` 的 `stop_conditions_hit` |
| 18–19 | 因果 probe；不过则 STOP 并修场景/输入 | 完成 | `probe_ftmoe_protocol021_learnability.py`、`learnability/probe_result*.json` |
| 20+ | 3400 步开发流与后续模型阶段 | **未执行** | 门禁决定后另行登记 |

## 2. 预注册内容（执行前固定，事后不得追调）

```text
候选：cascade_task_probability = 0.15 / 0.25 / 0.35           （§6）
时长：每个候选 1200 个评分区间 + 1 个 guard 区间              （§7）
cohort：P20 `online`（94 VM，与 S6 train/dev 交集为空）        （§8.4 授权）
replay seed：600（P20 未使用过的开发种子）
环境：熟悉的 P20 baseline 相位（容量尺度 1.0/1.0/0.9，适配器含 ram_upper=1400）
机制种子：21021；包络只由 replay_seed + creation_id + mechanism_seed 决定
选择规则：只看数据物理指标与事件结构，模型分数从不参与                （§7.1）
```

## 3. 与方案的偏离及理由（全部登记）

| 偏离 | 理由 | 留证 |
|---|---|---|
| CPU burst 改为**两段式**（age 0 熟悉、age 1 起 burst），而非从 age 0 起持续超容 | `Simulator.getPlacementPossible()` 用创建时刻需求做准入检查，age 0 超容 → 93.7% 部署拒绝、零标签（部署墙）。这不是放宽门槛，而是让机制在该模拟器中物理可行 | 问题日志 P21-01；smoke `failure`/`unseen_data_audit` |
| 为 CPU/RAM/Disk 登记"达到/超过主机容量"的下限与目标，而不是只乘系数 | 实测：适配器把 CPU 钳在 1860、RAM 钳在 1400，在线 cohort VM 中位 CPU 需求约 160，乘法系数被 clip 吞掉；级联将不可见 | 问题日志 P21-08；`CASCADE_V1` 物理依据注释 |
| 采用 P20 `online` cohort | 方案 §8.4 明确允许（前提是与 S6 train/dev 不相交，实测交集为空）；S1 审计的"禁用"登记在其原文中自述该组**未被使用**，属保守登记 | 问题日志 P21-06；`cascade_v1.json` |
| probe 同时实现 `target=next`（门禁口径）与 `target=same`（与 FT-MoE 运行器可比） | 两处对"t+1"的定义不同；若混用，"过去故障状态"特征可能等于目标（标签泄漏） | 问题日志 P21-07；`test_ftmoe_protocol021_audit.py` 错帧断言 |

## 4. 待执行的下一步（依门禁结果二选一）

**若 Data Gate + U4 + Learnability 全通过**（方案 §10/§12）：

```text
构建 3400 步开发流：熟悉 500 / 未见首次 1200 / 熟悉返回 500 / 未见再现 1200
起点：protocol_020/s6/adapted_v4_seed1/best.pt（禁止用 P21 数据做离线适配）
方法：frozen A、C-residual-off、C-residual-on（第一轮只跑这三个）
分析：按"时间"与"已成熟独立事件数"双口径（事件分箱 0–4 / 5–14 / 15–29 / >=30）
门禁：≥2/3 开发 replay 的 late-unseen 上 C 优于 A，才允许讨论 D（STOP-C）
```

**若任一不过**（方案 §39）：停止，登记第二轮机制参数（`cascade_v2`）并重跑全部三个候选；不得扩大搜索、不得改阈值迁就结果。

## 5. 纪律声明

1. 本轮**未加载任何 FT-MoE 模型**、未训练、未做离线适配；试点是纯数据采集。
2. 未修改 P19/P20 任何产物；未覆盖原始 v4 与 019/020 起点 checkpoint。
3. 所有失败（smoke 部署墙、TypeError、空表 KeyError）与矛盾（P21-04/P21-05）均保留并登记，未删除、未改写历史。
4. 候选选择不含任何模型分数；模型阶段未启动。
5. **门禁结论**：Data Gate 2/3 通过（p=0.15、p=0.25），**U4 预注册门禁 FAIL → STOP-A 成立**，`selected=null`，不进入 §10 模型阶段。可学习性（§9）PASS，但按纪律不构成继续的授权。
6. 每个试点流目录中的 `generation.log` 仅捕获 stdout（进度打印走终端），且被仓库全局 `*.log` 规则忽略（与 P20 一致），**不属于 §7 要求的证据文件**；证据文件为 `stream.npz`、`manifest.json`、`events.json`、`unseen_data_audit.json`。
