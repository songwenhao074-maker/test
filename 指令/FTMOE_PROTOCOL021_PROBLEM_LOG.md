# FT-MoE Protocol 021 — 实验执行问题日志（Problem Log）

> 协议：**Protocol 021 — Audited Unseen-Regime Online Adaptation + Additive Dynamic Residual Experts**
> 上游方案：`指令/FTMOE_PROTOCOL021_UNSEEN_REGIME_EXECUTION_PLAN_FINAL.md`
> 分支：`protocol-021`（自 `protocol-020` 的 `b56d63f901a79cf9bd1c2d1a8e39e88f23fc63ac` 分出）
> 规则：**任何 Gate 失败 → 停止后续阶段并记录**；记录只追加，不改写历史条目。机读副本：`artifacts/ftmoe_online/protocol_021/problem_log.jsonl`。

## 问题汇总表（最新在上）

| # | 阶段 | 严重度 | 状态 | 摘要 |
|---|---|---|---|---|
| P21-09 | S4 | **高** | ⛔ **STOP-A 成立** | **预注册 U4 门禁失败**：候选 `Corr(CPU_t,RAM_{t+4})`=0.549 < 离线 p97.5=0.582（max 0.638）；`Corr(CPU_t,Disk_{t+8})`=0.443 < 离线 p97.5=0.521（max 0.572）；转移概率 0.428 < 离线 max 0.764。诊断：**聚合尺度的仪器无法区分任务级时间级联与同机共存共动**。按 §39 不进入模型阶段 |
| P21-11 | S3/S4 | 低 | ⚠️ 保留失败 | 候选 p=0.35 迁移拒绝率 **43.76%** 超门禁（≤40%）；密度越高过载容器越多、越无法迁移。不调参迁就 |
| P21-10 | S4 | 中 | ✅ 已修 | 选择逻辑 bug：U4 全失败时按事件数把**未过 Data Gate** 的 p=0.35 选为 selected。改为门禁结果主导排序，无候选双过时 `selected=null` |
| P21-01 | S2/S3 | 高 | ✅ 已修 | 平坦 CPU burst（age 0 起 ≥4200 > 主机容量 4029）导致**部署拒绝率 93.7%**（P20 基线 5.7%）。根因：`getPlacementPossible()` 在**创建时刻**检查 CPU/RAM/Disk 可行性，age 0 就超容的任务永远无法放置 → 既不产生过载也不产生标签。改为**两段式**：age 0 保持熟悉变换（可放置），age 1 起进入持续 burst。修复后拒绝率 0.088 |
| P21-08 | S2 | 中 | ✅ 已修 | 纯乘系数在轻负载 VM 上被 clip 吞掉：在线 cohort VM 中位 CPU 需求≈160，而适配器钳在 CPU 1860 / RAM 1400，乘 2 倍无效。改为按物理定义登记下限/目标（burst ≥4400、RAM 目标 4500 上限 6000、disk retained 峰值 16000 且自清理） |
| P21-07 | S5 | 中 | ✅ 按设计解决 | 探针目标帧语义歧义可能造成标签泄漏（计划 §9.2 写 t+1，仓库运行器约定为步内）。两种目标都实现并分别报告，加错帧不变式单元测试 |
| P21-05 | S1 | 高 | ⚠️ 已登记 | `dev500/dev501` 同时被 R1 的**方法选择**（8 run 冻结网格）与**开发评估**消耗；P21 不得再把它当干净开发流 |
| P21-04 | S1 | 中 | ⚠️ 已登记 | P28/P36 记录的陈旧错配**现在依然存在**：P20 adaptation manifest 顶层 `profiles`（1.0/1.0/0.30 等）≠ 实际 `episode_hashes[*].profile`（1.0/1.0/0.90 ru1400 等）。P21 一律以 episode/raw manifest 为准 |
| P21-06 | S1/S3 | 低 | ✅ 已决策 | S1 审计把 `p20_online` cohort 保守登记为禁用集，但其原文说明该组**未被任何 019/020 run 使用**；计划 §8.4 明确允许该组（前提是与 S6 train/dev 不相交，实测交集为空）。决策与证据写入 registry |
| P21-02 | S3 | 低 | ✅ 已修 | 采集器路径构造运算符优先级错误（`bitbrain / "%d.csv" % i`）→ TypeError；`failure.json` 按设计落盘 |
| P21-03 | S3 | 中 | ✅ 已修 | 审计窗口查表 `event_windows` 在生成任务前构建（空表）→ 任何 cascade 事件都会 KeyError；改为按 append-only 下标查 `cascade_events[event_id]` |

## 详细记录

### 2026-09-10 — P21-09 U4 门禁失败（本轮最终判定，STOP-A 成立）

三个候选的预注册 U4 判据**全部未通过**（数字见 `docs/FTMOE_ONLINE_PROTOCOL_021.md` §5.3 与 `unseen_data_audit/candidate_p0NN.json`）：

```text
Corr(CPU_t, RAM_{t+4})   候选 0.549  vs  离线 p97.5 0.582 / max 0.638   -> 未超出
Corr(CPU_t, Disk_{t+8})  候选 0.443  vs  离线 p97.5 0.521 / max 0.572   -> 未超出
P(RAM_elev@t+4|CPU_elev@t) 0.428      vs  离线 max 0.764                -> 未通过
窗口 exact hash 重叠 0 / 与 S6 train+dev VM 交集 空                     -> 通过
```

诊断（有证据，不是借口）：门禁的滞后与转移判据作用在**主机聚合**上；离线语料里同机共存的重任务本身就使 CPU/RAM/Disk 强共动，因此该仪器无法把"任务级时间级联"从"共存共动"里分离出来。

**事后机制级诊断**（明确不属于门禁、不得用于推翻它）：以任务自身 CPU 骤升为锚点、对候选与 41 条离线流施加同一判据，候选 p=0.25 的 RAM 窗口最大值 `[t+4, t+14)` 为 p50 1381 / **p90 4500**，而离线 p50 580 / p97.5 1422 / **max 1497**；79% 的 onset 为正响应。离线单任务 RAM 上限是 1400，因此 4500 的窗口响应在离线不可能出现——**机制在单任务层面真实存在**。

按 §39/§43 纪律：**不据此通过门禁**，不进入 §10 模型阶段；第二轮须以 `cascade_v2` + **预先固定的机制级仪器**重跑全部候选，且不得用本轮结果挑参数。

### 2026-09-10 — P21-01 部署墙（最高优先级，已修）

把 cascade 任务的 CPU burst 从 age 0 起设为 ≥4200（> 主机 CPU 容量 4029）后，40 步 smoke 给出：

```text
deployment_attempts = 395, deployment_rejected = 370, rate = 0.937
positive_hoststeps  = 0, prevalence = 0.0
```

诊断过程：先读 `scheduler/GOBI.py` 确认 GOBI 是**学习型放置优化器、不做容量可行性检查**；再读 `simulator/Simulator.py:89-105` 确认真正的约束在 `getPlacementPossible()`：

```text
ipsreq <= host.getIPSAvailable() and ramsizereq <= host.getRAMAvailable() and disksizereq <= host.getDiskAvailable()
```

且该检查用的是**容器在当前时刻（创建时刻即 age 0）的需求**。因此超容任务在准入处即被拒绝、随后指针置空销毁（`Simulator.simulationStep` 第 181-182 行），既不产生过载也不产生标签——正是 P20 问题日志 P16 记录的"部署墙"在需求侧的对应现象。

修复：CPU 阶段改为**两段式**（准入保持熟悉水平，age 1 起 burst），RAM/Disk 窗口本就在 age 4/8 之后、不受准入检查影响。修复后同一 smoke：`rate=0.088`、`prevalence=0.058`、`worst_event_share=0.162`。

机制层面的含义已写入生成器 docstring 与 `unseen_registry/cascade_v1.json`：**本模拟器中"增量式需求抬升"是唯一能同时通过准入并制造持续过载的路径**（准入用当前需求，已放置容器的后续需求增长不再受约束）。

### 2026-09-10 — P21-08 clip 吞掉级联（已修）

实测（只读）：

- P20 熟悉基线流 `dev_seed500_steps2000`：每容器 CPU 需求 `p50=p90=p99=max=1860`、RAM `=1400`（适配器上限），disk law 值域 0–9000，主机容量 CPU 4029 / RAM 4295 或 8192 / Disk（0.9 尺度）28990.8；
- VM cohort 原始 trace：`online` cohort 的 VM 中位 CPU 需求约 160（`vm_split.json` 的 `bucket_rule` 按 CPU 活跃度分层：train 0-5 / dev 6-7 / online 8-9），单个 VM 峰值可达 3.7e4。

结论：对被钳住的资源，乘法系数无效；级联必须按**物理定义**登记"该阶段至少达到/超过主机容量"的下限。相关数值全部在 `CASCADE_V1` 中登记，改动须新建 `cascade_v2`。

### 2026-09-10 — P21-04 / P21-05（S1 审计遗留，必须遵守）

- **P21-04**：P20 `adaptation_data/v1/manifest.json` 顶层 `profiles` 仍是早期候选值（`1.0/1.0/0.30`、`0.75/1.0/0.30`、`1.0/0.45/0.30`、`1.0/1.0/0.20`），与 `episode_hashes[*].profile` 及 raw episode manifest（`1.0/1.0/0.90(ru1400)`、`0.65/1.0/0.90(ru1400)`、`1.0/0.55/0.90(ru2400)`、`1.0/1.0/0.35`）不一致；`adaptation/adaptation_profiles.json` 同样未更新。属 P28/P36 已登记问题，现仍存在。**P21 禁止引用顶层 profiles。**
- **P21-05**：R1 冻结网格 `execute_registered_grid_primary.py:45-58` 用 `dev_seed500` + `dev_seed501` 选择方法与保护阈值，之后又把同一对流当作开发评估流。**P21 的试点与后续开发流不使用这两条流。**

### 2026-09-10 — P21-06 cohort 决策（已决策并留证）

S1 审计的 `exclusion_registry.json` 把 `p20_train`(277) / `p20_dev`(86) / `p20_online`(94) 三组都登记为禁用集；其中 `p20_online` 条目的原文是"本次审计未在 019/020 的任何 run 中见到该组被使用；**登记为禁用集以保隔离**"。计划 §8.4 明确：若使用 P20 online cohort，只需检查与 P20 S6 train/dev 的 VM 交集为空。实测三组两两交集为空、并集 457 = eligible。

决策：P21 试点采用 `online` cohort，并把判断、证据与 **U1 限制**（相对原始 v4/019 的完整链仍为 `UNVERIFIED`，因为 P014 上游 VM 选择未在任何产物中登记）写入 `unseen_registry/cascade_v1.json` 与 `protocol.json`。这是"按当前用户要求/计划执行"与"审计保守登记"之间的显式取舍，不是忽略审计。

### 2026-09-10 — P21-07 探针语义（按设计解决）

计划 §9.2 写 "t+1 detection"，而仓库既有在线运行器约定是"动作前输入 → 同一步动作后标签"。若混用，作为特征的"过去已成熟故障状态"可能等于目标。实现同时支持：

- `target=next`（门禁口径）：t 帧特征 → t+1 标签，过去故障特征取 t 帧；
- `target=same`（与 FT-MoE 运行器可比）：t 帧动作前特征 → t 帧标签，过去故障特征取 **t-1** 帧。

`test_ftmoe_protocol021_audit.py::test_matured_label_offset_matches_target` 断言该错帧不变式。

## 未决 / 待用户或后续模型决定

1. **是否进入 §10 的 3400 步模型开发流**：取决于 Data Gate + U4 + Learnability 三个门禁（见 `unseen_data_audit/selected.json` 与 `learnability/probe_result*.json`）。
2. **动态专家 D 仍被阻塞**：按 H3/§12 与 STOP-C，固定在线 C 未显示可重复正信号前不得启动 D；R0-B 的 Top-k ramp 不连续问题也未解决（禁止复用旧部署方式）。
3. **P21-04/P21-05 的历史文件**：本协议不修改 P20 任何产物（冻结资产），仅在 P21 侧登记与规避。
