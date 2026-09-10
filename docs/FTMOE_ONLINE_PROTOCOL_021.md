# Protocol 021 — Audited Unseen-Regime Online Adaptation + Additive Dynamic Residual Experts

> 状态（2026-09-10）：**P21-S1 已核验、P21-S2 生成器与测试完成、P21-S3 data-only 试点完成（2/3 通过 Data Gate）、P21-S4 U4 门禁 FAIL → STOP-A 成立、P21-S5 可学习性 PASS**。结论：**不进入模型阶段**（§10/§12 未启动），动态专家 D 仍被阻塞。详见 §5。
> 上游方案（不可改写）：[`指令/FTMOE_PROTOCOL021_UNSEEN_REGIME_EXECUTION_PLAN_FINAL.md`](../指令/FTMOE_PROTOCOL021_UNSEEN_REGIME_EXECUTION_PLAN_FINAL.md)
> 执行记录：[执行计划](../指令/FTMOE_PROTOCOL021_EXECUTION_PLAN.md) · [问题日志](../指令/FTMOE_PROTOCOL021_PROBLEM_LOG.md)
> 分支：`protocol-021`，自 `protocol-020` 的 `b56d63f901a79cf9bd1c2d1a8e39e88f23fc63ac` 分出。仓库：`songwenhao074-maker/FT-MoE`。

## 0. 为什么新开 Protocol 021

Protocol 020 的科学问题（"动态专家能否在 CPU/RAM/Disk 相位漂移中优于 A/B/C"）在 S6 训练已经覆盖这些 profile 之后，无法再回答"未见 regime 上的在线适应"。Protocol 021 冻结 020 为历史基线，把问题改为：

> **当运行时出现相对于完整离线初始化链真正未覆盖、但具有因果可观察性与可学习规律的新 regime 时，在线方法能否利用成熟反馈改善后续预测；若固定在线模型仍有持续容量不足，动态专家能否提供独立收益？**

### 0.1 必须接受的三条既成事实

1. `dev500` 不是离线未见故障模式（S6 的 12 个训练 episode 已含 baseline/cpu/ram/disk 主要配置），R1 的"未稳定胜过 A"只能限定在**已覆盖 profile 及其切换**上。
2. 数据层三类 fault 已能构造，因此新一轮的主变量必须换成**离线链未出现过的时间规律与跨资源关系**，而不是再找一个容量倍率。
3. 现有 Dynamic 部署（Top-k ramp 重归一化）在成熟边界**不连续**（`max detection-logit jump ≈ 3.47`），在修复前不得复用。

## 1. 五个假设（H1–H5）

| # | 假设 | 判定门禁 |
|---|---|---|
| H1 | **Unseen validity**：新 regime 相对实际在线起点的完整学习链可证明未覆盖 | U3（机制未使用）+ U4（实际分布结构差异） |
| H2 | **Causal learnability**：当前/过去 12 步内含有足以预测未来 fault 的线索 | 因果探针（§9.4） |
| H3 | **Fixed online adaptation**：固定拓扑在线 C 能在成熟新事件后改善后续未训练预测，且熟悉模式退化受控 | A vs C 的 late-unseen 指标；否则 **STOP-C** |
| H4 | **Dynamic capacity benefit**：只有 C 能学但出现持续错误平台时，D 才有意义 | 同信息/同预算/同保护的 D-C 对照 |
| H5 | **Recurring knowledge reuse**：retire/reactivate 的收益 | 仅在 H4 成立后 |

## 2. Unseen 分级（禁止一个布尔值糊弄）

| 等级 | 定义 | 是否足够支持主 claim |
|---|---|---|
| U0 | 新 seed / 新任务实例 / 新 replay | 否 |
| U1 | VM/source cohort 不重叠 | 否（只证明来源隔离） |
| U2 | 静态参数组合离线没用过 | 不充分 |
| U3 | **生成机制/时间因果规律离线未使用** | 主 claim 必须 |
| U4 | **实际数据统计结构显著超出离线覆盖** | 主 claim 必须与 U3 配合 |

本协议一律**分别报告** U1/U2/U3/U4，不输出单一的 `unseen=true`。

## 3. 注册机制：Temporal Resource Cascade（`cascade_v1`）

```text
CPU burst（age 0 熟悉水平用于准入；age 1..d 起持续 >= 主机容量）
        │  lag 4
        ▼
RAM 目标渐进抬升（从熟悉水平插值到注册目标，窗口末释放）
        │  lag 8
        ▼
Disk 有上限、自清理的 retained-data 累积项
```

| 参数 | 注册值 | 物理依据 |
|---|---|---|
| `cpu_duration` | 3–5 区间 | 持续 burst |
| `cpu_to_ram_lag` / `cpu_to_disk_lag` | 4 / 8 | 二者都必须落在 12 步可观察历史内（"陌生但可学习"） |
| `ram_duration` / `disk_duration` | 6–10 区间 | 同上 |
| `cpu_burst_floor` / `upper` | 4400 / 5200 | 1.092× / 1.291× 主机 CPU 容量 4029.0 |
| `ram_target_floor` / `upper` | 4500 / 6000 | 1.048× 小容量主机 RAM 4295.0；低于 8192 主机 |
| `disk_retained_peak` / `cap` | 16000 / 24000 | 叠加 Markov 占用（≤9000）后需同机共存才越过 28990.8 |
| `cascade_task_probability` | 候选 0.15 / 0.25 / 0.35 | §6 预登记候选 |
| `observable_history` | 12 | — |
| `mechanism_seed` | 21021 | 包络只由 `replay_seed + creation_id + mechanism_seed` 决定 |

**为什么必须登记"下限/目标"而不是只乘一个系数**：实测 P20 熟悉环境中适配器把 CPU 钳在 1860、RAM 钳在 1400（`p50=p90=p99=max`），在线 cohort 的 VM 中位 CPU 需求仅约 160，因此乘法系数会被 clip 吞掉；同时 `Simulator.getPlacementPossible()` 用**创建时刻**的需求做准入检查，age 0 就超容的任务永远无法放置（实测拒绝率 93.7%）。因此机制定义为"准入保持熟悉水平、随后按注册幅度抬升"，并在 docstring 与问题日志 P21-01/P21-08 中完整登记。

**严禁作为模型输入**：`regime_id`、`phase_id`、`cascade_task_flag`、`cascade_event_id`、未来需求/容量、未成熟标签。

## 4. 阶段与门禁

| 阶段 | 内容 | 状态 | 证据 |
|---|---|---|---|
| P21-S1 | 完整学习链 Exposure Ledger + 禁止复用集合 | 完成 | `offline_coverage_audit/exposure_ledger.{json,md}`、`exclusion_registry.json` |
| P21-S2 | Temporal Cascade 生成器 + 确定性/因果/lag 测试 | 完成（11/11 通过） | `simulator/workload/BitbrainWorkloadProtocol021.py`、`test_ftmoe_protocol021_unseen.py` |
| P21-S3 | data-only 试点（p=0.15/0.25/0.35，1200 步 + guard，online cohort，seed 600） | 完成 | `pilot_streams/p0NN_seed600_steps1200/` |
| P21-S4 | U4 定量分布审计（跨资源滞后、事件结构、exact overlap） | 见 §5 | `unseen_data_audit/candidate_p0NN.json`、`offline_reference.json`、`selected.json` |
| P21-S5 | 因果可学习性探针 | 见 §5 | `learnability/probe_result.json` |
| P21-S6 | 3400 步模型开发流（熟悉 500/未见 1200/熟悉 500/再现 1200） | **未开始** | — |
| P21-S7 | Frozen A vs C-residual-off/on | **未开始** | — |
| P21-S8+ | 梯度诊断 → 稳定 C → Additive D-v1 → 资格验证 → C vs D | **未开始** | — |

### 4.1 Pilot Data Gate（§7.1，预登记）

| 判据 | 阈值 |
|---|---|
| 总体异常率 | 1%–15% |
| cascade 相关正 host-step | ≥150 |
| 独立 cascade fault 事件 | ≥30 |
| 正常 host-step | ≥5000 |
| 部署拒绝率 | ≤25% |
| 迁移拒绝率 | ≤40% |
| 单事件占全部正例 | <20% |

### 4.2 U4 Gate（§8.5，预登记）

1. `CPU_t→RAM_{t+4}` 或 `CPU_t→Disk_{t+8}` 的滞后峰超出**所有可审计离线语料**对应统计的历史 95% 区间；
2. cascade 转移概率相对离线最大值差 ≥0.15 或 ≥1.5×；
3. 与 P20 S6 的 12 步评分窗口 exact hash 重叠 = 0；
4. 与 P20 S6 train/dev 的 VM 交集为空；
5. Data Gate 通过。

### 4.3 Learnability Gate（§9.4，预登记）

```text
AP >= max(0.20, 3 x 正例率)
且 (ROC-AUC >= 0.75 或 top-decile recall >= 0.40)
时间划分：0–399 拟合 / 400–799 开发 / 800–1199 前瞻测试，禁止跨时间随机打乱
```

## 5. 本次执行结论（2026-09-10）

### 5.1 分级判定（分别报告，不合并）

| 等级 | 判定 | 证据 |
|---|---|---|
| U0 新 seed/实例 | 是 | replay seed 600、P20 未使用的任务实例 |
| U1 来源隔离 | **PASS**（相对 P20 S6 train/dev） | `online`(94) ∩ train(277) = ∅、∩ dev(86) = ∅；窗口 exact hash 重叠 = 0（P20 S6 与 P20 dev 流均为 0） |
| U1（相对完整 014/019 链） | **UNVERIFIED** | P014 上游 VM 选择未在任何产物中登记（§4 要求如实标注） |
| U2 参数组合未见 | PASS（构造性） | 该包络组合在离线链中不存在（见 U3） |
| U3 生成机制未见 | **PASS** | `exposure_ledger.json`：在 014/019/020 的 1003 个文件中检索 `cascade`/`cpu_to_ram`/`delayed`/`mechanism_seed`/`hysteresis`/`feedback` **全部 0 命中**；旁证：disk law 自述只保留 per-task 一阶持续性、drift 只有离散 capacity 跳变 |
| U4 分布新颖性（**预注册门禁**） | **FAIL** | 见 5.3 |

### 5.2 Pilot Data Gate（§7.1）

| 候选 | 异常率 | cascade 正例 | 独立 cascade 事件 | 部署拒绝 | 迁移拒绝 | 单事件占比 | 门禁 |
|---|---:|---:|---:|---:|---:|---:|---|
| p=0.15 | 4.43% | 846 | 302 | 8.47% | 25.85% | 1.41% | **PASS** |
| p=0.25 | 7.19% | 1377 | 495 | 14.10% | 37.99% | 1.01% | **PASS** |
| p=0.35 | 9.57% | 1832 | 666 | 17.47% | **43.76%** | 0.82% | **FAIL** |

p=0.35 唯一失败项是迁移拒绝率（>40%）：cascade 容器在过载阶段无法迁移，密度越高越明显。按纪律**保留失败、不调参迁就**。

### 5.3 U4 门禁失败（STOP-A 成立）

预注册的两项判据**都未通过**（三个候选一致）：

| 判据 | 候选 p=0.25 | 离线 95% 区间 | 判定 |
|---|---:|---|---|
| `Corr(CPU_t, RAM_{t+4})` 超出离线历史 95% | 0.549 | p97.5 = 0.582，max = 0.638 | 未超出 |
| `Corr(CPU_t, Disk_{t+8})` 超出离线历史 95% | 0.443 | p97.5 = 0.521，max = 0.572 | 未超出 |
| 转移概率 `P(RAM↑@t+4 | CPU↑@t)` 与离线最大值差 ≥0.15 或 ≥1.5× | 0.428 | 离线 max = 0.764 | 未通过（低于离线） |
| 12 步评分窗口 exact hash 重叠 = 0 | 0 | — | 通过 |
| 与 P20 S6 train/dev 的 VM 交集为空 | ∅ | — | 通过 |

**失败原因（有证据的诊断，不是借口）**：门禁的滞后/转移判据作用在**主机聚合**上，而主机聚合的跨资源共动在离线语料里本来就很强（同一批重任务同机共存 → 同时占 CPU 与 RAM，`Corr(CPU_t,RAM_{t+4})` 离线可达 0.638、转移概率可达 0.764）。也就是说，**这件仪器无法把"任务级时间级联"与"同机共存的共动"区分开**。

### 5.4 事后机制级诊断（明确不属于门禁，不得用于推翻 5.3）

以"任务自身 CPU 骤升"为锚点、对候选与离线语料施加**同一判据**：

| 仪器 | 离线（41 条流） | 候选 p=0.25 |
|---|---|---|
| 恰好 +4 的 RAM 差值（对渐进斜坡天然不利） | p50 = 198 | p50 = 168 |
| 与注册窗口一致的 RAM 窗口最大值 [t+4, t+14) | p50 = 580，p97.5 = 1422，**max = 1497** | p50 = 1381，**p90 = 4500**（≈离线最大值 3 倍），79% onset 为正响应 |
| Disk 窗口最大值 [t+8, t+18) | p50 = 4000，p97.5 = 6000，max = 7000 | p50 = 5000，75% onset 为正响应 |

解读：机制在**单任务层面真实存在且幅度远超离线**（离线每任务 RAM 上限 1400，不可能出现 4500 的窗口响应）；失败的更可能是**聚合尺度的测量仪器**，而不是机制本身。但按 §39/§43 纪律，**不得**在看过结果后重定义门禁——若要改用机制级仪器，必须作为新一轮预注册（`cascade_v2` + 预先固定的机制级仪器）重跑全部候选。

### 5.5 可学习性（P21-S5，§9.4）

| 目标口径 | AP（阈值） | ROC-AUC | top-decile recall | 门禁 |
|---|---|---|---|---|
| `target=next`（门禁口径） | 0.678（0.234） | 0.954 | 0.734 | **PASS** |
| `target=same`（与运行器可比） | 0.950（0.234） | 0.996 | 0.984 | **PASS** |

对照有效：打乱拟合集标签后 AP 降至 0.24（≈正例率，说明无泄漏）；人为注入"未来 ratio"特征后 AP 升至 0.96（说明探针能识别泄漏）。即 **H2 成立**：fault 可由当前/过去可部署特征预测。

### 5.6 本轮最终判定

```text
Data Gate      : 2/3 通过（p=0.15、p=0.25）
U4（预注册）   : FAIL  ->  STOP-A 成立
Learnability   : PASS  ->  H2 成立
selected       : 无（没有任何候选同时通过两项门禁）

=> 不进入 §10 的 3400 步模型开发流；不得以 unseen 名义运行模型
=> 登记第二轮：cascade_v2 机制 + 预先固定的机制级新颖性仪器（§5.4），
   重跑全部三个候选；不得把本轮结果用于挑选机制参数
```

已产出的价值：学习链暴露台账与禁止复用集合（可直接复用）、一个通过全部确定性与因果测试的级联生成器、一个可复现的数据层流水线、以及"聚合仪器为何无法证明任务级时间级联"这一有证据的结论。

## 6. 冻结与不可覆盖

- `artifacts/ftmoe_online/protocol_019/**`、`artifacts/ftmoe_online/protocol_020/**`、原始 v4 与 019/020 起点 checkpoint 一律不改（哈希见 `source_sha256_initial.json` 与 `unseen_registry/cascade_v1.json`）。
- 在线起点：`artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt`，sha256 `10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b`（P21-S1 重算核对，与 R1 configuration 及 protected hashes 一致）。
- 禁止用 P21 的 unseen 数据做任何新的离线适配（§12）。

## 7. 证据索引

| 内容 | 路径 |
|---|---|
| 协议登记 | `artifacts/ftmoe_online/protocol_021/protocol.json` |
| 初始源哈希 | `artifacts/ftmoe_online/protocol_021/source_sha256_initial.json` |
| 学习链暴露台账 | `artifacts/ftmoe_online/protocol_021/offline_coverage_audit/exposure_ledger.{json,md}` |
| 禁止复用集合 | `artifacts/ftmoe_online/protocol_021/offline_coverage_audit/exclusion_registry.json` |
| regime 注册表 | `artifacts/ftmoe_online/protocol_021/unseen_registry/cascade_v1.json` |
| 试点流 | `artifacts/ftmoe_online/protocol_021/pilot_streams/p0NN_seed600_steps1200/{stream.npz,manifest.json,events.json,unseen_data_audit.json}` |
| U4 审计 | `artifacts/ftmoe_online/protocol_021/unseen_data_audit/{offline_reference.json,candidate_p0NN.json,selected.json}` |
| 可学习性 | `artifacts/ftmoe_online/protocol_021/learnability/probe_result*.json` |
| 问题日志 | `artifacts/ftmoe_online/protocol_021/problem_log.jsonl`、`指令/FTMOE_PROTOCOL021_PROBLEM_LOG.md` |

## 8. 与 Protocol 020 的关系

- 019/020 的全部资产、结论与失败记录保留；P21 **不重训**任何离线模型。
- 020 的 R0/R1 工程与因果协议（prequential、冻结基础预测、保护规则）在进入模型阶段时继续沿用；但 020 的 dev500/dev501 已被"方法选择 + 开发评估"双重消耗（问题日志 P21-05），P21 不作为干净开发流使用。
- 020 记录的陈旧 metadata 错配（P21-04）在 P21 侧一律以 episode/raw manifest 为准。
