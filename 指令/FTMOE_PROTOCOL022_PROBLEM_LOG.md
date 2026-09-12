# FT-MoE Protocol 022 — 实验执行问题日志（Problem Log）

> 协议：**Protocol 022 — Mechanism-Level Unseen Validation → Fixed Residual Adaptation → Additive Dynamic Residual Experts**
> 上游方案：`指令/FTMOE_PROTOCOL022_EXECUTION_PLAN_20260910.md`
> 分支：`protocol-022`（自 `protocol-021` 的 `f0abb5580d92730eecdc1db4326382c81a5f1e31` 分出，与方案登记父 SHA 一致）
> 规则：**任何 Gate 失败 → 停止后续阶段并记录**；记录只追加，不改写历史条目。机读副本：`artifacts/ftmoe_online/protocol_022/problem_log.jsonl`。
> 阶段报告：[`docs/FTMOE_ONLINE_PROTOCOL_022.md`](../docs/FTMOE_ONLINE_PROTOCOL_022.md) · 机读状态：[`gate_status.json`](../artifacts/ftmoe_online/protocol_022/gate_status.json)

## 问题汇总表（最新在上）

| # | 阶段 | 严重度 | 状态 | 摘要 |
|---|---|---|---|---|
| P22-11 | S4 | **高** | ⚠️ 未决（重要保留） | 门禁 H2 通过，但 **lag ablation 显示注册级联 lag 不是承载信号的结构**：正确 lag 0.2392，全部错误 lag 在 ±0.014 内。故 H2 只能说「onset 可超越 persistence 预测」，**不能**说「级联时间律已被学到」 |
| P22-08 | S4 | 中 | ⚠️ 未决 | H2 余量薄：注册判据（> 各族 p99 = 0.2153）通过，但**更严格的界不通过**（600 次单抽最大 0.2562 > 0.2392）；单流单 seed，不是 confirmation |
| P22-07 | S4 | **严重** | ✅ 已修 | lagged 特征里的 NaN 毒化标准化器，探针退化为常量预测器，**首次运行给出错误的 STOP-B2**。修复后 onset AP 0.0259 → 0.2392，结论由 STOP-B2 变为 PASS |
| P22-06 | S2 | **高** | ✅ 已修 | 注册的 M0/M1 对照是空操作：所有级联任务共用同一包络，跨事件互换整条曲线完全无分离（对照 p90 4500.0 = 候选 4500.0）。改为逐事件 lag 移位 |
| P22-01 | S2 | **高** | ✅ 已修 | P21 式 median-baseline onset 检测器对注册包络**结构性失明**：0 个 onset，而任务级 CPU 确实达 4400–5200（max delta_cpu = −4400） |
| P22-02 | S2 | 中 | ✅ 已修 | 注册阈值下离线 task-level onset **恰为 0**（离线 per-task CPU 全被钳在 1860），方案 §7.2 的百分位规则**无定义**；改为双尺度并列报告 |
| P22-13 | S4 | 低 | ⚠️ 已登记 | onset 事件数与有效 follow-up 数不同（p=0.25：399 vs 207），因 RAM/Disk 窗口自 +4/+8 起而许多容器更早结束；两数处处并列 |
| P22-12 | S4 | 低 | ⚠️ 已登记 | h=4 onset 不达同一 AP 门槛（0.2025 < 0.2080），按方案 §8.1 作补充证据如实报告 |
| P22-10 | S3 | — | ⚠️ 已登记 | 离线参考是 profile 异质的（per-task RAM 最大 4336.9 / 9535.0 / 1400），使候选-离线分离偏保守 |
| P22-05 | S1 | 低 | ⚠️ 已登记 | `dev500/dev501` 只能作参考分布，不得当干净候选（沿用 P21-05） |
| P22-04 | S2 | 低 | ⚠️ 已登记 | P014 语料无 `creation_id` 列，永远不能进入 task-level 参考（登记为 host-aggregate only） |
| P22-03 | S1 | — | ⚠️ 已登记 | 方案 §8.2 的「18 特征探针」未定义特征族划分；v2 改为 25 列、按消融族划分 |
| P22-09 | S4 | 低 | ✅ 已回答 | §18 八问全部回答（见阶段报告 §5） |
| P22-00 | S0 | — | ✅ 完成 | 协议 bootstrap：父 SHA 一致、无 force reset、第一轮范围 S0–S4 |

## 详细记录

### 2026-09-10 — P22-07 NaN 毒化探针，首次运行给出**错误的** STOP-B2（最高优先级，已修）

第一次运行 `probe_ftmoe_protocol022_learnability.py` 得到 onset test AP **0.0259**（prevalence 0.0221），看起来是一个干净的负面结论 → STOP-B2。**它是假的。**

根因：`capacity` 族含 `past_fault_lag1_*`，该特征在每条流的第一个区间**无定义**，于是 25 列特征矩阵里有 **48 个 NaN**。`evaluate_target` 用含 NaN 的 fit 分割做标准化 → `mu`/`sigma` 变 NaN → 全部权重变 NaN → `predict()` 返回常量。

识别证据（正是方案自己要求的「不能检测出确定存在的信号的探针，不足以证明信号不存在」）：

```text
in-sample AP                 0.0652  （prevalence 0.0667）-> 连训练集都拟合不了
leaky control（加泄漏特征）  0.0826  = 基础模型 0.0826    -> 泄漏特征完全没被利用
planted 真实特征             0.1209  （应为 ~1.0）         -> 确定存在的信号检测不到
```

修复：`finite_rows()` 显式丢弃含非有限值的行并报告 `rows_dropped_non_finite_feature`；`evaluate_target` 拒绝非二值目标（原先 4 类 state 目标被喂进二值 logistic）；`build_groups` 对每个特征族强制 `[T, H]` 形状契约，使 off-by-one 不会静默广播。守护测试：`test_ftmoe_protocol022_learnability.NaNPoisoningTests` 与 `SignalDetectionTests`。

修复后（同一数据、同一特征）：in-sample AP 0.0652 → **0.6811**；planted 0.1209 → **1.0**；leaky 0.0826 → **0.8108**；state test AP 0.0826 → **0.6208**；onset test AP 0.0259 → **0.2392**。结论由 STOP-B2 变为 **H2 PASS**。

**教训（写入协议纪律）**：在仪器尚未在同一链路上证明「能给出正结果」之前，**不得**报告任何负面 Gate 结论。

### 2026-09-10 — P22-01 注册包络对 P21 式检测器结构性不可见（已修）

方案 §6.2 定义 `ΔCPU_t = CPU_task(t) − median(CPU_task[t−4:t]) ≥ τ`。注册包络在 **age 1 起 burst**（两段式是 `getPlacementPossible()` 的硬约束，非旋钮），因此：

- burst 内部每个区间的 median 窗口**已经全是 burst**；
- 唯一的大幅跳变是 burst **释放**。

实测第一个候选 `p015`（修复前）：

```text
task-level CPU 确实达到        5200.0
max delta_cpu                  -4400.0     （负的，即释放）
pooled task-level deltas       p99 304.0, max 1525.9
τ = 2600 下的注册 onset         0 个
同流主机级 Data Gate            通过（304 个独立级联故障事件）
```

这是与 P21-09 **同族**的仪器失配：P21 量的是主机聚合，朴素的 task-level 移植量的是任务自身轨迹的**错误部分**。**不是**数据或生成器问题。

修复：改为**无需基线的超越式规则**（`ftmoe_protocol022_core.onset_positions`）——任务自身 CPU 首次达到 τ 且前一观测区间严格低于 τ，持续 burst 只计一次。τ **不变**（2600）。response baseline 在任务没有 onset 前行时回退到注册熟悉水平，来源逐事件报告。旧形式保留为 `delta_cpu()` 并一并报告。守护测试：`RegisteredOnsetDetectorTests`。

修复后同一候选：244 个注册 onset、**244/244 落在注册级联窗口内**、RAM response p50 3919 / p90 4487 / max 4600、正响应占比 0.557、注册 lag-4 对齐率 0.794。

**未做**：包络未改、候选参数未改。修复发生在任何 U4-v2 门禁结果被记录**之前**。

### 2026-09-10 — P22-02 注册尺度上的百分位规则无定义（已修，规则在看候选前固定）

实测离线每语料 per-task CPU 最大值**均为 1860.0**（P019 1860.0 / P020 S6 1860.0 / P020 dev 1860.0；P014 产物根本没有 `creation_id` 列）。故 τ=2600 下离线 task-level onset 数**恰为 0**，方案 §7.2 的「candidate p90 > offline p97.5」没有 p97.5 可比——报 PASS 是空洞通过，报 FAIL 是怪罪数据。

修复：并列两个尺度，**不重定义门禁**。

| 尺度 | τ | 离线 task-level onsets | 角色 |
|---|---:|---:|---|
| registered | 2600 | **0** | 机制级新颖性证据（离线物理上不可能） |
| calibrated | 500 | 2074 | 参考仪器：同函数两侧可比 |

calibrated 阈值由**仅用离线数据**的登记规则选定，且在读任何候选**之前**固定：500 是严格高于离线非负 ΔCPU 合并 p99（481/419/327）且严格低于各语料最大值（861/1688/1688）的最小整值。同时明确：**per-task response 绝不与离线绝对需求量级比较**——曾有一版检查 A2 犯了这个错（拿 response 中位数比离线**绝对** RAM 峰值 9535），已改为与离线同函数算出的 response 最大值（2043.5）比较。

### 2026-09-10 — P22-06 注册对照 M0/M1 是空操作（已修）

方案 §6.5 要求 M0（marginal-matched lag-shuffled）与 M1（order-shuffled）。首版实现为**跨事件互换整条 RAM/Disk 偏差曲线**。但所有级联任务由**同一注册包络**生成，曲线形状近乎相同，于是：

```text
M0 对照 p90 4500.0   =   候选 p90 4500.0     完全无分离
M1 对照 p90 4500.0   =   候选 p90 4500.0     完全无分离
对照对齐率 0.5399    vs  候选 0.7536
```

无法与候选分离的对照不能支撑新颖性声明，条件 C 会被对着坏仪器评估。

修复：M0/M1/M2 改为对**每个事件自己的**偏差曲线施加逐事件圆周 lag 移位（`ram_shift_per_event`）。边际幅度、时长分布、事件数、任务队列、到达过程**精确保留**，只破坏注册 lag 对齐。M1 另外把固定移位多重集 {0..horizon} 随机重分配给各事件（恰一事件保留注册对齐）。跨事件互换版本仍报告为 `controls.M0_cross_event_repair_*` 并明确标注为**弱对照、不用于门禁**。

修复后（p025）：对照 RAM response p50 **128.6** vs 候选 **3982.7**；对齐率 0.37–0.57 vs 候选 0.7536；M0/M1/M2 的 alignment permutation **p = 0.0005**。

### 2026-09-10 — P22-11 H2 通过，但 lag ablation 不支持级联律归因（重要保留，未决）

方案 §8.4 明确要求：「若正确注册时序上下文与错误 lag 完全无差异，则不能把可学习性归因于 cascade temporal law。」实测：

```text
注册 lag（RAM 4 / Disk 8）      0.2392
disk_lag2                       0.2356      (-0.0035)
disk_lag11                      0.2425      (+0.0034)
ram_lag1                        0.2389      (-0.0002)
ram_lag10                       0.2379      (-0.0013)

去掉 past_fault 族               0.2385      (-0.0007)
去掉 current ratio 族           0.2415      (+0.0024)
去掉 slope 族                   0.2486      (+0.0094)
去掉 migration 族               0.2137      (-0.0255)   <- 唯一明显者
```

**结论**：探针学到的确实是「onset 之前存在可检测的 CPU 抬升」这一真实、因果可得的前兆信号（persistence / current-ratio / past-fault 在该目标上恰为 chance AP 0.0221），但**不能**说注册的 CPU→+4RAM→+8Disk lag 结构是被利用的结构。

**下一阶段约束**：S5 的容量诊断**必须**包含一个被迫使用注册 lag 上下文的变体；在证明出 lag-specific 优势之前，任何 D 收益都不得依靠 cascade-law 归因。

### 2026-09-10 — P22-08 H2 余量薄（未决，必须与通过一并报告）

```text
onset test AP                    0.2392
随机化各族 null p99（最大）      0.2153     -> 注册判据通过
随机化各族 null max（最大）      0.2562     -> 更严格的界不通过
观测值百分位                      global 99 / 其余五族 100
```

为此新增 **`host` 族**（打乱「哪台主机接收哪条标签流」），直接回应方案 §18 第 1 问。它是第二高 null（p99 0.1810 / max 0.2141）但**不占主导**：host identity 抬高所有 null，却不是观测值的唯一来源。同时只有一条 development 流、一个 replay seed——按方案 §17，这**不是** confirmation。

### 2026-09-10 — P22-13 onset 数 ≠ 有效 follow-up 数（已登记）

容器在熟悉水平准入、age 1 起 burst，因此**立即**构成注册 onset；但 RAM 响应窗口自 +4 起、Disk 自 +8 起，许多容器在此之前结束或被销毁。

```text
p=0.15   244 onsets / 136 follow-ups
p=0.25   399 onsets / 207 follow-ups
p=0.35   574 onsets / 294 follow-ups
```

两数处处并列报告，门禁 D 使用 follow-up 数，绝不混用。

### 2026-09-10 — P22-04 / P22-05 / P22-10 离线参考的边界（已登记）

- **P22-04**：`artifacts/ftmoe_end_to_end/data/protocol_004_physical/container_demand_series.npy` 是 `[8, 202, 112]`，**无 `creation_id` 列**，其 112 列是主机轴聚合而非任务。登记为 `kind=host_aggregate_only` 并写明 `exclusion_reason`，只进出参考清单，**绝不**静默替代进 task-level 参考。task-level 参考由 P019 adaptation/development 流、P020 S6 adaptation episodes、P020 development 流构成（已核验同时带 `demands` 与 `creation_ids`）。
- **P22-05**：`dev500/dev501` 被 P20 R1 同时用于方法选择与开发评估（P21-05）。本协议只允许它们进入离线**参考分布**（且参考在任何候选被读取之前建成），**绝不**作为独立确认流。
- **P22-10**：离线参考是 profile 异质的（per-task RAM 最大 4336.9 / 9535.0 / 1400），合并后已含异质变异，使候选-离线分离偏保守。逐语料分解保存在 `offline_reference_v2.json`，无需重建参考即可重算更窄的比较。

### 2026-09-10 — P22-03 探针特征族（已登记）

方案 §8.2 提到「full 18-feature logistic probe」，但 P21 探针没有特征划分，§8.4 要求的「移除 past_fault / current ratios / slope / migration」无法干净回答。v2 把每一列划入**恰好一个**族（`current_ratio` / `slope` / `past_fault` / `migration` / `host_context` / `capacity` / `causal_lag`），共 **25 列**，并补上 P20 方案 §1.4 第 4 项要求的因果容量/时序特征。门禁比较的是完整探针与简单基线、以及它自己的消融，**在同一批行上**。

### 2026-09-10 — P22-00 协议 bootstrap（已完成）

`git rev-parse HEAD` = `f0abb5580d92730eecdc1db4326382c81a5f1e31`，**与方案登记父 SHA 一致**；未 force reset、未 rebase。起点 checkpoint `artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt` 哈希核验为 `10c44bdb…`（与方案 §0 登记一致）。第一轮范围严格限定 S0–S4。

## 未决 / 待用户或后续模型决定

1. **是否进入 S5–S7（Fixed C 上界与严格 prequential）**：三个门禁 H0/H1/H2 均已通过，但 H2 带 P22-08 与 P22-11 两条保留意见。
2. **是否先补 confirmation 流**：因 P22-08 的薄余量与 P22-11 的归因限制，建议在把任何 C 结论当作正式结论之前，先按方案 §17 补 ≥2 个额外 replay seed 的新模式流。
3. **动态专家 D 仍被阻塞**：H3/H4 均未评估；R0-B 的 Top-k ramp 不连续问题也未解决（禁止复用旧部署方式，方案 §12.1）。
4. **`017` 模拟器改动自记**（历史遗留，P21 未决）：`capacity_017/confirmation/seed302_steps2000/failure.json` 含 `simulator_behavior_modified: true`，具体指哪些改动仍未在产物内展开，需用户确认后再决定归档措辞。
5. **R0/R1/P22 代码提交**：`ftmoe_protocol022_core.py`、`prepare_ftmoe_protocol022_unseen.py`、`run_ftmoe_protocol022_pilot.py`、`analyze_ftmoe_protocol022_unseen.py`、`probe_ftmoe_protocol022_learnability.py`、`register_ftmoe_protocol022_unseen.py`、两个测试文件与生成器目前均为 untracked；协议 020 的 R0/R1 代码同样未提交。建议在进入 S5 前提交并登记哈希，否则证据链依赖未跟踪的工作区文件。
