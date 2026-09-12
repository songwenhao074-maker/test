# Protocol 022 — Mechanism-Level Unseen Validation → Fixed Residual Adaptation → Additive Dynamic Experts

> 状态（2026-09-11）：**H0 PASS、H1(U4-v2) PASS（p=0.15/0.25）、H2 PASS（有保留）、H3(S5 容量门禁) PASS（带两条决定性保留）**；STOP 表未被触发，但 **S7 的 D 资格问题已被本轮结果预先回答：属于方案 §11 情形 C（是训练预算而非容量），故不得据此启动 D**。S6/S7 未开始。
> 问题日志见 [`指令/FTMOE_PROTOCOL022_PROBLEM_LOG.md`](../指令/FTMOE_PROTOCOL022_PROBLEM_LOG.md)（机读副本 `artifacts/ftmoe_online/protocol_022/problem_log.jsonl`，P22-00…P22-22）。
> 上游方案（不可改写）：[`指令/FTMOE_PROTOCOL022_EXECUTION_PLAN_20260910.md`](../指令/FTMOE_PROTOCOL022_EXECUTION_PLAN_20260910.md)
> 分支：`protocol-022`，自 `protocol-021` 的 `f0abb5580d92730eecdc1db4326382c81a5f1e31` 分出（与方案登记的父 SHA 一致，未 force reset）。
> 机读状态：[`gate_status.json`](../artifacts/ftmoe_online/protocol_022/gate_status.json) · [`protocol.json`](../artifacts/ftmoe_online/protocol_022/protocol.json) · [`bootstrap_state.json`](../artifacts/ftmoe_online/protocol_022/bootstrap_state.json)

## 0. 为什么需要 Protocol 022

Protocol 021 以 `STOP-A` 结束（P21-09）：它的 U4 门禁作用在 `[T, H, 7]` **主机聚合**上，而离线语料里同机共存的重任务本身就让 CPU/RAM/Disk 强共动，因此该仪器无法把「任务级时间级联」从「同机共存共动」里分离。Protocol 022 不推翻历史，而是**新登记一个新机制与一个新仪器**：

1. 仪器尺度：主机聚合 → **task（`creation_id`）/ event**；
2. 新增机制 seed（21021 → 22022，确认 seed 22023 本轮未用）；
3. 新增匹配负对照 M0/M1/M2（plan §6.5）。

**物理包络完全不变**（plan §5.1），以保证两轮可比。`cascade_v1` 与 `cascade_v2` 的物理参数逐项相等，由 `test_ftmoe_protocol022_audit.RegisteredPhysicsTests` 断言。

## 1. 第一轮范围

方案 §25 规定第一次提交只允许做到 S0–S4：

| 阶段 | 内容 | 状态 | 证据 |
|---|---|---|---|
| P22-S0 | task-level 审计器 + 强制单元测试 T-AUDIT-01..06 | **完成** | `ftmoe_protocol022_core.py`、`test_ftmoe_protocol022_audit.py`（31 测试全通过） |
| P22-S1 | 登记 `cascade_v2` + data-only 试点（0.15/0.25/0.35） | **完成** | `unseen_registry/cascade_v2.json`、`pilot_streams/p0NN_seed600_steps1200/` |
| P22-S2 | U4-v2 仪器（task/event 级 + M0/M1/M2） | **完成** | `analyze_ftmoe_protocol022_unseen.py` |
| P22-S3 | 离线参考锁定 → U4-v2 门禁 | **完成** | `audit_v2/offline_reference_v2.json`、`candidate_p0NN.json`、`selected.json` |
| P22-S4 | persistence-controlled 可学习性 | **完成** | `learnability_v2/learnability_p025_seed600_steps1200.json` |
| P22-S5 | Fixed C 容量诊断与门禁 | **PASS**（带保留，见 §9） | `fixed_c/s5_capacity_gate.json`、`fixed_capacity_diagnostic.json`、`budget_sweep.json` |
| P22-S6–S7 | 严格 prequential A/C 与残余平台 | **未开始** | — |
| P22-S8–S12 | Additive Dynamic D | **未开始**（S7 已判定情形 C，不得据此启动） | — |

**未做**（方案 §25 禁止）：未跑完整 C、未实现或运行 D、未改 FT-MoE backbone、未重训任何离线 v4/P19/P20 checkpoint。

## 2. 注册机制 `cascade_v2`

```text
CPU burst（age 0 熟悉水平用于准入；age 1..d 起持续 >= 主机容量）
        │  lag 4
        ▼
RAM 目标渐进抬升（从熟悉水平插值到注册目标，窗口末释放）
        │  lag 8
        ▼
Disk 有上限、自清理的 retained-data 累积项
```

物理参数与 `cascade_v1` **逐项相同**（`cpu_duration` 3–5、lag 4/8、`ram_duration`/`disk_duration` 6–10、`cpu_burst_floor/upper` 4400/5200、`ram_target_floor/upper` 4500/6000、`disk_retained_peak/cap` 16000/24000、`observable_history` 12），仅 `mechanism_seed` 与登记内容不同。

**为什么 CPU 是两段式**（不是可调旋钮）：`Simulator.getPlacementPossible()` 用**准入时刻**的需求做容量可行性检查，age 0 就超容的任务永远无法放置（P21-01 实测拒绝率 93.7%）。因此「先熟悉准入、后抬升」是本模拟器里唯一能同时通过准入并制造持续过载的路径。这条限制在方案 §19 要求下如实写成**construct validity 限制**：

> 本 regime 是 **controlled unseen temporal resource-demand regime**，不是「真实工业故障必然遵循该规律」的声明。

新增 simulator controls（方案 §19）：**S0** 相同准入但只 CPU burst、无 RAM/Disk 级联；**S1** 相同边际包络但随机 lag/order。

## 3. U4-v2 仪器（本协议的核心修正）

### 3.1 统计单元

```text
unit            = creation_id + CPU onset event        （持续事件只算一次）
RAM_response    = max RAM_task[t0+4 : t0+14] - baseline_RAM
Disk_response   = max Disk_task[t0+8 : t0+18] - baseline_Disk
baseline_*      = 任务自身 onset 前各行的 median；若无 onset 前行则取注册熟悉水平
                  （逐事件报告来源）
```

`creation_id` 是唯一稳定任务身份；**slot 会被复用**，`task_timeline.npz` 把 slot→creation_id 映射作为数据导出并由审计核验，而非假定。容器在迁移中对象不变，因此沿 `creation_id` 得到连续轨迹。

### 3.2 onset 规则的两次修正（P22-01）

方案 §6.2 原写 `ΔCPU_t = CPU_task(t) − median(CPU_task[t−4:t]) ≥ τ`。该形式对**注册包络结构性失明**：包络在 age 1 起 burst，因此 burst 内部每个区间的 median 窗口**已经全是 burst**，唯一的大幅跳变是 burst **释放**。实测第一个候选：`max delta_cpu = −4400`、注册 onset **0 个**，而任务级 CPU 确实达到 4400–5200、主机级 Data Gate 通过（304 个独立级联故障事件）。

修正为**无需基线的超越式规则**（τ 不变，仍为 2600）：

```text
onset = 任务自身 CPU 首次达到 τ 的观测区间，
        且其前一个观测区间严格低于 τ；持续 burst 只计一次
```

τ=2600 严格高于离线所有 per-task CPU（实测每语料最大值均为 1860.0），因此熟悉任务**不可能**产生事件。旧形式 `delta_cpu()` 保留并一并报告，`RegisteredOnsetDetectorTests` 永久守护该回归。

### 3.3 离线参考的两个尺度（P22-02）

离线语料**每一个 per-task CPU 需求都被钳在 1860.0**，因此 τ=2600 在离线侧的 task-level onset 数**恰为 0**：方案 §7.2 的「candidate p90 > offline p97.5」在注册尺度上**无定义**——报 PASS 是空洞通过，报 FAIL 是怪罪数据。两者都不可接受，故并列报告两个尺度：

| 尺度 | τ | 离线 task-level onsets | 用途 |
|---|---:|---:|---|
| **registered** | 2600 | **0** | 机制级新颖性证据（离线物理上不可能产生） |
| **calibrated** | 500 | 2074 | 同仪器两侧可比，给出可用的 p90/p97.5 参考 |

calibrated 阈值由**仅使用离线数据**的登记规则选定，且在读取任何候选**之前**固定：500 是严格高于离线非负 ΔCPU 合并 p99（P019 481 / P020 S6 419 / P020 dev 327）且严格低于各语料最大值（861 / 1688 / 1688）的最小整值。它被明确标注为**参考仪器**，不是对门禁的第二次尝试。

**只允许同类比较**：per-task *response* 绝不与离线的*绝对*需求量级比较；离线参考自带用同一函数算出的 response 分布（离线 p97.5 = 505.4、max = 2043.5）。

### 3.4 匹配负对照（P22-06）

| 对照 | 破坏什么 | 保留什么 |
|---|---|---|
| **M0** lag-shuffled | 注册 lag 对齐（逐事件均匀随机圆周移位） | 每个事件自己的偏差曲线：边际幅度、时长分布、事件数、任务队列、到达过程 |
| **M1** order-shuffled | 把固定移位多重集 {0..horizon} **随机重分配**给各事件（恰一个保留注册对齐） | 同上 |
| **M2** within-task circular shift | 逐资源独立圆周移位，保持边际与近似自相关 | 同上 |

跨事件**整条曲线互换**的版本仍然报告（`controls.M0_cross_event_repair_*`），但被明确标注为**弱对照**且**不用于门禁**：所有级联任务共用同一注册包络，曲线形状近乎相同，互换接近恒等操作（实测对照 p90 4500.0 = 候选 p90 4500.0，完全无分离）。

## 4. 门禁结果

### 4.1 Gate H1 / U4-v2（P22-S3）

| candidate | Data Gate | U4-v2 | task onsets | 有效 follow-ups | RAM p90 | 注册 lag-4 对齐 | exceedance（95% CI） |
|---|---|---:|---:|---:|---:|---:|---|
| p=0.15 | PASS | **PASS** | 244 | 136 | 4487.4 | 0.794 | 1.00 [1.00, 1.00] |
| **p=0.25（selected）** | PASS | **PASS** | 399 | 207 | 4500.0 | 0.754 | 1.00 [1.00, 1.00] |
| p=0.35 | **FAIL** | FAIL | 574 | 294 | 4487.4 | 0.789 | 1.00 [1.00, 1.00] |

p=0.35 的失败原因**正是预先登记的**：migration rejection 44.91% > 40%（方案 §5.2 明确「0.35 是 high-load stress candidate，若再次超过 40% 保留 FAIL，不降低门槛，不因为它事件多就选它」）。事实上它产生了最多事件（574），按事件数排序会选它——门禁结果主导排序，故未选中。

U4-v2 八项检查（p=0.25）：A 候选 p90 > 离线 p97.5 ✓；A2 候选中位数 > 离线最大值（3982.7 > 2043.5）✓；A3 优于全部 lag 对照（permutation p=0.0005）✓；B exceedance CI 下界 1.00 ≥ 0.20 ✓；C 注册 lag 对齐显著优于 M0/M1/M2（p=0.0005）✓；D follow-ups 207 ≥ 50 ✓；12 步窗口 exact overlap 与 P20 S6/dev 均为 0 ✓；与 P20 S6 train/dev 的 VM 交集为空 ✓。

**离线参考（calibrated 尺度）**：task-level onsets 2074（P019 1043 / P020 S6 590 / P020 dev 441），RAM response n=1157、p50 26.6、p90 184.5、p97.5 505.4、max 2043.5。候选 p90 4500.0 约为离线 p97.5 的 **8.9 倍**。Disk 次级证据：候选 p50 20000 / p90 22000 / max 24000，同样远超离线。

`claim_level` 分别报告：U0 ✓、U1（相对 P20 S6 来源隔离）✓、U2 ✓、U3 ✓、**U4-v2（task 级）✓**；**U1 限制**：相对**完整历史链**（P014 上游 VM 选择未在任何产物中登记）仍为 **UNVERIFIED**。主 claim 只写机制级 U3 + task 级分布 U4-v2。

### 4.2 Gate H2 / learnability-v2（P22-S4）

主任务为 **onset**（仅在该主机当前不处于目标故障时评估），h=1 为主门禁、h=4 为预警补充。

| 指标 | 值 |
|---|---:|
| onset test AP | **0.2392** |
| onset 门槛 `max(0.15, 2.5×prevalence)` | 0.15 |
| onset test prevalence | 0.0221 |
| onset test ROC-AUC | **0.9571** |
| onset test top-decile recall | 0.8917 |
| forward-test 正例数 | **120** |
| 最强简单基线 `max(B2,B3,B4)` | 0.0221 |
| 随机化 null p99（六族最大） | **0.2153** |
| 随机化 null max（六族最大） | 0.2562 |

**五项检查全部通过** → H2 **PASS**。

简单基线（同一批行、同一指标）：B0 随机 0.0221；B1 host prior 0.0414；**B2 persistence 0.0221**；B3 仅当前比值 0.0201；B4 仅过去故障 0.0221；B5 仅斜率 0.0188。**persistence 在 onset 任务上恰好等于 chance**——这正是方案 §8.1 把 onset 独立出来的目的：它把 P21 「当前异常→下一步仍异常」的解释力降到零，而完整探针仍有 0.2392。

任务 S（state，P21 可比）：test AP 0.6208、ROC-AUC 0.9327、prevalence 0.0641；对照 B2 persistence 0.3550、B3 0.4026、B4 0.4190 —— 完整探针显著高于全部简单基线。

资源分解（h=1，方案 §18 第 7 问）：CPU AP 0.2392（120 正例）、RAM AP 0.6003（14 正例）、Disk AP 0.2922（4 正例）；ROC-AUC 分别 0.957 / 0.978 / 0.998。**不是单一资源驱动整体指标**，但 RAM/Disk 的 onset 事件很稀疏，其 AP 只作参考。

### 4.3 必须在同一处报告的保留意见（不得只取好看的一半）

1. **H2 的余量很薄**（P22-08）。注册判据（优于各族 p99）通过，绑定族为 global permutation（p99 = 0.2153）。**更严格的界不通过**：600 次抽样中最好的单次达 **0.2562 > 0.2392**；观测值位于 global 族的第 99 百分位、其余五族的第 100 百分位。两者并列报告，不省略不利的那个。
2. **lag ablation 不支持「级联时间律已被学到」**（P22-11）。注册 lag（RAM 4 / Disk 8）AP 0.2392，所有错误 lag 变体在 ±0.014 内（disk_lag2 0.2356、disk_lag11 0.2425、ram_lag1 0.2389、ram_lag10 0.2379）。去掉 causal_lag / past_fault / current ratio / slope 任一族的变动 ≤ 0.004；唯一明显的是去掉 migration 族（0.2137，−0.025）。**结论**：探针学到的的确是「onest 前存在可检测的 CPU 抬升」这一真实且因果可得的前兆信号（persistence/past-fault 在该目标上恰为 chance），但**不能**说注册的 CPU→+4RAM→+8Disk lag 结构是被利用的结构。
3. **h=4 不达门槛**（P22-12）：AP 0.2025 < 0.2080，ROC-AUC 0.6766。按方案 §8.1 属补充证据，如实报告而非删除。
4. **只有一条 development 流、一个 replay seed**。方案 §17 要求最终 claim 只能基于 confirmation（3 个未用于调参的 replay seed + 新 mechanism seed）。本轮**不是** confirmation。
5. **U1 相对完整历史链仍 UNVERIFIED**（见 §4.1）。
6. **离线参考是 profile 异质的**（P22-10）：不同语料在不同 adapter 下采集（per-task RAM 最大 4336.9 / 9535.0 / 1400），合并后的离线 response 分布已含异质变异，这使候选-离线分离偏保守。
7. **onset 数 ≠ follow-up 数**（P22-13）：RAM 响应窗口自 +4 起、Disk 自 +8 起，许多容器在此之前结束，故有效 follow-up 少于 onset 数（p=0.25：399 vs 207）。门禁 D 用 follow-up 数，两数处处并列。

## 5. 方案 §18 的八个专项问题的回答

机读答案在 `learnability_v2/learnability_p025_seed600_steps1200.json -> review_question_answers`；此处为摘要。

1. **为什么单次 shuffled-label AP 达 0.24？** 因为 P21 的目标强烈 host-specific：一台主机的标签在长时间内近乎常量，全局 shuffle 保留类别边际，而一个主要学会「这台主机有多活跃」的探针仍能把真正故障的主机排在前面。单次抽样因此有很高的地板，不能当证据。本协议报告六族完整 null 分布。
2. **100 次 permutation 后是否仍异常？** 注册判据下是（0.2392 > 各族 p99，最大 p99 = 0.2153）；更严格的「优于全部 600 次单抽」**否**（0.2562）。为回答本问题专门新增 **`host` 族**（打乱「哪台主机接收哪条标签流」，彻底移除 host 活动签名）：它是第二高 null（p99 0.1810、max 0.2141）但**不占主导**，说明 host identity 会抬高所有 null，却不是观测值的唯一来源。
3. **persistence 基线 AP？** onset 任务上 **0.0221 = prevalence = chance**；state 任务上 0.3550。
4. **去掉 `past_fault_*` 后降多少？** onset 0.2392 → 0.2385（**−0.0007**）；state 0.6208 → 0.6025（−0.018）。
5. **只保留 current ratio？** onset **0.0201**（低于 chance）；state 0.4026。
6. **onset-only 任务是否仍可预测？** 是：AP 0.2392、ROC-AUC 0.9571、120 正例（详见表）。
7. **三个资源各 onset 是否都可预测？** CPU 0.2392（120）、RAM 0.6003（14）、Disk 0.2922（4）；不是单一资源驱动，但 RAM/Disk 事件稀疏。
8. **正确 lag 与错误 lag 是否显著不同？** **不显著**（见 §4.3 第 2 条）——这是本轮最重要的保留意见。

## 6. 复现入口

```text
# 数据层（单进程顺序；真实 subprocess 退出码，非 PowerShell 推断）
python run_ftmoe_protocol022_pilot.py --all

# 离线参考（在读取任何候选之前锁定）
python analyze_ftmoe_protocol022_unseen.py --build-reference

# U4-v2 门禁
python analyze_ftmoe_protocol022_unseen.py --skip-reference

# 可学习性（persistence-controlled）
python probe_ftmoe_protocol022_learnability.py --tag p025 --n-permutations 100

# 强制单元测试
python test_ftmoe_protocol022_audit.py
python test_ftmoe_protocol022_learnability.py
```

每次正式运行的 provenance（命令、git SHA、Python/torch 版本、seed、起止时间、**真实 exit code**、耗时、源码与 checkpoint 哈希）落在各 stream 目录的 `run_provenance.json`；三次采集的真实返回码为 **[0, 0, 0]**。

## 7. 下一轮（S5–S7）的入口条件与注意事项

1. 先做 **Fixed C 可训练上界**（方案 §9）：严格时间拆分、只训 residual、冻结 base、不得用 phase/cascade ID、不得在 test 上调参；并记录 §9.3 的梯度健康指标。
2. 因 P22-11，**必须包含一个被迫使用注册 lag 上下文的变体**，否则无法区分「容量不足」与「未学到级联结构」。
3. 因 P22-08，H2 的薄余量意味着在 S5–S7 之前或期间应补 **confirmation 流**（更多 onset 事件 + ≥2 个额外 replay seed）。
4. D 仍然阻塞：H3（strict online C 稳定优于 A）与 H4（C 可学但存在持续 residual plateau）**均未评估**。

## 8. 未改动的历史

- P21 `STOP-A` 仍为 FAIL，未改写。
- P019/P020/P021 的任何产物、checkpoint、manifest、normalization 均未修改、未重训（已重新哈希核验：7 个受保护文件 0 变更）。
- `dev500/dev501` 仅作为**参考分布**参与离线参考（P21-05 已登记其被方法选择消耗），**未**作为候选或确认流。

## 9. 第二轮（S5）：容量门禁 PASS，但两条保留限制了它的含义

### 9.1 数据与设置

开发流 `dev_seed600_steps2380` 采集成功（exit 0，2646 s，guard 2.5 GiB）：**2380 评分区间**（349/840/350/840 的相位几何按 0.7 缩放自方案的 500/1200/500/1200），数组 2381 行（+1 guard，与冻结 `ReplayV3` 契约一致）。按相位 Data Gate 在 **unseen_1 / unseen_recur 通过**（familiar 相位按构造必然不通过——那里机制是关的）；570 个注册 onset **全部**落在级联窗口内，任务级完整性 OK。

S5 用严格时间拆分（训练 = unseen_1 前段 [350,630)，未来 = mid [630,910)、**late test [910,1190)**、recurrence [1540,2380)），5 个变体共享同一次 base 前向，因此是配对比较。

**可复现性**：整个 S5 矩阵在流被规范化为 `steps+1` 行布局**前后各跑一次，全部指标逐位相同**（见 gate 文件的 `reproducibility`），说明布局修正没有改变任何报告数字。

### 9.2 结果

| 变体 | 可训练参数 | 更新次数 | late PR-AUC | recurrence PR-AUC | late onset AP | familiar_2 F1 |
|---|---:|---:|---:|---:|---:|---:|
| A（冻结，无修正） | 0 | 0 | 0.4228 | 0.4274 | **0.0917** | 0.0690 |
| C-current | 9 752 | 400 | 0.5200 (+0.097) | 0.5038 (+0.076) | 0.0907 | 0.0481 |
| C-wide | 37 424 | 400 | 0.5286 (+0.106) | 0.5131 (+0.086) | 0.0908 | 0.0546 |
| **C-budget** | 9 752 | **800** | **0.6061 (+0.183)** | 0.5728 (+0.145) | 0.0895 | 0.0562 |
| C-causal | 10 172 | 400 | 0.4838 (+0.061) | 0.4775 (+0.050) | 0.0908 | 0.0592 |

**门禁 PASS**：四个残差变体都满足容量判据（late PR-AUC ≥ +0.03），且保护判据成立（familiar_2 F1 降幅 0.010–0.021 ≤ 0.03；familiar_1 降幅 0.000）。因此 **STOP-CAP 未触发**。

### 9.3 两条保留（决定性，必须与 PASS 一并报告）

1. **训练预算在最大预算处仍在改善 → 方案 §11 情形 C 成立。** 同参数、同初始化的预算扫描（C-current）：

   ```text
   steps    100     200     400     800     1600
   latePR  0.4256  0.4374  0.5200  0.6568  0.6891     （末次边际 +0.0323，仍在上升）
   recurPR 0.4297  0.4393  0.5038  0.6105  0.6471
   onsetAP 0.0917  0.0915  0.0907  0.0897  0.0898
   ```

   §11 明确规定：若「单纯增加训练步数就能消除该 plateau」，则问题是训练预算而非专家容量，**D 缺少科学依据**。当前预算下 plateau **未成立**（C-budget 是最好变体这一事实本身就是证据）。
2. **增益是排序/校准增益，不是预警增益。** late-unseen onset AP 对 A 是 0.0917，对所有残差变体是 0.0898–0.0908（持平到略降），且随预算上升而下滑。结合 P22-11（错误 lag 与注册 lag 无法区分），一致的解释是：固定残差改善了未见 regime 上的整体检测排序与校准，**没有**改善 onset 前的预警，也**没有**利用注册级联 lag。

### 9.4 结论与下一步

- H3 按注册判据记为 **PASS**；STOP-CAP 未触发。
- 但 **S7 的「是否真的需要动态专家」问题已被本轮预先回答为情形 C**（预算而非容量），因此**不得据此启动 D**；若仍要讨论 D，必须先证明在足够大的预算下 residual error 形成真正的平台，且错误集中在可重复的新资源/时序簇。
- S6（严格 prequential A/C-off/C-on）**未开始**，且其意义因第 1 条保留而下降：S5 已显示离线容量诊断下固定残差有效，S6 要回答的是在线约束下是否仍有效。
- 资源说明：本轮采集在放宽后的 guard 2.5 GiB 下完成；此前五次失败的完整记录与修复见 §9.5 与 P22-17/18/19/20。

### 9.5 为完成本轮而修复的三个真实内存缺陷

| 问题 | 根因 | 修复后每区间增长 |
|---|---|---:|
| P22-17 | `Stats.saveStats` 每步 `np.append`（二次时间 + 无界增长） | 1.34 → 0.89 MB |
| P22-18 | 六个每区间 bookkeeping 列表，本链路无人读取 | 0.89 → 0.72 MB |
| P22-20 | recovery 可见的 `time_series`/`schedule_series` 历史（本链路 recovery 是 no-op） | 0.72 → **0.005 MB** |
| P22-23 | 流的行数与冻结 `ReplayV3` 契约差一行，且 manifest 用名为 `steps` 的字段记录行数 | 布局修正；矩阵重跑逐位复现 |

组件隔离实测（400 区间）：只加 bookkeeping 上界时进程增长 0.472 GiB；再加 series tail 上界后增长 0.002 GiB。`Stats` 默认行为不变，冻结的 P19/P20 流与哈希不受影响（受保护文件重新哈希：7 个 0 变更）。另有 P22-15：冻结的 `run_ftmoe_protocol020.tolerance_labels` 前向填充会写进临时视图而静默丢失（1-D 输入还会直接报错），P22 版本已修正并登记该差异，**未**改动冻结文件。

### 9.6 登记修正（不改写历史）

- **相位长度**（P22-21）：方案 §10.2 的 500/1200/500/1200 按 0.7 缩放为 **350/840/350/840 = 2380** 区间（+1 guard），结构性质全部保留，但**统计功效下降**。stream manifest 的 `phase_amendment` 同时记录方案值与实际值。
- **RAM guard**（P22-19）：采集器 guard 3.0 → 2.0/2.5 GiB（用户 2026-09-11 授权），每个新 stream 的 manifest 记录该修正。

### 9.7 未做

- S6（严格 prequential A / C-off / C-on）、S7 的平台分析、S8–S12 的 D：**均未开始**。
- 未实现或运行任何动态专家。
- 未把 S5 的离线容量诊断当作在线声明：它训练在带标签区块上，不是 §10.1 的 prequential 时序。
