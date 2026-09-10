# FT-MoE Protocol 020：多故障容量驱动漂移与动态专家 v3 解决方案

> **2026-09-09 修订：本文为原始方案历史存档。当前实施以 [修订实验计划 R1](FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md) 为准。**
> 新路线：R0 验证有效性 → R1 冻结基础＋固定在线修正 → R2 损失/采样 → R3 容量/长期记忆 → R4 动态独立收益 → R5 场景族 → R6 独立确认。
> 原始离线 v4 及已有适配起点均不重训；不重新执行本文 S6。旧 S8 具备运行代码但候选验证和生命周期仍需修订，不能直接进入旧 S10。用户已授权开发期调整场景，最终确认数据不得用于选择。以下旧门禁和执行顺序保留追溯，不覆盖新计划。

> 适用仓库：`songwenhao074-maker/FT-MoE`  
> 基础分支：`protocol-019`  
> 本文目的：针对 Protocol 019 暴露出的数据、模拟器、在线更新和动态专家机制问题，给出一套可直接交给其他模型/工程师执行的解决方案。  
> 建议新协议：**Protocol 020 — Capacity-Driven Multi-Fault Drift + Dynamic Expert v3**  
> 核心原则：**先让数据场景本身满足实验条件，再运行模型；禁止根据模型 F1、D-C 或 D-B 结果反向调整模拟器。**

---

# 0. 最终目标

Protocol 020 不再直接回答：

> “D 是否比 C 好？”

而是分三层回答：

1. **模拟器层**：是否能稳定地产生 CPU / RAM / Disk 三类物理异常，以及明确的 fault-mode drift？
2. **在线学习层**：固定专家 C 是否能够在长流中稳定微调，而不过度遗忘？
3. **动态专家层**：当真实 fault mode 发生变化且固定专家出现持续新误差模式时，动态专家 D 是否真正发生 topology change，并带来更短 adaptation lag 或更高 prequential performance？

只有第 1 层通过后才能执行第 2 层；只有第 2 层通过后才能执行第 3 层。

---

# 1. Protocol 019 已确认的问题

## 1.1 Data Gate 实际没有满足

Protocol 019 预登记的长流要求是：

```text
normal >= 5000 host-samples
CPU >= 100
RAM >= 100
Disk >= 100
```

但 S6 tolerance labels 约为：

```text
Normal = 31565
CPU    =   218
RAM    =    38
Disk   =   179
```

raw labels 约为：

```text
Normal = 31749
CPU    =   131
RAM    =    15
Disk   =   105
```

因此 S6 只能算 stationary topology sanity test，不能算完整三资源在线验证。

---

## 1.2 S7 的 “RAM phase” 没有形成 RAM fault

S7 设计：

```text
cpu -> mixed -> ram -> cpu
```

但 raw phase class counts 大致是：

```text
Phase cpu:
CPU 19 / RAM 0 / Disk 63

Phase mixed:
CPU 79 / RAM 0 / Disk 46

Phase ram:
CPU 71 / RAM 0 / Disk 23

Phase cpu recurrence:
CPU 20 / RAM 2 / Disk 50
```

所以所谓 RAM phase 实际上只是 RAM demand distribution 改变，而不是 RAM-dominant fault mode。

结论：

> S7 证明了 covariate drift 存在，但没有建立 CPU -> RAM -> Disk 这种 fault-mode drift。

---

## 1.3 Dynamic Expert v2 的触发条件在当前路由结构下几乎不可达

当前 v2 使用：

```python
eligible = score >= threshold
unmatched = ~eligible.any(dim=-1)
```

新增专家需要：

```text
routing samples >= 128
unmatched EMA >= 0.05
连续 3 个窗口满足
```

但当前 4 个专家的 acceptance region 很宽，而且：

```text
CAP_ACTIVE = 4
initial experts = 4
```

实际经常四个专家都被激活。

结果：

```text
unmatched = 0
expert_count = 4
birth = 0
retire = 0
reactivate = 0
```

因此：

```text
D == C
```

实际上说明的是：

> dynamic controller 从未真正启动。

不能解释为：

> dynamic topology 启动后没有收益。

---

## 1.4 Retirement 同样不可达

当前 retirement 大致要求：

```text
expert age >= 500
activation EMA < 0.005
连续 5 windows
```

但已有专家在几乎每个 routing window 都大量激活，因此 activation EMA 不可能接近 0.005。

所以 recurrence：

```text
CPU -> RAM -> Disk -> CPU
```

无法真正测试：

```text
CPU expert dormant -> CPU recurrence -> reactivate same expert
```

---

## 1.5 图语义 v2 仍只修复了一半

Protocol 019 已经使用 `creation_ids` 避免：

```text
slot replacement -> fake migration
```

这是正确修复。

但当前 migration 仍主要根据：

```text
previous proposed schedule
        ->
current proposed schedule
```

推断。

而 dev303 中 proposed migration 与 actual execution 存在明显 rejection，因此前一时刻 proposed host 不一定等于当前真实 source host。

正确的预测时刻 t 迁移应定义为：

```text
actual before_placement[t, c]
        ->
proposed destination[t, c]
```

这是因果可用信息，并且是真实当前源 Host。

---

## 1.6 S3 checkpoint selection 与 cold-start Gate 不一致

S3 最终验收主要看：

```text
PR-AUC
FPR
```

但 best checkpoint 仍用旧：

```text
Score = (F1 + HR + NDCG) / 3
```

在当前单资源标签下 HR=NDCG，因此诊断被双倍加权。

结果中已经出现：

```text
selected best checkpoint PR-AUC < last checkpoint PR-AUC
```

说明选模指标和最终 Gate 不一致。

---

## 1.7 S4/S6 的 online learning 不能直接称为“有效提升”

S4 中 B/C 相比 Frozen A 的 second-half F1 仍略低，只是下降没有超过允许门槛。

S6：

```text
A F1  ≈ 0.263
C/D F1≈ 0.269
```

提升很小，但 PR-AUC 反而下降。

因此合理表述应是：

> Online update 没有发生明显灾难性遗忘，固定 threshold F1 有轻微改善，但整体 anomaly ranking 没有明确提高。

---

## 1.8 Online sampling 对 rare faults 不友好

当前 S4：

```text
Recent 24
Anchor 8
interval stratification = none
```

S6 raw positive rate 约 0.78%，RAM 只有 15 个 host-step。

因此大量 update 根本看不到 RAM anomaly。

这会直接限制：

- RAM diagnosis；
- RAM specialist expert；
- fault-mode adaptation；
- dynamic expert birth。

---

## 1.9 Offline / dev / online 底层 VM 来源没有真正隔离

BWGD2 当前 `possible_indices` 是静态过滤得到的固定 VM 集。

不同 replay seed 主要改变：

```text
到达顺序
重复抽样
任务生命周期组合
```

并没有获得：

```text
不同 VM cohort
```

也没有获得：

```text
不重叠原始时间段
```

因此 seed401–408 与 seed303/304/305 不是严格的 source-disjoint split。

---

# 2. Protocol 020 总体结构

建议严格按以下阶段执行：

```text
P20-S0  冻结 Protocol 019，登记新协议
   ↓
P20-S1  模拟器 capacity control + 数据记录修复
   ↓
P20-S2  Graph source semantics 修复
   ↓
P20-S3  VM cohort/source split 修复
   ↓
P20-S4  Data-only capacity scan
   ↓
P20-S5  构建真正的 CPU→RAM→Disk→CPU fault-mode drift
   ↓
P20-S6  Same-domain offline adaptation / cold-start compatibility
   ↓
P20-S7  Online replay + loss 修复，先跑 A/B/C
   ↓
P20-S8  Dynamic Expert v3
   ↓
P20-S9  Stationary + controlled drift 小试
   ↓
P20-S10 5 model seeds × 3 source-disjoint streams
```

任一阶段失败：

```text
停止后续阶段
记录原因
不调模型来迎合场景
不调场景来迎合模型
```

---

# 3. P20-S0：冻结 Protocol 019

必须保留：

```text
docs/FTMOE_ONLINE_PROTOCOL_019.md
artifacts/ftmoe_online/protocol_019/
protocol-019 branch
```

禁止覆盖任何：

```text
S2/S3/S4/S5/S6/S7 result JSON
stream.npz
checkpoint
problem log
source hash
```

建议新目录：

```text
artifacts/ftmoe_online/protocol_020/
```

新文档：

```text
docs/FTMOE_ONLINE_PROTOCOL_020.md
docs/FTMOE_PROTOCOL020_PROBLEM_LOG.md
```

协议登记：

```json
{
  "protocol": "020",
  "base": "protocol019",
  "capacity_control_version": 1,
  "graph_semantics_version": 3,
  "source_split_version": 2,
  "online_optimizer_version": 3,
  "dynamic_expert_version": 3
}
```

---

# 4. P20-S1：把 Host Capacity 变成正式可控参数

## 4.1 不要直接写死 RAM=1000

当前硬件大致是：

```text
RPi4B   RAM = 4295
RPi4B8G RAM = 8192
```

如果全部写成：

```text
RAM = 1000
```

会：

1. 破坏 4GB/8GB 异构比例；
2. 容易导致大量 admission rejection；
3. 难以解释成真实系统配置；
4. 旧协议难以复现。

正确做法：

```text
RAM_CAP_SCALE
```

例如：

```text
RAM_CAP_SCALE=0.5
```

则：

```text
RPi4B   ≈ 2147.5
RPi4B8G = 4096
```

保留异构结构。

---

## 4.2 修改 `simulator/environment/RPiEdge.py`

当前已经有：

```text
CPU_CAP_SCALE
DISK_CAP_SCALE
```

增加：

```python
ram_cap_scale = float(
    os.environ.get("RAM_CAP_SCALE", "1.0")
)
```

RAM 构造：

```python
Ram = RAM(
    self.types[typeID]["RAMSize"] * ram_cap_scale,
    self.types[typeID]["RAMRead"] * 5,
    self.types[typeID]["RAMWrite"] * 5
)
```

旧协议不设置环境变量时：

```text
RAM_CAP_SCALE=1.0
```

必须 byte-level / metric-level 验证 legacy 行为不变。

---

## 4.3 不建议修改 RAMRead/RAMWrite 来制造当前 RAM fault

当前 physical label 主要使用：

```text
RAM space
```

不是 RAM I/O bandwidth。

所以：

```text
RAMRead
RAMWrite
```

即使变小，也不会自动生成当前 RAM-space fault。

除非后续明确重新定义 fault taxonomy，否则不要混在本协议。

Disk Read/Write 同理。

---

# 5. 动态容量 phase 的关键原则

如果 Protocol 020 使用：

```text
CPU phase
RAM phase
Disk phase
```

并让有效 capacity 随 phase 变化，那么**模型必须知道当前 capacity**。

否则模型看到：

```text
相同 workload
相同 schedule
```

但标签突然因为隐藏 capacity 改变而变化。

这属于 hidden-variable drift，会削弱模型比较的因果解释。

---

## 5.1 推荐解释：Effective Resource Quota

论文和实验中不要写：

> “把树莓派 8GB 物理内存改成 1800MB。”

建议写：

> “We vary the effective resource quota available to the target workload to emulate background-service occupancy, multi-tenant contention, and resource reservation.”

即：

\[
EffectiveCapacity =
PhysicalCapacity
-
BackgroundReservedCapacity
\]

这样 CPU/RAM/Disk phase 都有现实意义。

---

## 5.2 当前 capacity 必须进入模型输入

当前 `ScheduleGraphEncoder` 已有 3 维 host capacity，但是通常作为固定 buffer。

Protocol 020 应把它改为 per-sample capacity：

```python
graph_context = {
    "creation_ids": ...,
    "before_placement": ...,
    "capacities": ...
}
```

其中：

```text
capacities shape = [B, W, H, 3]
```

分别为：

```text
CPU capacity
RAM capacity
Disk capacity
```

Graph encoder 不再只能读取静态：

```python
self.host_capacity
```

如果 `graph_context["capacities"]` 存在：

```python
capacity = graph_context["capacities"]
```

否则 legacy path：

```python
capacity = self.host_capacity
```

必须保留 014/019 legacy compatibility。

---

# 6. P20-S2：Graph migration semantics v3

## 6.1 正确迁移定义

对每个窗口时间点 t、container c：

```text
creation_ids[t,c] >= 0
before_placement[t,c] = src
proposed_schedule[t,c] = dst distribution / one-hot
```

真实候选迁移：

```python
src = before_placement[t, c]
dst = argmax(proposed_schedule[t, c])

if valid(c) and src >= 0 and dst >= 0 and src != dst:
    migration[src, dst] += 1
```

不要再用：

```text
previous proposed -> current proposed
```

推断当前 source。

---

## 6.2 为什么这是因果的

预测发生在 `simulationStep()` 前，此时已知：

```text
container creation id
current actual host
new proposed schedule
current demand
current effective capacity
```

这些都是可用信息。

不能使用：

```text
after_placement[t]
actual migration result[t]
raw_label[t]
```

它们属于未来 outcome。

---

## 6.3 Replay 需要增加

`Replay.window()` 返回：

```text
time_window
schedule_window
graph_time_window
creation_id_window
before_placement_window
capacity_window
```

模型调用：

```python
model(
    time_window,
    schedule_window,
    graph_time_window,
    graph_context={
        "creation_ids": ids,
        "before_placement": before,
        "capacities": caps
    }
)
```

---

# 7. P20-S3：严格 source-disjoint VM split

## 7.1 不再用 replay seed 当作数据来源隔离

必须新增一个固定 cohort 文件：

```text
artifacts/ftmoe_online/protocol_020/vm_split.json
```

---

## 7.2 第一选择：扩大 Bitbrain VM pool

当前旧 BWGD2 只用一个 CPU 条件筛成约几十个 VM。

Protocol 020 建议重新扫描全部可用 VM 文件。

每个 VM 只用原始轨迹统计：

```text
CPU mean/p95
RAM mean/p95
Disk read/write（只作描述）
trace length
missing ratio
zero ratio
```

禁止使用：

```text
模型结果
fault label
D-C
```

筛选有效 VM。

---

## 7.3 分组方法

推荐使用 deterministic hash split：

```text
60% train
20% dev
20% online/test
```

例如：

```python
bucket = sha256(vm_id).int % 10

0-5 -> train
6-7 -> dev
8-9 -> online
```

如果资源分布差异太大，可以先按：

```text
CPU tercile × RAM tercile
```

分层，再在层内 hash split。

---

## 7.4 禁止

```text
同一 VM CSV
train 使用前 100 点
online 每创建任务又从同一 CSV 第 0 点开始
```

这种形式不能称为 source-disjoint。

如果继续使用同一个 VM，需要按真实时间区间切片：

```text
train segment
dev segment
online future segment
```

并且 workload model 必须从对应时间 offset 开始，而不是每次任务从 index0 开始。

---

# 8. P20-S4：先只做 simulator/data-only capacity scan

**这一阶段绝对不加载模型。**

目的：

> 找到能稳定形成三类异常，同时不会导致极端 rejection 的 capacity range。

---

## 8.1 CPU candidate grid

建议：

```text
CPU_CAP_SCALE:
1.00
0.90
0.80
0.75
0.70
```

当前单 container CPU 经过 adapter 上限约 1860。

原始 Host CPU 约 4029。

例如：

```text
CPU scale 0.75
capacity ≈ 3022
```

两个高 CPU container：

```text
1860 + 1860 = 3720
```

足以产生 CPU overload。

---

## 8.2 RAM candidate grid

不要一开始直接 1000。

建议：

```text
RAM_CAP_SCALE:
1.00
0.75
0.60
0.50
0.45
0.40
0.35
0.30
```

对应前 8 台：

```text
4295
3221
2577
2148
1933
1718
1503
1289
```

后 8 台仍保留约 1.9 倍容量。

优先关注：

```text
0.50
0.45
0.40
```

因为旧静态分析已经显示 0.5 很可能足以把 RAM fault 提升到百级。

---

## 8.3 Disk candidate grid

当前：

```text
DISK_CAP_SCALE ≈ 0.25
capacity ≈ 8053
```

synthetic disk state 可达到：

```text
9000
```

推荐：

```text
DISK_CAP_SCALE:
0.30
0.25
0.22
0.20
0.1875
0.17
```

重点：

```text
0.20 ~ 0.22
```

避免直接降到 0.125 导致 Disk fault 完全主导。

---

## 8.4 不做完整笛卡尔积

不要一次跑：

```text
5 CPU × 8 RAM × 6 Disk = 240 scenarios
```

推荐单因素扫描：

### Scan A：CPU

```text
RAM=1.0
Disk=0.30
只变 CPU
```

### Scan B：RAM

```text
CPU=1.0
Disk=0.30
只变 RAM
```

### Scan C：Disk

```text
CPU=1.0
RAM=1.0
只变 Disk
```

每个 candidate：

```text
300~500 intervals
```

只统计数据。

---

# 9. 每个 capacity candidate 必须输出的统计

至少保存：

```json
{
  "raw_class_counts": {},
  "tolerance_class_counts": {},
  "pure_fault_counts": {},
  "multi_resource_overload_counts": {},
  "anomaly_prevalence": 0,
  "event_counts": {},
  "mean_event_duration": {},
  "p95_event_duration": {},
  "deployment_rejection_rate": 0,
  "migration_rejection_rate": 0,
  "active_containers_mean": 0,
  "host_occupancy_p95": 0
}
```

---

# 10. 必须新增 `overload_mask` 和 `overload_ratio`

当前只保存最终：

```text
label = argmax(overload ratio)
```

这会丢失联合故障信息。

Protocol 020 collector 必须保存：

```python
overload_ratio[t,h,r] = demand[t,h,r] / capacity[t,h,r]
overload_mask[t,h,r]  = overload_ratio[t,h,r] > 1
```

shape：

```text
[T, H, 3]
```

例如：

```text
[1,1,0]
```

表示：

```text
CPU overload
RAM overload
Disk normal
```

最终 dominant class 仍可用：

```python
argmax(overload_ratio)
```

但分析时必须同时报告：

```text
pure CPU
pure RAM
pure Disk
CPU+RAM
CPU+Disk
RAM+Disk
triple
```

---

# 11. Capacity candidate 的数据筛选标准

第一阶段目标不是最大异常率。

建议：

```text
normal = 80%~95%
```

每类 dominant fault：

```text
CPU  >= 1%
RAM  >= 1%
Disk >= 1%
```

对于 2000 × 16 = 32000 host-step：

推荐最低：

```text
CPU  >= 150 host-step
RAM  >= 150 host-step
Disk >= 150 host-step
```

此外每种 fault 至少：

```text
30 independent events
```

避免一个单一长期事件贡献全部样本。

---

## 11.1 Rejection Gate

必须统计：

```text
initial deployment rejection
migration rejection
proposed != actual
```

建议：

```text
deployment rejection < 20%
migration rejection < 40%
```

如果 capacity 降低导致：

```text
大量任务直接无法部署
```

则该 candidate 不适合做 fault-generation scenario。

---

# 12. P20-S5：真正构造 fault-mode drift

完成单因素 scan 后，根据 data-only 结果选择 phase configuration。

禁止依据任何模型分数。

---

## 12.1 推荐五阶段

```text
Phase 0  Baseline
Phase 1  CPU
Phase 2  RAM
Phase 3  Disk
Phase 4  CPU recurrence
```

每 phase：

```text
400~500 intervals
```

总长度：

```text
2000~2500 intervals
```

---

## 12.2 推荐初始配置

这只是第一组预登记候选，最终必须由 S4 data-only scan 确认。

### Baseline

```text
CPU_CAP_SCALE  = 1.00
RAM_CAP_SCALE  = 1.00
DISK_CAP_SCALE = 0.30
```

### CPU phase

```text
CPU_CAP_SCALE  = 0.75
RAM_CAP_SCALE  = 1.00
DISK_CAP_SCALE = 0.30
```

### RAM phase

```text
CPU_CAP_SCALE  = 1.00
RAM_CAP_SCALE  = 0.45
DISK_CAP_SCALE = 0.30
```

### Disk phase

```text
CPU_CAP_SCALE  = 1.00
RAM_CAP_SCALE  = 1.00
DISK_CAP_SCALE = 0.20
```

### CPU recurrence

同 Phase 1：

```text
CPU=0.75
RAM=1.00
Disk=0.30
```

---

## 12.3 为什么非目标资源要放宽

当前 dominant class：

\[
label = argmax_r(Demand_r/Capacity_r)
\]

所以 RAM phase 如果仍：

```text
CPU scale = 0.75
RAM scale = 0.45
```

CPU 很可能继续先占主导。

因此 RAM phase：

```text
CPU 回到 1.0
```

Disk phase：

```text
CPU=1.0
RAM=1.0
```

这样才能制造真正的 resource-dominant drift。

---

# 13. Drift scene 必须满足的 Gate

每个 phase 单独验收。

例如 500 × 16 = 8000 host-step。

目标 phase：

```text
target fault dominant count >= 100
```

例如 RAM phase：

```text
RAM dominant >= 100
```

并且：

```text
RAM dominant > CPU dominant
RAM dominant > Disk dominant
```

最好：

```text
RAM share among anomalous samples >= 50%
```

同理 CPU/Disk phase。

---

## 13.1 不能再使用 phase 名称自证

只有数据满足：

```text
RAM fault 真正主导
```

才能把 phase 命名为：

```text
RAM phase
```

否则必须标成：

```text
high-RAM-demand phase
```

不能混淆 covariate drift 和 fault-mode drift。

---

# 14. 容量变化和 scheduler 的关系

当前 `getPlacementPossible()` 会检查：

```text
CPU available
RAM size available
Disk size available
```

所以容量降低可能导致：

```text
更多 reject
```

而不是更多 overload。

因此必须区分：

## 方案 A：真实 Host capacity change

scheduler / simulator 都看到新 capacity。

优点：

```text
物理一致
```

缺点：

```text
可能主要增加 rejection
```

推荐先使用这个方案。

---

## 方案 B：Effective quota / background occupancy

如果 A 无法形成足够异常，可以新增：

```text
background_reserved_capacity[t,h,r]
```

定义：

\[
EffectiveCapacity =
PhysicalCapacity -
BackgroundReservedCapacity
\]

但必须决定：

### B1

scheduler 也知道 effective capacity：

```text
capacity-aware scheduling
```

### B2

scheduler 不知道：

```text
unexpected background contention
```

两者是不同研究问题。

**不能混为同一实验。**

推荐优先：

```text
B1
```

因为预测模型和 scheduler 都可以看到当前 available capacity，解释最干净。

---

# 15. P20-S6：重新做 same-domain adaptation

如果 capacity profile 和 VM cohort 都变化，旧 adapted checkpoint 不应直接作为最终 online 起点。

建议：

```text
Protocol019 adapted v4
        ↓
Protocol020 same-domain common adaptation
```

---

## 15.1 训练数据

使用：

```text
train VM cohort
```

并包含多个固定 capacity condition：

```text
baseline
cpu capacity
ram capacity
disk capacity
```

但不能使用正式 online phase 顺序。

训练可以把不同场景作为独立 episode 混合。

---

## 15.2 Dev 数据

使用：

```text
dev VM cohort
```

与 train VM 不重叠。

---

## 15.3 Online/Test

使用：

```text
online VM cohort
```

不得参与：

```text
normalization
capacity choice
checkpoint selection
trigger tuning
```

---

# 16. S6 checkpoint selection 必须和最终目标对齐

不再使用旧：

```text
(F1 + HR + NDCG)/3
```

建议 primary selection：

```text
PR-AUC
```

如果还需要同时考虑资源分类：

```text
Selection =
0.6 * detection_PR_AUC
+
0.4 * resource_macro_F1
```

必须在训练前固定。

不要在实验后根据结果改权重。

---

# 17. Primary label 与 tolerance label

建议：

## Primary

```text
raw post-action physical overload
```

## Secondary

```text
±1 interval tolerance
```

在线训练仍可使用成熟后的 tolerance label，但必须额外报告 raw metrics。

论文主要物理结论建议优先 raw。

---

# 18. P20-S7：修复 online rare-event sampling

## 18.1 Recent Memory 继续保留 interval-level

不能按单 Host 切样本，因为：

```text
graph
schedule
cross-host attention
```

需要整个 interval。

---

## 18.2 Recent buffer 建立事件标签

成熟 label 后，将 interval 标记为：

```text
normal_only
contains_cpu
contains_ram
contains_disk
multi_resource
```

一个 interval 可以进入多个 event index。

---

## 18.3 每次 update 推荐采样

保留总量 24 recent：

```text
12 uniform recent
4 CPU-event
4 RAM-event
4 Disk-event
```

如果某类不足：

```text
从其他 event / uniform 补齐
```

再加：

```text
8 anchor
```

总 batch 仍约：

```text
32 interval windows
```

---

## 18.4 必须避免重复 exposure 爆炸

继续保留：

```text
max exposure <= 3
```

但是 rare event 可以独立设置：

```text
max rare exposure <= 5
```

如果采用不同 exposure 上限，必须预登记并统一用于 B/C/D。

---

# 19. P20-S7：Online loss v3

当前固定：

```text
normal weight 0.6
fault weight 2.0
```

对 1% 左右 anomaly 太弱。

推荐使用**由 same-domain training set 固定计算的 class-balanced weight**，不从 online test stream估计。

---

## 19.1 Detection

设 adaptation-train：

```text
N0 = normal
N1 = anomaly
```

推荐：

\[
w_1 = clip(\sqrt{N_0/N_1}, 2, 10)
\]

\[
w_0 = 1
\]

然后：

```python
CrossEntropy(weight=[1, w1])
```

或者使用 focal loss：

```text
gamma = 2
```

二者只选一个主方案，不要同时叠加太多超参。

推荐第一版：

```text
class-balanced CE
```

更容易解释。

---

## 19.2 Resource classification

CPU/RAM/Disk：

\[
w_c = clip(\sqrt{N_{max}/N_c},1,5)
\]

在 positive samples 上计算 weighted CE。

---

## 19.3 Anchor / distillation

可以继续：

```text
lambda_anchor = 0.25
lambda_distill = 0.10
```

但不要直接认定最优。

先固定为 Protocol 019 值，避免一次改变太多因素。

如果 C 稳定但适应过慢，再单独注册：

```text
regularization ablation
```

---

# 20. P20-S7：A/B/C 先跑，D 暂时不允许启动

必须先验证：

```text
A frozen
B full tuning
C fixed MoE tuning
```

---

## 20.1 Stationary Gate

在 stationary same-domain stream：

```text
anchor F1 drop <= 0.03
```

并要求 C：

```text
PR-AUC 不明显低于 A
```

建议：

```text
C PR-AUC >= A PR-AUC - 0.02
```

---

## 20.2 Drift Gate

在 fault-mode drift stream：

至少 C 应出现：

```text
可测 adaptation
```

例如：

```text
phase transition 后滚动 F1/PR-AUC 有恢复趋势
```

如果 C 完全没有学习能力，先修 online optimizer，不要启动 D。

---

# 21. P20-S8：Dynamic Expert v3 核心设计

Dynamic Expert v3 的最重要变化：

> **专家新增触发不再依赖 `unmatched == no eligible expert`。**

novelty detector 与 trainable router eligibility 解耦。

---

# 22. Dynamic Expert v3 Trigger

推荐使用两种信号：

## 22.1 Routing uncertainty

对每个 routing state：

```text
top1 score
top2 score
margin = top1 - top2
routing entropy
```

novel routing 可以定义为：

```text
margin < threshold
```

或：

```text
entropy > threshold
```

threshold 必须由：

```text
same-domain dev stationary data
```

预先估计，例如：

```text
stationary routing entropy p95
```

不能根据 D-C 结果调整。

---

## 22.2 Matured prediction error

标签成熟后记录：

```text
detection loss
classification loss
false negative
false positive
resource confusion
```

维护：

```text
loss_ema
```

相对 stationary baseline：

```text
loss_ratio = current_loss_ema / baseline_loss_ema
```

---

# 23. 推荐 Birth Gate v3

同时满足：

```text
candidate samples >= 128
routing novelty EMA >= stationary_p95 threshold
loss EMA >= 1.25 * stationary baseline
连续 >= 3 adaptation windows
expert_count < 8
```

才进入 candidate birth。

这里的：

```text
1.25
```

只是建议初值。

正式协议必须在 S8 前登记。

---

# 24. 为什么需要 routing + error 双触发

只使用 routing novelty：

```text
输入变了
但模型可能仍正确
```

没必要加专家。

只使用 loss：

```text
可能只是 label noise
```

也不一定需要新专家。

双条件：

```text
representation routing pattern 变新
+
模型持续犯错
```

更接近：

> 新知识容量不足。

---

# 25. Candidate Buffer

当 trigger 开始积累时保存：

```text
candidate routing states
对应完整 interval windows
成熟 labels
```

上限：

```text
256~512 samples
```

不能只存 hidden vector。

因为后面 shadow expert 训练需要完整 forward context。

---

# 26. Safe Expert Birth v3

## 26.1 Parent selection

candidate centroid：

```python
centroid = mean(candidate_routing_states)
```

选择最近 existing expert：

```python
parent = argmax cosine(
    centroid,
    expert_key
)
```

---

## 26.2 Clone parent

新 expert：

```text
expert MLP = deepcopy(parent)
detection head = deepcopy(parent)
class head = deepcopy(parent)
```

不要 zero-init 整个 expert。

---

## 26.3 新 key

```text
new_key = normalized centroid
```

threshold 可使用 candidate similarity：

```text
q70
```

但 threshold 不再负责 birth trigger。

它只负责新 expert 激活区域。

---

# 27. Shadow Expert 必须正式实现

Protocol 019 deferred shadow activation。

Protocol 020 建议正式实现。

新 expert：

```text
birth candidate
    ↓
shadow mode
```

shadow 模式：

```text
不影响主模型 prediction
不进入 routing normalization
```

训练：

```text
3~5 online updates
```

数据：

```text
candidate buffer
+
少量 anchor
```

---

## 27.1 Activation Gate

只有：

```text
candidate-buffer loss 至少比 parent 低 5%
```

同时：

```text
anchor loss 增幅 <= 2%
```

才正式 activate。

否则：

```text
discard candidate
```

---

# 28. Routing Ramp

正式 activate 后：

```text
0
0.25
0.50
0.75
1.0
```

前 4 个 update 逐步参与。

Protocol 019 的 birth continuity 修复继续保留：

```text
ramp=0 newborn 不占 CAP_ACTIVE slot
```

---

# 29. Retirement v3

不要再使用：

```text
activation EMA < 0.005
```

作为唯一条件。

推荐综合：

```text
expert age >= 500 intervals
routing load < 1%
连续 5 topology windows
AND
与其他 active expert 高度冗余
```

冗余：

```text
expert key cosine > 0.98
```

或者：

```text
两 expert outputs correlation > 0.98
```

---

## 29.1 Retire 仍然是 dormant，不删除

保留：

```text
expert params
key
threshold
optimizer metadata
expert id
```

从 active routing 移出。

---

# 30. Reactivation v3

candidate centroid 与 dormant expert：

```text
cosine > 0.90
```

且 dormant expert 在 candidate buffer 的 shadow loss 优于 nearest active expert：

```text
reactivate same ID
```

而不是创建新 ID。

这样：

```text
CPU -> RAM -> Disk -> CPU
```

才能真正测试 recurring knowledge。

---

# 31. CAP_ACTIVE 暂时不要改变

Protocol 020 第一版建议继续：

```text
CAP_ACTIVE = 4
```

原因：

- 014/019 checkpoint 是在当前结构上训练；
- 直接改成 top2 会改变已有 forward；
- 会把“dynamic trigger变化”和“routing sparsity变化”混为一个实验。

如果 v3 birth 能发生：

```text
E=5/6/...
CAP_ACTIVE=4
```

此时 top-k 才开始真正起作用。

若以后需要：

```text
CAP_ACTIVE=2
```

应该单独注册为 Protocol 021 或 matched-routing retraining。

---

# 32. Dynamic v3 必须记录的诊断日志

每个 topology evaluation：

```json
{
  "step": 0,
  "expert_count": 4,
  "routing_entropy_mean": 0,
  "routing_entropy_p95": 0,
  "top12_margin_mean": 0,
  "novelty_ratio": 0,
  "loss_ema": 0,
  "loss_ratio": 0,
  "candidate_samples": 0,
  "birth_trigger": false,
  "shadow_expert": null,
  "added": [],
  "dormant": [],
  "reactivated": []
}
```

---

# 33. P20-S9：实验顺序

## Experiment A — Stationary

目标：

```text
D 不应乱增长
```

要求：

```text
expert additions <= 1
D anchor forgetting <= 0.03
D-C 无显著负向差异
```

---

## Experiment B — CPU -> RAM

目标：

```text
RAM phase 真实出现 RAM dominant faults
```

观察：

```text
C adaptation lag
D trigger timing
D birth timing
D adaptation lag
```

---

## Experiment C — CPU -> RAM -> Disk

目标：

```text
至少两个新 fault mode
```

检查是否需要：

```text
1 个还是多个 expert
```

---

## Experiment D — CPU -> RAM -> Disk -> CPU

目标：

```text
recurrence
```

检查：

```text
old CPU expert 是否 dormant/reactivated
```

---

# 34. Adaptation Lag 定义

在 phase transition \(t_0\) 后：

先定义新 phase 稳态性能：

```text
phase 最后 20% rolling metric mean
```

达到：

```text
95% of stable level
```

所需 interval：

\[
Lag =
t_{recover} - t_0
\]

Primary comparison：

```text
Lag_D < Lag_C
```

---

# 35. Primary dynamic comparison

动态专家机制的主要 comparison 必须是：

```text
D - C
```

因为：

```text
C = same MoE online training, fixed topology
D = C + dynamic topology
```

B 继续作为：

```text
generic full-model online tuning
```

---

# 36. 最终评价指标

## Detection

```text
Raw F1
Tolerance F1
PR-AUC
Precision
Recall
FPR
Brier
ECE
```

---

## Diagnosis

```text
CPU recall / F1
RAM recall / F1
Disk recall / F1
Macro-F1
confusion matrix
support
```

---

## Online

```text
Prequential PR-AUC
Prequential F1
Adaptation lag
Area under rolling PR-AUC
Area under rolling F1
Anchor forgetting
```

---

## Dynamic expert

```text
expert additions
shadow candidates
shadow rejected
retirements
reactivations
peak experts
mean active experts
routing entropy
prediction latency
update latency
memory
```

---

# 37. Final Confirmation

只有以下条件同时满足才能进入最终 5×3：

```text
S5 每个目标 phase 真正由目标 fault 主导
S7 A/B/C 稳定
S8 D 至少在开发 drift 中发生一次真实 topology event
```

否则：

```text
停止
```

不要跑 60 个大矩阵得到无意义 null。

---

## 37.1 正式矩阵

```text
5 model seeds
×
3 source-disjoint online streams
×
A/B/C/D
```

如果计算成本过高，动态主比较至少：

```text
5 × 3 × C/D
```

A/B 可作为较小公共基线，但最好完整保留。

---

# 38. 统计单位

统计单位：

```text
model seed × online stream
```

不能把：

```text
16 hosts × 2000 intervals
```

当成 32000 次独立重复。

报告：

```text
mean
std
median
paired D-C
bootstrap CI
```

---

# 39. 必须新增的单元测试

## Test 1 — RAM_CAP_SCALE legacy

```text
RAM_CAP_SCALE=1.0
```

必须与旧 RPiEdge capacity 完全一致。

---

## Test 2 — RAM heterogeneity preserved

例如：

```text
RAM_CAP_SCALE=0.5
```

必须：

```text
first8 = 4295*0.5
last8  = 8192*0.5
```

---

## Test 3 — Dynamic capacity visible to model

phase boundary 前后：

```text
graph_context["capacities"]
```

必须变化。

---

## Test 4 — No future capacity leakage

修改：

```text
capacity[t+1:]
```

不得影响：

```text
prediction <= t
```

---

## Test 5 — Graph source host correctness

```text
before_placement[t,c]=2
proposed dst=8
```

必须：

```text
edge 2->8
```

不依赖 previous proposed host。

---

## Test 6 — Rejected previous proposal 不污染 current source

```text
t-1 proposal Host8
actual remained Host2
t current proposal Host3
```

必须：

```text
edge 2->3
```

不能：

```text
8->3
```

---

## Test 7 — overload_mask

同时 CPU/RAM overload：

```text
ratio=[1.2,1.5,0.8]
```

必须：

```text
mask=[1,1,0]
dominant=RAM
```

---

## Test 8 — phase fault dominance

synthetic mini scenario：

```text
RAM capacity drastically lower
CPU/Disk generous
```

应产生 RAM dominant。

---

## Test 9 — source split disjoint

必须：

```text
train_vm ∩ dev_vm = empty
train_vm ∩ test_vm = empty
dev_vm ∩ test_vm = empty
```

---

## Test 10 — online sampler rare class

buffer 存在 CPU/RAM/Disk event 时：

```text
一次 update 必须能够按注册比例抽到三类
```

---

## Test 11 — mature label causality

t label 只能在：

```text
t+1 or registered maturation point
```

进入 replay。

---

## Test 12 — dynamic novelty independent of eligibility

即使：

```text
所有 token 至少一个 expert eligible
```

只要：

```text
entropy 高 + matured loss 持续升高
```

v3 candidate gate 仍可触发。

---

## Test 13 — shadow expert no prediction impact

shadow mode：

```text
logits before == logits after
```

误差建议：

```text
<1e-6
```

---

## Test 14 — shadow activation continuity

activate+ramp step0：

```text
max logit diff <1e-4
```

---

## Test 15 — recurrence same ID

```text
Pattern A
Pattern B
Pattern C
Pattern A
```

旧 dormant expert A 应优先 reactivate。

---

## Test 16 — optimizer state

active expert topology 变化后：

```text
surviving Adam exp_avg
surviving Adam exp_avg_sq
```

必须保持。

---

## Test 17 — resume exact

连续 run 与 save/resume：

```text
model hash
optimizer
topology
predictions
```

一致。

---

# 40. 推荐新增/修改文件

## 新增

```text
docs/FTMOE_ONLINE_PROTOCOL_020.md
docs/FTMOE_PROTOCOL020_PROBLEM_LOG.md

recovery/PreGANSrc/src/ftmoe_online_s7.py
recovery/PreGANSrc/src/ftmoe_dynamic_expert_v3.py

simulator/workload/BitbrainWorkloadProtocol020.py

prepare_ftmoe_protocol020_capacity_scan.py
analyze_ftmoe_protocol020_capacity_scan.py
prepare_ftmoe_protocol020_drift.py
build_ftmoe_protocol020_adaptation_dataset.py
train_ftmoe_protocol020_samedomain.py
run_ftmoe_protocol020.py
analyze_ftmoe_protocol020.py

test_ftmoe_protocol020_simulator.py
test_ftmoe_protocol020_graph.py
test_ftmoe_protocol020_online.py
test_ftmoe_protocol020_dynamic_v3.py
```

---

## 修改

```text
simulator/environment/RPiEdge.py
```

增加：

```text
RAM_CAP_SCALE
```

---

```text
recovery/PreGANSrc/src/ftmoe_ablation.py
```

增加：

```text
before_placement graph semantics
per-sample capacity
```

---

```text
Replay
```

增加：

```text
before_placement
capacity window
overload metadata
```

---

# 41. Capacity Controller 推荐实现

不要在不同 phase 中直接重新创建整个 simulator。

建议新建：

```python
class CapacityController:
    def __init__(self, schedule):
        self.schedule = schedule

    def current(self, step):
        ...
```

返回：

```text
cpu_scale
ram_scale
disk_scale
```

---

## 41.1 在 phase boundary 更新 Host

由于 RAM/Disk object 目前只保存：

```text
size/read/write
```

可以更新：

```text
host.ipsCap
host.ramCap.size
host.diskCap.size
```

但实施前必须检查：

```text
Host.getIPSAvailable()
Host.getRAMAvailable()
Host.getDiskAvailable()
```

确认 available 是：

```text
current capacity - current demand
```

实时计算，而不是缓存旧 capacity。

如果有缓存，必须同步更新。

---

# 42. 重要：capacity phase 切换时不能瞬间制造非法状态而不记录

例如：

```text
RAM capacity:
4295 -> 1500
```

当前已运行 workload 可能立即变成：

```text
RAM demand 2500 > 1500
```

这可以解释为：

```text
background process / quota reduction
```

但必须记录：

```text
capacity-change induced fault
```

不要称为 workload demand surge。

建议 event 增加：

```json
{
  "fault_cause": "capacity_drop",
  "capacity_before": ...,
  "capacity_after": ...
}
```

---

# 43. 更推荐 gradual capacity drift 作为补充

除了 abrupt phase：

```text
RAM 1.0 -> 0.45
```

还可以做 gradual：

```text
1.00
0.90
0.80
0.70
0.60
0.50
0.45
```

每 50 interval 下降一次。

这更适合测试：

```text
dynamic expert 是否提前检测持续 drift
```

但第一轮优先 abrupt，容易解释。

---

# 44. 不建议做的事情

## 不要

```text
直接把所有 RAM 写死成 1000
```

---

## 不要

```text
根据 D-C 最好的容量选 scenario
```

---

## 不要

```text
RAM phase 只看 RAM demand mean
```

必须看真实 RAM fault label。

---

## 不要

```text
target fault <100 仍进入模型实验
```

---

## 不要

```text
把 capacity 改变对模型隐藏
```

除非明确注册为 hidden background contention 实验。

---

## 不要

```text
同时改 capacity、VM pool、loss、dynamic threshold
然后把性能变化归因于某一个组件
```

必须阶段化。

---

## 不要

```text
继续使用 unmatched-only 作为 dynamic birth 的唯一信号
```

---

## 不要

```text
PR-AUC 低归因于固定 threshold=0.5
```

PR-AUC 与单一 threshold 无关。

---

## 不要

```text
看到 F1 +0.005 就称 online learning 显著提升
```

同时看 PR-AUC 和 paired statistics。

---

# 45. 推荐其他模型实际执行顺序

后续模型收到本文件后，应严格执行：

```text
1. Checkout protocol-019，仅阅读，不覆盖。
2. 新建 protocol-020 分支。
3. 建 docs/FTMOE_ONLINE_PROTOCOL_020.md。
4. 建 problem log。
5. 实现 RAM_CAP_SCALE。
6. 写 legacy capacity tests。
7. 实现 overload_mask / overload_ratio。
8. 修 graph：before_placement -> proposed destination。
9. 增加 per-sample capacities 到 graph_context。
10. 写 graph causality tests。
11. 建 VM source-disjoint split。
12. 不加载模型，跑 CPU 单因素 capacity scan。
13. 不加载模型，跑 RAM 单因素 capacity scan。
14. 不加载模型，跑 Disk 单因素 capacity scan。
15. 输出所有 candidate data-only report。
16. 按预登记 data criteria 选 CPU/RAM/Disk capacity。
17. 建 CPU->RAM->Disk->CPU phase stream。
18. 检查每 phase fault dominance。
19. 若任何目标 fault 支持不足，停止模型实验。
20. 用 train cohort 建 same-domain adaptation dataset。
21. 用 dev cohort 选 checkpoint，指标与 PR-AUC gate 对齐。
22. 冻结 checkpoint。
23. 实现 rare-event interval sampling + online loss v3。
24. 先跑 A/B/C。
25. A/B/C 不稳定则停止。
26. 实现 Dynamic Expert v3。
27. Test novelty trigger、shadow expert、birth/ramp、retire/reactivate。
28. stationary stream 跑 C/D。
29. dynamic topology 无故增长则停止。
30. controlled drift dev 跑 C/D。
31. 必须确认 D 至少发生真实 topology event。
32. 如果 D 从未触发，不得解释为“动态专家无效”。
33. 只有机制真正触发后才能比较 D-C。
34. 开发通过后固定全部超参。
35. 最后跑 5 model seeds × 3 source-disjoint streams。
36. 保存所有失败、hash、配置、stream SHA 和代码 SHA。
```

---

# 46. 推荐第一轮 Capacity Scan 参数

为了让其他模型可以直接开始：

## CPU

```text
RAM_CAP_SCALE=1.0
DISK_CAP_SCALE=0.30

CPU_CAP_SCALE ∈:
1.00
0.90
0.80
0.75
0.70
```

---

## RAM

```text
CPU_CAP_SCALE=1.0
DISK_CAP_SCALE=0.30

RAM_CAP_SCALE ∈:
1.00
0.75
0.60
0.50
0.45
0.40
0.35
0.30
```

---

## Disk

```text
CPU_CAP_SCALE=1.0
RAM_CAP_SCALE=1.0

DISK_CAP_SCALE ∈:
0.30
0.25
0.22
0.20
0.1875
0.17
```

---

# 47. 推荐初始 Drift Config

如果 S4 的实际重放支持，可优先尝试：

```json
{
  "phase_len": 500,
  "phases": [
    {
      "name": "baseline",
      "cpu_scale": 1.0,
      "ram_scale": 1.0,
      "disk_scale": 0.30
    },
    {
      "name": "cpu_fault",
      "cpu_scale": 0.75,
      "ram_scale": 1.0,
      "disk_scale": 0.30
    },
    {
      "name": "ram_fault",
      "cpu_scale": 1.0,
      "ram_scale": 0.45,
      "disk_scale": 0.30
    },
    {
      "name": "disk_fault",
      "cpu_scale": 1.0,
      "ram_scale": 1.0,
      "disk_scale": 0.20
    },
    {
      "name": "cpu_recurrence",
      "cpu_scale": 0.75,
      "ram_scale": 1.0,
      "disk_scale": 0.30
    }
  ]
}
```

强调：

> 这只是预登记初始候选，不是最终保证有效的值。

最终 phase config 必须由 data-only scan 选出。

---

# 48. Protocol 020 的最终成功条件

Protocol 020 只有同时满足以下条件才可以称为成功：

## Simulator

```text
CPU/RAM/Disk 三类都有足够独立异常事件
phase 名称与真实 dominant fault 一致
rejection 没有失控
```

## Cold start

```text
same-domain adapted model 在 source-disjoint dev 上具有合理 PR-AUC
```

## Online

```text
C 在 stationary 长流中无明显遗忘
```

## Dynamic

```text
D 在 drift phase 中至少发生真实 topology event
```

## Mechanism

```text
新增 expert 不是由 normalization bug 或单次 outlier 触发
```

## Comparison

```text
D-C 用 paired source-disjoint streams 评估
```

---

# 49. 如果 Protocol 020 最终仍然 D-C≈0，如何解释

只有在以下条件都满足时：

```text
三类 fault drift 真实存在
Data Gate 满足
D trigger 确实发生
expert 真正 birth/activate
C/D 使用同样 replay/loss
D topology 真的发生变化
```

之后仍：

```text
D-C≈0
```

才能比较有力地得出：

> 在当前 workload/fault complexity 下，固定 4-expert MoE 的容量已足够，动态扩容没有额外收益。

这将是一个可信的负面结论。

而 Protocol 019 当前还没有走到这一步。

---

# 50. 最终建议

当前最优先的工作不是继续调：

```text
EAGate threshold
EMA 0.05
expert max
learning rate
```

而是：

> **先建立一个真正满足 CPU/RAM/Disk 三类 fault coverage、并且存在明确 fault-mode drift 的模拟环境。**

最推荐路线：

```text
RAM_CAP_SCALE
+
CPU_CAP_SCALE
+
DISK_CAP_SCALE
+
per-sample effective capacity input
+
actual before_placement graph
+
source-disjoint VM split
```

在数据层通过后，再进行：

```text
rare-event online training
+
Dynamic Expert v3
```

这是当前最能同时提升实验有效性、论文可解释性和动态专家可验证性的路线。
