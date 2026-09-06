# FT-MoE 在线微调实验问题复盘与系统性解决方案

> 适用仓库：`songwenhao074-maker/FT-MoE`  
> 目的：系统回顾协议 015/016 及后续 017/018 所暴露的问题，并给出足够详细、可直接交给其他模型/工程师继续实施的修复与验证方案。  
> 建议新实验代号：**Protocol 019 — Common-Input Repair + Same-Domain Initialization + Drift-Aware Online FT-MoE**

---

## 0. 执行结论

当前在线微调实验的主要问题**不是“在线微调本身已经证明失败”**，也不是“动态专家机制一定无效”，而是在线实验尚未建立一个足够合理的前提：

> **离线训练得到的 Protocol 014 v4 checkpoint，在进入当前 online workload 时，输入语义、资源尺度、任务生命周期、调度关系和异常分布已经发生明显改变。模型在第一步附近就处于 domain shift，而不是先处于一个合理工作点，再随时间产生 drift。**

因此，当前的在线结果不能直接回答“动态专家能否比普通在线微调更好”。  
正确的实验逻辑应该拆成三层：

1. **Cold-start compatibility**：模型刚进入在线环境时，输入和标签关系是否仍与训练时足够一致；
2. **Online adaptation stability**：在不改变专家数量时，普通微调或 MoE-only 微调能否稳定更新；
3. **Dynamic expert benefit**：只有环境确实出现新的、持续的路由/故障模式后，动态增删专家是否比“固定专家、同参数更新范围”的方法更快适应。

当前 015/016 把第 1 层和第 2/3 层混在一起，因此即使 D 失败，也不能据此否定动态专家。

**推荐停止继续对 015/016 直接搜索学习率、压力倍率或专家阈值。**  
下一步应新建独立 Protocol 019，保留 014/015/016 全部结果和归档不动。

---

# 1. 当前在线实验的事实回顾

## 1.1 Protocol 015：原生 BWGD2 在线实验

四组定义：

| 方法 | 在线更新 |
|---|---|
| A | 不更新 |
| B | 全参数在线微调 |
| C | 仅 MoE/EAGate 参数在线微调，专家数固定 |
| D | 与 C 相同的可训练参数 + 动态增删 EAGate 专家 |

共同设置：

- 起点：Protocol 014 的 v4 checkpoint；
- 模型种子：1；
- 回放种子：301；
- 300 个评分 interval；
- interval = 300 s；
- B/C/D 学习率：`3e-5`；
- 每 10 interval 更新一次；
- 最近 64 interval 为 replay buffer；
- 每次对 buffer 进行一轮训练；
- 延迟 1 interval 才让 ±1 容差标签成熟；
- 动态专家每 10 次 update（约 100 interval）检查一次。

后半程 F1：

| 方法 | Second-half F1 |
|---|---:|
| A | 0.412500 |
| B | 0.416667 |
| C | 0.419067 |
| D | 0.408207 |

关键现象：

1. A 不更新也只有约 0.41；
2. B/C/D 与 A 接近；
3. 离线参考集 F1 仍约 0.905；
4. 因此这不是典型的“在线更新导致遗忘”；
5. D 在 100 步时确实从 4 expert 增到 5 expert，但没有带来收益；
6. 300 步中容差异常 555 个，但**全部是 CPU 类**，RAM/Disk 为 0。

### 结论

Protocol 015 最主要证明的是：

> **014 模型对 native BWGD2 流存在明显冷启动域失配。**

它没有建立足够证据证明：

- B 会发生持续 catastrophic forgetting；
- C 比 B 更稳定；
- D 的动态专家机制能在真实新故障模式下发挥作用。

---

## 1.2 Protocol 015 分布审计发现的问题

### A. Offline 与 Online workload 不是同一个分布

离线训练：

- `OfflineTraceWorkloadV2`
- 16 固定 slot
- 固定短 trace
- CPU/RAM/Disk 使用专门映射
- I/O 四列基本为训练约定值
- CPU/Disk capacity 有缩放
- 任务机制和生命周期固定

在线：

- `BWGD2`
- 从 Bitbrain VM 轨迹池随机创建任务
- CPU/RAM/Disk 原始语义不同
- I/O 输入语义不同
- capacity 不同
- 任务完成、重新创建、slot 身份变化显著

这不是简单的数值倍率变化，而是 **feature semantics + temporal dynamics + workload lifecycle + schedule-outcome mapping** 的共同变化。

---

### B. 第 14 台主机存在病态归一化

原离线 normalization 是“每 Host × 每 Feature 使用训练最大值”。

某些 Host/Feature 在训练中没有被有效覆盖，例如第 14 台主机：

- RAM scale ≈ 0
- Disk scale ≈ 0
- CPU scale 极小

代码用 `1e-8` 兜底，因此在线正常负载会被放大到：

- `1e8`
- `1e11`
- 甚至更高

这些数仍然 finite，所以旧的 `isfinite()` 检查无法发现。

仅修这个问题后 F1 有改善，但改善很有限，因此：

> 它是确定存在的工程 bug，但不是全部根因。

---

### C. 图分支把“slot 替换任务”误判成“同一容器迁移”

当前 `ScheduleGraphEncoder` 通过比较：

```text
schedule[t-1, slot] -> schedule[t, slot]
```

构造 migration edge。

但在线环境中：

```text
slot 7 at t-1 = creationID 100
slot 7 at t   = creationID 208
```

这两者不是同一个容器。

如果前一个容器在 Host 2，后一个新容器被调度到 Host 8，旧图编码会把它当成：

```text
Host 2 -> Host 8 migration
```

实际上只是：

```text
old task died
new task created
```

审计中大量“slot target change”来自身份变化而非真实迁移。

这会直接污染：

- graph adjacency；
- incoming migration count；
- outgoing migration count；
- occupancy；
- graph attention；
- cross-attention 的 K/V。

---

### D. 拟调度与实际执行存在差异

模型看到的是 proposed schedule，但物理标签来自 `simulationStep()` 后的真实状态。

由于：

- 资源约束；
- 拒绝部署；
- migration feasibility；
- task completion；

部分 proposed placement 最终没有真实执行。

这本身不是 bug，因为预测时确实只能看到 proposed plan；但它意味着：

> 模型需要学习 `proposed schedule -> feasible execution -> overload` 的关系，而不是简单把 schedule 当成真实 future state。

旧离线环境中的这种关系与 BWGD2 在线环境不同。

---

## 1.3 Protocol 016：Adapted BWGD2

016 已尝试：

- CPU 限幅到训练量级；
- RAM ×2；
- I/O 固定为训练约定；
- 恢复训练 capacity；
- 增加基于训练数据统计的 synthetic dynamic disk。

结果：

| 指标 | 016 |
|---|---:|
| 主异常比例 | 2.90% |
| CPU | 106 |
| RAM | 3 |
| Disk | 30 |
| 冻结 A 全程 F1 | 0.213813 |
| 冻结 A 后半程 F1 | 0.143911 |

后半程模型：

- Recall ≈ 79.59%
- 但 False Positive Rate ≈ 19.31%
- Precision 只有约 7.91%

### 为什么均值接近仍然失败？

因为以下分布仍不匹配：

#### CPU 状态分布

训练数据近似呈现：

```text
大量低状态 + 大量高状态
```

而 016 更像：

```text
大量中间状态
```

也就是说：

> mean 相似，但 distribution shape 和 temporal persistence 完全不同。

#### Host 聚集不同

训练：

```text
≥3 container 聚集到同 Host ≈ 10%
```

016：

```text
≈ 1%
```

异常往往不是由单容器值决定，而由**主机级聚合和调度聚集**决定。

#### RAM 高尾不足

训练 RAM 的高分位明显高于 016。

简单 ×2 不能同时匹配：

- mean；
- p95；
- temporal persistence；
- host aggregation。

---

# 2. 当前问题的根因分级

## P0：必须先解决，否则后续在线实验没有解释力

### P0-1 冷启动 domain shift

**严重度：最高**

只要 Frozen A 一进入在线流就从约 0.90 跌到 0.2–0.45，D 的表现就无法解释成“持续学习能力”。

### P0-2 归一化病态

**严重度：高，确定性高**

必须修复所有：

```text
training_scale ≈ 0
training_scale 极小
```

的列。

### P0-3 图身份语义错误

**严重度：高，确定性高**

必须让图迁移边只代表“同一个 creation ID 的迁移”。

---

## P1：必须在正式 online comparison 前解决

### P1-1 Online buffer 严重重复利用

当前：

- buffer = 最近 64 interval；
- 每 10 interval update；
- 每次遍历整个 buffer。

一个样本从加入 buffer 到被淘汰，会出现在约 6～7 次 update 中。

因此即使写的是：

```text
1 epoch/update
```

同一条成熟样本实际上会被训练很多次。

这会造成：

- online overfitting；
- recent distribution 过度放大；
- full-parameter B 容易发生漂移；
- C/D 的 router 也容易被短期局部状态带偏。

---

### P1-2 在线类别极不平衡，但仍使用固定 offline loss

当前 detection class weight 基本固定：

```text
normal = 0.6
fault = 2.0
```

但 016 的异常率只有约 2%。

这与训练分布不同。

同时 classification loss 只有 positive 样本才计算。

如果最近 64 interval 中：

- 只有 CPU；
- 或没有 RAM/Disk；
- 或完全无异常；

那么 online classification / ranking gradient 非常不稳定。

---

### P1-3 固定阈值 0.5 在 prior shift 下可能严重失配

016 Frozen A：

- recall 不低；
- FPR 仍很高；
- precision 极低。

这说明概率分布本身可能发生 calibration shift。

因此必须同时报告：

- Fixed-threshold F1；
- PR-AUC；
- Precision；
- Recall；
- FPR；
- calibration metrics。

不能只用 F1 判断 representation 是否完全失效。

---

## P2：动态专家机制本身需要重新设计

### P2-1 “只要有 unmatched 就加 expert”过于敏感

当前 D 大致是：

```text
每 100 interval：
    如果存在 unmatched
    且 experts < 8
        新增 1 expert
```

015 首个周期：

```text
1600 host-samples
118 unmatched
```

就新增了 expert。

问题是：

> unmatched 并不等于“出现了一个稳定的新故障模式”。

它可能只是：

- OOD noise；
- normalization outlier；
- 某个 Host scale 爆炸；
- transient schedule；
- 一个短暂 workload cluster。

---

### P2-2 新 expert 的出生方式可能造成输出扰动

当前新 expert：

- key = unmatched vectors 均值；
- threshold = 0；
- expert 最后一层输出初始化为 0；
- detection/class heads 初始化为 0。

虽然新 expert 自身初始输出接近 0，但它一旦被路由选中，会参与 weight normalization。

例如原本：

```text
E1 0.7
E2 0.3
```

新增 E5 后可能变成：

```text
E1 0.5
E2 0.2
E5 0.3
```

而 E5 仍未训练。

因此即使 E5 输出是 0，也会**稀释成熟 expert 的贡献**。

这是一种很可能导致短期 performance drop 的机制。

---

### P2-3 “activation_count == 0 就删除”不适合 recurring drift

如果环境模式：

```text
CPU-heavy -> RAM-heavy -> CPU-heavy
```

旧 CPU expert 在 RAM 阶段可能暂时不用。

直接删除意味着：

> 环境恢复到 CPU-heavy 时，原知识已经被物理删除。

真正的 continual/dynamic MoE 更适合：

```text
retire / sleep / reactivate
```

而不是立即永久删除。

---

# 3. 推荐总体方案：Protocol 019

## 总体原则

Protocol 019 必须分阶段。

任何阶段失败，都**停止后续阶段**，而不是继续调参数直到 D>B。

建议流程：

```text
P19-S0 归档与协议登记
      ↓
P19-S1 共同输入修复
      ↓
P19-S2 Cold-start compatibility
      ↓
P19-S3 Same-domain common offline adaptation（必要时）
      ↓
P19-S4 Online optimizer stabilization
      ↓
P19-S5 Safe dynamic expert
      ↓
P19-S6 Stationary long-stream validation
      ↓
P19-S7 Natural / controlled drift validation
      ↓
P19-S8 5 seeds × 3 streams final confirmation
```

---

# 4. P19-S0：建立独立协议，禁止覆盖旧结果

## 必须创建

建议：

```text
docs/FTMOE_ONLINE_PROTOCOL_019.md
artifacts/ftmoe_online/protocol_019/
```

记录：

```json
{
  "base_checkpoint": "protocol014",
  "input_contract_version": 2,
  "normalization_version": 2,
  "graph_semantics_version": 2,
  "online_optimizer_version": 2,
  "dynamic_expert_version": 2
}
```

禁止修改：

```text
backup/ftmoe_protocol014_accepted_20260905
artifacts/ftmoe_online/pilot
artifacts/ftmoe_online/adapted_bwgd2_016
```

017/018 如果已有运行结果，也必须保持独立。

---

# 5. P19-S1：先修“共同输入层”

这是整个方案最重要的一步。

---

## 5.1 建立统一 Input Contract

建议新增：

```text
recovery/PreGANSrc/src/ftmoe_input_contract.py
```

定义明确 schema：

```python
HostFeature = [
    cpu_demand,
    ram_space,
    ram_read_or_network_rx,
    ram_write_or_network_tx,
    disk_space,
    disk_read,
    disk_write,
]
```

必须把每一列的：

- 物理含义；
- 单位；
- 是否真实数据；
- 是否 synthetic；
- normalization；
- capacity；
- missing-value 规则；

写入 manifest。

如果某列在训练中其实恒为 1，那么 online 不能突然换成另一个真实物理量而仍声称语义相同。

---

## 5.2 Legacy checkpoint 的安全归一化兜底

为了先诊断旧 014 checkpoint，可以增加**兼容模式**：

```python
def safe_scale(scale, training_stats, host_group):
    if scale < epsilon:
        return same_hardware_group_feature_scale
    if scale < low_coverage_threshold:
        return same_hardware_group_feature_scale
    return scale
```

推荐 host group：

```text
Host 0–7   : Raspberry Pi 4GB
Host 8–15  : Raspberry Pi 8GB
```

禁止使用 online label 或 online future statistics。

### 推荐判据

训练时为每列记录：

```text
count_nonzero
max
p50
p95
p99
```

若：

```text
count_nonzero < 12
```

或：

```text
max < 1e-4 * same_group_p95
```

则标记：

```text
under-covered
```

对 under-covered 列使用：

```text
same_hardware_group + same_feature
```

统计。

### 必须记录

```json
{
  "fallback_columns": [
    {"host": 13, "feature": "ram", "reason": "training_max_zero"}
  ]
}
```

### 验收

在任何 online run 前：

```text
normalized_abs_max < 50
```

必须只是**报警阈值**，不是强行 clip。

建议额外输出：

```text
p50
p95
p99
max
fraction(|x| > 5)
fraction(|x| > 10)
```

不能只做 `isfinite()`。

---

## 5.3 推荐的长期方案：capacity-ratio normalization

如果进行新的 common retraining，不建议继续使用 per-host max。

建议核心资源显式构造：

\[
u^{CPU}_{h,t} = \frac{CPUDemand_{h,t}}{CPUCapacity_h}
\]

\[
u^{RAM}_{h,t} = \frac{RAMDemand_{h,t}}{RAMCapacity_h}
\]

\[
u^{Disk}_{h,t} = \frac{DiskDemand_{h,t}}{DiskCapacity_h}
\]

然后：

\[
x=\log(1+u)
\]

这样：

```text
1.0 capacity violation
```

在不同硬件/数据集下具有相同物理意义。

仓库已有 `CapacityRatioGraphEncoder` 候选，可以作为参考，但正式使用必须新协议重新训练，不得直接把旧分数解释成同一个模型。

---

# 6. P19-S1：修正 Graph Identity

## 6.1 不推荐继续使用

```text
schedule[t-1, same_slot] -> schedule[t, same_slot]
```

来推断 migration。

---

## 6.2 推荐新图定义：Current Proposed Migration Graph

在预测时刻 t 已知：

```text
creation_ids[t, c]
before_placement[t, c]
proposed_schedule[t, c, :]
```

因此直接构造：

```python
for each active container c:
    src = before_placement[t, c]
    dst = argmax(proposed_schedule[t, c])

    if src >= 0 and dst >= 0 and src != dst:
        add edge src -> dst
```

这是：

- causal；
- 不依赖未来；
- 不依赖 slot 跨时间连续；
- 明确表示“当前计划迁移”。

新的 occupancy：

```python
valid = creation_ids[t] >= 0
occupancy[h] = sum(
    proposed_schedule[t, c, h]
    for c if valid[c]
)
```

空 slot 不再计入 occupancy。

---

## 6.3 如果暂时必须保留旧图结构

至少增加：

```python
same_identity = (
    creation_ids[t-1] >= 0
    & creation_ids[t] >= 0
    & (creation_ids[t-1] == creation_ids[t])
)
```

migration 计算必须乘：

```text
same_identity
```

新任务和旧任务结束：

```text
NO migration edge
```

---

## 6.4 需要修改的数据接口

当前 `stream.npz` 已经保存：

```text
creation_ids
before_placement
after_placement
```

但 `Replay.window()` 没有传给模型。

Protocol 019 应新增：

```text
identity_windows
before_host_windows
valid_slot_windows
```

模型接口可改为：

```python
model(
    time_windows,
    schedule_windows,
    graph_time_windows,
    graph_context={
        "creation_ids": ...,
        "before_placement": ...,
        "valid_mask": ...
    }
)
```

为保持历史代码兼容：

```text
graph_context=None
```

继续走 legacy path。

---

# 7. P19-S2：Cold-start compatibility 必须单独验收

在任何 B/C/D 更新之前，先只运行 A。

目的不是“选择一个对模型有利的场景”，而是判断：

> 当前 scenario 能否合理用于研究 online adaptation。

---

## 7.1 Data-only gate

必须先检查，不看模型结果：

### 资源

每类资源：

```text
p10
p50
p90
p95
p99
zero ratio
training-range exceed ratio
```

### Host aggregation

```text
mean active containers
P(host >= 2 containers)
P(host >= 3 containers)
max host occupancy
```

### Event

```text
anomaly prevalence
CPU/RAM/Disk counts
mean anomaly duration
p95 anomaly duration
recovery duration
```

### Schedule

```text
proposed migrations
executed migrations
rejection rate
proposed != actual ratio
```

### Temporal structure

至少：

```text
lag-1 autocorrelation
lag-12 autocorrelation
lag-30 autocorrelation
```

不能再仅比较：

```text
mean demand
class fraction
```

---

## 7.2 Model compatibility gate

Data-only 配置冻结后，运行 Frozen A。

这个结果**不允许再用于反向修改 scenario 参数**。

建议报告：

```text
fixed threshold F1
PR-AUC
precision
recall
FPR
raw-label F1
tolerance-label F1
```

若：

```text
Frozen A F1 极低
且 PR-AUC 也低
```

则说明旧 014 checkpoint 不适合作为这个 domain 的 online 起点。

正确动作：

```text
进入 S3：common offline adaptation
```

而不是继续调 capacity 直到 A/D 分数变高。

---

# 8. P19-S3：建立 Same-Domain Common Starting Point

这是当前最推荐的方向。

## 8.1 为什么需要

真正研究：

```text
online learning
```

应该是：

```text
same-domain pretrained model
    +
future stream changes
```

而不是：

```text
offline synthetic domain
    ->
completely different online domain
```

后者测试的是 zero-shot domain generalization，不是 online continual learning。

---

## 8.2 数据划分必须按来源隔离

不允许：

```text
同一批 VM trace 开头
只换 random seed
```

当成独立 train/dev/test。

推荐：

### Bitbrain 路线

按 VM ID 划分：

```text
train VM set
dev VM set
confirmation VM set
```

或按原始时间段划分：

```text
train time range
dev future time range
test later time range
```

不能混用。

### Google 2011 路线

优先：

```text
calibration segment
offline adaptation train segment
offline adaptation dev segment
online stream 1
online stream 2
online stream 3
```

所有时间区间严格不重叠。

---

## 8.3 Common adaptation 方式

推荐先使用 014 v4 warm-start，而不是立刻 from-scratch。

所有模型 seed 分别：

```text
014 v4 seed_i
    ↓
same-domain offline adaptation
    ↓
adapted_v4_seed_i
```

共同训练：

- 修复后的 normalization；
- 修复后的 graph semantics；
- 同一 online feature contract；
- 同一 workload domain。

### 预算

建议开发阶段固定：

```text
15 epochs
lr = 1e-4
AdamW
batch = 32
```

如果 warm-start 过度改变模型，可比较：

```text
5 / 10 / 15 epochs
```

但选定后全部 seed 使用同一预算。

禁止对 D 单独寻找更好的起点。

---

# 9. P19-S4：重新设计在线 replay/update

---

## 9.1 不再每 10 步完整重复训练最近 64 interval

推荐改成两层 memory：

```text
Recent Buffer      = 128 intervals
Anchor Buffer      = offline/same-domain 128–256 intervals
```

每 10 interval：

```text
sample 24–32 recent intervals
+
sample 8–16 anchor intervals
```

进行 1 次 update。

每条 recent interval 记录：

```text
times_sampled
```

设置：

```text
max_replay_count <= 3
```

避免同一样本被 6～7 次重复遍历。

---

## 9.2 必须保留 interval-level sample

不能只把某个 anomalous Host 抽出来，因为：

- graph 使用 16 Host；
- schedule 使用 16 slot；
- cross-attention 使用全 Host。

所以 balanced sampling 的单位是：

```text
interval
```

可把 interval 分类为：

```text
normal-only
contains CPU anomaly
contains RAM anomaly
contains Disk anomaly
multi-event
```

然后分层抽样。

---

## 9.3 Anchor loss 防止遗忘

建议增加：

\[
L =
L_{online}
+
\lambda_{anchor}L_{anchor}
+
\lambda_{distill}L_{distill}
\]

其中：

```text
L_online
```

是成熟 online label。

```text
L_anchor
```

是同域离线 anchor buffer 原标签。

```text
L_distill
```

让当前模型在 anchor 上保持接近 online 开始时的 teacher：

\[
KL(p_\theta(x_a)\,||\,p_{\theta_0}(x_a))
\]

建议初始：

```text
lambda_anchor = 0.25
lambda_distill = 0.10
```

开发时可以做小范围固定比较，但所有 B/C/D 使用相同原则。

---

## 9.4 B 的学习率应该分组

当前 B 全部参数同一个 LR。

更合理：

```text
backbone / graph / cmha: 0.1 × LR
heads / MoE / router:     1.0 × LR
```

例如：

```text
base lr = 3e-5

encoder      = 3e-6
graph        = 3e-6
cmha         = 3e-6
detection    = 3e-5
class heads  = 3e-5
moe/eagate   = 3e-5
```

这样不会故意削弱 B，也更接近实际 online fine-tuning。

---

# 10. P19-S5：重新设计 Dynamic Expert

这是 D 能否真正成为论文贡献的核心。

---

## 10.1 增加 expert 不能只看一次 unmatched

建立 EMA：

```python
unmatched_ema =
    0.9 * unmatched_ema
    + 0.1 * current_unmatched_ratio
```

只有同时满足：

```text
unmatched_count >= 128
unmatched_ema >= 0.05
连续 >= 3 个 routing windows
expert_count < max_experts
```

才进入“候选新增”。

这只是建议初值，必须在协议前登记。

---

## 10.2 必须区分 normalization OOD 和 semantic OOD

如果：

```text
|normalized_feature| > 20
```

之类的 extreme scale outlier 占主要 unmatched，

禁止通过新增 expert 去“学习 normalization bug”。

routing event 日志必须保存：

```text
unmatched feature p95
unmatched host IDs
unmatched resource ratios
```

---

## 10.3 Safe Expert Birth

不要继续使用：

```text
zero-output new expert immediately participates in routing
```

推荐以下策略。

### Step A：聚类 unmatched

从最近 unmatched vectors：

```text
最多保存 256–512
```

计算：

```text
centroid
```

第一版可以只 k=1，避免引入复杂超参。

---

### Step B：找到 nearest existing expert

按 key cosine similarity：

```python
parent = argmax cosine(centroid, expert_key)
```

---

### Step C：clone parent expert

新 expert：

```text
weights = deepcopy(parent weights)
detection head = deepcopy(parent)
class head = deepcopy(parent)
```

而不是 zero init。

这样 birth moment 的输出近似连续。

---

### Step D：新 key 使用 centroid

```text
new_key = normalize(centroid)
```

threshold 不要直接设 0。

建议根据 unmatched score 分布初始化到：

```text
70%–80% quantile
```

使其只接收目标 cluster。

---

### Step E：Routing Ramp

新增后前 N 次 update：

```text
gate_multiplier:
0.0 -> 0.25 -> 0.5 -> 0.75 -> 1.0
```

例如 N=5。

这样避免专家刚出生就抢走大量 routing probability。

---

## 10.4 推荐更安全的 Shadow Expert

更严格的版本：

```text
Candidate Expert
    ↓
先不参与真实 prediction
    ↓
用 unmatched + anchor 训练 3–5 update
    ↓
比较 candidate loss
    ↓
通过后 activate
```

activation 条件可定义为：

```text
candidate unmatched loss
至少优于 parent 5%
且 anchor loss 增幅 < 2%
```

这比“立刻添加”更容易产生可解释结果。

---

## 10.5 Expert Removal 改成 Retire

不要：

```text
activation_count == 0 -> delete
```

推荐：

### Dormant 条件

```text
expert age >= 500 intervals
activation_ema < 0.005
连续 5 个 adaptation windows
```

则：

```text
active -> dormant
```

dormant expert：

- 不参与正常 forward；
- 参数仍保留；
- optimizer state 可保留或压缩；
- key 继续保留用于 recurrence matching。

如果新 unmatched centroid 与 dormant key：

```text
cosine > 0.90
```

优先：

```text
reactivate old expert
```

而不是新增。

这能直接支持：

> recurring workload / recurring fault pattern。

---

# 11. Primary Comparison 必须改成 D vs C

如果论文要证明：

> 动态专家数量有价值

真正最干净的比较是：

```text
C：同样 MoE/EAGate 参数在线训练，但固定专家数
D：完全相同训练范围 + 动态专家
```

因此：

\[
D-C
\]

应作为 dynamic expert 的**主要机制比较**。

B 仍然保留，因为它代表：

```text
generic full-parameter online fine-tuning
```

但是：

\[
D-B
\]

同时包含：

- 可训练参数范围差异；
- backbone freezing；
- dynamic topology；

不能单独证明 dynamic expert。

推荐最终声明逻辑：

```text
D > C
=> dynamic topology contribution

C/B
=> restricted MoE tuning vs full-model tuning

D > B
=> overall proposed online strategy advantage
```

---

# 12. P19-S6：先跑 Stationary Long Stream

动态专家不应该在平稳环境下不断增长。

因此第一条正式长流必须是：

```text
stationary / same-domain
```

目标：

1. A 保持稳定；
2. B/C 不明显忘记；
3. D 不应该频繁新增专家；
4. D 至少不能显著劣于 C。

建议 2000 interval。

必须报告：

```text
expert_count(t)
mean_active(t)
unmatched_ratio(t)
F1(t)
PR-AUC(t)
offline-anchor F1(t)
```

---

# 13. P19-S7：真正建立 Drift Scene

当前 BWGD2 长时间运行并不天然等于 concept drift。

如果新任务不断从同一批 VM 轨迹开头生成，2000 步只是在重复同一种任务生成机制。

要证明动态专家，需要明确的“新模式出现”。

---

## 13.1 Natural Drift：优先

### Google 2011

使用真实连续时间段：

```text
Phase A
Phase B
Phase C
```

保持：

- task ID；
- CPU/RAM/Disk 同源；
- 原始绝对时间；
- 不循环短流。

每个 phase 从不同连续时间范围获得。

---

## 13.2 Controlled Drift：作为补充

如果真实数据自然变化不明显，可以预登记：

```text
Phase 1: CPU-heavy VM group
Phase 2: mixed group
Phase 3: RAM-heavy group
Phase 4: return to Phase 1
```

VM group 的定义只能根据：

```text
training/development workload statistics
```

不能根据 D-B 结果选择。

Phase 4 的 recurrence 非常重要：

> 可测试 retired expert 是否被重新激活。

---

# 14. Protocol 017 应该如何看待

017 的 capacity sweep 有一个合理点：

> capacity 选择不使用模型 F1。

这是正确的。

但是当前 selection 主要依据：

```text
normal/CPU/RAM/Disk class fraction
```

接近训练分布。

这仍然不足。

两个流可以：

```text
class fraction 完全一样
```

但：

```text
CPU temporal pattern
host aggregation
event duration
migration feasibility
normalized OOD ratio
```

完全不同。

因此 017 最多作为：

```text
scenario diagnostic
```

不能把“异常比例匹配”当成 domain alignment 已解决。

Protocol 019 如果继续做 capacity selection，应将 score 扩展为：

\[
S_{scenario} =
w_1 D_{class}
+w_2 D_{resource\ quantile}
+w_3 D_{occupancy}
+w_4 D_{duration}
+w_5 D_{temporal}
+w_6 D_{schedule}
\]

所有项只由 data statistics 决定，不读取模型分数。

---

# 15. Protocol 018 / Google 2011 的推荐定位

Google 2011 是目前更有价值的长期方向，但应定位为：

> **real-trace-driven simulated overload**

而不是：

> Google 原始数据自带 CPU/RAM/Disk fault labels。

推荐在使用前进一步修正：

1. calibration segment 与 online evaluation 完全隔离；
2. source task cohort 不能因未来 performance 选择；
3. 长任务筛选造成 survival bias 必须报告；
4. 三条 confirmation stream 尽量使用：
   - 不重叠时间；
   - 不重叠 task cohort；
5. 第一次使用 Google 前必须先完成：
   - normalization repair；
   - graph identity repair；
6. 不建议拿原 014 checkpoint 直接对 Google 流做 D vs B 结论；
7. 应先 same-domain adaptation，再进入 online comparison。

---

# 16. Evaluation Protocol 重构

---

## 16.1 Prequential 顺序保持

当前这一点是正确的：

```text
predict(t)
↓
execute / reveal physical outcome
↓
mature label
↓
update
```

必须继续保持。

绝对禁止：

```text
label(t) -> update -> predict(t)
```

---

## 16.2 Raw label 与 ±1 tolerance 必须分开

建议：

### Primary physical metric

```text
raw post-action overload
```

### Alignment-sensitive secondary metric

```text
±1 tolerance
```

如果为了与旧协议一致仍以 tolerance 为主，报告中也必须同时给 raw。

在线训练可以继续使用 delayed tolerance，但必须标注：

```text
t-1 sample at time t
```

才成熟。

---

## 16.3 新增必须报告的 detection metrics

每条流：

```text
F1
Precision
Recall
FPR
PR-AUC
ROC-AUC（辅助）
Brier score / ECE（calibration）
```

固定 threshold：

```text
0.5
```

继续保留作为 primary comparable result。

如果研究 adaptive threshold，应作为额外方法，不得偷偷改变主实验阈值。

---

## 16.4 Diagnosis 不再只看 HR=NDCG

当前单资源 label 下：

```text
HR@100% == NDCG@100%
```

因此在线实验应增加：

```text
CPU recall
RAM recall
Disk recall
macro-F1
balanced accuracy
confusion matrix
```

每个类别必须报告 support。

---

## 16.5 Online adaptation 专属指标

新增：

### Adaptation Lag

发生 drift 后达到：

```text
95% of post-adaptation stable F1
```

所需 interval。

### Area Under Prequential F1

对整个 stream：

\[
AUPF1
=
\frac{1}{T}
\sum_t F1_{rolling}(t)
\]

### Forgetting

固定 anchor/reference set：

\[
Forget =
F1_{start}
-
F1_{end}
\]

### Dynamic Expert Cost

```text
number of expert additions
number of retirements
number of reactivations
average active experts
peak experts
prediction latency
update latency
memory usage
```

---

# 17. 最终统计设计

正式确认：

```text
5 model seeds
×
3 disjoint online streams
=
15 paired runs / method
```

四组全部使用：

```text
同一 stream
同一起点 seed
同一成熟 label
```

主要 paired difference：

```text
D - C
D - B
C - B
```

统计单位是：

```text
model-seed × stream
```

不能把 16×2000 个 Host 样本当作 32000 个独立实验重复。

推荐报告：

```text
mean
std
median
15 paired differences
bootstrap CI over seed-stream pairs
```

如需时间 block bootstrap：

```text
block length >= 12 intervals
```

避免破坏窗口依赖。

---

# 18. 推荐代码修改清单

## 18.1 新增

```text
recovery/PreGANSrc/src/ftmoe_input_contract.py
recovery/PreGANSrc/src/ftmoe_normalization.py
recovery/PreGANSrc/src/ftmoe_dynamic_expert_v2.py
analyze_ftmoe_domain_alignment.py
build_ftmoe_online_adaptation_dataset.py
run_ftmoe_protocol019.py
docs/FTMOE_ONLINE_PROTOCOL_019.md
```

---

## 18.2 修改

### `prepare_ftmoe_scenario.py`

确保导出：

```text
creation_ids
valid_slot_mask
before_placement
after_placement
proposed_schedule
actual_migrations
capacities
```

并在 manifest 写清 feature semantics。

---

### `run_ftmoe_online.py`

修改：

```text
Replay.window()
```

返回 identity graph context。

替换：

```text
buffer=deque(maxlen=64)
```

为：

```text
RecentReplayMemory
AnchorReplayMemory
```

并记录每样本 replay count。

---

### `ftmoe_ablation.py`

给 `ScheduleGraphEncoder` 增加：

```text
identity-aware migration
valid occupancy mask
```

旧 path 继续兼容：

```text
graph_context=None
```

---

### `ftmoe_online.py`

替换：

```text
OnlineEAGate.adapt()
```

为：

```text
collect_stats()
propose_candidate()
shadow_train()
activate_candidate()
retire_expert()
reactivate_expert()
```

---

# 19. 必须新增的测试

其他模型实施时，**每修改一个阶段先写测试，再运行实验。**

---

## Test 1：Normalization Zero Coverage

构造：

```text
training scale = 0
online value > 0
```

验证：

```text
使用同硬件同 feature fallback
不存在 1e8 / 1e11 放大
```

---

## Test 2：Identity Replacement Is Not Migration

```text
t-1:
slot 3, creationID=10, Host 2

t:
slot 3, creationID=20, Host 8
```

必须：

```text
NO edge 2 -> 8
```

---

## Test 3：Real Migration Is Migration

```text
same creationID
Host 2 -> proposed Host 8
```

必须：

```text
edge 2 -> 8
```

---

## Test 4：Empty Slot Does Not Count Occupancy

空 slot 即使 scheduler matrix 有默认 row，也不能增加 occupancy。

---

## Test 5：No Future Label Leakage

改变：

```text
raw_labels[t+2:]
```

必须不影响：

```text
prediction <= t
update <= t
```

---

## Test 6：Replay Exposure Cap

每条 interval：

```text
times_sampled <= configured_max
```

---

## Test 7：Expert Birth Output Continuity

新增 expert 前后，同一个 batch：

```text
max |logit_after - logit_before|
```

在 birth 的第一步应接近 0。

推荐阈值：

```text
< 1e-4
```

如果使用 ramp/shadow activation。

---

## Test 8：Dormant Expert Can Reactivate

模拟：

```text
Pattern A
Pattern B
Pattern A
```

确认：

```text
old A expert ID 被恢复
```

而不是重复创建无限新 expert。

---

## Test 9：Optimizer State Preservation

expert topology 改变后：

```text
surviving parameter IDs
Adam exp_avg
Adam exp_avg_sq
```

必须保持。

---

## Test 10：Resume Exact Equivalence

连续运行 N 步：

```text
Run A
```

和：

```text
Run B:
N/2 save
resume
continue
```

最终：

```text
model hash
optimizer hash
expert topology
predictions
```

必须完全一致。

---

# 20. 建议的实验执行顺序

## Step 1

不要再启动新的 D/B comparison。

先：

```text
实现 normalization v2
实现 graph identity v2
```

---

## Step 2

重新生成一个 300-step development stream。

只做：

```text
data-only distribution report
```

---

## Step 3

冻结 scenario。

运行：

```text
Frozen A
```

不更新。

---

## Step 4

如果 A 冷启动仍很低：

```text
停止
```

进入：

```text
same-domain common offline adaptation
```

不要继续调 capacity 或 D 参数。

---

## Step 5

获得合理 common starting checkpoint 后，先比较：

```text
A/B/C
```

验证 online optimizer。

如果 B/C 都不稳定：

```text
不要启动 D
```

---

## Step 6

只有 C 稳定后，启动：

```text
D = C + dynamic topology
```

这样 D-C 才可解释。

---

## Step 7

300 步通过后：

```text
2000-step stationary stream
```

先证明动态机制不会无理由增长。

---

## Step 8

最后再上：

```text
natural drift
controlled drift
recurring drift
```

测试 dynamic expert。

---

# 21. 推荐验收门槛

这些门槛应在 Protocol 019 开始前固定，不根据结果改变。

## Data gate

完整长流至少：

```text
normal >= 5000 host-samples
CPU >= 100
RAM >= 100
Disk >= 100
```

如果真实场景无法满足 Disk，则明确改成 CPU/RAM 实验，不用 synthetic label 强行补齐。

---

## Cold-start gate

不建议用它“选择场景”，只用于决定是否需要 common adaptation。

建议最低可用线：

```text
Frozen A PR-AUC >= 0.60
Frozen A FPR <= 0.15
```

如果固定阈值 F1 很低但 PR-AUC 尚好，应先研究 calibration，而不是立即认定 representation 无效。

---

## Online stability gate

B/C 在 stationary stream：

```text
anchor/reference F1 drop <= 0.03
```

且：

```text
second-half F1 不低于 Frozen A 超过 0.03
```

否则先修 online optimizer。

---

## Dynamic expert gate

D 与 C：

```text
D-C mean > 0
```

并报告 15 个 paired run。

对于明确 drift 场景，建议额外要求：

```text
D adaptation lag < C
```

这比只看最后一个 F1 更符合 dynamic expert 的机制目标。

---

# 22. 不建议继续做的事情

## 不要继续

```text
不断调 RAM / Disk capacity
直到 D > B
```

---

## 不要

```text
根据 D-B 结果选择 workload VM
```

---

## 不要

```text
增加异常注入
只为了达到 0.65
```

---

## 不要

```text
把同一个 202-step trace 循环 10 次
然后称 2000-step continual stream
```

---

## 不要

```text
把 slot replacement 当 migration
```

---

## 不要

```text
只看平均 demand 接近就称分布对齐
```

---

## 不要

```text
把 016 Frozen A 后半程下降称为 catastrophic forgetting
```

因为 A 根本没有更新参数。

---

# 23. 最推荐的论文级实验结构

最终可以形成下面四个实验层级。

## Experiment 1 — Cold-start Domain Compatibility

比较：

```text
014 checkpoint
014 + input repair
same-domain adapted checkpoint
```

目的：

> 证明为什么需要共同输入和同域初始化。

---

## Experiment 2 — Online Fine-Tuning Strategy

比较：

```text
A frozen
B full fine-tune
C fixed MoE fine-tune
```

目的：

> 判断 MoE-only 是否比 full tuning 更稳定。

---

## Experiment 3 — Dynamic Expert Adaptation

比较：

```text
C fixed MoE
D dynamic MoE
```

环境：

```text
stationary
abrupt drift
gradual drift
recurring drift
```

目的：

> 直接证明 dynamic topology 的贡献。

---

## Experiment 4 — External Long-Trace Validation

使用：

```text
Google 2011
```

或其他独立真实长 trace。

目的：

> 证明方法不只对旧合成 replay 有效。

---

# 24. 给后续模型的任务指令模板

后续模型开始工作时，应严格按照以下格式：

```text
1. 读取：
   PROJECT_CONTEXT_LATEST.md
   docs/FTMOE_ONLINE_EXPERIMENT.md
   docs/FTMOE_ONLINE_PILOT_015.md
   docs/FTMOE_ONLINE_DISTRIBUTION_AUDIT_015.md
   docs/FTMOE_ADAPTED_BWGD2_PILOT_016.md
   docs/FTMOE_DATASET_AND_CAPACITY_REVIEW.md
   docs/FTMOE_CAPACITY_GOOGLE_EXECUTION_017_018.md

2. 检查：
   artifacts/ftmoe_online/capacity_017/result.json
   artifacts/ftmoe_online/google2011_018/result.json

3. 如果 result.json 不存在：
   不得称协议已完成。

4. 不修改：
   protocol014 archive
   015 outputs
   016 outputs

5. 新建 Protocol 019。

6. 第一阶段只做：
   normalization + graph identity repair。

7. 写测试并全部通过。

8. 生成新 development stream。

9. 先进行 data-only audit。

10. 冻结 scenario 后再跑 Frozen A。

11. Frozen A 明显不兼容：
    进入 common offline adaptation，
    禁止继续调 simulator 迎合模型。

12. B/C 稳定前不得启动 D 的正式比较。

13. D 的机制主比较使用 D-C。

14. 最终确认必须：
    5 model seeds × 3 disjoint streams。

15. 所有失败、停止条件、配置和 hash 均保留。
```

---

# 25. 证据来源（仓库内）

本方案基于以下当前仓库文件整理：

```text
PROJECT_CONTEXT_LATEST.md
docs/FTMOE_ONLINE_EXPERIMENT.md
docs/FTMOE_ONLINE_PILOT_015.md
docs/FTMOE_ONLINE_DISTRIBUTION_AUDIT_015.md
docs/FTMOE_ONLINE_MATCHED_SIMULATOR_PROPOSAL.md
docs/FTMOE_BWGD2_INPUT_ADAPTER_PROPOSAL.md
docs/FTMOE_ADAPTED_BWGD2_PILOT_016.md
docs/FTMOE_DATASET_AND_CAPACITY_REVIEW.md
docs/FTMOE_CAPACITY_GOOGLE_EXECUTION_017_018.md

run_ftmoe_online.py
prepare_ftmoe_online.py
prepare_ftmoe_scenario.py
run_ftmoe_capacity_study.py
prepare_google2011_pilot.py

recovery/PreGANSrc/src/ftmoe_online.py
recovery/PreGANSrc/src/ftmoe_ablation.py
recovery/PreGANSrc/src/ftmoe_end_to_end.py
```

---

# 26. 最终建议

如果只允许选择一个方向，推荐：

> **停止继续微调 015/016 的 D 参数，建立 Protocol 019：先做 normalization + graph identity 的共同修复，然后用与目标 online workload 同域的数据对 014 v4 做共同 offline adaptation；之后先验证 A/B/C，再引入“safe dynamic expert”，最后用具有明确自然/受控 drift 的长流验证 D-C。**

这是当前最能同时解决：

- 冷启动域失配；
- online loss 不稳定；
- dynamic expert 误触发；
- 图输入语义错误；
- 归一化爆炸；
- 论文实验因果解释不清；

的一条路线。

而 017 的容量调整和 018 的 Google 接入仍有价值，但应该被放在这个更严格的输入/实验框架下使用，而不是用来绕过 cold-start compatibility 问题。
