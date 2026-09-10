# FT-MoE Protocol 021：经审计的离线未见 Regime 在线适应与动态残差专家实验计划

> 日期：2026-09-10  
> 上游仓库：`songwenhao074-maker/FT-MoE`  
> 审计基线：`protocol-020`，HEAD `b56d63f901a79cf9bd1c2d1a8e39e88f23fc63ac`  
> 建议新分支：`protocol-021`  
> 协议名：**Protocol 021 — Audited Unseen-Regime Online Adaptation + Additive Dynamic Residual Experts**  
> 用途：可直接交给其他模型/工程 agent 执行。不得跳过 Gate，不得依据 D 是否优于 C 反向筛选场景。

## 0. 为什么新开 Protocol 021

Protocol 020 已经完成 RAM capacity control、`overload_ratio/mask`、Graph semantics v3、per-sample capacity、VM cohort split、CPU/RAM/Disk 主导场景、S6 same-domain adaptation、rare-event replay、Dynamic Expert v3 初版、R0 prequential 修复以及 R1 frozen-base residual correction。这些资产应全部保留。

但 Protocol 020 已同时包含 S0–S10、continuation、R0/R1、P01–P36 及多份修订计划，部分顶层 metadata 与真实 episode 配置也存在历史不一致。更关键的是，科学问题已经改变：

旧问题是“动态专家能否在 CPU/RAM/Disk phase 漂移中优于 A/B/C”；新问题应是：

> **当运行时出现相对于完整离线初始化链真正未覆盖、但具有因果可观察性和可学习规律的新 regime 时，在线方法能否利用成熟反馈改善后续预测；若固定在线模型仍有持续容量不足，动态专家能否提供独立收益？**

因此将 Protocol 020 冻结为历史基线，Protocol 021 专门回答 unseen-regime 在线适应。

---

## 1. 当前新增结果必须接受的事实

### 1.1 dev500 不是离线未见故障模式

Protocol 020 S6 的真实训练 episode 已包含：

```text
baseline:   CPU/RAM/Disk=1.00/1.00/0.90, RAM upper=1400
cpu_fault:  0.65/1.00/0.90, RAM upper=1400
ram_fault:  1.00/0.55/0.90, RAM upper=2400
disk_fault: 1.00/1.00/0.35
```

dev500 的主要 phase 正是这些模式；`cpu_recurrence` 只是已知 CPU profile 再现。因此不同 VM、不同 seed、不同任务实例、不同 phase 顺序不能单独称作“离线未见类型”。

R1 的“没有稳定胜过 A”只能解释为：**在已经覆盖的 profile 及其切换上，当前 online correction 没有稳定优势**，不能外推为“在线学习新模式无效”。

### 1.2 数据层三类 fault 已经能构造出来

Protocol 020 S5 已能形成 CPU/RAM/Disk dominant phase，并满足开发期 dominance/rejection 条件。Protocol 021 不应继续把主要精力放在“再找一个 RAM/CPU/Disk 容量倍率”；容量保留为物理环境变量，新主变量必须变成**离线链没有出现过的时间规律、跨资源关系和持续结构**。

### 1.3 R1 的在线修正主要改变 detection operating point，而非资源专长

dev500：

```text
A:                F1≈0.5118, AP≈0.6779, FP=1159, FN=142
C-residual-off:   F1≈0.5026, AP≈0.6782, FP=1223, FN=137
C-residual-on:    F1≈0.5091, AP≈0.6769, FP=1176, FN=141
```

AP 几乎不变，主要变化是 FP/FN operating point。更重要的是，在约 824 个 anomaly host-step 上，off 只改变 1 个资源类别判决，on 为 0。这说明当前 residual learner 尚未展示明显的新资源知识学习。

### 1.4 loss/sampling 存在明显但未完全因果定位的失衡

dev500 原异常率约 2.575%，近期抽样约 3.473%，同时 detection positive weight≈8.008742。历史实验中将该权重改为 1 可明显降低 FP、提高某些固定阈值 F1，但 recall/AP 同时下降。因此不能简单“去掉权重”，而应先直接测量 detection/classification/ranking/anchor/distill 对各参数组的梯度大小与冲突，再决定消融。

### 1.5 当前 Dynamic D 尚不具备正式主实验资格

R0-B 已证实旧动态 Top-k ramp 在成熟边界不连续：

```text
ramp -> 1 前后，参与专家可从 5 变 4
max detection-logit jump ≈3.47~3.48
```

差值在 epsilon→0 时不消失。因此 Protocol 021 禁止直接复用“把 newborn 插入重新归一化 Top-k expert pool”的部署方式。

---

## 2. Protocol 021 的五个假设

### H1 — Unseen Validity
新 regime 相对于实际在线起点的**完整学习链**具有可证明的未覆盖性。主 claim 至少要求“生成机制未覆盖 + 实际统计结构新颖”。

### H2 — Causal Learnability
当前/过去 12-step 内包含足以预测未来 fault 的线索。简单 causal probe 必须先证明这不是随机、不可观察的变化。

### H3 — Fixed Online Adaptation
固定拓扑在线 C 在观察一定数量成熟新事件后，能改善**后续未训练预测**，且熟悉模式退化受控。这是进入 D 前的硬前提。

### H4 — Dynamic Capacity Benefit
只有当 C 能学习但出现持续错误平台/容量不足时，D 才增加专家。D-C 必须在同信息、同基础 learner、同监督预算原则、同保护规则下比较。

### H5 — Recurring Knowledge Reuse
只有 H4 已成立后，再测试 retire/reactivate。第一版动态主实验只做 add，不把生命周期全部捆绑。

---

## 3. 分支、目录与冻结资产

执行第一步：

```bash
git checkout protocol-020
git status
git log -1
```

记录 HEAD，然后创建：

```bash
git checkout -b protocol-021
```

如果分支已存在，不得 force-reset；先记录状态。

禁止覆盖：

```text
protocol-019
protocol-020
artifacts/ftmoe_online/protocol_019/
artifacts/ftmoe_online/protocol_020/
原始 protocol014 v4 checkpoints
Protocol019 adapted checkpoints
Protocol020 S6 checkpoints
```

新产物统一放：

```text
artifacts/ftmoe_online/protocol_021/
```

至少建立：

```text
protocol.json
source_sha256_initial.json
problem_log.jsonl
docs/FTMOE_ONLINE_PROTOCOL_021.md
指令/FTMOE_PROTOCOL021_EXECUTION_PLAN.md
指令/FTMOE_PROTOCOL021_PROBLEM_LOG.md
```

---

## 4. 将“Unseen”分级，禁止一个布尔值糊弄

| 等级 | 定义 | 是否足够支持主 claim |
|---|---|---|
| U0 | 新 seed / 新任务实例 / 新 replay | 否 |
| U1 | VM/source cohort 不重叠 | 否，只证明来源隔离 |
| U2 | 静态参数组合离线没用过 | 不充分 |
| U3 | **生成机制/时间因果规律离线未使用** | 主 claim 必须 |
| U4 | **实际数据统计结构显著超出离线覆盖** | 主 claim 必须配合 U3 |

主实验最低要求：

```text
U3 + U4
```

如果能满足 U1 更好。若相对于原始 v4/Protocol019 的源文件历史无法完全恢复，则必须写：

```text
source-unseen relative to full historical chain: UNVERIFIED
```

不能宣称“绝对未见”。

---

## 5. P21-S1：完整 Exposure Ledger

创建：

```text
artifacts/ftmoe_online/protocol_021/offline_coverage_audit/
    exposure_ledger.json
    exposure_ledger.md
    exclusion_registry.json
```

必须审计整个在线起点学习链：

```text
Protocol014:
  gradient train
  dev/checkpoint selection
  normalization
  physical mapping/prototype source

Protocol019:
  adaptation train/dev
  normalization v2
  anchor
  teacher/distillation source

Protocol020:
  capacity scan
  scenario selection
  disk-law fitting
  S5 drift dev
  S6 adaptation train/dev
  normalization
  anchor
  R1 dev500/dev501
  保护阈值/方法选择使用过的流
```

每个数据源区分 exposure type：

```json
{
  "source": "...",
  "exposure_types": [
    "gradient_training",
    "checkpoint_selection",
    "normalization_fit",
    "generator_fit",
    "scenario_selection",
    "anchor_regularization",
    "teacher_distillation",
    "development_evaluation"
  ]
}
```

并生成禁止复用集合：

```text
used VM IDs
used time ranges
used capacity profiles
used adapters
used disk laws
used phase-transition rules
used temporal-coupling rules
used development seeds
```

没有进入梯度，不代表没有影响实验设计。

---

## 6. P21-S2：第一主 U3 机制固定为 Temporal Resource Cascade

第一轮不要同时造多种机制。

主新 regime：

```text
CPU burst
   ↓ lag
RAM retention/ramp
   ↓ lag
Disk accumulation
```

即：

\[
CPU(t) \rightarrow RAM(t+L_{CR}) \rightarrow Disk(t+L_{CD})
\]

初始登记：

```text
CPU burst duration: 3–5 intervals
CPU->RAM lag:       4
RAM elevated:       6–10 intervals
CPU->Disk lag:      8
Disk elevated:      6–10 intervals
cascade_task_probability candidates: 0.15 / 0.25 / 0.35
history window: 12
```

lag 选择 4/8 的原因是它们在 12-step 输入历史内可观察，满足“陌生但可学习”。

### 6.1 新 generator

新增：

```text
simulator/workload/BitbrainWorkloadProtocol021.py
```

不要改 Protocol020 generator。

每个新任务：

1. 从合法 cohort 采原始 Bitbrain trace；
2. 使用冻结基础 adapter；
3. 以注册概率决定是否 cascade；
4. cascade task 创建 lifecycle envelope；
5. 随机性只由 `replay_seed + creation_id + mechanism_seed` 决定；
6. 同 seed 完全复现。

不要对所有任务整段乘同一个系数。

### 6.2 Envelope

推荐：

```python
cpu_start
cpu_duration ∈ [3,5]
ram_start  = cpu_start + 4
ram_duration ∈ [6,10]
disk_start = cpu_start + 8
disk_duration ∈ [6,10]
```

RAM 使用渐进 ramp，而非瞬时跳变；Disk 在已有 Markov occupancy 上增加**有上限且会清理**的 retained-data term，不能无限单调增长。

### 6.3 只作审计的 metadata

允许保存：

```text
regime_id
cascade_task_flag
cascade_event_id
phase_name
```

但这些字段严禁作为模型输入。

---

## 7. P21-S3：Data-only Pilot，禁止加载 FT-MoE

当前 unseen plan 直接安排 7000 step 太重。先做 3 个候选：

```text
cascade probability = 0.15
0.25
0.35
```

每个：

```text
1200 scored intervals + guard
```

只跑一个开发 cohort/seed，不加载 A/B/C/D。

保存：

```text
stream.npz
manifest.json
events.json
unseen_data_audit.json
```

`stream.npz` 至少：

```text
host_features
demands
schedules
capacities
overload_ratio
overload_mask
raw_labels
before_placement
after_placement
creation_ids
after_creation_ids
cascade_event_ids
cascade_task_flags
simulator_intervals
```

其中事件/flag 仅审计，禁止模型读取。

### 7.1 Pilot Data Gate

1200×16=19200 host-step，建议：

```text
overall anomaly prevalence: 1%–15%
cascade-related positive host-step: >=150
independent cascade fault events: >=30
normal host-step: >=5000
deployment rejection <=25%
migration rejection <=40%
任何单一事件占全部 positive <20%
```

候选只依据数据物理指标和事件结构选择，绝对不能看 A/C/D 分数。

如果三个候选都失败，停止并登记第二轮机制参数，不能偷偷扩大搜索。


---

## 8. P21-S4：定量证明 U4，而不是只写“生成器不同”

新增：

```text
analyze_ftmoe_protocol021_unseen.py
```

至少比较：

```text
Protocol021 candidate
vs Protocol014 train/dev（可恢复部分）
vs Protocol019 adaptation train/dev
vs Protocol020 S6 train/dev
```

### 8.1 资源统计

对 CPU/RAM/Disk `demand/capacity`：

```text
p10/p50/p90/p95/p99
zero ratio
ratio>1 tail
```

### 8.2 时间结构

对 host aggregate：

```text
ACF lag 1..11
```

并计算跨资源 lag correlation：

\[
Corr(CPU_t,RAM_{t+k}),
Corr(CPU_t,Disk_{t+k}),
Corr(RAM_t,Disk_{t+k}),\quad k=0..11
\]

Temporal Cascade 的主证据必须主要体现在这种时序结构，而不是只体现在均值更大。

### 8.3 事件结构

统计：

```text
independent event count
duration p50/p90/p95
inter-event interval
burstiness
CPU->RAM transition probability
RAM->Disk transition probability
event transition matrix
```

### 8.4 exact overlap

检查：

```text
scoring-window exact hash overlap with P20 S6 adaptation == 0
```

如果使用 P20 `online` cohort或新的 dev-unseen 子集，检查与 P20 S6 train/dev VM：

```text
intersection == empty
```

### 8.5 U4 Gate

预先固定：

1. `CPU_t -> RAM_{t+4}` 或 `CPU_t -> Disk_{t+8}` 的 lag peak 至少一项超出所有可审计 offline corpus 对应统计的 historical 95% range；
2. cascade transition probability 相比 offline maximum：
   ```text
   absolute difference >=0.15
   ```
   或
   ```text
   >=1.5× offline maximum
   ```
3. exact scoring-window overlap=0；
4. P20 S6 source overlap=0；
5. Data Gate 已 PASS。

输出必须分别写：

```text
U1 source-disjoint: PASS/FAIL/UNVERIFIED
U2 parameter-combo unseen: PASS/FAIL
U3 generator-mechanism unseen: PASS/FAIL
U4 distributional novelty: PASS/FAIL
```

禁止只输出一个 `unseen=true`。

---

## 9. P21-S5：Causal Learnability Gate

这是现有 `UNSEEN_REGIME_PLAN` 最需要补强的一步：

> 未见不等于可学习。

在 FT-MoE 前运行一个简单 causal probe：

```text
probe_ftmoe_protocol021_learnability.py
```

### 9.1 输入

只用当前/过去可部署特征：

```text
current CPU/RAM/Disk demand-capacity ratio
1-step delta
3-step slope
6-step slope
host occupancy
proposed migration indicator
past observed fault state（只能已成熟）
```

禁止：

```text
phase ID
cascade flag
event ID
future demand
future capacity
未成熟 label
```

### 9.2 任务

与 FT-MoE 一致：

```text
t+1 detection
t+1 CPU/RAM/Disk dominant class
```

### 9.3 时间划分

例如 1200-step pilot：

```text
0–399   fit/calibration
400–799 dev
800–1199 held-forward test
```

禁止 random shuffle across time。

### 9.4 Learnability Gate

不要再次使用固定 `AP>=0.60`。

建议：

```text
AP >= max(0.20, 3 × positive_prevalence)
```

同时至少满足：

```text
ROC-AUC >=0.75
```

或：

```text
top-decile recall >=0.40
```

如果 Probe 失败，先检查历史长度/可观察特征，再修场景；禁止直接加 expert 或提高模型容量。

---

## 10. P21-S6：开发流先用 3400 步

只有：

```text
Data Gate PASS
U3 PASS
U4 PASS
Learnability PASS
```

才能构建模型开发流。

第一轮：

```text
Familiar           500
Unseen first      1200
Familiar return    500
Unseen recurrence 1200
----------------------
Total             3400
```

不要一开始直接 7000。

若 3400-step 上固定 C 已有可信正信号，再构建完整：

```text
Familiar          1000
Unseen first      2000
Familiar return   1000
Unseen recurrence 2000
Familiar final    1000
----------------------
Total             7000
```

---

## 11. 按“已成熟独立事件数”分析，而不仅按时间分析

所有 unseen 结果必须同时按：

```text
time since onset
```

和：

```text
number of matured independent events observed
```

报告。

建议 event bins：

```text
0–4
5–14
15–29
>=30
```

这样才能回答：

> 模型在观察多少次新错误反馈后开始真正改善？

持续 20 step 的同一个 host fault 必须按一个事件处理，不能当 20 个独立反馈。

---

## 12. P21-S7：第一轮只跑 A 与固定 C

统一在线起点：

```text
artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt
```

**绝对禁止用 Protocol021 unseen 数据做新的 offline adaptation。**

第一轮方法：

```text
A
C-residual-off
C-residual-on
```

“residual no-update”只需做一次：

```text
prediction == A
```

等价测试，不必每条 replay 重复。

### 12.1 主指标

Detection primary：

```text
late-unseen AP
prequential detection log loss
```

Secondary：

```text
F1@0.5
precision
recall
FPR
Brier
ECE
```

Diagnosis：

```text
target resource recall/F1
CPU/RAM/Disk macro-F1
end-to-end resource macro-F1
```

### 12.2 late-unseen 定义

优先按事件：

```text
从 >=15 个 matured independent unseen events 之后开始
```

若事件密度不足，再使用固定时间窗口，但必须在运行前登记。

### 12.3 单流继续 Gate

满足：

```text
late-unseen AP(C)-AP(A) >= +0.03
```

或：

```text
prequential supervised loss(C) <=0.95 × A
```

至少一项。

同时：

```text
familiar-return FPR increase <=0.005 absolute
familiar-return AP drop <=0.02
target-resource recall(C)-recall(A) >=+0.05
```

如果只提高固定阈值 F1、AP 不变/下降，不算充分学习证据。

---

## 13. 扩展到 3 个开发 replay 后才能调 D

单流出现正信号后，固定场景与当前 C，扩展：

```text
3 development replay seeds
```

要求：

```text
至少 2/3 replay 的 late-unseen improvement sign 为正
```

如果 1/3 正、2/3 负：

```text
停止 D
进入固定 online learner 诊断
```

---

## 14. P21-S8：Gradient / Task Diagnostic 优先于大网格调参

R1 已经观察到资源判决几乎不改变，因此先直接测各任务梯度。

预登记 update：

```text
unseen onset 后第 5/10/20/40/80 次 online update
```

记录：

```text
||grad_detection||
||grad_classification||
||grad_ranking||
||grad_anchor||
||grad_distill||
```

并计算：

```text
cos(det, cls)
cos(det, ranking)
cos(det, anchor)
cos(cls, anchor)
```

按参数组分别统计：

```text
residual router
expert first layer
expert output layer
detection-correction rows
classification-correction rows
```

单独梯度诊断必须：

```text
不 optimizer.step
不改变 RNG
不改变 model/buffer
```

---

## 15. Gradient Diagnostic 决策树

### Case A：Detection 学会，classification 不学

若：

```text
Detection AP improves
classification decision 거의不变
classification gradient 极小/稀疏
```

则下一版拆成：

```text
DetectionResidualBank
ClassificationResidualBank
```

不要继续共用一个 5-output correction bank。

### Case B：F1 变但 AP 不变/下降

优先定位 calibration：

```text
sampling × detection positive weight
```

### Case C：Probe 能学，但 C 的 AP/F1/分类都不学

说明 frozen z 可能没有把新规律表达成 residual 易用特征，进入 Feature Sufficiency Test。

### Case D：Probe 也学不会

回到场景/可观察性；不要通过增加模型参数掩盖输入不可预测。

---

## 16. R2 loss/sampling 消融只在诊断支持时做

如果属于 Case B，固定：

```text
architecture
features
LR
memory
stream
```

只做：

```text
Sampling:
  uniform
  event-stratified

Detection positive weight:
  1
  current registered weight
```

2×2。

第一轮不要同时加 focal loss、阈值优化、额外 memory。

### 16.1 R2 选择标准

按以下联合证据选择：

1. late-unseen AP；
2. prequential loss；
3. target resource recall/macro-F1；
4. familiar-return FPR；
5. familiar-return AP；
6. retention probe 仅辅助。

若：

```text
F1 +0.05
AP -0.08
Recall -0.15
```

不得称为优胜。

---

## 17. P21-S9：Feature Sufficiency Test

当前 R1 residual 只读 frozen final `z(64d)`。虽然 full base model看过容量/图信息，但不保证 z 对新 temporal mechanism 足够可利用。

先做两个 probe：

### Probe-Z

```text
frozen z
```

### Probe-Physics

```text
z
+ current actual demand/cap ratio (3)
+ proposed-placement demand/cap ratio (3)
+ ratio delta 1-step (3)
+ ratio slope 3-step (3)
+ ratio slope 6-step (3)
+ capacity delta (3)
+ occupancy
+ migration indicator
```

若：

```text
AP(Physics)-AP(Z) >=0.05
```

或 target resource recall：

```text
>=+0.10
```

则固定加入一个小的 shared physical projection：

```text
phi(physics) ->16/32d
concat(z, phi)
```

未来 C 和 D 必须获得完全相同信息。

禁止给 D 独享 capacity/physics 特征。

---

## 18. Long-term memory 后置

只有固定 C 已经能学习 unseen pattern，但 recurrence 时明显遗忘，才增加长期 memory。

推荐：

```text
event-level long-term memory =256 intervals
```

按：

```text
independent fault event
resource type
hard normal
```

分层，避免一个持续事件占满缓存。

更新预算保持 32：

```text
recent 16
long-term 8
anchor 8
```

与原 `recent24+anchor8` 保持总预算一致。

---

## 19. Protocol021 Dynamic D：改为 Additive Dynamic Residual

正式 D 不再把 newborn 插入重新归一化的 Top-k expert pool。

固定 C：

\[
y_C=y_A+\Delta_{fixed}
\]

新 D：

\[
y_D=y_A+\Delta_{fixed}+\sum_j \alpha_j g_j(x)\Delta_j(x)
\]

其中：

```text
alpha_j∈[0,1]
g_j(x)=independent sigmoid gate
```

dynamic expert 是**加法 residual**，不会在出生时抢走固定专家的 softmax mass。

### 19.1 为什么能解决 R0-B

birth：

```text
alpha=0
```

严格有：

```text
D == C
```

随后：

```text
0 -> .25 -> .5 -> .75 ->1
```

只改变新增 residual 幅度，不发生“5 experts -> top4 -> 某成熟专家突然消失”。

### 19.2 第一版只允许 ADD

命名：

```text
D-additive-v1
```

第一版仅实现：

```text
trigger
candidate evidence buffer
shadow train
prospective qualification
activate
continuous ramp
```

暂时不做 retire/reactivate。

先证明“新增 expert 真有未来预测贡献”，再做 lifecycle。


---

## 20. Dynamic Trigger：固定 novelty monitor + mature error plateau

Protocol020 历史使用 trainable router entropy 作为重要触发信号，但 router/key 自身持续更新，且 entropy p95 曾接近 0.996，区分性不足。

Protocol021 采用：

```text
Frozen novelty monitor
+
Mature prediction-error plateau
```

双触发。

### 20.1 Frozen Monitor

monitor 来源：

```text
online-start frozen base
```

使用 `frozen z`；若 Feature Sufficiency Test 已证明 physical features 必要，则使用 `frozen z + frozen physical projection`。进入动态实验后 monitor 参数禁止更新。

### 20.2 Novelty Score

第一版推荐使用固定 familiar calibration centroids 的 cosine/Mahalanobis distance。threshold 仅由 familiar development prefix 的 distance p95 产生，不得在 confirmation 上重新校准。

### 20.3 Error Plateau Trigger

只有 novelty 高不代表需要扩容。推荐首版：

```text
observed independent novel events >=20
novelty-ratio >=0.20，连续3 topology windows
mature supervised loss EMA >=1.25 × familiar/reference loss，连续3 windows
最近3 windows loss improvement slope >= -0.02 relative/window
expert_count < registered max
```

含义是“输入陌生 + 固定 C 持续犯错 + 常规 C 更新改善趋于平台”才创建 candidate。任何修改必须生成新 config version，旧结果保留。

---

## 21. Candidate Evidence Buffer 必须绑定触发错误

不能再只存 centroid/router vector。每条记录至少包含：

```json
{
  "window_id": 123,
  "event_id": 9,
  "input_reference": "...",
  "mature_label": "...",
  "base_prediction_version": 0,
  "c_prediction_version": 18,
  "error_type": "FN|FP|resource_confusion",
  "novelty_score": 1.23,
  "created_at": 456
}
```

Candidate training 主要来自：

```text
50% trigger-causing hard events
25% matched hard normals
25% anchor/retention
```

以 interval 为单位。禁止用“最新10个窗口”替代真正触发扩容的错误证据。

---

## 22. Shadow Training

candidate 创建后：

```text
alpha=0
不参与 live prediction
```

shadow 可训练 3–5 个注册 update blocks；candidate optimizer/state 独立保存。训练结束后冻结 candidate，进入 prospective qualification。

---

## 23. Qualification 必须是真正 prospective

Qualification block 中 candidate 参数固定。每个时间点在 label 出现前同时缓存：

```text
Frozen A prediction
C live prediction
C + candidate prediction
```

标签成熟后才结算。禁止成熟后用新参数重算过去 candidate score。

最低资格支持同时要求：

```text
>=10 independent novel events
>=50 positive host-step
>=500 normal host-step
```

不足时状态为 `UNVERIFIED`，不是 PASS/FAIL。

推荐 Admission Gate：

```text
candidate supervised loss <=0.95 × C
AP(candidate) >= AP(C)
normal FPR increase <=0.005 absolute
target resource recall >= C
familiar retention AP drop <=0.02
```

必须有主要 loss improvement 且无明显 detection/retention 副作用才允许启用。

---

## 24. Continuous Activation

candidate 通过后：

```text
alpha=0 -> 0.25 -> 0.50 -> 0.75 -> 1.00
```

测试点必须覆盖：

```text
0
1e-7
0.25
0.50
0.75
1-1e-7
1
```

特别要求 `1-epsilon -> 1` 不存在结构跳变。

---

## 25. D-off 必须逐位等于 C

dynamic trigger 关闭或尚无 candidate 时：

```text
D prediction == C prediction
D common learner params == C
D common optimizer state == C
recent/anchor/long-term sample sequence == C
```

需要逐步/hash 对拍，而不是只比较最终指标。

---

## 26. Dynamic 公平对照

第一主比较：

```text
C-residual
D-additive-v1
```

只有 D 有可信正收益，再补：

- **C-wide**：从开始就配置与 D 最大容量相同的固定专家，排除“只是参数更多”；
- **C-budget**：固定 C 获得与 D shadow/candidate 相当的训练计算预算，排除“只是多训练”。

若 C-budget 需要按 D 实际 candidate 次数事后匹配，只能标为成本解释性分析。主 topology effect 始终是 `D-C`，不是 `D-B`。

---

## 27. 生命周期 D-v2 后置

只有 D-additive-v1 在真实 unseen event 上通过 prospective qualification，并在 future predictions 上有重复收益，才实现 retire/dormant/reactivate。

Retirement 不要只用低 activation，建议：

```text
age >=500
AND regime absence >= registered duration
AND rolling counterfactual contribution≈0
```

Counterfactual `with expert / without expert` 必须在 label 前缓存。

Reactivate 流程：

```text
similarity 只检索 dormant candidate
-> prospective with/without predictions
-> 新成熟 evidence
-> requalification
-> PASS 后 same-ID reactivate
-> alpha=0 开始 ramp
```

相似度本身不能授权部署。

---

## 28. 第二个 unseen regime

Temporal Cascade 完成后，至少增加一个不同机制的 U3 regime。推荐 **Gradual Effective-Capacity Ramp**：

```text
RAM effective capacity:
1.00 -> .90 -> .80 -> .70 -> .60 -> .55
```

每 2–4 interval 缓慢变化，然后恢复。模型可看到 current capacity 与 past capacity delta。主 novelty 是 capacity trajectory，而不是静态 0.55 endpoint。

不要第一次就把 cascade、capacity ramp、arrival burst 全混在一起。

---

## 29. Final Confirmation 数据必须预先锁定

Protocol020 `online` cohort 约 94 VM。在任何 final model result 前，把仍未用于开发的 online cohort deterministic stratified split 为：

```text
online_confirm_0
online_confirm_1
online_confirm_2
```

三组 VM 不重叠。生成 `confirmation_registry.json`，写入 VM IDs、mechanism config、event seed、stream seed、method version、thresholds、hashes，并冻结 SHA。

若某 confirm stream 被用于修方法/调参，它自动降级为 development。

---

## 30. 关于 model seed 的现实边界

当前 P20 S6 主要正式起点是 model seed1。因此第一阶段可：

```text
fixed base model seed1
× 3 source-disjoint confirm streams
× registered online RNG
```

但必须写明：

```text
base-model-seed generalization not established
```

若后续用户明确允许多 base seeds，可从已有 Protocol014 v4 seed2/6/... 出发，严格复用被冻结的 P20 familiar-only S6 recipe 生成 P21 起点，但不得接触 unseen 数据、不得覆盖旧 checkpoint、不得挑最好 seed。

---

## 31. Statistical Unit

不能把 `16 hosts × 7000 intervals` 当成独立重复。实验单元：

```text
base-model seed × source stream × online RNG
```

base seed 固定时，主要独立来源是 source stream；同 stream 多 RNG 只是算法随机性重复。

报告 mean/std/median、paired differences、bootstrap CI 和 per-stream results。

---

## 32. 最终指标

### Detection
```text
AP/PR-AUC                  PRIMARY
prequential log loss       PRIMARY
F1@0.5
precision/recall/FPR
ROC-AUC auxiliary
Brier/ECE
```

### Diagnosis
```text
CPU/RAM/Disk recall/F1
macro-F1
balanced accuracy
confusion matrix
support
```

必须区分 conditional classification 与 end-to-end diagnosis。

### Continual/Online
```text
cold-start cost
events-to-improvement
time-to-improvement
late-unseen AP
familiar-return degradation
recurrence retention
area under prequential AP
cumulative excess loss vs A
```

### Dynamic
```text
candidate creations/rejects/accepts
shadow updates
active experts
peak params
candidate counterfactual contribution
extra FLOPs
prediction/update latency
memory
```

---

## 33. 推荐新的 Adaptation Lag

同时报告：

### Event-based lag
从 unseen onset 到 `AP gain over A >=0.03` 且持续两个 evaluation blocks 所需 matured independent events 数。

### Cumulative Excess Loss
\[
CEL_M(t)=\sum_{\tau=t_0}^{t}[L_M(\tau)-L_A(\tau)]
\]

真正学习时早期 CEL 可以上升，但后期 slope 应转负。这比单点 F1 recovery 更稳健。

---

## 34. 必须新增的测试

| Test | 验收 |
|---|---|
| 01 Protocol020 regression | 新功能默认关闭时 020 输入/预测不变 |
| 02 Cascade deterministic | 同 replay+creationID+mechanism seed 完全一致 |
| 03 Lag correctness | CPU t、RAM t+4、Disk t+8 对拍 |
| 04 No future leakage | 改 future envelope/capacity/label 不影响过去 |
| 05 Metadata exclusion | phase/regime/cascade/event ID 不在模型输入 |
| 06 Source split | 可核查范围 train/dev vs unseen source 不相交 |
| 07 Window overlap | P21 scoring vs P20 S6 exact window hash overlap=0 |
| 08 Event segmentation | 连续同一 fault 只算一个独立事件 |
| 09 Probe causality | 只能 forward split |
| 10 Frozen A immutable | base parameter/buffer hash 恒定 |
| 11 Prequential order | predict→mature→settle→train |
| 12 finish idempotent | 最后一条成熟 label 恰好一次 |
| 13 Gradient diagnostic | 不改变参数/optimizer/RNG |
| 14 D disabled==C | 预测、公共训练、抽样一致 |
| 15 Additive birth alpha0 | birth 前后 logits diff <1e-7 |
| 16 Full ramp continuity | 1-epsilon→1 无 hard jump |
| 17 Candidate provenance | candidate 数据绑定 trigger evidence |
| 18 Prospective qualification | candidate frozen，预测先于 label |
| 19 Candidate reject | live C 完全不变 |
| 20 Candidate accept | 第一步 alpha=0 无 jump |
| 21 Resume exact | candidate/alpha/optimizer/ledger/RNG 全恢复 |
| 22 C/D fairness | birth 前采样完全配对 |
| 23 Counterfactual logging | with/without candidate 均在 label 前 |
| 24 Final cohort lock | confirm registry 后禁止搜索式读取 confirm labels |

---

## 35. 建议新增文件

```text
docs/FTMOE_ONLINE_PROTOCOL_021.md
指令/FTMOE_PROTOCOL021_EXECUTION_PLAN.md
指令/FTMOE_PROTOCOL021_PROBLEM_LOG.md

audit_ftmoe_protocol021_offline_coverage.py
register_ftmoe_protocol021_unseen.py
analyze_ftmoe_protocol021_unseen.py
probe_ftmoe_protocol021_learnability.py

simulator/workload/BitbrainWorkloadProtocol021.py
prepare_ftmoe_protocol021_unseen.py

recovery/PreGANSrc/src/ftmoe_online_p21.py
recovery/PreGANSrc/src/ftmoe_dynamic_residual_v1.py
run_ftmoe_protocol021.py
analyze_ftmoe_protocol021.py

test_ftmoe_protocol021_audit.py
test_ftmoe_protocol021_unseen.py
test_ftmoe_protocol021_online.py
test_ftmoe_protocol021_dynamic.py
test_ftmoe_protocol021_resume.py
```

---

## 36. Unseen Registry 建议 schema

```json
{
  "protocol": "021",
  "regime_id": "cascade_v1",
  "claim_level": {
    "U1_source_disjoint": true,
    "U2_parameter_combo_unseen": true,
    "U3_generator_mechanism_unseen": true,
    "U4_distributional_novelty": null
  },
  "mechanism": {
    "type": "temporal_resource_cascade",
    "cpu_duration": [3, 5],
    "cpu_to_ram_lag": 4,
    "cpu_to_disk_lag": 8,
    "ram_duration": [6, 10],
    "disk_duration": [6, 10],
    "cascade_task_probability": 0.25
  },
  "observable_history": 12,
  "forbidden_inputs": [
    "regime_id",
    "phase_id",
    "cascade_task_flag",
    "future_demand",
    "future_capacity",
    "unmatured_label"
  ],
  "development_only": true,
  "generator_sha256": "...",
  "registry_sha256": "..."
}
```

---

## 37. Offline Coverage Audit 建议 schema

```json
{
  "start_checkpoint": "...",
  "learning_chain": [
    {"stage": "protocol014", "gradient_data": [], "selection_data": [], "normalization_data": []},
    {"stage": "protocol019", "gradient_data": [], "anchor_data": []},
    {"stage": "protocol020_s6", "gradient_data": [], "selection_data": []}
  ],
  "mechanism_exclusion": {
    "temporal_cpu_ram_disk_cascade": "not_found"
  },
  "source_scope": {
    "relative_to_p20_s6": "source-disjoint",
    "relative_to_full_014_019": "unverified"
  }
}
```

---

## 38. 其他模型直接执行的 64 步 Checklist

```text
01. Checkout protocol-020，登记 HEAD/status/hash。
02. 新建 protocol-021，不覆盖020。
03. 建 protocol.json/problem log/source hash。
04. 枚举 online-start 完整 learning chain。
05. 建 exposure_ledger.json/md。
06. 区分 gradient/selection/normalization/generator-fit/anchor/dev exposure。
07. 建 exclusion_registry.json。
08. 核查未来 unseen cohort 与 P20 S6 train/dev VM overlap。
09. 实现 Temporal Cascade generator，不运行模型。
10. 写 deterministic/lag/no-future tests。
11. 注册 cascade-v1。
12. Data-only 跑 probability=.15/.25/.35。
13. 只依据事件/rejection/physics选候选。
14. 输出 unseen_data_audit。
15. 审计 U3 generator exclusion。
16. 审计 U4 cross-lag/event statistics。
17. U3/U4 不过则 STOP。
18. 跑 causal simple probe。
19. Learnability 不过则 STOP 并修场景/输入。
20. 构建3400-step dev replay。
21. 冻结 stream SHA。
22. 跑 Frozen A。
23. 跑 C-residual-off。
24. 跑 C-residual-on。
25. 按事件数+时间分析。
26. 检查 late-unseen AP/loss/target recall/familiar-return。
27. C 无正学习信号则禁止 D。
28. 同一机制扩展到3 dev replay。
29. 至少2/3 improvement sign一致。
30. 执行 gradient/task-conflict audit。
31. 按决策树决定 sampling/weight 或 split-head。
32. 做 Probe-Z vs Probe-Physics。
33. 必要时为 C/D 同时加入相同 physical features。
34. 固定最终 C。
35. recurrence不足时才加 event-level long-term memory。
36. 冻结 C architecture/loss/features/memory/protection。
37. 实现 additive dynamic residual D-v1。
38. 禁止接入旧 EAGate Top-k mature ramp。
39. Test alpha=0..1 全路径连续。
40. Test D-off == C。
41. 建 frozen novelty monitor。
42. 实现 error-plateau trigger。
43. candidate buffer绑定触发事件。
44. shadow train。
45. prospective frozen qualification。
46. evidence不足标 UNVERIFIED。
47. PASS 后 alpha=0 开始 activate。
48. 第一版只 ADD，不 retire/reactivate。
49. 跑 C vs D 3400-step dev。
50. D未触发不得解释“动态专家无效”。
51. D触发但不优于C，先分析贡献/干扰。
52. D有收益后补 C-wide。
53. D有收益后补 C-budget。
54. 动态v1有效后才做 lifecycle v2。
55. 注册第二 unseen mechanism：gradual capacity ramp。
56. 重复 U3/U4/learnability/fixed C/dynamic 流程。
57. 开发结束后冻结 generator/method/thresholds。
58. 在任何 final result 前拆分并锁定3个 confirmation source sets。
59. 写 confirmation_registry + SHA。
60. 第一条 confirm 仅工程验收；若因此改方法则降级开发。
61. 按冻结方案运行全部 confirmation。
62. 统计 paired D-C/C-A/B-A。
63. 保留所有失败、null、rejection、unverified。
64. 禁止依据 confirmation 结果重新选场景/超参。
```

---

## 39. STOP 条件

### STOP-A：Unseen validity 失败
U3 未证明或 U4 无实际差异，不得继续以 unseen 名义跑模型。

### STOP-B：Learnability 失败
simple causal probe 接近 prevalence/random baseline，先修可观察性。

### STOP-C：Fixed C 失败
3 dev replay 中至少 2 条 late-unseen 不优于 A，禁止正式 D。

### STOP-D：Dynamic continuity 失败
任何 alpha/topology 边界出现非预期 hard discontinuity，D 无资格进入效果矩阵。

### STOP-E：Qualification contamination
candidate 使用对应 qualification label 训练后才生成“历史预测”，该 run 作废。

### STOP-F：Confirmation contamination
confirm 数据用于选场景、loss、threshold、architecture，该 stream 自动降级 development。

---

## 40. 最终可支持的论文 Claim

### Claim A — Online adaptation to unseen runtime regime
需要：
```text
U3+U4 PASS
causal learnability PASS
C 对 post-feedback future predictions 优于 A
多个 replay 方向一致
```

### Claim B — Dynamic experts provide additional adaptation capacity
需要：
```text
D 真实创建并通过 prospective qualification
dynamic expert 实际参与 future predictions
D-C paired positive
C-wide/C-budget 不能完全解释收益
```

### Claim C — Recurring knowledge reuse
需要：
```text
真实 dormant
再现时 prospective requalification
same-ID reactivate
recurrence future prediction gain
```

单元测试通过不够。

---

## 41. 当前最值得立刻执行的内容

如果下一模型只能先做一项：

> **Exposure Ledger + Temporal Cascade data-only pilot + U3/U4/causal-learnability audit。**

当前暂时不要：

```text
继续调 EAGate threshold
调 expert max
跑5×3
重新用 unseen 数据离线适配
直接跑 D
```

最大的科学缺口不是“dynamic expert 参数没调好”，而是尚未先建立并证明一个相对于实际初始化链真正未覆盖、又具有可观察规律、可以从成熟反馈中学习的 runtime regime。

---

## 42. 下次通过 GitHub 反馈时必须提交的最小证据

至少上传：

```text
docs/FTMOE_ONLINE_PROTOCOL_021.md
指令/FTMOE_PROTOCOL021_PROBLEM_LOG.md
artifacts/ftmoe_online/protocol_021/protocol.json

artifacts/ftmoe_online/protocol_021/offline_coverage_audit/
    exposure_ledger.json
    exposure_ledger.md
    exclusion_registry.json

artifacts/ftmoe_online/protocol_021/unseen_registry/
    cascade_v1.json

artifacts/ftmoe_online/protocol_021/unseen_data_audit/
    candidate_*.json
    selected.json

artifacts/ftmoe_online/protocol_021/learnability/
    probe_result.json
```

若已进入模型 pilot，再提交：

```text
development_streams/.../manifest.json
runs/A.../summary.json
runs/C.../summary.json
analysis/fixed_online_pilot.json
```

若已实现 D，再提交：

```text
dynamic_config.json
dynamic_events.jsonl
candidate_qualification.jsonl
counterfactual_predictions.jsonl
C_D_paired_analysis.json
```

---

## 43. 执行纪律

```text
1. 不为让 D 赢而改场景。
2. 不因 A 太强而删除已登记 unseen 场景。
3. 不因某方法恢复慢而单独延长它的 phase。
4. 不把新 VM、新 seed 当新机制。
5. 不把 S6 训练过的 profile 再叫 unseen。
6. phase/cascade/event ID 不进入模型。
7. 不用 future labels/capacity/demand。
8. 不用更新后的模型重算过去 prediction 做资格验证。
9. 不把 candidate 单元测试成功当真实收益。
10. 不把专家数量增长本身当成功。
11. 不把 F1 上升等同于 AP/ranking 上升。
12. 不把 AP 下降但 F1 上升称全面改善。
13. 未证明 C 可学习前，不启动正式 D。
14. R0-B discontinuity 未解决前，不复用旧 Top-k 动态部署。
15. 所有失败、停止、metadata 不一致必须保留。
```

---

## 44. Protocol 021 总路线

```text
完整学习链审计
        ↓
U3 mechanism registry
        ↓
Data-only pilot
        ↓
U4 distribution audit
        ↓
Causal learnability probe
        ↓
Frozen A vs fixed C
        ↓
Gradient/loss/feature diagnosis
        ↓
稳定 fixed C
        ↓
Additive Dynamic D-v1
        ↓
Prospective candidate qualification
        ↓
C vs D
        ↓
C-wide/C-budget
        ↓
第二 unseen regime
        ↓
独立 confirmation
```

核心是把 Protocol020 中“工程越来越完整，但主科学问题没有真正隔离”的状态，转化为：

```text
先证明场景确实未见
再证明它可学习
再证明在线更新能学
最后才证明动态容量是否额外有用
```

这样后续无论得到正结果还是负结果，都有可解释、可复现和论文价值。
