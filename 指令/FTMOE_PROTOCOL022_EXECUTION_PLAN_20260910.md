# FT-MoE Protocol 022 实验执行计划
## Mechanism-Level Unseen Validation → Fixed Residual Adaptation → Additive Dynamic Experts

**日期：2026-09-10**  
**仓库：** `songwenhao074-maker/FT-MoE`  
**建议分支：** `protocol-022`  
**冻结父分支：** `protocol-021`  
**父分支固定 HEAD：** `f0abb5580d92730eecdc1db4326382c81a5f1e31`  
**统一在线起点 checkpoint：**  
`artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt`  
SHA256：`10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b`

> 本文档是给执行模型/工程 Agent 的强约束实验协议。  
> **不得为了得到“D 优于 C”的结果跳过 Gate、修改历史失败、事后筛选场景、事后改主指标或用未来标签重新计算过去预测。**
>
> Protocol 021 的 `STOP-A` 是有效实验结论，**不得修改为 PASS**。Protocol 022 是新的、独立登记的后续实验。

---

# 0. 本轮最终要回答的科学问题

Protocol 022 不直接问“动态专家是不是更好”，而分成四个递进问题：

1. **新 regime 是否真的在完整离线/开发链之外表现出新的联合时间结构？**
2. **这种新结构是否能从部署时可获得的信息中预测，而且收益不是由 fault persistence、当前过载状态或泄漏造成？**
3. **固定拓扑在线模型 C 是否能利用成熟反馈，改善之后尚未用于训练的预测？**
4. **只有当 C 已能学习但出现持续容量/专业化瓶颈时，新增动态专家 D 是否提供独立收益？**

因此顺序必须是：

```text
Data validity
    ↓
Mechanism-level U4
    ↓
Predictive learnability beyond persistence
    ↓
Fixed C capacity / upper-bound test
    ↓
Strict prequential C vs A
    ↓
[只有 C 可学但仍有残余平台]
    ↓
Additive Dynamic D continuity
    ↓
D vs C / C-wide / C-budget
```

---

# 1. 必须接受的历史事实

## 1.1 Protocol 020：固定在线修正没有稳定超过冻结 A

R1 两条开发流结果：

| replay | method | F1 | PR-AUC | anchor F1 | end-to-end resource F1 |
|---|---|---:|---:|---:|---:|
| 500 | A | 0.5118 | 0.6779 | 0.6120 | 0.4598 |
| 500 | C-residual-off | 0.5026 | 0.6782 | 0.5750 | 0.4547 |
| 500 | C-residual-on | 0.5091 | 0.6769 | 0.6120 | 0.4577 |
| 501 | A | 0.1053 | 0.1270 | 0.6120 | 0.0381 |
| 501 | C-residual-off | 0.1057 | 0.1284 | 0.6157 | 0.0383 |
| 501 | C-residual-on | 0.1053 | 0.1285 | 0.6170 | 0.0381 |

这说明：

- residual 结构基本成功实现“**不破坏 A**”；
- 但没有证明“**在线学习产生新知识**”；
- dev500/dev501 已参与方法开发，不能再当独立确认数据；
- C-legacy 在某些固定阈值 F1 上变化，但 PR-AUC/旧能力可能变差，不能把 operating-point 改变误判为表示学习。

## 1.2 Protocol 021：数据物理门禁基本可行，但 U4 预注册门禁失败

`cascade_v1`：

- `p=0.15`：Data Gate PASS；
- `p=0.25`：Data Gate PASS；
- `p=0.35`：migration rejection = 43.76%，超过 40%，FAIL。

`p=0.25` 的旧 U4：

```text
Corr(CPU_t, RAM_{t+4}) = 0.549
offline p97.5          = 0.582

Corr(CPU_t, Disk_{t+8}) = 0.443
offline p97.5           = 0.521

P(RAM↑@t+4 | CPU↑@t) = 0.428
offline max            = 0.764
```

所以 Protocol 021 的 **U4=FAIL，STOP-A 成立**。不得用后续分析反向改写。

## 1.3 Protocol 021 的 U4 仪器尺度有问题

旧审计主要在：

```text
[T, H, 7] host_features
```

上做 host-level correlation / onset response。

尤其当前 `analyze_ftmoe_protocol021_unseen.py::onset_response()` 的 docstring 写的是：

```text
task's own CPU onset
```

但实现实际先调用：

```python
arr = as_host_features(features)
```

随后按 `h in range(H)` 遍历 **host**。

因此当前所谓“任务自身 onset”的事后结果，严格来说仍是 host-level response，不能作为真正 task-level U4 证据。

## 1.4 Learnability PASS，但存在 persistence/confounding 风险

Protocol 021 `target=next`：

```text
test prevalence      = 0.07785
AP                   = 0.67804
ROC-AUC              = 0.95397
top-decile recall    = 0.73441
conditional class acc= 0.48893
```

信号明显高于 prevalence。

但：

```text
time-shuffled-label AP = 0.24189
```

仍显著高于 prevalence 0.07785。

同时 probe 输入含：

```text
current CPU/RAM/Disk ratio
past_fault_0..3
```

如果故障连续存在，则“当前已经异常 → 下一步仍异常”本身即可得到很强结果。

因此现在只能说：

> 当前数据对 `t+1 fault state` 有可预测性。

还不能严格证明：

> 模型能够学习 cascade 的新时间因果关系并提前预测新的 fault onset。

## 1.5 旧 Dynamic D 存在结构性不连续

R0-B 已证明：

```text
candidate ramp: 0.9999999 -> 1.0
active experts: 5 -> 4
max detection-logit jump ≈ 3.475
```

原因是 candidate 成熟后进入旧 Top-k/重归一化竞争，导致成员集合硬切换。

**在这个问题修复前，禁止执行正式 D effectiveness matrix。**

---

# 2. Protocol 022 的核心假设与 Gate

## H0 — Audit Correctness

审计代码必须真的在它声称的尺度工作。

**Gate H0：**

- task-level 指标必须使用 `creation_id / per-container demand / event_id`；
- host-level 与 task-level 结果分开命名；
- unit test 用人工两任务同机案例确认不会把 host aggregation 当成单任务；
- metadata 只用于审计，不进入模型输入。

FAIL → STOP-0。

---

## H1 — U3 + U4-v2 Unseen Validity

新 regime 的生成机制在实际学习链中未出现，并且其**联合时序结构**相对离线参考显著不同。

**U3：** 沿用 P21 Exposure Ledger，不重新解释历史。  
**U4-v2：** 改为 task/event-level + matched-control 统计。

FAIL → STOP-A2。  
不得跑 A/C/D 主模型。

---

## H2 — Predictive Learnability Beyond Persistence

部署时可见历史不仅能预测“故障持续”，还必须能预测：

```text
normal/non-target at t
        ↓
new fault onset at t+1...t+h
```

并优于 persistence / current-ratio / host-prior 等简单基线。

FAIL → STOP-B2。  
先修数据可观察性，不调 FT-MoE。

---

## H3 — Fixed Residual Online Adaptation

固定 C 在看到成熟新 regime 样本后，能改善**之后未训练样本**，同时熟悉能力退化受限。

FAIL → STOP-C2。  
不得增加动态专家。

---

## H4 — Dynamic Capacity Necessity

只有出现以下情况才允许进入 D：

```text
C 明确能在线学习
AND
C 在 mature unseen 段出现持续 residual error plateau
AND
增加普通训练步数不能解释该 plateau
AND
C-wide / representation upper bound 显示额外容量可能有价值
```

否则：

- C 已解决问题 → D 没必要；
- C 根本学不会 → D 缺少科学依据。

---

## H5 — Dynamic Expert Independent Benefit

D 相对同信息、同主 learner、同监督原则的 C：

- 在未来未训练预测上有稳定增益；
- 不主要来自更多参数；
- 不主要来自更多梯度更新；
- 不依赖未来标签或 phase/cascade ID；
- 没有显著破坏 familiar/anchor。

---

# 3. 分支与资产冻结

执行：

```bash
git checkout protocol-021
git status --porcelain
git rev-parse HEAD
```

必须得到：

```text
f0abb5580d92730eecdc1db4326382c81a5f1e31
```

若不是该 SHA：

1. 不 force reset；
2. 在 `protocol_022/bootstrap_state.json` 记录实际 HEAD；
3. 若新增提交只是后续反馈，可由执行模型重新审计差异；
4. 未审计前不得开始正式数据生成。

然后：

```bash
git checkout -b protocol-022
```

禁止覆盖：

```text
artifacts/ftmoe_online/protocol_019/**
artifacts/ftmoe_online/protocol_020/**
artifacts/ftmoe_online/protocol_021/**
原始 v4 checkpoint
P019 adapted checkpoint
P020 S6 adapted checkpoint
```

新产物：

```text
artifacts/ftmoe_online/protocol_022/
```

至少创建：

```text
protocol.json
source_sha256_initial.json
problem_log.jsonl
gate_status.json

audit_v2/
pilot_streams/
learnability_v2/
fixed_c/
dynamic_d/
confirmation/

docs/FTMOE_ONLINE_PROTOCOL_022.md
指令/FTMOE_PROTOCOL022_PROBLEM_LOG.md
```

任何失败都 append 到 `problem_log.jsonl`，不删除。

---

# 4. P22-S0：先修审计器，不生成正式数据

## 4.1 新增文件

建议复制并改名，而不是原地修改 P21：

```text
analyze_ftmoe_protocol022_unseen.py
probe_ftmoe_protocol022_learnability.py
register_ftmoe_protocol022.py
test_ftmoe_protocol022_audit.py
```

generator 也独立：

```text
simulator/workload/BitbrainWorkloadProtocol022.py
```

## 4.2 真正的 task-level 索引

必须能对每个时间点构造：

```text
(time,
 slot/container index,
 creation_id,
 vm_id,
 host_id,
 demand[7],
 cascade_event_id,
 cascade_flag,
 age)
```

其中：

- `creation_id` 是主要实体键；
- slot index 不能当永久任务 ID；
- host migration 后仍要沿 `creation_id` 跟踪同一任务；
- 一个 creation_id 被释放后不能错误拼接到复用 slot。

新增审计输出：

```text
task_timeline.parquet 或 task_timeline.npz
task_event_index.json
```

如果不希望引入 parquet 依赖，优先 NPZ + JSON。

## 4.3 强制单元测试

至少增加以下测试：

### T-AUDIT-01：同机混合隔离

构造：

```text
Task A: CPU burst only
Task B: RAM burst only
same host
```

host 聚合会表现出 CPU/RAM 共动，但 task-level A 不应被判成 CPU→RAM cascade。

### T-AUDIT-02：migration identity

任务 t0 在 host0，之后迁移 host1。  
task-level trajectory 必须连续。

### T-AUDIT-03：slot reuse

旧 container 删除，新 container 复用同 slot。  
两者 creation_id 不同，绝不能拼接。

### T-AUDIT-04：future metadata isolation

修改：

```text
future cascade_event_id
future phase metadata
```

不得影响任何模型可见 feature。

### T-AUDIT-05：P20 regression

Protocol022 probability=0 时，生成路径与冻结 familiar generator 在核心需求/标签口径上一致。

### T-AUDIT-06：determinism

相同 replay seed + creation_id + mechanism seed → byte-equivalent envelope/event registry。

全部通过才可进入 S1。

---

# 5. P22-S1：登记 cascade_v2，但禁止根据 P21 效果调参“追 Gate”

## 5.1 设计原则

为了避免 HARKing，**不要因为 P21 U4 失败就把 CPU/RAM/Disk 幅度继续放大直到通过**。

建议 cascade_v2 保留 P21 的核心物理 envelope：

```text
CPU duration      : 3–5
CPU->RAM lag      : 4
CPU->Disk lag     : 8
RAM duration      : 6–10
Disk duration     : 6–10
CPU floor/upper   : 4400 / 5200
RAM floor/upper   : 4500 / 6000
Disk retained peak/cap: 16000 / 24000
observable history: 12
```

变化主要是：

1. 审计尺度修正；
2. 增加 matched negative-control generator；
3. 使用全新的 mechanism seed；
4. 重新登记正式 Gate；
5. 明确 development 与 confirmation 数据隔离。

推荐：

```text
mechanism_seed_dev     = 22022
mechanism_seed_confirm = 22023
```

若 seed registry 已占用，可顺延，但必须在第一次正式生成前冻结。

## 5.2 probability

仍保留：

```text
0.15 / 0.25 / 0.35
```

理由不是重新搜索最优，而是保持与 P21 可比。

但预先注明：

- `0.35` 是 high-load stress candidate；
- 它如果再次超过 migration rejection 40%，保留 FAIL；
- 不降低门槛；
- 不因为 0.35 事件多就选它。

---

# 6. P22-S2：U4-v2 的新仪器

## 6.1 Primary unit：独立 task-event

不要再把 19200 host-step 当成独立样本做显著性。

统计单元：

```text
creation_id + CPU onset event
```

同一持续事件只算一次。

用 event bootstrap / permutation：

```text
bootstrap unit = event
not host-step
```

## 6.2 CPU onset 定义

仅使用任务自身 demand：

```text
ΔCPU_t = CPU_task(t) - median(CPU_task[t-4:t])
```

onset 至少满足：

```text
ΔCPU_t >= registered_onset_threshold
CPU_task(t-1) 未处于同一 burst
creation_id 连续存在
```

threshold 只允许由 generator 注册值和 offline-only calibration 决定。

## 6.3 RAM response

对每个 CPU onset `t0`：

```text
baseline_RAM = median RAM_task[t0-4:t0]

RAM_response =
max RAM_task[t0+4 : t0+14]
- baseline_RAM
```

同时报告：

```text
RAM_at_exact_lag4
RAM_window_max
time_to_RAM_peak
response_positive
```

## 6.4 Disk response

```text
baseline_Disk = median Disk_task[t0-4:t0]

Disk_response =
max Disk_task[t0+8 : t0+18]
- baseline_Disk
```

报告：

```text
Disk_at_exact_lag8
Disk_window_max
time_to_Disk_peak
response_positive
```

## 6.5 必须加入三个 matched controls

### Control M0：marginal-matched, lag-shuffled

保留每个资源的：

```text
marginal amplitude
duration distribution
event count
task cohort
arrival process
```

只随机化 CPU/RAM/Disk 的事件配对/lag。

目的：证明新颖性来自**联合时序结构**，而不是资源值变大。

### Control M1：order-shuffled

保留相同资源事件，但随机顺序：

```text
RAM -> CPU
Disk -> CPU
...
```

### Control M2：within-task circular shift

在每个 creation_id 内对 RAM/Disk trajectory 做 circular temporal shift，保持 marginal/autocorrelation 近似不变，破坏注册 causal alignment。

审计 metadata 可以使用 cascade flag/event id，但这些 control 绝不能进入模型输入。

---

# 7. P22-S3：U4-v2 Gate

先用所有可恢复离线语料建立 reference，然后锁定阈值，再读取 P22 candidate。

## 7.1 必须报告的指标

每个 candidate：

```text
n_task_onsets
n_valid_followup_events
RAM_response p50/p90/p95
Disk_response p50/p90/p95
share_RAM_positive
share_Disk_positive
lag4 alignment rate
lag8 alignment rate
transition-order fidelity
event-level bootstrap CI
distance vs offline
distance vs M0/M1/M2
exact scoring-window overlap
VM/source overlap
Data Gate
```

## 7.2 U4-v2 通过规则

为了避免“某一个 max 大就 PASS”，使用分布规则。

推荐主规则：

### RAM joint-temporal novelty

必须同时满足：

```text
A. candidate RAM_response p90 > offline RAM_response p97.5
B. bootstrap 95% CI of
   P(candidate RAM_response > offline_p97.5)
   lower bound >= 0.20
C. registered-lag alignment significantly > M0/M1/M2
   permutation p < 0.01
D. n_valid_followup_events >= 50
```

### Disk

Disk 作为次级证据，不要求单独把主 Gate 卡死，但完整报告：

```text
candidate vs offline
candidate vs M0/M1/M2
```

### 其他硬条件

仍要求：

```text
exact 12-step scoring-window overlap with P20 S6 = 0
VM overlap with P20 S6 train/dev = empty
Data Gate = PASS
```

对于 P014/P019 全历史 VM 来源无法恢复的问题：

```text
source-unseen relative to full historical chain = UNVERIFIED
```

继续如实报告。  
**主 unseen claim 只写 mechanism-level U3 + distributional U4-v2，不写 absolute source-unseen。**

## 7.3 禁止事项

不得：

- 用 P22 candidate 反过来确定 offline percentile；
- 看 A/C/D 的性能决定哪个 probability 通过 U4；
- 把 post-hoc 指标静默提升为 preregistered Gate；
- 修改 p-value / percentile 直到 candidate PASS。

全部 candidate FAIL → STOP-A2。

---

# 8. P22-S4：重新验证“可学习性”，重点排除 persistence

这是本轮非常重要的一步。

## 8.1 保留两个任务，但区分含义

### Task S（state prediction）

```text
y_state(t+1) = fault state at t+1
```

用于和旧实验比较。

### Task O（onset prediction，新的主任务）

仅在 t 时刻未处于目标故障时评估：

```text
y_onset(t,h)=1
if host is non-fault/other-fault at t
and target fault begins in t+1...t+h
```

至少报告：

```text
h = 1
h = 4
```

主 Gate 优先看 `h=1`，h=4 作早期预警补充。

## 8.2 必须加入的简单基线

### B0 random prevalence
### B1 host prior
### B2 persistence

```text
predict y(t+1) = y(t)
```

### B3 current-ratio only

仅：

```text
CPU/RAM/Disk current ratio
```

### B4 past-fault only
### B5 slopes only
### B6 full 18-feature logistic probe

关键结论必须来自：

```text
B6 > max(B2, B3, B4)
```

而不是只比 random。

## 8.3 permutation control 修正

旧实验只做一次 label shuffle。

Protocol022 必须：

```text
>=100 independent permutations
```

建议进行：

1. global fit-label permutation；
2. within-host permutation；
3. block permutation（block size 8/12）；
4. event-label permutation。

保存 null AP distribution：

```text
mean
std
p95
p99
observed percentile
```

Gate 不允许只写单个 `time_shuffled_label_ap`。

## 8.4 lag ablation

Full probe 分别移除：

```text
past_fault
current ratios
slope4-like temporal terms
migration
```

同时做错误 lag：

```text
RAM lag = 1 / 10
Disk lag = 2 / 11
```

如果“正确 registered temporal context”与错误 lag 完全无差异，则不能把可学习性归因于 cascade temporal law。

## 8.5 Learnability-v2 Gate

主 Gate 使用 onset task：

```text
AP_onset >= max(0.15, 2.5 × onset prevalence)
AND ROC-AUC >= 0.75
AND AP_full >= AP_best_simple + 0.05 absolute
AND observed AP > permutation null p99
```

同时要求至少：

```text
>=50 positive onset events in forward test
```

如果 onset 太稀，扩大**数据长度/seed 数**，不要放宽 Gate。

FAIL → STOP-B2。

---

# 9. P22-S5：Fixed C 先做“可训练上界”测试

不要马上跑完整 online adaptation。

目的：

> 区分“C 的结构/特征根本学不了”与“在线优化策略没学好”。

## 9.1 Frozen A

完全冻结现有起点。

## 9.2 C-oracle-lite（仅作 capacity diagnostic）

严格时间拆分：

```text
unseen block:
early mature segment -> train residual
later segment        -> evaluate
```

例如：

```text
0–399   collect / train after labels mature
400–799 future validation
800–1199 future test
```

注意：

- 不是最终 online claim；
- 只训练 residual；
- base logits / backbone 冻结；
- 不能使用 phase/cascade ID；
- 不在 test 上调参。

比较：

```text
A
C residual with current features
C residual + causal capacity/temporal features
C-wide fixed residual
```

## 9.3 记录梯度健康

每次诊断抽样记录：

```text
grad_norm_detection
grad_norm_classification
grad_norm_ranking
grad_norm_distill
cosine(det, class)
cosine(det, rank)
parameter_delta_norm
residual_logit_abs_mean/p90
base_logit_abs_mean/p90
correction_strength
zero_grad_fraction
saturation_fraction
```

## 9.4 Capacity Gate

至少满足：

```text
future PR-AUC(A -> best fixed C) +0.03 absolute
OR
future onset AP +0.05 absolute

AND
anchor/familiar F1 drop <=0.03 absolute
```

如果所有固定 residual 版本都没有 capacity gain：

```text
STOP-CAP
```

优先修 representation/feature path，不进入 online optimizer 网格，更不能跑 D。

---

# 10. P22-S6：固定 C 的严格 prequential 在线实验

只有 S5 PASS 才运行。

## 10.1 时间顺序不能违反

每个 t：

```text
1. 用当前部署版本预测，立即保存 raw logits/probability/model_version/input_hash
2. 模拟环境推进
3. 等标签成熟
4. 用历史保存预测结算误差
5. qualification / protection decision
6. 最后才允许用该成熟样本训练
```

绝对禁止：

```text
训练后重新 forward 老窗口，再称它为“online prediction”
```

## 10.2 Development stream 结构

建议每条 3400 step：

```text
Familiar-1      500
Unseen-1       1200
Familiar-2      500
Unseen-recur   1200
```

但前提是新的 P22 generator + U4-v2 PASS。

建议先 1 条 development stream；通过后再 2 个 replay seeds。

## 10.3 A/C 最小矩阵

第一轮只跑：

```text
A
C-residual-off   # 无保护/无资格切换
C-residual-on    # 完整 protection
```

不要同时加入 D。

## 10.4 C 的核心指标

### Detection

```text
PR-AUC
F1@0.5
precision
recall
FPR
Brier/ECE
```

### Onset/event

```text
event recall
onset AP
false alarms / 1000 host-step
detection delay
time-to-recovery
```

### Diagnosis

```text
conditional resource macro-F1
end-to-end resource F1
CPU/RAM/Disk per-class F1
```

### Adaptation-specific

把 unseen 划分：

```text
early-unseen
mid-unseen
late-unseen
recurrent-unseen
```

主比较：

```text
C - A on late-unseen
C - A on recurrent-unseen
```

early 段可用于学习，不能拿 early 的训练后重算结果充当收益。

### Familiar protection

```text
familiar-2 F1
anchor F1
familiar PR-AUC
```

## 10.5 H3 Gate

进入 D 前至少要求：

```text
late-unseen PR-AUC: C >= A + 0.03
AND onset AP:       C >= A + 0.05
AND 至少 2 个 development replay seed 同方向
AND familiar F1 drop <= 0.03
AND anchor F1 drop   <= 0.03
```

若 PR-AUC 已很高导致 ceiling，可预注册用 paired event-level bootstrap：

```text
95% CI of C-A > 0
```

FAIL → STOP-C2。

---

# 11. P22-S7：判断是否真的需要“更多专家”

H3 PASS 也不能立即说 D 必要。

计算 late-unseen residual plateau：

```text
rolling event AP/F1
rolling residual loss
per-resource error
representation cluster error
gradient alignment
```

以下任一情况不允许启动 D：

### 情况 A：C 继续稳定改善

说明容量尚未明显不足。继续 C 即可。

### 情况 B：C 的错误主要是 calibration threshold

先做 causal calibration 对照，不新增专家。

### 情况 C：C-budget 增加同样训练步数就解决

说明问题是训练预算，不是专家容量。

### 情况 D：C-wide 从开始就能明显改善

这说明“更多容量有价值”，但还没有证明“动态增加”有价值；允许进入 D，对照 C-wide。

进入 D 的最低证据：

```text
C 有稳定学习收益
+
late-unseen 后半段仍存在至少 50 个独立错误事件
+
错误集中在可重复的新资源/temporal cluster
+
单纯增加训练步数不能消除
```

---

# 12. P22-S8：重新实现 Additive Dynamic Residual Expert

这是替代旧 Top-k ramp 的关键。

## 12.1 禁止旧方式

不得：

```text
newborn 加入 normalized Top-k pool
成熟瞬间切回 k=4
成熟时重新归一化全部专家
```

## 12.2 推荐结构

冻结 base：

\[
L_{final}
=
L_{base}
+
\sum_{e=1}^{E_{fixed}}\alpha_e R_e
+
r_{new}(t)\,\beta_{new}\,R_{new}
\]

其中：

```text
r_new(t) ∈ [0,1]
```

连续增长。

关键：

- old residual weights 不因 new expert 出生而重新归一化；
- new expert 在 birth 时贡献严格为 0；
- maturation 只是 ramp 达到 1，不触发 membership hard switch；
- retire 也必须 smooth ramp-down；
- 如果所有 expert 都实际 forward，再加权，就如实称 dense/additive routing，暂不宣称 sparse compute。

## 12.3 初始化

new expert：

```text
clone best parent or zero-safe initialization
```

但输出参与系数从 0 开始。

候选训练可以在 shadow path 进行：

```text
shadow_train=True
deployed_contribution=0
```

这样避免“部署为零导致梯度也为零”。

## 12.4 continuity Gate

创建专门：

```text
audit_ftmoe_protocol022_additive_continuity.py
```

检查：

```text
birth:       r=0
ramp:        0, .01, ..., .99, 1
mature:      1-eps vs 1
retire:      1 -> 0
reactivate:  0 -> 1
resume/save
```

对于相同 input/weights：

```text
state-transition-only logit jump <= 1e-6
```

ramp 自身预期变化单独报告：

```text
|Δlogit| / |Δr|
```

不能把正常连续 ramp 变化误判成 discontinuity。

必须对：

```text
eps = 1e-3, 1e-5, 1e-7
```

均通过。

FAIL → STOP-D0。  
禁止 D performance run。

---

# 13. P22-S9：Dynamic candidate 的因果资格验证

## 13.1 Trigger

trigger 只能基于：

```text
已成熟持续错误
固定/冻结 representation drift
独立事件数
现有 C 的 residual plateau
```

不能仅凭：

```text
高 routing entropy
单次大 loss
phase_id
cascade_flag
```

## 13.2 Candidate training set

必须包含真正触发它的：

```text
hard event windows
对应 mature labels
相关 normal hard negatives
anchor/familiar preservation samples
```

不能再出现：

```text
触发用 A 样本
训练却取“最近 10 个无关窗口”
```

## 13.3 Qualification

candidate 出生后才开始积累 qualification predictions。

对于每个未来窗口，在 label 前同时保存：

```text
C deployed
D candidate shadow
D without-new-expert counterfactual
Frozen A
```

标签成熟后才比较。

资格至少需要：

```text
>=3 independent validation blocks
>=20 independent relevant events
candidate AP/PR-AUC improvement
FPR budget pass
familiar side-effect pass
```

证据不足状态写：

```text
UNVERIFIED
```

不能当 PASS，也不能当 FAIL。

---

# 14. P22-S10：D 的公平对照

正式矩阵：

```text
A
C-residual
D-additive
C-wide
C-budget
```

公平原则：

## C vs D

相同：

```text
base checkpoint
input features
normalization
label delay
main online learner
loss
memory
anchor
protection
random stream
replay seed
```

## C-budget

C 获得与 D shadow candidate training 相当的额外优化预算。

记录：

```text
main updates
shadow updates
total backward passes
samples exposed
unique events exposed
wall time
peak RSS
parameter count
active/shadow/dormant parameter count
```

## C-wide

从 t=0 就拥有与 D 最终近似的固定容量，用于排除：

```text
“D 更好只是因为参数更多”
```

---

# 15. P22-S11：D Benefit Gate

D 主 claim 至少要求：

```text
1. D - C late-unseen PR-AUC >= +0.02 absolute
2. D - C onset AP       >= +0.03 absolute
3. paired event bootstrap 95% CI > 0
4. 至少 2 个 development replay seeds 同方向
5. familiar/anchor F1 drop <=0.03
6. D 优势不能被 C-budget 消除
7. D 相对 C-wide 至少体现 adaptation speed 或 cost/accuracy 优势
```

此外必须报告：

```text
expert birth count
qualified count
rejected count
time from trigger to qualification
expert contribution ablation
per-expert event specialization
```

如果新专家出生但贡献消融近乎 0：

```text
不能宣称 dynamic expert 有效
```

---

# 16. P22-S12：Recurrence / retire / reactivate（仅在 D 主收益成立后）

第一轮 D 主实验只需要 **add**。

只有 S11 PASS 后才验证 lifecycle：

```text
unseen-A
familiar gap
unseen-A recurrence
```

要求出现真实：

```text
expert active
→ retire/dormant
→ same expert_id retrieved
→ requalification
→ same expert_id reactivated
```

不能只做 unit test。

报告：

```text
cold new expert
vs
reactivated old expert
```

比较：

```text
time-to-benefit
samples-to-benefit
PR-AUC
onset AP
compute
```

---

# 17. 独立 Confirmation，禁止继续在 development 上循环汇报

一旦配置冻结：

```text
generator params
U4-v2 thresholds
probe
C configuration
D trigger/qualification
ramp
memory
metrics
```

写：

```text
artifacts/ftmoe_online/protocol_022/confirmation/frozen_config.json
```

随后至少使用：

```text
3 个完全未用于调参的 replay seeds
新的 mechanism_seed_confirm
```

development 的任何失败/成功都不能改变 confirmation 配置。

建议 paired bootstrap 的 unit：

```text
independent fault/cascade event
```

同时提供 host-level secondary CI。

**最终论文 claim 只能基于 confirmation。**

---

# 18. 对当前 learnability 结果的专项复查要求

执行模型必须专门解释旧结果：

```text
target=next AP              0.678
prevalence                  0.0779
time-shuffled-label AP      0.2419
conditional fault class acc 0.4889
```

至少回答：

1. 为什么单次 shuffled-label AP 会达到 0.24？
2. 100 次 permutation 后它是否仍异常？
3. persistence baseline AP 是多少？
4. 去掉 `past_fault_*` 后 AP 降多少？
5. 只保留 current ratio 后 AP 是多少？
6. onset-only task 是否仍可预测？
7. CPU/RAM/Disk 各 onset 是否都可预测，还是只有一个资源驱动整体指标？
8. 正确 lag 与错误 lag 是否存在显著差异？

在这些问题回答前，**不要把 H2 写成“cascade causal learnability 已证明”。**

---

# 19. 对模拟器 construct validity 的专项控制

P21 已证明：

```text
age0 直接超 capacity → placement rejection
age1 之后 demand 上升 → 已放置 container 可制造 overload
```

这意味着当前模拟器天然偏好“先准入、后需求上升”的故障机制。

因此论文表述不要写成：

```text
真实工业故障必然按该规律发生
```

应写成：

```text
controlled unseen temporal resource-demand regime
```

并增加两类 control：

### Simulator Control S0

相同 admission-safe age0，之后仅 CPU burst，不产生 RAM/Disk cascade。

### Simulator Control S1

相同 CPU/RAM/Disk marginal envelope，但随机 lag/order。

如果模型只在 cascade 上提升，而 S0/S1 不表现出相同收益，更能支持其学习联合时序关系。

---

# 20. 代码修改优先级

推荐顺序：

## 第一批：只改 data/audit

```text
BitbrainWorkloadProtocol022.py
register_ftmoe_protocol022.py
prepare_ftmoe_protocol022_unseen.py
analyze_ftmoe_protocol022_unseen.py
probe_ftmoe_protocol022_learnability.py
test_ftmoe_protocol022_audit.py
```

不动模型。

## 第二批：Fixed C diagnostics

复用/派生：

```text
recovery/PreGANSrc/src/ftmoe_online_r1.py
```

新增 Protocol022 runner，不原地覆盖 P20 R1。

## 第三批：Additive D

只有 C Gate PASS 后新增：

```text
recovery/PreGANSrc/src/ftmoe_online_protocol022_dynamic.py
audit_ftmoe_protocol022_additive_continuity.py
test_ftmoe_protocol022_dynamic.py
```

不要提前大规模实现 lifecycle。

---

# 21. 资源控制

沿用已有环境约束：

```text
Python 3.8 compatible
single process for heavy model/simulator runs
RAM guard >= 3.0 GiB free
torch thread count explicitly recorded
```

每个正式 run 保存：

```text
command
git SHA
Python version
torch version
device
seed
start/end time
exit code
peak RSS
elapsed time
source SHA256
checkpoint SHA256
```

PowerShell wrapper 的 stderr warning 不得再伪装成 observed exitcode。  
用 `subprocess.run(...).returncode` 或同等真实进程返回码。

---

# 22. 强制 STOP 表

| STOP | 条件 | 之后禁止做什么 |
|---|---|---|
| STOP-0 | task-level audit correctness FAIL | 禁止正式数据生成 |
| STOP-A2 | U3/U4-v2 FAIL | 禁止 unseen 模型实验 |
| STOP-B2 | onset learnability / permutation FAIL | 禁止调 C/D |
| STOP-CAP | fixed residual upper bound无收益 | 禁止 online grid / D |
| STOP-C2 | strict online C 不优于 A | 禁止 D |
| STOP-D0 | additive continuity FAIL | 禁止 D effectiveness |
| STOP-D1 | D 无独立收益 | 禁止宣称动态专家有效 |
| STOP-R | recurrence reuse FAIL | 只能保留 add-only claim |

**执行模型不能把 STOP 当成“任务没完成”。**  
STOP 本身就是实验结果，必须提交证据。

---

# 23. 每个阶段必须提交的文件

## S0–S3

```text
protocol.json
source_sha256_initial.json
audit_tests.json
u4_v2_registry.json
offline_reference_v2.json
candidate_*.json
matched_controls_*.json
selected.json
```

## S4

```text
learnability_state.json
learnability_onset.json
simple_baselines.json
permutation_null.json
lag_ablation.json
```

## S5–S6

```text
fixed_capacity_diagnostic.json
gradient_diagnostic.jsonl
A_summary.json
C_off_summary.json
C_on_summary.json
prequential_audit.json
```

## S8–S10

```text
continuity.json
expert_registry.jsonl
candidate_predictions.jsonl
qualification.jsonl
D_summary.json
C_wide_summary.json
C_budget_summary.json
fairness_audit.json
```

## Confirmation

```text
frozen_config.json
paired_results.json
bootstrap.json
final_acceptance.json
FINAL_REPORT.md
```

---

# 24. FINAL_REPORT.md 必须按这个结构写

```text
1. Git commit / source hashes
2. Historical facts accepted
3. Data Gate
4. U1/U2/U3/U4-v2 separately
5. Audit correctness
6. Learnability beyond persistence
7. Fixed C capacity diagnostic
8. Strict A vs C
9. Whether D was scientifically eligible
10. Additive continuity
11. C vs D vs C-wide vs C-budget
12. Familiar/anchor retention
13. Runtime / memory / update budget
14. Recurrence if executed
15. Every failed gate
16. What can and cannot be claimed
```

禁止只给一张“最终 F1 表”。

---

# 25. 给执行模型的首轮任务边界

**第一次提交只允许做到 S0–S4。**

即：

```text
修 task-level audit
→ register cascade_v2
→ data-only pilot
→ U4-v2
→ learnability-v2
```

**第一次提交禁止：**

```text
跑完整 C
实现/跑 D
改 FT-MoE backbone
重新训练离线 v4/P19/P20 checkpoint
```

原因：当前最大不确定性仍然是：

```text
新颖性仪器是否正确
+
learnability 是否超越 persistence
```

只有第一轮结果通过后，第二次提交才执行 S5–S7。

这能显著减少无效 GPU/CPU 实验。

---

# 26. 建议第一轮 Git commit

建议 commit message：

```text
protocol-022: task-level unseen audit and persistence-controlled learnability
```

至少包含：

```text
代码
tests
registry
3 个 data-only candidates
matched controls
U4-v2 result
learnability-v2 result
problem log
阶段报告
```

不要提交不必要的大模型 checkpoint；可提交 hash / manifest / summary。

---

# 27. 下一次反馈时需要提供的信息

下一次通过 GitHub 反馈时，只需让审查模型读取：

```text
branch: protocol-022
latest commit
docs/FTMOE_ONLINE_PROTOCOL_022.md
artifacts/ftmoe_online/protocol_022/gate_status.json
artifacts/ftmoe_online/protocol_022/problem_log.jsonl
artifacts/ftmoe_online/protocol_022/audit_v2/**
artifacts/ftmoe_online/protocol_022/learnability_v2/**
```

如果第一轮 Gate 全部 PASS，再进入固定 C。

---

# 28. 当前最重要的判定

根据 Protocol 019–021 的现有证据，下一步**不是**：

```text
继续调 learning rate
继续增加 expert
继续扩大 D matrix
继续制造更极端的 overload
```

下一步应该是：

```text
1. 把“task-level unseen”真正测对；
2. 排除 fault persistence 对 learnability 的解释；
3. 证明 fixed residual 本身存在可学习上界；
4. 再证明 strict-online C 真能学；
5. 最后才允许讨论 dynamic expert 的独立价值。
```

只要这五步按 Gate 顺序完成，之后得到的 D-C 结果无论正负，都比继续在当前 protocol 上堆实验更有科学解释力。
