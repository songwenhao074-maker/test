# FT-MoE Protocol 023：面向“动态增删专家优于固定在线微调”的多 Regime 持续学习实验计划

**日期：2026-09-12**  
**仓库：** `songwenhao074-maker/FT-MoE`  
**建议分支：** `protocol-023`  
**建议父提交：** `protocol-022 @ 6c5db86fde8afd10e3997d4ff0bf8016e2e87130`  
**目标：** 在一个合理、可复现、具有工业边缘计算意义的在线非平稳场景中，证明 D（动态增删专家的在线微调）相对于 A/B/C 具有稳定、可解释、不可由“更多训练步数/更多参数/更多信息”解释的优势。

---

# 0. 这轮实验的唯一最终目标

整个在线实验最终不再围绕：

```text
“某一个 cascade 场景能不能被 C 学会”
```

而围绕：

> **当 IIoT/边缘节点连续经历多种性质不同、驻留时间有限、并会再次出现的资源故障机制时，固定拓扑在线模型面临稳定性–可塑性冲突；D 能否通过按需新增专家、移除/休眠低贡献专家，并在旧机制再现时重新激活已有专家，在相同信息和公平训练预算下取得更高的持续在线性能？**

最终论文 claim 建议限定为：

> **D is advantageous under recurring heterogeneous non-stationarity where regime dwell time is shorter than the convergence horizon of a fixed online learner.**

中文可表述为：

> **在故障机制异构、持续切换且具有再现性的非平稳 IIoT 边缘环境中，当单一固定拓扑在线模型无法在有限驻留时间内同时兼顾快速适应和旧知识保持时，动态专家增删机制能够通过按需专业化与知识复用获得更稳定的检测与诊断性能。**

这比“D 在所有环境都优于 C”合理，也比人为扩大故障强度直到 D 胜出更有说服力。

---

# 1. Protocol 022 新结果的核心结论

Protocol 022 已经证明：

```text
H0 task-level audit correctness          PASS
H1 U3 + U4-v2 unseen validity           PASS
H2 learnability beyond persistence      PASS
H3 fixed residual capacity diagnostic   PASS
```

但 S5 给出的真正结论不是“需要动态专家”，而是：

```text
C 仍能通过更多优化预算持续改善。
```

C-current 的 late-unseen PR-AUC：

```text
100  updates -> 0.4256
200  updates -> 0.4374
400  updates -> 0.5200
800  updates -> 0.6568
1600 updates -> 0.6891
```

冻结 A：

```text
0.4228
```

800→1600 仍增加约：

```text
+0.0323
```

所以当前没有建立 fixed-expert capacity plateau。

同时 late-unseen onset AP：

```text
A             0.09173
C-current     0.09073
C-wide        0.09077
C-budget      0.08946
C-causal      0.09083
```

即：

```text
PR-AUC ↑
onset AP ↔ / ↓
```

当前 residual 更像 ranking/calibration correction，而不是 new-fault warning knowledge。

另外，registered lag 与错误 lag 几乎没有区别：

```text
registered     0.2392
disk lag2      0.2356
disk lag11     0.2425
ram lag1       0.2389
ram lag10      0.2379
```

feature ablation 中移除 migration 的影响最大，因此不能把当前 learnability 归因于 CPU→RAM→Disk 级联本身。

---

# 2. 为什么当前单一 cascade 不适合证明 D

S5 中：

```text
C-current = 4 experts
C-wide    = 8 experts
```

单一 unseen regime 下：

```text
C-wide    = 0.5286
C-current = 0.5200
```

差值只有约：

```text
+0.0086
```

而 C-budget：

```text
0.6061
```

说明当前主要瓶颈是优化预算，不是 expert capacity。

因此 Protocol023 的场景必须出现：

```text
多个互相干扰、需要不同专业化映射的 regime
+
有限驻留时间
+
旧机制再现
```

这才是动态专家真正应该发挥作用的情形。

---

# 3. 当前 familiar retention Gate 也要修改

S5 中：

```text
A familiar_1 F1 = 0.0
A familiar_2 F1 ≈ 0.069
```

所以“F1 drop <= 0.03”虽然数学上容易 PASS，但保护意义很弱。

Protocol023 必须拆成：

## Normal familiar

报告：

```text
FPR
specificity
false alarms / 1000 host-step
Brier
ECE
```

## Familiar fault probes

单独保留已知：

```text
CPU familiar
RAM familiar
Disk familiar
```

报告：

```text
PR-AUC
F1
Recall
resource classification
```

这些 probe 不参与 online training，也不参与 expert qualification。

---

# 4. Protocol 023 主场景：Multi-Regime Recurring Drift

定义三种 task-level resource mechanism。

每个 phase 只激活一种。

以下字段只能审计，禁止输入模型：

```text
phase_id
regime_id
mechanism_id
event_id
```

模型必须自己从资源序列和 scheduler context 推断。

---

# 5. Regime A：Compute-first Cascade

沿用 Protocol022 已经验证的数据机制：

```text
CPU burst
   ↓ lag 4
RAM retention/ramp
   ↓ lag 8
Disk accumulation
```

建议保持：

```text
CPU burst      4400–5200
RAM target     4500–6000
disk retained  16000–24000
probability    0.20–0.25
```

物理解释：

```text
compute-intensive inference burst
```

---

# 6. Regime B：Memory-pressure / Paging Cascade

物理解释：

```text
working-set growth / memory leak
         ↓
RAM pressure
         ↓ lag 3
Disk/page activity
         ↓ lag 6
CPU recovery overhead
```

形式：

```text
RAM ramp
   ↓3
Disk retained/read-write burst
   ↓6
CPU secondary burst
```

要求：

- age0 admission-safe；
- 后续才进入超额资源需求；
- bounded；
- self-cleaning；
- 事件规模和 prevalence 尽量与 A 接近；
- 不能只靠把 RAM 调得极端来制造差异。

---

# 7. Regime C：I/O-backpressure Cascade

物理解释：

```text
logging/checkpoint/data-upload accumulation
       ↓
Disk pressure
       ↓ lag 3
CPU cleanup / compaction
       ↓ lag 6
RAM cache disturbance
```

形式：

```text
Disk accumulation
   ↓3
CPU burst
   ↓6
RAM disturbance
```

同样要求：

```text
admission-safe
bounded
self-cleaning
matched severity
```

---

# 8. 为什么三种 Regime 更适合动态专家

三种机制具有：

```text
不同起点资源
不同 temporal order
不同 dominant fault class
不同 correction direction
```

例如：

```text
A: CPU -> RAM -> Disk
B: RAM -> Disk -> CPU
C: Disk -> CPU -> RAM
```

固定模型连续学习时有机会出现：

```text
gradient interference
router competition
catastrophic forgetting
representation compromise
```

而动态专家可以形成：

```text
Expert_A -> compute-first specialist
Expert_B -> memory-first specialist
Expert_C -> IO-first specialist
```

在 recurrence 时直接复用。

---

# 9. 必须匹配 Marginal，而改变 Joint Structure

三个 regime 尽量匹配：

```text
overall anomaly prevalence
event duration
event count
deployment rejection
migration rejection
资源 peak ratio
normal host-step 数量
```

主要差异来自：

```text
resource order
lag
cross-resource dependency
```

建议 Data Gate：

```text
anomaly prevalence           3%–12%
deployment rejection         <= 25%
migration rejection          <= 40%
independent fault events     >= 80 / regime
valid onset/follow-up        >= 50 / regime
worst event share            < 10%
regime prevalence difference <= 4 percentage points
```

---

# 10. 主时间轴

Development：

```text
F0 familiar            300
A1 compute-first       420
B1 memory-first        420
C1 IO-first            420
F1 familiar            240
A2 recurrence          360
C2 recurrence          360
B2 recurrence          360
```

总：

```text
2880 scored intervals
```

如果资源允许，可整体扩大到约 3500–3600 intervals，但 phase 比例必须在正式收集前冻结。

---

# 11. 为什么选择 420 左右的 first-exposure dwell

Protocol022 已实测：

```text
400 updates  -> PR-AUC 0.520
800 updates  -> 0.657
1600 updates -> 0.689
```

说明 fixed C 的收敛时间明显长于少量在线更新。

因此研究条件定义为：

```text
regime dwell time < fixed learner convergence horizon
```

有实验依据。

但不能只挑一个 D 最容易赢的速度。

必须做：

```text
Fast:   ~240–300 / regime
Medium: ~420–500 / regime
Slow:   ~900–1000 / regime
```

期望：

```text
Fast/Medium: D advantage明显
Slow: C逐渐追平
```

这种 sensitivity 会使最终结论更可信。

---

# 12. 数据生成 Seed

建议：

```text
development: 700
confirmation: 701, 702, 703
optional:     704, 705
```

701+ 不允许参与调参。

---

# 13. Multi-Regime Learnability / Specialization Opportunity Gate

先用简单 probe 分别测：

```text
A train -> A/B/C test
B train -> A/B/C test
C train -> A/B/C test
```

至少报告：

```text
state AP
onset AP
resource classification
```

如果 cross-regime generalization 几乎等于 within-regime：

```text
A-trained 在 B/C 只下降 <0.02
```

说明三个机制太相似，不适合证明专家专业化。

建议进入模型阶段前要求：

```text
within-regime AP
-
mean cross-regime AP
>= 0.05
```

至少 2/3 regime 满足。

---

# 14. Gradient Interference Gate

这是进入 D 前最关键的新 Gate。

用固定 C，从 A/B/C 各取相同数量成熟事件。

计算 residual 参数梯度：

```text
g_A
g_B
g_C
```

报告：

```text
cos(g_A,g_B)
cos(g_A,g_C)
cos(g_B,g_C)
```

进一步报告：

```text
router gradient cosine
expert gradient cosine
per-layer cosine
```

建议至少满足：

```text
one pair mean cosine <= -0.05
```

或：

```text
>=30% sampled cross-regime event pairs have cosine < 0
```

再做：

```text
train A -> evaluate A/B/C
train B -> evaluate A/B/C
train C -> evaluate A/B/C
```

如果学 B 后 A 明显下降，才建立真实 continual-learning conflict。

若三个 regime 梯度几乎完全同向：

```text
STOP-MR
```

不要继续实现 D。

---

# 15. A/B/C/D 方法冻结

## A — Frozen

不更新。

## B — Full Online Fine-tuning

全模型在线更新，但：

```text
same matured labels
same update opportunities
same replay memory
same anchors
same inputs
```

## C — Fixed-MoE / Fixed Residual

```text
4 residual experts
fixed topology
router + experts online trainable
```

## D — Dynamic Add/Delete Experts

起点和 C 完全一致：

```text
same checkpoint
same 4 experts
same router
same optimizer state
same features
same normalization
same memory
```

唯一核心差异：

```text
dynamic expert lifecycle
```

---

# 16. 必须加入 C-wide / C-budget

## C-wide

从 t=0 就拥有 D 的最大 expert capacity。

例如：

```text
8 experts
```

用于排除：

```text
D 只是参数更多
```

## C-budget

固定 4 experts，但获得：

```text
与 D candidate/shadow training 相同 backward pass 数
```

用于排除：

```text
D 只是训练更多
```

---

# 17. D 的结构必须使用 Additive Residual

最终：

```text
final = base + sum(active residual contributions)
```

new candidate：

```text
final = old + alpha_new(t) * residual_new
```

其中：

```text
alpha_new: 0 -> 1 continuously
```

旧专家权重不能因为 candidate 加入突然重新归一化。

禁止复用 P20 的：

```text
newborn -> normalized Top-k pool
mature boundary active 5 -> 4
```

旧实现已经出现约 3.47 logit jump。

---

# 18. D expert 数量

建议：

```text
initial active = 4
max active     = 6
max stored     = 8
```

运行中允许：

```text
4 -> 5 -> 6 -> 5 -> 4
```

C-wide：

```text
8 fixed experts
```

---

# 19. Spawn Trigger

禁止：

```text
phase_id
regime_id
mechanism_id
cascade_flag
```

只使用 mature information。

建议触发同时满足：

### 持续错误

最近 64–96 mature windows：

```text
rolling residual BCE > reference p90
```

且：

```text
>=20 independent fault events
```

### 现有专家无法短期解决

连续两个评价块：

```text
rolling performance无明显改善
```

### 表征新颖性

固定 monitor 上：

```text
error centroid 与已有 expert prototype 距离大于冻结阈值
```

至少前两项必须满足。

---

# 20. Candidate Training

candidate 训练集：

```text
50% triggering/relevant hard events
25% hard normal negatives
25% familiar anchors
```

相似事件只能通过：

```text
fixed-representation nearest neighbour
```

检索。

禁止按 regime ID 选样本。

---

# 21. Candidate 初始化

建议：

```text
clone nearest expert hidden layers
+
zero final residual layer
```

这样：

```text
initial contribution = 0
```

但内部表示已有合理起点，利于快速专业化。

---

# 22. Qualification 必须是未来预测

candidate 出生后才能积累验证预测。

每个 future step 在标签出现前保存：

```text
A
B
C
D-current
D + shadow candidate
D without candidate counterfactual
```

label mature 后再结算。

至少：

```text
3 validation blocks
20 relevant independent events
```

建议 PASS：

```text
candidate PR-AUC +0.03
OR resource macro-F1 +0.05
AND normal FPR increase <=0.01
AND familiar-anchor PR-AUC drop <=0.02
```

---

# 23. Additive Ramp Continuity

qualification PASS 后：

```text
alpha = 0
```

随后 32 intervals：

```text
0 -> 1
```

专门测试：

```text
epsilon=1e-3
1e-5
1e-7
```

要求 mature boundary：

```text
state-transition-only max logit jump <=1e-6
```

---

# 24. Expert Removal / Retirement

不能因为低 usage 就删除。

连续 3 个评价块同时满足：

```text
routing usage <2%
AND
counterfactual remove metric delta <0.005
```

才进入 retirement。

alpha：

```text
1 -> 0
```

32 intervals。

之后从 active pool 删除，权重放入 dormant archive。

必须记录：

```text
active parameter count
dormant parameter count
archive memory
active FLOPs
```

---

# 25. Reactivation

未来 error cluster 与 dormant expert 相似时：

```text
retrieve same expert_id
```

但先：

```text
shadow requalification
```

PASS 后：

```text
alpha 0 -> 1
```

记录：

```text
reactivation latency
samples-to-benefit
cold-new-expert latency
```

如果 recurrence 时旧专家明显更快恢复，这就是 D 的核心复用证据。

---

# 26. 实时 Online Budget 必须冻结

Protocol022 已证明给 C 无限训练会继续提升。

所以 Protocol023 必须模拟真实在线计算约束。

例如：

```text
每 4 intervals 最多 1 次 main update
batch=32
```

或：

```text
每个 420-step regime 最多 100 main updates
```

具体预算由实际 runtime benchmark 决定，而不是根据 D/C 结果调。

原则：

```text
所有主方法共享相同 main-update opportunity
```

D 的 shadow candidate extra updates 要被 C-budget 匹配。

---

# 27. 主评价指标

不能只看 full-stream F1。

每个 regime 报告：

```text
PR-AUC
F1
precision
recall
resource macro-F1
HR@100
NDCG@100
```

还要：

## Worst-regime

```text
min regime PR-AUC
min regime resource F1
```

## Switch adaptation

切换后：

```text
first 50
first 100
first 200 intervals
```

计算：

```text
PR-AUC
F1
resource F1
```

以及：

```text
adaptation curve area
```

## Time-to-recover

达到该 regime 最终性能 90% 所需：

```text
intervals
mature events
```

## Forgetting

例如：

```text
A1 end score
- A score after learning B1/C1
```

## Recurrence

A2/B2/C2 前 100 intervals：

```text
D 应快速恢复
```

---

# 28. Familiar Retention 新指标

## Normal familiar

```text
FPR
false alarms / 1000
specificity
Brier
ECE
```

## Familiar-fault anchors

CPU/RAM/Disk 已知故障 probe：

```text
PR-AUC
F1
Recall
resource F1
```

---

# 29. D 的 Primary Success Gate

至少 3 个 confirmation seeds。

## Macro performance

相对最佳 B/C：

```text
D macro PR-AUC >= +0.03 absolute
```

## Worst-regime

```text
D worst-regime PR-AUC >= +0.05
```

或：

```text
resource macro-F1 >= +0.05
```

## Adaptation

```text
D adaptation AUC >= C +0.03
```

或：

```text
time-to-90% 至少缩短25%
```

## Recurrence

至少两个机制：

```text
D first-100 recurrence PR-AUC >= C +0.05
```

且：

```text
same expert_id reactivated
```

## Retention

```text
familiar-anchor PR-AUC drop <=0.02
normal familiar FPR increase <=0.01
```

---

# 30. Fairness Gate

## vs C-budget

```text
D macro PR-AUC >= C-budget +0.02
```

或：

```text
D adaptation AUC >= C-budget +0.03
```

## vs C-wide

优先要求：

```text
D macro PR-AUC >= C-wide +0.01
```

若 accuracy 持平：

```text
|D-Cwide| <=0.01
```

则至少：

```text
D active FLOPs <=80% of C-wide
```

才能宣称 efficiency advantage。

如果最终论文必须严格“性能优于”，则 efficiency-only 不作为最终结论。

---

# 31. Statistical Gate

统计单位：

```text
independent fault event
```

paired bootstrap：

```text
>=2000 samples
```

报告：

```text
mean difference
95% CI
bootstrap sign probability
```

至少在：

```text
macro PR-AUC
adaptation AUC
worst-regime metric
```

三项中两项：

```text
D-C lower CI > 0
```

seed：

```text
>=3
```

最好 5。

---

# 32. 专家专业化证据

必须保存：

```text
expert routing histogram
expert contribution by regime
expert ablation by regime
expert centroid
expert pairwise cosine
```

希望事后看到：

```text
Expert4 -> A
Expert5 -> B
Expert6 -> C
```

但 regime ID 不能参与训练。

---

# 33. Expert Ablation Gate

对新增专家 i：

```text
full D
vs
D without expert_i
```

按 regime 分析。

建议：

```text
own-regime metric drop >=0.03
other-regime mean drop < own-regime drop/2
```

否则不能说该新增专家形成有效专业化。

---

# 34. Sensitivity：Switching Speed

必须测试：

```text
Fast
Medium
Slow
```

最理想趋势：

```text
             Fast      Medium      Slow
A            poor      poor        poor
B            unstable  moderate    better
C            poor/mod  moderate    good
D            best      best        ≈C
```

这会直接说明：

```text
D 的优势来自 non-stationarity speed
```

而不是模型无条件更强。

---

# 35. Sensitivity：Regime Count

资源允许时：

```text
1 regime
2 regimes
3 regimes
```

理想：

```text
1: C≈D
2: D开始领先
3: D明显领先
```

这是动态 expert necessity 最直接的因果验证。

---

# 36. 执行阶段

## P23-S0

冻结 P22，不修改历史。

## P23-S1

实现 A/B/C 三种 generator 和 tests。

## P23-S2

Data-only pilot，完成 marginal matching。

## P23-S3

cross-regime probe + gradient interference + sequential forgetting。

如果没有专业化/冲突：

```text
STOP-MR
```

## P23-S4

严格 prequential A/B/C。

证明固定 C 在有限实时预算下确实存在：

```text
adaptation / forgetting tradeoff
```

如果 C 可以快速解决所有 regime：

```text
STOP-D-NEED
```

## P23-S5

只实现 D continuity：

```text
birth
ramp
mature
retire
reactivate
save/resume
```

## P23-S6

seed700 development：

```text
A/B/C/D/C-wide/C-budget
```

仅允许调：

```text
trigger
qualification
retirement
ramp
```

不允许改 generator。

## P23-S7

冻结配置。

## P23-S8

seed701/702/703 confirmation。

## P23-S9

speed / regime-count / expert-cap sensitivity。

---

# 37. 第一次交给其他模型的任务边界

第一次不要实现 D。

只做到：

```text
S0
S1
S2
S3
S4
```

必须回答：

```text
1. 三种 regime 是否真的不同？
2. cross-regime generalization 是否下降？
3. 是否存在 gradient conflict？
4. C 学新 regime 时是否忘记旧 regime？
5. recurrence 时 C 是否需要重新学习？
6. 给定实时 update budget 后，C 是否存在稳定性–可塑性矛盾？
```

只有这些成立，第二轮才允许 D 登场。

---

# 38. 进入 D 前必须同时满足

```text
1. A/B/C 三种机制 Data Gate PASS
2. 至少2/3 regime有 specialization opportunity
3. 至少一对 regime有 gradient conflict
4. C 能学习当前 regime
5. regime switch 后 C 明显退化
6. 学新 regime 后旧 regime 性能下降
7. recurrence 时 C 需要重新适应
8. 相同有限预算不能立即恢复
```

---

# 39. 不再建议做的事情

不要继续：

```text
提高 cascade_v2 probability
继续放大 CPU/RAM/Disk 幅度
给当前 single-regime C 无限训练
直接在现有 stream 上跑 D
只比较 D 和 C4
只看 full-stream F1
只用 seed600
```

尤其不要：

```text
先跑 D -> 找 D 赢的 phase -> 再定义主场景
```

必须：

```text
先注册 dynamic-need scene
-> 证明 fixed model conflict
-> 冻结 scene
-> 再跑 D
```

---

# 40. 最终论文主表建议

| Method | Macro PR-AUC | Worst-Regime PR-AUC | Macro F1 | Resource F1 | Adaptation AUC | Recurrence PR-AUC | Familiar FPR | Time/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A Frozen | | | | | | | | |
| B Full-Online | | | | | | | | |
| C Fixed-MoE | | | | | | | | |
| C-budget | | | | | | | | |
| C-wide | | | | | | | | |
| **D Dynamic** | **best** | **best** | **best** | **best** | **best** | **best** | controlled | acceptable |

D 最重要的不是 overall F1 略高，而是同时改善：

```text
worst-regime
switch adaptation
recurrence
forgetting
```

---

# 41. 最重要的动态图

建议最终生成：

```text
X-axis: time
background: F/A/B/C/F/A/C/B

upper:
rolling PR-AUC/F1 of A/B/C/D

lower:
active expert count
expert birth
expert retire
expert reactivation
```

理想可视化：

```text
A1 -> Expert4 born
B1 -> Expert5 born
C1 -> Expert6 born
F  -> low-contribution expert retired
A2 -> Expert4 reactivated
```

如果 recurrence 后 D 快速恢复，而 B/C 重新学习，这张图会非常直观地证明动态专家价值。

---

# 42. 下一次 GitHub 反馈需要提交

第一轮完成后：

```text
branch protocol-023

artifacts/ftmoe_online/protocol_023/
    protocol.json
    gate_status.json
    problem_log.jsonl

    data_audit/
        regime_A.json
        regime_B.json
        regime_C.json
        marginal_match.json

    specialization/
        cross_regime_probe.json
        gradient_interference.json
        sequential_forgetting.json

    fixed_baselines/
        A_summary.json
        B_summary.json
        C_summary.json
        prequential_audit.json

docs/FTMOE_ONLINE_PROTOCOL_023.md
```

第一次反馈不要先实现 D。

---

# 43. 当前最终判断

Protocol022 的价值非常大，因为它已经证明：

```text
单一 unseen regime + 足够训练预算
并不能构成动态专家的科学必要性。
```

下一步应该转向：

```text
多机制
短驻留
连续切换
旧机制再现
实时 update budget
```

这一类动态专家真正应该发挥优势的环境。

在该环境中，D 的优势可以明确拆成：

```text
新增专家 -> 快速专业化
移除专家 -> 减少干扰/控制计算
再激活专家 -> 避免重新学习
```

如果最终 D 在相同信息、相同主在线预算原则下稳定超过：

```text
A
B
C
C-budget
C-wide
```

那么“动态增删专家”的贡献才真正可以在论文中站住。
