# FT-MoE 下一步实验：C-Learnability Rescue → Specialist Necessity

> **当前唯一实验执行文档。** 后续模型默认只需先读 `PROJECT_CONTEXT_LATEST.md` 和本文件。不要递归读取历史 Protocol、artifacts 或 problem logs。

## 0. 当前问题与本轮目标

Protocol023 Round2A 已确认：

- 数据层可用：H1-v2 PASS，H2-v2 PASS；
- prequential 实现正确；
- Fixed-C 在当前 online budget 下几乎没有学会新 regime；
- first-exposure late-100 PR-AUC 相对 Frozen A 只有：compute +0.0049、memory +0.0020、io +0.0203；
- mature gradient 大多同向，没有预注册的强负梯度 conflict；
- 因此当前 `D_eligible=false`。

本轮不再尝试制造 gradient conflict，也不实现 D。

本轮只回答两个问题：

1. **Q1：在仍然合理的实时在线预算内，Fixed-C 能否真正学会至少两个新 regime？**
2. **Q2：当 Fixed-C 已经具备学习能力后，regime-specific specialist 是否显著优于一个共享 Fixed-C？**

只有 Q1=YES 且 Q2=YES，下一轮才实现动态专家 D。

---

# 1. 冻结项

以下全部冻结，不得因为模型结果改变：

- branch：`protocol-023`
- calibrated generator；
- development stream：`dev_seed700_steps2880_calibrated`；
- A/B/C 三个 regime 的顺序与物理定义；
- Frozen base checkpoint；
- seed700 仅用于开发；
- seeds 701/702/703 继续禁止使用；
- label delay、prequential prediction-before-update 顺序；
- 模型输入特征；
- base network 冻结；
- Fixed-C 仍为 4 residual experts。

禁止重新调数据场景来迎合模型。

---

# 2. 为什么先做 C-Learnability Rescue

当前 C 每 4 interval 更新一次，每个 420-step first-exposure phase 约只有 105 次更新。

但 Protocol022 的单-regime诊断已经观察到：固定 residual 在 400、800、1600 updates 时 PR-AUC 才逐步出现明显提升。

因此 Round2A 的失败更像：

```text
online plasticity / optimisation strength不足
```

而不是：

```text
固定专家容量一定不足
```

必须先建立一个强而公平的 Fixed-C 基线 `C*`，否则将来 D 赢了也可能只是赢一个弱 baseline。

---

# 3. Stage R3-1：最小 C Rescue 网格

只在 seed700 和同一 calibrated development stream 上运行。

不要做大网格，只跑下列 5 个配置。

## C0 — 当前 Round2A 配置

```text
update_every = 4
batch = 32
grad_steps = 1
replay = 64
LR = 1e-4
anchor = 0.25
distill = 0.10
```

作为复现基线，不重新调参。

## C1 — 提高更新频率

```text
update_every = 1
batch = 32
grad_steps = 1
replay = 64
LR = 1e-4
anchor = 0.25
distill = 0.10
```

目的：只测试“在线更新次数不足”。

## C2 — 相同时间频率，提高每次优化强度

```text
update_every = 4
batch = 32
grad_steps = 4
replay = 64
LR = 1e-4
anchor = 0.25
distill = 0.10
```

目的：区分 update frequency 与 optimisation depth。

## C3 — 扩大 replay

```text
update_every = 1
batch = 32
grad_steps = 1
replay = 256
LR = 1e-4
anchor = 0.25
distill = 0.10
```

目的：测试当前 64-interval replay 是否限制了新 regime 学习。

## C4 — 增强 plasticity

基于 C1，只降低保护强度：

```text
update_every = 1
batch = 32
grad_steps = 1
replay = 64
LR = 1e-4
anchor = 0.05
distill = 0.02
```

目的：测试 anchor/distillation 是否压制 residual 学习。

**本轮不搜索 LR。LR 固定 1e-4。**

---

# 4. 每个 Rescue 配置必须保存的指标

每个配置只需要一个 compact JSON：

`artifacts/ftmoe_online/protocol_023/rescue/<variant>/summary.json`

必须包含：

```text
variant
update_count
backward_passes
samples_seen
elapsed_seconds
update_time_total
prediction_time_mean
max_RSS
trainable_params

A1/B1/C1:
  whole_phase PR-AUC
  late_100 PR-AUC
  F1
  resource_macro_F1
  gain_vs_frozen_A

F1 familiar:
  FPR
  false_alarms_per_1000
  Brier

probe_A/B/C at boundaries
```

不保存大规模逐步调试 JSON，除非出现 bug。

---

# 5. C Rescue Gate

一个配置成为候选 `C*`，必须同时满足：

### 学习能力
至少 2/3 first-exposure regime：

```text
late_100 PR-AUC(C) - PR-AUC(A) >= +0.03
```

并且第三个 regime 不得严重退化：

```text
gain_vs_A >= -0.01
```

### 正常阶段保护

相对 C0 / Frozen A：

```text
normal familiar FPR increase <= 0.01 absolute
```

### 资源诊断
至少 2/3 regime：

```text
resource macro-F1 不低于 Frozen A - 0.02
```

如果多个配置通过，选择顺序不是“PR-AUC最高优先”，而是：

1. 最小 backward passes；
2. 最低 update wall-time；
3. 若成本接近，再选 macro PR-AUC 更高者。

这样得到的是“最小可部署强 C”，不是为了后续故意让 D 好赢的弱 C。

---

# 6. Rescue 失败时如何处理

如果 C1/C2/C3/C4 全部无法让至少 2 个 regime 达到 +0.03：

```text
STOP-C-LEARN
```

此时禁止实现 D。

只允许进一步做一个诊断：

```text
Oracle-C
```

做法：每个 first-exposure regime 的前半段用成熟标签训练 residual，后半段评估；不受实时 update budget 限制，但仍冻结 base。

若 Oracle-C 也不能明显超过 A：

```text
representation/residual architecture itself is inadequate
```

下一步应该改 residual representation，而不是动态增专家。

若 Oracle-C 能学而在线 C 不能：

```text
optimization / label-delay / sampling bottleneck
```

再针对该瓶颈改 online learner。

---

# 7. Stage R3-2：冻结 C*

一旦某个配置通过 Rescue Gate：

创建：

`artifacts/ftmoe_online/protocol_023/rescue/frozen_c_star.json`

写入：

```text
variant
all hyperparameters
source commit
stream hash
update budget
performance summary
```

从此以后 C* 不再调整。

---

# 8. Stage R3-3：Specialist Upper-Bound Test

这是决定 D 是否还有必要的核心实验。

目标不是先实现动态专家，而是回答：

> 同样的 residual capacity，如果不同 regime 拥有独立 specialist，是否比一个共享 Fixed-C 更好？

建立 3 个 specialist：

```text
S_A: 只使用 A1 的 mature samples 更新
S_B: 只使用 B1 的 mature samples 更新
S_C: 只使用 C1 的 mature samples 更新
```

三者：

- 起点与 C* 完全相同；
- 每个 specialist 仍是一个与 C* expert/residual 规模一致的 residual learner；
- 训练预算使用 C* 在对应 first-exposure regime 实际消耗的相同 backward-pass 数；
- 不使用 regime ID 作为模型输入；
- **训练时可按已知实验 phase 切分数据，因为这是 upper-bound diagnostic，不是最终部署算法。**

然后在 recurrence 流上比较：

```text
Shared C* on A2/B2/C2
vs
Matched Specialist S_A/S_B/S_C on对应 recurrence
```

这个实验只是验证“专业化有没有价值”，不是最终 D。

---

# 9. Specialist Necessity Gate

至少 2/3 regime 满足：

```text
Specialist recurrence PR-AUC >= Shared C* + 0.03
```

或者：

```text
Specialist resource macro-F1 >= Shared C* + 0.05
```

并且至少一个 regime 的：

```text
first_100 recurrence PR-AUC >= Shared C* + 0.05
```

若满足：

```text
SPECIALIST_NEEDED = true
```

这将成为 D 的新科学依据：

> 不同 recurring regime 具有可复用的 specialist advantage；动态专家的作用是在线发现、形成、休眠和再激活这些 specialist，而不是解决已经被否定的“强负梯度冲突”。

若不满足：

```text
STOP-D-NEED
```

说明共享 C* 已足够，不再实现 D。

---

# 10. 为什么这个 Gate 比旧 Gradient-Conflict Gate 更合适

Round2A 已经测得 mature gradients 大体同向，因此继续人为寻找负 cosine 没有意义。

动态专家仍可能有价值，因为：

- 同向梯度不代表最优参数相同；
- 不同 regime 可以共享大方向，但需要不同局部 residual correction；
- recurring regime 的 specialist 可以被存储并快速复用；
- D 的优势可以来自 conditional specialization 和 memory reuse，而不是 catastrophic interference。

因此下一轮 D 的准入依据改为：

```text
C* learns + specialists beat shared C*
```

不再要求：

```text
negative gradient cosine
```

旧 H3 STOP-GI 作为负结果永久保留。

---

# 11. 只有 Specialist Gate PASS 后，才允许实现 D

届时 D 的结构继续遵循已确定原则：

```text
additive residual experts
continuous contribution ramp 0 -> 1
no Top-k membership discontinuity
expert birth only from mature error evidence
expert retirement only after counterfactual harmlessness
recurrence uses dormant expert reactivation
```

并且最终一定要比较：

```text
A Frozen
B Full-online
C* Strong Fixed-C
C-budget
C-wide
D Dynamic
```

但这些**本轮都不执行**。

---

# 12. 本轮唯一输出

请只新增：

```text
artifacts/ftmoe_online/protocol_023/rescue/
  C0/summary.json
  C1/summary.json
  C2/summary.json
  C3/summary.json
  C4/summary.json
  rescue_comparison.json
  frozen_c_star.json          # only if rescue passes
  specialist_upper_bound.json # only if C* exists
  gate_status.json
```

以及一个简短：

```text
RESCUE_REPORT.md
```

最多 2–3 页，不写长 problem log。只有真实代码 bug 才增加 `issues.jsonl`。

---

# 13. 下一轮反馈给 ChatGPT 时的读取规则

只需让我读取：

1. `PROJECT_CONTEXT_LATEST.md`
2. `NEXT_EXPERIMENT_LATEST.md`
3. `artifacts/ftmoe_online/protocol_023/rescue/gate_status.json`
4. 必要时 `rescue_comparison.json` / `specialist_upper_bound.json`

不要让我扫描整个仓库。

---

# 14. 本轮最终布尔结论

`gate_status.json` 必须只给出：

```text
C_STAR_FOUND = true/false
SPECIALIST_NEEDED = true/false/not_run
D_NEXT_ROUND_ELIGIBLE = true/false
```

规则：

```text
D_NEXT_ROUND_ELIGIBLE = C_STAR_FOUND AND SPECIALIST_NEEDED
```

只有为 true，下一轮才开始实现 D。