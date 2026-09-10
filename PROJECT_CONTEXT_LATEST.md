# 项目最新上下文

本文件是唯一当前状态入口。**本次订正：2026-09-10**，订正范围与遗留矛盾见 [§9](#9-本次订正记录2026-09-10)。历史原文见 [文档归档](docs/archive/README.md)；协议 020 的完整证据在 `指令/` 各文件与 `artifacts/ftmoe_online/protocol_020/`。

## 0. 文档优先级（协议 020 修订计划 §0 登记，本次未改动该链条）

当前用户要求 > [离线未覆盖模式计划](指令/FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md) > [修订实验计划](指令/FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md) > [最新 HANDOFF](指令/FTMOE_PROTOCOL020_HANDOFF.md)（取其顶部"当前计划入口"）> 2026-09-08 复盘 > [原始详细方案](指令/FTMOE_PROTOCOL020_DETAILED_SOLUTION_PLAN.md)。

原始详细方案的 S0–S6、旧门禁与失败过程保留历史价值；其"重训 S6、直接推进 S8/S10、禁止开发期模型反馈调整场景"等规定**不是**当前执行路线。历史结论不得改写为 PASS。

## 1. 当前阶段：协议 020（在线修正与动态专家）

**一句话状态**：R0-A 因果基础通过、R0-B 动态部署**未通过**；R1（冻结基础＋固定在线修正）已完成 8×2000 步开发对照，结论是**减轻了旧 C 的漂移退化，但尚未稳定超过 A**；下一项是完整离线覆盖审计与新模式注册，**尚未开始**；R2–R6 未实施，S10 未开始。

本次核查（2026-09-10）：仓库内**没有**训练/采集/重放进程在运行；`artifacts/ftmoe_online/status.json` 已同步改为本状态。

### 1.1 阶段状态

| 阶段 | 状态 | 证据入口 |
|---|---|---|
| S0–S3 | 完成：019 冻结登记 / RAM_CAP_SCALE / graph v3（per-sample capacity + before_placement）/ VM source-disjoint split（eligible 457：train 277 / dev 86 / online 94） | `protocol_020/protocol.json`、`vm_split.json`、`test_ftmoe_protocol020_{simulator,graph}.py` |
| S4 | 完成（曾按 P16 正式停止，用户 2026-09-08 授权方向 (b) 后以 ~75 data-only 候选 + disk-law 手术重做） | `capacity_scan/`、`capacity_scan_report.json` |
| S5 | **PASS**：dev drift 流 v4 全相位 dominance，dep 18.7%（≤25）、mig 30.9%（≤40）；相位计数 baseline 3 / cpu_fault 43·share100% / ram_fault 173·97.2% / disk_fault 226·90.4% / cpu_recurrence 40·88.9% | `drift_streams/dev_seed500_steps2000`、问题日志 P23 |
| S6 | 完成：同域适配 `s6/adapted_v4_seed1/best.pt`（epoch4；dev PR-AUC 0.526 / macroF1 0.395），数据 `adaptation_data/v1`（train12/dev8） | `protocol_020/s6/`、`protocol_020/adaptation_data/v1`；配置含义见 §1.3 |
| S7-stationary | **PASS**：A/B/C 各 2000 步，无遗忘、C PR-AUC ≥ A−0.02、采样零违规 | `S7_stationary_result.json`、`runs/*_seed501` |
| S7-drift | 完成 A/C：**C ≈ A，无可测 adaptation** | `S7_drift_AC_result.json`、`runs/*_seed500_drift` |
| S8 | 触发逻辑 v3 已实现（`ftmoe_dynamic_expert_v3.py`，测试 7/7）；容器级集成后续在续跑中以 `ftmoe_online_s8.py` 完成，12 次新 2000 步运行显示 **D 未稳定胜过 A/B/C**，真实休眠/唤醒未观察到 | `runs_continuation/`、`continuation/audit_final.json`、`continuation/comparison_tables.md` |
| R0 | **A 通过、B 未通过**：23 项旧回归＋12 项新测试通过，runner 复验 39/39；ramp 连续性检查未通过（`ramp_continuity_passed: false`） | `revision_20260909/r0/verification_final.json` |
| R1 | 工程与 8 组开发对照**完成**，性能目标未达成（见 §1.2） | `revision_20260909/r1/final_runs_20260909_1435/` |
| S9/S10 | 未开始（S10 不得直接启动） | — |

另有历史续跑：12 次 2000 步开发运行（`runs_continuation/`），D 在漂移流真实启用过第五个专家（1e-4 设置 step660、1e-5 设置 step1900），平稳流无新增专家；提高学习率加重误报，检测正类权重 8→1 后 F1 改善但召回与 PR-AUC 下降、且 D 与 C 预测相同。原始 v4、019 与 020 起点在全程均未重训、哈希未变。

### 1.2 R1 固定对照结果（开发数据，**不是**独立确认）

统一源模型 `protocol_020/s6/adapted_v4_seed1/best.pt`（模型种子 1）；流 `dev_seed500_steps2000`（漂移）与 `dev_seed501_steps2000`（平稳）；共 8 次 2000 步运行、16000 个时间步骤，单进程顺序执行。指标为标签容忍口径、固定阈值 0.5，PR-AUC 按既有实现为 average precision。

| 流 | 方法 | 检测 F1 | PR-AUC | 召回 | FP | FN | 端到端资源宏 F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 漂移 500 | A | 0.5118 | 0.6779 | 0.8277 | 1159 | 142 | 0.4598 |
| 漂移 500 | C-legacy | 0.4701 | 0.6179 | 0.8337 | 1412 | 137 | 0.4512 |
| 漂移 500 | C-residual-off | 0.5026 | 0.6782 | 0.8337 | 1223 | 137 | 0.4547 |
| 漂移 500 | C-residual-on | 0.5091 | 0.6769 | 0.8289 | 1176 | 141 | 0.4577 |
| 平稳 501 | A | 0.1053 | 0.1270 | 0.7875 | 1054 | 17 | 0.0381 |
| 平稳 501 | C-legacy | 0.1301 | 0.0969 | 0.6750 | 696 | 26 | 0.0557 |
| 平稳 501 | C-residual-off | 0.1057 | 0.1284 | 0.7750 | 1031 | 18 | 0.0383 |
| 平稳 501 | C-residual-on | 0.1053 | 0.1285 | 0.7750 | 1036 | 18 | 0.0381 |

结论（与 [R1 实验记录](指令/FTMOE_PROTOCOL020_R1_EXPERIMENT_20260909.md) 一致，逐条可核）：

1. 固定小修正结构明显减轻旧 C 在漂移流上的退化，但**尚未稳定超过 A**（漂移 F1 与排序仍略低于 A；平稳流微小排序变化不足以宣布优势）。旧 C 在平稳流的 F1 与资源诊断更高，但漏报与排序更差，不能称新结构全面优于旧 C。
2. 基础路径保留有效：四个残差运行的每条完整基础预测均与同流 A 一致，基础参数与 buffer 哈希不变；learner 确实更新过，不是因梯度被阻断而等价于 A。
3. 保护减少了部分伤害：漂移 off 相对 A 改对 10 个检测错误、新增 69 个错误；on 改对 9 个、新增 25 个；FP 从 1223 降至 1176，仍高于 A 的 1159。
4. 实际保护参与：漂移接受 6 个快照、拒绝 12 个、回退 1 次，修正参与 1717/2000 步（85.85%）；平稳接受 3 个、拒绝 15 个、无回退，参与 991/2000 步（49.55%）。基础参与比例含初始等待，不能全称为"回退时间"。
5. 资源诊断学习有限：漂移 824 个异常 host-step 上 off 只改变 1 个资源类别判决、on 为 0；平稳两组均为 0。当前配置主要产生小幅分数修正。
6. 训练锚点 F1：A 恒为 0.6120；旧 C 最低 0.4778、最终 0.5217；off 最低及最终 0.5750；on 最低 0.6021、最终 0.6120（末段回退所致）。锚点用于训练与蒸馏，**不是**独立保留测试。

完整性与成本（本次复核）：16 项测试通过；`verify_ftmoe_protocol020_r1_results.py` 独立重算 **106/106 通过**（我于 2026-09-10 重数 `verification_results.json`：106 passed / 0 failed，`status: PASS`，其 scope 明示仅代表记录完整性、不代表性能成功）；两对 off/on 的 learner 哈希、优化器状态、200 次更新与抽样曝光序列一致，故部署差异可归于保护策略。总参数/可训练参数：A 242786/0、C-legacy 242786/134452、固定修正（on/off 相同）281794/9752；平均预测耗时约 A 7 ms、残差 9.7–11.6 ms，单次完整运行 A 22.09 s、旧 C 39.34 s、残差 102.90–133.91 s（含每步多路预测记录与周期 checkpoint I/O，不能全部解释为专家计算）。

冻结与未变（我 2026-09-10 重新计算哈希，与登记一致）：

- 模型模块 `recovery/PreGANSrc/src/ftmoe_online_r1.py` = `72ae878eebaf8a5abca76123c1127a9c1910b49c39dd60aa8d1d04b92acd80cb`
- 运行器 `run_ftmoe_protocol020_r1.py` = `bd3c0f72d8cef4a763a995830277758ebd6b83dfd9f55e797d60738e0453f8e8`
- 三个受保护模型哈希不变：`physical_lr0003_e30/v4_seed1/checkpoints_by_epoch/epoch021.pt` = `e3513575…`、`protocol_020/s6/adapted_v4_seed1/best.pt` = `10c44bdb…`、`protocol_019/s3/adapted_v4_seed1/best.pt` = `670c56fe…`

### 1.3 场景纠偏 P36（限制 R1 结论的适用范围）

实际 S6 起点 `adaptation_data/v1` 的 12 个训练 episode 已覆盖 baseline / cpu_fault（CPU×0.65）/ ram_fault（RAM×0.55，RAM 上限 2400）/ disk_fault（Disk×0.35），与在线 dev500 各相位的主要容量及 RAM 适配配置**一致**；dev501 为同域平稳流。顶层候选 `profiles` 陈旧，不能据此误判未见性；来源或轨迹变化不等于类型未见。

因此：R1"未稳定胜过 A"只在这些**已覆盖模式及其切换**上成立，**不能**外推为"在线学习新模式无效"。完整 v4→019→020 学习链、选模数据、初始锚点与数据分布覆盖仍待审计（P36 措辞为"目标与实验设计不充分匹配"，未指控样本泄漏）。

### 1.4 下一步（均未开始，不得当作已完成）

1. `offline_coverage_audit` 与 `unseen_regime_registry`：记录实际 checkpoint 链及数据用途、候选生成规约、离线排除证据、特征可观察性、物理标签、源/时间划分、固定时序、回放种子与开发/确认身份。**本次全库检索未见任何同名产物**（仅计划文档提到该交付名）→ 未开始。
2. 新模式小试流：登记时序为熟悉 1000 / 新模式 2000 / 熟悉 1000 / 新模式再现 2000 / 熟悉 1000，共 7000 个评分步骤另加 guard；先一类一个开发种子核验工程与数据。→ 新场景**未采集**。
3. R2 损失/采样 2×2 消融（均匀/事件分层抽样 × 检测正类权重 1/8.008742），主要评价放在核验后的新模式流，旧 dev500/dev501 仅作熟悉模式回归。→ 未实施。
4. R3 容量关系输入与长期情景记忆 → R0-B 修复与 R4 动态专家独立价值 → R5 场景族扩展 → R6 独立确认。→ 未实施。

门禁：R0-B 未通过前不得扩大 D 的效果矩阵；不直接启动 S10；确认流不得参与选场景/选模型/调参；原始离线 v4 与 019、020 起点不得重训或覆盖。

### 1.5 已知缺陷与口径限制（须一并报告，不得只取最好看的指标）

- **P34（已修复）**：完整运行首个含正例评估块暴露单窗口账本 logits/labels 拼批维度不一致，`C-residual-off_dev500_m1` 在 cursor 301 中止；已在模型评分函数规范为 host 行并补真实 320 步含正例回归。原失败目录与"兼容层重试"中止目录保留、**不作正式结果**；正式表只取 `final_runs_20260909_1435/`。
- **P35（未解决，转 R2）**：近期采样异常占比高于流实际（漂移 3.473% vs 2.575%；平稳 0.675% vs 0.250%），检测正类权重仍为 8.008742。该统计不含锚点、也不是实测梯度贡献，不能直接当作退化原因。
- **退出码证据缺口**：首个正式 off/dev500 运行的 PowerShell 包装器把 PyTorch warning 当作错误，该进程退出码**未被直接捕获**，日志末尾 `EXITCODE=0` 是执行代理依据完成文件追加的推断；后五组改为直接捕获 subprocess 返回码。该组仍通过 2000 步产物与独立验收。
- **P29/P30/P31/P32 的修复边界**：P32 的"基础预测保留"部分已由 R1 修复并逐预测验证，长期记忆部分仍待 R3；P30 触发证据与候选培养脱节、P31 完整启用/休眠/再唤醒有效性仍待 R4；R0-B 与 P30 的动态生命周期问题**均未通过**。
- 所有 R1 数字来自两条**开发**流（dev500/dev501）且已用于开发比较，不是独立确认；端到端资源指标计入误报、漏检与错分，条件分类 F1 另见分析文件，两者不可互相替代。

## 2. 状态订正：协议 017 / 018 的真实终态

原文写作"正在执行：容量 017 与 Google2011 协议 018"，与产物不符，据实订正如下（其后项目转入协议 019/020，这两项不再处于执行中）：

- **协议 017（容量）**：9 组开发扫描已跑完并选出配置 `ram0.5_disk1`（`capacity_017/selection.json`：主类占比 0.9525/0.02625/0.01167/0.00958，distribution_score 0.614368，`roughly_similar: false`）。2000 步确认运行**未完成**：`confirmation/seed302_steps2000/failure.json` 记录 `RuntimeError: RAM guard below 4.5 GiB (available 4.461 GiB)`，completed_steps 1479/2000，并自记 `simulator_behavior_modified: true`、`next_action: "Report to user before simulator changes"`。**不存在** `capacity_017/result.json`。
- **协议 018（Google2011）**：数据未下载、未运行，**不存在** `google2011_018/result.json`。
- 两者均不得称为已完成；模拟器此前是否被改动过，须以该 `failure.json` 的自记为准并等用户确认，本次订正未触碰模拟器。

## 3. 归档：协议 014（已验收回档基线，内容保留）

B_best 为每个模型种子在公共 30 轮预算内，按三个开发重放的平均综合分数选出的 checkpoint；不是 F1 最高轮，也不是跨五种子统一选某一轮。五种子汇总如下：

| 模型 | F1 | HR=NDCG |
|---|---:|---:|
| v0 | 0.525479 | 0.860048 |
| v1 | 0.567550 | 0.860922 |
| v2 | 0.901334 | 0.936297 |
| v3 简单融合 | 0.894437 | 0.962851 |
| v4 完整注意力 | 0.905134 | 0.972003 |
| v4 门控替换 | 0.903579 | 0.970907 |

HR、NDCG 的五种子均值在 A_last 和 B_best 下均满足 v0<v1<v2<v3<v4，完整 v4 也高于门控。F1 的研究主链为 v0<v1<v2<v4，不要求 v3>v2。

v4 相比 v2、门控的 B F1 增量为 0.003799、0.001554，均小于此前登记的 0.005，条件区间跨零。新增种子 17/42 中门控检测均略高于完整注意力，v1 诊断也没有稳定改善。0.005 是本项目开发筛选门槛，不是论文消融有效性的普遍要求；失败结果仍可如实报告。

原三种子历史组 B F1=0.546070/0.596316/0.898028/0.893558/0.903615，已核验可复用，但不能只保留这组而省略新增种子。用户阶段验收不等同于新增统计证据。

## 4. 模型及贡献边界

实现入口：`recovery/PreGANSrc/src/ftmoe_ablation.py`，全模型包装：`ftmoe_end_to_end.py`，训练器：`train_ftmoe_end_to_end.py`。

| 版本 | 实际增加的计算 |
|---|---|
| v0 | 逐主机两层 Transformer、公共检测/分类头 |
| v1 | 普通 softmax MoE，4 专家 |
| v2 | 第二个 4 专家 EAGate 池，以及拟调度后的资源聚合输入 |
| v3 | 调度感知图编码；图特征残差与图预测 logits 相加 |
| v4 | 保留 v3，增加跨主机多头交叉注意力、同主机逐元素交互、归一化与输出 adapters |

v2-v1 不是纯路由替换；v4-v3 不是纯注意力消融。移除图分支及依赖融合对应 v2，只能检验整套图扩展。门控对照保留交互等其他计算，以同参数量 16,640 的同主机门控替换注意力。

## 5. 数据、选模与已知局限

- 离线全模型数据：`artifacts/ftmoe_end_to_end/data/protocol_004_physical`；训练重放 42/1/6/17/23，开发重放 31/101/102。协议 020 另用 `protocol_020/` 下的 VM split、适配集与两条 dev 流（§1.1）。
- 16 主机、7 特征、12 步窗口；离线每个完整重放 202 步，不跨块构造窗口；归一化仅使用训练数据。
- 已修复生命周期/trace 绑定；输入为真实动作前主机信息，标签为实际动作后过载及主导资源类别，块内 ±1 步容差。
- HR@100%、NDCG@100% 在当前单资源标签下均等于真实异常样本上的资源 top-1 准确率，包含检测漏报；不能混用旧全局 top-100 检索指标。
- 当前 S=(F1+HR+NDCG)/3，实际给诊断两倍权重。早期基线可能几乎只预测 CPU；始终预测 CPU 的开发诊断准确率约 0.898224。
- 单一底层 workload、多次开发集选模及类别不均衡均限制结论。无训练物理规则的开发 F1≈0.93057、诊断≈0.98194，高于当前候选。
- 论文参考 F1/HR/NDCG=0.8766/0.6496/0.6021；数据、标签及指标实现有差异，不能据数值更高声称复现成功或优于论文。
- 用户已要求搜索选模依据；调研见 [选模方法说明](docs/MODEL_SELECTION_REVIEW.md)。尚未批准或实施新的主选模规则，旧结果不重写。

## 6. 继续工作入口

协议 020（当前，按优先级）：

0. [离线未覆盖模式计划](指令/FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md)：当前执行顺序与"新类型"定义、数据隔离、成功判据。
1. [修订实验计划](指令/FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md)：R0–R6 方法与因果门禁规格。
2. [最新 HANDOFF](指令/FTMOE_PROTOCOL020_HANDOFF.md) 顶部：协作分工、路线与产物索引；[问题日志](指令/FTMOE_PROTOCOL020_PROBLEM_LOG.md)（P01–P36）、[R1 实验记录](指令/FTMOE_PROTOCOL020_R1_EXPERIMENT_20260909.md)、[R0 验收](指令/FTMOE_PROTOCOL020_R0_REVIEW_20260909.md)、[协议说明](docs/FTMOE_ONLINE_PROTOCOL_020.md)。
3. 机器结果：`artifacts/ftmoe_online/protocol_020/revision_20260909/r1/final_runs_20260909_1435/` 下的 `r1_analysis.json`、`r1_analysis.md`、`acceptance_primary.json`、`verification_results.json`、`parameter_counts.json`；R0 门禁 `revision_20260909/r0/verification_final.json`；续跑 `continuation/audit_final.json` 与 `continuation/comparison_tables.md`。

协议 014 归档（历史，仍然有效）：

4. [已验收版本与回档说明](docs/BASELINE_ACCEPTED_20260905.md)：完整快照、校验清单与恢复方法。后续实验不得覆盖此归档；启动下一实验须等待用户指示。
5. [现行实验规则](FTMOE_END_TO_END_ABLATION_PLAN.md)：比较关系、训练公平性、选模与测试边界。
6. [实验索引](FTMOE_END_TO_END_EXPERIMENT_LOG.md)：协议 001–014、失败原因与产物入口。
7. [历史复用审计](FTMOE_HISTORY_REUSE_AUDIT.md)、[五种子确认](FTMOE_HISTORICAL_CONFIRMATION.md)、[选模诊断](FTMOE_HISTORICAL_SELECTION_DIAGNOSTIC.md)。
8. 机器结果：`artifacts/ftmoe_end_to_end/comparison_014_complete.json`；图：`artifacts/ftmoe_end_to_end/figures/historical_confirmation_014.png`。
9. 旧冻结增量训练：[历史最终报告](FTMOE_ABLATION_FINAL_REPORT.md)。该报告不是当前全模型实验的成功证明；其原文、权重与测试结果保持原样。

协议 012 评估器 `evaluate_ftmoe_source_alignment.py` 需等待两个来源融合臂的训练及公共阈值校准均完成后使用。旧 `evaluate_ftmoe_end_to_end.py` 仍按五节点顺序链验收，不适用于最新分支消融，不可直接启动最终测试。

历史低学习率组 B 权重必须读取 `diagnosis_reassessment/physical_lr0003_e30` 指定的每轮 checkpoint，不能拿原 best.pt 代替。全部逐 epoch 权重保留，允许后续统一重评选模规则。

## 7. 运行环境与纪律

- 运行 Python：`D:\Anaconda\envs\dynmoe\python.exe`（CPU-only torch 2.4.1+cpu、numpy 1.19.2、Python 3.8）。仓库 `F:\PreGANPlus-master`，分支 `protocol-020`。
- **单进程顺序执行**：同一时刻只运行一个模型或模拟器进程（`run_ftmoe_online_stages.py` 在启动时会拒绝与其它实验 Python 进程并发），CPU 3 线程、interop 1、DataLoader 0、BelowNormal。
- **内存保护分两档，勿混用**：协议 020 在线运行沿用 019 的 **3.0 GiB** 可用内存 guard（`FTMOE019_RAM_GUARD_GIB`）；离线/场景采集链路（如 `prepare_ftmoe_scenario.py`）使用 **4.5 GiB** guard，并要求磁盘 ≥20 GiB。协议 017 确认运行正是撞在 4.5 GiB 这一档。
- 不得覆盖历史 predictions、summary、manifest、checkpoint 或 normalization；新产物写入协议自己的目录。
- git 客户端较旧（无 `switch` / `--show-current`，用 `checkout -b`）；远端不主动 push（019 惯例）。**R0/R1 的代码当前仍未提交**：`recovery/PreGANSrc/src/ftmoe_online_r1.py`、`run_ftmoe_protocol020_r1.py`、`analyze_ftmoe_protocol020_r1.py`、`test_ftmoe_protocol020_r1.py`、`verify_ftmoe_protocol020_r1_results.py`、`test_ftmoe_protocol020_r0.py` 均为 untracked（HEAD `8740da6`，branch `protocol-020`），另有 7 个已跟踪文件被修改未提交。
- Python 进程 stderr 的 torch 警告会使 pwsh 报 exit 1——以产物（`summary.json` / `failure.json` / `.log.exitcode`）为准，不要据 shell 退出码判定成败；同时注意 §1.5 中首个正式运行的退出码本身缺失。

## 8. 维护记录

维护与文档索引见 [文档索引](docs/README.md)。本文件在每次阶段状态变化后更新，且只写"产物可证明"的状态。

## 9. 本次订正记录（2026-09-10）

订正动因：本文件此前停留在 2026-09-05（协议 016/017 阶段），而 `artifacts/ftmoe_online/protocol_020/**` 与 `指令/FTMOE_PROTOCOL020_*.md` 已记录到 2026-09-09 的 R1 结果，入口文档与产物互相矛盾；`README.md` 顶部横幅亦仍停在协议 016。

本次改动（仅文档，**未运行任何实验、未修改模拟器或模型文件**）：

1. 本文件：加入协议 020 当前状态（§1）、订正 017/018 真实终态（§2）、更新继续工作入口（§6）与运行纪律（§7），并保留协议 014 归档内容（§3–§5）。
2. `artifacts/ftmoe_online/status.json`：由 015 小试的 `awaiting_user_discussion` 改为协议 020 的机器可读状态（该文件此前只被 `run_ftmoe_online_stages.py` 写入、无脚本读取；016 的独立状态在 `artifacts/ftmoe_online/adapted_bwgd2_016/status.json`，未改动）。
3. `docs/FTMOE_ONLINE_PROTOCOL_020.md` §5：原表停在 2026-09-07（"S4 停止、S5–S10 未开始"），与同文件头部 2026-09-09 状态行矛盾，按产物订正为真实阶段状态。
4. `README.md` 顶部横幅：改为指向协议 020 当前状态。
5. `docs/README.md`（文档索引）：017/018 行补注"两者均未完成"，新增协议 019/020 的入口行，并订正结尾停在 015 小试的状态句。

遗留矛盾（本次未自行决定，留待用户）：

- **子智能体口径**：本文件旧版（2026-09-05）记载"由当前助手独立执行，不创建子智能体"；2026-09-09 的 [HANDOFF](指令/FTMOE_PROTOCOL020_HANDOFF.md) 与 [R1 记录](指令/FTMOE_PROTOCOL020_R1_EXPERIMENT_20260909.md) 则登记了用户指定的分工：边界明确的实现/证据/检查交 `gpt-5.6-luna` 子 agent（推理强度 max），主 agent 负责设计、复核与验收。按"当前用户要求优先"，以 09-09 的口径为准，但请确认是否延续。
- **017 模拟器改动自记**：`capacity_017/confirmation/seed302_steps2000/failure.json` 含 `simulator_behavior_modified: true` 与"先向用户报告再改模拟器"的自我约束；该字段具体指哪些改动未在该文件内展开，需用户确认后再决定其归档措辞。
- **R0/R1 代码未提交**：见 §7，建议在进入下一步前提交并登记哈希，否则证据链依赖未跟踪的工作区文件。
