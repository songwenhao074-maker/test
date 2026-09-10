# Protocol 020 — 精简交接（Session Handoff，2026-09-09）

## 当前计划入口（2026-09-09，优先于以下历史交接）

- **协作分工（用户指定）**：边界明确的代码实现、证据整理和检查执行交给 `gpt-5.6-luna` 子 agent，推理强度 `max`。主 agent 负责设计任务、审查证据、验收修改、纠错与调整实验方向。子 agent 的完成报告不自动等于验收通过；模型/模拟器运行仍须单进程顺序执行，共享文件的写入范围须分开。
- **当前场景与执行顺序：[离线未覆盖模式计划](FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md)；方法规格：[修订实验计划](FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md)。** R0-A 因果基础已通过，R0-B 动态部署门禁未通过；R1 工程与 8 组开发对照已完成，稳定优于 A 尚未实现。下一项为完整离线覆盖审计与新模式注册，随后新模式小试，再进入 R2；新场景及 R2–R6 尚未实施。
- **场景纠偏**：S6 实际训练已包含 dev500 的正常/CPU/RAM/Disk 主要配置，dev501 是同域平稳流。不同 VM 与运行轨迹不等于离线未见类型；R1 未证明新模式学习无效。核查范围目前仅确认 S6 主要配置重合，原始 v4→019→020 的完整覆盖、初始锚点及选模暴露仍须审计。
- 目标保持 D 在运行中稳定、较高性能，相对 A/B/C 有可重复价值；原始离线 v4 与已有 019/020 起点均不重训、不覆盖。模拟器和在线方法可以在开发期调整，最终确认来源不得用于选择。
- 路线：已完成 R0-A/R1 → 离线覆盖审计与新模式注册 → 新模式 A/C 小试 → R2 损失/采样 → R3 容量/长期记忆 → R0-B 修复与 R4 动态独立收益 → R5 扩展场景族 → R6 独立确认。旧 dev500/dev501 保留回归用途，暂停旧两流立即 8 组调参。
- **R1 固定在线修正完成**：[R1 实验记录与结果](FTMOE_PROTOCOL020_R1_EXPERIMENT_20260909.md)。16 项测试、106 项独立结果检查、两条流 off/on 的 learner/optimizer/抽样一致性通过。正式 8×2000 结果仅取 `revision_20260909/r1/final_runs_20260909_1435/`；早期失败及中止目录保留，不混入对照。
- **R1 结论**：在现有熟悉模式及其切换上，冻结基础＋小修正减轻旧 C 的漂移退化，保护减少伤害，但尚未稳定超过 A，也未全面优于旧 C。漂移 protected F1/AP 为 0.5091/0.6769，A 为 0.5118/0.6779；平稳 protected 为 0.1053/0.1285，A 为 0.1053/0.1270。漂移修正参与 85.85%、6 次快照准入、1 次回退；不是全程使用 A。权重/采样 2×2 消融保留为 R2，但须先完成新模式审计和小试，再以新模式为主评估；训练权重与准入评分分开冻结。
- R0 的 `prequential_v1` 修复了旧 S8 验证顺序，必须显式指定；默认 `legacy_v3` 保留历史行为。R1 使用独立 `run_ftmoe_protocol020_r1.py` 与 `r1_fixed_residual_v1`，不能把 R0/R1/legacy 结果或恢复文件混用。
- 23 项旧回归＋12 项新 R0 测试通过；主 agent 对新测试进行了原始输出独立复核。runner 恢复检查修复 NumPy/Python RNG 初始化后 39/39 通过。总证据：`artifacts/ftmoe_online/protocol_020/revision_20260909/r0/verification_final.json`。
- 新增合成诊断确认 ramp 接近 1 到等于 1 时会发生 Top-4 硬切换。R0-B 尚未通过，不启动 D 正式效果矩阵。完整错误样本培养、重新验证唤醒与贡献评估仍待 R4；新回退机制需在实现后补测试。
- 历史 12 次运行、模型哈希和整体性能结论保留。旧“候选通过”按旧规则解释；不能称为双方共同未见数据上的收益。
- 对照采用 A/B-legacy/C-legacy/D-legacy 与 C-residual/D-residual，并加入参数量、训练预算控制；新方法不能不改名字就与旧 C/D 混比。
- 按修订计划在新目录 `artifacts/ftmoe_online/protocol_020/revision_20260909/` 保存后续产物。继续单进程顺序运行，3.0 GiB 内存保护，不直接启动 S10。

## 历史续跑结果（2026-09-08；方法有效性以 09-09 更正为准）

- 用户已授权继续调整在线机制及模拟器，目标是 D 稳定优于 A/B/C；原始离线 v4 不得重训。见 `指令/FTMOE_PROTOCOL020_CONTINUATION_20260908.md`。本轮没有重训任何离线模型，原始 v4、019 和 020 起点哈希全部保持一致。
- **S8 容器级集成已完成**：`recovery/PreGANSrc/src/ftmoe_online_s8.py`；影子试训、后续成熟数据验证、零权重启用、渐进参与、休眠/唤醒及恢复已实现。D 可通过 `--dynamic-config` 运行。
- **完成 12 次新的 2000 步开发运行**，目录 `artifacts/ftmoe_online/protocol_020/runs_continuation/`；共用 dev500 漂移和 dev501 平稳流。D 在漂移流真实启用第五个专家：1e-4 设置为 step660，1e-5 设置为 step1900。平稳流没有新增专家。真实运行尚未出现休眠/唤醒。
- **D 尚未稳定胜过 A/B/C**。提高学习率会加重误报；检测正类权重从约 8 改为 1 后 F1 改善，但召回、PR-AUC 下降，且该设置下 D 与 C 预测相同，无动态收益。
- 修复 S7 恢复遗漏 state；保留能力检查改用 P20 锚点完整输入；修复 lag 返回绝对时间、跨相位搜索、滚动面积归一化问题。旧结果保留，新旧 reference 不能混比。
- 详细结论与下一步计划：**`指令/FTMOE_PROTOCOL020_D_REVIEW_AND_NEXT_PLAN.md`**。
- 机器可读审计：`artifacts/ftmoe_online/protocol_020/continuation/audit_final.json`；对照表 `comparison_tables.md`；冻结模型哈希 `protected_checkpoint_hashes.json`。
- **下一步建议**：先做冻结基础预测＋在线修正的保留能力保护；为 C/D 同时提供容量感知特征；再优化按持续错误模式培养专家；最后扩展阶段长度和任务分布，并在未参与调整的 VM 来源上确认。上述下一步尚未实施；S9 仅完成初步开发小试，S10 未开始。
- 本轮结束时没有留下训练或采集进程。
- 验证：23 项测试通过；旧 C 重跑以及未启用新专家的 D/C 对照均逐元素复核。详见 `continuation/verification.json`。

### 历史 D 复现示例（非修订方案启动命令）

```text
D:\Anaconda\envs\dynmoe\python.exe run_ftmoe_protocol020.py --method D --model-seed 1 --checkpoint-path artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt --stream artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed500_steps2000 --output <新的输出目录> --base-lr 1e-5 --dynamic-config artifacts/ftmoe_online/protocol_020/continuation/dynamic_v3_lr1e5.json
```

以下为本次续跑之前的历史交接，保留追溯。

> 新对话请以此文件为主上下文；完整历史见 `指令/FTMOE_PROTOCOL020_PROBLEM_LOG.md`（P01–P23）与
> `docs/FTMOE_ONLINE_PROTOCOL_020.md`。仓库 = `F:\PreGANPlus-master`，分支 `protocol-020`
> （旧版 git：用 `checkout -b`，无 switch）。Python = `D:\Anaconda\envs\dynmoe\python.exe`
> （CPU-only torch 2.4.1，内存整机 15.2G/可用 ~4G，**所有采集/运行单进程顺序执行**，
> guard 3.0 GiB；勿并行）。

## 1. 目标与纪律
按 `指令\FTMOE_PROTOCOL020_DETAILED_SOLUTION_PLAN.md` 分层执行：
模拟器/数据层(S1–S5) → 同域适配(S6) → 在线 A/B/C(S7) → Dynamic v3(S8) → 小试/矩阵(S9/S10)。
**任一阶段门禁不过即停并记录**；已发生的方法学让步全部登记在问题日志（见 §4）。

## 2. 阶段状态（截至交接）
| 阶段 | 状态 |
|---|---|
| S0–S3 | ✅ 完成：019 冻结登记 / RAM_CAP_SCALE / graph v3(per-sample capacity+before_placement) / VM source-disjoint split（train277/dev86/online94） |
| S4 | ✅ 完成（大量探测后定稿场景；~75 data-only 候选 + disk-law 手术） |
| S5 | ✅ **PASS**：dev drift 流 v4（`drift_streams/dev_seed500_steps2000`）全相位 dominance + dep 18.7%/mig 30.9% |
| S6 | ✅ 完成：同域适配 `s6/adapted_v4_seed1/best.pt`（epoch4；dev PR-AUC 0.526/macroF1 0.395）；数据 `adaptation_data/v1`（train12/dev8） |
| S7-stationary | ✅ PASS：`runs/*_seed501` A/B/C 2000 步（无遗忘、C PR≥A−0.02、采样零违规）；结果 `S7_stationary_result.json` |
| S7-drift | ⚠️ 完成 A/C：**C ≈ A，无可测 adaptation**（`runs/*_seed500_drift` + `S7_drift_AC_result.json`）；§20.2 建议“先修 optimizer 再启 D” |
| S8 | 触发逻辑已实现（`recovery/PreGANSrc/src/ftmoe_dynamic_expert_v3.py`，测试 7/7）；**容器级集成未做** |
| S9/S10 | 未开始 |

## 3. 关键产物索引
- 场景/数据契约：
  - `artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json`（v4：arrival 1.0、rm 2.0、ru None、disk-law ls）
  - `.../drift/drift_config.json`（v4 五相位：baseline/cpu/ram/disk/recurrence，**相位级 adapter**：cpu 类 ram_upper 1400、ram 相位 0.55+ru2400、disk 0.35）
  - `.../stationary/stationary_config.json`、`.../vm_split.json`、`.../disk_law_p20_{v1,v2,lm,ls}.json`、`.../normalization_v2_time_scale.json`
- 流：`drift_streams/dev_seed500_steps2000`（漂移 v4 最终）、`drift_streams/dev_seed501_steps2000`（stationary 最终）
- 运行：`runs/{A,B,C}_model1_seed501`（stationary）、`runs/{A,C}_model1_seed500_drift`
- 结果：`S7_stationary_result.json`、`S7_drift_AC_result.json`、`capacity_scan/capacity_scan_report.json`
- 代码（新/改，均已提交）：root 下 `prepare/analyze/build/run/train_ftmoe_protocol020*.py`、
  `test_ftmoe_protocol020_{simulator,graph,s4,online,dynamic_v3}.py`、`simulator/environment/RPiCapacity.py`、
  `simulator/workload/BitbrainWorkloadProtocol020.py`（adapter knobs：cpu_upper/cpu_mult/ram_mult/ram_upper/disk_mult + set_adapter）、
  `recovery/PreGANSrc/src/ftmoe_ablation.py`（graph v3）、`ftmoe_online_s7.py`（S7 session）、`ftmoe_dynamic_expert_v3.py`（v3 trigger）

## 4. 已登记的门禁/方法学让步（新对话勿擅自再改）
- 相位主导下限（per 8000，按实际相位 horizon 缩放）：**CPU 25 / RAM 100 / Disk 60**（CPU 弱模式）
- 部署拒绝 **≤25%**（原 20）、迁移拒绝 ≤40%、share：CPU 类相位 0.45 / 其他 0.50
- **允许相位级需求画像**（用户 2026-09-08 批准 α）：相位间同时改变容量与需求 = **covariate+fault 混合漂移**，报告中必须如实声明（019 教训回归）
- disk 引擎 = 专用 law-ls（初始截断≤4000 + 上行增长）；CPU 类相位 ru1400 消除满容量 RAM 噪声；RAM 相位保留 tail（0.55+ru2400）
- 关键机制记录：RAM 过载≈(ram_mult/ram_scale) 比率驱动；GOBI 对 RAM/Disk 容量盲视；容量过深→部署墙（事件反降）；CPU 相位受“大 RAM tail 容器噪声”结构性限制（P21）

## 5. 待用户决策（下一步）
S7-drift 已显示 **C≈A（无可测 adaptation）**；选项：
(a) 接受结论收尾：输出 Protocol 020 完整报告（数据层重建成功 + 在线/动态无增益负面结论 + 让步声明）——推荐；
(b) 继续 S8：先实现 OnlineEAGateV3 容器级（shadow/birth/ramp/retire/reactivate，参照 `ftmoe_online_s5.py` v2 结构 + v3 trigger 注入 matured-loss/novelty EMA）→ D 运行，预期大概率 D−C≈0；
(c) 暂停。

## 6. 常用命令
```text
python analyze_ftmoe_protocol020_drift.py <stream_dir>            # S5 相位门禁
python analyze_ftmoe_protocol020.py --stream <s> --runs A=<d>,C=<d> # 在线分析(相位/lag/面积)
python run_ftmoe_protocol020.py --method A|B|C --model-seed 1 \
  --checkpoint-path artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt \
  --stream <stream_dir> --output <runs_dir>
python train_ftmoe_protocol020_samedomain.py --model-seed 1        # S6 重训(可选)
```
注意：python 进程 stderr 的 torch 警告会使 pwsh 报 exit 1——以产物（summary.json/failure.json）为准；
`run_ftmoe_protocol020.py` D 方法会拒绝运行（需 S8 gate）。
