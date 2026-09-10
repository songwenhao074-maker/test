# Protocol 020 — Capacity-Driven Multi-Fault Drift + Dynamic Expert v3

> 状态（2026-09-09）：**R0-A 因果基础已通过；R0-B 动态部署未通过。R1 固定在线修正完成 8×2000 步开发对照、16 项测试及 106 项独立结果检查。新结构减轻漂移退化，但未稳定超过 A；D 优势仍未实现。R2–R6 未实施，S10 未开始。**
> 当前执行顺序以 [离线未覆盖模式计划](../指令/FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md) 为准，方法规格见 [修订实验计划](../指令/FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md)，验收与结果见 [R1 实验记录](../指令/FTMOE_PROTOCOL020_R1_EXPERIMENT_20260909.md)。S6 已训练在线旧流的主要模式，现有结果不足以评估离线未见类型学习。下一项先审计完整离线覆盖并注册新模式，完成新模式小试后再进入 R2；旧流保留回归用途。新场景尚未采集。原始离线 v4 及已有起点不重训；动态效果矩阵仍受 R0-B 门禁约束。最新交接为 `指令/FTMOE_PROTOCOL020_HANDOFF.md` 顶部。
> 以下 *The Plan* 指原始 `指令/FTMOE_PROTOCOL020_DETAILED_SOLUTION_PLAN.md`，其初始门禁和旧阶段表保留历史口径，不是当前执行指令。
> 仓库：`songwenhao074-maker/FT-MoE` @ `F:\PreGANPlus-master`（git 分支 protocol-020，自 protocol-019 分出）
> 当前纪律：开发调整在用户授权内执行并保留失败记录；有效性问题先修后评。允许开发期模型反馈调整场景，必须标记已使用来源；最终确认数据不得用于选配置。具体对照与推进条件见修订计划。

## 0. 协议身份与版本登记

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

登记文件：`artifacts/ftmoe_online/protocol_020/protocol.json`

## 1. 不可修改的既有资产（immutable，Protocol 019 全量冻结）

- `docs/FTMOE_ONLINE_PROTOCOL_019.md`、`指令/FTMOE_ONLINE_TUNING_REVIEW_AND_SOLUTION_PLAN.md`、`指令/FTMOE_PROTOCOL019_PROBLEM_LOG.md`
- `artifacts/ftmoe_online/protocol_019/` 全部内容（S2–S7 result JSON、stream.npz、checkpoint、problem log、source hash）
- protocol-019 git 分支（本地 commit `57e856d`，仅读取）
- 旧模拟器/训练脚本：`prepare_ftmoe_online.py`、`prepare_ftmoe_adapted_bwgd2.py`、`prepare_ftmoe_scenario.py`、
  `prepare_ftmoe_protocol019_*.py`、`run_ftmoe_online.py`、`run_ftmoe_protocol019.py`、`run_ftmoe_protocol019_s4.py` 等
  一律不改；Protocol 020 全部新增/修改走本协议自己的文件（见 §6）。

## 2. 阶段流程（The Plan §2 / §45，任一阶段失败即停止后续阶段）

```text
P20-S0  冻结 Protocol 019，登记新协议                        <- 本文档
P20-S1  模拟器 capacity control（RAM_CAP_SCALE）+ 数据记录（overload_ratio/mask）
P20-S2  Graph source semantics v3（before_placement -> proposed dst）+ per-sample capacity
P20-S3  VM cohort / source-disjoint split（vm_split.json）
P20-S4  Data-only capacity scan（CPU / RAM / Disk 单因素，不加载模型）
P20-S5  构建真正的 CPU→RAM→Disk→CPU fault-mode drift（每 phase 真实 dominant fault）
P20-S6  Same-domain offline adaptation / cold-start compatibility
P20-S7  Online replay + loss v3（rare-event sampling），先跑 A/B/C
P20-S8  Dynamic Expert v3
P20-S9  Stationary + controlled drift 小试
P20-S10 5 model seeds × 3 source-disjoint streams
```

## 3. Gate 定义（The Plan §11 / §13 / §20 / §23，预先固定）

- Data gate（S4 scan 候选）：normal 80%–95%；每类 dominant fault ≥1%（≥150 host-step @2000×16）；
  每类 ≥30 独立事件；deployment rejection <20%；migration rejection <40%。
- Drift phase gate（S5）：每目标 phase dominant fault count ≥100 且 target > 其余两类，
  phase 名称只能由真实 dominant fault 命名。
- Stationary gate（S9）：D expert additions ≤1；anchor F1 drop ≤0.03；C PR-AUC ≥ A PR-AUC − 0.02。
- Drift gate（S9）：C 出现可测 adaptation；D 至少发生一次真实 topology event（否则不得解释为
  “动态专家无效”）。
- 最终确认（§37）：S5/S7/S8 全部通过后才进入 5×3 矩阵；统计单位 = model seed × online stream。

## 4. 环境与已知约束（S0 登记，2026-09-07）

- torch 全为 CPU-only（dynmoe env：torch 2.4.1+cpu / numpy 1.19.2 / Python 3.8；另见 019 日志 P02）。
- 内存 ~15.2 GiB 总量、可用常年 ~4 GiB；沿用 019 的 3.0 GiB RAM guard（`FTMOE019_RAM_GUARD_GIB`）。
- 本地 Bitbrain 数据集 500 个 VM CSV 在位；GOBI energy_latency_16 ckpt 在位。
- git 客户端较旧（无 switch/show-current），远端不主动 push（019 惯例）。

## 5. 阶段状态

| 阶段 | 状态 | 产出 | 日期 |
|---|---|---|---|
| S0 | 完成 | 本文档 + protocol.json + 问题日志（指令/） | 2026-09-07 |
| S1 | 完成（RAM_CAP_SCALE，Test 1/2 等 5/5 通过；overload_mask/ratio 随 P20 采集器在 S4 工具中落地） | RPiEdge RAM_CAP_SCALE + test_ftmoe_protocol020_simulator.py | 2026-09-07 |
| S2 | 完成（graph semantics v3 + per-sample capacities；Test 3–6 等 7/7 通过；019 legacy 回归 8/8、9/9 无退化） | ftmoe_ablation.py（ScheduleGraphEncoder v3）+ test_ftmoe_protocol020_graph.py | 2026-09-07 |
| S3 | 完成（eligible 457 VM：train 277 / dev 86 / online 94；Test 9 不相交断言通过） | build_ftmoe_protocol020_vm_split.py + vm_split.json + BitbrainWorkloadProtocol020.py | 2026-09-07 |
| S4 | **曾按 P16 正式停止**（2026-09-07：31 候选全量 data-only 扫描无一双门禁通过，机制=深度容量触发部署拒绝墙、中等容量事件率不足）；用户 2026-09-08 授权方向 (b) 后重做并完成（~75 data-only 候选 + disk-law 手术） | capacity_scan/（含 capacity_scan_dev/）+ capacity_scan_report.json + P16–P23 问题日志 | 2026-09-08 |
| S5 | **PASS**（dev drift v4 全相位 dominance：dep 18.7% ≤25、mig 30.9% ≤40；cpu 43/100%、ram 173/97.2%、disk 226/90.4%、recurrence 40/88.9%、baseline 干净） | drift_streams/dev_seed500_steps2000 + P23 记录 | 2026-09-08 |
| S6 | 完成：同域适配 `s6/adapted_v4_seed1/best.pt`（epoch4；dev PR-AUC 0.526 / macroF1 0.395）；数据 `adaptation_data/v1`（train12/dev8） | s6/ + adaptation_data/v1 | 2026-09-08 |
| S7-stationary | **PASS**：A/B/C 各 2000 步（无遗忘、C PR-AUC ≥ A−0.02、采样零违规） | S7_stationary_result.json、runs/*_seed501 | 2026-09-08 |
| S7-drift | 完成 A/C：**C ≈ A，无可测 adaptation**（§20.2 建议先修 optimizer 再启 D） | S7_drift_AC_result.json、runs/*_seed500_drift | 2026-09-08 |
| S8 | 触发逻辑 v3 完成（ftmoe_dynamic_expert_v3.py，测试 7/7）；容器级集成在续跑中以 `ftmoe_online_s8.py` 完成，12 次新 2000 步运行显示 **D 未稳定胜过 A/B/C**，真实休眠/唤醒未观察到 | ftmoe_online_s8.py、runs_continuation/、continuation/audit_final.json | 2026-09-08 |
| S9 | 初步开发小试完成（未达"D 稳定优于 A/B/C"目标） | continuation/comparison_tables.md、continuation/audit_final.json | 2026-09-08 |
| S10 | 未开始（R0-B 未通过前不得启动） | — | — |
| R0（修订） | **R0-A 因果基础通过；R0-B 动态部署未通过**（23 项旧回归 + 12 项新测试通过、runner 复验 39/39；ramp 连续性 `false`） | revision_20260909/r0/verification_final.json | 2026-09-09 |
| R1（修订） | 工程与 8×2000 开发对照完成（16 项测试、106/106 独立检查通过），**尚未稳定超过 A** | revision_20260909/r1/final_runs_20260909_1435/ | 2026-09-09 |
| 下一步 | 离线覆盖审计（`offline_coverage_audit`）与新模式注册（`unseen_regime_registry`）**未开始**，随后新模式小试，再进入 R2；R2–R6 未实施 | 指令/FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md | 2026-09-09 |

> 上表于 2026-09-10 依产物订正：原表停在 2026-09-07 的 "S4 停止、S5–S10 未开始"，与本文件头部 2026-09-09 状态行及 `artifacts/ftmoe_online/protocol_020/` 产物矛盾。S0–S3 行未改动。当前执行顺序、场景纠偏（P36）与门禁以头部链接的《离线未覆盖模式计划》《修订实验计划》为准。

## 6. Protocol 020 专属文件清单（推荐路径，按需新增）

- 数据/运行：`prepare_ftmoe_protocol020_capacity_scan.py`、`analyze_ftmoe_protocol020_capacity_scan.py`、
  `prepare_ftmoe_protocol020_drift.py`、`build_ftmoe_protocol020_adaptation_dataset.py`、
  `train_ftmoe_protocol020_samedomain.py`、`run_ftmoe_protocol020.py`、`analyze_ftmoe_protocol020.py`
- 代码：`recovery/PreGANSrc/src/ftmoe_online_s7.py`、`recovery/PreGANSrc/src/ftmoe_dynamic_expert_v3.py`
- 模拟器：`simulator/workload/BitbrainWorkloadProtocol020.py`
- 测试：`test_ftmoe_protocol020_simulator.py`、`test_ftmoe_protocol020_graph.py`、
  `test_ftmoe_protocol020_online.py`、`test_ftmoe_protocol020_dynamic_v3.py`
- 修改（仅列出的两处 + Replay）：`simulator/environment/RPiEdge.py`（RAM_CAP_SCALE）、
  `recovery/PreGANSrc/src/ftmoe_ablation.py`（graph semantics v3 / per-sample capacity）
- 产物：`artifacts/ftmoe_online/protocol_020/`（capacity_scan/、vm_split.json、drift_streams/、
  adaptation_data/、runs/、result json）
