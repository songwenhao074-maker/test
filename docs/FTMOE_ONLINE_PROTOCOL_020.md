# Protocol 020 — Capacity-Driven Multi-Fault Drift + Dynamic Expert v3

> 状态：**执行中（S0 登记完成；S1 进行中）**
> 依据：`指令/FTMOE_PROTOCOL020_DETAILED_SOLUTION_PLAN.md`（下称 *The Plan*）
> 仓库：`songwenhao074-maker/FT-MoE` @ `F:\PreGANPlus-master`（git 分支 protocol-020，自 protocol-019 分出）
> 纪律：**先让数据场景本身满足实验条件，再运行模型；禁止根据模型 F1 / D-C / D-B 结果反向调整模拟器。任一阶段失败 → 停止后续阶段、记录原因（指令/FTMOE_PROTOCOL020_PROBLEM_LOG.md）。**

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
| S1 | 进行中 | RPiEdge RAM_CAP_SCALE（Test 1/2）+ test_ftmoe_protocol020_simulator.py | 2026-09-07 |
| S2 | 未开始 | — | — |
| S3 | 未开始 | — | — |
| S4 | 未开始 | — | — |
| S5 | 未开始 | — | — |
| S6 | 未开始 | — | — |
| S7 | 未开始 | — | — |
| S8 | 未开始 | — | — |
| S9/S10 | 未开始 | — | — |

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
