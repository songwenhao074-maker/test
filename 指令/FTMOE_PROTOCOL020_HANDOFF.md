# Protocol 020 — 精简交接（Session Handoff，2026-09-08）

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
