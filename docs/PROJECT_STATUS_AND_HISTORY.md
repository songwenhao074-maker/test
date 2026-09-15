# 2026-09-15 当前状态补充

Protocol-024 next_round_v1 已完成真实 A/C/D pilot（Actions 34953810201）。A AP=0.510442，C=D AP=0.667584；
D candidate_created=0，development_signal=false。本轮未启动动态专家拓扑变化，不能声称 D 优于 C。

当前入口：[结果审阅](PROTOCOL024_REVIEW_20260915.md) 与 [next_round_v2 指示](PROTOCOL024_NEXT_DIRECTIVE_20260915.md)。
本次审阅核验成功运行及 stdout 的比较结果，并发现 pressure/persistence 的可得时点错位；
未独立重算原 NPZ。确认种子仍未使用。下文保留历史记录，旧“尚无生命周期性能运行”等只描述当时状态。

---

# 项目现状与历史沿革

**当前进度补充（Protocol-024，2026-09-14）：** 已审阅 `674c626` 的 continuation 结果，15项单测通过，
追加 lifecycle-off 1140步/285更新 C-D 等价检查通过，覆盖825个原始异常host-step。尚无生命周期开启的性能结果。
新增发现：实际阶段指标仍继承旧 evaluator、保存函数仍指向023、拒绝候选后ID空缺恢复失败；休眠不释放总容量。
请先读 [最新审阅](PROTOCOL024_REVIEW_20260914.md) 与 [下一轮指示](PROTOCOL024_NEXT_DIRECTIVE_20260914.md)，
以下历史章节保留为历史记录。新场景拟注册 raw next-fault 预测任务，不能与旧tol1检测AP混表。

本文件是**唯一**的当前状态入口，取代此前的 `PROJECT_CONTEXT_LATEST.md` 与 6 份根目录历史报告。
日常只需读这一份 + [`docs/FTMOE_ONLINE_PROTOCOL_023.md`](FTMOE_ONLINE_PROTOCOL_023.md)（当前协议细节）。

**最后更新：2026-09-14**（协议 014 基线重训及归档完成；协议 023 第二轮 A 状态不变）

**2026-09-14 审计与交接补充（优先于下文旧结论概括）：** D 在 023 尚未实现或测试。
首次接触增益小，但再次接触的末段检测 AP 增益为 compute +0.0571、memory +0.0972、io +0.1297，
不能概括为全程“几乎不学习”。阶段 onset 存在展平时间轴错误，且倒数第二行漏结算；
旧完整性检查虽 PASS，并未覆盖这两个问题。详见 [最新复核](ONLINE_D_ANALYSIS_AND_NEXT_SCENARIO_20260914.md)。
新实验以寻找合理有利情形下 D 的实测优势为目标，可合理修改新场景并直接实现 D；旧门禁、数据和结果仍保留。
从 GitHub 接手先读 [交接说明](GITHUB_EXPERIMENT_HANDOFF.md)。以下历史章节按当时协议记录，勿据旧 STOP 门禁禁止新协议探索。

**归档重建补充：** 用户随后要求恢复 2026-09-05 的消融快照，并在确认无其他副本后授权按原配置重训。协议 014 的 30 个模型现已全部重训、保存 900 份逐轮权重并生成新 snapshot.zip，2,532 个归档文件通过哈希及实际恢复校验。108 项历史汇总指标一致，五个完整 v4 的权重张量与幸存原权重完全一致。当前协议 023 不变；详见 [归档重建说明](PROTOCOL014_RETRAINING_20260913.md)。这是基线重训快照，不是所有已删除历史文件的逐字节恢复。

---

## 1. 项目是什么

上游是 **PreGAN+**（Imperial College，BSD-3，作者 Shreshth Tuli）：面向移动边缘计算的
故障容忍框架。核心论点是「预测争用 → 定位具体资源 → 生成抢占式迁移决策」：用 GAN 预测
过载、few-shot 分类器诊断资源类型，从而在节点被打满前主动迁移任务。配套有一个 16 主机的
Raspberry-Pi 边缘模拟器（`simulator/`、`framework/`、`scheduler/`、`stats/`、`metrics/`）。

本仓库在上游之上附加了一条独立的**研究线：FT-MoE 在线学习协议**（`recovery/FTMoE*.py`、
`recovery/PreGANSrc/src/ftmoe_*.py`、根目录 `*_ftmoe_protocol023_*.py`）。它研究的问题是：

> 边缘环境的资源需求机制会漂移、切换并再现。**固定拓扑**的在线微调（fixed C）是否足够，
> 还是必须**动态增删专家**（D）？

历史上这条线一直试图证明 D 有必要，但**协议 023 给出了第一个反证**（见 §3）。

### 模型版本（消融链）

| 版本 | 相对前一级增加的计算 |
|---|---|
| v0 | 逐主机两层 Transformer、公共检测/分类头 |
| v1 | 普通 softmax MoE，4 专家 |
| v2 | 第二个 4 专家 EAGate 池，以及拟调度后的资源聚合输入 |
| v3 | 调度感知图编码；图特征残差与图预测 logits 相加 |
| v4 | 保留 v3，增加跨主机多头交叉注意力、同主机逐元素交互、归一化与输出 adapters |

v2−v1 不是纯路由替换，v4−v3 不是纯注意力消融；移除整套图扩展对应 v2。门控对照以同参数量
（16,640）的同主机门控替换注意力。

---

## 2. 当前状态：协议 023 第二轮 A 已完成，`D_eligible = false`

| 阶段 | 状态 |
|---|---|
| S0–S3（第一轮） | 完成。H0 **PASS**、H2 **PASS**、**H1 FAIL**（严格口径 24/36/31 < 50）、H3 **INADMISSIBLE**、H4 仅探针级 |
| S2.5 数据校准 | 完成。仅调 B/C 的 `cascade_task_probability` 与 io-first 的 `disk_retained_peak` |
| H1-v2 / H2-v2 | **PASS**。io-first 正例 12 → 31，三机制全部可测；prevalence spread 1.42 pp |
| S4 严格 prequential 固定 C | 完成。720 次更新 / 2880 区间，完整性六项检查全通过 |
| H3-v2（成熟学习器） | 可评估了，读出 **STOP-GI**：均值余弦 +0.71…+0.95，负比例 0.020…0.094 → **无梯度冲突** |
| §15 复现 | 冻结基线反而显示更大的 gap（+0.2121 vs B2 的 +0.1536） |
| **最终判定** | **`D_eligible = false`**，停止条件 **`STOP-NO-LEARN`** |

**最关键的一个数字**（固定 C 相对冻结基线 A 的 late-100 PR-AUC 增益）：

| regime | C | A | 增益 | 门槛 |
|---|---:|---:|---:|---:|
| compute_first | 0.4111 | 0.4062 | +0.0049 | +0.03 |
| memory_first | 0.5683 | 0.5663 | +0.0020 | +0.03 |
| io_first | 0.5946 | 0.5743 | +0.0203 | +0.03 |

**0/3 达标。** 在协议 022 的*离线*预算下 C 随更新继续增长；在本轮冻结的*在线*预算下几乎不动。
这是本协议族第一个**反对**需要动态专家的测量，且受限因素是预算而非机制。

固定 C 也没有遗忘（三个探针切片最差偏移 −0.0030，且单调改善），说明残差学到的东西
**不是 regime 专属的**。

### 尚未做的下一步（没有任何一项已开始）

1. **在线预算扫描**（2× / 4× 更新）——整个负结果压在这个冻结预算上，需确认平台期是否为预算假象。
2. **确认种子 701–703**——目前只用开发种子 700；io-first 仅以 1 个正例（31 vs 30）过线。
3. 若审阅者认为**事件计数**比 prevalence 更该匹配，需要新的 `cascade_v4` 响应时长律（不是校准）。
4. 是否接受 S2.5 的校准代价（事件数 +56–75%，换 5× 正例）。
5. §15 是否保留「冻结基线对照」这一额外标准（指令本身不要求；按字面规则该项会读作 PASS）。

**门禁纪律：** confirmation 种子 701–703 不得用于调参或选场景；不得重训或覆盖冻结 checkpoint。

详见 [`docs/FTMOE_ONLINE_PROTOCOL_023.md`](FTMOE_ONLINE_PROTOCOL_023.md)，机器可读状态见
`artifacts/ftmoe_online/protocol_023/gate_status.json` 与 `round2a/gate_status_round2a.json`。

---

## 3. 协议演进（结论已压缩，原始长文已于 2026-09-13 精简）

| 协议 | 问什么 | 结论 |
|---|---|---|
| 001–003 | 独立全模型训练；修复容器生命周期与 trace 绑定 | 001 旧数据保留；003 修复后重建重放，需求对齐检查通过 |
| 004 | 真实动作前主机输入、实际过载资源标签 | **当前物理数据基准**；`physical_lr0003_e30` 全组完成 |
| 005 | 修正资源诊断指标 | HR@100%/NDCG@100% 改为逐异常样本诊断；旧全局 top-100 降为补充 |
| 006–008 | 学习率 / 轮数 / 阈值校准 | 选定 lr=0.003；阈值校准未通过，不改变原最佳 epoch |
| 009–010 | MoE 上下文类型；专家 dropout | 上下文未采用；dropout=0.1 由单种子 pilot 选出 |
| 011 | 用户修订主链（v0/v1/v2/v4 主链，v3 补充） | 原始与校准结果均未达 0.005 门槛 |
| 012 | 同主机时间/图来源融合 | source_attention 三种子完成；配对 source_gated **未完成** |
| 013 | 回顾旧 30 轮低学习率候选 | 均值方向为正，融合检测增量不足 0.005 |
| 014 | 新增模型种子 17/42 | 12 个模型完成；五种子 B 主链 F1 = 0.525479 / 0.567550 / 0.901334 / **0.905134**；**未取得冻结测试资格** |
| 015–016 | 在线微调小试；BWGD2 输入适配 | 未达标（016 冻结 A 后半程 F1=0.143911）；预留测试未使用 |
| 017 / 018 | 容量扫描 / Google2011 | **两者均未完成**：017 确认运行在 1479/2000 步撞 4.5 GiB RAM guard 中止；018 数据未下载 |
| 019 | 搭建在线阶段基础设施 | 冻结登记、VM split、适配集；`s3/adapted_v4_seed1/best.pt` 沿用作后续起点 |
| 020 | 在线修正与动态专家（R0–R6） | R0-A 因果基础通过、**R0-B 动态部署未通过**；R1 固定修正减轻旧 C 漂移退化但**未稳定超过 A**；R2–R6 未实施 |
| 021 | 时序资源级联生成器（`cascade_v1`）+ 未见机制 | 数据门禁与 learnability 完成；未见机制未建立可学性 |
| 022 | 单机制未见 regime 的容量平台 | S5 容量诊断证明**单一 regime 下 C 的收益随预算继续增长、未建立容量平台** → 单机制下 D 无科学必要性 |
| **023** | 三异构机制 + 短驻留 + 再现 | **`D_eligible = false`**（见 §2） |

### 协议 014 的关键限定（历史最重要的一组数字）

| 模型 | B_best F1 | HR=NDCG |
|---|---:|---:|
| v0 | 0.525479 | 0.860048 |
| v1 | 0.567550 | 0.860922 |
| v2 | 0.901334 | 0.936297 |
| v3 简单融合 | 0.894437 | 0.962851 |
| v4 完整注意力 | 0.905134 | 0.972003 |
| v4 门控替换 | 0.903579 | 0.970907 |

v4 相比 v2、门控的 B F1 增量仅 0.003799 / 0.001554，均 < 0.005，条件区间跨零；新增种子
17/42 中门控检测均略高于完整注意力。**没有因结果不利而丢弃种子**；失败结果如实报告。

### 已确立的口径限制（不得只取最好看的指标）

- **HR@100% 与 NDCG@100% 在本数据上相等**（单资源标签），都等于真实异常样本上的资源 top-1
  准确率、包含检测漏报；因此 S=(F1+HR+NDCG)/3 实际给诊断两倍权重。不能混用旧全局 top-100 指标。
- 当前开发集异常以 CPU 类为主，**近乎恒定预测 CPU 的诊断准确率约 0.898224**；早期 checkpoint
  可在检测 F1 很差时靠此拿到高综合分（选模诊断已记录）。
- **无训练物理规则的开发基线 F1≈0.93057、诊断≈0.98194**，高于当时候选的完整模型。满足内部
  消融方向不能替代与此基线的比较。
- 论文参考 F1/HR/NDCG = 0.8766/0.6496/0.6021；数据、标签与指标实现有差异，**不能据数值更高
  声称复现成功或优于论文**。
- 单一底层 workload、多次开发集选模、类别不均衡均限制结论。开发集的置信区间不是独立确认。
- 预留测试 201–205 **从未生成**，全程保持封闭。

---

## 4. 运行环境与纪律

- **Python**：`D:\Anaconda\envs\dynmoe\python.exe`（Python 3.8.20、torch 2.4.1+cpu、numpy 1.19.2）。
  仓库 `F:\PreGANPlus-master`，当前分支 `protocol-023`。
- **单进程顺序执行**：同一时刻只运行一个模型或模拟器进程；CPU 3 线程、interop 1、
  DataLoader 0、BelowNormal 优先级。采集器会拒绝与其它实验进程并发。
- **内存保护分两档，勿混用**：在线运行（协议 019–023）用 **2.5–3.0 GiB** 可用内存 guard；
  离线/场景采集链路用 **4.5 GiB** guard，并要求磁盘 ≥20 GiB。协议 017 正是撞在 4.5 GiB 这档。
- **退出码**：Python 进程 stderr 的 torch warning 会让 pwsh 报 exit 1。**以产物为准**
  （`summary.json` / `failure.json` / `.log.exitcode`），不要据 shell 退出码判定成败；
  协议 023 的采集器直接捕获子进程真实返回码。
- **不得覆盖**历史 predictions、summary、manifest、checkpoint 或 normalization；新产物写入
  协议自己的目录。
- git 客户端较旧（2.15，无 `switch` / `--show-current`，用 `checkout -b`）；远端不主动 push。

### 不可再生的本地资产（**删除即永久丢失**，不在 git 中）

| 路径 | 大小 | 为什么不能删 |
|---|---:|---|
| `simulator/workload/datasets/bitbrain/rnd/1..499.csv` | 476 MB | 协议 023 采集器**硬门禁**：`prepare_ftmoe_protocol023_stream.py:670` 要求 1–499 全部存在，否则抛 `FileNotFoundError`。上游下载 URL 已失效，代码明确拒绝下载 |
| `recovery/PreGANSrc/checkpointsplus/*.ckpt` | 6.3 MB | `main.py -r ftmoe` 的推理起点 |
| `recovery/PreGANSrc/data/` | 76 MB | 离线训练/评测数据 |
| `artifacts/ftmoe_online/protocol_023/**` | 41 MB | 当前协议的注册与全部证据 |
| `artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt` | 1 MB | 协议 023 的冻结起点，hash `10c44bdb…` |

> 2026-09-13 精简时已按用户指示删除 `backup/`（014 回档快照 7.6 GB + 旧链 ckpt 1.36 GB）、
> `logs/`、`recovery/PreGANSrc/plots/`（2005 个 PDF）、`checkpoints_qos_overload_ms_tol1/`。
> 这些不在 git 中，**不可恢复**。014 阶段的历史结论保留在本文件 §3。

---

## 5. 文件地图

### 上游模拟器（保留，`main.py` 可用）

```bash
D:\Anaconda\envs\dynmoe\python.exe main.py -r ftmoe
```
`-r` 可选 `ftmoe`（默认，映射到 `FTMoEProgressiveRecovery`）、`preganplus`、`pregan`、`pcft`、
`dftm`、`eclb`、`cmodlb`。注意 `main.py` 在解析参数**之前**就 import 全部 9 个 recovery 模块
（`main.py:54-62`），因此 `recovery/*.py` 一个都不能删。

| 目录 | 作用 |
|---|---|
| `framework/` | 真实/虚拟化环境的执行框架（Docker、Ansible、InfluxDB） |
| `simulator/` | 16 主机 RPi 边缘模拟器；`simulator/workload/BitbrainWorkloadProtocol0{20,21,22,23}.py` 是本研究线的生成器链 |
| `scheduler/` | 放置/迁移调度器（GOBI、GA、DRL、HOGOBI…）；`scheduler/BaGTI/` 被 `GOBI` 以 `sys.path.append` 引入 |
| `stats/`、`metrics/` | 统计与资源/能耗模型 |
| `recovery/` | 故障恢复与预测模型；`recovery/PreGANSrc/src/ftmoe_*.py` 是 FT-MoE 实现 |

### 协议 023 实验链

| 阶段 | 入口 |
|---|---|
| 注册 | `register_ftmoe_protocol023.py` |
| 生成器 | `simulator/workload/BitbrainWorkloadProtocol023.py` |
| 审计仪器 | `ftmoe_protocol023_core.py` |
| 采集 / 运行 / 独立验证 | `prepare_ftmoe_protocol023_stream.py`、`run_ftmoe_protocol023_s2.py`、`verify_ftmoe_protocol023_stream.py` |
| 数据门禁 | `analyze_ftmoe_protocol023_s2.py` |
| 专业化 / 梯度 / 遗忘探针 | `probe_ftmoe_protocol023_{specialization,specialization_v2,gradient}.py` |
| 数据校准 | `calibrate_ftmoe_protocol023_s25.py`、`score_ftmoe_protocol023_s25_candidates.py`、`select_ftmoe_protocol023_s25_candidate.py` |
| S4 prequential | `run_ftmoe_protocol023_s4.py` |
| 判定汇总 | `analyze_ftmoe_protocol023_round2a{,_gates}.py`、`assemble_ftmoe_protocol023_round2a_status.py` |
| 测试 | `test_ftmoe_protocol023_{core,regimes,s2,gradient,s4_smoke}.py`（**206 个用例**） |

**跨协议硬依赖（名字带旧协议号但不可删）**：`ftmoe_protocol022_core.py`（被 8 个 023 脚本
import）、`probe_ftmoe_protocol022_learnability.py`（被 9 个 import）、`run_ftmoe_protocol020.py`、
`run_ftmoe_protocol019.py`、`run_ftmoe_online.py`、`analyze_ftmoe_online.py`、
`train_ftmoe_{end_to_end,ablation_existing}.py`、`train_ftmoe_protocol019_s3.py`。
另有数据依赖：`protocol_020/{vm_split.json, adapter/scenario_adapter.json, drift/drift_config.json,
disk_law_p20_ls.json, normalization_v2_time_scale.json, adaptation_data/v1}` 与
`adapted_bwgd2_016/disk_law.json`。

**回归基线**（精简后必须保持）：
```powershell
$env:PYTHONPATH="F:\PreGANPlus-master"
D:\Anaconda\envs\dynmoe\python.exe -m unittest test_ftmoe_protocol023_core test_ftmoe_protocol023_regimes test_ftmoe_protocol023_s2 test_ftmoe_protocol023_gradient
# 期望：66 + 28 + 41 + 71 = 206 tests, OK
```

### 文档

| 文件 | 内容 |
|---|---|
| `README.md` | 项目介绍与快速开始 |
| `docs/PROJECT_STATUS_AND_HISTORY.md` | **本文件** — 当前状态 + 精简后的历史 |
| `docs/FTMOE_ONLINE_PROTOCOL_023.md` | 协议 023 完整设计与两轮结果 |
