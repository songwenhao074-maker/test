## 会话交接（2026-09-07 留档，用户关机，下次继续）

**当前状态**：S0–S3 完成（代码已提交）；S4 data-only 全扫描 31 候选完成 → **P16 正式停止条件**
（无候选满足相位主导×拒绝率双门禁），已停止 S5+；方向选项待用户拍板（P16 详细记录见下）。

**下次恢复命令**（工作目录 F:\PreGANPlus-master，Python = D:\Anaconda\envs\dynmoe\python.exe，
内存约束 3.0 GiB guard，勿并行多个采集进程）：
- 回顾：`git log --oneline -12`；`docs/FTMOE_ONLINE_PROTOCOL_020.md`；本日志 P13–P16。
- S4 报告：`python analyze_ftmoe_protocol020_capacity_scan.py`（读 capacity_scan/capacity_scan_report.json）。
- 待用户选方向（P16 选项 a/b/c/d）→ 选定后按纪律先记录到本日志再执行：
  (b) 需改 `simulator/workload/BitbrainWorkloadProtocol020.py` 的 demand adapter（新协议文件，
  不动 019）；(a) 需重登记 §13 阈值；(c) 需重跑 `build_ftmoe_protocol020_vm_split.py`；
  (d) 收尾存档。
- 若走通 S4→S5 已备好工具：`prepare_ftmoe_protocol020_drift.py --config drift/drift_config.json`
  （dev seed500，5 相位）+ `analyze_ftmoe_protocol020_drift.py`；
  S6：`prepare_ftmoe_protocol020_adaptation_episodes.py` → `build_ftmoe_protocol020_adaptation_dataset.py`
  → `train_ftmoe_protocol020_samedomain.py --model-seed 1`；
  S7：`run_ftmoe_protocol020.py --method A/B/C`（已含 loss v3 与分层采样，测试 4/4）；
  S8 触发逻辑已实现（测试 7/7），容器级 v3 集成待 S7 通过后继续。
- 归一化注意：P20 工件 `artifacts/ftmoe_online/protocol_020/normalization_v2_time_scale.json`
  由 `build_ftmoe_protocol020_adaptation_dataset.py` 生成（基于 019 工件 + train-cohort 覆盖），
  S6 checkpoint 与 P20 runner 均须与其一致。

---

# FT-MoE Protocol 020 — 实验执行问题日志（Problem Log）

> 本文件与 `FTMOE_PROTOCOL020_DETAILED_SOLUTION_PLAN.md` 同目录（`F:\PreGANPlus-master\指令\`）。
> 用途：记录尝试执行 Protocol 020 过程中遇到的所有问题、失败、停止条件、配置与 hash，供后续修改与复盘。
> 规则：**任何阶段失败 → 停止后续阶段并记录**；记录后不改写历史条目，只追加（最新在最上）。

---

## 问题汇总表（最新在上）

| # | 阶段 | 严重度 | 状态 | 摘要 |
|---|---|---|---|---|
| P18 | S4(方向b) | 高 | 已记录（(b) 探测穷尽，等用户决策） | (b) 21 个额外探测后：RAM 可构造（rm2.5×scale0.6 → 166/8000，dep 19.3% ✓）；CPU 机制上限 ≈60/8000（am2×scale0.6，dep 17.3% ✓，GOBI 均衡约束）；Disk 拒绝门内仅 ~15–20/8000（更深容量一律触发部署墙 60–98%）→ 三资源同达 §13（≥100/8000）在 demand/到达率杠杆内不可行；待选：按实测重登记非对称下限 / 继续 disk-law 剖面手术 / 接受负面结论 |
| P17 | S4(恢复) | 中 | 已登记方向 (b)，执行中（v1/v2 探测完成，P18 判定） | 用户"继续任务"→ 方向 (b)：P20 专用 demand adapter（cpu_upper/ram_mult/disk_mult/arrival_mean），默认=016 契约；探测 21 候选 data-only |
| P16 | S4 | 高 | **已记录（正式停止条件）→ 2026-09-08 用户指示继续，方向 (b) 执行中** | 31 候选全量 data-only 扫描：CPU/RAM/Disk 三轴**无任何候选同时满足 §13 相位主导（≥100/8000 host-step）与 §11 拒绝率（dep<20%/mig<40%）**；极深容量触发部署拒绝墙（97%+）且事件反降，中等容量事件率不足 → 按纪律停止，用户拍板方向 (b) 恢复 |
| P15 | S4 | 中 | 已执行（Extension 1 全量跑完） | 预登记网格频率不足 → 数据驱动扩展网格（cpu 0.65–0.40、ram 0.25–0.10、disk 0.15/0.12），全部跑完并纳入 P16 判定 |
| P14 | S4 | 中 | 已修复（一次性线程配置 flag） | 进程内多候选采集时每个候选重复调用 `torch.set_num_interop_threads(1)` → 第 2 个候选 RuntimeError（019 P07 同类）；cpu 轴仅产出 cpu_0.7 一个候选即崩；三个采集器（scan/drift/adaptation）均已加进程级一次性配置保护，重启扫描 |
| P13 | S4 | 中 | 已处理（改为顺序运行） | 3 个并行扫描进程使可用内存跌破 3.0 GiB RAM guard（2.77/2.76 GiB），cpu/ram/disk 首候选约 t=200–250 中止；已清理失败残留并改为单进程顺序重跑 |
| — | S0/S1 | — | 进行中 | 2026-09-07 登记 Protocol 020（分支 protocol-020）；S1 RAM_CAP_SCALE 已实现，测试待跑 |

---

## 详细记录

### 2026-09-08 — P16 恢复：用户指示"继续任务"，登记方向 (b)

- 用户返回并指示"继续任务"（未逐字选择 a/b/c/d）；按推荐路径登记为 **方向 (b)：P20 专用
  demand adapter 加深需求/占用（pure scenario/data-layer 改动，不碰模型与门禁）**。
  若 (b) 探测后仍无法构造达标场景，将记录并再次请示 (c)/(a)/(d)。
- 实现原则：只扩展 Protocol 020 自己的 workload 层（`BitbrainWorkloadProtocol020.py` 可选
  adapter 参数：cpu_upper / ram_mult / disk_mult），016/019 契约文件一律不动；默认参数 =
  016 契约原值（cpu clip ≤1860、ram×2、disk×1）；探测值逐候选登记于 manifest["adapter"]。
- 探测 v1 候选集（400 interval，data-only）：CPU (cpu_upper, scale) ∈ {(2200,0.7),(2600,0.7),
  (2600,0.8)}；RAM (ram_mult, scale) ∈ {(2.5,0.5),(3.0,0.5),(2.5,0.6),(3.0,0.6)}；
  Disk (disk_mult, scale) ∈ {(1.5,0.4),(2.0,0.4),(2.0,0.5)}——目标：相位投影 ≥100/8000
  且 dep-rej<20%/mig-rej<40%。
- **v1 结果（2026-09-08）**：
  - **关键机制**：RAM/disk 过载由"有效需求/容量比率"驱动（demand_mult / cap_scale 之比不变
    时流完全等价：ram_0.5_rm2.5 ≡ ram_0.6_rm3 逐位同构，已用 stream hash + 全量统计核对）。
  - **RAM 候选通过两臂**：ram_mult 2.5 × RAM scale 0.6（比率 4.17）→ RAM 133/6400
    （8000 投影 166 ≥100）、事件 42、RAM 占比 97.8%、dep-rej 19.3% ✓、mig-rej 20.9% ✓。
  - **CPU 否定**：提高 clip 上限无效/反效果（au2200@0.7: 25；au2600@0.7: 13；au2600@0.8: 5；
    高需求容器被 GOBI 更分散放置）→ 转向**占用率杠杆（到达率 arrival_mean）**。
  - disk 0.4/0.5 值不在磁盘网格 → 未跑（v1 命令空转），已登记磁盘网格扩展 {0.35,0.45,0.5}。
- **v2 探测（进行中）**：CPU (arrival_mean, scale) ∈ {(1.6,0.8),(2.0,0.8),(2.0,0.75),
  (2.5,0.9)}（默认 adapter）；Disk (disk_mult, scale) ∈ {(1.5,0.35),(2.0,0.35),(2.0,0.45),
  (1.5,0.30)}（比率 4.3–5.7 区间）。
- **P18 结论与用户决策（2026-09-08）**：21 个 (b) 探测后：RAM 相位可构造（rm2.5×0.6 →
  166/8000、dep 19.3%）；CPU 机制上限 ≈60/8000（am2×0.6、dep 17.3%，GOBI 均衡约束）；
  Disk 拒绝门内 ≤20/8000（更深容量=部署墙 60–98%）。用户选择 **b2：磁盘 law 手术 +
  非对称下限**。
- **b2 登记（预注册，跑探测前固定）**：相位主导下限（每 8000 host-step、拒绝率门不变）：
  **CPU ≥50、RAM ≥100、Disk ≥60**；仍要求 target>其余类 且 anomalous share≥50%。
- disk law 手术方案：016 law 为 10 态线性 0–9000、初始概率 65% 在 ≤2000 但 ~0.9% 尾部分布在
  7000–9000（正是部署墙来源）、转移自留 0.78–0.9（增长极慢）。P20 law v1/v2：初始概率截断
  >4000（重归一），转移改"上行偏置"（L1: stay.3/+1.4/+2.2/+3.1；L2: stay.15/+1.45/+2.3/+3.1），
  使多数容器在寿命内增长到 6000–9000 → 在拒绝门内（cap 0.20–0.28）制造 solo 主导事件。
  探测网格：L1/L2 × cap {0.20, 0.25}（必要时补 0.22/0.28）。

### 决策数据附录（P16 附件，2026-09-07 留档）

按 8000/相位投影降序的轴内表（dep-rej 门 <20%；§13 目标 ≥100/8000 且 target>其余）：

| CPU scale | CPU/RAM/Disk 投影 | dep-rej | mig-rej |
|---|---|---|---|
| 0.65 | 48/6/10 | 12.4% | 18.4% |
| 0.70 | 31/11/24 | 9.6% | 21.4% |
| 0.80 | 20/5/20 | 5.4% | 13.9% |
| 0.45 | 20/0/0 | **97.4%** | 0.0% |
| 0.40 | 9/0/0 | **97.3%** | 25.0% |

| RAM scale | RAM 投影 | dep-rej | mig-rej |
|---|---|---|---|
| 0.30 | 246 | **83.8%** | 28.0% |
| 0.35 | 222 | 24.9% | 34.0% |
| 0.40 | 139 | 21.6% | 24.7% |
| 0.50 | 120 | 22.2% | 23.6% |
| 0.60 | 70 | 15.1% ✓ | 10.7% |

| Disk scale | Disk 投影 | dep-rej | mig-rej |
|---|---|---|---|
| 0.17 | 82 | **94.1%** | 16.7% |
| 0.15 | 78 | **96.9%** | 65.2% |
| 0.22 | 41 | 69.3% | 19.4% |
| 0.30 | 15 | 4.1% ✓ | 8.1% |

读数：唯一"拒绝门内且投影最高"的干净候选是 **CPU 0.65（48/8000，12.4%）** 与 **RAM 0.60（70/8000，15.1%）**、Disk 0.30（15/8000）——三轴均达不到 §13 的 100/8000；任何拉高频率的更深容量都触发部署拒绝墙。选项 a–d 见 P16 主记录。

### 2026-09-07 — S4 阶段（终）—— 正式停止条件 P16

#### P16. 31 候选 data-only 全扫描：无候选满足相位主导 × 拒绝率双门禁（S4 FAIL）

- **执行**：train cohort（457 合格 VM 中 277）seed410×400 interval（6400 host-step scored）×31 候选
  （cpu 11 / ram 12 / disk 8，含 Extension 1 全部深度值），全部为 data-only 采集
  （无模型加载），产出 `capacity_scan/capacity_scan_report.json`（§9 schema + 相对水位门禁）。
- **结果摘要**（每轴最佳候选 vs 门禁：相位投影按 rate×8000；dep-rej<20%、mig-rej<40%；§13 ≥100/相位）：

  | 轴 | 最佳候选 | 目标事件 | 相位投影 | dep-rej | 结论 |
  |---|---|---|---|---|---|
  | CPU | 0.65 | 38/6400 | 48 | 12.4% | 频率 FAIL（<100） |
  | CPU | 0.50 | 12/6400 | 15 | 20.1% | 双 FAIL |
  | CPU | 0.40/0.45 | 7/16 | 9/20 | **97.3%/97.4%** | 部署墙 |
  | RAM | 0.30 | 197/6400 | 246 ✓ | **83.8%** | 拒绝 FAIL |
  | RAM | 0.60 | 56/6400 | 70 | 15.1% ✓ | 频率 FAIL |
  | RAM | 0.10–0.25 | 25–52 | 31–65 | 94.8–96.8% | 部署墙，事件反降 |
  | Disk | 0.30 | 12/6400 | 15 | 4.1% ✓ | 频率 FAIL |
  | Disk | 0.17 | 66/6400 | 82 | **94.1%** | 拒绝 FAIL |
  | Disk | 0.12/0.15 | 42/62 | 52/78 | 98.4%/96.9% | 部署墙 |
- **机制观察**：
  1. **容量越深 → 部署拒绝墙（dep-rej→95%+），事件不升反降**（plan §14 预警"更多 reject 而非更多
     overload"在 0.10–0.45 区间全面成立：目标事件 7–52，均低于 0.30–0.35 邻域峰值）；
  2. 频率可用区（0.55–1.0 / 0.6–1.0 / 0.25–0.30 disk 之外的浅容量）事件率 0.1–0.6%，
     GOBI 能量均衡 + 需求适配 clip（cpu≤1860、ram×2 上限 ~3.4k）使单机过载成为稀有事件；
  3. RAM/Disk 容量对 GOBI 决策不可见（其输入仅 CPU 利用率），约束只在执行期生效——与设计一致，
     但使"以容量制造故障"只能走拒绝路径，无法走"调度硬塞"路径；
  4. 正常占比 96.9–99.9%（远超 §11 80–95% 目标带的场景侧差异，记录不改）。
- **判定（The Plan §0/§11.1/§45 规则）**：**S4 无可行容量配置 → 协议分层第 1 层（模拟器/数据层）
  未通过 → 停止 S5 及后续阶段**。不调整模型，不修改标签/门禁，不反向调 scenario。
- **候选方向（需用户拍板，不擅动）**：
  (a) **降低相位频率要求**（如 §13 ≥100 → 重新登记 ≥50 或按 32000 折算 0.47%），接受更低
      事件率的场景并据实报告（改动=协议门禁重登记）；
  (b) **加大需求/占用**（P20 专用 demand adapter：提高 cpu clip/ram multiplier/到达率，或提高
      初始占用数）使中等容量即可过载——scenario 改动（019 先例：相位化到达池经用户授权）；
  (c) **提高 host 饱和度**：用更少 host 或改变 cohort 分层（如按高需求 VM 分层 train cohort）——
      scenario 改动，需授权；
  (d) 接受**数据层负面结论**：在该 workload/调度器/容量控制组合下，容量驱动的三资源 fault-mode
      drift 在注册门禁内不可构造 → 关闭 Protocol 020 数据路径并报告（可信负面结论）。

### 2026-09-07 — S4 阶段（续 2）

#### P15. 预登记网格对 P20 train-cohort 频率不足 —— 网格扩展 Extension 1（已注册）

- **中间数据（cpu 轴 0.70/0.75/0.80/0.90，400×16=6400 host-step scored）**：
  | candidate | CPU | RAM | Disk | norm% | 相位投影(×8000) CPU | dep-rej | mig-rej |
  |---|---|---|---|---|---|---|---|
  | cpu_0.70 | 25 | 9 | 19 | 99.2% | 31 | 9.6% | 21.4% |
  | cpu_0.75 | 9 | 1 | 17 | 99.6% | 11 | 9.5% | 23.1% |
  | cpu_0.80 | 16 | 4 | 16 | 99.4% | 20 | 5.4% | 13.9% |
  | cpu_0.90 | 0 | 4 | 11 | 99.8% | 0 | 8.0% | 12.9% |
- **读数**：CPU 主导事件率 0–0.39%；§13 相位水位 ≥100/8000（1.25%）全部不合格；正常占比
  99%+ 也超 §11 上限（80–95%）；拒绝率均在门内。
- **判定（未动任何模型/标签逻辑）**：plan §46 网格是"推荐初值"，容量选择须由 data-only
  scan 完成 → **Extension 1 已注册**（2026-09-07，prepare 脚本 docstring 同步）：
  cpu 增 {0.65,0.60,0.55,0.50,0.45,0.40}；ram 增 {0.25,0.20,0.15,0.10}；
  disk 增 {0.15,0.12}。已完成的候选按 manifest 自动跳过；补扫后按 §11/§13 相对水位选容量。
- 相位级门禁按 §13（≥100/相位、target>其余、anomalous share≥50%）作为最终判定水位；
  §11 事件/host-step 门禁按 32000 基准折算到实际候选步数（分析器已实现）。

### 2026-09-07 — S4 阶段（续）

#### P14. 进程内多候选采集：torch.set_num_interop_threads 二次调用崩溃（019 P07 同类）

- **现象**：顺序扫描重启后 cpu 轴第 1 个候选（cpu_0.7）正常完成（class [6347, 25, 9, 19]），
  第 2 个候选（cpu_0.75）在 `configure()` 即崩：
  `RuntimeError: cannot set number of interop threads after parallel work has started`。
- **原因**：`prepare_ftmoe_protocol020_capacity_scan.py` 的 main() 在**同一进程内**循环多个候选，
  每个候选调用一次 `configure()`；torch 的 `set_num_interop_threads` 只允许进程级设置一次。
- **处理**：三个采集器（capacity scan / drift / adaptation episodes）的 `configure()` 全部改为
  进程级一次性配置（模块 flag `_THREADS_CONFIGURED`，与 019 P07 修复同法）；
  清理失败残留目录后重启顺序扫描链（后台 pwsh-34）。
- **经验**：任何"同进程多次进入收集循环"的新采集器必须带该 flag（已加入 P20 工具模板）。

### 2026-09-07 — S4 阶段

#### P13. 并行扫描触发 RAM guard（3 进程 × ~0.7 GiB > 可用水位）

- **现象**：cpu/ram/disk 三轴各起 1 个后台进程并行（计划 3×~0.7 GiB RSS），约 t=200–250
  时可用内存降到 2.76–2.77 GiB，低于 3.0 GiB guard → 首个候选中途 RuntimeError 中止
  （cpu_0.7、disk_0.17、ram_0.3 各自 partial dir + failure.json，现场已保留/清理对照日志）。
- **背景**：延续 019 P05 的机器约束：整机 15.2 GiB，常驻占用（浏览器/杀软等）约 11 GiB，
  空闲 ~4.0–4.4 GiB；单进程扫描 RSS ~0.7 GiB 可行，但 3 个并发进程不可行。
- **处理（Protocol-020 内执行偏差，已显式记录）**：改为**单进程顺序扫描三轴**（后台链式任务）；
  采集器增加自愈：目录内已有 failure.json 时自动删除并干净重试（原语义保持 manifest 幂等 skip）。
  guard 下限仍为 3.0 GiB（沿用 019 授权值，不继续下调）。
- **影响**：总耗时从 ~20 min（并行）变为 ~35–45 min（顺序），无正确性影响（进程内 seed 确定性）。

### 2026-09-07 — S0 登记（Protocol 019 冻结）

- 分支：`protocol-020` 自 `protocol-019`（commit `57e856d`）分出；019 全量资产只读冻结。
- 登记文件：
  - `docs/FTMOE_ONLINE_PROTOCOL_020.md`（本协议文档，阶段状态表）
  - `artifacts/ftmoe_online/protocol_020/protocol.json`
  - 本问题日志（与 020 指令同文件夹，按用户要求）
- 环境复核（延续 019 日志 P02/P05）：torch 全部 CPU-only（dynmoe: Python 3.8 / torch 2.4.1+cpu /
  numpy 1.19.2；Python310: torch 2.13.0+cpu）；RAM guard 沿用 3.0 GiB；Bitbrain 500 个 VM CSV 在位；
  磁盘余量 ~109 GiB；git 客户端过旧（无 switch/show-current，用 checkout -b / rev-parse）。
- 目标路线（The Plan §45）：S1 实现并测试 RAM_CAP_SCALE → overload_mask/ratio → graph v3 →
  per-sample capacity → S3 VM split → S4 data-only scan（不加载模型）→ S5 相位流 → 后续各层门禁。

### 2026-09-07 — S1：RAM_CAP_SCALE 实现

- 修改 `simulator/environment/RPiEdge.py::generateHosts`：
  - 读取 `RAM_CAP_SCALE`（缺省 1.0）；RAM 构造改为 `RAMSize * ram_cap_scale`（读写带宽不动，
    与 The Plan §4.2 一致——本协议不靠 RAMRead/Write 制造 RAM-space fault）。
  - 校验 scale 必须为有限正数，否则 ValueError（防 0/负/NaN 静默破坏容量）。
  - 环境变量缺省时与 Protocol 019 逐位一致（测试 Test 1）。
- 新增 `test_ftmoe_protocol020_simulator.py`：Test 1（无变量/显式 1.0 = legacy）、
  Test 2（0.5 → 2147.5/4096.0、8GB:4GB=8192:4295 保持、CPU/Disk 不动）、非法 scale 拒绝；
  **5/5 通过**。
- 注意：此改动只影响 Protocol 020 自己新采集的流；旧 prepare 脚本不设 RAM_CAP_SCALE，行为不变。

### 2026-09-07 — S2：Graph semantics v3 + per-sample capacity

- 修改 `recovery/PreGANSrc/src/ftmoe_ablation.py::ScheduleGraphEncoder`：
  - `graph_migration_and_occupancy` 增加 v3 路径：`graph_context` 携带 `before_placement [B,W,C]`
    时，每个 window 位置独立解码 `src=before_placement[t,c]`（实际当前 host）→
    `dst=argmax(proposed schedule[t,c])`，shape [B,W,H,H]；被拒的上一时刻 proposal 不再污染
    当前源 host（019 §1.5）。无 `before_placement` 时 v2/legacy 路径代码原样保留。
  - `forward` 容量输入支持 `graph_context['capacities'] [B,W,H,3]`（per-sample），否则回退
    静态 `host_capacity` buffer（legacy 逐位不变）。
- 新增 `test_ftmoe_protocol020_graph.py`：Test 5（2→8 边）、Test 6（被拒 8→3 不算 8→3 边）、
  Test 3（动态容量改变输出）、Test 4（window 切片后未来容量不可泄漏）、未部署槽无迁移边、
  v2/legacy 一致性——**7/7 通过**。
- 回归：`test_ftmoe_protocol019_s1.py` 8/8、`test_ftmoe_online.py` 9/9 通过（legacy 无退化）。
- 注：Replay/window 层的 v3 context 组装（把 before_placement/capacities 切片传入模型）属于
  P20 自己的 runner（run_ftmoe_protocol020.py 系列），旧 runner 不动。

### 2026-09-07 — S3：VM source-disjoint split

- 新增 `build_ftmoe_protocol020_vm_split.py`（纯原始轨迹统计，无任何模型输出）：
  - 预登记合格标准：rows≥400、nan<0.2、CPU active ratio≥0.05；扫描 500 个 VM CSV。
  - 457 个合格 VM；按 (cpu_ips_mean×ram_units_mean) tercile 分层，层内 sha256('p20:v1:<id>')%10：
    bucket 0-5 train（277）/ 6-7 dev（86）/ 8-9 online（94）；Test 9 不相交断言通过。
  - 产物：`artifacts/ftmoe_online/protocol_020/vm_split.json`（含 per_vm 统计与分层明细）。
- 新增 `simulator/workload/BitbrainWorkloadProtocol020.py`（Protocol020AdaptedBWGD2）：
  - 与 016/019 相同的 demand adapter（cpu clip[2,1860]、ram×2、io=1、synthetic disk law），
    到达池 = 指定 cohort 的 VM 列表；跳过 BWGD2.__init__ 的全池静态扫描（每次 ~20 s），
    属性初始化逐项对齐并校验 cohort VM 文件在位。

### 2026-09-07 — S4：data-only capacity scan（进行中）

- 新增 `simulator/environment/RPiCapacity.py`（RPiCapacity controller v1）：
  - phase/候选容量 = 改写 live host 容量字段（ipsCap/ramCap.size/diskCap.size）；Host available
    每次现算（Host.py 无缓存）→ 下一次 placement 立即生效；GOBI 对 RAM 容量盲视属预期（实验对象）。
  - 前置断言：host 必须以物理容量构造（无任何 CAP_SCALE env），防双重缩放。
- 新增 `prepare_ftmoe_protocol020_capacity_scan.py` + `analyze_ftmoe_protocol020_capacity_scan.py`：
  - 预登记网格（§46）：cpu∈{0.70,0.75,0.80,0.90,1.00}（RAM1.0/Disk0.30）、
    ram∈{0.30,…,1.00}（CPU1.0/Disk0.30）、disk∈{0.17,0.1875,0.20,0.22,0.25,0.30}（CPU/RAM1.0）；
    train cohort、seed 410、400 scored interval（+1 guard）。
  - 采集器记录 overload_ratio/overload_mask [T+1,16,3]（§10）、per-interval capacities、
    每 interval deploy/migrate attempt+rejected 计数（decision×executed 差分）、host 级
    run-length 事件与时长统计。
  - 分析器按 §9 schema 输出 + §11 预登记门禁（各 ≥150 host-step / ≥30 events / normal 80-95% /
    deployment<20% / migration<40%）。
- smoke 验证（60 interval）：cpu0.75 [950 N,5 CPU,5 Disk]、ram0.45 [956 N,4 RAM]，拒绝率低位，
  方向正确 → 三轴全量扫描（400 interval）已在后台并行启动（cpu×5 / ram×8 / disk×6）。
- 口径记录：deployment rejection 的"尝试"含被拒容器跨 interval 重试（每 interval 各计一次）；
  初始部署（t=0 的 addContainersInit）在循环外执行，不计入拒绝统计。
