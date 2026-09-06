# FT-MoE Protocol 020 — 实验执行问题日志（Problem Log）

> 本文件与 `FTMOE_PROTOCOL020_DETAILED_SOLUTION_PLAN.md` 同目录（`F:\PreGANPlus-master\指令\`）。
> 用途：记录尝试执行 Protocol 020 过程中遇到的所有问题、失败、停止条件、配置与 hash，供后续修改与复盘。
> 规则：**任何阶段失败 → 停止后续阶段并记录**；记录后不改写历史条目，只追加（最新在最上）。

---

## 问题汇总表（最新在上）

| # | 阶段 | 严重度 | 状态 | 摘要 |
|---|---|---|---|---|
| P14 | S4 | 中 | 已修复（一次性线程配置 flag） | 进程内多候选采集时每个候选重复调用 `torch.set_num_interop_threads(1)` → 第 2 个候选 RuntimeError（019 P07 同类）；cpu 轴仅产出 cpu_0.7 一个候选即崩；三个采集器（scan/drift/adaptation）均已加进程级一次性配置保护，重启扫描 |
| P13 | S4 | 中 | 已处理（改为顺序运行） | 3 个并行扫描进程使可用内存跌破 3.0 GiB RAM guard（2.77/2.76 GiB），cpu/ram/disk 首候选约 t=200–250 中止；已清理失败残留并改为单进程顺序重跑 |
| — | S0/S1 | — | 进行中 | 2026-09-07 登记 Protocol 020（分支 protocol-020）；S1 RAM_CAP_SCALE 已实现，测试待跑 |

---

## 详细记录

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
