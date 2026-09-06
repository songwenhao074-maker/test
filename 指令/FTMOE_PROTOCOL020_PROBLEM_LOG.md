# FT-MoE Protocol 020 — 实验执行问题日志（Problem Log）

> 本文件与 `FTMOE_PROTOCOL020_DETAILED_SOLUTION_PLAN.md` 同目录（`F:\PreGANPlus-master\指令\`）。
> 用途：记录尝试执行 Protocol 020 过程中遇到的所有问题、失败、停止条件、配置与 hash，供后续修改与复盘。
> 规则：**任何阶段失败 → 停止后续阶段并记录**；记录后不改写历史条目，只追加（最新在最上）。

---

## 问题汇总表（最新在上）

| # | 阶段 | 严重度 | 状态 | 摘要 |
|---|---|---|---|---|
| — | S0/S1 | — | 进行中 | 2026-09-07 登记 Protocol 020（分支 protocol-020）；S1 RAM_CAP_SCALE 已实现，测试待跑 |

---

## 详细记录

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
  Test 2（0.5 → 2147.5/4096.0、8GB:4GB=8192:4295 保持、CPU/Disk 不动）、非法 scale 拒绝。
- 注意：此改动只影响 Protocol 020 自己新采集的流；旧 prepare 脚本不设 RAM_CAP_SCALE，行为不变。
- 下一步：运行 S1 测试 → S1 收尾（overload_ratio/mask 由 P20 采集器实现，随 S4 扫描脚本落地）。

---

（待续——后续阶段执行中出现的问题按上述规则追加在本文件。）
