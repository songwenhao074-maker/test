<h1 align="center">PreGAN+ · FT-MoE online-learning track</h1>

> **入口：** [项目现状与历史沿革](docs/PROJECT_STATUS_AND_HISTORY.md)（唯一状态入口）·
> [协议 023 设计与结果](docs/FTMOE_ONLINE_PROTOCOL_023.md)

本仓库 = 上游 **PreGAN+** 边缘计算故障容忍框架 + 一条附加的 **FT-MoE 在线学习研究线**。

---

## 1. 这是什么

**PreGAN+** 针对移动边缘基础设施的争用与过载：用 GAN 预测争用、用 few-shot 分类器定位
具体过载资源，并生成**抢占式迁移**决策，在节点被打满前主动迁移任务。仓库内含 16 主机的
Raspberry-Pi 边缘模拟器与多种调度器、恢复策略基线。

在其之上，本仓库附加研究一个问题：

> 边缘环境的资源需求机制会**漂移、切换并再现**。固定拓扑的在线微调（fixed C）是否足够，
> 还是必须**动态增删专家**（D）？

**当前结论（协议 023 第二轮 A）：`D_eligible = false`。** 在冻结的在线预算下，fixed C 相对
冻结基线的增益只有 +0.0049 / +0.0020 / +0.0203（门槛 +0.03），0/3 达标；既不遗忘，也不产生
梯度冲突。这是本协议族第一个**反对**需要动态专家的测量。详见
[项目现状](docs/PROJECT_STATUS_AND_HISTORY.md)。

---

## 2. 快速开始

```console
git clone https://github.com/imperial-qore/PreGANPlus.git
cd PreGANPlus
python3 -m pip install -r requirements.txt
python3 -m pip install "torch>=1.11"
```

本机已配置好的环境：**`D:\Anaconda\envs\dynmoe\python.exe`**
（Python 3.8.20、torch 2.4.1+cpu、numpy 1.19.2）。

### 跑模拟器（上游路径）

```bash
D:\Anaconda\envs\dynmoe\python.exe main.py -r ftmoe
```

`-r` 可选 `ftmoe`（默认 → `FTMoEProgressiveRecovery`）、`preganplus`、`pregan`、`pcft`、
`dftm`、`eclb`、`cmodlb`。无 `-e` 时走 `RPiEdge` 模拟器 + `BWGD2` 合成负载。

> 注意：`main.py` 在解析参数**之前**就 import 全部 9 个 recovery 模块（`main.py:54-62`），
> 因此 `recovery/*.py` 不能删任何一个。

### 跑协议 023 实验链

```powershell
$env:PYTHONPATH="F:\PreGANPlus-master"
cd F:\PreGANPlus-master
$py = "D:\Anaconda\envs\dynmoe\python.exe"

# 回归基线（必须保持 206 tests OK）
& $py -m unittest test_ftmoe_protocol023_core test_ftmoe_protocol023_regimes test_ftmoe_protocol023_s2 test_ftmoe_protocol023_gradient

# 生成器独立验证（A-01..A-04）
& $py verify_ftmoe_protocol023_generator.py

# 采集 + 独立验证（单进程，约 7 分钟/流）
& $py run_ftmoe_protocol023_s2.py
& $py verify_ftmoe_protocol023_stream.py artifacts/ftmoe_online/protocol_023/development_streams/dev_seed700_steps2880

# 数据门禁 / 专业化 / 梯度
& $py analyze_ftmoe_protocol023_s2.py
& $py probe_ftmoe_protocol023_specialization.py
& $py probe_ftmoe_protocol023_gradient.py
```

**运行纪律：** 单进程顺序执行（同一时刻只跑一个实验进程）；在线链路 2.5–3.0 GiB 内存 guard，
离线/采集链路 4.5 GiB guard + 磁盘 ≥20 GiB；不得覆盖历史 checkpoint 或产物。
torch 的 stderr warning 会让 pwsh 报 `exit 1`，**以产物判定成败**。

---

## 3. 目录

| 路径 | 内容 |
|---|---|
| `main.py` | 上游入口（模拟器 + 恢复策略选择） |
| `framework/` `simulator/` `scheduler/` `stats/` `metrics/` `utils/` | 上游模拟器与基线 |
| `recovery/` | 恢复/预测模型；`recovery/PreGANSrc/src/ftmoe_*.py` 是 FT-MoE 实现 |
| `*_ftmoe_protocol023_*.py`、`ftmoe_protocol02{2,3}_core.py` | 协议 023 实验链（根目录，不移动以免破坏 import 与哈希登记） |
| `artifacts/ftmoe_online/protocol_023/` | 当前协议的全部注册与证据 |
| `artifacts/ftmoe_online/protocol_020/`、`adapted_bwgd2_016/` | 协议 023 依赖的冻结起点与配置 |
| `docs/` | [项目现状与历史](docs/PROJECT_STATUS_AND_HISTORY.md) · [协议 023](docs/FTMOE_ONLINE_PROTOCOL_023.md) |
| `指令/` | 协议 023 的上游计划与两轮 directive（SHA256 已登记在 `protocol.json`） |

---

## 4. 已知限制（不得只取最好看的指标）

- **HR@100% 与 NDCG@100% 在本数据上相等**（单资源标签），都等于真实异常样本上的资源 top-1
  准确率、含检测漏报；`S=(F1+HR+NDCG)/3` 实际给诊断两倍权重。
- 开发集异常以 CPU 类为主，**近乎恒定预测 CPU 的诊断准确率约 0.898224**。
- **无训练物理规则的开发基线 F1≈0.93057、诊断≈0.98194**，高于当时候选的完整模型。
- 论文参考 F1/HR/NDCG = 0.8766/0.6496/0.6021；数据、标签与指标实现有差异，**不能据数值更高
  声称复现成功或优于论文**。
- 全部在线结论基于**开发**种子 700；confirmation 种子 701–703 未使用。预留测试 201–205
  从未生成，保持封闭。

---

## 5. External Links & License

| Items | Contents |
| --- | --- |
| **Video** | https://youtu.be/Pp82aZu5dJw |
| **Contact** | Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli)) |
| **Funding** | Imperial President's scholarship |

BSD-3-Clause. Copyright (c) 2022, Shreshth Tuli. See [LICENSE](LICENSE).
