# 从 GitHub 继续 FT-MoE 实验

更新：2026-09-14。本次同步合并远端 `ca39074` 的文档历史与本地 `1ccd6b1` 的精简历史；不改写历史实验结果。

## 先读什么

1. 本文件：可运行范围和交接限制。
2. `docs/ONLINE_D_ANALYSIS_AND_NEXT_SCENARIO_20260914.md`：最新复核、确定的代码缺陷和下一轮建议。
3. `docs/PROJECT_STATUS_AND_HISTORY.md`：需要时查项目历史；旧判断以该文件顶部审计补充为准。

用户当前目标是在合理、现实可能出现的特定在线情形中证明 D（动态增删专家）相对于指定对照的优势，允许合理修改新场景与公开的资源不对称。原 023 结果保留；旧“必须先通过 C/specialist/梯度门禁才允许实现 D”是历史计划，不是新协议的通用限制。

远端原 `PROJECT_CONTEXT_LATEST.md` 和 `NEXT_EXPERIMENT_LATEST.md` 的完整内容保留在 `ca3907466d8b549adbb5c0238fcc3c61859e3f45`；根目录同名文件现在只指向当前入口，避免互相冲突的“唯一执行文档”。

## 当前真实状态

- 023 Round2A 完成；只有 A/C 完整在线对比，D 尚未实现。
- C 首次接触收益小，复现阶段末段收益明显。历史 `D_eligible=false` 不等于 D 比较失败。
- 阶段 onset 先展平再移位的错误、倒数第二行漏结算及诊断宏 F1 口径均已记录，**旧运行器尚未修复**。
- 本次上传是仓库同步与交接；未执行新场景、C Rescue 或 D 训练。
- seed700 用于开发；701～703 继续保留确认用途；201～205 预留测试未生成。

## 新机器准备

默认 `main` 与 `protocol-023` 同步到本次交接内容。推荐浅克隆，避免下载全部历史中间产物：

```console
git clone --depth 1 https://github.com/songwenhao074-maker/FT-MoE.git
cd FT-MoE
python -m pip install -r requirements-online.txt
python maintenance/verify_experiment_handoff.py
python -m unittest test_ftmoe_protocol023_core test_ftmoe_protocol023_regimes test_ftmoe_protocol023_s2 test_ftmoe_protocol023_gradient
python verify_ftmoe_protocol023_generator.py
python verify_ftmoe_protocol023_stream.py artifacts/ftmoe_online/protocol_023/development_streams/dev_seed700_steps2880_calibrated --tag-suffix _calibrated
```

使用有权访问该私有仓库的 GitHub 账户认证，不把 PAT 写进 URL、文件或提交。

参考环境是 **Python 3.8.20、torch 2.4.1 CPU、numpy 1.19.2、Windows**。`requirements-online.txt` 从当前正常环境提取关键依赖，补齐旧 requirements 缺少的 psutil 等；本机 `pip check` 和回归测试通过。没有在全新 Linux 环境重装验证；PowerShell 脚本和维护重训脚本中的 Windows Python 路径需要按新机器调整。运行核心 Python 脚本时从仓库根目录启动。

## 为什么这些文件不能再删

- `simulator/workload/datasets/bitbrain/rnd/1..499.csv`：023 采集的硬依赖，原下载链接已失效；本次保留已跟踪的 500 个 CSV，约 476 MiB。不要因体积大而删。
- `artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt`：冻结起点，SHA256 `10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b`。
- `protocol_023/round2a/checkpoints/C_after_*.pt`：7 份成熟 C 检查点，已确认远端与本地一致，本次完整保留以支持梯度复核/诊断。
- 每个已发布流的 `generation.log`：验证器要求的七类输出之一；8 份日志已补入，不能按普通运行日志省略。
- `recovery/PreGANSrc/data/`、`checkpointsplus/`，以及协议 020 的 adaptation_data、normalization、adapter、drift、disk_law：当前链路跨协议依赖。
- 根目录 `run_ftmoe_protocol019.py`、`run_ftmoe_protocol020.py`、`ftmoe_protocol022_core.py` 等旧协议名文件仍被 023 import。
- 上游 `recovery/*.py`、调度器及其资产保留：`main.py` 会提前导入多个恢复基线，不能按名字随意剪掉。

已精简的是本地先前删除的历史诊断脚本、重复报告、旧中间流与大量账本。旧分支与 Git 历史保留；本次不做历史重写。因此当前文件树更集中，但整个仓库历史存储不会同步缩小；补齐不可下载数据后当前树字节数也可能增加。

## 014 重训归档的边界

`docs/PROTOCOL014_RETRAINING_20260913.md` 与轻量结果/校验摘要一并发布。826 MiB 的 `snapshot.zip`、其展开的 900 份逐轮权重及大份恢复缓存保留在原本机，不放入普通 Git。**它们不是继续 023/D 的输入**。

GitHub 克隆可继续在线实验，不等于取得完整 014 归档。若后续任务需要恢复所有 014 模型，应另行转交 ZIP 并按文档哈希验证。不要把 GitHub 中未包含的 ZIP 链接误认为已经下载。历史恢复脚本还依赖完整 Git 历史及原本机资产；浅克隆不能保证执行该恢复流程。

## 后续实验执行顺序

先在新版本修复评估与完整性检查，旧 predictions/summary/manifest/checkpoint 保持原样；输出写入新协议目录。随后有限诊断可学性与表征，再做 A/C/D 的新场景 pilot，最后补固定大池、D 消融与独立确认。具体候选场景和比较范围见最新复核文档。

同一时刻只跑一个模型或模拟器进程，CPU 3 线程、interop 1；在线可用内存 guard 2.5～3.0 GiB，离线/采集 4.5 GiB，磁盘至少 20 GiB。不要直接重新运行会写入已存在注册目录的旧 runner。

发布文件与关键输入的哈希清单见 `maintenance/github_sync_20260914/`；检查脚本只读取文件，不训练或覆盖实验产物。

本次另外修正了 `test_ftmoe_protocol023_s4_smoke.py` 的测试范围：它只运行 24 步，却原先要求整条 2880 步流的末行已经结算，所以会误报失败。现在显式检查所有已成熟前缀、预测写入与预期更新次数，保留完整流审计读数并说明末尾未测；不修改在线算法，也不掩盖待修的完整流末尾漏结算问题。
