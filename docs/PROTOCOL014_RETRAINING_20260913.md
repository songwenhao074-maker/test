# 协议 014 归档重建

2026-09-13 用户确认原归档没有其他副本，并授权：如果无法恢复 snapshot.zip，就找到当时训练设置，重新训练并保存。

**完成状态（2026-09-14）：30/30 个模型已完成，快照及实际恢复校验通过。**

**GitHub 发布范围：** 本文及轻量结果/校验摘要已同步；下文的 826 MiB `snapshot.zip`、展开权重和恢复缓存仍只保存在原本机，不在普通 Git 克隆中。继续 023/D 不依赖它们；完整恢复 014 时需另行转交 ZIP。

## 完成结果

- 新快照：[snapshot.zip](../artifacts/ftmoe_protocol014_retraining_20260913/snapshot.zip)，865,943,486 字节，约 826 MiB，已设为只读。
- 包含 2,532 个文件：历史源码和数据、新训练的 900 份逐轮权重、30 份 best、30 份 last、30 份 optimizer/RNG 断点、训练日志、结果和模型路径索引。
- 30 次训练的源码指纹及 13 个核心源码/数据文件与历史记录一致。
- 原三种子、新两种子、合并五种子的 A_last/B_best 共 108 项指标逐项一致；150 项逐种子配对差值在 1e-12 内一致。
- 幸存的五个完整 v4 原权重均与新训练的 B_best 逐张量完全一致；每个模型比对 151 个张量，最佳轮次仍为 21/23/15/22/23（种子 1/2/6/17/42）。
- 对 ZIP 中全部 2,532 个文件重新读取并核验 SHA-256，再实际恢复到临时目录逐文件核验；两次均通过，临时恢复副本已清理。

新 ZIP SHA-256：`f14f72a7c5da6d323a64c966552e5a435ed50aeb8fdd0c810721e0aab9642b15`。

核验记录：[快照校验](../artifacts/ftmoe_protocol014_retraining_20260913/snapshot_verification.json)、[恢复校验](../artifacts/ftmoe_protocol014_retraining_20260913/restore_verification.json)、[历史一致性](../artifacts/ftmoe_protocol014_retraining_20260913/historical_equivalence_check.json)。新的训练与快照均单独保存，原快照内其他未找回的历史实验没有伪造或混入新结果。

## 恢复调查

原 2026-09-05 ZIP 与完整 manifest 均已删除。本地 Git 全部 2,091 个 blob、项目残留文件、回收站相关条目及上游 ZIP 已检查。从维护时的 7,361 文件 SHA-256 清单中，找回 1,082 个字节一致文件，另有 6,279 个清单内文件未找到。该清单小于原快照的 9,760 文件范围，不能代表原快照完整清单。

已找回当前重训需要的历史代码、协议 004 物理数据、协议 014 比较结果及原完整 v4 的五个 B_best 权重。原五个权重只作为历史证据保留，不用于重训初始化。恢复明细在 `artifacts/ftmoe_protocol014_retraining_20260913/recovery/`。

## 新训练

v0、v1、v2、v3、完整 v4、门控 v4 各训练种子 1/2/6/17/42，共 30 个独立模型。配置为 lr=0.0003、30 轮、expert dropout=0；全部参数进入优化器，保留原损失、优化器、调度、阈值及选模方法。

训练重放 42/1/6/17/23，开发重放 31/101/102。每个种子按开发集 S=(F1+HR+NDCG)/3 选择最早最高分轮次，固定第 30 轮作为末轮对照。仍保留 HR=NDCG 的历史定义。

数据和八个训练依赖源码均与维护清单 SHA-256 一致。注意力和门控训练前检查均通过：公共参数初始化匹配，各模块获得更新、损失有限、全部参数可训练。

运行限制：CPU 3 线程、interop 1、BelowNormal、单训练进程、RAM 4.5 GiB 和磁盘 20 GiB 下限。每个模型保留全部 30 轮 checkpoint、best/last、optimizer/RNG 断点、逐轮指标和训练日志。

本次是基线重训，不是新方法实验；不重训原快照内其他协议，不捏造无法恢复的历史文件，不修改协议 023，不使用预留测试或启动 QoS。重训结果单独保存，不能声称与原 ZIP 逐字节相同。

## 产物入口

根目录：`F:\PreGANPlus-master\artifacts\ftmoe_protocol014_retraining_20260913`。

- `retraining_registration.json`：完整训练登记和历史源码/数据哈希。
- `status.json`：当前模型和完成计数；只有 `state=complete` 才表示全部重训及归档完成。
- `controller.log`、`controller.stderr.log`、`logs/`：执行日志。
- `workspace/`：隔离的历史源码、物理数据与新权重。
- `retraining_results.json`：30 个模型全部完成后生成的逐模型及汇总结果。
- `snapshot.zip`、`snapshot_manifest.json`、`snapshot_verification.json`：全部完成后自动打包、逐文件校验，设为只读。
- `restore_snapshot.py`：校验和恢复到空目录的标准库工具。

新快照生成后校验：

```powershell
& 'D:\Anaconda\envs\dynmoe\python.exe' -X utf8 'F:\PreGANPlus-master\artifacts\ftmoe_protocol014_retraining_20260913\restore_snapshot.py' verify
```

恢复到新目录：

```powershell
& 'D:\Anaconda\envs\dynmoe\python.exe' -X utf8 'F:\PreGANPlus-master\artifacts\ftmoe_protocol014_retraining_20260913\restore_snapshot.py' extract --destination 'F:\PreGANPlus-restored-retrained014'
```

如训练被系统中断，先确认没有仍在运行的控制器或训练进程，再执行 `maintenance/retrain_protocol014_20260913.py`。已完成模型会跳过，未完成模型读取其独立 resume.pt。不要同时启动两份控制器，不要清理本目录。
