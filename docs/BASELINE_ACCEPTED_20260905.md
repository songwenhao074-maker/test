# 已验收的消融版本与回档说明

## 验收决定

2026-09-05 用户指示：“目前消融实验可以认为已经通过，将现在这个版本归档保存（便于之后实验出现问题回档），之后我会指示进行下一个实验”。当前消融阶段据此标记为用户验收通过，并作为后续实验的回档基线。

基线为协议 014：lr=0.0003、30 轮、expert dropout=0、独立全参数训练、模型种子 1/2/6/17/42；保留 v0–v4 与门控 v4。B_best 仍按原综合分数选模，完整 v4 的五种子 B F1=0.905134，HR=NDCG=0.972003。原协议的小效应门槛及新增种子确认未通过，其记录不改写。用户验收是阶段决策，不是新的统计检验结果。

协议 012 的 source_gated 仍未训练。测试 201–205 未生成或评估；本次不启动训练、最终测试或 QoS。

## 归档内容

归档目录：`F:\PreGANPlus-master\backup\ftmoe_protocol014_accepted_20260905`。

- `snapshot.zip`：独立文件快照，包含项目源码、活动文档、全部 artifacts、数据、现存逐 epoch/best/last 权重、日志、历史备份及维护记录。
- `manifest.json`：逐文件相对路径、字节数、SHA-256；同一清单嵌入 ZIP。
- `verification.json`：对归档内容逐文件重新读取的校验结果、ZIP 哈希及文件数量。
- `environment.json`：Python、平台和已安装包版本；不是 Conda 环境或系统依赖的二进制备份。
- `pre_acceptance_docs/`：本次修改前的五份活动文档，保留原始表述。
- `snapshot_tool.py`：标准库归档、校验与恢复工具；恢复时拒绝覆盖已有非空目录。

快照排除当前归档目录自身、`.git`、`.codex`、`.agents` 及可再生 Python/测试缓存。旧 `backup/` 的内容纳入快照；不使用硬链接。权重中已于此前维护删除的完整训练 optimizer/RNG 恢复状态无法补回，参见 `MAINTENANCE_20260905.md`；现存恢复文件全部保留。

此归档存放于同一 F: 磁盘，用于实验回档；不是独立磁盘故障备份。后续实验不得修改或覆盖本归档。

## 校验与恢复

从 PowerShell 校验，无须加载模型或运行实验：

```powershell
& 'D:\Anaconda\envs\dynmoe\python.exe' 'F:\PreGANPlus-master\backup\ftmoe_protocol014_accepted_20260905\snapshot_tool.py' verify
```

恢复到一个新目录（此命令目前仅作为回档说明，归档时不执行）：

```powershell
& 'D:\Anaconda\envs\dynmoe\python.exe' 'F:\PreGANPlus-master\backup\ftmoe_protocol014_accepted_20260905\snapshot_tool.py' restore --destination 'F:\PreGANPlus-restored-protocol014'
```

工具先验证 ZIP 与清单，再逐文件恢复并核对恢复文件 SHA-256；非空目录不会被覆盖。恢复后需保留原 Python 环境，或按 `environment.json` 重建环境。部分历史配置记录 `F:\PreGANPlus-master` 绝对路径；正式替换当前工作目录前，先保存后续实验，再将已核验的恢复副本切换至原路径。不要直接合并覆盖当前项目，以免混入后续新增文件。

恢复得到项目文件快照；归档目录自身的工具、清单和本次修改前文档保留在原归档目录。归档产物存在且 `verification.json` 显示 `passed: true` 才表示归档完成。
