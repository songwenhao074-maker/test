# 在线微调实验（协议 015）

用户已批准四组同源 v4 对比：A 不微调、B 全参数微调、C 固定专家 MoE 微调、D 动态专家 MoE 微调。主目标是五模型种子、三回放种子的后半程整体 F1 等权均值中 D≥0.65 且 D>B。普通微调是否退化、D 是否优于 C 分别报告，不预设结果。

登记文件为 `artifacts/ftmoe_online/protocol_015_online.json`。原生 BWGD2、RPiEdge 和 GOBI 生成一次公共轨迹，预测器不改变调度。采集全部原始资源、身份、拟调度和实际落点；每条流额外采集一步以确认最后的容差标签。保留 ±1 容差主指标和无容差辅助指标。

`prepare_ftmoe_online.py` 为独立采集入口；`run_ftmoe_online.py` 严格先记录预测，再暴露当前物理标签，仅将已延迟确认的样本加入最近64间隔缓冲区，每10间隔更新一轮。`ftmoe_online.py` 是独立模型包装，旧模型文件未修改；仅 D 动态调整 EAGate，稳定专家 ID 和逐专家路由参数保证存活 AdamW 状态不被重置。

`analyze_ftmoe_online.py` 从原始预测重新计分，检查完整运行网格，输出指标、配对区间和图。`run_ftmoe_online_stages.py` 串行执行300步小试、开发参数比较和五种子确认；每一阶段未过门槛或运行失败即写入 `status.json` 并停止，等待用户讨论。不得更改环境以制造退化。

## 已批准的命令

```powershell
& 'D:\Anaconda\envs\dynmoe\python.exe' 'test_ftmoe_online.py'
& 'D:\Anaconda\envs\dynmoe\python.exe' 'run_ftmoe_online_stages.py' --through confirmation
```

运行目录中每100步保存状态，包含优化器、专家拓扑与编号、缓冲区、路由统计、预测、标签可见性、随机数和游标。单运行可使用原命令附加 `--resume`；配置和源代码哈希必须匹配。源代码修改后不得直接续跑旧协议产物。四组 A 的学习率记录为0，仅表示无优化器。

固定参考集来自原开发重放31/101/102，仅用于遗忘诊断。评估期间暂用其原主机容量，评估后恢复当前环境容量；不参与在线更新或策略选择。缺少异常的窗口 F1 标为 null，保留误报数。

原归档、原协议001–014、历史权重与预留测试保持原样。当前运行阶段及问题以 `artifacts/ftmoe_online/status.json` 和各阶段报告为准。
