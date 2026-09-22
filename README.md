# PreGAN+ / FT-MoE 在线实验

当前目标：在合理的业务更替与复现场景中检验**动态残差专家D相对固定残差专家C的预测优势**。先研究D/C，再由用户决定A/B。

**当前唯一下一步：[Protocol-028：满容量时保护已验收记忆的一次D/C试跑](docs/PROTOCOL028_SINGLE_TASK_DIRECTIVE_20260922.md)。028尚未实现/运行；完成后停止。**

## 从这里继续

阅读[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)、[027问题分析](docs/PROTOCOL027_ANALYSIS_20260922.md)和[028单任务](docs/PROTOCOL028_SINGLE_TASK_DIRECTIVE_20260922.md)。先实现独立028入口，再通过专用分支push启动；不要重跑已完成的旧027方法。

## 最新已完成事实

Protocol-027 run35682811782已完成。C/D全程AP为0.719730/0.711054；九窗口等权AP为0.769551/0.766751。D正差3/9，成功复用0，墙钟约10.10倍、CPU约6.67倍；当前未显示D的准确率或总成本优势。

下一步只检验满容量时不为未验收候选预删旧记忆的策略。两组仍用同一冻结数据、seed700/model1；不自动追加A/B、消融、种子或调参。D额外驻留记忆和计算如实披露。

## 历史和依赖

[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [状态历史](docs/PROJECT_STATUS_AND_HISTORY.md) / [原交接与资产清单](docs/GITHUB_EXPERIMENT_HANDOFF_HISTORY_20260921.md)。历史STOP与D_eligible不构成027门禁，历史失败不改写为通过。

保留 `recovery/`、模拟器、调度器、Bitbrain数据、冻结checkpoint和被当前代码import的旧模块。实验输出按run_id隔离；大产物保存在有保留期的GitHub Actions artifacts，Git记录链接及哈希。

本仓库基于PreGAN+，附加FT-MoE研究线。上游许可：[BSD-3-Clause](LICENSE)。
