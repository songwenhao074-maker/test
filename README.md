# PreGAN+ / FT-MoE 在线实验

当前目标：在合理业务更替与复现场景中检验**动态残差专家D相对固定残差专家C的预测优势**。先研究D/C，暂不追加A/B。

**当前唯一下一步：[Protocol-029：一次休眠记忆效用诊断](docs/PROTOCOL029_SINGLE_TASK_DIRECTIVE_20260922.md)，尚未实现/运行。**

## 从这里继续

阅读[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)、[028分析](docs/PROTOCOL028_ANALYSIS_20260922.md)及[029登记计划](artifacts/ftmoe_online/protocol_029/plan.json)。执行模型只实现一次原样028线上回放的旁路审计，完成并回写结果后停止。

## 最新已完成事实

[Protocol-028结果](docs/PROTOCOL028_RESULTS.md)，run35705211072：C/D全程AP为0.719730/0.711047，九回归等权AP为0.769551/0.766681，D正差3/9；purge=0，reactivation=0。D墙钟约为C的10.35倍、CPU约6.47倍，未显示准确率或总成本优势。C是固定残差专家，不是全量微调。

下一任务只区分“有用记忆被筛选/调度遗漏”和“现有冻结记忆没有收益”。不改数据或验收门槛，不将事后最优专家作为D胜出证据，不自动追加种子/方法/调参。

## 历史和依赖

[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [状态历史](docs/PROJECT_STATUS_AND_HISTORY.md) / [原交接与资产清单](docs/GITHUB_EXPERIMENT_HANDOFF_HISTORY_20260921.md)。历史失败保留，旧计划不作为当前入口。

保留recovery、模拟器、调度器、Bitbrain数据、冻结checkpoint和当前代码依赖。输出按run_id隔离，大产物存GitHub Actions artifacts，Git记录链接与哈希。本仓库基于PreGAN+，上游许可：[BSD-3-Clause](LICENSE)。
