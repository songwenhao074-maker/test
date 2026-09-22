# PreGAN+ / FT-MoE 在线实验

当前目标：在合理业务更替与复现场景中检验动态残差专家D相对固定残差专家C的预测优势，并保留所有负结果与诊断边界。

**当前状态：Protocol-029一次休眠记忆效用诊断已完成，暂无登记的下一实验。**

## 最新结果

029诊断未通过028同轨迹一致性门禁；blocker=runner_exception。 详见[Protocol-029 Results](docs/PROTOCOL029_RESULTS.md)。

029保持028线上算法、数据、训练、阈值和生命周期决策不变，只增加标签前只读旁路。它没有重跑C或新增训练组；事后最好专家是post_hoc_oracle_diagnostic，不能作为线上D>C结论。

## 历史

[Protocol-028结果](docs/PROTOCOL028_RESULTS.md) / [Protocol-028分析](docs/PROTOCOL028_ANALYSIS_20260922.md) / [历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)。

保留冻结数据、注册、历史失败和大artifact。当前没有自动后续任务。
