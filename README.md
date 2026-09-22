# PreGAN+ / FT-MoE 在线实验

目标：在合理业务更替与复现场景中检验动态残差专家D相对固定残差专家C的预测优势，保留负结果。

**当前唯一下一步：[Protocol-031非阻塞复用D/C试跑](docs/PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)，尚未实现/运行。**

## 最新结论

[030结果](docs/PROTOCOL030_RESULTS.md)确认同机旁路无干扰：开关旁路的检测/分类概率逐元素相同，生命周期一致。两次回放后的JSON序列化故障已从完整artifact恢复，不需重跑030。[分析](docs/PROTOCOL030_ANALYSIS_20260922.md)。

五个潜在验收通过机会都在S6首次出现，且同时被busy和相似度硬准入阻挡；不是实际复用或D>C证据。[028](docs/PROTOCOL028_RESULTS.md)仍是已完成D/C参照：C/D全程AP=0.719730/0.711047，九窗均值=0.769551/0.766681，D正差3/9；D总成本并不更低。C是固定残差专家，不是全量微调。

## 下一模型入口

阅读[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)与[031计划](artifacts/ftmoe_online/protocol_031/plan.json)。只实现一个联合复用策略包并跑C/D各一次：相似度只排序，因果验证与新生训练并行，原预测质量门槛保持。九回归窗口主指标不变，完成后停止。

保留冻结数据、checkpoint、recovery及代码依赖；输出按run_id隔离，大产物存Actions artifacts并记录哈希。[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
