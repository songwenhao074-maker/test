# PreGAN+ / FT-MoE 在线实验

目标：在合理业务更替与复现场景中检验动态残差专家D相对固定残差专家C的预测优势，保留负结果与证据边界。

**当前唯一下一步：[Protocol-030同机成对验证旁路无干扰](docs/PROTOCOL030_SINGLE_TASK_DIRECTIVE_20260922.md)，尚未实现/运行。**

## 当前结论

[029结果](docs/PROTOCOL029_RESULTS.md)已交付，但一致性门禁失败。检测/分类概率差分别为1.19209e-6/2.08616e-6，均超过1e-6；生命周期一致，原因未确定。[完整分析](docs/PROTOCOL029_ANALYSIS_20260922.md)。

五个潜在验收通过机会全部位于S6首次出现阶段，同时被busy与相似度条件阻挡；不是已成功复用，更不是回归D>C证据。有效D/C结果仍以[028](docs/PROTOCOL028_RESULTS.md)为准：全程AP C/D=0.719730/0.711047，九窗均值0.769551/0.766681，D正差3/9，墙钟约为C的10.35倍。C为固定残差专家，不是全量微调。

## 下一模型从这里继续

读[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)和[030计划](artifacts/ftmoe_online/protocol_030/plan.json)。仅做同机audit_off/audit_on两个原D回放，不改科学策略，不跑C/A/B，不改旧失败判定；配对检查与历史匹配分开报告，交付后停止。

保留冻结数据、checkpoint、recovery及代码依赖，输出按run_id隔离，大产物保存Actions artifact链接与哈希。[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [上游许可证](LICENSE)。
