# PreGAN+ / FT-MoE 在线实验
目标：在明确合理的部署场景中检验动态残差专家D的优势。

**当前唯一任务：[Protocol-032正常NLL容忍额试验](docs/PROTOCOL032_SINGLE_TASK_DIRECTIVE_20260927.md)，尚未运行。**
[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [计划JSON](artifacts/ftmoe_online/protocol_032/plan.json) / [6800行结果分析](docs/PROTOCOL031_INTERIM6800_ANALYSIS_20260927.md)。

031的6800行前缀C/D回放已完成：两回归平均AP为C=0.819464、D=0.816672；D未产生真实休眠专家和复用，耗时约为C的18倍。完整9869行流未完成。不能据此声称D更好或更省。

下一次只改变D的F0正常NLL容忍额，其他质量门槛保持；沿用6799计分＋1保护行，C/D各一次。不继续模拟、不追加A/B或种子。实验实现位于既有实验分支，接手前读交接。

历史结果、模拟器、checkpoint与恢复依赖保留。[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
