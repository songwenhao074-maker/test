# PreGAN+ / FT-MoE 在线实验

目标：通过结构修改，使保留动态新增、知识保存/选择、休眠复用和容量管理思想的D优于持续学习C。

**下一轮已登记、未运行：[Protocol-042 revision 2：固定窗口效用与互补专家接入](docs/PROTOCOL042_WINDOWED_UTILITY_20261003.md)。**

本版替代尚未执行的困难样本加权计划。固定四臂：U_parent严格复现041；R_win128作为替换策略对照；A_hist作为累积评分对照；A_win128为预指定主臂。重点检验有用旧专家的保留，以及故障变化时固定128成熟区间评分的适应性。

已发行加性贡献用于低开销移除损失评分；样本不足为未知，正负贡献分别保护，退出活动池先休眠、保留复用记忆。总专家optimizer步骤上限2576；明确计入两个活动专家的额外成本。先完成恢复与门控测试，U通过才执行三新臂。未启动科学运行。

[执行交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [机器登记](artifacts/ftmoe_online/protocol_042/plan.json) / [当前上下文](PROJECT_CONTEXT_LATEST.md)。

**最新完成：[Protocol-041](docs/PROTOCOL041_RESULTS.md)，run37110969550。** 暖启动候选优于零初始化但未超过live，候选拒绝，池尚未形成多专家部署。[041证据](artifacts/ftmoe_online/protocol_041/runs/run_37110969550/)。

历史033–041预算关闭；042 revision1撤销，旧预算不追加。当前仍是已见seed3601开发实验，不代表跨流确认、永久删除有效或端到端加速。

[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
