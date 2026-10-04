# PreGAN+ / FT-MoE 在线实验

目标：通过合理结构使保留动态新增、窗口评分、休眠复用及容量管理思想的完整D优于持续学习C。

**最新指示：[Protocol-044 revision 2 工程恢复](docs/PROTOCOL044_REVISION002_DURABLE_RECOVERY_20261005.md)，已登记、未执行。**

[r1结果](docs/PROTOCOL044_RESULTS.md)：工程核心验收通过，正式数据生成在本地5200/5953行后超时，检查点未远端持久化，四条训练序列均未开始。新增静态审查发现正式C入口参数不符、训练恢复门控不可达。

r2要求先修复实际入口与跨任务恢复；找不到旧状态时仅一次同seed4401重建；每段远端上传并回读后继续。模型、阈值、四序列与2360总梯度预算不变。原r1科学配置和历史报告保留，r2独立交付。

[执行入口](NEXT_EXPERIMENT_LATEST.md) / [交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [r2机器计划](artifacts/ftmoe_online/protocol_044/revision_002/plan.json) / [r1科学方案](docs/PROTOCOL044_CLOSEOUT_AND_CAPACITY_20261004.md)

本次只发布指示，未启动实验；不自动启动045。
