# 当前项目入口

先读 [Protocol-024 最新结果审阅](docs/PROTOCOL024_REVIEW_20260914.md) 和 [下一轮实验指示](docs/PROTOCOL024_NEXT_DIRECTIVE_20260914.md)。

当前成果：15项P24单测通过；lifecycle-off C/D扩展到1140步，各285更新、覆盖825个异常host-step，概率差在浮点容差内。D尚未启用动态策略做性能对比。

当前缺口：实际阶段评估仍走旧代码；继承的保存函数指向023旧目录；拒绝候选后ID空缺无法恢复；优化器、行为哈希、完整resume与容量回收待完成。

目标：在合理有利的特定在线情形下实测D的优势，公开比较条件并保留失败结果。优先完成可审计的动态pilot，不再只重复容器等价测试。

资产与环境见 [GitHub 交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)，历史见 [项目现状](docs/PROJECT_STATUS_AND_HISTORY.md)。
