# 当前交接：Protocol-029一次记忆效用诊断

以main为准。028已完成；用户最新要求分析并安排下一任务，现有历史“等待新指示”已由本指示替代。

先读[028分析](PROTOCOL028_ANALYSIS_20260922.md)，再严格执行[029单任务](PROTOCOL029_SINGLE_TASK_DIRECTIVE_20260922.md)及[plan.json](../artifacts/ftmoe_online/protocol_029/plan.json)。029尚未实现或运行。

唯一产出：一次保持028线上算法不变的休眠专家旁路效用审计，回答“未选中的记忆是否具有同期收益”。不改数据/训练/门槛，不重新跑C，不运行A/B或新种子。所有事后最好候选标明oracle诊断，不能当作D>C证据。

执行模型实现独立029入口、隔离输出和专用push触发工作流；不要重启028。数据/历史预测缺失则如实记录阻塞。发布结果到main并同步README、AGENTS、NEXT、PROJECT_CONTEXT及本页后停止。

028原完成交接保存在[GITHUB_EXPERIMENT_HANDOFF_PROTOCOL028_COMPLETED_20260922.md](GITHUB_EXPERIMENT_HANDOFF_PROTOCOL028_COMPLETED_20260922.md)，其历史停止状态不阻止此次新任务；027/028注册、代码和结果保留。
