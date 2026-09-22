# 当前唯一任务：Protocol-029休眠记忆效用诊断

状态：planned_not_implemented_not_run。先从main读取本文件。

028已完成：九回归窗口D-C=-0.00286984，正差3/9，purge=0，reactivation=0。保留专家9成功，但其S3回归两次验证损失更高；8/9个已完成reuse验证未满足1%改善。

[分析](docs/PROTOCOL028_ANALYSIS_20260922.md) / [唯一执行指示](docs/PROTOCOL029_SINGLE_TASK_DIRECTIVE_20260922.md) / [登记计划](artifacts/ftmoe_online/protocol_029/plan.json)。

只做一次不改变028线上轨迹的D旁路诊断，评估全部休眠专家，区分匹配/调度漏检与记忆缺乏收益。复用既有C结果；不重新跑C、不改算法、不放宽门槛、不重做数据。诊断oracle不能作为D胜C证据。

执行模型实现独立029入口及专用push工作流，完成检查、一次回放、结果回写main后停止。不要自动追加任务。
