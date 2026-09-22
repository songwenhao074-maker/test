# 当前交接：Protocol-030同机配对诊断

以main为准。029已交付但diagnostic_valid=false；用户最新要求授权本次新任务。

先读[029问题分析](PROTOCOL029_ANALYSIS_20260922.md)，再执行[030单任务指示](PROTOCOL030_SINGLE_TASK_DIRECTIVE_20260922.md)与[plan.json](../artifacts/ftmoe_online/protocol_030/plan.json)。030尚未实现/运行。

唯一任务：同job固定单线程确定性CPU环境，原Protocol028 D关闭/开启旁路各执行一次，验证旁路无干扰并汇报全体休眠记忆效用。正式回放预算两次，不改模型策略、阈值、数据或seed，不跑C/A/B。不得扩大容差或改写029历史失败。

新030配对门禁与历史028匹配分别输出；若新配对失败，报告首个分歧位置后停止。即使有效，事后最佳候选仍不是可部署D或D>C证明。

执行模型实现独立030入口/登记/专用push工作流，输出按run_id隔离；文档与结果更新不启动训练。结果写回main并同步所有当前入口后停止。

[029完成时旧交接](GITHUB_EXPERIMENT_HANDOFF_PROTOCOL029_COMPLETED_20260922.md)为历史，不阻止本次已授权的新任务。其失败证据和所有旧注册保持原样。
