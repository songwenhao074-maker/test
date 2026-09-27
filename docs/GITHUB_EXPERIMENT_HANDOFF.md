# 当前交接：Protocol-032单任务
先读[执行指示](PROTOCOL032_SINGLE_TASK_DIRECTIVE_20260927.md)、[031前缀分析](PROTOCOL031_INTERIM6800_ANALYSIS_20260927.md)和[plan.json](../artifacts/ftmoe_online/protocol_032/plan.json)。

031 interim运行36295124291成功完成两组回放，ZIP10923099870已核验；此前main“待实现”与实验分支“生成失败”交接均已过时。真实结论是前缀已完成但D未领先、未复用，9869全流未完成。

032唯一任务：补审并冻结同一6800行输入，实施F0正常NLL绝对容忍额下限0.01这一项变化，C_fixed5/D_guard_budget各一次。误报、局部改善、标签时序和其他训练设置保持；完成或明确阻塞后交付停止。不自动继续生成或发起更多试验。

代码来源：protocol-031-rare-recurrence-20260922的4e7df7e5f7abca2e2470633864b35440858bd4bf，可复用该分支并先同步本指示。新workflow只对自己的文件push触发，可保留dispatch；本次文档提交不启动。不要启动旧031生成/试跑工作流。
上传完整冻结bundle/结果artifact，并把紧凑证据和结果回写main入口；仅上传artifact不算交接完成。旧代码、数据、失败证据不改写。
