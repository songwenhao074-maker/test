# 当前交接：027已完成，028单任务待实现

当前状态以main的NEXT_EXPERIMENT_LATEST.md为准。[027结果](PROTOCOL027_RESULTS.md)已经完成，两组均有真实结果；[问题分析](PROTOCOL027_ANALYSIS_20260922.md)给出证据与限制。

唯一后续任务：[Protocol-028：满容量时保护已验收记忆的一次D/C试跑](PROTOCOL028_SINGLE_TASK_DIRECTIVE_20260922.md)。这是一个计划，028代码/工作流尚未实现，不能直接运行旧027工作流当作新实验。

接手先实现新策略与独立028入口，再完成C_fixed5与D_memory_protected各一次并交付。冻结数据直接复用run35682811782的protocol027-frozen-data-35682811782，不重新模拟。028方法注册与原027数据快照分开，保留全部旧证据。

连接器没有workflow_dispatch时，为新028工作流启用专用分支protocol-028-memory-protected-20260922的工作流文件push触发。实现完成后的明确启动提交不含[skip ci]；普通结果提交不触发。不要再要求用户因缺少dispatch工具而手动启动，也不要自动追加实验。

原027数据与启动交接留作[历史快照](GITHUB_EXPERIMENT_HANDOFF_PROTOCOL027_COMPLETED_20260922.md)，其中“当前唯一任务”等字样仅适用于已完成的027。
