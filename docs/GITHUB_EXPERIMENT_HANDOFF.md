# 当前交接：Protocol-031一次非阻塞复用D/C试跑

以main为准。030配对诊断已通过；用户最新指示授权下一单任务，旧“等待新指示”已经满足。

先读[030分析](PROTOCOL030_ANALYSIS_20260922.md)，再执行[031单任务](PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)和[plan.json](../artifacts/ftmoe_online/protocol_031/plan.json)。031尚未实现/运行。

唯一任务：将一位旧专家的复用验证与新生训练解耦，相似度用于排序、不作硬准入，保留全部预测质量验收；C_fixed5和D_nonblocking_reuse各一次，共两次完整回放。按照指示处理同点接管、birth取消、角色改变及流尾截尾，不能借复用突破驻留容量或读取未来标签。

原九回归窗口为主指标，S6_first只能次要报告；复用次数和预测优劣分开，不能用oracle当在线结果。旧数据/代码/注册/负结果保留。不自动加A/B、其他种子或场景。

执行模型先实现独立031入口、登记、必要检查和结果序列化fixture，再经专用push工作流一次启动；报告故障从已保存产物恢复，不重跑模型。结果回写main并同步README/AGENTS/NEXT/PROJECT_CONTEXT/当前handoff后停止。

[030完成时交接](GITHUB_EXPERIMENT_HANDOFF_PROTOCOL030_COMPLETED_20260922.md)仅为历史，不阻止本次新任务。
