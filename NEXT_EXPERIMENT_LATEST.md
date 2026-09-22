# 当前唯一任务：Protocol-031非阻塞复用D/C试跑

状态：planned_not_implemented_not_run。030已完成并通过同机旁路无干扰检查，不再重跑复现。

[030分析](docs/PROTOCOL030_ANALYSIS_20260922.md) / [031执行指示](docs/PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md) / [计划](artifacts/ftmoe_online/protocol_031/plan.json)。

唯一任务：实现“相似度只排序、复用验证与新生训练并行”的策略包，运行C_fixed5与D_nonblocking_reuse各一次。旧专家仍须通过原1%损失改善及全部正常/FPR/guard验收才接管。数据、seed和九窗口主指标不变。

不要把030五次S6_first事后机会当作在线成功，也不要为了复现五次而指定专家或阶段。实际复用、回归AP领先与开发参考分别报告。先验证JSON产物序列化，避免重复030报告故障。结果回写main并停止，不自动追加实验。
