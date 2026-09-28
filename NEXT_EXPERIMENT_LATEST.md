# 当前唯一任务：Protocol-034，三层在线微调机制实验

状态：planned_not_implemented_not_run。用户授权后续执行者实现并检验三件事：活动专家继续学习、完整记忆保留旧能力、因果路由利用记忆。

读取[完整执行方案](docs/PROTOCOL034_THREE_HYPOTHESES_20260928.md)和[机器可读登记](artifacts/ftmoe_online/protocol_034/plan.json)。先运行 maintenance/validate_protocol034_plan.py。

输入固定为033 run36417604442的5969行artifact10968752245；禁止新生成。源码固定0463cd365b4cda5a50423b4fccc941fe8d074bbf；本次只提交计划，没有034实现、训练或结果。

预算：同环境顺序训练回放C_ref、D_frozen_ref、D_live各一次；后两层使用D_live只读快照和预测缓存，进行Full/Fragment配对诊断及两次固定因果路由评分。事后最佳专家只作诊断，不能报成在线结果。三次训练＋两次缓存评分后停止，不追加种子或调参。

执行后将紧凑结果、原始artifact索引和SHA256写回GitHub，同时更新main和实际执行分支的入口，不能仅把结果留在Actions artifact。
