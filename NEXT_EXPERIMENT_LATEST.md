# 当前唯一任务：Protocol-033，5969行完整回归实验

状态：`implementation_ready_not_run`。执行分支：`protocol-033-5969-gpt56-20260928`。目前尚未生成任何Protocol-033数据，也尚未运行C/D模型。

权威约束见 `docs/PROTOCOL033_6000_STEP_DIRECTIVE_20260928.md`、`artifacts/ftmoe_online/protocol_033/scenario_registration.json`、`artifacts/ftmoe_online/protocol_033/plan.json`；实现交接见 `docs/PROTOCOL033_IMPLEMENTATION_HANDOFF.md`。

实现已完成：独立033生成/断点身份、最多200行的连续流分段、30个chunk（29×200+169）、每20步RAM采样与80%软上限、生成退出后独立组装、独立资格审计、C_fixed5/D_guard_budget独立顺序回放、六个128步回归窗比较，以及每次D复用事件独立记录首次影响索引。

冻结源 `a215c1161adba4ff29514eb88806762c2f068023` 上的031/032科学实现未修改。启动前workflow会再次编译新代码、运行033静态测试和计划validator，并用git diff验证冻结历史文件未变；这些检查全部在第一条033生成数据之前执行。

启动入口：`.github/workflows/protocol033-5969.yml`，仅允许 `workflow_dispatch`。首次启动后，同一注册流的后续生成段和最终assemble/audit/C/D阶段由workflow自行dispatch。若资格审计或工程步骤失败，停止并保留数据/证据，不重抽seed、不调事件强度、不删除失败窗口、不自动增加模型回放。
