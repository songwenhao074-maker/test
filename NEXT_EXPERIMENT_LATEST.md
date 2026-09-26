# 当前唯一任务：031 revision003——构建稀有回归场景并做一次D/C试跑

状态：planned_not_implemented_not_run。用户目标是在明确特定场景下体现D优势，不要求旧场景或所有场景获胜。

[唯一执行指示](docs/PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md) / [场景](docs/PROTOCOL031_RARE_RECURRENCE_SCENARIO.md) / [计划JSON](artifacts/ftmoe_online/protocol_031/plan.json) / [变更说明](docs/PROTOCOL031_PLAN_REVISION_003_20260926.md)。

新场景：F0 300；U/V初学各1600；W长驻1600；六次U/V交替回归各128，中间W各800，共9868计分步＋1保护步，总9869步。U/V/W采用S1/S3/S4物理规律。C/D常规更新均每16步、训练重放64、标签t+2；D新生周期1600，非阻塞复用验收保留。

只构建一个新冻结数据流并跑C_fixed5与D_nonblocking_reuse各一次。主指标为六回归first128等权AP差，报告误报、W退化和总成本。尚未证明D最优；不从未测A/B作推断。

旧031未执行方案已归档。旧“只用027 revision002/不重新模拟/九窗first100”不再是当前任务限制。数据审计失败或未形成有用记忆必须如实报告，不换seed/场景重试。完成或明确阻塞后写回main并停止。
