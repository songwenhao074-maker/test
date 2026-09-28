# 当前唯一任务：Protocol-033，5969行完整回归实验

状态：planned_not_implemented_not_run。用户要求将9869行长流缩短到约6000行，后续由其他模型执行。

读取[完整执行指示](docs/PROTOCOL033_6000_STEP_DIRECTIVE_20260928.md)、[时间表与生成登记](artifacts/ftmoe_online/protocol_033/scenario_registration.json)、[预算](artifacts/ftmoe_online/protocol_033/plan.json)。

5968步计分＋1行末尾标签支持；F0=300，U/V/W初学各1000，U/V各三次128步回归，五个W间隔各380。新生周期改为600+1000k成熟步，其余沿用032。按200行分段、独立进程组装，记录RAM峰值。全六窗资格通过后C_fixed5/D_guard_budget各一次。

实现来源：protocol-031-rare-recurrence-20260922 的 a215c1161adba4ff29514eb88806762c2f068023；main仅有计划与历史实现，不能替代该来源。先同步033指示，再实现和手动启动。旧031/032数据与结论保留，不截旧流、不复用旧断点、不追加种子。本次只上传方案，未生成或训练。
