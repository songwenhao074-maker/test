# PreGAN+ / FT-MoE 在线实验

目标：在明确、合理的特定部署场景中检验动态残差专家D的优势，不要求D在所有场景获胜。

**当前唯一任务：[Protocol-031 revision002——构建稀有业务回归场景并做一次D/C试跑](docs/PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)。尚未实现、生成或运行。**

## 从这里继续

[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [场景设计](docs/PROTOCOL031_RARE_RECURRENCE_SCENARIO.md) / [计划JSON](artifacts/ftmoe_online/protocol_031/plan.json) / [计划变更](docs/PROTOCOL031_PLAN_REVISION_002_20260922.md)。

U/V首次各1600步，W长驻3200步；U/V之后交替短回归六次，每次128步，中间W各1600步，加F0共15468步。两组每16步更新、64步训练重放，D可保存少量旧专家并验证复用。生成一个新冻结流，C/D各跑一次；主指标为六回归first128平均AP差，并报告误报、W退化及资源开销。D额外记忆/后台计算如实披露，不从未测A/B推断最优。

## 历史事实

[030](docs/PROTOCOL030_RESULTS.md)同机旁路无干扰通过，后处理故障已恢复；[028](docs/PROTOCOL028_RESULTS.md)旧场景中D未领先C。旧结果保留。旧031“旧流＋九窗口”方案未执行，已被当前revision002替代；无需继续在旧流上追求D胜出。

保留模拟器、工作负载、checkpoint、recovery及代码依赖；输出按scenario_id/run_id隔离，大产物记录artifact与哈希。[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
