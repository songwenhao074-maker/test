> 历史修订记录：本页revision002已被[revision003（总9869步）](PROTOCOL031_PLAN_REVISION_003_20260926.md)替代。以下保留当时内容，当前执行以revision003为准。

# Protocol-031计划revision002变更记录
用户明确要求：在特定场景下D最优即可；将长驻业务与稀有回归场景构建加入下一目标。031 revision001尚无实现、注册结果或运行，本次在任何031模型结果之前替换计划。

| 项目 | 已替代revision001 | 当前revision002 |
| --- | --- | --- |
| 数据 | 历史027 revision002，5520步 | 独立新生成031 revision001，15468步 |
| 业务 | 六业务更替流 | U=S1、V=S3、W=S4，U/V短回归 |
| 主指标 | 九回归first100 | 六回归first128等权AP差 |
| 常规在线更新 | 每4步 | C/D共同每16步 |
| D新生提议 | 从600起每256步 | 从600起每1600步，限制同类专家重复占位 |
| reuse | 非阻塞、相似度排序、原质量验收 | 保留同一策略包 |
| 模型回放 | C/D各一次 | C/D各一次；先构建一个新冻结流 |
| 执行入口 | protocol031-nonblocking-reuse | protocol031-rare-recurrence |
| 科学范围 | 在旧流检验策略 | 在事前固定的条件性场景检验D/C，不是单因素比较或全场景最优 |

[旧指令](PROTOCOL031_SUPERSEDED_OLD_STREAM_DIRECTIVE_20260922.md)与[旧计划JSON](../artifacts/ftmoe_online/protocol_031/history/plan_revision_001.json)保留。旧030分析/025/027注册中的数据不可变约束不限制本次新场景；历史数据和失败记录本身仍不得覆盖。

本次只修改计划、场景登记与交接，不实现代码、不生成数据、不启动Actions。接手模型按当前[唯一指令](PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)执行。
