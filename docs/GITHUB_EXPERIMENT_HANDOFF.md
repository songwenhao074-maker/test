# 当前交接：031 revision002稀有回归场景构建与单次试跑

从main开始，当前计划revision=2。用户已明确要求把新场景纳入下一目标，替换尚未运行的旧031；无需再询问是否允许新场景。

读[唯一执行指示](PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)、[场景](PROTOCOL031_RARE_RECURRENCE_SCENARIO.md)、[复用规范](PROTOCOL031_NONBLOCKING_REUSE_SPEC.md)与[plan.json](../artifacts/ftmoe_online/protocol_031/plan.json)。只做一个任务：生成/审计/冻结一个15468步新流，然后C_fixed5/D_nonblocking_reuse各跑一次，交付并停止。

主要新值：U/V/W=S1/S3/S4；六次128步回归；两组在线更新16、replay64、label t+2；D新生从600起每1600步一次；相似度只排序，16步未来标签验收可与birth并行，预测质量门槛不降低。

独立031 collector/audit/verifier/data_lock，不能套用旧027固定SHA和九窗口入口。旧数据只作历史，不复用为新场景输入。先数据审计，冻结真实哈希，再模型；必要源资产缺失则报告阻塞，不替换来源。新生周期只是限频，不保证有效专家一定出现；失败也必须交付。

使用protocol031-rare-recurrence专用push工作流（可断点恢复同一生成任务）。不要启动旧protocol031-nonblocking-reuse。做好JSON小型验收，报告故障从已存产物恢复，不追加模型重跑。结果回写main并同步README/AGENTS/NEXT/PROJECT_CONTEXT/当前handoff，结束后不自动做其他场景/A/B/种子。

[修订记录](PROTOCOL031_PLAN_REVISION_002_20260922.md)保留旧计划位置。历史029无效与030恢复证据保持原样；旧限制不覆盖本次用户明确授权。
