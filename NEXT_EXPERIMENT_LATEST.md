# 下一任务：Protocol-040 最小双专家动态池

**状态：方案已登记；尚未实现、尚未启动。** 用户将把执行交给其他模型。本次只发布指示。

先读[完整执行指示](docs/PROTOCOL040_TWO_EXPERT_POOL_20261003.md)、[机器计划](artifacts/ftmoe_online/protocol_040/plan.json)与[GitHub交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)。

唯一实验：保留B=C+D_lin，最多两个动态专家、一个部署专家，检验因果新增/选择/重复休眠复用后D是否仍优于C。只新增一条D_pool2完整序列；缓存C/B/D_keep/D_039不重训。累计性能损失预算固定对D_keep。16次额外shadow更新是该一条系统序列的内部训练，不是另一条完整对照回放。

收到用户“执行Protocol-040”的交接后，按冻结指示实现、测试、运行一次并将结果发布到main；完成即停止。真实没有第二专家或复用时如实报告，不改阈值凑事件。

上一轮：[039结果](docs/PROTOCOL039_RESULTS.md)，run36982845152，progress_to_next_step（已见单流开发信号）。历史033–039已关闭，不重跑。F研究暂缓；本轮没有永久删除、新流确认或自动后续实验。
