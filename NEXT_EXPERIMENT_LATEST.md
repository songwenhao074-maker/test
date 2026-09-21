# 下一步实验入口（2026-09-21）

唯一执行指示：[Protocol-027：六业务更替场景中的一次D/C开发试跑](docs/PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)。

只运行C_fixed5与D_dynamic，replay seed700/model1，复用Protocol-025 revision1的5520步数据。重点检验九个业务回归前100步的AP差，同时报告全程性能与实际开销。完成结果上传后停止，无自动追加实验。

本入口替代026五方法计划；旧协议与失败记录保留。027正常保护集沿用026定义，数据恢复只参考026第1节。登记：[registration.json](artifacts/ftmoe_online/protocol_027/registration.json)。

用户已明确：从现在起，每份实验指示只规划一个有交付和结束点的任务。先集中D对固定残差C，A/B后置；不跑其他对照、消融、确认或参数搜索。

当前仅完成任务设计和发布，尚未实现027入口或运行027模型。最新完整结果仍为024 v2c。仓库：songwenhao074-maker/test。
