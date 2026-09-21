# 当前唯一任务与阻塞

当前目标仍是[Protocol-027的一次C_fixed5/D_dynamic试跑](docs/PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)，只允许seed700/model1；不加入A/B、其他固定拓扑、额外种子、消融或调参。

**当前不得运行C/D。唯一前置阻塞是：从原保全的精确state_5520无法重新得到登记stream/final chunk SHA。**

最新复现已经进一步排除main代码问题：旧成功run35596448071按同一commit/workflow原样执行attempt 2，也得到当前稳定的 `fdea8430.../21cd7335...` 而不是登记 `46b1dbdd.../fc3e9887...`。保全state_5520已证明直接来自原run35254809015的overlay-5520；当前final chunk又与9/16 revision1进度artifact里的孤立final chunk逐字节完全一致。详细证据见[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)和 `artifacts/ftmoe_online/protocol_027/maintenance_20260921/recovery_reproduction_blocker.json`。

继续此任务时只允许寻找原登记完整副本，或定位并复现最后一个interval的运行时/硬件差异。只有原登记两个SHA真实通过并完成资格审计后，才可用Protocol-027工作流的 `run_models=true` 运行C_fixed5与D_dynamic一次并停止。不可替换登记SHA。
