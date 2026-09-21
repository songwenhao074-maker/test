# 当前唯一任务与阻塞

当前目标仍是[Protocol-027的一次C_fixed5/D_dynamic试跑](docs/PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)，先检验D/C，暂不加入A/B。每份指示只规划一个任务。

**仓库入口与特征错误已修复；目前仍被原登记数据的恢复哈希阻塞，C/D均未运行。** 不要直接重跑训练。

接手先读[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)和[整理及验证记录](docs/REPOSITORY_CLEANUP_20260921.md)。沿用main；旧计划、D_eligible/STOP和五方法runner不支配027。

先解决当前任务的数据恢复前置问题；原登记哈希和完整资格审计通过后，才用Protocol-027工作流的run_models=true运行这两组并停止。不可替换登记SHA，不自动追加实验。
