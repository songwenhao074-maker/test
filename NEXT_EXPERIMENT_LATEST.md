# 当前唯一任务

继续 [Protocol-027两组试跑](docs/PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)，先读 [当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)，使用默认分支main。

run35596448071因开头补齐窗口的9维特征不一致停止；C/D都未运行。修复已提交，维护审计结果以 [结果页](docs/PROTOCOL027_RESULTS.md) 为准，不能把代码修复写成模型成功。

当前工作流默认仅审计；手动选择run_models=true才运行C_fixed5、D_dynamic。两组结果交付后停止，每份指示只安排一个任务。A/B、其他C、消融与种子搜索后置。
