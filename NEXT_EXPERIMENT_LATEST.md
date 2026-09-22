# 当前唯一任务：Protocol-027两组试跑

2026-09-22用户已授权明确登记新数据版本，解除“必须复现不可取得的旧文件哈希”的限制。**采用protocol027_data_revision_002；旧版本不被改写为成功。**

从main读取[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)，执行[同一份单任务指示](docs/PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)。只运行C_fixed5与D_dynamic，seed700/model1，完成后停止。

工作流手动启动：run_models=true。首次frozen_data_run_id留空，自动拼装现存数据、审计、完整归档后再运行两组；如已有完整冻结归档，填入其run ID直接复用。当前维护提交没有启动C/D。

不再寻找旧46b1…/fc3e…哈希，不重新模拟，不从分数选择数据；物理、因果、覆盖、特征一致性和冻结文件校验仍必须通过。
