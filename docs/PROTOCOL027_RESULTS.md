# Protocol-027当前状态

首次试跑 [run35596448071](https://github.com/songwenhao074-maker/test/actions/runs/35596448071) 停在数据资格审计，C_fixed5/D_dynamic均未运行。

具体阻塞：`runtime_9d_features_match_stored_causal_features=false`，最大误差0.11541324853897095。物理数据及最后chunk哈希、正常guard和九窗口覆盖通过。原报告的`execution_stopped_before_runner_status`只是缺失状态文件的占位，不是根因。

仓库已修复开头补齐窗口的斜率定义与失败状态持久化，19项针对性检查通过。完整真实流审计将在维护工作流中验证，尚不预先记为通过。审计通过也不等于C/D性能实验完成。

历史紧凑证据原样保存在 `artifacts/ftmoe_online/protocol_027/runs/run_35596448071/`。当前唯一任务与入口见 [交接说明](GITHUB_EXPERIMENT_HANDOFF.md)。
