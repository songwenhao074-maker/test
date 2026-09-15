# 下一轮实验入口

当前执行链仍遵循：[Protocol-024：修复实际链路，交付首个动态 pilot](docs/PROTOCOL024_NEXT_DIRECTIVE_20260914.md)。

**最新续跑与问题交接：**[Protocol-024 seed700 恢复续跑：结果、阻塞与交接](docs/PROTOCOL024_RESUME_RESULTS_20260915.md)。机器可读审计见 `artifacts/ftmoe_online/protocol_024/next_round_v1/recovery_review_20260915.json`。

最新 generation-only run `34918496058` 已生成完整 4981-row seed700 `stream.npz`，但在生成结束后的 finalization 因 `SCORED_STEPS` / `SCORDED_STEPS` 拼写不一致失败。恢复分支 `protocol-024-resume-gpt56-20260915` 不重新生成科学数组，直接从不可变 stream 续接官方 learnability → budget → lifecycle calibration → lifecycle-on A/C/D pipeline；恢复 workflow run 为 `34948819437`。

已验证 stream SHA-256 为 `468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`，4980 scored + 1 guard，无 NaN/Inf，物理 overload ratio 可逐元素重算 raw labels。注册 13-interval 隔离的 raw-history learnability 子门禁已复现通过：history AP 0.86656、validation prevalence 0.08825；response_law_v1 不应因此修改，701～703仍禁止使用。

需重点保留的风险：全程 deployment rejection 14.68%、migration rejection 32.90%；R2 分别 22.99%/48.67%。即使后续 D−C 为正，也必须检查可学习性/收益是否被 rejection 与调度压力中介，不能直接升级为确认实验。

更早 C Rescue / 负梯度 / specialist 必要性门禁不再是禁止新协议实施 D 的通用条件。旧 tol1 检测结果保持独立，不与 raw-next-fault 新任务混表。
