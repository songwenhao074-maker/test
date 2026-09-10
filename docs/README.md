# 文档索引

日常只需先读 [项目最新上下文](../PROJECT_CONTEXT_LATEST.md)。它是唯一的当前状态入口，其他文档分别承担方法、历史、结果或维护记录，不再各自维护一套“当前主线”。

| 需要了解的内容 | 入口 |
|---|---|
| 当前状态、数据、模型、待办、运行限制 | [PROJECT_CONTEXT_LATEST.md](../PROJECT_CONTEXT_LATEST.md) |
| 用户已验收的版本、完整快照与回档方法 | [BASELINE_ACCEPTED_20260905.md](BASELINE_ACCEPTED_20260905.md) |
| 在线微调实现、执行命令与阶段门槛 | [FTMOE_ONLINE_EXPERIMENT.md](FTMOE_ONLINE_EXPERIMENT.md) |
| 在线微调300步结果、性能停止与首次资源故障 | [FTMOE_ONLINE_PILOT_015.md](FTMOE_ONLINE_PILOT_015.md) |
| 原生/离线分布审计、归一化和图身份问题、待审批方案 | [FTMOE_ONLINE_DISTRIBUTION_AUDIT_015.md](FTMOE_ONLINE_DISTRIBUTION_AUDIT_015.md) |
| 保留原模型、恢复训练环境及连续长流的待审批方案 | [FTMOE_ONLINE_MATCHED_SIMULATOR_PROPOSAL.md](FTMOE_ONLINE_MATCHED_SIMULATOR_PROPOSAL.md) |
| 保留BWGD2、输入适配及近似异常分布的待审批方案 | [FTMOE_BWGD2_INPUT_ADAPTER_PROPOSAL.md](FTMOE_BWGD2_INPUT_ADAPTER_PROPOSAL.md) |
| 协议016适配小试结果、场景停止及改进措施 | [FTMOE_ADAPTED_BWGD2_PILOT_016.md](FTMOE_ADAPTED_BWGD2_PILOT_016.md) |
| 替代真实数据集、容量静态敏感性及异常生成方案评审 | [FTMOE_DATASET_AND_CAPACITY_REVIEW.md](FTMOE_DATASET_AND_CAPACITY_REVIEW.md) |
| 容量017与Google2011协议018的授权与执行方法（**两者均未完成**：017 确认运行在 1479/2000 步撞 4.5 GiB RAM guard 中止且无 `result.json`；018 数据未下载、未运行） | [FTMOE_CAPACITY_GOOGLE_EXECUTION_017_018.md](FTMOE_CAPACITY_GOOGLE_EXECUTION_017_018.md) |
| **协议020 当前执行顺序、新模式定义、数据隔离与成功判据** | [指令/FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md](../指令/FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md) |
| 协议020 R0–R6 方法规格与因果门禁 | [指令/FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md](../指令/FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md) |
| 协议020 交接（路线、产物索引、协作分工）与问题日志 P01–P36 | [指令/FTMOE_PROTOCOL020_HANDOFF.md](../指令/FTMOE_PROTOCOL020_HANDOFF.md)、[问题日志](../指令/FTMOE_PROTOCOL020_PROBLEM_LOG.md) |
| 协议020 R1 结果（8×2000 开发对照，未稳定超过 A）与 R0 验收 | [R1 实验记录](../指令/FTMOE_PROTOCOL020_R1_EXPERIMENT_20260909.md)、[R0 验收](../指令/FTMOE_PROTOCOL020_R0_REVIEW_20260909.md) |
| 协议020 阶段流程、门禁、环境约束与文件清单 | [FTMOE_ONLINE_PROTOCOL_020.md](FTMOE_ONLINE_PROTOCOL_020.md) |
| 协议019 执行说明与问题日志 | [FTMOE_ONLINE_PROTOCOL_019.md](FTMOE_ONLINE_PROTOCOL_019.md)、[019 问题日志](../指令/FTMOE_PROTOCOL019_PROBLEM_LOG.md) |
| 消融关系、训练公平性、现行选模及测试边界 | [实验规则](../FTMOE_END_TO_END_ABLATION_PLAN.md) |
| 协议 001–014 的状态及产物位置 | [实验索引](../FTMOE_END_TO_END_EXPERIMENT_LOG.md) |
| 哪些历史结果能复用 | [历史审计](../FTMOE_HISTORY_REUSE_AUDIT.md) |
| 五种子 A/B、确认结果 | [确认报告](../FTMOE_HISTORICAL_CONFIRMATION.md) |
| 综合分数为何选中早期 CPU 类预测 | [选模诊断](../FTMOE_HISTORICAL_SELECTION_DIAGNOSTIC.md) |
| 平均分、主指标、约束与 Pareto 选模的搜索依据 | [方法调研](MODEL_SELECTION_REVIEW.md) |
| 更早的逐级冻结实验 | [冻结历史报告](../FTMOE_ABLATION_FINAL_REPORT.md)，不可与全模型结果拼表 |
| 清理前的旧上下文、长文和零散文件 | [归档说明](archive/README.md) |
| 本次清理范围及验证结果 | [维护记录](MAINTENANCE_20260905.md) |

机器结果及配置的权威来源仍为 `artifacts/ftmoe_end_to_end/protocol_*.json`、各运行 `configuration.json`、`summary.json`、每轮日志、checkpoint 和比较结果 JSON。清理没有改写这些实验数据。

根目录训练、分析及测试脚本保留原路径，避免破坏 import、配置指纹和历史复现。当前消融阶段已经用户验收通过并归档至协议 014；在线阶段已推进到协议 020（R0-A 因果基础通过、R0-B 动态部署未通过、R1 固定修正完成 8×2000 步开发对照但尚未稳定超过 A），下一步是离线覆盖审计与新模式注册，尚未开始。最新状态一律以当前状态入口为准。
