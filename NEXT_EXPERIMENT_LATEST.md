# 下一轮实验入口（2026-09-16）

Protocol-024 v2c 已完成，接受 run `35074464132`；全部工程 gate 通过，但六个 recurrence first100 的平均 D-v2c−C AP=-0.00009536，`development_signal=false`。详见 [v2c结果](docs/PROTOCOL024_V2C_RESULTS.md)。

按 [2026-09-16 指示](docs/PROTOCOL024_NEXT_DIRECTIVE_20260916.md)，已进入独立 Protocol-025：6 种物理响应服务、固定 5520 scored + 1 guard 时间线、服务进入/退出/复现、4 generalists、D 最多 1 live specialist、resident 上限 8。方法注册仍冻结为 `artifacts/ftmoe_online/protocol_025/method_registration.json`；同一活动预算主比较为 `D_dynamic - C_fixed5` 的全部注册 recurrence first100 等权 AP，同时保留原数据注册中的 `D-C_fixed4` 口径并列报告。

## Canonical Protocol-025 execution

当前唯一 canonical 执行分支是 `protocol-025-revision1-gpt56-20260916`。

在任何 Protocol-025 模型结果出现之前的 source audit 发现原生成器有两处“注册物理语义未真正进入曲线”的问题：S5 `periodic_workingset_release` 原代码只执行一个 36-step 周期且 `release_fraction=0.42` 未参与计算；S4 `cache_warmup_decay` 的 `floor_capacity_fraction=0.55` 未实际用于 CPU 衰减。依据预注册允许的最多一次 pre-model 数据修订，已提交 `artifacts/ftmoe_online/protocol_025/data_revision_001.json` 并消耗该唯一修订额度。S1–S6 的注册数值、timeline、seed700/model1、event probability=0.30、raw_next_fault、共同 9D 因果特征和五 comparator 均未因模型结果改变；此时尚无 Protocol-025 模型结果。

旧分支 `protocol-025-gpt56-20260916` 的 generation run `35101949642` 是 **pre-revision evidence only**，无论其最终成功与否都不得创建 `data_lock.json`、不得用于 C/D 模型实验。

revision-1 已通过两道 pre-model gate：

- method/real-forward gate run `35114510374`: **success**；
- revision-1 generation semantic preflight run `35114553811`: **success**。

revision-1 正式数据生成与 strengthened pre-model audit run：`35114741391`。只有该 run（或其仅工程恢复后的同 revision-1 成功 run）完成且所有 strengthened audit gates 通过，才允许创建 `artifacts/ftmoe_online/protocol_025/data_lock.json`，随后触发冻结的五 comparator：`C_fixed4 / C_fixed5 / C_fixed8_dense / C_fixed8_top5 / D_dynamic`。

`data_revision_001` 已用尽 Protocol-025 唯一数据修订额度；若 strengthened audit 出现科学数据 gate 失败，不允许再次修改 service law 绕过结果。确认种子 701–703 和测试种子 201–205 继续封存，直到 seed700/model1 的完整开发比较和所有适用条件对照结束后另有明确指示。
