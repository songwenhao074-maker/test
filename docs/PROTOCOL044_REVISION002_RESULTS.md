# Protocol-044 revision 2｜四组已完成，未通过完整机制验证

2026-11-24（Asia/Shanghai） · [GitHub Actions 运行 45236302789](https://github.com/songwenhao074-maker/test/actions/runs/45236302789) · [预登记方案](PROTOCOL044_CLOSEOUT_AND_CAPACITY_20261004.md) · [恢复规则](PROTOCOL044_REVISION002_DURABLE_RECOVERY_20261005.md)

**在 seed 4401 的这条模拟流上，动态模型 D 的检测表现好于固定在线模型 C。但新增专家没有接入，也没有发生永久回收；两条本应逐值一致的对照出现了微小数值差异。** 正式结论保留为 `D_over_C=true`、`S_validity=false`、`full_lifecycle_goal_completed=false`。工作流成功表示实验执行完毕，不等于假设全部成立。

## 检测：多检出 214 个故障主机时刻，少报错 150 个

5953 行原始数据含 1 行用于结算最后预测；实际评分 5952 个区间 × 16 台主机，共 95232 次主机预测。固定阈值 0.5。AP 衡量把故障排在正常情况前面的能力，越高越好。

| 同流模型 | AP | 检出/故障总数 | 召回率 | 误报/正常总数 | 误报率 |
|---|---:|---:|---:|---:|---:|
| C：固定在线模型 | 0.619658 | 6599/13621 | 48.447% | 4174/81611 | 5.115% |
| D_lin：共享线性修正 | 0.633498 | 6617/13621 | 48.579% | 3970/81611 | 4.865% |
| D_no_gc：禁用永久回收 | 0.642795 | 6813/13621 | 50.018% | 4024/81611 | 4.931% |
| D_bounded：完整 D | 0.642795 | 6813/13621 | 50.018% | 4024/81611 | 4.931% |

四个故障模式再次出现后的前 128 个区间，D 相对 C 的 AP 增益依次为 +0.04999、+0.04306、+0.01395、+0.01397，四窗平均 +0.03024。全流增益 +0.02314，各项预登记的性能保护条件均通过。共享线性修正本身已优于 C，不能把全部增益归于增删专家。[四组全程、F0、分阶段及复现窗口指标](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/four_arm_metrics.json)

## 生命周期：旧专家两度复用，两个候选都没接入

- 在区间 367 建立初始专家 E0；E0 在 1343、2991 休眠，在 1775、3183 重新激活。
- 在 1663 启动候选 E1。训练了 7 步后，E0 通过复用验收，E1 训练取消。
- 在 4495 启动候选 E2。训练 16 步后完成考核，4783 时未达到相对当前组合的损失改善门槛，未接入。
- 最多同时驻留 2 个专家，低于上限 3；最多活动 1 个。永久删除 0 次，因而没有“腾位置并接入新人”的事件，也没有删除前后窗口可比较。复用达标，但预登记的生命周期条件还要求初始专家之外至少一个候选被接纳并工作 64 个区间，所以该条件为假。[候选记录](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/D_bounded_candidate_decisions.jsonl) · [生命周期记录](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/D_bounded_lifecycle.jsonl)

## 审计：数值差异很小，但不是逐值相同

没有永久回收时，启用与禁用回收的两组应产生完全相同的轨迹。两组的活动专家 ID、版本和活动集合变化一致；从第 368 个预测起专家参数指纹及修正值开始不同，第 434 个预测首次出现概率差异。95232 次主机预测中 1889 个概率值不同，最大绝对差 5.96×10⁻⁸；阈值 0.5 下的最终报警 **0 个不同**。两组 AP 仅相差约 2.85×10⁻⁹。原始数值足以排除“只是报告显示位数不同”，但未定位到具体哪一步计算引入差异。[逐项量化](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/paired_precision_check.json) · [原始审计](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/divergence.json)

两条动态序列各 310 个效用评分、各 5 个候选或复用评估通过独立复算。输入、预算、资源约束通过；总优化步数 1412/2360（372、372、334、334）。两组数值不一致触发预登记的严格门槛，因此正式标签是 `invalid_execution`，不因误差小而事后改门槛。[正式状态](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/analysis_status.json)

## 数据与恢复

本轮从远端 5800 行状态恢复，生成并上传最后 153 行。首次立即回读新 artifact 遇到 HTTP 404；随后取回同一 5953 行状态并验证，没有重新生成。完整数据通过无模型验收后锁定，四组按 C_ref → D_lin → D_no_gc → D_bounded 顺序运行，每段均保存检查点。四组训练及独立分析已结束；没有训练重跑、换种子或启动 045。[输入审计](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/input_audit.json) · [原始产物索引](../artifacts/ftmoe_online/protocol_044/revision_002/runs/run_45236302789/artifact_index.json) · [r1 中断报告](PROTOCOL044_RESULTS.md)
