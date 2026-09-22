# PreGAN+ / FT-MoE 在线实验

目标：在业务更替与复现场景中检验动态残差专家D，同时严格保留负结果和诊断有效性边界。

**当前状态：Protocol-030同机成对旁路无干扰验证已完成并通过；暂无登记的下一实验。**

## 最新结果

[Protocol-030 Results](docs/PROTOCOL030_RESULTS.md)：在同一个Actions job、固定单线程确定性CPU配置下，原Protocol-028 D的`audit_off`与`audit_on`两次回放检测/分类概率逐元素完全一致，九窗口AP、全程AP、生命周期及最终拓扑一致，`paired_audit_valid=true`。

有效旁路审计发现5条满足全部原复用验收门槛的候选记录，全部位于S6_first、专家9，并同时受busy和similarity门槛阻挡。它们仍是post_hoc oracle诊断，不是实际在线复用或D>C证据。

相对历史Protocol-028，class probability最大点误差为1.430511474609375e-6，因此`historical_028_match=false`；Protocol-029历史无效判定保持不变。030未重跑C/A/B，未改变数据、seed、阈值或线上策略。

Actions父进程在两次回放结束后因numpy.int64 JSON序列化失败而显示failure；完整结果从已上传artifact离线恢复，没有启动第三次回放。

历史有效D/C性能结论仍见[Protocol-028 Results](docs/PROTOCOL028_RESULTS.md)。当前等待新的明确实验指示。
