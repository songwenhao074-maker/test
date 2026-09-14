# Protocol-024：当前问题与下一步阻塞项（2026-09-14）

本文件只记录 **Protocol-024 continuation** 的问题，不修改 Protocol-023 已注册结论。

## 1. D 的模型公平性问题已经定位

Protocol-023 的固定 C 使用 `FrozenResidualFTMoE`：冻结 v4 base，外加 4 个 `64 -> 32 -> 5` residual experts 和一个 dense softmax router。Protocol-020 的旧 D-v3 则管理原模型内部 EAGate。直接把旧 `OnlineEAGateV3` 当作 D 会同时改变专家生命周期和模型/路由容器，因此不能作为公平主对照。

当前 `ftmoe_protocol024_dynamic_residual.py` 已实现同 residual bank 上的动态容器原型，并通过：t=0 等价、shadow 不参与部署、ramp=0 输出连续、retire/reactivate 保留 id、max=8 等测试。

**未完成：** 还没有接入完整 strict-prequential session。

## 2. 必须修复的评估问题

### 2.1 onset 时间轴
旧 `phase_metrics()` 在 `[time, host]` 展平后做一位 shift，会把相邻 host 当成未来时间。Protocol-024 必须用二维数组沿 time 轴、同 host 计算 h-step onset，并丢弃 phase 尾部未来不完整的样本。

### 2.2 尾部结算
严格 t-2 结算在在线循环结束时还剩 `steps-2` 与 `steps-1` 两条预测待成熟。旧 `finish()` 只结算最后一条，导致已保存的 023 A/C 都有 index 2878 的 16 个 host-step 未结算。Protocol-024 finalization 必须同时结算两条并 assert 全部 scored row 已成熟。

### 2.3 diagnosis 指标
CPU/RAM/Disk macro-F1 不应把 normal row 强制归入三种资源类别。至少同时报告：fault-positive-only resource macro-F1、注册 detection threshold 后的 end-to-end diagnosis、normal phase FPR/specificity。

## 3. DynamicResidualBank 接入前还缺的工程项

1. 与 C 完全相同的 prediction-before-label-before-update 顺序。
2. 与 C 相同 replay、anchor、supervised loss、optimizer family、update opportunity。
3. shadow optimizer 与主 learner optimizer 分离，candidate 不得影响 live forward。
4. candidate 只能用未来已经 matured 的窗口做 causal qualification。
5. topology 变化后 optimizer state 要可追踪；不能静默丢失旧专家的 Adam moments。
6. save/resume 必须包含 active/dormant/shadow ids、expert/router tensor、ramp、age、activation EMA、novelty/loss trigger、candidate validation ledger。
7. 加一条 resume equivalence test：resume 后下一次预测必须与不中断运行一致。

## 4. Trigger 不能直接沿用旧 EAGate 的数值阈值

旧 V3 的结构性规则可复用：candidate samples >=128、连续 novelty window、matured loss ratio、max expert count=8。但 Protocol-024 residual router 是 dense router，novelty 分布不同，所以 numeric threshold 要在新协议 seed700 stationary/baseline 上重新校准，不能从 D-C 结果倒推。

建议记录：router entropy、top1-top2 margin、matured supervised-loss EMA。禁止使用 `regime_id`、`phase_id`、`mechanism_id`、event id 或未来标签。

## 5. reactivation 匹配仍需冻结实现

可选方案：

- 把 dense router 每个 weight row 归一化后作为 expert key；或
- 对每个 expert 维护其主要路由到的 frozen 64-D `z` centroid EMA。

novelty centroid 与 dormant key 的 cosine 超过预注册阈值时恢复同一 id，否则在 capacity 允许时创建 shadow。旧实现的 0.90 只能作为初始参考，最终数值要在 seed700 冻结。

## 6. 新场景不能只是继续换 CPU/RAM/Disk 排列

023 的 mature gradient 大多同向，说明三个资源顺序最终仍在学习共享的“压力修正”。新场景应改变 **response law / future outcome**，同时让 clues 在预测时可观察。

推荐从业务模式进入/退出构造：短计算脉冲并快速释放、持续推理导致 working-set 增长、write-back 积压、paging/recovery、负趋势 batch drain、周期 checkpoint/upload 等。相似当前负载应可能因为不同历史趋势产生不同未来风险。

模型只接收共同的 12-step telemetry + scheduler context；mode/event id 全部 audit-only。

## 7. 新场景先检查 residual 输入 `z` 是否真的可分

023 的 H2 使用 49-D handcrafted features，不能证明 fixed-C 实际接收的 frozen 64-D `z` 保留相同 regime 信息。Protocol-024 在跑 D 前需要同时做：

- allowed raw/history probe；
- frozen 64-D `z` probe。

如果 raw 可学而 `z` 不可学，应给 C 与 D 同时增加相同轻量 temporal feature；不能只给 D。

## 8. 初次驻留时间和更新预算仍未冻结

最新复核建议第一次学习 800–1600 intervals、recurrence 120–240。先在 seed700 做 learnability/budget diagnosis，再冻结正式 pilot。不能沿用 023 的 420 first exposure 然后默认 D 一定能冷启动。

## 9. 主指标应在运行前固定

建议主指标：所有注册 mode switch 后前 W=100 intervals 的平均 detection AP，并报告每个 switch 的 paired D-C、worst-mode AP、full-stream AP。

若论文主张提前预警，则 corrected same-host raw onset AP/event recall 必须作为对应主指标之一；不能再用当前 fault-state AP 替代 warning ability。

## 10. 控制实验顺序

先做最小 A/C/D seed700 pilot。只有 D-C 出现有意义开发集差异后再增加：

- fixed sparse 8-expert pool；
- C-budget（与 D 额外 backward 数对齐）；
- D-no-birth；
- D-no-reactivation；
- D-no-retirement。

然后冻结全部规则，再使用 701/702/703 confirmation；不得在 confirmation 上调参。

## 11. 当前尚未产生正式模型结果

本轮已产生的是 **structural validation**，9/9 tests PASS；没有新 stream collection，也没有 Protocol-024 A/C/D 训练，因此不能把本轮写成“D 优于 C”或“新实验已经完成”。机器可读状态见：

`artifacts/ftmoe_online/protocol_024/continuation_20260914/structural_validation.json`
