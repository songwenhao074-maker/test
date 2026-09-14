# Protocol-024 最新结果审阅与追加验证

审阅基点：`protocol-024-continuation-gpt56 @ 674c6267dc9e3aec268c1da186278f77d45224f5`。审阅日期：2026-09-14。

**结论：C/D 共同在线起点已经得到更充分的验证，但尚无 D 动态生命周期带来性能收益的实验结果。下一轮应修复实际接线与恢复缺口，随后交付一次生命周期开启的开发 pilot，不再停留在容器测试。**

当前执行指示：[下一轮实验指示](PROTOCOL024_NEXT_DIRECTIVE_20260914.md)。本文件是审阅记录，不修改此前结果。

## 1. 哪些结果已经可靠

GitHub [Actions run 34800910318](https://github.com/songwenhao074-maker/FT-MoE/actions/runs/34800910318) 的 job `103843227377` 已独立查询，结论 success。原 24-step smoke 的 C/D 各更新 6 次、输出完全一致，确实关闭了“D 因历史 arm==C 条件而不更新”的接线错误。

本次在本地运行现有三组 Protocol-024 单元测试，**15/15 PASS**。同时将 lifecycle-off 等价性检查延长到 **1140 intervals**：

| 项目 | 追加验证结果 |
|---|---:|
| 覆盖阶段 | F0、A1_compute、B1_memory |
| 输入流 | 已保存的 calibrated replay seed700 |
| residual/model seed | 1 |
| 原始 fault host-step | 825，包含 CPU/RAM/Disk 三类 |
| C / D 在线更新数 | 285 / 285 |
| detection probability 最大绝对差 | 1.1921e-7 |
| diagnosis probability 最大绝对差 | 4.4703e-7 |
| 原冻结网络未改变 | 两臂 PASS |
| lifecycle | 关闭，4 个原专家，无 shadow/dormant |

原 24 步全部位于前 300 步的正常 F0；追加检查覆盖了异常样本与两个机制，证明范围更充分。这里是浮点容差内等价，不是“逐 bit 完全一致”，也不是 D 优于 C。没有生成新场景、开启动态策略或改变历史 checkpoint。

## 2. 已复现的问题

### P0：评估 helper 已修，session 的实际输出仍走旧代码

`Protocol024Session` 没有覆写 `phase_metrics()` 或 `probe_scores()`，仍继承 023 版本。前者继续 flatten 后算 onset，后者继续用旧诊断口径；阶段划分还固定在旧 `s4.PHASES`。

构造唯一 onset 在 t=301/host0，最高风险预测在 t=300/host0 的确定性输入：正确 same-host AP=**1.0**，实际 `session.phase_metrics()` 输出 **0.0625**。因此“新增修正函数”不能记作“最终评估链已修复”。

另有两个 onset helper 实现：`ftmoe_protocol024_eval.temporal_onset_metrics` 排除未知标签，session 中的 `same_host_onset_metrics` 则会把 −1 当正常。输入 `[0,-1,1]` 时前者有效行 0，后者有效行 2、AP=0.5。需要保留单一实现，并通过真实输出文件测试。

### P0：继承的保存函数可能覆盖 023 注册检查点

`Protocol024Session.save_checkpoint` 原样继承 023；它使用全局 `s4.CKPT_DIR` 和 `CHECKPOINT_NAMES`，目标仍是 `protocol_023/round2a/checkpoints/C_after_*.pt`，还写入 protocol=023。给 session 传新的 out_dir 并不能改变这个保存路径。

本次只做静态确认，没有调用该保存函数。下一轮必须先隔离路径、阶段表和 protocol metadata，再运行包含 checkpoint 的实验。不能靠“目前 smoke 没调用保存”当作安全验证。

### P0：拒绝候选后的合法 ID 空缺无法恢复

复现：创建 shadow 4 → 拒绝 → 创建 shadow 5 → 激活 → export/restore。恢复报 `snapshot active topology cannot be reconstructed`。

原因是恢复函数按 `0..next_id-1` 顺序重新生长，错误地把已拒绝 ID 当成应存在的专家。当前 round-trip 只覆盖连续 ID。恢复应直接按 snapshot 中实际存在的 active/dormant/shadow 集合构建，next_id 仅作为下一次分配的单调计数器。

### P0（启用生命周期前）：优化器与行为哈希尚未覆盖动态状态

optimizer 在拓扑变化前构造，直接激活新专家不会自动加入 optimizer；复现中新专家 6 个 tensor 均不在原 param_groups。完整集成仍待实现，不能只给它设置 requires_grad=True。

仅变 ramp 就能使输出最大变化约 **0.05423**，但 `active_state_hash()` 不变。模型的 `learner_state_hash()` 也只哈希 tensor state_dict，不包含普通 Python ramp/ID 顺序。动态预测版本、checkpoint 和因果 ledger 必须包括真正决定前向的 topology/ramp 状态。

### 方法边界：retire 是休眠，不是释放总容量

4 个起始专家扩到 8 后，retire 一个，状态变为 active=7、dormant=1；再次创建 shadow 仍报 `expert capacity reached`。这是当前容量定义的必然结果，不是内存泄漏，但不支持“不断删除旧专家后学习更多新模式”的主张。

若研究删除，需要真正移除 dormant 权重、router 行及 optimizer state 的 purge；若只研究休眠，应把 claim 写成活动计算节省与知识复用，不能称已释放模型总存储。

### 稳健性：ramp=0 的有限精度连续性并非无条件成立

新候选 router logit=1000、ramp=0 时，先 softmax 后乘 ramp 会使老概率下溢，再触发 `all active routing ramps are zero`。这是极端数值压力用例，并非已证明实际数据会出现，但它说明已有“ramp=0 连续”测试覆盖有限。可在 log 空间中加入 log(ramp)，对 ramp=0 明确 mask，然后稳定归一化。

当前公式是 ramp 加权后重新归一化的 mixture。只有对冻结主干而言是 residual addition，新增专家仍会改变旧专家权重。不可将其描述为旧专家贡献始终不变的独立加法。

## 3. 对实验方向的建议

不再要求出现负梯度或 C 永久无法收敛才测试 D；也不需要第一轮就同时完成 8 模式、全部消融与确认种子。优先完成三个模式的真实动态 pilot，再根据实际缺口扩展。

原方案说改变“未来 response law”，却沿用 tol1 当前检测目标，这可能仍让模型只学习共同压力修正。本轮建议在**新的场景比较**中明确注册下一时刻 raw fault prediction，C/D 使用相同目标与标签延迟。旧 tol1 等价性检查保留为工程回归，不与新预测 AP 混表。具体目标、预算和场景见下一轮指示。

允许合理有利于 D 的首次长期学习、短周期再现和小 replay 等条件；需要实际记录额外 shadow/qualification 开销和动态事件，不预先承诺 D 胜出。若生命周期未触发，先区分策略未启用、触发过慢与场景无差异；不能把空事件运行当成动态方法的有效性试验。

## 4. 证据位置与本次边界

- `artifacts/ftmoe_online/protocol_024/review_20260914/review_evidence.json`：最小复现及 1140-step lifecycle-off 对照。
- `artifacts/ftmoe_online/protocol_024/review_20260914/static_and_test_audit.json`：实际保存路径、15 项单测和 CI 核对。
- `maintenance/review_protocol024_20260914.py`：独立诊断脚本，拒绝覆盖已有审阅结果。

本次没有修补上述模型/评估实现；将修复任务写成下一轮可验收指示。旧 023 和 024 continuation 证据保持原样。
