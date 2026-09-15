# Protocol-024 next round：阶段结果与当前问题

更新时间：2026-09-15。

## 当前结论

本轮尚未得到可用于论文结论的 A/C/D 动态性能结果。当前已经完成的是工程链路继续执行、一次真实 4980-step response-law 长跑的故障定位，以及不改变科学参数的修复与重跑启动。

### 已确认

- 当前执行分支：`protocol-024-next-round-gpt56`，基于 main 的 Protocol-024 下一轮指示继续实现。
- 下一轮 runner、动态 residual/lifecycle、response-law generator、learnability/budget/pilot 入口已存在于该分支。
- 2026-09-15 的 generation run `34918496058`：依赖安装成功；4 个 response-law 单测全部通过；seed700 生成过程运行约 11039 秒。
- 该 run 在主 stream 已写盘后、最终 per-response-law audit 构造阶段失败：`NameError: SCORED_STEPS is not defined`。
- 根因是纯符号拼写错误：注册常量为 `SCORDED_STEPS=4980`，最终审计块有一处误写成 `SCORED_STEPS`。
- 失败 run 上传了 artifact `10381312155`，包含 3 个中间文件；因此这不是 simulator 未启动或早期 OOM，而是收尾审计错误。
- 修复提交：`4bab9047f0494372b2540f9a622ca7ccd427d5ac`。修复未改变 seed、timeline、response-law 数值、scheduler、label rule、模型输入或训练超参数。
- 修复后 push 已启动 generation run `34951444062`；记录本文时状态为 `in_progress`。

## 当前未完成

以下项目不能标记为完成，也不能据此声称 D 优于 C：

1. seed700 的 finalized `stream.npz + manifest.json + audit.json` 尚未由成功 workflow 产出并通过最终 hash/label 重算审计。
2. learnability probe 尚未在 finalized 新流上完成。
3. C 的 update_every=4 / update_every=1 预算比较尚未完成。
4. lifecycle-on A/C/D pilot 尚未完成；因此暂无 birth/reactivate/retire/purge 的真实性能对比证据。
5. `development_signal` 与确认种子 701-703 均未执行；confirmation_run 仍为 false。

## 问题记录

完整问题表见：
`artifacts/ftmoe_online/protocol_024/next_round_v1/problem_log.json`。

新增问题 `P24-NR-06` 记录了 run `34918496058` 的收尾 NameError、修复 commit、证据与重跑 run。此前 P24-NR-01~05 主要是 GitHub-hosted Ubuntu 的路径/优先级/磁盘/RAM guard 适配问题。

## 对下一位模型的直接建议

不要重新设计场景或修改 response-law 参数。先检查 run `34951444062` 的结论：

- 若成功：下载 generation artifact，核对 manifest SHA、4980 scored + 1 guard、raw label 由 overload ratio 可重算，然后进入 `protocol024-next-round-valid.yml` 的 learnability/budget/A-C-D 链路。
- 若仍失败：优先修复确定性的工程错误，并保持 registration/lifecycle_config/response_law_config 不变；不要使用 701-703 确认种子绕过开发流问题。
- 只有 lifecycle-on D 真正发生出生/复用/退役等事件并完成同流 A/C/D 预序贯比较后，才讨论 D-C 性能差异。
