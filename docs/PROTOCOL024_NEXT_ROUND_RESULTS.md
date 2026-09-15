# Protocol-024 next round：阶段结果与当前问题

更新时间：2026-09-15。

## 当前结论

本轮已经把 seed700 response-law 新流从“模拟完成但 finalizer 崩溃”的状态恢复为可审计的 immutable generation package，并启动正式的 learnability → budget → lifecycle-on A/C/D 开发链路。**目前仍不能声称 D 优于 C**；性能结论必须等 downstream valid run 完成。

## 已确认的生成结果

- 执行分支：`protocol-024-next-round-gpt56`。
- 原 generation run：`34918496058`，主模拟过程运行约 11039 秒；4 个 response-law 单测通过。
- 原 run 在 `stream.npz` 已经写盘后，因 `SCORED_STEPS` / `SCORDED_STEPS` 拼写错误在最终 audit 阶段失败。
- 原 artifact：`10381312155`，其中 `stream.npz` 完整存在。
- immutable stream SHA256：`468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`。
- stream 维度已核对：4980 scored intervals + 1 guard row；`raw_labels=(4981,16)`，`host_features=(4981,16,7)`，调度、容量、overload 和 audit 数组均存在。
- raw label 可以严格由 `overload_ratio` 重算；phase vector 与 response-law vector 与冻结时间线一致。
- 从 audit 数组恢复到 1954 个 response-law event；未把 audit ID 用作模型输入。

## finalization 恢复

新增：

- `maintenance/finalize_protocol024_existing_stream.py`
- `.github/workflows/protocol024-finalize-existing-generation.yml`

finalizer workflow run：`34953715809`，**success**。

该 run 做的是纯 finalization：下载原 `stream.npz`，验证固定 SHA，重算 label/phase/law audit，生成 `manifest.json`、`audit.json`、`events_recovered.json` 和 `recovery_audit.json`，然后上传 artifact：

- artifact id：`10389922674`
- artifact name：`protocol024-next-round-v1-finalized-generation`

关键 provenance：`finalization_only=true`、`simulator_rerun=false`、`stream_bytes_mutated=false`。response-law 参数、seed700、timeline、scheduler、label rule、模型输入和训练超参数均未改变。

## 后续 workflow 接线问题与修复

原 `protocol024-next-round-valid.yml` 还有两个确定性工程问题：

1. 硬编码旧 generation run `34839589954`，不是当前有效来源；
2. 假设 artifact 解压后存在 `data/stream.npz`，但实际上传目录会展开为 artifact 根目录的 `stream.npz`。

因此新增 `.github/workflows/protocol024-recovered-valid-pilot.yml`，固定读取 finalizer run `34953715809` 的 immutable artifact，并正确复制 artifact 根目录到本次 `$ROOT_OUT/data`。

## 正在执行

正式 downstream run：`34953810201`。

当前 GitHub Actions 状态：

- immutable stream / provenance / label / anchor 校验：**通过**；
- 13-interval 隔离的 raw-history / frozen-z learnability probe：**完成**；
- 预注册 learnability gate：**通过**；
- 当前正在执行固定 C 的 `update_every=4` 与 `update_every=1` 预算比较；
- lifecycle threshold calibration 与 lifecycle-on A/C/D pilot 尚未开始。

learnability gate 已通过意味着 response_law_v1 至少满足本轮预注册的“可继续进入模型开发”条件；没有因为场景不可学而触发 response-law 修订。具体 AP/prevalence/z-gap 将在 run 完成并取得 artifact 后写入本文件。

后续顺序：

1. 完成 C update budget 选择；
2. 冻结 lifecycle loss threshold；
3. 在同一 seed700 / raw_next_fault 目标下运行 lifecycle-on A/C/D pilot；
4. 输出 `status.json`、`comparison.json`、各 arm predictions/checkpoints 与 D lifecycle ledger。

## 尚未完成

以下项目仍不能标记完成：

- learnability 的具体 AP / prevalence / frozen-z gap 数值落盘汇总；
- C 的最终 update budget；
- lifecycle-on D 是否实际发生 candidate birth / acceptance / reactivation / retirement；
- recurrence first-100 窗口中的 D−C AP；
- `development_signal`；
- 确认种子 701–703（本轮仍禁止使用）。

## 下一步判定规则

优先读取 run `34953810201` 的 artifact `protocol024-next-round-v1-recovered-valid-development`。

- 若后续 lifecycle 没有事件：根据 `lifecycle.jsonl` 区分 trigger 未发生、candidate 样本不足、qualification 失败或 cooldown/阈值问题。
- 若 lifecycle 已执行但 D≈C：比较同一 recurrence 群体上的专业化与部署前后 paired loss，不无限增加容量。
- 若 D 更差：先查 topology action 前后 logit jump、选择性训练、旧业务退化和 D 额外预算。
- 只有 lifecycle-on A/C/D 完成并有有效 recurrence coverage 后，才讨论是否进入 701–703 确认阶段。
