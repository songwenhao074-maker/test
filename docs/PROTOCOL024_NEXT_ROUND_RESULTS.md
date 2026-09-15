# Protocol-024 next round：阶段结果与当前问题

更新时间：2026-09-15。

## 当前结论

seed700 / model seed1 的 Protocol-024 valid development pilot 已完成，GitHub Actions run `34953810201` 全部步骤成功。工程链路、immutable stream、learnability、预算选择、lifecycle threshold calibration 和 lifecycle-on A/C/D 都完成。

**本轮不能声称 D 优于 C。** 原因不是 D 性能下降，而是冻结 lifecycle trigger 在正式 pilot 中没有被真正触发：D 没有创建 candidate，因此 D 与 C 的预测和指标完全相同，`development_signal=false`。按预注册规则，本轮不使用确认种子 701–703，也不为得到有利结果事后修改 threshold。

## 生成与恢复来源

- 分支：`protocol-024-next-round-gpt56`。
- 原 generation run：`34918496058`；主模拟约 11039 秒后完成 stream 写盘，但 final audit 因 `SCORED_STEPS` / `SCORDED_STEPS` 拼写错误失败。
- immutable stream SHA256：`468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`。
- 4980 scored intervals + 1 guard row；`raw_labels=(4981,16)`、`host_features=(4981,16,7)`。
- raw label、phase vector、response-law vector 均通过重算审计；恢复 1954 个 response-law event。
- finalizer run `34953715809`：success；`finalization_only=true`、`simulator_rerun=false`、`stream_bytes_mutated=false`。
- downstream valid run `34953810201`：success；artifact `10391150168`，名称 `protocol024-next-round-v1-recovered-valid-development`。

## Learnability probe

13-interval isolation gap 满足要求，train/validation history 与 future target 不重叠。validation prevalence=`0.0882523`。

- persistence current fault AP：`0.602924`
- current pressure AP：`0.807905`
- current raw 7D logistic AP：`0.744792`
- raw history 12x7 logistic AP：`0.866506`
- frozen-z 64D logistic AP：`0.837763`
- history gain over current raw：`+0.121713`
- history gain over current pressure：`+0.058600`
- frozen-z gap to raw history：`0.028742`

因此 `raw_history_learnable=true`、`z_retains_history_signal=true`、`proceed_to_budget_and_pilot=true`。response_law_v1 没有因为不可学而被修改。

## 固定 C 的 update budget

比较规则预注册为：只有 u1 相比 u4 的 first-exposure mean AP 增益 `>0.05` 且 p95 update latency 有限时才选择 u1，否则选择 u4。

- u4：first-exposure mean AP=`0.481261`，full AP=`0.667584`，update p95=`0.04773 s`，总耗时约 `77.65 s`。
- u1：first-exposure mean AP=`0.474242`，full AP=`0.720371`，update p95=`0.05331 s`，总耗时约 `281.08 s`。
- u1−u4 first-exposure mean AP=`-0.007019`。

因此冻结选择：`update_every=4`。虽然 u1 full AP 更高，但不满足预注册的 first-exposure 选择条件，不能事后改规则。

## Lifecycle calibration

最终使用第二个且最后一个预注册 calibration config `loss_trigger_v2_pre_result_protocol_fix`。600 interval calibration prefix 包含 F0=300 和 R1_first 的前 300 intervals，避免旧 256-prefix 全落在零故障 F0 的协议问题。

冻结参数：32-interval matured supervised-loss window，p99 threshold=`0.7022458374191777`，连续异常窗口数=2；candidate train=64 intervals，validation=32；accept relative loss improvement>=1%，normal anomaly probability allowance=0.01；cooldown=64；retirement/hard-delete=false；无 forced trigger。正式 D 重算 threshold 与冻结值一致。

## Lifecycle-on A/C/D pilot

A/C/D 均使用同一 seed700 stream、model seed1、raw_next_fault target 和相同 anchor；C/D 使用 `update_every=4`。D 的 lifecycle 开启。

### Full detection

- A：AP=`0.510442`，Recall=`0.498050`，FPR=`0.039240`。
- C：AP=`0.667584`，Recall=`0.678463`，FPR=`0.038141`。
- D：AP=`0.667584`，Recall=`0.678463`，FPR=`0.038141`。

future-positive resource macro-F1：A=`0.764456`，C=`0.839168`，D=`0.839168`。raw same-host onset AP：A=`0.105217`，C=`0.160114`，D=`0.160114`。C/D 最差 response law 均为 R1，AP=`0.326094`。

### First-100 switch windows

所有 D−C AP 都为 0：

- R1_first：C=D=`0.169100`
- R2_first：C=D=`0.664624`
- R3_first：C=D=`0.610061`
- R1_rec1：C=D=`0.424176`
- R3_rec1：C=D=`0.674653`
- R2_rec1：C=D=`0.819272`
- R1_rec2：C=D=`0.544380`
- R2_rec2：C=D=`0.778640`
- R3_rec2：C=D=`0.698141`

六个 recurrence window coverage=1.0；mean D−C AP=`0.0`，positive D−C switches=`0/6`，normal FPR D−C=`0.0`。预注册 development rule 要求 mean>=0.03、至少 4/6 positive、FPR delta<=0.01，因此 `development_signal=false`。

## Lifecycle ledger：本轮真正的阻塞

D ledger 只有：`calibration_frozen=1`、`loss_window=136`。没有 candidate_created / accepted / rejected，没有 birth/reactivation/retirement/purge；最终 topology 仍为 active experts `[0,1,2,3]`，topology_version=0，extra shadow compute=0。

136 个 monitoring loss windows 中只有两个超过冻结 threshold：cursor 1529 的 mean loss=`0.740665`，cursor 2169 的 mean loss=`0.747093`；两次都只是 `consecutive_abnormal=1`，中间被正常窗口打断。由于预注册 trigger 要求连续 2 个异常窗口，**candidate 从未创建**。因此当前 artifact 中通用说明“candidate causally trained/qualified but none accepted”并不精确；ledger 证明更准确的原因是“trigger condition never reached”。

D 总耗时约 `125.31 s`，C 约 `80.09 s`；D prediction p95=`0.00524 s`，C=`0.00445 s`；D update p95=`0.05656 s`，C=`0.04907 s`。这些是 lifecycle monitoring/controller 的执行开销，但没有 shadow candidate train/validation compute。

## 当前科学判定与下一步

这不是确定性工程失败，因此不应修改当前结果后重跑同一个已注册 pilot，也不应直接进入 701–703 confirmation。当前结论是：response-law stream 可学，online C 明显优于 A，但当前冻结的 p99 + consecutive-2 lifecycle trigger 对该 stream 太保守，导致 D 的动态专家机制没有被 exercised，因而 D=C。

下一位模型应把问题作为 **lifecycle trigger sensitivity / protocol redesign** 分析，而不是把本轮包装成 D 优于 C。若要启动新的开发轮，应在新的预注册 protocol/version 中、在读取确认种子前明确制定 trigger 修订依据（例如基于本轮 ledger 的触发覆盖诊断），并保留本轮 negative development result；不得回写或覆盖本轮 `development_signal=false`。

确认种子 701–703：**未使用**。
