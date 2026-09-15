# 下一轮实验入口

当前执行指示：[Protocol-024 next_round_v2](docs/PROTOCOL024_NEXT_DIRECTIVE_20260915.md)。

先读 [2026-09-15 结果审阅](docs/PROTOCOL024_REVIEW_20260915.md) 和 [v1 原始阶段结果](docs/PROTOCOL024_NEXT_ROUND_RESULTS.md)。

最新有效 pilot：seed700/model1，A AP=0.510442，C=D AP=0.667584；D candidate_created=0，development_signal=false。不能称 D 已优于 C。

下一步：复用不可变 stream，修正基线可得时点；注册周期性后台候选训练并因果验收，再检验专家保留、退役和快速复用。只有实际发生硬删除时才讨论其贡献。确认701–703、测试201–205继续封存。

本次审阅不运行新实验，不覆盖 v1 证据。下一模型直接执行 v2 指示，并将结果保存于独立 run_id。
