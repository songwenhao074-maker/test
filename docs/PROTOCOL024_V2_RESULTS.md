# Protocol-024 next_round_v2 实验结果

更新：2026-09-16。v2a/v2b均已运行完毕，原“尚未写入结果”为历史登记状态。

v2a首预算0接受，第二预算4接受，复现AP等权平均增益+0.0035665；v2b出生2、退役3、复用2、硬删除0，
复现AP增益+0.0017037，全程AP低于C。两者development_signal=false，确认种子未使用。

- [原始v2b结果摘要](../artifacts/ftmoe_online/protocol_024/next_round_v2/results.md)
- [2026-09-16审阅与实现问题](PROTOCOL024_REVIEW_20260916.md)
- [下一步执行指示](PROTOCOL024_NEXT_DIRECTIVE_20260916.md)
- [stdout证据摘录](../artifacts/ftmoe_online/protocol_024/review_20260916/v2b_from_job_stdout.json)

## 历史登记说明

# Protocol-024 next_round_v2 实验结果

状态：v2a 已注册，尚未写入运行结果。

固定输入：replay seed700 / model seed1；immutable stream SHA256 `468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`；4980 scored + 1 guard；不重跑模拟器。

本轮先执行 v2a 周期性后台候选。首版为 64 个不同成熟 interval 各一次 shadow 更新，随后 32 个全新 interval 因果验证；若首版 0 accepted，才执行预注册的 `buffer128_4x4` 第二预算。只有至少一个 candidate 因果验收通过才进入 v2b specialist memory/retire/reactivate。

确认种子 701–703 与测试种子 201–205 继续封存。上一轮 v1 阴性结果不覆盖。
