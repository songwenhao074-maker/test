# 下一实验：Protocol-042 revision 2 固定窗口效用与互补接入

状态：仅登记，未实现、未运行（2026-10-03）。完整替代未运行的revision 1困难样本加权；旧预算不追加。

[完整执行指示](docs/PROTOCOL042_WINDOWED_UTILITY_20261003.md) / [机器登记](artifacts/ftmoe_online/protocol_042/plan.json) / [交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)。

固定顺序：U_parent复现041暖启动 → R_win128替换对照 → A_hist累积评分补充 → A_win128固定128成熟区间评分补充。主臂预先指定A_win128。64/256窗仅只读诊断，不择优冒充主结果。

近期效用=去掉专家后的issued BCE减live BCE；按全局时间过期、正负分开、支持不足为未知。保留有用旧专家，低效用先休眠，记忆可复用。三个新臂最多3驻留/2活动/1shadow、两次后续新生尝试；新残差零初始化，避免复制父贡献再叠加。

先修真实恢复和门控，冻结全部实现后执行。U复现失败不运行其他臂。四条完整序列总梯度上限2576，明确高于被撤销旧版预算，计入多活动专家成本；不永久删除、不加权、不增加seed/F，不自动下一协议。

最新完成仍为[041结果](docs/PROTOCOL041_RESULTS.md)，run37110969550；042无科学结果。
