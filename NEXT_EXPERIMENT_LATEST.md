# 下一轮实验入口（2026-09-16）

执行 [v2c与后续情形指示](docs/PROTOCOL024_NEXT_DIRECTIVE_20260916.md)，先读 [v2结果审阅](docs/PROTOCOL024_REVIEW_20260916.md)。

v2b已完成：birth=2、retirement=3、reactivation=2、purge=0；复现first100等权AP增益+0.0017037，全程AP略低于C，development_signal=false。不是上一轮的零触发状态。

本次发现birth训练/验收模型与实际替换部署拓扑不一致，已运行真实bank合成反例。先进行一次v2c修复重放与限定诊断；若收益仍小，推进已说明的Protocol-025服务进入/退出/复现与容量约束情形。不要继续无界扫阈值。

只用seed700/model1，确认701–703及测试201–205继续封存。旧v1/v2结果不覆盖。本次审阅未重跑性能实验。
