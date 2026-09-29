# 下一轮实验：Protocol-035

当前状态：**执行分支已提交实现，main已发布手动工作流；本次修订不启动实验**。此前“算法未实现”的入口状态已过时。

权威入口：[能力覆盖诊断与保留C学习路径的最小纠错结构](docs/PROTOCOL035_C_PRESERVING_CORRECTION_20260929.md)。
机器登记：[plan.json](artifacts/ftmoe_online/protocol_035/plan.json)。先运行 `python maintenance/validate_protocol035_plan.py`。

执行分支：`codex/protocol-035-c-preserving-correction-20260929`；基于034完整执行代码，勿从仅同步报告的main或本地024旧checkout推断实现齐全。

执行顺序：核验033/034原始artifact → 固定历史能力诊断 → 一次C参考回放及因果tape → 一次2401参数纠错分支训练 → 原始产物归档、完整指标与结果回传。总计2条新完整训练序列，不新生成数据、不重跑旧D、不扫描参数。

本轮验证额外能力的来源。新生/休眠/受保护记忆/复杂路由留待后续另行授权。全程与回归能力分别判定，禁止用误报下降掩盖召回损失；负结果如实交付。

首次启动先读[启动方式修订](docs/PROTOCOL035_LAUNCH_AMENDMENT_20260929.md)。连接器无dispatch时，允许REST/gh；只有仓库写入能力时，允许创建一次性push启动助手并由其派发正式workflow_dispatch。外层ref=main，inputs.execution_ref=035执行分支。模板尚未激活；已有科学run则跟踪已有run，不重复启动。
