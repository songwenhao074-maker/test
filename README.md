# PreGAN+ / FT-MoE 在线实验

目标：通过结构修改，使保留动态新增、知识保存/选择、休眠复用及容量管理思想的完整D优于持续学习C。

**下一轮已登记、未运行：[Protocol-042 暖启动候选的成熟困难样本加权](docs/PROTOCOL042_MATURE_ERROR_WEIGHTING_20261003.md)。**

比较U_uniform与H_hard：两者均继承当前E0权重并重置Adam，只改变候选16次训练的逐样本BCE权重。权重依据过去实际发行的live预测误差，标签须成熟，原始权重1到3并归一化；验收始终使用原始未加权未来指标。先修工程与门控，再复现U，通过后运行H。两臂梯度总上限736，F暂缓。

[执行交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [042机器登记](artifacts/ftmoe_online/protocol_042/plan.json) / [当前上下文](PROJECT_CONTEXT_LATEST.md)

**最新完成：[Protocol-041](docs/PROTOCOL041_RESULTS.md)，run37110969550。** 暖启动候选明显优于零初始化，但仍未超过活动专家；候选被拒绝，系统与040相同，双专家池未实际形成。[041紧凑证据](artifacts/ftmoe_online/protocol_041/runs/run_37110969550/)

本次只发布042指示，后续由用户交执行模型。历史033–041预算关闭；所有结果仍属已观察seed3601开发证据，不是跨流确认、复用因果优势或整体部署加速。

[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
