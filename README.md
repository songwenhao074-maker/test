# PreGAN+ / FT-MoE 在线实验

目标：在合理部署场景中，通过结构修改，使保留动态新增、知识保存/选择、休眠复用及容量管理思想的完整D优于持续学习C。

**当前下一任务：[Protocol-040 最小双专家动态池](docs/PROTOCOL040_TWO_EXPERT_POOL_20261003.md)。方案已登记，尚未实现或启动；由用户交接给执行模型。**

[执行交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [机器计划](artifacts/ftmoe_online/protocol_040/plan.json) / [最新已完成结果：039](docs/PROTOCOL039_RESULTS.md)

040保留B=C+D_lin，在其上最多保存两个动态专家、一次只部署一个，检验新增、选择与重复休眠复用能否保住D>C。只新训练一条D_pool2序列；缓存控制不重训，累计性能损失预算固定对D_keep。F研究、新流确认与永久删除不在本轮范围。

039在已观察seed3601上，D相对C全程AP+0.018781、六窗+0.012708；一次休眠/唤醒后保持优势，但只验证一个动态专家的一次周期。不能据此宣布完整动态系统或跨流优势已确认，也不宣称整体部署加速。

历史数据、源码、checkpoint和失败结果保留。[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
