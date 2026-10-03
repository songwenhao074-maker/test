# PreGAN+ / FT-MoE 在线实验

目标：在合理部署场景中，通过结构修改，使保留动态新增、知识保存/选择、休眠复用及容量管理思想的完整D优于持续学习C。

**最新完成：[Protocol-040 最小双专家动态池](docs/PROTOCOL040_RESULTS.md)，run 37098618766，结果标签 quality_retained_pool_not_exercised。**

[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [040紧凑证据](artifacts/ftmoe_online/protocol_040/runs/run_37098618766/) / [040冻结方案](docs/PROTOCOL040_TWO_EXPERT_POOL_20261003.md)

本轮只新训练一条D_pool2序列；C/B/D_keep/D_039不重训，F未加载。结果仅属于已观察seed3601开发证据，不能自动外推为跨流确认、复用因果优势、完整动态删除成功或整体部署加速。

Protocol-040到此停止。历史数据、源码、checkpoint和失败结果保留。[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
