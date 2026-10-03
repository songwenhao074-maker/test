# PreGAN+ / FT-MoE 在线实验

目标：在合理部署场景中，通过结构修改，使保留动态新增、知识保存/选择、休眠复用及容量管理思想的完整D优于持续学习C。

**下一轮已登记、未运行：[Protocol-041 第二专家当前父模型初始化](docs/PROTOCOL041_PARENT_INITIALIZATION_20261003.md)。**

比较Z_zero与W_parent：只改变第二候选初始化，检验复制当前E0权重并重置Adam能否解决冷启动、保留D>C并形成有用的双专家池。两条新科学序列合计梯度上限736；先完成工程预检及Z对040的复现，再执行W。F继续DEFERRED，不额外调参或追加候选。

[执行交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [041机器登记](artifacts/ftmoe_online/protocol_041/plan.json) / [当前上下文](PROJECT_CONTEXT_LATEST.md)

**最新完成：[Protocol-040 最小双专家动态池](docs/PROTOCOL040_RESULTS.md)，run37098618766，quality_retained_pool_not_exercised。**
040仍优于C，但第二候选被拒绝，只有E0反复睡醒。[040紧凑证据](artifacts/ftmoe_online/protocol_040/runs/run_37098618766/) / [038初始化复核](docs/PROTOCOL038_REVIEW_AND_HANDOFF_20261002.md)

本次只发布041指示；科学实现和执行由用户交接给其他模型。历史033–040科学预算关闭。结果均属于已观察seed3601开发证据，不是跨流确认、复用因果优势、完整动态删除成功或整体部署加速。

[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [许可证](LICENSE)。
