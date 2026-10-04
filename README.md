# PreGAN+ / FT-MoE 在线实验

目标：在合理部署场景中，通过结构修改，使保留动态新增、知识保存/选择、休眠复用及容量管理思想的完整D优于持续学习C。

**最新状态：[Protocol-044 revision 1](docs/PROTOCOL044_RESULTS.md) 工程阶段通过，但科学Stage S因CI超时后精确checkpoint未持久化而阻塞关闭。**

Stage E: E_gate=true，19/19恢复case通过，内存有界与043固定轨迹复核通过。Stage S: seed4401生成最后完整边界next_t=5200；run 37184889231命中350分钟超时且无取消后artifact，因此exact resume不可用。登记禁止restart-from-zero，四条训练序列均未启动，scientific optimizer.step=0。

[执行交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [044结果](docs/PROTOCOL044_RESULTS.md) / [044冻结方案](docs/PROTOCOL044_CLOSEOUT_AND_CAPACITY_20261004.md)

Protocol-044 到此停止，不自动启动045。
