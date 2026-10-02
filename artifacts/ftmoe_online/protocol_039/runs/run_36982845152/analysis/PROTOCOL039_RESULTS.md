# Protocol-039 结果

本轮只新增一条 D_sleepwake 科学序列；C、B=C+D_lin 与 D_keep 均为缓存控制，不重训。F 研究保持 DEFERRED，本轮没有加载或比较 F。

登记结果标签：**progress_to_next_step**。这是已见 seed3601 上的开发结果，不构成跨流或统计学确认。

## 三个预登记问题

- **完整 D 相对 C：True**。全程 ΔAP +0.018781；六窗 ΔAP +0.012708；late4 +0.011576；prefix32 +0.012336；正向复现窗 6/6；护栏 True。
- **生命周期管理相对 D_keep 保持：True**。全程 ΔAP -0.000471；六窗 -0.000775；late4 -0.000437；prefix32 -0.000791；护栏 True。
- **真实 cycle/resource：True**。birth=351，sleep=1599，wake=2255；sleep predictions=656；expert prediction-forward=4960 (<5616)，optimizer.step=311 (<352)。
- **progress_to_next_step_signal：True**。

## 贡献归因（只作解释）

- B−C：全程 ΔAP +0.011091；六窗 +0.007954。
- D_keep−B：全程 ΔAP +0.008160；六窗 +0.005529。
- D_sleepwake−B：全程 ΔAP +0.007690；六窗 +0.004754。
因此若 D>C 主要由固定 D_lin 保留而来，应表述为“候选完整系统保住静态增益并减少新增专家计算”，不能把全部 AP 增益归因于 sleep/wake。

## 工程有效性与成本边界

- 合成真实入口磁盘恢复、未来扰动、休眠零调用和终末实计数 fixture：**True**。
- 科学输入/出生时刻/D_keep 前缀一致性/真实调用计数/预算审计：**True**。
- 新科学 optimizer.step：**311 / 352 max**；相对 D_keep 少 41 次。
- 新专家 prediction-forward：**4960 / 5616 D_keep reference**；少 656 次。
- sleep 保留 74 参数权重与 Adam 状态，因此**不节省该专家模型内存，也不等价于永久删除**。
- 本轮没有同机完整在线端到端 B 重测，因此不宣称整体部署加速。
- Protocol-039 到此停止；不自动调阈值、加种子、永久删除或恢复 F 研究。
