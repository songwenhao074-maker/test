# 当前执行交接：Protocol-042 revision 2

唯一有效入口：[完整指示](PROTOCOL042_WINDOWED_UTILITY_20261003.md)及[plan.json revision 2](../artifacts/ftmoe_online/protocol_042/plan.json)。旧困难样本加权方案撤销，不能运行U_uniform/H_hard或叠加旧预算。

1. 核验注册hash、036/041 raw及固定041源码。运行maintenance/validate_protocol042_plan.py；仅验证登记，不代表科学有效。
2. 在新042入口实现U_parent、R_win128、A_hist、A_win128。保留旧核只读；修订规则完整见文档，不从旧checkpoint或另一臂中段开始。
3. 合成测试实际覆盖多专家梯度、窗口出队、未知支持、加性消融、victim选择、资格/复用、原子保存和磁盘恢复；提交事件证据，不能只有all_pass。
4. 共同冻结四臂、分析器/schema和execution SHA。先执行计费的U；双层fail-closed确认严格复现041 W_parent，才按序运行R/A。后臂不因前臂科学负结果而被取消，也不能据此调整。
5. 主臂固定A_win128；全局时间128窗只读过去issued值，等待标签成熟；正负贡献保护；有用旧专家可保留并加入互补残差。休眠保留同ID Adam，永久删除为0。64/256只读诊断不产生新科学序列。
6. 四序列总<=2576专家optimizer steps。最多两次后续新生尝试/新臂，第二次需冷却256成熟区间并重新满足压力。不增加窗口扫描臂/seed/新流/F/加权。
7. 封存raw，独立重算全部控制评分和AP/BCE/混淆矩阵；发布docs/PROTOCOL042_RESULTS.md及revision2证据。报告窗口与接入机制是否实际触发，不用D>C代替机制有效。
8. 精确恢复外禁止从零重跑；分析/发布恢复只读原产物。交付或阻塞后停止。

执行分支codex/protocol-042-windowed-utility-20261003；执行者创建protocol042-windowed-utility.yml，workflow_dispatch-only，锁revision2、expected_execution_sha和跨run预算。本次只发布计划，未创建或启动workflow。
