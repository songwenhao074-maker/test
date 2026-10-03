# 当前执行交接：Protocol-041 已登记，未运行

执行模型在收到用户交接后，按 [Protocol-041 完整指示](PROTOCOL041_PARENT_INITIALIZATION_20261003.md) 和 [plan.json](../artifacts/ftmoe_online/protocol_041/plan.json) 实施。先运行 maintenance/validate_protocol041_plan.py；本次发布不含科学运行器或工作流，不应误认训练已启动。

1. 从040实际checkout 562979a1a826de828f7d2bdc0017b3993c8623a1取得并核验历史内核，按plan锁下载036/040输入；不加载F，不从040 checkpoint续训041。
2. 新建041两臂共用的运行器/分析器。Z_zero与040算法相同；W_parent仅在第二候选创建时、同t更新前深复制active E0参数，候选Adam为空。
3. 先修复子步骤持久化和恢复；完整执行生产入口合成恢复、竞争事件、未来扰动、实际预算和AP fixture。不能用硬编码True或出生前零梯度前缀代替。
4. 冻结两臂及分析器execution SHA，再执行Z_zero一次（计科学预算）。逐项复现040；失败保全并停止。通过后执行已经冻结的W_parent一次，不看结果改方案。
5. 两臂每臂至多352 live+16 shadow梯度；同一seed3601、单次候选机会、原阈值/窗口/训练规则。禁止追加真实Adam继承臂、donor前缀、重试或参数扫描。
6. 先上传不可变raw，再完整复算指标并发布docs/PROTOCOL041_RESULTS.md和紧凑证据，同步main/执行分支指针。科学、分析、发布状态分开；发布失败不重训。
7. 两条登记序列完成或出现明确阻塞后停止，不自动做后续协议。

最新已完成仍是 [040结果](PROTOCOL040_RESULTS.md)，run37098618766；其历史预算关闭。041正式工作流由执行模型创建，仅workflow_dispatch，分支codex/protocol-041-parent-initialization-20261003并锁expected_execution_sha。重复启动不得重新分配科学预算。
