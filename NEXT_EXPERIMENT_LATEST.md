# 下一实验交接：Protocol-036

035完成，六窗AP +0.005737、全程AP +0.004534，登记标签recurrence_only。独立核对支持预测与因果批次，但局部召回、机制归因和新流稳定性仍需检验。

读取：
1. docs/PROTOCOL035_INDEPENDENT_REVIEW_20260930.md
2. docs/PROTOCOL036_CORRECTION_CONTROLS_AND_NEW_STREAMS_20260930.md
3. artifacts/ftmoe_online/protocol_036/plan.json 与 plan.sha256

执行分支：codex/protocol-036-correction-controls-20260930，基于84282143b849dd3b5684e7f879964a2ab485fab0。

给执行模型的指示：在用户要求执行本交接时，先修复实际SHA记录、真实断点恢复、因果访问证据和发布冲突处理，再冻结全部036代码与分析。A阶段只在700的035缓存上训练D_cal/D_lin；B阶段对3601/3602/3603三流各执行C_ref/D_cal/D_lin/原D_corr。共14条新训练序列、3条新流；不改阈值、不扫超参、不加记忆/路由，不按A结果改B。执行后完整报告每流结果、全部局部召回和预算，停止，不自动扩展。

仅有GitHub连接器时允许036一次性push helper调用正式workflow_dispatch；必须锁SHA/plan并查重，helper不训练。不再启动035。本次只发布指示，没有启动036。
