# 当前项目上下文（2026-09-18）

目标：在合理现实、允许公开后台计算/有限存储不对称的在线场景中验证动态增删专家D的优势。

最新已完成模型结果为Protocol-024 v2c：birth5、retirement4、reactivation0、purge2，复现AP均值增益-0.00009536，development_signal=false。

Protocol-025新数据已生成，12/13强化审计通过；F0正常保护集没有正例导致原协议停止，尚无五方法性能结果。本次明确转入独立Protocol-026：保留原门禁失败，复用同一数据，仅重新定义正常保护集的用途及运行/报告口径。

执行 [新指示](docs/PROTOCOL026_NEXT_DIRECTIVE_20260918.md)，依据 [最新审阅](docs/PROTOCOL025_REVIEW_20260918.md)。
恢复材料已在run35292728416归档，包含next_t=5520状态，不需要重跑数小时模拟器。
仓库改名为songwenhao074-maker/test；确认与测试种子均未使用。
