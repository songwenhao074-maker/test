# 下一实验交接

当前：Protocol-038 已登记，尚未实现或启动。

先读 [Protocol-038 实验指示](docs/PROTOCOL038_BIRTH_INITIALIZATION_20261001.md) 与 [冻结计划](artifacts/ftmoe_online/protocol_038/plan.json)，运行 `python maintenance/validate_protocol038_plan.py`。

收到用户执行交接后，按顺序完成：已有037结果只读诊断 → 重建一次21步历史donor → 两条新序列D_bias/D_warm。两组均重置优化器，保持结构、出生规则、后续batch不变。新增科学更新上限725，B/F/D_zero仅复用；原始工件不可访问或不匹配就停止，不重训补齐。

本轮检验历史偏置与特征权重能否改善冷启动。donor成本必须计入，不能把效果接近F当成已经证明动态增删成功。完成后上传证据并停止，下一轮跨流、结构或休眠/删除实验需新登记。

037历史结果：[结果说明](docs/PROTOCOL037_RESULTS.md)。本次指示提交不启动训练。
