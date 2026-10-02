# 下一实验交接：Protocol-039

目标优先级：完整D优于C。F相关实验暂缓，不再作为当前推进门槛。

038科学训练已经完成，勿重跑；参见[独立复核与停止交接](docs/PROTOCOL038_REVIEW_AND_HANDOFF_20261002.md)。当前039仅登记，尚未实现或启动。

收到用户执行交接后：读取[完整039指示](docs/PROTOCOL039_D_VS_C_SLEEP_WAKE_20261002.md)与[冻结plan](artifacts/ftmoe_online/protocol_039/plan.json)，运行 `python maintenance/validate_protocol039_plan.py`；补齐真实入口的恢复、未来扰动和调用计数合成测试，然后实施唯一D_sleepwake新序列。

沿用零初始化出生与原预测结构，只加一次休眠/唤醒。C为主基线，缓存D_keep衡量管理损失，B仅解释贡献。最多352科学更新，不重训基线、不增加流、不做F/donor或永久删除。未触发也交付并停止，不改阈值凑事件。成功后也必须交付并停止，下一步另登记。
