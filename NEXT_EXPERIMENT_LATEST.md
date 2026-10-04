# 当前执行指示：Protocol-044 revision 2 工程恢复

**状态：已登记，尚未执行。** 本次仅发布指示，未触发 Actions。

请执行 [完整修订指示](docs/PROTOCOL044_REVISION002_DURABLE_RECOVERY_20261005.md) 和 [机器可读计划](artifacts/ftmoe_online/protocol_044/revision_002/plan.json)。

已确认：r1 的数据生成到本地 5200/5953 行后因350分钟超时取消；中断上传仅在 failure 时运行，被跳过，远端artifact=0，四条模型序列均未开始。另发现正式 C 入口 make_c 参数数目不符，以及 validate_gate 阻止所有已开始序列恢复。

执行顺序：
1. 只读核查旧状态；修复实际入口和远端持久化，验证旧 E artifact 与源码兼容性。
2. 合成数据通过真实入口、跨进程/跨runner恢复及预算幂等验收，冻结r2代码。
3. 若有完整旧状态则恢复；若无，只允许一次同seed4401重建。先远端保存初始化t=0状态，每job至多200行，每段先上传、下载核验，再继续。
4. 完成5953行并锁定数据，顺序运行 C_ref、D_lin、D_no_gc、D_bounded，训练中途同样持久化。
5. 保持原2360总梯度预算与全部阈值，发布r2独立报告后停止。

原r1 plan/scenario原字节保留作为科学合同；它不是新的恢复授权入口。r1历史结果保持未完成，不覆盖。禁止换seed、反复从零训练、加实验或自动045。
