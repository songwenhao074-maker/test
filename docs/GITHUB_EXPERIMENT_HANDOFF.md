# 当前交接：Protocol-044 revision 2 恢复方案（未执行）

从 [最新入口](../NEXT_EXPERIMENT_LATEST.md) 开始，执行 [完整指示](PROTOCOL044_REVISION002_DURABLE_RECOVERY_20261005.md)。

本次原因是生成任务超时且检查点只保存在runner本地；不是D/C性能结论。另有C入口参数错误和生产resume门控错误，需要先修复、用实际入口验收。

r1报告 [PROTOCOL044_RESULTS.md](PROTOCOL044_RESULTS.md) 保留不变；r2报告写到 `PROTOCOL044_REVISION002_RESULTS.md`。当前没有r2结果文件，不得冒充完成。

执行模型收到用户交接后可进行登记内恢复：旧状态优先；确无旧状态时仅一次同seed重建；每段远端持久化；原四条序列、2360总梯度与阈值不变。不要再受旧交接“无新许可不能恢复”的文字阻塞，本修订就是新的有限恢复授权。不得由本次文档提交自动触发实验。
