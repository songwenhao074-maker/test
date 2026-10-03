# 下一实验：Protocol-042 暖启动候选困难样本加权

状态：仅登记，未实现、未运行（2026-10-03）。用户将交给执行模型。

执行 [完整指示](docs/PROTOCOL042_MATURE_ERROR_WEIGHTING_20261003.md) 和 [机器登记](artifacts/ftmoe_online/protocol_042/plan.json)。
U_uniform严格复现041 W_parent；H_hard仅对16次shadow训练BCE按成熟issued live误差加权，v=1+2*abs(y-p)，再除以batch所有样本出现次数上的均值。候选验收仍用原始未加权未来BCE，live训练不变。

先补完整事务恢复、真实含更新/事件的磁盘恢复测试及fail-closed门控，再冻结两臂。U不通过禁止启动H。两条新科学序列梯度合计<=736，单次候选机会不变，F暂缓，完成后停止。

最新完成仍为 [041结果](docs/PROTOCOL041_RESULTS.md)，run37110969550；042尚无结果。
