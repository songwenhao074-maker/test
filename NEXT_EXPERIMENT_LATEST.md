# Protocol-044 revision 1：工程收尾 + 一次容量压力验证

状态：**仅计划已登记，尚未实现、生成或运行**。日期：2026-10-04。

执行入口：[完整指示](docs/PROTOCOL044_CLOSEOUT_AND_CAPACITY_20261004.md)。
登记：[plan.json](artifacts/ftmoe_online/protocol_044/plan.json) / [plan.sha256](artifacts/ftmoe_online/protocol_044/plan.sha256) / [固定场景](artifacts/ftmoe_online/protocol_044/scenario_registration.json)。

用户目标仍是结构合理且超过C，不是找到最好的D。不要再扩展窗口、初始化、门槛或结构搜索。

## 固定两阶段

1. **E：补齐043工程验收。** 19类断点逐项恢复到同一终点；修复滚动统计、在线有界存储、完整内存与入口门控测试。只读复核043全部控制证据，默认零真实梯度、不重训043。
2. **S：仅一条新科学流。** seed4401，5952预测区间，四种已有物理机制U/S1、V/S3、W/S4、X/S2，固定首次出现与回归各704区间。从空专家池启动，C_ref、共享D_lin底座、D_no_gc、D_bounded各训练一次。

E全部门控通过后才进入S。S总optimizer.step上限2360，动态D每臂最多4次后续候选。真实流没有发生永久回收时如实交付未覆盖，不调参、不增加流、不强制事件。

## 执行交接

用户要求执行后，按完整指示实现、合成预检、冻结代码、生成并封存数据、运行四条固定序列、上传原始工件和中文结果。E若发现会改变旧决策的语义问题则停止，不擅自补跑旧流。
交付docs/PROTOCOL044_RESULTS.md，分别说明工程收尾、同流D>C、真实流回收与新增闭环、成本与剩余限制。完成或阻塞即停止，不自动045。
