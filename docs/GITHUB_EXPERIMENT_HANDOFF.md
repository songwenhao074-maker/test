# 当前执行交接：Protocol-042 已登记，未运行

执行模型收到用户交接后，读取 [完整042指示](PROTOCOL042_MATURE_ERROR_WEIGHTING_20261003.md) 及 [plan.json](../artifacts/ftmoe_online/protocol_042/plan.json)。运行 maintenance/validate_protocol042_plan.py；该程序仅检查登记。

1. 核验036/041 raw源工件和固定041源码a1b43179fdbe18dc7a71e8a6d5a8537d9e4e522a。不要从旧checkpoint启动042，不加载F；历史raw失败报告原样保留。
2. 实现共用042状态机：U_uniform重现041 W_parent；H_hard仅在shadow BCE上使用v=1+2*abs(y-p_live_issued)、batch全体归一权重。标签必须成熟，概率来自本臂过去实际issued输出，不能用当前模型补算。正则/所有live训练/未来验收不加权。
3. 先修复shadow更新事务及ready持久化；合成恢复测试必须真正跨出生、梯度、冻结、资格ready、接受/拒绝、睡醒等事件并继续到终末。提交每个断点实证，不接受只列名称或16到19无更新测试。
4. 实现双层fail-closed门控并验证缺失/空/假报告和失败退出都阻止H。冻结两臂及分析器，先跑U一次计科学预算，复现不通过立即保全；通过才跑已冻结H一次。
5. 两臂各<=352 live+16 shadow更新，总<=736；相同已见seed3601。不得加权重系数扫描、重试、额外臂或更改验收规则。
6. 封存并上传raw，再独立重算指标和每个训练权重，发布docs/PROTOCOL042_RESULTS.md及紧凑证据，区分候选改善、资格通过、系统收益与池运行。同步main/执行分支指针。
7. 两条登记序列完成或阻塞后停止。分析/发布恢复只读取原产物，不重训。

本次仅发布指示，未创建/启动工作流。执行分支codex/protocol-042-mature-error-weighting-20261003；执行模型创建workflow_dispatch-only工作流并锁expected_execution_sha及跨run预算。最新已完成仍是 [041](PROTOCOL041_RESULTS.md)。
