# Protocol-043 revision 1 结果

本轮仅新增两条已登记科学序列：D_no_gc → D_bounded。主臂预先固定为 D_bounded，唯一必须性能比较为冻结 C_ref；其它D仅作描述。

## 主结论

- validity=True；D_over_C=True；core_lifecycle_exercised=True；gc_engineering_verified=True；resource_bounds_pass=True。
- core_goal_supported=True；gc_exercised_on_stream=False；gc_admission_closed_loop=False；full_reclamation_demonstrated=False。
- 系统标签：**core_goal_supported_gc_unexercised_on_stream**。
- D_bounded相对C：full ΔAP +0.022906；six +0.016401；late4 +0.014640；prefix32 +0.016537；正向窗口 6/6；guardrails=True。

## 生命周期与回收

- E0后接受候选：1；窗口sleep：1；same-ID reuse满足64/64：True。
- 真实流永久回收次数：0；回收创建候选闭环>=64：False。
- 若真实流未触发GC，合成闭环仅证明工程机制可运行，不代表真实回收提高性能。

## 描述性比较（非成功门槛）

- D_bounded−D_no_gc full ΔAP +0.000000；D_bounded−D_keep +0.003654；D_bounded−042 A_hist -0.000430；D_bounded−042 A_win128 -0.000000。

## 审计

- 两臂独立128-window评分全量复算：True；首次真实控制分歧审计：True。
- 042旧报告保持原invalid_execution；043/audit042只读诊断不会覆盖旧结果。
- 数据仍是已观察seed3601开发流；不声称独立泛化、最优结构/窗口、等算力胜C、永久删除永远安全或端到端加速。
- Protocol-043 revision1 到此停止，不自动启动044。
