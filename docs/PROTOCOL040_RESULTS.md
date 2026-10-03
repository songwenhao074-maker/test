# Protocol-040 结果

本轮只新增一条 **D_pool2** 科学序列；C、B=C+D_lin、D_keep、D_039 均来自冻结缓存，不重训。F 保持 DEFERRED，本轮没有加载或比较 F。

结果标签：**quality_retained_pool_not_exercised**。这是已观察 seed3601 上的开发结果，不是跨 seed/新流确认，也不能把联合结构包的效果归因到单一模块。

## 预登记判定

- **D_pool2 相对 C：True**。full ΔAP +0.018371；六窗 ΔAP +0.012523；late4 +0.011372；prefix32 +0.012091；正向窗 6/6；护栏 True。
- **累计保持相对 D_keep：True**。full ΔAP -0.000880；六窗 -0.000960；late4 -0.000641；prefix32 -0.001037；护栏 True。
- **相对上一版 D_039**：full ΔAP -0.000410；六窗 -0.000185；late4 -0.000204；prefix32 -0.000246。
- **双专家池实际运行到：False**。接受 ID=[0]；active 预测计数={0: 4704}；满足 >=64 dormant 后同 ID 复用并再 active>=64 的事件数=2。
- **两个专家局部 first64 均优于 B：None**。该项仅作描述，不证明专业化或复用因果收益。

## 生命周期与成本

- first birth=351；最终 accepted experts=1；second attempt consumed=True；reuse slots started=11。
- live optimizer.step=295/352；shadow=16/16；动态部署 forward=4704/5616；reuse preview=352/2048；shadow qualification=32/32。
- 休眠专家保留权重和 Adam 状态，因此不会节省该专家内存；本轮没有永久删除，也没有同机完整 B 路径重测。

## 有效性边界

- 冻结输入、因果成熟 batch、容量/梯度/preview 预算、终末零更新、无 F 加载审计：**True**。
- 真实 <=256 前缀零梯度磁盘恢复与合成隔离/资格 fixture：**True**。
- 核心 full AP 独立重算一致：**True**。
- Protocol-040 到此停止；不调阈值、不补候选、不加 seed/新流、不自动启动 Protocol-041。
