# Protocol-037 结果

本轮只在已观察的 Protocol-036 seed3601 上进行机制开发：共同底座 B=C+D_lin 不重训；新增两条科学序列 F_extra 与 D_birth。没有生成新流、没有扫描触发阈值、没有休眠/唤醒/删除试验。

结果标签：**dynamic_birth_extra_capability_without_incremental_advantage**。本轮不是独立统计确认。

## 核心结果

| 方法 | 是否出生/存在 | 全程 ΔAP vs B | 六窗 ΔAP vs B | late4 ΔAP | prefix32 ΔAP | 护栏 | 额外能力信号 |
|---|---|---:|---:|---:|---:|:---:|:---:|
| F_extra | 从 cursor0 固定存在 | +0.010222 | +0.006315 | +0.005254 | +0.006579 | True | True |
| D_birth | t=351 | +0.008160 | +0.005529 | +0.004768 | +0.005931 | True | True |

## 动态出生相对固定额外专家

- D_birth − F_extra：全程 ΔAP **-0.002062**；六窗 ΔAP **-0.000786**。
- 相对 F 的同一护栏通过：**True**。
- 登记的 dynamic_birth_incremental_signal：**False**。

## 触发与资源

- 真实出生是否发生：**True**；birth_t=351；第一条可受影响预测=352。
- F_extra 优化步：**373**；D_birth 优化步：**352**。
- F_extra 推理调用：**5968**；D_birth 推理调用：**5616**。
- F_extra 驻留参数×interval：**441632**；D_birth：**415584**。
- 两臂只具有相同新增参数上限和相同时刻更新上限；实际训练计算不同，因此不声称严格同计算。

## 有效性与边界

- 工程恢复/未来扰动/终末结算 fixture：**True**。
- 科学因果、B隔离、批次成熟、一次出生上限审计：**True**。
- 两条训练序列预算与零新流账本：**True**。
- science checkout/workflow/源码 provenance：**True**。
- 本轮只使用已观察 seed3601，因此不能声称跨流或跨模型初始化泛化。
- 本轮没有休眠、唤醒或永久删除，不能据此宣称完整动态增删专家系统有效。
- 完成 Protocol-037 后按登记停止，不自动进入下一协议。
