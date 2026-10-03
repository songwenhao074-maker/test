# Protocol-041 结果

本轮按登记顺序完成 Z_zero 与 W_parent 两条新科学序列。Z 是041预算内的完整040零初始化复现；W 唯一处理因素是在第二候选创建时复制当前活动E0权重/偏置，并使用全新 Adam。F 未加载。

## Z复现与候选层

- **Z 对缓存040逐项复现：True**。预测数组、更新/Adam step/批次、候选/复用/生命周期离散事件按冻结 schema 核对。
- **候选结果：candidate_improved_but_not_qualified**。W候选 BCE=+0.501081，Z候选 BCE=+0.520620，L_W-L_Z=-0.019540，相对Z BCE改善=+0.037532。
- W原资格通过=False；Z原资格通过=False；candidate_qualification_signal=False。

## 系统层

- W相对C：D_over_C=True；full ΔAP +0.018371；六窗 +0.012523；late4 +0.011372；prefix32 +0.012091。
- W相对D_keep：累计保持=True；full ΔAP -0.000880；六窗 -0.000960；late4 -0.000641。
- W-Z 初始化系统信号=False；full ΔAP +0.000000；六窗 +0.000000；late4 +0.000000；正向窗 0/6。
- 双专家池 exercised=False；accepted IDs=[0]；active counts={0: 4704}。

## 六个独立判定

- validity=True；candidate_qualification_signal=False；initialization_system_signal=False；D_over_C=True；cumulative_preservation=True；pool_exercised=False。
- **041_joint_development_success=False**。系统标签 **quality_retained_pool_not_exercised**。

## 有效性与边界

- 独立AP/BCE/混淆矩阵重算一致：True。
- 结果仍是已观察 seed3601 的开发证据；不构成跨流/多seed确认，不证明复用因果收益、永久删除有效或整体端到端加速。
- Protocol-041 到此停止；不追加重试、第三初始化臂、训练长度、阈值、seed/新流或F研究。
