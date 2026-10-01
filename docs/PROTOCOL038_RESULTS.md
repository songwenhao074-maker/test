# Protocol-038 结果：固定出生机制下的初始化机制对照

- GitHub Actions run：**36884471028**
- 科学有效性：**True**；结果标签：**early_only**。
- 本轮严格复用 seed3601；B/F/D_zero 均为缓存只读对照，没有重训。
- 新科学更新严格为 donor 21 + D_bias 352 + D_warm 352 = **725** 次。

## 主要预登记判断
- warm_init_signal：**False**；D_warm−D_zero 六窗等权 AP = **0.000188**，全程 AP = **0.000785**。
- bias_init_signal：**False**；D_bias−D_zero 六窗等权 AP = **-0.000006**，全程 AP = **-0.000002**。
- feature_weight_increment：**False**；D_warm−D_bias 六窗等权 AP = **0.000193**，全程 AP = **0.000787**。

## 与固定专家 F 的对照
- D_bias extra_capability：**True**；dynamic_increment_vs_F：**False**。
- D_warm extra_capability：**True**；dynamic_increment_vs_F：**False**。
- D_bias 相对 F 的出生后[352,5968) AP差：**-0.002098**。
- D_warm 相对 F 的出生后[352,5968) AP差：**-0.001238**。

## 初始化成本与边界
- donor 是真实的 21 步历史前缀模型，不是免费初始化；单独部署 D_bias 或 D_warm 均应计 21+352=373 步。
- 037 固定 F 前21更新采样正例比例：**0.008819**；D_zero 出生后前21更新：**0.266462**。该差异只作历史分布背景，不是受控因果证据。
- 六个复现窗来自同一条已观察开发流，不能视为六个独立随机种子，也不支持统计显著性或等价性结论。
- Protocol-038 到此停止；不得自动追加新 seed、结构、休眠/唤醒/删除或下一协议。
