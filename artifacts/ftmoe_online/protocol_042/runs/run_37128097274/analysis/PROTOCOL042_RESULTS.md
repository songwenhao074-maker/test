# Protocol-042 revision 2 结果

四条预登记序列按固定顺序完成：U_parent → R_win128 → A_hist → A_win128。主臂固定为 A_win128；64/256窗口仅作为只读诊断。

## 执行与主结果

- U_parent 对041 W_parent严格复现：**True**。
- validity=False；A_win128 D>C=False；cumulative_preservation=False；pool_exercised=True；main_development_success=False。
- A_win128相对C：full ΔAP +0.022906；six +0.016401；late4 +0.014640；prefix32 +0.016537。
- A_win128相对D_keep：full ΔAP +0.003654；six +0.002918；late4 +0.002627；prefix32 +0.003410。
- 系统标签：**invalid_execution**。

## 机制结果

- 接入策略 A_win128-R_win128：exercised=True；signal=False；full ΔAP +0.004213；six +0.003718；late4 +0.003046。
- 窗口策略 A_win128-A_hist：exercised=False；signal=False；标签=window_mechanism_not_exercised；full ΔAP -0.000430；six -0.000620；late4 -0.000364。
- multi_active_exercised=True（最长连续3616区间）；victim_selection_exercised=False；sleep_selection=True；replacement_selection=False。

## 审计边界

- issued margin/delta/label独立评分复算及AP/BCE/混淆矩阵复算：**False**。
- 评分本身额外专家forward为0，但多活动部署、训练和休眠专家reuse preview均单独计费；R/A不是等算力比较。
- 数据仍为已观察seed3601开发证据，不构成独立确认；不证明永久删除、复用因果优势或整体部署加速。
- Protocol-042 revision 2 到此停止，不自动增加窗口臂、seed、F、困难加权、永久删除或Protocol-043。
