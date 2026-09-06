# 选模方法调研与当前决策

状态：根据用户要求完成搜索与讨论，尚未实施新的选模规则。原始训练权重、A/B 结果和既有协议保持原样。

## 当前实现与问题

每个模型种子分别按三个开发重放的平均 S 选择最佳 epoch，然后汇总模型种子。S=(F1+HR+NDCG)/3；本数据 HR=NDCG，因此诊断实际占 2/3、检测占 1/3。B_best 不保证 F1 高于末轮 A_last。

平均分本身不是无效方法，但指标重复、权重与研究目标是否匹配、类别不均衡和反复开发集选择均需解释。当前已观察到部分早期基线近乎恒定预测 CPU，诊断准确率高而检测及资源 macro F1 较差。

## 查到的直接依据

| 方法 | 原始来源支持的内容 | 本项目可借鉴之处 |
|---|---|---|
| 综合分数 | Hugging Face GLUE 示例计算多个指标的算术平均 combined_score；Trainer 可指定返回的指标选最佳 checkpoint | 平均分有实现先例，但需显式指定，且不证明本项目三项平均合理 |
| 单一主指标 | Keras ModelCheckpoint 用 monitor 指定最佳指标及最大化/最小化方向 | 若检测是主任务，可统一按验证 F1 选择，再报告该 checkpoint 的诊断指标 |
| 约束与优先级 | scikit-learn 官方自定义 refit 示例先筛 precision，再筛 recall，最后比较预测速度 | 可以先规定不可牺牲的任务表现，再选择另一项指标；属于通用模型选择，非原论文的直接 epoch 规则 |
| Pareto 多目标 | Optuna 官方教程保留多目标非支配候选，再选择所需折中 | 可展示检测/诊断取舍，但前沿通常不止一个点，仍需预先规定最终选择规则 |
| 早停耐心值 | Keras EarlyStopping 支持 min_delta、patience、restore_best_weights | 用于停止规则；它本身不证明最佳分数显著，也不解决重复计权 |

参考来源：

- [Hugging Face GLUE 官方代码](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py)
- [Trainer 最佳模型选择参数](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- [Keras ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)
- [scikit-learn 自定义选择策略](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html)
- [Optuna 多目标优化](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)
- [Keras EarlyStopping](https://keras.io/api/callbacks/early_stopping/)
- [Cawley 与 Talbot：模型选择过拟合及评估偏差，JMLR 2010](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)

JMLR 论文说明有限样本选模准则的方差会导致选择过程过拟合；换成 F1 最大并不会自动消除这一问题。这些来源提供通用方法依据，不代表 FT-MoE 原论文使用了当前具体公式。

## 尚未实施的建议

- 若检测优先，统一验证 F1 选模，诊断指标完整报告。
- 若检测与诊断同等重要，先去掉重复计权，明确权重或约束，并用资源 macro F1 辅助检查多数类问题。
- 可以利用已保存的逐 epoch 权重做规则敏感性分析，但所有版本和种子同等处理，保留每一种规则的结果。
- 不根据哪种选法最容易产生单调链，就将其事后定义为主规则；不从不同 epoch 拼出同一行的各项最大值。

下一项工作由用户另行指定。当前没有训练、新的选模重评或最终测试在运行。
