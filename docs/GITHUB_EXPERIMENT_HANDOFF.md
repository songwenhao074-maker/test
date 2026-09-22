# 当前实验交接：Protocol-027 / data revision 002

以main为准。2026-09-22用户授权修订数据版本，旧“不得替换登记SHA、只能追溯旧完整文件”的限制已由本次明确修订替代。旧登记保存在`artifacts/ftmoe_online/protocol_027/registration_original_20260921.json`；原失败证据不变。[原交接快照](GITHUB_EXPERIMENT_HANDOFF_HISTORY_20260921_RECOVERY.md)仅供追溯。

## 唯一任务

完成一次C_fixed5/D_dynamic开发试跑，seed700/model1，交付后停止。[单任务指示](PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)中的方法、六业务时间线、九窗口主指标及预算披露保持不变，不追加A/B、其他C、消融、种子或调参。

## 数据准备已改变

新身份为`protocol027_data_revision_002`，stream SHA为`fdea84306ac752611e4d0b1b4cd2300d0e0dcbc62ace07ac94096d904b310761`，final chunk SHA为`21cd73359a4837b7c2d58456e4c85da63e0023c9088bab5dc87e87dccf529d48`。这是看到任何027模型分数前，选定已有恢复产物的新登记；不是声称旧46b1…/fc3e…版本恢复成功，也不要求再复现旧文件。

首次准备由脚本自动完成：从run35292728416保全artifact读取27个前缀chunk与transient；从run35604239834的`protocol027-restored-stream-35604239834`读取已存在的完整stream、特征、manifest和尾chunk；从仓库固定的`events.json.gz`读取已恢复并核验的事件表。逐数组核对前5520行和尾chunk，再执行完整027资格审计。**不重新模拟，不加载2.3GB的dill对象。** 事件表来源、SHA和数据选择理由见[修订记录](PROTOCOL027_DATA_REVISION_002_20260922.md)。

025源审计的旧“两类guard”仍记录为失败；027使用原已登记的正常保护集。025的历史修订次数限制不禁止本次经用户授权的独立027数据身份变更。

## 唯一执行入口

GitHub Actions选择 **Protocol-027 single D/C pilot**，分支main，手动启动：

- 首次直接完成本任务：`run_models=true`，`frozen_data_run_id`留空。工作流先准备、审计和完整归档，归档成功后才运行两组。
- 如果只准备数据：`run_models=false`。这不等于实验完成；接手者继续同一任务时使用该run ID。
- 已有完整数据归档：`run_models=true`，`frozen_data_run_id=<完整归档所在run ID>`。直接下载`protocol027-frozen-data-<run ID>`，验证所有文件后复用，不再恢复模拟状态。

仅`protocol027-frozen-data-*`是可训练的完整归档。原`protocol027-restored-stream-*`缺事件与前缀块，不能单独用于训练。完整归档含38个受校验文件及冻结清单：stream、特征、events、全部28个chunk、审计、来源核验及注册快照；不包含大dill状态。每次再上传完整副本延续90天保留期，receipt记录artifact ID、digest和清单SHA。若源artifact过期，先寻找完整冻结归档，不能以反复重模拟代替。

工作流已取消push触发。此次维护不会自动训练。结果推送先fetch/rebase当前分支，遇到冲突保留artifact并报告，不强推。

## 结论边界

两组必须读取同一冻结版本，数据不得随模型成绩改变。物理标签、因果特征、正常保护、九窗口覆盖和文件完整性仍需真实通过。D允许额外后台计算及驻留记忆，不能预先声称更省算力或必胜。

最新性能仍没有027两组结果；本地维护审计通过不能代替云端归档或模型结果。输出comparison、成本、生命周期与局限，交付后停止。
