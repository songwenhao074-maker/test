> 历史维护记录：下文旧哈希恢复限制已由用户授权的[2026-09-22数据修订](PROTOCOL027_DATA_REVISION_002_20260922.md)替代，原失败与诊断事实保留。当前执行以GITHUB_EXPERIMENT_HANDOFF.md为准。

# 2026-09-21 仓库整理与027阻塞修复

本次范围为修复执行链路和交接，不增加实验任务，不运行C/D训练。

## 已核验失败

[run35596448071](https://github.com/songwenhao074-maker/test/actions/runs/35596448071) 的正确stream和最后chunk哈希均已恢复。正常guard、物理标签及九窗口覆盖通过；唯一027失败项是runtime_9d_features_match_stored_causal_features，最大绝对差0.11541324853897095。两组均未运行。

生成器的slope4在t<4时为0；ReplayV3将首行重复填满12步窗口，原模型在t=1..3用填充行计算了非零斜率。修复显式传递有效历史长度，将不足真实历史的差分置零。预测、回放和两个guard路径一致使用这一规则，不改物理数据和5e-5容差。

另有最终报告缺陷：runner未启动时生成了内存中的默认status，却未写入文件，且未提取eligibility具体失败项。现先持久化状态再归档，并区分维护审计通过、模型完成、模型失败。

## 整理

- 将027实验分支的实现和真实失败证据并入main，避免默认分支只有计划没有实现。
- README、LATEST、交接说明统一指向027；原交接保留为历史快照，旧计划标明历史用途。
- 新增AGENTS.md及历史索引，明确旧D_eligible/STOP、025两类guard与026五方法不支配027；物理、因果、哈希、特征门禁保留。
- 旧工作流仅标注历史参考，当前工作流默认维护审计；显式run_models=true才训练两个登记方法。
- 完整恢复数据单独归档，避免模型审计失败后又丢失数据。
- 成功复用事件只能说明发生过复用，不能单凭事件数宣称复用有因果收益。

## 验证与未解决项

- 19项针对性测试在本地和GitHub Actions通过；随后新增恢复失败报告、文件替换拒绝及缺失文件诊断，本地恢复/交接5项通过。
- [维护run35604239834](https://github.com/songwenhao074-maker/test/actions/runs/35604239834) 只恢复和检查数据，模型训练开关为false。恢复的stream SHA为`fdea84306ac752611e4d0b1b4cd2300d0e0dcbc62ace07ac94096d904b310761`，final chunk为`21cd73359a4837b7c2d58456e4c85da63e0023c9088bab5dc87e87dccf529d48`，不匹配原登记值，工作流正确阻止了后续执行。完整资格审计没有通过。
- 下载保全归档和本次恢复stream，逐数组比较前5520行与27个原chunk及transient，完全一致；27个chunk原SHA也一致。Windows再恢复一次的完整数组与云端一致，但文件SHA因ZIP平台元数据不同而不同。**尚不能证明任一新副本与原登记完整stream等价，也未确定原登记哈希不符的根因。**
- 对本次恢复stream的5520步模型输入单独验证，旧补齐逻辑复现最大误差`0.11541324853897095`；修复后`3.9411090790864023e-07`，小于原容差`5e-5`。这证明特征修复有效，不代表数据完整资格通过。
- 数值诊断：[feature_and_recovery_diagnostic.json](../artifacts/ftmoe_online/protocol_027/maintenance_20260921/feature_and_recovery_diagnostic.json)。各次原失败记录保留。

恢复失败现在会先写`recovery_verification.json`，报告实际/期望哈希和具体失败项，最终status明确记`data_recovery_failed`。恢复归档补齐events和resume_manifest，便于后续检查。

当前唯一未完成前置项是恢复原登记数据并通过哈希校验。没有C/D性能结果；不能说实验已经恢复可运行，也不能说D输给C。没有更换登记哈希、放宽容差、增加参数搜索或模型训练。
