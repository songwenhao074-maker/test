> 历史计划，仅供查证。027只执行 [当前单任务](PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)，不要将本文的STOP、额外方法或后续任务叠加到027。

# Protocol-026：复用Protocol-025数据，完成首个五方法比较

日期：2026-09-18。先读 [最新审阅](PROTOCOL025_REVIEW_20260918.md)。

## 当前唯一执行任务

本次作出明确协议决策：**Protocol-025按原规则保留为数据/方法保护集门禁失败；启动独立Protocol-026开发实验。** 使用同一物理数据，改正F0保护集的评估角色，不生成第三版服务数据。

这项修订基于seed700的数据审计，属于开发集复用；截至本指示，未见Protocol-025五方法性能结果。不得创建假的Protocol-025 data_lock、回写旧audit_pass=true或篡改旧注册文件。

只用replay seed700/model1。确认701–703、测试201–205继续封存。新输出根`artifacts/ftmoe_online/protocol_026/`，每个run_id独立，禁止覆盖。本次注册见同目录registration.json；状态为registered_not_executed。

## 1. 先恢复完整数据副本，禁止重新生成整流

读取归档run35292728416、artifact10526683176：
`protocol025-recovery-evidence-preserved-20260918`，ZIP SHA256
`f2337ce4e00f4419b64b618b4eef72ab2d95f7def882ded4f5ddeb68bf2cd092`。

先核验ZIP与内部preservation_manifest。目录包含chunks、state_5520、state_5400；恢复时使用chunks+state_5520，后备5400只用于核对，不重新跑前120步。保留旧源run35254809015及原artifact元数据。

若已找到完整stream.npz，直接验证并使用；否则：
- 在独立恢复目录恢复next_t=5520状态，核对27个不可变chunk截至5400、transient覆盖5400:5520。
- 用原注册的revision1生成器和已固定恢复逻辑，仅执行剩余t=5520这一guard interval，再完成文件组装。它不计入5520 scored，也不是新科学数据修订。
- 实际重算stream SHA，必须等于`46b1dbdd885683bd45c146ffc3cbe68dd151bf60a10c1663eda45b68f12d7c42`。
- 最终chunk 5400:5521 SHA须核对`fc3e9887961e6e99f29d0f386e4da9dd318367010f9028a6233599829eef4c10`。不匹配则检查恢复与序列化原因，不能悄悄将新哈希登记为同一数据。
- 恢复生成器仍使用其Protocol-025原注册快照，不把新的Protocol-026注册文件传入旧生成器造成registration hash漂移。

产出完整stream、common_observable_features、events、manifest、旧audit和来源清单；无论旧audit是否通过，都上传原始证据。后续“证据保全”与“允许跑模型”必须是两个独立步骤。

## 2. 独立登记新协议的数据资格

原强化audit原封保留，结果仍为false。新增Protocol-026 eligibility.json，引用旧失败原因，并独立核验：
- stream真实字节哈希、5521×16尺寸、物理容量标签重算；
- 特征因果顺序、9D有限值、存储特征与模型实时重建的一致性（含开头padding边界）；
- revision1来源与S4/S5语义、各服务有效事件、每服务正负覆盖、全部九个复现first100可评估；
- F0正常保护集normal_rows>0，记录positive_rows，**不要求positive_rows>0**。

现有13项审计中，仅替换F0类别覆盖这一项的方法意义，其余实质完整性要求不降低。新资格文件的`source_protocol025_audit_pass=false`与`protocol026_data_eligible`分别存储。只有新资格真正核验通过才建立新协议data_lock，严禁直接复制布尔值。

## 3. F0保护集只保护正常状态

索引固定0,5,…,295，目标同主机raw[t+1]，仍在F0内。只在这些观察的标签均已成熟后用于决策；首个候选提议仍从600个成熟interval开始。不得使用未来service标签补充F0。

明确角色为`historical_normal_regression_guard`：
- 在该固定子集中，只对已知正常标签行计算normal NLL与FPR；正类计数保留作描述。
- 正常NLL采用二分类logits的交叉熵，候选≤live×1.02+1e-6。
- 阈值0.5的正常FPR候选−live≤0.01。
- normal_rows=0或出现非有限预测时，guard unavailable并拒绝部署；positive_rows=0本身不是失败。
- 故障Recall、positive-only resource F1及正类AP在该guard上记null与明确原因，不能填0后参与判定，也不能因为这些指标不可定义就使整个实验失败。
- 这不是独立验证集：通用模型可以已经按原在线规则学习F0；shadow不得使用guard行训练。报告为历史正常行为回归保护，不能称为全部旧故障知识的独立保留证明。

birth和reuse两条guard路径都接入同一策略；不能只删build_f0_guard中的异常，遗漏继承代码里的两类available判断。其他验收规则保持：prospective完整部署模型loss相对改善≥1%、正常平均异常概率及FPR保护、严格标签成熟顺序、替换一致性。

新故障是否学习得更好仍由后续因果候选验证及整流评估回答。不得用正常guard通过替代这个问题，也不新增“必须先证明specialist必要”的门禁。

## 4. 方法、场景和主要比较保持明确

保持Protocol-025的6种物理服务、5520时间线、event probability0.30、revision1曲线与9D公共特征，所有已学习方法相同73D输入。保持AdamW lr1e-4、u4、batch32、replay64、distill0.10、候选buffer128_4x4、1%改善门槛、reuse与purge策略。只改变本指示明确列出的guard语义、运行隔离和报告口径。

一次完整运行：
1. C_fixed4；
2. C_fixed5；
3. C_fixed8_dense；
4. C_fixed8_top5；
5. D_dynamic；
并报告同信息pressure参考。

首要比较为九个注册复现first100的等权 **D_dynamic−C_fixed5 AP**；D−C_fixed4并列保留，其他对照也全量报告。不挑窗口，不用池化AP差替代等权均值。全程AP、正常FPR/Recall、onset、resource F1同时报告。

D允许额外后台训练和有限记忆，这符合服务进入/退出、短期复现的部署情形，但须公开成本：
- 通常D为4+1活动专家，渐变最多4+2；明确峰值6及总执行次数，不称严格每步同5专家预算。
- resident上限8包含shadow、dormant；记参数、optimizer archive及私有状态字节，FLOPs/MLP执行与驻留存储分别报告。
- pressure原始容量比采用threshold=1.0报告阈值指标；AP不依赖阈值。模型概率继续threshold=0.5。

## 5. 最小工程验证后直接跑完整比较

不要再重复几轮只有容器或常量测试的CI。必须通过的最小检查：
- 用实际960个正常guard行，验证guard可计算、两条动态路径均不因缺正类拒绝整个run；人工空guard仍不可用。
- 无生命周期的C_fixed4/D初始等价；公共9D/73D特征一致，含有故障的短回放覆盖正常/故障目标。
- birth/reuse训练、验收、guard与最终replacement同拓扑；原v2c已验证链路做针对性回归。
- opportunity与candidate守恒，结尾pending标记正确；代表性保存/恢复沿用已修复实现。
- 单arm独立子进程、顺序运行；结束后释放模型。父进程只汇总预测与summary，不能保留所有session后用全进程ru_maxrss比较每个arm峰值。
- 实际重算数据哈希与配置指纹；检查新入口没有暗中调用旧“F0必须有正类”的workflow gate。

检查通过后立即进行五方法全流开发比较。任何arm异常都标failed并保留已完成arm，不把异常记作D优胜，也不隐藏其失败。修确定性工程错误可以新run_id继续，不能借此调整科学参数。

## 6. 结果如何决定后续

不提前保证D胜出。完整输出9个窗口、每个服务的表现、所有候选事件、等待/训练/验证/渐变延迟及成本。分别回答出生、退役、复用、硬删除是否发生，以及各自是否有收益证据。

预先使用开发参考：9/9窗口有效，平均D−C_fixed5 AP≥0.03、至少6/9为正、合并这些窗口正常FPR差≤0.01。它是后续投入的开发判断，不是统计显著性或普遍优越性证明。

- 若满足参考：运行D-no-reuse与需要的计算预算对照；若存在真实容量阻塞与purge，再运行D-no-purge。固定8-top5等强对照不省略。
- 若不满足：保留负结果，提交一次原因分解（候选学习、可用记忆、匹配、等待、实际部署作用），不要按这个新结果再改guard、阈值、服务曲线或筛选seed。下一科学版本须重新登记，不能静默续调。
- 有purge但没有no-purge比较，不称删除有效；没有复用不称记忆复现有效。
- 本轮不启动确认701–703和测试201–205。

## 7. 必交付与续跑入口

提交：
- registration、source_data_manifest、eligibility、data_lock；
- guard_manifest与normal_guard_audit；
- 五arm predictions/summary/checkpoints、candidate_pairs/lifecycle；
- comparison、cost_profile、status；
- `docs/PROTOCOL026_RESULTS.md`及两个LATEST入口。

status至少区分data_restored、eligibility_verified、normal_guard_valid、completed_comparators、failed_comparators、development_signal、confirmation_run=false。不得把归档成功写成数据资格通过，也不得把数据资格通过写成模型结果完成。

仓库当前为songwenhao074-maker/test；使用仓库ID1358898099核对来源，修正脚本/文档的活动入口，保留原历史链接与证据。
