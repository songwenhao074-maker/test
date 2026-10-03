# Protocol-042：暖启动候选的成熟困难样本加权

日期：2026-10-03。状态：instructions_registered_not_implemented_not_started。
本次仅发布实验指示、机器登记及交接；不实现科学运行器、不创建/启动workflow。用户将交给执行模型执行。收到交接后，可按本协议完成实现、预检及两条登记序列；交付后停止。历史033–041预算关闭；042的复现对照是独立记账的新序列。F保持DEFERRED。

## 1. 研究问题与本轮边界

目标仍是使保留动态新增、知识保存/选择、休眠复用和容量管理思想的完整D优于持续在线学习C。
041表明暖启动候选能继承能力：同一首验收窗口中，零初始化BCE=0.5206202995032794，父模型暖启动=0.501080578368035，live=0.5008784185917762；暖启动消除了约98.98%的候选-live BCE差距，却仍略弱于live。两臂候选均拒绝，最终预测与040逐文件相同。父/子同结构、同批次、同损失，仅Adam历史不同，尚未形成足够增量能力。

本轮唯一科学改变：**在保留父模型暖启动的情况下，仅对候选16次shadow训练的逐样本BCE，按当前D已经发出的成熟预测误差加权。** 检验候选是否能从继承父能力进一步学习父系统处理较差的样本。
不同时改初始化、优化器、学习率、训练次数、batch成员、候选冻结、未来验收、单次机会、休眠/复用及容量规则；不加新结构、不采用B+E0+E1叠加、不延长或重试以凑通过。

权重使用连续误差、范围有限且平均归一为1；这是一个固定开发假设，不宣称最优权重。困难样本可能含噪声，权重还可能改变有效正负比例并损害校准，所以必须用原始未加权未来BCE及误报/召回护栏验收。加权训练损失下降本身不是成功。

## 2. 固定源码、输入与历史问题

注册基于main 28eec9f190d5337b1d5d6d42d54406b42547fc28。041真实执行源码commit a1b43179fdbe18dc7a71e8a6d5a8537d9e4e522a。main不保证包含这些运行器；按plan.kernel固定提交/文件hash读取，新增042入口，不修改历史内核或历史结果。

仅用两个源工件，锁见plan：
- 036 run36831958978 / artifact11147931152：seed3601 issued feature_tape、实际C更新批次、B=C+D_lin预测、C预测和stream manifest。
- 041 run37110969550 / artifact11269043834 / protocol041-raw-37110969550：W_parent预测/资格预测/更新/初始化/生命周期记录及cached C/B/D_keep/D_039/D_040。ZIP 64506487 bytes，登记SHA256 baa36f1fbb3489f76d8f0ffb5d8e4ebc88317e231bd934146e185ac2e4ffde6e；执行者须下载实算，不把API digest当已实测。

核验原始不可变manifest所有条目；只materialize允许文件。041 raw中Z复现报告原始all_pass=false属于历史事实，不能静默改raw；本轮只以已核验W_parent为科学复现参照，不依赖旧all_pass。Git中的后续纠正报告只作解释，不覆盖raw源锁。严禁从041 checkpoint启动042或加载F、其他seed、038 donor。

041已知工程缺口必须先修复：
1. 复现比较器把新增审计字段当差异；后续分析排除字段后匹配，但workflow空输出条件未阻止W继续运行。042必须独立fail-closed门控，不使用“两个可缺失字符串中任一等于0”的条件。
2. 旧恢复fixture从16到19，不跨due16更新；列出断点名称不代表真的测试。
3. shadow内部_do_update保存时，外层shadow.updates尚未递增，next_substep已指向finish；恢复可能漏记更新。shadow.ready也必须显式保存。新实现必须把“参数更新+计数/状态变更+下一未完成操作”一致持久化。

输入缺失/过期/hash不符即source_unavailable/source_mismatch，停止而不生成新流、不重训历史基线。结果仍是已见seed3601开发证据，不是多seed确认。

## 3. 实验矩阵与顺序

| 新序列 | 第二候选初始化 | shadow训练BCE | 其余行为 |
| --- | --- | --- | --- |
| U_uniform | 复制当前active E0权重/偏置，新Adam | 原始均匀BCE，严格复现041 W_parent | 相同 |
| H_hard | 与U相同 | 第4节固定成熟误差加权BCE | 相同 |

缓存对照C_ref/B_ref/D_keep/D_039/D_040/W_041，其中W_041来自041 W_parent。U完整训练计入042科学预算，不是免费工程回放。

先冻结两臂、分析器、比较schema和所有fixture，再执行U一次。U与W_041严格复现通过、输入锁及预检证据齐全后，才运行已经冻结的H一次。不得看到U结果再修改H。U失败立即保全，不启动H、不放宽容差、不从零重跑。两臂独立执行，无共享可变参数/RNG/优化器；不从历史checkpoint或U中段复制状态来省略序列。

第一专家仍为零初始化、按原因果规则出生。第二候选在_start_shadow内部，完成本t结算后、任何本t梯度更新前复制active父参数。预期出生351、候选创建1055（父已更新44次）、第16次训练1295、首验收[1296,1328)、验收时刻1343，仅用于复现核对，不能写成强制事件或按已知phase安排出生。

候选是替代当前修正：B+candidate，不是B+parent+candidate。父专家复制操作前后权重/Adam/RNG不变，候选参数无别名、Adam state为空，首次更新step=1；接受后保留候选自己的Adam，休眠复用恢复同ID状态。若首候选无active父，标donor_unavailable并停止该机制检验，不换donor。

## 4. 唯一处理因素：精确权重公式

每次候选训练时刻t，仍读取C原登记32个interval索引，每个含16 host，保留顺序与重复项；令展平后的样本出现次数N=512（若实际源batch不同则源锁/复现失败，不自行补齐）。

对其中第j个出现的样本(i,h)：
- y_j = 1[raw_target(i,h)>0]，只允许i+2<=t的成熟真值。
- p_j = 本臂实际完整live D在时刻i、标签出现之前已经发行并保存的检测概率 out.probability[i,h]。
- e_j = abs(y_j - p_j)。
- v_j = 1 + 2*e_j。
- w_j = v_j / mean(v_1,...,v_N)。

p_j必须读取不可改写issued缓存，不是B概率、不是候选概率、不是用当前/流尾父权重补算旧样本。出生前的live为B也按真实记录使用，不特殊筛掉。禁止使用未成熟标签、phase/业务ID、未来表现、batch外样本或新的模型forward来决定权重。样本重复出现则按出现次数重复计入权重归一化和梯度。

p/y/e/v/w按float32、行优先展平，np.mean(dtype=float32)；转torch后requires_grad=False。p有限且在[0,1]、y在{0,1}，否则执行失败，不静默clip/替换。
原始v范围[1,3]；归一后均值1，最大/最小比<=3（浮点诊断容差1e-6）。不把归一后的w错误宣称仍在[1,3]。无Top-k、无按类/host再次归一、无截断重采样、无alpha扫描；系数2本轮固定。

设delta_j=2*tanh((w_model·issued_z_j+b)/2)，margin_B_j=B1-B0：
- U shadow损失 = mean(BCEWithLogits(margin_B+delta,y)) + 0.001*mean(delta²)。
- H shadow损失 = mean(w_j * BCEWithLogits(margin_B+delta,y,reduction="none")) + 0.001*mean(delta²)。

只加权数据BCE，正则项保持不加权。两个臂live专家始终使用原始均匀损失；H候选接受转为live后也恢复原始均匀损失。休眠不训练。区别严格限定为唯一候选获得资格之前的16次shadow更新，不扩大为整个系统持续困难挖掘。
U直接保留原未加权训练运算路径，避免通过重写成乘全1权重而引入归约差异，破坏精确复现。

每步只做原本一次候选forward/backward/Adam step，使用同一forward结果额外记录未加权BCE、加权BCE与未加权正则；不能额外调用模型作日志。AdamW lr1e-4、weight_decay1e-4、betas(0.9,0.999)、eps1e-8、clip_grad_norm1、Linear(73,1)/74参数、delta有界形式和B路径完全不变。

每次shadow更新写weight_trace：t、batch有序索引、host索引、每个出现的y/p/e/v/w、引用issued预测版本/有效索引、成熟性、参数/Adam前后hash、未加权/加权损失和gradient norm、ESS=(sum w)^2/sum(w²)、正负各自样本数/权重总量/平均权重、权重极值及mean。U可只读算同样e/v供配对诊断，但训练实际w固定为1，明确区分diagnostic_weight与applied_weight，不改变优化路径。
独立分析直接从本臂issued输出、标签、批次重算H权重并与trace核对，不能只相信记录中的pass。

## 5. 不变的控制器、验收与预算

plan.inherited_controller逐项复制041冻结规则：
- 容量2包含shadow，最多1active/1shadow，总ID最多2，第二候选机会最多1。拒绝/取消不重试，不永久删除已接受专家。
- E0误差上升出生：320成熟支持、最近64/之前256、ratio>=1.25及差>=0.02、正负至少16、连续两次due16；不能强制351。
- 第二候选压力来自实际live issued BCE，同epoch近期64且当前active epoch>=256成熟（无active>=64），连续两次；有休眠记忆时先满足原复用质量拒绝条件。
- shadow每16时段使用相同登记批次，16次后冻结，仅用接下来32个新issued时段验收。active继续训练。原模型睡眠/取消规则可能终止候选，照实censored/cancelled，不补跑。
- due16至多一次live转换，优先级first_birth > reuse_accept > shadow_accept > sleep；接入从t+1生效。
- sleep依据同active epoch>=256成熟、最近128的mean(B_BCE-live_BCE)<=0连续3次，保存权重/Adam；reuse每64时段、最多32槽、每槽32未来时段，冻结休眠候选作preview，按原规则取消和恢复。
- 所有候选资格使用未加权BCE：candidate<=0.99*live并且<=B；相对live及B的FPR增量各<=0.01、recall增量各>=-0.02；正负host rows各>=16，阈值0.5，全部finite。样本不足不扩窗。不能用加权BCE/AP/训练损失替代原资格。

每臂5968预测时段、16host；live更新<=352、shadow<=16、总<=368；两臂合计<=736。每due16最多2次。每臂动态部署forward<=5616、reuse preview<=2048、shadow资格forward<=32、总preview<=2080；两臂各上限翻倍。训练forward按实际批次另计，困难权重额外模型forward=0。
0新流/额外seed/F加载训练比较/donor训练/历史控制重训/参数扫描/额外候选。U复现是042新增科学成本。
真实工程全协议最多一次<=256时段、零梯度、出生前；所有梯度工程测试用合成数据，不能用真实prefix试权重。科学开始后只允许完整精确恢复，不从零重启；实际重复训练也占预算，无法确定动作是否发生即ambiguous_step保全停止。

## 6. 先完成的工程验收：需要事件证据，不接受名称清单

两臂共用修复后的状态机，工程修复不得改变完整连续轨迹。不能复制041的fixture_report或只增加新的布尔字段。
必须显式持久化：cursor及下一未执行操作、参数/Adam/RNG、shadow.updates/status/ready、issued/settled资格内容、两次终末结算进度、所有控制器计数、权重日志进度、预算账本及已提交action ID。

不能在_do_update内部保存“外层计数尚未完成但next_substep已经跳过它”的检查点。一次训练动作提交必须包含参数、优化器、shadow更新计数/第16次状态转换和下一操作的一致状态；可用完整事务或更细的next_operation实现。保存中断后不能跳过外层计数、漏训练或重复训练。终末补标签不控制/训练/预测。保存失败/未知pending不得自动重放。

正式科学之前完成下列生产入口合成测试，每项须有非空证据，汇总all_pass只从真实断言归约：
1. 权重公式金标准：y=[0,1,0,1]、p=[0.1,0.9,0.9,0.1]，e=[0.1,0.1,0.9,0.9]、v=[1.2,1.2,2.8,2.8]、w=[0.6,0.6,1.4,1.4]（float32容差1e-6）。实测梯度/权重改变；同一概率下正负对称。用非均匀权重确认正则未加权；检查重复样本、全相同误差、0/1概率边界及非法p拒绝。
2. 成熟性与出处：不可见标签不能进入权重；改变未来标签/特征后从头跑生产入口，前缀预测/决策/更新/权重不变；分别测试标签i+2=t与i+2=t+1边界。保存过去issued预测后改变当前父权重，历史p来源不得变；若重新预测会变化，也必须仍用issued缓存。
3. 初始化：复制前同t尚无梯度、父状态不变、无别名、子Adam空且首step1；两专家梯度/Adam隔离，休眠同ID恢复。零专家精确B和零初始化可学习。
4. 真正的磁盘断点续跑：连续基准从合成流起点到终末；中断分支也从相同起点前进到真实事件、保存、销毁对象、独立进程/对象恢复到终末。每个断点记录已经发生的事件、实际步数、最近更新ID、checkpoint hash、恢复后的至少一次梯度更新与至少一次控制决策（终末断点除外），对齐每条预测、参数、Adam、torch/numpy/python RNG、所有后续事件/权重和实际预算。禁止手工填cursor=16再到19，禁止仅比较同一内存checkpoint两个副本。
   必须覆盖：E0出生及更新后；shadow创建后；live更新完成而shadow未更新；shadow第1次及第15次更新后；第16次冻结后；32条资格已成熟ready=true但未决策；shadow接受、拒绝、sleep及reuse后；终末两次settle之间。U/H均覆盖含训练/权重的关键断点。合成数据可预先设计触发事件，但同一真实状态机必须执行该事件；无法触发即fixture_missing，不写pass。
5. 崩溃注入：在更新后计数提交前、原子文件替换前后、ready保存和action提交之间中断；证明恢复一致或按要求ambiguous_step拒绝。对正常恢复不能丢shadow.updates/ready。对已提交动作恢复不得再次optimizer.step；实调用插桩核验。
6. 竞争与预算：同点reuse/shadow均合格经生产_control验证reuse优先；sleep取消两种槽；容量满、机会耗尽、样本不足、流尾censored；每个due16/due64及全部阻塞理由有日志。终末实际更新/转换/预测计数差为0，不硬编码True。
7. 比较器测试：冻结比较schema，U与旧W的参数/批次/候选/事件语义相同应通过；仅arm/protocol审计名不同可通过；注入一个候选通过位、参数hash、batch索引或权重来源变化必须失败。未知新增字段显式分类，不能递归删除所有叫optimizer_step的字段。历史候选审计step=16与新step必须相等。
8. 启动门控测试：复现报告缺失、空值、字符串"true"、all_pass=false、源码/输入不符、U非零退出、fixture缺项、预算槽已消费、报告hash错误，全部不得启动H；只有真实bool True且全部锁一致才允许。工作流外直接调用H入口也须强制同样检查。
9. 评分金标准：独立AP实现，ties示例AP=0.5、单类AP=null、固定混淆矩阵；验证分类实际复制B。独立权重重算、计数和实际调用匹配。

保存每项fixture的输入/程序hash、确实触发的事件与断点、实际更新数和比较结果。恢复项目必须显示“恢复后更新数>0”以及两种状态内容一致，而非只有同一个digest字符串。
缺项或失败即engineering_incomplete，不启动任何真实科学臂。登记validator通过不代表工程验收通过。

## 7. U复现与H配对约束

U必须复现041 W_parent：保存的概率/logits/delta/active_id/epoch/version/hash/labels/class_probability数组精确相同；候选资格预测、更新批次顺序及重复项、参数/Adam step/hash、初始化父子映射、控制器事件/接受拒绝相同。所有登记AP/BCE及混淆矩阵重算一致，AP/BCE绝对差<=1e-12。npz压缩元数据和计时不比较。
协议名、臂名、增加的审计字段必须在科学启动前冻结schema排除；模型/优化器实际step及资格结果不可忽略。对照hash取模型state_dict规范化算法，不能因改容器名称改变模型键。

预期U：候选BCE0.501080578368035，候选AP0.5741380830998513，live BCE0.5008784185917762；拒绝；E0生命周期351/1599/2351/4047/4207；295 live+16 shadow更新；部署4704、reuse352、shadow资格32。只核对，不硬编码决策。
比较失败即停H。若执行者认为差异是审计schema遗漏，先保全并交付，不在原科学run中事后修报告绕过门控。已发生的U不重跑来修发布；后续处理需用户交接。

H与U在候选创建及第一次加权更新之前应参数/Adam/live历史完全一致；H权重只能改变shadow。直到第一处不同live转换生效前，两臂live输出和用于首验收的live/B/labels精确相同；首资格窗口若因实现变化不再可比，记invalid_execution或mechanism_not_observed并解释，不能换窗。所有16次shadow的batch相同，H权重可从相同的过去live输出独立重算。

## 8. 预登记指标与结论

候选层以相同首未来窗口比较H候选、U候选、共同live、B，报告原始未加权BCE/AP/recall/FPR/混淆矩阵、H-U差、相对BCE改善、每项资格条件和训练/验证时序。
candidate_qualification_signal = validity AND H原资格通过 AND H候选BCE严格低于U。
H BCE低于U但仍拒绝：candidate_improved_but_not_qualified；没有改善：no_candidate_improvement；无可比资格：mechanism_not_observed。不能以加权资格或挑子窗口替代。

加权机制诊断：
- H/U相同批次上的未加权损失、实际权重分布/ESS、正负权重质量、梯度裁剪前后范数、parent-child参数距离/预测差异（只用已有forward/资格输出，不增加真实模型调用）。
- 首验收按共同live在同一issued行的e=abs(y-p_live)固定分箱[0,0.25)、[0.25,0.5)、[0.5,0.75)、[0.75,1]报告各组BCE差和支持，空组null。这是看到成熟标签后的描述性误差分组，不是新的独立测试或接受标准，不能宣称困难组改善已经证明泛化专业化。
- 报告正/负各自未加权BCE，特别检查改善是否只通过抬高异常概率而伤害正常类。全程AP升而BCE/召回/FPR恶化须明确。
- 不能把加权训练损失与U未加权训练损失直接比大小；效益评价必须使用共同未加权指标。

系统层：分别报告U/H相对C、B、D_keep、D_039、D_040、W_041及H-U，沿用041窗口和护栏：
- 主D>C：full AP>=+0.002、six>=+0.002、至少4/6正向、late4>=0、prefix32>=-0.005，全部护栏/有效性通过。
- 累计保持始终相对D_keep：full/six/late4 AP差>=-0.001，prefix32>=-0.005，护栏通过。不能重置退化额度。
- 加权系统开发信号H-U：full AP>=0、six>=+0.0005、至少4/6正向、late4>=0、相对U护栏通过。0.0005是操作性开发阈值，不是统计显著性。
- 护栏：全程和六窗pooled FPR差<=0.01、recall差>=-0.01；六W块均值AP差>=-0.02；每复现窗recall差>=-0.02、每prefix32 recall差>=-0.03。
- pool_exercised：两个已接受不同ID各实际部署>=64时段，且至少一次同ID实际休眠>=64后再次连续部署>=64。
- 每个专家首次部署连续同epoch前64、事件前64/后32/64/128、训练/休眠/复用区间和成本完整报告。跨epoch窗口标混合；phase只事后描述，不输入控制器。

分别输出validity、candidate_qualification_signal、hard_weighting_system_signal、D_over_C、cumulative_preservation、pool_exercised；六项都真才是042_joint_development_success。
系统标签顺序沿用：invalid_execution / D_over_C_not_established / D_over_C_with_excess_pool_loss / quality_retained_pool_not_exercised / quality_retained_pool_exercised。
H被拒绝且系统轨迹相同，应明确“候选训练改变但没有部署收益”；接受第二ID也不自动证明互补能力/复用因果收益。无独立seed、无no-reuse对照、无永久删除，本轮不作这些结论。

## 9. 实施门控、成本、交付与停止

执行者创建run_ftmoe_protocol042.py、analyze_ftmoe_protocol042.py、来源materializer、真实生产入口synthetic tests和fail-closed gate脚本。先运行maintenance/validate_protocol042_plan.py，再实现测试并冻结execution SHA/依赖/fixture/schema/hash，之后才能跑U。修复适用于两臂，不可一臂用旧坏恢复、一臂用新实现。

workflow protocol042-mature-error-weighting.yml仅workflow_dispatch，execution_branch=codex/protocol-042-mature-error-weighting-20261003，锁expected_execution_sha、协议并发与跨run两臂槽账本。用一个不可歧义gate输出ok=true，并在H进程入口再次读取布尔报告/源码锁/fixture/预算确认；不依赖缺失输出的隐式类型转换。U非零退出可先上传证据，但绝不能继续H。
首次梯度前持久化arm.started；每实际动作记账，崩溃不能将已消费槽写回False。重复run只允许已核验精确恢复，不重新发两条预算；发布失败不重训。

成本分别记录live/shadow训练、部署/reuse/shadow资格调用、困难权重计算、参数复制、控制器、checkpoint/I/O时间；实际参数/Adam/权重缓存/验证缓存/序列化副本内存和峰值。权重计算模型forward为0但CPU/缓存成本非0；保留父模型历史训练成本。B历史成本与新增成本分栏，不拼接旧机器耗时宣称整体加速。
若H与U调用次数相同而AP无增益，照实报告无部署质量收益。不能用单次wall小差宣称提速。

先封存不可变科学raw manifest并上传artifact（>=90天），后分析；status科学/分析/发布分离。raw包含冻结必要源码/测试/schema、输入锁、两臂预测、issued资格原始数组、完整训练/权重/初始化/机会日志、检查点、每项测试实证、预算和费用。Git紧凑目录artifacts/ftmoe_online/protocol_042/runs/run_<id>/及docs/PROTOCOL042_RESULTS.md至少交付：
source_lock、implementation_manifest、fixture_report+breakpoint evidence、U_reproduction、paired_audit、weight_trace（体积小则直接进Git）、weight_recompute、candidate_comparison、comparison、per_window/per_expert、生命周期/复用/机会日志、budget_ledger、cost_profile、independent_recompute、artifact_index（ID/URL/ZIP digest/大小/有效期）、compact_manifest及三种status。
独立从已封存预测复算full/six/late4/prefix32/64 AP、BCE、混淆矩阵，并从issued缓存重算每个训练权重。工程有效性按必需fixture内容核验，不只读取all_pass。

同步main及执行分支交接指针；保留原始失败报告。用最新main为父提交、[skip ci]、不force push；只有非快进最多3次重建，认证错误单列publication_pending。分析/序列化修复只读原产物，不重训。
两条登记序列完成或明确阻塞后停止。无论正负，不自动调权重系数、加第三臂、改验收、重试候选、增加seed/新流、永久删除、恢复F或开启043。后续学习目标/结构/有界重试需另行登记和用户交接。
