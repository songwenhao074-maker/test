# Protocol-041：第二专家继承当前专家权重的受控实验

日期：2026-10-03。状态：instructions_registered_not_implemented_not_started。
本次仅发布方案、机器登记和交接，不实现运行器、不创建或启动工作流。用户将把本指示交给执行模型；收到该交接后，执行模型可按本协议完成实现、测试和两条登记序列，交付后停止。历史033–040预算关闭；041的对照复现是单独记账的新科学序列，不能重开040。F研究继续DEFERRED。

## 1. 研究问题与证据边界

最终目标仍是使保留动态新增、知识保存/选择、休眠复用、容量管理思想的完整D优于持续在线学习C。本轮只问：
**在完全相同的出生、训练和验收规则下，第二专家继承出生时活动专家的权重并重置Adam，能否缩小冷启动差距、通过验收，并形成有用的双专家池？**

040只有E0被接受。第二候选在1055创建，1295完成16次更新，验收[1296,1328)，1343拒绝；候选/live/B的BCE分别0.5206202995032794/0.5008784185917762/0.528111444341405。它优于B却弱于活动专家，只有bce_vs_live失败。候选训练结束时E0已更新60次，验收32时段有28个跨到V阶段。拒绝后又出现20次持续压力检查，但单次机会已耗尽。040相对C全程/六窗AP为+0.018371/+0.012523，相对D_keep为-0.000880/-0.000960。

038测的是第一专家继承出生前21次更新的历史donor；warm相对zero全程AP+0.000785256、六窗+0.000187899，BCE+0.000562202，未达登记信号。不能据此否定本轮当前E0暖启动，也不能把AP微升当成BCE资格必过。038原报告early_only和有效性字段有独立复核指出的限制，见docs/PROTOCOL038_REVIEW_AND_HANDOFF_20261002.md。

本轮刻意不同时改变单次候选机会、训练长度、冻结验收、出生/休眠/复用门槛、采样和目标函数。否则不能判断收益来自初始化还是其他修改。若结果仍为负，不表示所有暖启动、重试机制或互补学习无效。已观察seed3601只用于开发，不是独立确认。

## 2. 源码和冻结输入

注册基于main 5ac353e3fd0772c74fb5778155d525b62e8ef85c；040实际科学源码在562979a1a826de828f7d2bdc0017b3993c8623a1。不要假设main已有运行器。按plan.kernel.files核验固定提交中的源码，新增041入口，保留037/039/040历史内核；禁止直接调用旧science入口重训历史实验。

输入只需两组核验后的工件：
- 036 run36831958978、artifact11147931152：seed3601 feature_tape、actual update_batches、B预测、C预测和manifest，逐文件锁见plan.source036。
- 040 run37098618766、artifact11265231995：C_ref/B_ref/D_keep/D_039/D_pool2预测与040控制器/更新日志，作为缓存参照。ZIP SHA256为319d9f75aac25346e09027dabbe212acc33c14c81e06d8f1e09f8e418574e9fd，大小32011607；该digest来自artifact_index，执行时须下载实算。科学manifest逐项核验，文件白名单和manifest锁见plan.source040。

040原工件无需加载历史checkpoint来启动041；不能把旧检查点当可靠恢复状态。041从头执行两条已登记序列，缓存040只作独立复现核对。B是C+D_lin，禁止用C logits冒充B。C/B在原轨迹中持续训练，缓存不等于模型原来被冻结。

仅加载seed3601所需文件；不加载其他种子、F模型/预测或038 donor，不产生新模拟流。文件缺失/过期/hash不匹配，分别记录source_unavailable/source_mismatch并停止，不重新生成流或重训基线替代。

## 3. 两条新序列与执行顺序

| 名称 | E0初始化 | 唯一第二候选初始化 | 优化器 | 作用 |
| --- | --- | --- | --- | --- |
| Z_zero | w=b=0 | w=b=0 | 各专家新建独立AdamW | 在修复工程实现上复现040 |
| W_parent | w=b=0 | 复制该时刻活动E0的w和b | 候选新建独立AdamW，空state，step从1开始 | 唯一算法处理因素 |

C_ref、B_ref、D_keep、D_039、D_040均为缓存；D_040对应源工件D_pool2。Z_zero不是免费工程回放，完整科学训练计入041预算。

先冻结两臂实现、分析器、fixture、源码SHA和依赖，再运行Z_zero。Z完整结果与缓存040通过第8节复现检查后，再运行已冻结W_parent。不得根据Z的结果修改W。若Z复现失败，保全并停止，不消耗W、不放宽容差、不自动从零重跑。两臂独立从头执行；不共享可变状态，不用结果挑分支，不通过复制中途科学checkpoint省略登记序列。

两臂均应因果重现E0出生351、候选创建1055、完成训练1295、验收1343；这些仅为复现预期，不能按时刻强制触发。若首候选时无活动专家，记录donor_unavailable并判本轮机制不可检验，不临时改成休眠专家/历史F/随机权重。

## 4. 精确复制时点与训练核

每t顺序：先live及已存在资格槽预测；结算i=t-2；due16依040优先级执行最多一次live转换；若无转换才作压力/资格槽启动；随后active更新一次、shadow更新一次。

W在_start_shadow事件内部、完成本t结算但在本t任何梯度更新之前，深复制当前active专家参数。预期t1055的E0已完成截至1039的44次更新；记录donor_id、model_hash、optimizer_hash/step、最大训练标签可用时刻、candidate_initial_hash与复制时点。不能误用1055更新后的E0，否则候选同批次多看一次数据；不能用1295或流尾权重。donor的权重和Adam在复制操作前后不变，复制只读且不共享storage、不改变任何RNG。

W新优化器state为空，不继承donor的step、exp_avg或exp_avg_sq；第一次实际更新后step=1。候选被接受后保留它自己的Adam；休眠复用则保留同ID原Adam。不要把“出生时重置”误用于“复用时重置”。

所有专家仍为Linear(73,1)，74参数：
- delta=2*tanh((w·issued_z+b)/2)，输出[B0-delta/2,B1+delta/2]。
- BCEWithLogits(issued_B_margin+delta,y>0)均值 + 0.001*mean(delta²)。
- AdamW lr=1e-4、weight_decay=1e-4、betas=(0.9,0.999)、eps=1e-8；clip_grad_norm=1。
- 每16时段用C登记的相同成熟批次，保留顺序/重复项/全部16 host；禁止额外热身、补训练、改正负样本比例。
- 候选取代当前修正，绝不输出B+E0+E1；分类数组精确复制B。
- E0构造维持隔离seed3701及零初始化；W候选在复制后允许非零delta，不能复用“所有新专家必须输出零”的断言。断言应改为参数精确相等、复制前候选不影响live、输出有限有界及优化器隔离。
- 每臂候选最多16次更新，然后冻结。仅用随后32个新issued时段验收，不能把训练批次作为验收窗口。active继续训练；候选不追踪复制后E0的更新。

在合成fixture中补做“复制权重+完整Adam”的机制负对照：相同批次同步更新时，两专家参数/预测应相同。此项无真实数据、无第三条科学臂；冻结候选后不再要求同轨。该测试说明单纯复制全部状态不会自动产生专长。

## 5. 完全继承的控制器规则

plan.inherited_controller是040的逐项冻结值，不依赖执行者自行解释“沿用”。

E0：至少320成熟时段；最近64相对之前256的B issued BCE，比例>=1.25且绝对增加>=0.02，至少16正16负host rows，连续2个due16检查才出生；不能强制351。

候选压力：同样64/256损失上升规则，但依据实际完整live D issued BCE；最近64须属于当前epoch；活动epoch至少256成熟、无活动至少64，连续2次。压力不永久锁存。

池：驻留容量2（包括shadow），最多1active/1shadow，总创建ID最多2、第二候选最多1次。拒绝/取消后不重试，不永久删除已接受专家，不能为了形成双专家而强行接受。容量限制与尝试次数分别记账。

休眠：当前active epoch至少256成熟，最近128均同epoch，mean(B_BCE-live_BCE)<=0连续3次due16；保留权重/Adam，取消过时资格槽。允许重复睡醒，不增加ACTIVE_FINAL。

复用：每64时段且当前epoch>=64成熟，已接受休眠专家全部进入固定32个未来时段预览；最多32个槽，资格候选冻结、active继续学习。没有phase/业务/相似度硬过滤。准备完成后下一个due16验收；优先级first_birth > reuse_accept > shadow_accept > sleep，每点至多一次live转换，生效预测为t+1。复用接受取消shadow；shadow接受取消过时reuse；sleep取消两者。

第二候选需当前压力、容量、无待处理reuse；有休眠专家时，还需同epoch、相同ID/hash集合、最近64游标内有足够样本的复用质量拒绝；样本不足不能授权出生。reuse资格可与已有shadow训练/验收并行。

新候选和复用共用资格：
- 至少16正16负host rows，全部finite。
- candidate BCE <= 0.99*live BCE，并且 <= B BCE。
- 相对live及B分别满足FPR增量<=0.01、recall增量>=-0.02（阈值0.5）。
- 只评最初32个新时段；样本不足拒绝，不延长、不事后选窗；多个合格复用候选选最低BCE，平局小ID。
- 验收不含训练正则项。AP同时报告，但不能拿AP改善替代BCE资格。
- 接受时旧active休眠，新ID带自己的Adam转active，同t正常更新一次，t+1开始部署。

终末只结算最后两条标签，不作控制/梯度/新预测；未完成的槽标censored，不补训练和验收。

## 6. 硬预算

两条完整新序列，每条5968预测时段、16 host、73维特征；同一已见seed3601。每臂live更新<=352、shadow<=16、总<=368；两臂科学optimizer.step合计<=736，每个due16最多两次。复制当前E0不新增donor训练；任何隐藏donor/追赶训练均禁止。

每臂部署动态预测forward<=5616；reuse preview<=2048；shadow qualification<=32，preview合计<=2080。两臂部署合计<=11232、reuse<=4096、shadow资格<=64、总preview<=4160。单位为一次16-host interval调用；训练forward另按实际更新批次逐项计数。用于分析的已发行预测读取不记新forward；不得用后来的模型补算历史候选预测。

0新流、0额外种子、0F加载/训练/比较、0基线重训、0真实完整Adam继承臂、0随机初始化/学习率/阈值扫描、0额外候选尝试。
真实工程检查全协议最多一次<=256时段、零梯度、出生前；其余工程训练/未来扰动/恢复用合成数据。两条科学序列开始后分别只可精确恢复，不从零重启。真实测试若执行了梯度也计科学预算，不能称工程步骤免账。

## 7. 必须先修复并验证的工程问题

040前缀恢复fixture只在出生前继续运行；learned-state测试只读回hash，未覆盖更新后继续执行。原更新后checkpoint仍cursor=t，恢复再advance会重复结算。因此041必须新建可靠的子步骤恢复语义，不能直接继承旧all_pass。

要求持久化next_substep/下一未执行操作、issued预测、settled掩码、槽的status/ready/issued/settled_rows、shadow更新数、live角色、优化器/RNG、预算计数和已提交action ID。每次更新/拓扑/槽事件后保存状态时，恢复须从下一个未执行子步骤开始；不能仅粗暴cursor+1而跳过本t尚未执行的shadow更新。完整interval结束才进入t+1。终末结算进度也可恢复。

每个真实训练/预测动作写唯一ID并预留账本，完成后把状态和已提交ID一致持久化；检测到无法确定是否执行的pending动作，标ambiguous_step、保全并停止，不重放。没有可信精确恢复状态即交付中断，不能“修好后再跑一遍”消耗未登记科学预算。

合成fixture必须调用生产041入口并保存可复查数值/计数/digest，至少覆盖：
1. 零专家精确B；E0零初始化梯度非零且能学习。
2. W复制前父模型44步类状态、复制后参数相等/非别名/父Adam不变/子Adam空，第一步子step=1；完整Adam继承负对照同步等价。
3. 两专家梯度与Adam隔离，休眠冻结，复用恢复同ID及权重/Adam hash；容量包含shadow。
4. 同点reuse与shadow均可接受、reuse优先取消shadow；sleep取消；shadow第16次更新后冻结；无支持拒绝；满容量和一次尝试耗尽分别记录；流尾censored；at-most-one live转换。
5. 连续执行与“磁盘保存、销毁对象、恢复并继续到终点”逐项一致：在E0出生/更新后、shadow创建后、live更新后但shadow更新前、第16次shadow更新后、32条资格标签已齐但未决策、接受/拒绝/休眠/复用后及终末两次结算中间分别断开。比较每条预测、参数、Adam、RNG、全部后续决策和实际预算；不能比较同一内存对象的两个副本冒充。
6. 对同一合成流实际改动非空未来特征/标签，从头运行生产入口，改变数据尚不可见时的issued预测/决策/更新保持相同；记录变更位置和非零差值，确认用到变更后的输入。分别检验预测i+2<t与决策/训练i+2<=t。
7. 每次optimizer.step及forward实调用插桩；终末调用前后更新/转换/预测计数差实际为0。能检测重复settlement/重复action，不能硬编码True。
8. 独立AP实现检查并列分数组、单类null、阈值混淆矩阵；分类数组实际逐项等于B。
9. 记录每个理论due16/due64机会及全部阻塞理由集合：忙碌槽、尝试耗尽、容量、无压力、无donor、无有效拒绝；不能将没有新增槽等同“没有需求”。
10. 真实<=256前缀零梯度恢复若执行，单独记账；不能代替以上含事件/梯度的合成测试。

缺任一必需项即engineering_incomplete，正式科学不可启动。fixture_report的all_pass必须由逐项真实断言归约，文件含测试ID、输入hash、断点、前后计数和结果；报告与source manifest一致。禁止在科学完成后为了通过而补造预检证据。

## 8. Z复现和配对因果检查

Z必须与缓存040逐项对齐：
- labels、class_probability、概率、logits、active_id/deployment_epoch、参数hash、批次顺序和重复项、每步Adam step、候选/复用接受拒绝和生命周期离散事件。
- 数值预测在冻结CPU/Python/torch/numpy环境要求数组精确相同；不比较npz压缩时间戳或运行耗时。AP全程及每个登记窗口绝对差<=1e-12。
- 忽略新加的审计/检查点文件名、wall/CPU时间和action序号等工程元数据；事先在comparison schema列明排除字段，不能忽略模型/控制语义差异。
- 预期候选1055/1295/1343及拒绝、E0睡醒1599/2351/4047/4207、live295+shadow16更新、4704/352/32 forward。预期只作核对，不能把轨迹写入控制器。
如无法逐字段比较，列出缺失和原因并停止W；不能只比较舍入主表就宣称复现。

Z/W在第一次候选初始化前参数/优化器/控制器状态完全一致；初始化后shadow可不同，但live轨迹及已发行预测应在第一个不同的live转换生效前相同。首次资格预期[1296,1328)对齐，两个验收的live/B/labels必须逐项相同。如果更早出现live分叉，先判隔离或因果错误，不解释为初始化收益。若候选被取消而无验收，标mechanism_not_observed，不强行补事件。

## 9. 预登记分析与解释

候选层主比较是在相同首验收窗口：
- 三方表：Z候选、W候选、共同live、B的BCE/AP/recall/FPR/混淆矩阵。
- 输出L_W-L_Z，(L_Z-L_W)/L_Z，以及两臂各自相对live和B的差；分母非正记null。
- 按原资格逐条件输出通过/失败，不重定义“接近通过”。candidate_qualification_signal仅当有效性通过、W原资格通过且BCE严格优于Z时成立。
- W BCE优于Z但仍拒绝：candidate_improved_but_not_qualified；W未优于Z：no_candidate_improvement；资格不可比：mechanism_not_observed。
- 记录复制后首次更新前参数和输出等价诊断、16次更新的模型hash/step/训练损失。未更新候选对已出现输入的诊断不得改写issued预测或参加验收，也不得增加真实forward；通过权重相等及已缓存父输出表达初始化等价，合成测试验证函数等价。
- 训练损失下降不能替代未来验收；AP与BCE方向不同必须突出。不能从出生所在U/V推定专业化。

系统层完整输出W/Z各自相对C、B、D_keep、D_039、D_040，以及W-Z。复用040全部窗口/指标/护栏：
- D>C主条件：full AP>=+0.002、六窗均值>=+0.002、至少4/6为正、late4>=0、prefix32>=-0.005，并满足共同护栏与有效性。
- 累计保持始终相对D_keep：full/six/late4 AP差>=-0.001，prefix32>=-0.005，护栏通过。不能每轮重赠退化额度。
- 初始化系统开发信号W-Z：full AP>=0、six>=+0.0005、至少4/6为正、late4>=0，并且相对Z的同形式护栏通过。0.0005只是操作性开发阈值，不是显著性界限。
- 护栏：全程及复现pooled FPR差<=+0.01、recall差>=-0.01；六W块平均AP差>=-0.02；每复现窗recall>=-0.02、每prefix32 recall>=-0.03。
- pool_exercised严格为两个不同已接受ID各实际部署>=64时段，并且至少一个ID实际休眠>=64后同ID再部署连续>=64。
- 各ID首次部署前64连续同epoch时段vsB的局部效用、各事件前64/后32/64/128比较、实际训练/部署/休眠区间、候选预览成本、各阶段使用分布全部报告。跨epoch窗口标混合并另报连续段；阶段只做事后描述。
- W接受E1后，系统差异包括初始化导致的后续训练/控制轨迹变化，不能全部归因于某一时刻参数差；没有no-reuse对照，不宣称记忆复用因果优势。

分别输出validity、candidate_qualification_signal、initialization_system_signal、D_over_C、cumulative_preservation、pool_exercised六个独立字段，不让一项通过替代其他。
系统标签依次为invalid_execution、D_over_C_not_established、D_over_C_with_excess_pool_loss、quality_retained_pool_not_exercised、quality_retained_pool_exercised。只有有效性、候选资格、初始化系统信号、D>C、累计保持、pool六项都通过，才可标041_joint_development_success；仍不是独立确认或完整D普遍优于C的证明。

结果决定后续讨论，不授权自动执行：
- 候选改善并通过，但系统退化：查接受后收益持续性及睡醒规则。
- 改善但不过：再讨论训练跨度/阶段变化与验收，不先松门槛凑事件。
- 未改善：再考虑改变训练分布/目标或互补修正。
- 单次机会未形成池：只能报告本次尝试结果；有界重试是独立控制器因素，另行登记。
本轮所有已登记有效序列完成即停，不根据负结果追加臂/种子/训练。

## 10. 成本与交付

分别记录每臂live/shadow梯度、部署/reuse/shadow资格forward、控制器、参数复制、checkpoint/I/O的调用/时间；参数、Adam、资格缓存、序列化副本的内存范围及进程峰值。休眠保留状态不节省专家参数内存。W复制既有E0不新增donor梯度，但E0先前的训练成本已计在本臂，不能宣称免费获得能力。B历史成本独列；不拼接跨机器旧耗时宣称端到端加速。合成fixture成本与真实科学成本分开。

新增run_ftmoe_protocol041.py、analyze_ftmoe_protocol041.py、synthetic fixtures及materializer；运行器不得加载F。先运行maintenance/validate_protocol041_plan.py；该validator仅验证登记文件，不证明科学正确。先冻结execution SHA/依赖和源码hash，再运行预检及Z/W。
工作流protocol041-parent-initialization.yml仅workflow_dispatch，分支codex/protocol-041-parent-initialization-20261003、expected_execution_sha双锁、协议级并发锁、一次登记两臂账本；重复dispatch先查已有科学槽，不按run ID重新发预算。没有用户交给执行模型的启动指示时，本次发布者不启动。

科学/分析/发布三种状态分离。先封存不可变科学manifest（逐文件sha256）并上传raw artifact、保留至少90天，再分析/发布；分析复算失败从原预测修复，不重训。raw含输入锁、必要源码bundle、两臂全部预测/issued资格输出/更新及机会日志、初始化审计、恢复checkpoint、真实预算、fixture证据。含F的历史整包可核验archive，但不得把F文件materialize给本轮运行器。

Git紧凑目录artifacts/ftmoe_online/protocol_041/runs/run_<id>/至少含：
status、source_lock、implementation_manifest、fixture_report、Z_reproduction、paired_initialization_audit、candidate_comparison、comparison、per_window、per_expert、lifecycle/reuse/candidate/opportunity日志、budget_ledger、cost_profile、core_metric_recompute、artifact_index、compact_manifest及中文docs/PROTOCOL041_RESULTS.md。
独立从封存预测重算全程、六窗、prefix32/64、late4、BCE和混淆矩阵，不能只重算full AP。按plan锁和fixture逐项独立审计有效性，缺项不得用工作流绿灯替代。

同步main及执行分支交接指针，保留历史方案/失败证据。以最新main为父提交，只对非快进最多3次重建，不force push；认证失败记publication_pending并保全，不把认证错误说成非快进、不重训。提交带[skip ci]，禁止自动开启下一协议。
