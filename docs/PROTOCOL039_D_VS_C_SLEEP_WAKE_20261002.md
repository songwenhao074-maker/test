# Protocol-039：以完整D优于C为目标，先接入一次休眠与唤醒

状态：只登记实验指示，尚未实现或启动。用户2026-10-02明确要求以完整D>C为当前目标，F相关实验暂缓，待完成D>C目标后再探索。本次提交不创建/启动workflow。后续收到用户执行交接的模型可实施本轮固定预算；交付后停止，路线图不自动授权下一轮。

## 目标调整与历史状态

Protocol-038已在run36884471028完成科学训练，共725次更新，结果在执行分支commit c8b4ec85ccbcc90c006ba3403e99ed011a3fd8c9。主分支旧指针“尚未开始”是发布失败造成的陈旧状态，不能据此重跑038。独立复核见 docs/PROTOCOL038_REVIEW_AND_HANDOFF_20261002.md。

当前首要问题是加入生命周期管理后D能否保持相对C的优势，不再要求战胜F。本轮不训练、不重算、不调优F，不实施继承F权重/优化器或donor实验，也不将任何F比较作为通过条件。历史F文件和033–038结果保持原样。本轮不继续追求初始化改进。

本路线的D明确定义为B=C+D_lin，再添加可动态管理的一个额外线性专家；不是早期另一套专家组织方式的自动等价物。039仅是完整D的一步：出生→一次休眠→一次唤醒，没有永久删除、多专家路由或完整系统跨流确认。单流优于C仅能叫开发信号，不能提前宣布完整动态增删D成功。

## 冻结来源与四个输出（仅一条新科学序列）

计划基于main e2854dad6b1e4df15f01e9ffd1de150e70c1b6e8；执行分支 codex/protocol-039-d-vs-c-sleep-wake-20261002 应从含本039登记的main创建。复用037科学内核时，从固定commit 7440a3973f97599ea2ad4de45d5bacaaf8b517c8读取 run_ftmoe_protocol037.py、analyze_ftmoe_protocol037.py及其依赖，保存来源hash。新增039入口，不调用旧完整science入口，不重训旧臂。修正测试缺口，不照搬038的伪通过fixture。

只用已见开发流seed3601，5968预测×16hosts，issued特征73维。036 run36831958978/artifact11147931152 ZIP SHA256为040bacac8f23659f56599b71d6d6706b62a544e44161c7c42fb5af73650f56c0；037 run36871854703/artifact11167124178 ZIP SHA256为498ead6571ef34e56702fd23f8cc57ac30afe92e99a8ffbfbd85a0b92e413c77。逐文件锁写入plan。artifact内虽含历史F文件，本轮运行器/分析器不加载这些文件；整包下载不算授权分析F。

|输出|定义|本轮新增训练|
|---|---|---|
|C_ref|036 seed3601 C原始预测|0|
|B_ref|036 seed3601 C+D_lin原始预测|0|
|D_keep|037 D_birth原始预测，出生后始终运行|0|
|D_sleepwake|与D_keep相同出生规则和零初始化，最多一次休眠、一次唤醒|仅此1条|

B仅用于共同底座、因果状态信号与贡献归因；主要成败看D_sleepwake相对C，保留D_keep检查管理损失。不能只报D>B，也不能把B已有增益全归因于动态机制。

四臂标签/raw标签/分类输出/时序必须一致；C、B、D_keep逐字节复制来源。C是036持续在线学习的五专家参照，不是冻结C；B亦是历史在线输出。只读重算C/B/D_keep主表，缺源/过期/哈希不一致就停止，不重生成或重训。

## 不变的预测与学习结构

额外专家为Linear(73,1)，74参数，构造隔离RNG seed3701，出生时w=b=0；δ=2*tanh((w·z+b)/2)，输出[B0−δ/2,B1+δ/2]，分类输出始终精确复制B。loss=mean BCEWithLogits(issued_B_margin+δ,y>0)+0.001*mean(δ²)。AdamW(lr=1e−4,weight_decay=1e−4,betas=(0.9,0.999),eps=1e−8)，clip_grad_norm=1。

实际batch沿用036 C登记的update_batches，保留顺序、重复项和全部host。每16步一次机会，active时执行该次更新，absent/sleeping时跳过；不积攒、不补训练。新臂没有donor，无新初始化随机试验。参数保存方式、loss、特征、底座、路由和预测阈值不改。

## 状态机与严格事件顺序

四状态：ABSENT → ACTIVE_BEFORE_SLEEP → SLEEPING → ACTIVE_FINAL。每个状态转换最多一次；ACTIVE_FINAL持续到流结束，不再休眠。没有永久删除或第二次出生。

每个cursor t执行：
1. 按当前状态发出t预测，记录issued_state、参数版本/hash。ABSENT/SLEEPING逐项复制B，不能调用额外专家forward，不允许后台shadow推理或训练。
2. 结算i=t−2标签，计算当时实际issued预测的BCE。不得用当前权重重算旧预测来做决策。
3. 若(t+1)%16==0，按当前状态执行唯一对应的birth/sleep/wake检查，同一t最多一次状态转换。
4. 若转换后的状态为ACTIVE_BEFORE_SLEEP或ACTIVE_FINAL，执行t的一个正常登记更新；否则跳过。因此休眠事件t的本次更新跳过，唤醒事件t允许一次正常更新；不添加热身更新。
5. 变化只影响t+1及之后的预测，不改写已经发出的t预测。

预测可见标签i+2<t；检查/更新可见i+2<=t。终末两次结算只评分，不做状态检查、创建或optimizer.step。所有阶段/服务名称仅用于事后报表，不能参与状态决策。

## 出生：原样沿用037，不用t351强制触发

只在ABSENT检查。m=t−2，至少320个成熟interval；recent[m−63,m]64步、previous[m−319,m−64]256步，B issued mean-host BCE用float64稳定softplus；recent中至少16正例16负例。R>=1.25*max(P,1e−6)且R−P>=0.02，连续两次合格则创建零初始化专家，失败清零。该源预期t351出生，首条影响预测352；不符则停止审计，不能改阈值。

在首次sleep事件之前，新臂issued输出及更新参数hash须与D_keep一致，包含sleep事件t已发出的预测；sleep事件t之后的更新不再要求一致。禁止把D_keep逐步参数注入新臂以伪造一致性。

## 休眠：固定历史贡献规则

只在ACTIVE_BEFORE_SLEEP检查。从出生后首条active预测起，至少256个已成熟active interval；这用于避免把刚出生的适应过程立即判为无用。必须连续128个最新成熟interval的issued_state都属于当前active阶段，否则本次不合格并清零streak。

对最新成熟窗口[m−127,m]：
U=mean_interval(mean_host(BCE(B_issued,y)−BCE(D_issued,y)))。
U>0表示额外专家降低了检测损失。mean-host/mean-interval等权，使用稳定float64计算；不含训练正则项，不按类别事后重加权。候选条件只有U<=0，连续3次登记检查满足才休眠，U>0或成熟量不足时streak清零。记录正负支持数但本规则不另设正负最小门槛，正常期也可判断实际贡献。

休眠时保留精确权重、Adam一二阶矩及step计数，并保存checkpoint。暂停该专家全部forward/训练，不重置参数或优化器；B照常按历史输出提供预测。保存内存仍计入，不宣称已删除。

注意：该规则以BCE作为因果管理信号，但目标是AP/召回；这是需要检验的机制假设，不能假定BCE无收益等于AP无收益。报告sleep附近AP、召回变化，不据结果修改规则。

## 唤醒：复用底座误差上升信号

只在SLEEPING检查，至少64个已成熟sleeping interval，且最新64个成熟interval都在sleeping状态；recent与previous窗口仍为64/256，总计320成熟interval。用B当时实际issued BCE，不读取休眠专家输出：R>=1.25*max(P,1e−6)、R−P>=0.02，recent至少16正例16负例，连续两次合格才唤醒；不满足清零。

唤醒载入原样保存的权重与优化器，并核对睡前/醒前hash相同，进入ACTIVE_FINAL。在该t做一次普通成熟batch更新（可含睡眠期已经兑现的历史样本，这是原登记batch，不是额外回放），首个受影响预测t+1。Adam step延续上次计数，不从1重启。不得后台训练、使用未来数据、直接复制D_keep/F状态或把休眠期间跳过的更新补回来。

若未休眠：记no_sleep_observed，新臂应逐项等于D_keep，不能声称已验证管理收益。若休眠未唤醒：记sleep_without_wake，只证明停用路径，不证明复用。只有真实发生完整sleep→wake才报告cycle_observed。无论哪种情况都交付并停止；不能调低阈值、指定phase触发或加跑种子来凑事件。合成数据可强制事件测试工程正确性，但不计作真实实验事件。

## 预算、冻结与恢复

新增科学训练槽仅1个D_sleepwake；梯度上限352步（对应D_keep所有出生后机会），实际可少于352，不允许为达到上限补训练。0新流、0C/B/D_keep重训、0F相关训练/诊断对比、0donor、0其他真实流、0阈值/结构/初始化扫描；最多1birth/1sleep/1wake。

真实工程前缀最多1次≤256预测，0梯度，不用于扫描睡眠触发或择优阈值。运行器、分析器、plan和窗口定义须在真实生命周期诊断/新科学训练之前一并冻结。先做固定只读基线校验，不按读到的分数改登记。Py3.8、numpy1.24.4、torch2.4.1+cpu与单线程确定性沿用源，依赖和源码实际hash写入science provenance。

对真实optimizer.step和额外专家forward插桩：执行前检查/预留预算，执行后记录；checkpoint与ledger一致后再推进。崩溃落在不能判定某步是否已执行的边界，标ambiguous_step并停止，不能从旧checkpoint重放隐瞒成本。已启动科学槽只能精确恢复，禁止从零重跑。每次状态转换前后、每次实际更新后及每512预测保存可恢复状态，另保存终态。

checkpoint包含cursor及子步骤位置（预测前、结算后、转换后、更新后）、专家权重/优化器/RNG、issued状态与输出、成熟loss buffer、全部streak/计数、事件、时间累计、plan/source/code hash和预算ledger。恢复须验证并恢复RNG，避免出生或唤醒边界重复预测、重复结算或重复更新。完成、未触发、不可恢复、预算耗尽都终止本轮。

## 科学run前必须补齐的真实入口测试

合成fixture必须调用039同一step/state/checkpoint接口，不能另写简化状态机冒充生产路径。测试不允许读取真实流选择阈值。

- 原始出生映射、出生前/休眠期精确复制B；休眠期间实际forward和step增量均为0；保留的参数/优化器hash不变。
- 连续运行与磁盘中断恢复逐项比较：出生前、出生已创建但未更新、睡眠转换前后、唤醒已恢复但未更新、正常更新后。比较输出、事件、cursor/子状态、优化器、RNG及预算，不只比较两份内存模型的一次更新。
- 未来特征/标签真实扰动后重新调用生产入口，确认扰动非空，并确认截断时刻之前所有输出/决策/更新状态不变。
- 真实插桩验证终末只结算零optimizer.step、零生命周期变化；不能写terminal_zero_update=True或用日志长度冒充实测调用计数。
- 有效性汇总从各实测结果计算，不写死all_pass。失败阻断科学入口，不能忽略异常继续跑。

这些测试是本轮前置工程修复，不要求重新训练或覆写038结果。

## 指标与通过条件：主看D>C

沿用036/037并列分数组非插值AP、阈值0.5、六复现窗等权、late4、每窗prefix32/64、W六块、全程/回归pooled recall/FPR及混淆矩阵。不得改窗口或只挑有利片段。单类指标null，任何主门槛所需指标未定义都不能认定通过。六窗不是六个独立种子。

primary_D_over_C_signal需同时满足：
- 全程AP(D_sleepwake)−AP(C)>=0.002；六窗平均差>=0.002；六窗至少4个严格正向；late4差>=0。
- prefix32平均AP差>=−0.005；相对C全部护栏通过；全部实际有效性检查通过。

相对C护栏：全程及复现pooled FPR增量<=0.01、recall增量>=−0.01；W六块均值AP差>=−0.02；每复现窗recall差>=−0.02；每窗prefix32 recall差>=−0.03。阈值是开发阶段操作性标准，不是统计显著性声明。

management_preserved_signal单独判断D_sleepwake相对D_keep：full和six AP差均>=−0.001，late4差>=−0.001，prefix32平均AP差>=−0.005，并通过上面同形式护栏（比较基准换D_keep）及全部有效性。这是预先允许的小损失预算，不称统计非劣效证明；不能在D>C仍成立时隐藏超出预算的退化。

cycle_resource_signal需真实一次sleep和一次wake、至少64条实际sleep预测、休眠期额外专家forward和step均为0、总额外forward<5616且实际step<352，并通过实际计数审计。

progress_to_next_step_signal=primary_D_over_C_signal AND management_preserved_signal AND cycle_resource_signal。分别展示三个分量与具体失败原因，不用单一绿色状态掩盖未触发、性能退化或计数不合格。只有D>C但没有真实cycle不算动态管理已验证。

B对照仅作贡献解释：列出B−C、D_keep−B和D_sleepwake−B，不设置新的B超越门槛，不额外加载或比较F。若优势主要来自固定D_lin，结论明确为“完整候选系统保住静态增益并节省新增专家计算”，不能称动态管理本身提高AP。

固定事件诊断：对sleep和wake分别取预测切换前64步及切换后64/128/256步（切换后从事件t+1起；窗口裁剪，报告实际长度与重叠），各臂AP/BCE、召回/FPR、TP/FP/FN变化。按每个实际睡眠段统计贡献与省掉的调用。只作解释，不代替主门槛，不用未来回放结果反向决定何时睡醒。

## 成本、发布与停止状态

报告额外专家真实forward/step次数、active/sleep时长、状态检查CPU、逐进程累计wall/CPU、参数驻留积分与峰值状态字节。暂停专家计算仍需B和控制器计算；D还包含固定D_lin，其训练、推理和存储成本不能遗漏。保存权重与Adam状态的睡眠不省这部分内存。未经同机完整在线端到端测量，不宣称整体部署加速；不同run历史耗时分栏，不拼成严格速度比。

科学文件与publication_status分开：科学文件封存后不能再被分析/发布脚本修改；final manifest逐项自检后上传。保留037历史status.json那一项已知manifest异常原貌，除该项外全量核验，不扩大白名单。038原始证据不修补；本轮引用独立复核说明其测试局限。

正式入口protocol039-d-vs-c-sleep-wake.yml为workflow_dispatch_only，锁execution_ref和expected_execution_sha；并发组锁定协议，检查启动账本避免多次科学启动。连接器缺dispatch时，可用一次性push helper验证冻结SHA/plan及不存在同批次run，再POST一次正式dispatch；helper不能训练，响应不明先查runs，不盲重试，成功后移除。此次指示发布不创建helper或workflow，不重跑037/038。

原始artifact先上传，紧凑结果后发布。用Git tree API或明确配置认证的干净clone，保留最新main、不force push。仅真正非快进最多3次重建commit；认证错误立即记authentication_error/publication_pending，不伪装成非快进重试，更不能重训。发布失败保留可访问执行分支及停止状态；结果发布成功须读回确认主分支三个入口指向最新完成状态。

至少交付：source/provenance锁、合成fixture实测报告、三缓存预测与新臂预测、真实forward/step事件与budget ledger、全部生命周期检查/状态事件/checkpoints、各对照/护栏/事件诊断、累计cost、分离科学/发布status、原始和紧凑manifest、artifact_index、中文结果。AGENTS/NEXT_EXPERIMENT_LATEST/PROJECT_CONTEXT_LATEST须一致。

## 后续路线（本轮不授权执行）

039先判断一次休眠/唤醒是否保住D>C与可量化计算节省。之后另立协议测试永久删除和重建代价，逐步形成完整D。再冻结完整算法，在未用于选规则的多个流seed上与持续学习C配对确认。F研究处于DEFERRED状态：在完整D>C目标解决之前不自动恢复，不因本轮信号失败而绕回F；目标达到后再登记F探索。
