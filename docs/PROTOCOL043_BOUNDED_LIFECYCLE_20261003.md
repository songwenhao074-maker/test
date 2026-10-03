# Protocol-043：按需新增、窗口退出与有限专家池回收

日期：2026-10-03（Asia/Shanghai）。revision=1。状态：instructions_registered_not_implemented_not_started。

本次用户授权是设计并上传下一轮实验指示；本次发布不实现训练器、不触发工作流、不启动科学运行。执行模型收到用户交接后，按本文完成修复、合成预检、冻结、两条科学序列和交付，随后停止。本文取代根目录“停止等待042分析”的下一步指示；033–042预算与历史产物仍关闭、只读，不重跑042。

## 1. 唯一目标与成功条件的改变

用户明确目标：结构逻辑合理，效果超过C；不要求得到精度最高的结构，不要求新专家严格替换旧专家，也不要求窗口组超过累计组。

本轮D：有用专家保留；近期无贡献的专家退出活动集合并休眠；现有组合不足时优先尝试复用，仍不能满足需求则训练新增专家；驻留容量不足时，在证据满足条件的休眠专家中回收一个。新增与退出是独立决策，绝不强制一进一出。

仅将预指定D_bounded与冻结C_ref的比较作为主要性能判据。与D_keep、042 A_hist/A_win128及本轮D_no_gc的差值均报告，不作为必须击败的门槛。042的“累计保持必须通过”不再进入043成功定义；这是根据用户目标在运行前明确修改，而非看043结果后放宽。所有方法仍需报告误报、召回、局部窗口和实际成本。

## 2. 已知证据与不能继承的结论

042 revision2 science run=37128097274，发布基线提交19445006dee657884bca5cd800f477a8428f6493；执行代码提交da5eaa7e71f44497ede538c884e7ca688e6afbf4。
已发布AP：C=0.6828006987524914，R_win128=0.7014930612986227，A_hist=0.7061360620708045，A_win128=0.7057063219482922。这是已见开发流上的历史数值，042原始validity=false，不覆盖原报告。

原始日志记录：A_win128在1343接受E1、1615休眠E0、2351复用E0；1615时E0累计效用+0.0122337193，而窗口效用-0.0062407245。R的两次候选均未被接受，因此不能把R>C称作替换机制成功。
上述时刻仅用于旧产物核对，严禁在043控制器中硬编码事件。

本輪必须修复：
1. 042在线float32先做live-delta、离线float64先转换再相减，违反统一评分口径。
2. prefix审计同时要求同一动作集合相同和不同，且把导致首次输出分叉的前一步合法控制差异误作污染。
3. sleep事件缺active_ids_before，使多候选选择被默认空集合误判。
4. hist保留全部原始历史、评分重复拼接扫描；“无额外专家forward”不能掩盖CPU/I/O/存储成本。
5. 恢复fixture只比较载入状态，未完整续跑；所谓独立评分fixture调用同一实现，未覆盖共同错误。

## 3. 执行顺序、来源和冻结

阶段A：只读复核042封存产物，不梯度、不模型重放。输出043/audit042/下的更正分析，原042文件不变。独立重算全部控制点，不只打印前20个差异；同时复核支持数、窗口成员、连续计数、排序、选择、资格和因果时序。
分别给出“与原float32实现一致”和“与原登记float64算术一致”的结果。若规范算术会改变旧动作，记录首次差异并停止对旧轨迹后续闭环的反事实推断；不得把固定旧输出上的重算当作修正后模型轨迹。原validity不自动改成true。
旧实验仍invalid但已被完整诊断不阻止新的043；若源缺失/损坏或无法完成必要复核则保全停止。

阶段B：实现043与生产入口合成测试。所有真实梯度前冻结两臂代码、分析器、阈值、fixture证据、输入/运行环境、计划hash和预算。提交implementation_manifest和gate_report。没有再训练U_parent的预算。

阶段C：依次执行D_no_gc、D_bounded，每臂各一次完整5968区间。前臂性能差或未触发机制不阻止后臂；工程错误、输入错误或超预算立即停止。不得看前臂结果修改后臂。

来源：
- 036 raw run36831958978/artifact11147931152，ZIP sha256=040bacac8f23659f56599b71d6d6706b62a544e44161c7c42fb5af73650f56c0。
- 041 raw run37110969550/artifact11269043834，ZIP sha256=baa36f1fbb3489f76d8f0ffb5d8e4ebc88317e231bd934146e185ac2e4ffde6e，用于冻结C/B/D_keep等缓存；不是重训授权。
- 042 raw artifact11276470062，ZIP sha256=764913d9ffb3e26d2895b1ba802595458b5efdeb46c8bc118c226c2c2c7bc991，用于只读审计。
完整文件锁见plan.json；取得实际ZIP字节并核验manifest，不以服务器摘要替代下载验证。源不可用则source_unavailable，不换流或重新生成。
042参考代码文件在执行提交仓库根目录：protocol042_common.py、protocol042_engine.py、run_ftmoe_protocol042.py、analyze_ftmoe_protocol042.py；不是scripts/目录。仅从固定SHA提取参考，历史文件不回写。

数据固定seed3601、5968×16、73维issued特征、真实C更新批次和原顺序/重复。数据已被多轮观察，本轮属于开发确认，不声称独立泛化验证。
预测i标签于i+2结算；时刻t先预测再结算t-2，再控制和训练。阶段/故障类型/事件ID只用于离线分析。C和B=C+D_lin缓存只读，分类分支仍精确复制B，F禁用。

## 4. 两臂与共同结构

| 臂 | 活动退出、新增、复用 | 满驻留容量处理 | 角色 |
| --- | --- | --- | --- |
| D_no_gc | 本文统一策略 | 拒绝新增并记录capacity_blocked；不永久删除已接受专家 | 回收关闭对照 |
| D_bounded | 与D_no_gc相同 | 满足第7节才回收一个休眠专家并创建候选 | 唯一主臂 |

唯一策略差异为永久回收是否启用；两臂相同预算、初始化、输入、随机数和控制代码。若自然流无合格回收机会，两臂完全一致是正常结果，不追加实验。不能要求主臂一定优于此对照。

每个专家Linear(73,1)，74参数，delta=2*tanh((w·z+b)/2)。live margin=B_margin+sum(active delta)；不重归一、不裁剪总和，空集合精确复制B。最多2 active、1 shadow，active+dormant+shadow总驻留上限3；因此dormant上限随其他状态变化，绝不是无限档案池。参数/Adam和临时复制的真实内存另计。
状态：shadow→active（验收成功）/disposed（失败）；active→dormant；dormant→active（复用成功）/deleted（容量回收）。无直接active→deleted。新增与复用均只在active<2时提议；active满时先由独立退出规则决定是否休眠，不能为了创建候选主动挑一个仍有用的专家替换。
ID全程单调不复用；最多E0+4次后续候选，即最多5个创建ID，而非最多3个终生ID。已删除ID不能从审计文件/旧检查点重新导入。审计保存证据不是在线可复用内存。

E0保留042的首次压力出生规则，零初始化直接活动，是明确的bootstrap特例。之后包括空活动集合时的新增均需16次shadow训练和未来验收。
训练保持042：每due16使用同一登记32区间×16 hosts批次，live联合BCE+0.001*sum(mean(delta²))；各专家独立AdamW(lr=1e-4,weight_decay=1e-4,betas=.9/.999,eps=1e-8)，各自clip=1、各自step计账。
shadow零权重/偏置、新Adam，目标BCE(B+detach(所有活动专家更新前贡献)+delta_new)+0.001*mean(delta_new²)，复用已有背景forward；只训练16次后冻结，不能对历史困难样本加权、复制父修正或额外训练。
休眠完全停止正常部署/训练，保留参数和Adam；复用接受后保留同ID/Adam继续训练。永久删除释放在线参数、Adam、缓存和可加载引用，预算与审计tombstone保留。

## 5. 窗口退出与精确低成本评分

只用当前deployment_epoch的最近128个全局已成熟issued区间[m-127,m]，m=t-2。活动集合变化才切epoch；普通梯度更新、shadow创建、休眠对象删除均不切epoch。跨epoch迟到标签仍计算评价指标，但不得加入新epoch的控制统计。
从保存的float32 issued live margin和各专家delta开始，分别转换float64，再计算：
u_i=BCE(live64-delta64,y)-BCE(live64,y)。
正值表示专家有帮助。禁止先float32相减再转double，禁止用当前权重重算旧样本。稳定BCE全部float64。

评分支持：当前epoch128成熟区间、128完整成员、正负host各>=16、含正标签的不同区间>=4。未知/null与0严格区分，休眠专家没有当前在线移除分数。
连续3个due16同时满足S<=0、S_pos<=0、S_neg<=0、移除FPR增量<=.01、移除recall增量>=-.02才可退出；未知/不满足清零，epoch切换全部重置。多个符合者选S最低，精确同分选较小ID，每次最多休眠1个。不能用正分中的最低分作为无用证据。三个重叠窗口不是独立显著性检验。

维护每区间sum(u)、sum_pos/neg、正负计数、positive_interval指示、移除/不移除的TP/FP/FN/TN，环形容量128；增量入队/出队维护float64累计值。不在线调用AP排序，不在线保留不断增长的原始历史数组。压力只需有界320区间统计；资格至多32区间/槽。审计输出流式写盘。
独立审计从封存issued原始数组重新扫描，不能调用生产tracker/同一聚合函数；绝对容差1e-10、rtol=0仅用于数值核验，不用于改变<=阈值。所有离散条件、候选排序和动作必须完全一致，包括近0测试。若增量误差导致动作分叉，视作工程失败，不事后扩容差。
可以在接近阈值时用当前有界128区间统计重算以防漂移，但规则须冻结；不得增加专家forward。单独报告该回退次数与耗时。原始数据的逐项重算器必须能发现人为注入的dtype/成员/符号错误。

## 6. 缺口触发、复用与新增

压力使用本臂已发行live BCE：最近64均值R与紧前256均值P，R>=1.25*max(P,1e-6)且R-P>=.02，近期正负host各>=16，连续2个due16成立。首次E0至少320成熟区间；后续active非空要求当前epoch>=256成熟，空集合>=64成熟；近期64必须同epoch，之前256可跨epoch。拓扑变化/条件失败清零。压力是“值得尝试新专家”的代理，不能证明所有旧专家无用，也不能检测所有持续性能力不足；本轮不改为oracle或调低阈值。

复用：每due64、epoch>=128成熟、active<2、有dormant且无pending reuse时可开始一个槽，不要求压力；最多32槽/臂。同槽对全部休眠专家（至多3）作32个未来区间preview，冻结其参数，live继续训练；记录全部候选的系统输出与支持。preview系统=实际live+休眠delta，不减任何活动专家。不能用休眠前旧分数决定当前是否值得复用。
新生：当前压力成立，active<2、无pending reuse/shadow、无同t活动转换、尚有候选尝试预算、上次候选结束已过256区间。若有dormant，必须具有最近64个区间内已作出的、同epoch且休眠ID/hash集合匹配的充分支持复用质量拒绝（该槽所有候选都未通过）；支持不足不授权新增。无dormant时无需此证据。先检查这些非容量条件，再判断是否需要回收。

资格统一：shadow完成16更新后冻结；从下一次预测起，取恰好32个新issued区间。reuse也是从槽开始后的下一次预测取32个未来区间。标签全部成熟后的首due16决策。候选系统BCE<=.99*live BCE且<=B BCE；相对live和B，FPR增量各<=.01、recall增量各>=-.02；正负host各>=16，全部有限，threshold=.5，不计训练正则。质量不通过/支持不足/取消不延长窗口、不补训练。多个复用候选通过时选BCE最低，同分较小ID。
验收证明“加入该专家后的组合有改善”，不要求新专家单独超过旧专家。
shadow与reuse可并行，但同一时刻仅一个shadow、一个reuse槽；reuse接受优先，活动拓扑变化取消所有旧epoch pending槽，已消耗预算不返还。
控制顺序：发行live及已有资格preview→结算→due16至多一次活动转换(first_birth > reuse_accept > shadow_accept > sleep)→无转换时，符合全部新生条件则先尝试新生；否则在无pending reuse时按due64条件启动复用（允许已有shadow继续）→活动更新→shadow更新。
新建槽从t+1采样；转换从t+1预测生效，t的训练按转换后活动集合。shadow创建/永久回收不改变active集合，可在同一原子事务内发生，但不能与同t活动转换混在一起。终末两次settle只结算，不控制、不训练、不接受。

## 7. 有限休眠池与永久回收

正常运行不按年龄定时删除；年龄只是保护期和合格者的排序依据。仅当全部新生非容量条件已经成立、驻留数=3导致无法创建shadow时考虑回收。

一个dormant专家可回收必须同时满足：
1. 自最后一次退出活动集合已休眠>=256个预测区间，权重/Adam未变，非任何pending槽占用。
2. 在当前epoch、当前同一休眠ID/hash集合下，最近两次已完成的复用窗口均有充分支持、互不重叠，且两窗起点均>=t-255，最新决策满足0<=t-decision_t<=64。支持额外要求每窗正负host各>=16、含正标签区间>=4。固定delay成熟时间逐项核验。
3. 两窗中该专家均未通过复用；并且其整体、正类、负类增量效用Q=BCE(live)-BCE(live+delta_dormant)都<=0。仅因为达不到1%改善门槛而被拒绝，但Q>0，不能据此删除。
4. 最新槽中全部休眠专家都有充分支持并被质量拒绝（第6节新生前提）；未知不作无用，不拿历史高分或低分代替近期preview。

多个合格者按last_active_prediction升序、再expert_id升序选择一个（最久未使用者优先）。不比较来自不同时间窗的陈旧原始分数。候选表必须列全部驻留ID、状态、保护期、两次证据、Q_pos/neg与否决原因。无合格者则capacity_blocked_no_safe_eviction，允许本次不新增。
D_no_gc在同一机会只读计算并记录would_delete_id，但不删除/不创建，且不消费候选尝试；D_bounded原子记录“删除休眠专家+消费一次尝试+创建零初始化shadow+预算/next_action”，任一阶段崩溃不得造成重复删除/重复候选。
删除发生在候选训练前，因为shadow也占驻留槽；候选随后失败时，被删专家不会复活。这是有限容量下的明确风险，必须统计；不能临时允许第4个专家来偷偷避免风险，也不能删除活动专家。
永久回收不证明该专家永远无用，只表示在当前证据和有限内存约束下放弃保留。未形成长期回收证据时，宁可阻塞而不强删。
每次删除最多一个；总删除数不超过4次后续候选尝试。不因流尾或试图凑事件而删除；创建前必须还有足够未来due16训练点与完整32窗、成熟和决策时刻，否则tail_budget_unavailable且不删除。此检查只能用流长度/时钟，不能读取未来标签。

## 8. 必须通过的工程测试

所有有梯度和生命周期的预检用合成输入，经过生产tick/训练/控制入口；不通过直接调用sleep/delete私有方法伪造机制证据。可以构造合成初始权重与带标注的初始池，但压力、评分、未来验收和状态转换必须真实计算，不能给决策函数预填pass。
每个fixture记录输入种子、真实事件序列、参数/Adam前后hash、实际optimizer调用、checkpoint hash和断言。缺证据/空列表/字符串true均fail closed。

必测：
- 加法/移除算术、float32取消误差反例、float64金标准、阈值两侧与精确同分；独立评分器能抓故意修改的成员、符号和dtype。
- 固定全局窗口到期、epoch切换后迟到标签、支持不足/null、稀有正类保护、三个检查计数；正常/故障阶段信息不能进入控制器。
- 两个有用专家不退出；有害者退出；全体未知不删；active满不强行替换；空活动集合退化B并允许随后候选。
- 训练共同更新前图、独立Adam、detach背景、shadow16次后冻结、32未来资格、尾部取消；未来标签扰动不改变此前预测/动作。
- 休眠保留参数/Adam、真实复用接受；少量正收益但不到1%不得永久删除；仅一次拒绝/陈旧拒绝/不同epoch或hash/重叠窗口/支持不足/保护期不足都不得回收。
- 满容量下所有非容量条件成立→两次真实非重叠复用拒绝→选择dormant→永久回收→新候选训练→验收接受并至少64次活动预测；必须由生产入口完成。另一fixture覆盖回收后新候选失败、删除对象不复活、无合格对象阻塞。
- 上述回收fixture对两臂成对执行：首个GC动作前参数、Adam、输出、批次和离散事件一致；之后no_gc阻塞、bounded继续，非强制把两臂同时改池。
- 真正磁盘恢复：连续运行和销毁对象后的恢复运行都继续到同一终点，逐项比较后续预测、参数、Adam、RNG、窗口/压力、槽、事件、删除tombstone和预算。覆盖出生、双活动联合step事务、shadow1/15/16、ready未决策、accept/reject、sleep/reuse、窗口出队、回收创建事务、终末两次settle之间。只比较刚载入的快照不算通过。
- 崩溃注入覆盖step与计数、删除与候选创建、预算commit前后；原子恢复或明确ambiguous_state停止，不重放未知step。多专家联合训练作为事务处理，不能恢复成一半专家已更新。
- 缺失报告/错hash/非布尔pass/错revision/超预算/非法状态都阻止每个科学入口，不只在workflow检查；分析器面对缺失字段返回schema_error而非默认假/真。

## 9. 预算、运行与持久化

只授权两条新科学序列；同一已见流，无额外seed/流/F/窗口扫描/初始化扫描/U复现/额外臂。每臂E0一次，后续候选尝试最多4；拒绝、取消均消费尝试，ID不复用；每次结束冷却256区间，压力须重新满足。
每臂硬上限：live专家step=704，shadow step=64，总optimizer step=768；部署专家区间forward=11232，reuse preview=32槽×32区间×3专家=3072，shadow资格=4×32=128，预测相关合计14432；训练forward另计<=768，背景forward复用。每due16最多2 live+1 shadow step。
两臂合计：optimizer steps<=1536；部署<=22464，reuse<=6144，shadow资格<=256，预测相关<=28864。上限不是目标，不为用满预算制造事件。E0不应早于320成熟，具体step上限沿用036流时钟推导并在预检验证；出现超限即停止，不扩大预算。
真实工程prefix最多一次<=256区间、零梯度（仅输入检查），其余工程训练全为合成。只读042复算不计科学序列，但计CPU成本。性能计时不重跑训练。

环境Python3.8.18/Torch2.4.1+cpu/NumPy1.24.4、单CPU线程。执行分支codex/protocol-043-bounded-lifecycle-20261003；后续由执行模型创建protocol043-bounded-lifecycle.yml，仅workflow_dispatch，不随计划push启动。两个入口均验证同一execution SHA、完整源码bundle、schema/计划hash、非空fixture证据、输入hash与预算。
跨run预算key=(protocol043,revision1,arm)，科学开始前落盘started；原子checkpoint包含next_operation、所有参数/Adam/RNG、窗口/压力/资格、候选计数、接受/退出/永久删除注册表、预算及终末进度。只允许同冻结代码精确续跑；不能从零重跑或修改算法继续。
分析/发布失败只处理封存产物；不重训。工程失败保全并停止，报告实际完成范围；不自动进入044。

## 10. 分开报告性能、结构与回收证据

主要性能D_over_C沿用042对C的固定门槛：
- full AP差>=.002，六复现窗等权AP差>=.002，至少4/6窗AP差>0；
- late4等权AP差>=0，prefix32等权AP差>=-.005；
- 全程与复现窗合并FPR差<=.01、recall差>=-.01；W段等权AP差>=-.02；
- 每个复现窗recall差>=-.02，各prefix32 recall差>=-.03。
六窗、late4和分析算法锁在plan.analysis，threshold=.5，AP为tie-grouped noninterpolated，单类AP=null。全部指标独立从封存预测计算。操作性门槛不是统计显著性；不挑最好臂、不以BCE代替AP、不要求胜过A_hist/D_keep。

独立布尔量：
- validity：043来源、无泄露、预算、状态机、全量评分/指标复算、实际恢复证据都通过。
- D_over_C：主臂达到上述性能门槛；即使validity=false也展示原始数值，但不称有效结论。
- core_lifecycle_exercised：真实主流中至少一个E0之后的候选经资格接受且活动>=64预测；至少一次窗口退出且该ID保持休眠>=64预测；两类事件不要求同一ID或同一时刻。不把bootstrap或未发生替换当作新增证据。
- reuse_exercised：真实主流同ID休眠>=64后复用接受并活动>=64；独立报告，不替代新增。
- gc_engineering_verified：第8节真实生产入口合成回收闭环与恢复fixture通过。
- gc_exercised_on_stream：真实主流至少一次符合第7节永久回收。
- gc_admission_closed_loop：被回收事务创建的候选随后接受且活动>=64预测。
- resource_bounds_pass：每时刻驻留<=3、active<=2、shadow<=1、窗口/压力/资格有界；内存审计包括临时副本，审计磁盘单列。
- core_goal_supported = validity AND D_over_C AND core_lifecycle_exercised AND gc_engineering_verified AND resource_bounds_pass。
- full_reclamation_demonstrated = core_goal_supported AND gc_admission_closed_loop。

若core_goal_supported=true而自然流没回收：结论为“D的新增/退出已运行且超过C；回收通过工程测试，真实流上的回收效果尚未验证”。不得把合成事件填入真实生命周期；也不能因未触发就宣布结构失败或自动换流补跑。
D_bounded不必优于D_no_gc；若有回收但差于对照且仍超过C，仍可满足目标，同时如实报告回收的代价。候选回收后失败、复用损失和局部退化均不能隐藏。
没有独立seed/流，不输出泛化确认、最优窗口、最优结构、等算力胜C、永久删除永远安全、整体部署提速等结论。

两臂比较以“首次真实控制分歧”作边界：其前输出、参数、Adam、更新和动作应一致；分歧应来自回收开关，在动作生效后的输出/更新可不同。禁止重新使用042中actions_equal AND actions_not_equal的逻辑。若没有GC分歧则要求全程科学轨迹一致（臂名/时间/审计字段按冻结schema排除）。
针对042窗口误判的修复单列验证控制时刻1615→首受影响预测1616的旧日志关系；不能要求产生分歧的控制动作仍相同。

## 11. 交付与停止

新增代码建议protocol043_common.py、protocol043_engine.py、run_ftmoe_protocol043.py、analyze_ftmoe_protocol043.py、maintenance/validate_protocol043_plan.py；不将旧错误代码原样改名视为完成。
先封存raw manifest和源码bundle，上传artifact保留>=90天，再分析并提交：
- docs/PROTOCOL043_RESULTS.md；
- artifacts/ftmoe_online/protocol_043/runs/run_<id>/，含source/implementation/gate/fixture/registration/budget与恢复证据；
- audit042/：旧结果的全量独立复算、首次差异、机制判定更正及不能修复的限制；
- 两臂issued live/B/delta/ID/version/epoch、预测与标签结算索引、更新/压力/资格/机会/退出/复用/回收完整日志；
- 每次候选和回收的全部候选表，active_ids_before/after、resident_ids_before/after、action ID、首受影响预测、释放字节、tombstone和取消原因；
- 独立指标/评分复算、性能比较、分开的机制/有效性布尔量、首次控制分歧检查；
- 成本：专家部署/训练/preview调用，实际step，评分/控制器/训练/预测/I/O耗时，在线常驻和峰值内存、持久化体积。缓存B的成本与D增量分开，不能把缓存实验当端到端加速；
- 实际删除后未来表现只从已观测轨迹描述；没有被删专家未来输出就标unknown，不能声称知道其未来无用。

结果提交使用[skip ci]、非force推送，保留原失败证据。同步根目录AGENTS/NEXT/CONTEXT为本轮完成或阻塞状态，不提前宣称通过。两条序列完成或任何工程阻塞即停止，等待用户分析交接。
