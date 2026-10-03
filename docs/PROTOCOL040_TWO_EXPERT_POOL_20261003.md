# Protocol-040：保留通用学习路径的最小双专家动态池

日期：2026-10-03。状态：**仅登记指示，尚未实现、尚未启动**。
用户要求设计并推送方案，再由其他模型执行并把结果放回 GitHub。本次提交不创建或启动工作流。执行模型收到用户明确交接后，可完成本文件规定的一次有界实验及结果发布；不追加下一轮。

## 1. 唯一问题与目标

用户最终目标是通过合理修改结构，让保留动态新增、知识保存/选择、休眠复用与容量管理思想的完整 D 优于持续学习 C；无需复刻早期 D 的组织形式。本轮只探索一个结构包：**保留 B=C+D_lin，在其上建立最多两个动态专家、通常只激活一个的可重复管理池**。首要判定仍是 D>C，不要求战胜 F，不要求每个管理模块单独提高 AP。

039 在已见 seed3601 上全程 D−C AP=+0.01878116846464073、六窗=+0.012708018448589875，但只允许一次出生/休眠/唤醒。040 将其扩展为双专家池及可重复休眠/复用；不把 039 标成完整动态增删已成功。
本轮仍不做永久删除；容量满时拒绝继续新增并记录。后续是否需要回收，依据真实容量/效用结果另定。039 文末“下一步删除”是历史路线建议，由本轮用户认可的结构探索优先级取代，不改写039历史。

这是一组预先冻结的联合结构修改，不能把结果单独归因于专家数量、复用、验收或触发中的某一个。只用已观察 seed3601 做开发；不以6个窗口或16台主机充当独立重复。

## 2. 源码与数据身份

方案发布前 main：3518e3d5f9c2cca78d0d0467ed45a04f610c851d。
039实际科学checkout：600ebb0a1691c72fe7f451bae2c8e8357ad4add4。
执行分支建议：codex/protocol-040-two-expert-pool-20261003。
**main没有完整039运行器**。从包含本登记的main建执行分支；从上述固定039 checkout读取所需内核/依赖，核对Git blob与SHA256，添加040专用入口，不覆盖旧科学实现。不能从本地旧024分支猜测重建当前算法。

输入只需：
- 036 run36831958978，artifact11147931152；ZIP SHA256 040bacac8f23659f56599b71d6d6706b62a544e44161c7c42fb5af73650f56c0。
- 039 run36982845152，raw artifact11216467452；登记API digest为 sha256:03a25b90b28f09f3dab3f1704a3d52053b0750f3616f3c0ed03dbda374c2ae0a。039逐文件manifest固定引用方案基点下 artifacts/ftmoe_online/protocol_039/runs/run_36982845152/science/scientific_raw_manifest.json；下载后执行实际ZIP与逐文件校验，不把本指示引用digest当成已经下载核验。
- seed3601 stream SHA256 09fadb02f2017f8d528ee8284129c29adcb6be137b1a9e93f4a249eebcd1b659。5968计分时刻×16hosts；73维issued特征；目标是同host raw[t+1]异常，标签在cursor t+2的预测之后成熟。
- 036提供 feature_tape.npz、实际update_batches.json和timeline manifest；039提供C_ref、B_ref、D_keep、D_sleepwake的原始预测。具体逐文件hash写入plan.json。
- C_ref是持续学习的五专家C；B_ref是036已在线训练的C+D_lin输出；D_keep来自037出生后常开的D；D_039是039结果。它们全部作为缓存控制，不重训。B仅为本协议符号，不能混同早期“全模型微调B”。
- 不加载/重算/比较F文件。缺少或过期输入则交付blocked_input并索取同一份原始文件，不能重生成、换seed或重新训练历史控制。

## 3. 冻结模型与训练内核

最终检测logits=B logits+[-delta/2,+delta/2]，其中delta来自唯一活动动态专家；无活动专家时逐项复制B。分类输出始终精确复制B。本轮不宣称资源分类改进。
每个动态专家沿用039：Linear(73,1)，74参数，零初始化，delta=2*tanh((w·issued_z+b)/2)；独立AdamW(lr=1e-4,weight_decay=1e-4,betas=(0.9,0.999),eps=1e-8)，clip_grad_norm=1。
loss=mean BCEWithLogits(issued_B_margin+delta,y>0)+0.001*mean(delta²)。
构造使用隔离RNG seed3701，零初始化不扫描随机种子。两个专家都学相对于B的完整单专家修正；**不是把候选叠加到当前活动专家上**。

活动专家每16步最多更新一次，使用036 C当时登记的成熟batch，保留重复、顺序和全部host；不重新采样、不筛选phase、不补回休眠跳过的更新。通用B轨迹、特征、标签、归一化、检测阈值0.5均不改。
休眠专家的权重与Adam状态冻结；复用时恢复同ID和原优化器step。专家选择是整个16-host批次共享的全局选择，不做逐host路由。除固定有限资格验证外，不后台调用休眠专家。

池总容量=2，包含已接受活动/休眠专家和未接受shadow；只允许一个active和一个shadow。单独计入检查点/序列化副本的内存，不把副本当可部署专家；不复制多个隐藏模型做不记账搜索。出生ID单调递增，失败候选ID不复用。

## 4. 统一因果顺序与状态

每个cursor t严格：
1. 用预测前状态发出live预测；若有验证槽，给该槽尚缺的前瞻interval发出候选完整预测，均在标签揭示前。活动专家一次forward可复用为live，不重复调用。
2. 结算i=t−2的已发行预测与候选预测，计算float64稳定BCE及阈值计数；禁止用当前权重回算历史表现。
3. 在(t+1)%16==0时处理到期控制决策，最多一次live拓扑改变，优先级：首次出生 > 已完成reuse验收 > 已完成shadow验收 > sleep。若高优先动作发生，跳过本次其他live动作。
4. 拓扑改变使尚未决定的旧reuse/候选槽失效，记录具体取消原因。没有改变时，按第6/7节启动新的检查或shadow；不补偿错过的机会。
5. 更新转换后的active一次；若shadow处于TRAINING且未取消，再用同一成熟batch更新shadow一次。shadow达到16次实更新后冻结，开始等待后续新预测验证。任何变化只能影响t+1之后的预测。

控制器只读取已经结算的标签；预测可见i+2<t，结算后决策/更新可见i+2<=t。cursor5968/5969只补结算计分，不做验收部署、事件或梯度；流尾未决槽统一标censored。
active替换时“旧active入库＋候选出库”是一笔原子拓扑转换，可以包含两个事件记录，不是允许同t连续作两次独立决策。
显式保存active_id或null、各ID角色、活动/休眠epoch、首次部署/上次休眠时刻、streak、reuse槽、shadow训练/验证槽、全局预算；不存在ACTIVE_FINAL，接受过的专家可以多次active↔dormant。

## 5. 首次出生、重复休眠与未解决错误

首次专家E0：
- 完全沿用039的初次birth：至少320成熟interval，recent64/previous256 B issued mean-host BCE，R>=1.25*max(P,1e-6)、R−P>=0.02、recent至少16正/16负，连续两次16步检查。
- 由数据触发；在该固定源应t351出生，检查该事实但禁止按t强制创建。首次创建直接active、同t允许正常更新；首条影响预测352。
- 在040首次实际区别于039的模型动作之前，核对live预测和参数更新hash前缀一致；控制器诊断不应改变模型RNG。不得注入039参数伪造一致。

重复sleep：
- 当前active epoch至少有256条已成熟active预测；最近128条成熟预测全部由当前active epoch发出。
- U=mean(B_issued_BCE−D_live_issued_BCE)。连续3次16步检查U<=0则active转dormant，保留权重及Adam；该t不再更新该专家，下一预测用B。
- 不满足资格或U>0清零sleep streak；每次激活均重置epoch/streak。没有永久睡眠/活动终态。
- 正在做reuse或shadow不阻止sleep；sleep取消旧槽及未接受shadow，记录cancelled_by_sleep，保留已接受记忆。

未解决错误pressure：
- 对**实际完整D issued BCE**计算同样的recent64/previous256比例与绝对增量、16正/16负、连续两次16步检查；首次birth仍单独用B规则。
- 最近64成熟interval必须具有当前同一active_id/epoch（或同一无active epoch）；active存在时其epoch至少256成熟interval，无active时至少64。
- 前256窗口允许跨历史状态，但全部是当时真实live预测；禁止用当前模型补算。
- pressure只是一次检查时的布尔条件，不永久锁存；失效清零，拓扑改变全部清零。不能因B损失高但当前D已解决而创建新专家。
- 此压力规则不保证每次稳定高误差都触发；未触发属于结果，不能事后新增绝对阈值补救。

## 6. 旧专家的非阻塞前瞻复用

每(t+1)%64==0，若存在已接受dormant、当前部署epoch已有至少64条成熟预测、无pending reuse、reuse启动次数<32，启动一次reuse槽。它不要求pressure为真，不用phase、相似度硬阈值或未来回归边界。
最多对池内两个dormant都验证（有active时至多一个）。候选保持冻结；记录全部ID与hash。从t+1开始取**连续32条新预测**，每条计算B+候选完整输出与同期实际live，不把候选自身与完整模型比较。获得32条成熟配对后，在下一个16步检查点验收；采样满32后不再追加推理、不挑选窗口。
若期间active_id/epoch变更，取消整个槽；active正常梯度更新不取消，比较对象就是这32条真实在线live输出。

每个候选资格同时满足：
- 32×16配对至少16个正例、16个负例；所有数值有限；
- 候选平均BCE <= 0.99×live平均BCE，且 <= 同期B平均BCE；
- 候选相对live和B分别满足FPR增量<=0.01、recall增量>=−0.02（阈值0.5）。
这是固定资格阈值，不是统计显著性，也不假设BCE改善必然提高AP。正负支持不足直接rejected_insufficient_support，不延长验证。
多个合格候选选平均BCE最低者；数值严格相等选较小ID；没有合格候选则保持live。接受时旧active（若有）休眠，选中ID恢复active及保存的Adam，重置epoch，同t可进行一次正常更新，输出从t+1切换。
记录权重/Adam恢复前hash与上次休眠hash一致。验证会调用休眠专家，须单列preview调用；只能说“没有训练且未部署”，不能说休眠全程零forward。

reuse槽可以与shadow训练并行，不能因为birth busy直接跳过。若reuse接受，取消shadow训练/验证、释放未接受候选，记cancelled_by_reuse；不会删除已接受专家。reuse失败不重置shadow或active训练。

## 7. 第二专家：一次有界候选机会

全流最多创建E0加一个新shadow，共2个专家ID；第二候选只尝试一次，拒绝/取消后不得重新初始化再试。这样避免在同一流上筛选多个隐形候选；本轮不会检验无限持续出生。
创建shadow仅在16步检查点同时满足：
- E0已创建，第二尝试尚未消费，总容量<2，pressure当前为真；
- 当前没有pending reuse；若存在dormant，必须先有最近64个cursor内、相同部署epoch且相同候选ID/hash集合的reuse资格失败记录。支持不足不算“已有记忆无效”，不能据此进入新生；需要有效支持但质量不合格。无dormant时无需此记录。
- 此步没有live拓扑改变。先尝试复用已有知识，再学习新知识；容量满只记录capacity_block，不purge、不替换已保存权重。新生机会日志按16步到期点记录pressure、容量、既有记忆验证与尝试余额，不把第二尝试已耗尽误写成容量阻挡。

候选零初始化，独立Adam；初始输出相对B为零。候选不参与live、不影响B/active，沿共同16步机会进行最多16次额外shadow更新，每次同一个C成熟batch，无额外采样/热身。active可同时正常更新。第二候选学习的时间段由上述因果触发决定，不按U/V/W标识筛样本。
完成16次实际更新后冻结，从下一cursor开始积累连续32条新预测做shadow资格；该32条不得反流训练候选。资格、支持、tie规则与第6节完全相同。live拓扑变化取消候选，active普通梯度变化不取消。
通过则旧active休眠，候选成为已接受active；保留其训练所得Adam moments；同t再做一次正常live更新可行，必须分别计入shadow16与live预算。失败记rejected并释放未接受候选。若reuse/shadow同点完成，先处理reuse；reuse失败后仍可处理shadow，reuse接受则shadow取消。
shadow验证期间仍可启动/推进reuse。reuse需要的候选仅为已接受dormant，不能把shadow伪装成历史记忆。

## 8. 预算与实现前验证

科学预算只有**一条D_pool2完整新训练序列**，不得把每个专家视为额外独立流：
- 0新模拟流、0其他seed、0控制臂重训、0F训练/比较、0donor、0超参扫描。
- live optimizer.step全池累计<=352；shadow累计<=16；科学梯度总计<=368，每16步机会最多live1+shadow1，不补训练。
- 已接受动态参数最多148，单次live预测动态参数最多74。已部署prediction forward总数<=5616；reuse preview最多32槽×32interval×2候选=2048；shadow qualification preview<=32；preview总计<=2080。分别计数，不能合并成“与039严格同计算”。
- 不给反复sleep/wake另设任意时刻表；受固定检查时钟、成熟支持、最多32次reuse槽自然限制。记录未触发与预算耗尽，不改规则强凑第二专家或重复复用。
- 一次真实工程前缀<=256步、0梯度，仅查数据/零输出；其余工程测试用合成输入，必须与生产状态机/序列化入口共用。
- 科学开始后只允许精确checkpoint恢复，不能从零重新跑。预算事件先预留、实调用后提交；无法确认某次更新是否已执行就标ambiguous_step并保全停止。
- 工作流单job、CPU单线程、Python3.8/torch2.4.1+cpu/numpy1.24.4，完整依赖冻结；timeout建议350分钟，接近上限先保存，不换seed或跳过尾窗。

实现后、访问040真实后段结果之前冻结代码/plan/hash，验证：
1. 无动态专家时精确B；禁止操作时能复现039规定的固定行为fixture，真实前缀不误触发。
2. 验收预测在标签前产生；历史缓存不能拿未来权重补算；候选32条不混入训练。
3. 2容量计数包括shadow；两专家的梯度/Adam隔离；休眠不更新，复用hash恢复，拒绝/取消不伤已接受记忆。
4. 同点reuse与shadow接受、sleep取消、候选满16、样本不足、满容量、流尾censored、机会守恒和至多一个live转换。
5. 真实生产入口合成连续/磁盘恢复对齐预测、参数、Adam、RNG、控制槽、后续决策、预算；覆盖上述竞争边界，不比较两份同内存副本冒充恢复。
6. 改变非空未来特征/标签再运行，已发行前缀、决策和更新不变；末尾补结算实际零更新/转换；AP ties与null支持检查。
每次实更新、拓扑事件、槽创建/结束及每512预测保存完整状态，包括子步骤、pending issued预测、hash与账本。禁止用硬编码true代替fixture证据。

## 9. 评价：质量、结构运行与效用分开

复用039同一计分器：并列分数组非插值AP，单类别AP=null；六窗U/V_rec1/2/3固定[3300,3428)、[3808,3936)、[4316,4444)、[4824,4952)、[5332,5460)、[5840,5968)；late4后四窗；prefix32/64逐窗；W_long及五W_gap按manifest。所有窗口与完整支持都报告，不事后扩窗。

主要D_pool2−C信号：full AP>=+0.002、six>=+0.002、正差窗>=4/6、late4>=0、prefix32>=−0.005，并通过全部有效性及共同护栏。
**累计保持预算固定对D_keep**：full/six/late4 AP差均>=−0.001、prefix32>=−0.005及同形式护栏；不能每轮相对上一版重新赠送−0.001。D_pool2−D_039全部指标另列，无论正负。
共同护栏（分别相对C和D_keep）：全程与六窗pooled FPR差<=+0.01、recall差>=−0.01；六W块平均AP差>=−0.02；每复现窗recall差>=−0.02、每prefix32 recall差>=−0.03。

结构指标：
- pool_exercised：两个不同ID都通过各自规定的接受路径并各产生>=64条真实active预测；至少一次同ID在有>=64条实际dormant-issued预测后被重新部署，随后再产生>=64条active预测。
- 仅一次E0睡醒、第二候选拒绝或始终不触发，不算pool_exercised；真实发生的有限能力照实报告。
- 每个ID首次激活之后前64条实际active预测（需连续同epoch，否则censored）的BCE/AP/recall/FPR与同期B对比，作为局部贡献描述；有效支持仍至少16正16负。两个ID各自BCE优于B可记local_two_expert_utility，但不证明专业化或相对另一专家优越。
- 每次reuse/新生切换报告前64、后32/64/128的四缓存控制与D_pool2指标、窗口重叠、期间再次切换；若再次切换，完整窗口是系统轨迹，另报连续epoch片段，不能归因于单个专家。
- 报告所有候选未来验证表，包括未被选择的候选，不选最好一次。按事后phase汇总ID使用，不把出生于U直接叫U专家。phase不进入在线选择。
- 没有no-reuse独立对照，本轮不声称已证明记忆复用的因果收益或最优路由。不是对所有固定拓扑的优势证明，F保持DEFERRED。

结果标签按顺序：
invalid_execution；否则primary未过→D_over_C_not_established；
primary过但累计保持未过→D_over_C_with_excess_pool_loss；
两质量项过但pool_exercised假→quality_retained_pool_not_exercised；
三者过→quality_retained_pool_exercised。
另列local_two_expert_utility，不把不充分支持填成true。
上述均为开发判定，不叫完整动态删除成功或统计非劣效证明。不能因为结果不理想在原run中改阈值、延长shadow训练或追加候选。

## 10. 成本、交付与执行交接

报告B历史成本与本轮增量成本分开：live、reuse preview、shadow training/qualification、控制器、检查点/I/O的调用/时间；参数、Adam、验证缓存和检查点的内存范围分清。休眠保留状态不省该模型内存；新池可能比039更贵。未同机重测完整B路径，不宣称整体部署加速。
给出专家部署占比、每ID实际训练数、等待/取消/拒绝/容量阻挡的完整计数与机会守恒，两个专家接受/退役时间、实际first_affected_prediction，不复用全局首事件时间。

执行模型收到用户交接后：核验源→实现040入口/分析器/固定测试→冻结实现→只启动一次科学序列→分析已封存产物→推送结果→停止。
不得启动旧工作流或恢复F，不自动开展删除/新流确认/下一协议。新增工作流仅workflow_dispatch，执行分支和expected_execution_sha双锁，协议并发锁；重复启动检查本协议账本和Actions运行/产物，pre-start失败与已消费科学预算分开。
若连接器没有dispatch工具，后续执行模型可用一次性push helper调用已冻结正式workflow一次；helper只校验/dispatch不训练，返回不明先查运行，不盲重试。**本次方案发布不创建helper或workflow**。

交付路径：
- docs/PROTOCOL040_RESULTS.md：中文短结论、对照表、所有失败原因、下一步单一建议。
- artifacts/ftmoe_online/protocol_040/runs/run_<id>/：status（science/analysis/publication分离）、source_lock、implementation_manifest、fixture_report、budget_ledger、comparison、per_window、per_expert、lifecycle_events、reuse_decisions、candidate_decisions、cost_profile、artifact_index、compact_manifest。
- 原始artifact含四缓存预测、新臂预测/issued状态、训练/forward/资格日志、完整恢复checkpoint、冻结输入及科学manifest。只有校验过的必要源文件进入040输入bundle；不要把F文件带入分析。保留至少90天，Git记录artifact ID、URL、ZIP digest、大小/有效期和逐文件hash。
- raw先封存并上传，再只读分析；用独立计算重算核心AP/计数核对最终JSON。分析不得修改科学文件；publication状态单独保存，不能因git失败重训。
- 更新main与执行分支的README、AGENTS、NEXT_EXPERIMENT_LATEST、PROJECT_CONTEXT_LATEST及docs/GITHUB_EXPERIMENT_HANDOFF.md，指向同一040结果/run。按最新main创建仅新增结果和入口修改的快进提交，不force push；真正非快进最多3次重建提交，认证失败立即保全publication_pending。
- 固定保留所有033–039历史结果、源码与预算。040一条序列完成即停，交回用户和后续分析模型。

机器登记见 artifacts/ftmoe_online/protocol_040/plan.json 与 plan.sha256。先验证文档hash/plan hash。数字与状态机若有矛盾，在真实040训练之前报告并修订登记；不能执行时默选更有利口径。
