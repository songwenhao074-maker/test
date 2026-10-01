# Protocol-038：固定出生机制，区分偏置初始化与历史权重初始化

状态：仅登记指示，尚未实现或启动。用户授权更新 GitHub；后续收到用户执行交接的模型可按本协议实施，完成后停止。不得自动进入新流确认、结构修改、休眠、唤醒或删除。本文与 plan.json 一起冻结；037 的原始结果、预算和协议保持不变。

## 研究问题与已知边界

B 为 Protocol-036 的 C+D_lin issued 预测；F 为 Protocol-037 从头训练的额外线性专家；D_zero 为037按历史误差出生的额外专家。037在已观察流3601上出生t=351、第一条可影响预测352。F全程/六复现窗相对B的AP增益为0.010222/0.006315，D_zero为0.008160/0.005529；动态出生没有超过固定专家。

F出生前21次更新的采样异常比例约0.88%，D出生后前21次约26.65%。D在早期固定窗口的BCE更低但AP更弱，说明“没有学会”并非充分解释。这是历史数据与初始状态的混杂线索，不是已证实因果机制。线性73→1零初始化可正常学习；本轮不测试随机初始化。

本轮只回答：在同样出生时间和后续训练下，继承历史偏置是否改善D？完整历史权重是否比偏置提供额外收益？固定F仍是强参照，不能只对较弱C报告优势。所有结论是单条已见开发流上的探索性机制证据，不是统计确认或完整动态增删成功。

## 源码、输入与只读参照

计划基于main commit `02fa8cfd4b3efe93ae9f318952c54a9faba78450`。执行分支 `codex/protocol-038-birth-initialization-20261001`。main未包含037运行器；从已完成执行分支的固定commit `7440a3973f97599ea2ad4de45d5bacaaf8b517c8` 读取 `run_ftmoe_protocol037.py`、`analyze_ftmoe_protocol037.py` 及实际依赖，核对其科学内核与原始运行provenance。新增038运行器/分析器，不能覆盖037文件或修改历史结果。禁止直接调用037入口同时重训F与D。

只复用036 seed3601，无新流、无其他流诊断/筛选。036 run36831958978、artifact11147931152，ZIP SHA256 `040bacac8f23659f56599b71d6d6706b62a544e44161c7c42fb5af73650f56c0`；输入逐文件锁继承037 plan并复制至038 plan。B直接取036 D_lin logits，z取C feature tape，不能把C logits当B；B在原运行中持续学习，本轮复用历史输出，无B梯度。

只读对照来自037 run36871854703、artifact11167124178，ZIP SHA256 `498ead6571ef34e56702fd23f8cc57ac30afe92e99a8ffbfbd85a0b92e413c77`。B_ref、F_extra、D_birth预测与F更新日志的逐文件SHA见plan。完整源工件必须可访问且匹配；缺失、过期或关键文件不匹配，登记source_unavailable/source_mismatch并停止，禁止重训基线或重新生成流。

## 顺序与硬预算

Stage A：只读复核与固定诊断。先对B/F/D_zero重新计算037主表，检查标签/分类输出/出生前精确等于B、触发器及批次成熟性。输出诊断文件后继续预登记的Stage B/C，不根据诊断结果更换初始化、阈值或窗口。仅有效性失败可停止。

Stage B：重建一次历史donor前缀，精确执行F从零开始的21次更新t=15,31,...,335；只保存该前缀的历史权重，不继续到t351更新。每一步核对037 F日志的batch、hash_before/hash_after、version。最终state_dict SHA必须为 `95e290ede276a29c8d56097e5ddbfd584e37256a5e49b864fb8c9a6d06b324b6`（沿用037的state_dict哈希算法；不是torch.save文件哈希）。同时保存checkpoint文件SHA、参数、donor优化器/RNG以供精确恢复与审计。历史工件只有该时刻hash，不应假定已有可免费读取的donor权重。

Stage C：两条新增完整科学序列D_bias、D_warm；两组算法及分析器必须在查看任一新结果前一起冻结。可以一个作业交错更新或顺序执行，不能看D_bias结果后修改D_warm。每组预期352次更新，合计704次；共享donor21次，因此本轮新增科学optimizer.step合计上限725次。donor是单独登记的第三个训练槽（短前缀），不是“免费工程测试”。B/F/D_zero不重训。

0新流、0其他真实种子、0学习率/触发器/初始化缩放/随机种子扫描、每臂最多1次出生和1个额外专家。真实数据工程检查最多1次≤256步且0梯度，只检查读取、shape与出生前复制；涉及参数更新和恢复的工程测试全部用合成fixture。不得用真实工程前缀偷跑donor或新臂。科学槽一旦开始只能从已持久化完整状态精确恢复；没有足够checkpoint恢复则交付中断状态，不能从零重启。每次真实optimizer.step计入可恢复账本，重复执行也占预算；预算耗尽立即停止。

## 唯一初始化差异

共同模型、损失、优化器及batch完全继承037：Linear(73,1)，74参数；δ=2*tanh((w·z+b)/2)；输出[B0−δ/2,B1+δ/2]，分类数组逐项复制B；mean BCEWithLogits(issued_B_margin+δ,y>0)+0.001*mean(δ²)。AdamW(lr=1e−4,weight_decay=1e−4,betas=(0.9,0.999),eps=1e−8)，clip_grad_norm=1。构造使用隔离RNG seed3701；之后按组复制值，禁止为得到更好结果扫描随机种子。

| 臂 | 出生时w | 出生时b | 优化器 | 出生后可训练参数 |
|---|---|---|---|---|
| D_zero（缓存037） | 0 | 0 | 出生时新建 | 全部74 |
| D_bias（新增） | 0 | donor b | 出生时新建 | 全部74 |
| D_warm（新增） | donor w | donor b | 出生时新建 | 全部74 |

D_bias不是永久冻结权重的校准器。两新臂均不得继承donor的Adam动量、二阶矩或step计数；出生后第一个优化步必须step=1。donor与两个专家参数不能共享可变存储。两臂仅初始化不同，不能改变loss、batch、梯度剪裁或出生后的训练次数。

不能从B中的D_lin直接复制，避免重复叠加已有修正；不能用t351及之后的F权重。完整复制F权重和优化器会在相同后续batch下重现固定F，不是本轮新臂。只复制权重的D_warm与F的区别还包括优化器重置和出生前输出；D_warm−F不能单独归因于出生机制。主要因果对照为D_warm−D_zero、D_bias−D_zero、D_warm−D_bias，三组出生时均为新优化器。

本轮非零初始化故意允许出生时δ非零：删除038对应的“出生瞬间δ必须为零”断言，换为正确参数复制、优化器空状态与有限/有界δ断言，不能改037断言。记录出生时未更新参数对当前已出现z_t的反事实delta/概率跳变，它只作诊断，不能改写已经发出的t预测。不能为了平滑跳变新增gate、缩放或中心化残差结构。

## 时序和出生规则完全固定

按037全部规则因果重算，不能把t351写成强制出生条件。每t先预测，再结算i=t−2，再在(t+1)%16==0时检查出生，最后正常更新。至少320个成熟interval；m=t−2；recent[m−63,m]64步、previous[m−319,m−64]256步；issued B mean-host BCE用float64稳定softplus。recent至少16正例、16负例；R>=1.25*max(P,1e−6)且R−P>=0.02，连续两次合格才出生，失败清零。出生后停止检查。只用B触发，因此两新臂应在同一t351出生；第一条可影响预测352。源锁一致却不符，记因果复现失败并停止，不能改阈值。

t351的顺序：先精确复制B发出预测351；结算标签349；触发出生；载入仅含t335及之前更新的donor参数并创建新优化器；用原登记t351 batch进行一次正常更新。没有补训练、额外热身步或隐含t351 donor更新。未来输入/标签不能参与donor、初始化和触发。出生前所有数值输出逐项等于B。

后续复用C实际update_batches，保留重复项、原顺序及全部host。预测可用标签i+2<t，更新可用i+2<=t；末端两次settlement只评分不训练。结构和输出组合恒定，没有路由、sleep/wake/delete、第二专家或第二次出生。

## 固定诊断与指标

Stage A及最终结果对所有臂报告：全程AP、六复现窗等权AP、late4、各窗prefix32/64、W六块、全程与复现pooled recall/FPR、逐窗混淆矩阵；采用037同一窗口定义及tie-grouped noninterpolated AP、阈值0.5，单类指标null，缺失主指标不得认定通过。

固定出生后诊断窗口为[352,416)、[352,480)、[352,608)；报告AP/BCE、正负例分开的delta均值/绝对值均值/分位数、|δ|>=1.98比例、纠正/误伤计数、误报与漏报变化。所有窗口都展示，不能挑最好窗口作主结果。参数诊断只使用当时真实保存的checkpoint，列出其cursor和hash；报告b、w·z以及非线性合成后的δ，不把δ误写成两项线性相加。不存在的早期checkpoint标missing，不为此额外训练；不得用终态权重解释早期预测。

只读复核重现F前21更新及D_zero前21更新的实际采样正例比例，重复样本计入，附batch时刻与分母。两比例不是受控因果证据。六个复现窗口来自一条流，不当作六个独立种子做显著性检验。

## 预登记判断，禁止事后改门槛

主要对照D_warm−D_zero；D_bias−D_zero和D_warm−D_bias为预登记机制对照。原始差值均完整报告，不因为标签未通过而隐藏。

定义 init_signal(A,Q)：六窗平均ΔAP>=0.0005、全程ΔAP>=0、六窗至少4个严格正向、late4ΔAP>=0，且A相对Q与B均通过037护栏，全部有效性通过。该0.0005是开发阶段的操作性门槛，不是统计显著性或非劣效界限。

037护栏不变：全程及复现pooled FPR增量<=0.01、recall增量>=−0.01；W六块均值AP增量>=−0.02；每复现窗recall增量>=−0.02，每窗prefix32 recall增量>=−0.03。任何召回/误报违例另标risk，不得用AP掩盖。

分别输出warm_init_signal=init_signal(D_warm,D_zero)、bias_init_signal=init_signal(D_bias,D_zero)、feature_weight_increment=init_signal(D_warm,D_bias)。后二者是路径对照，不是把最终训练后收益精确分解成独立偏置贡献和权重贡献。

保留037 extra_capability规则（相对B full>=0.002、six>=0.002、4/6正向、late4>=0、prefix32均值>=−0.005、护栏通过）。另原样报告每新臂对F的dynamic_increment：extra_capability通过、sixΔ>=0.001、fullΔ>=0及相对F护栏通过。即便通过也只能叫“带历史初始化的出生方案在该开发流上优于F”，不能宣称无需前缀训练的纯动态优势。

报告相对F的出生后[352,5968) AP差异，避免把出生前输出差异误归因于初始化；全程指标仍保留。报告gap closure=(AP_A−AP_Dzero)/(AP_F−AP_Dzero)，分别算full/six，分母非正则null，不裁剪大于1或负值；这是描述量，不是置信界限。

解释：bias与warm均有信号但warm−bias无信号，只能说尚未检测到完整权重的额外收益，不能证明两者等价；warm−bias有信号支持历史特征权重值得继续研究；只有早期窗口改善而主门槛不通过记early_only；均无主信号则保留空结果，优先再查触发时机、目标函数和采样分布，不能自动加结构。所有分支完成即停止；跨流验证和后续路线需新登记与用户交接。

## 成本与证据完整性

分别报告共享donor21步、每臂352步、本轮总计725步；每个暖启动方法单独部署应归属21+352=373步，不能因两臂共享donor就给每臂只算10.5步，也不能仅报出生后352步宣称省训练。缓存基线本轮新增成本为0，但其历史部署成本仍列出。donor本身在出生前也是74参数模型，需要计算其训练时间、历史数据保存或在线维持状态的内存成本。新专家推理可延迟，但不等于无需维护历史模型。

保存逐阶段/每次进程调用wall/CPU时间、实际训练/推理调用和硬件；精确恢复后累加历史账目，不能用最后一次序列化耗时替代总耗时。B原运行成本与038新增成本分栏，跨进程旧时间不可直接合成严格速度提升。未实测同机完整B部署时，标端到端加速未知。

037原始manifest有status.json在生成后被改写的已知不一致：只允许将这一项列为历史manifest异常，保留原ZIP及该文件原始/登记SHA，不能静默修补历史工件；其余文件全部核验，不扩大白名单。038必须分开scientific_status与publication_status；科学文件封存后不可再修改，生成final manifest后逐项验证再上传。发布状态另存，不纳入已冻结科学manifest。紧凑结果每次更新status后重建compact manifest。

donor每次更新、新臂出生前/出生并更新后、每512预测及终态保存完整checkpoint：cursor、输入/源码/plan hash、参数、优化器、RNG、触发buffer/streak、输出、事件、step ledger与耗时累计。实际optimizer.step计数须插桩，不能硬编码True；因果未来扰动须确认发生真实扰动。合成fixture调用真实038入口覆盖两个初始化映射、无参数别名、Adam不继承、出生前/后恢复、下一步等价与terminal零更新。读取B分类输出使用真实存在的class_probability；不得再次因不存在class_logits而在导出阶段崩溃。

## 启动、发布与交付

本次只发布指示、plan及其validator，不创建或启动workflow。执行模型先运行 `python maintenance/validate_protocol038_plan.py`，实现 `run_ftmoe_protocol038.py` / `analyze_ftmoe_protocol038.py` 与fixtures，冻结execution SHA、所有源码/分析文件hash和环境，完成Stage A有效性后再开始donor和两臂。validator只检查登记，不证明科学执行有效。

正式workflow `protocol038-birth-initialization.yml` 只接workflow_dispatch，execution_ref与expected_execution_sha锁定checkout，concurrency禁止同一协议并发科学run。若连接器不能dispatch，允许一次性push helper仅校验SHA/plan、检查无同批次启动记录后POST一次正式dispatch；helper不得训练。POST响应不确定先查run，禁止盲目重试；成功后移除helper。不得因无法dispatch去重跑037。指示和结果提交带[skip ci]，不连锁启动后续实验。

先持久化并上传原始结果，再发布紧凑结果。用Git tree API或配置好认证的干净临时clone保留最新main；禁止force push。只对真实非快进最多3次重建commit；遇到凭据错误如could not read Username，报告authentication_error/publication_pending，不伪装非快进或重跑训练。发布失败不改变science完成状态。

交付至少含：stage_A_diagnostics、source_lock、donor逐步匹配与checkpoint、两新臂预测/实际batch/出生事件/参数初始化审计、所有对照表和护栏、checkpoint恢复/因果审计、budget ledger及事件、分阶段cost、科学/发布状态、原始与紧凑manifest、artifact_index（ID/ZIP SHA/大小/有效期）、中文结果。同步AGENTS.md、NEXT_EXPERIMENT_LATEST.md、PROJECT_CONTEXT_LATEST.md，明确完成/中断后停止，不能让下一个模型继续耗尽新实验预算。
