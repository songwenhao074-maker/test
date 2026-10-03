# Protocol-042 revision 2：固定时间窗口效用与互补专家接入

日期：2026-10-03。状态：instructions_registered_not_implemented_not_started。
本文件与 protocol_042/plan.json revision=2 完整替代未运行的042 revision 1困难样本加权方案。旧版U_uniform/H_hard预算作废，不累计追加；旧版内容保存在提交c20ff2b4e8593100ccd73ce6b5503d14c42961e0。本次仅发布计划，不实现或启动科学运行。执行模型收到用户交接后按本文件实现、预检、冻结、执行四臂并交付；之后停止。

## 1. 问题、已知证据和范围

最终目标：通过结构修改，使包含新增、选择、休眠复用和容量管理思想的D优于持续学习C。不是要求复刻原023结构。

041在同一未来验收窗：暖启动候选BCE=0.501080578368035，零初始化=0.5206202995032794，live=0.5008784185917762。暖启动继承能力但未形成部署收益；拒绝替换不能证明新专家没有互补价值。过去实验常态只有一个活动专用专家，未实现一般的“从多个当前专家中淘汰无贡献者”。

本轮检验：
H1：保留有用旧专家、训练新专家补充当前系统，优于强制让新专家替代旧专家的接入策略。
H2：只使用固定全局时间窗内的实际移除损失，较同一部署epoch内累积评分更能适应故障变化。
H3：利用已经发行的加性贡献即可评分，额外专家推理为零，但统计、存储和控制器成本必须实测。

本轮不混入困难样本加权、ADWIN、EMA门控、相似度路由训练、永久删除或初始化扫描。近期低效用决定休眠；过去价值保存在档案，不得覆盖当前评分。休眠记忆是否应永久删除留待后续；容量满记录阻塞，不按年龄清除。无额外seed/流/F/自动043。

## 2. 唯一实验矩阵与顺序

| 臂 | 接入策略 | 控制评分 | 用途 |
| --- | --- | --- | --- |
| U_parent | 完整沿用041 W_parent | 原041控制器 | 新预算内严格复现，工程门控 |
| R_win128 | 有活动专家时必须替换一个，再验收；无活动则加入 | 最近128个全局成熟区间 | 替换策略对照 |
| A_hist | 有活动空位就新增；满时只允许替换低效用专家 | 自当前部署epoch开始累积，最低128个成熟区间 | 累积评分对照 |
| A_win128 | 与A_hist完全相同 | 最近128个全局成熟区间 | 唯一预指定主臂 |

U通过才依次执行已冻结的R_win128、A_hist、A_win128。某个机制没有触发或科学结果为负不阻止后臂；工程/输入/预算失败则停止。所有臂、分析器和阈值必须在U第一步真实梯度之前共同冻结，不能看早期结果再改后臂。

R/A三臂均零初始化新增残差，相同优化器、机会数、验收、成熟性和驻留上限；不同接入策略导致候选学习参考与实际计算量不同，属于策略包比较，不能声称完全等算力或只改变一个加号。A_hist与A_win128仅控制评分时间范围不同，可比较窗口策略的闭环效果；轨迹分叉后不能声称使用了相同训练历史。
U与三新臂差异不止一个变量，只作旧方法参考，不能把U差异全归因于窗口。64/256窗只对已记录输出做只读敏感性诊断，不产生额外闭环科学臂、不挑选最好窗口当主结果。

## 3. 来源和运行锁

沿用旧042所锁036与041 raw工件、seed3601、5968预测区间、16 hosts、73维issued特征和真实C更新批次；完整ID/hash见plan.source036/source041。旧041内核固定a1b43179fdbe18dc7a71e8a6d5a8537d9e4e522a，所有文件hash沿用；旧文件只读，新增042入口实现修订。
从raw重建必要输入，不从旧检查点或其他臂中段开始。C、B=C+D_lin、D_keep、D_039、D_040及W_041只读缓存，不重训。校验ZIP实际字节与所有允许文件、manifest；缺失/过期则source_unavailable，hash不符则source_mismatch，保全停止而不换来源生成新流。

预测i的标签只在i+2结算；时刻t先发行预测，再结算i=t-2，再控制和训练。未来阶段/故障机制/切换点/事件ID不进入控制器。正常/故障划分只使用成熟二分类标签；独立故障事件ID只在离线分析使用，在线只计含正标签的不同时间区间，不冒充独立事件。

旧041工程缺口必须修复：shadow参数更新与updates/ready/下一操作的事务一致性、真正跨更新和控制事件的磁盘恢复、缺失或失败报告的fail-closed门控。旧raw中的失败报告不篡改，不用旧validity布尔值代替新证据。

## 4. 三新臂的结构、训练与预算公平性

每个专家仍为74参数Linear(73,1)，delta_i=2*tanh((w_i·z+b_i)/2)。
活动集合S的检测margin=m_B+sum(delta_i,i in S)，logits=[B0-sum(delta)/2,B1+sum(delta)/2]。
不重归一旧专家、不对总和再tanh/clip，不改变分类分支；空集合逐元素复制B。最多两个活动专家，因此总修正可达[-4,4]，这是显式结构变化，不能隐瞒为相同函数容量。使用稳定BCEWithLogits，非有限值失败。

每16区间使用同一登记C batch（32 intervals×16 hosts，保持顺序/重复）。训练开始前对各活动专家各forward一次，复用输出：
live loss=mean BCE(m_B+sum(delta),y)+0.001*sum_i mean(delta_i²)。
一次backward计算所有活动专家梯度，分别clip_grad_norm=1，各自AdamW step一次；参数/Adam独立。三个新臂AdamW固定lr=1e-4, weight_decay=1e-4, betas=(0.9,0.999), eps=1e-8。最多两次live专家step；每专家分别记账，不把一次joint backward伪装成一个专家更新。

shadow创建时锁定将保留的活动ID集合K及victim_id（可为空）。训练目标：
mean BCE(m_B+stopgrad(sum(delta_i,i in K))+delta_new,y)+0.001*mean(delta_new²)。
背景贡献复用同t活动更新前的forward，并detach；背景继续按live loss学习，shadow梯度不能回流。无活动背景直接B。每个due16最多一次shadow更新，累计16次后冻结，不能追加步数。接受后保留其自己的Adam并按live joint loss训练。
三新臂新增专家权重/偏置全零、Adam空，不复制父修正再叠加（会重复父贡献）。U保持原父拷贝与新Adam，用于保留暖启动的历史依据；本轮不检验其他新初始化。

三新臂每臂：最多3个驻留专家（含shadow）、2 active、1 shadow；E0直接出生一次，之后最多两次新候选尝试，ID单调0/1/2且不复用。拒绝/取消同样消费尝试；不是同机会反复训练。第二次尝试必须等待上次结束后至少256个成熟区间，并重新经历压力条件，不能失败立即补发。已接受专家绝不永久删除。

## 5. 固定窗口评分：精确口径

预测时保存m_live、每个活动ID的delta_i、m_B、deployment_epoch与专家参数版本。标签成熟后用这些当时发行的数值计算：
u_i=stable_BCE(m_live-delta_i,y)-stable_BCE(m_live,y)。
这是保持其他贡献不变的真实单次消融损失；不使用当前权重重算过去样本，不使用训练后的拟合误差。评分反映近期实际运行轨迹，不等同于当前参数在历史集上的重评估，更不是永久删除后重新学习的因果效果。

窗口按全局issued时间索引定义：t结算后m=t-2，最近W窗为[m-W+1,m]，所有行必须已成熟；固定延迟下等价最近W个成熟区间。不是每专家最近W次调用，也不是W个host rows。过期观测按时间移除。
一个deployment_epoch由活动ID集合定义；新生/接受/休眠/复用改变集合即进入新epoch。模型正常梯度更新不切epoch。控制评分只使用当前epoch，拓扑变化清空控制累计量/连续计数，旧值转入记忆档案。新epoch不足128成熟区间不得休眠或选择替换victim。
A_hist累计当前epoch全部成熟观测；A_win128/R_win128仅用最近128。A_hist不是专家终生混合评分，须准确命名。

每个专家计算overall S、positive S_pos、negative S_neg、移除前后FPR/recall、正负host数、含正标签区间数。有效评分要求：
- 当前epoch>=128个成熟区间；win臂128区间全部属当前epoch；
- 正、负host rows各>=16；
- 含至少一个正标签的不同区间>=4（不是4个独立事件）。
分数与差异用float64从issued float32 margin/delta重算；阈值比较<=为含等号，不设事后容差。不存在/休眠/支持不足返回null及reason，不写0。
稀有故障保护：删除总体收益不能掩盖故障类贡献；不将同一故障的多host/相邻行当独立统计样本。

“近期可退出”需在连续3个due16检查中均满足有效支持、S<=0、S_pos<=0、S_neg<=0、移除FPR-live FPR<=0.01且移除recall-live recall>=-0.02。任一失败/未知清零streak。重叠窗不是3次独立显著性检验，不输出虚假置信度。
有多个合格者时选择最小S，精确同分选较小ID；仅低排名但S>0不授权A臂替换。R作为强制替换对照：有活动专家则在有效支持者中取最小S，不要求上述无用条件；全部未知则不提议。所有专家仍有用时A必须允许“无人可淘汰”。

同次控制最多休眠一个专家，保存同ID权重/Adam/step及历史效用摘要。历史效用只用于报告记忆价值，不作为当前活动分数加分。当前未观测不等于无用。

评分仅用已存在输出，额外专家forward=0。控制使用128环形缓冲及累计和；hist只维护当前epoch累计状态，不无限存全部原始样本于在线内存。审计文件可流式写盘。诊断64/256窗不参与控制。
REAP式mean(abs(delta_i))可作为无标签描述性代理，不能替代损失贡献或称为已验证在线REAP。

## 6. 新生、接入、未来验收、休眠和复用

### 6.1 新生压力与顺序

E0沿用原041首次出生规则和零初始化，预期t351仅作核对，禁止按时刻强制创建。
后续压力沿用原窗口：实际本臂live已发行BCE的最近64均值R与之前256均值P；R>=1.25*max(P,1e-6)且R-P>=0.02；近期正负host各>=16，连续两次due16成立。
近期64须同epoch；有活动当前epoch成熟数>=256，无活动>=64；旧256窗可跨epoch。失败或拓扑变化清零压力streak。须有驻留容量、无进行中的shadow/复用、无同t转换、尝试预算和冷却满足。
有休眠记忆时，须先在同epoch获得最近64区间内、同休眠ID/参数hash集合的有充分支持的复用质量拒绝；无支持不能授权新生。压力只代表尝试候选的理由，不证明旧专家已无用。

事件顺序：发行live/现有资格preview→结算→due16完成至多一次拓扑转换（first_birth > reuse_accept > shadow_accept > sleep）→无转换时优先启动到期复用，否则检查新生→活动训练→未取消shadow训练。同t建立的资格槽从t+1取样。转换从t+1预测生效，更新按转换后集合进行。

### 6.2 接入计划

A臂：若|S|<2，则K=S、victim=null（保留全部旧能力）；若|S|=2，只能选第5节连续3次合格的最低分victim，K=S\{victim}；没有合格者记录capacity_active_all_useful_or_unknown并跳过，不先删除旧专家。
R臂：S非空时按第5节选victim并令K=S\{victim}；S为空则K为空。R从一个E0出发通常只有一个活动专家，这正是强制替换策略的后果，不能据此声称验证了多活动排名。
接入计划在创建/复用槽开始时锁定。训练和验收期间活动拓扑变化则取消槽，消费已发生的尝试/slot预算；梯度变化不取消。验收通过前不改变旧专家活动状态。

### 6.3 候选未来验收

新候选完成16次更新后冻结，从下一时刻开始固定32个新issued区间；复用专家冻结权重，以相同32区间preview，不训练。候选系统margin=当时实际live margin-当时victim delta（无victim则0）+candidate delta；背景直接复用同次live输出。
32条标签全成熟后的首due16决策，不能延窗、挑窗、重训/重复验收。不足流尾记censored，终末结算只记账不接受/训练。

相同32窗比较整个候选系统与实际live、B：
candidate BCE<=0.99*live BCE且<=B BCE；相对live和B的FPR各不增加超过0.01、recall各不降低超过0.02；正负host各>=16，全部finite，threshold=0.5，正则不参与验收。
若涉及A臂victim，决定时还须同epoch、同victim，且当前仍满足第5节连续3次可退出条件，否则reject_victim_no_longer_eligible，不能临时换victim。R仅要求victim仍活动且评分有支持。
通过后原victim休眠（若有），新/复用专家活动；其他活动专家保持；未通过新shadow释放，已接受的复用记忆仍保留。接入通过并不等于新专家优于所有旧专家，其意义是候选组合优于当前组合。

### 6.4 睡眠和复用

没有更高优先级接受时，每due16按第5节可退出规则最多休眠一个；任何拓扑转换取消过时pending槽并重置评分epoch。接受/休眠取消同t冲突shadow的规则需记录，不允许隐式重试。

每due64、当前epoch>=128成熟、存在休眠且无pending复用时，可启动一个复用槽，最多32槽/臂。依第6.2节锁定统一接入计划，针对所有休眠ID（最多3）各作一次专家forward/区间，最多32未来区间；参数冻结，live继续训练。即使无新生压力也允许复用。有多个通过者选候选系统BCE最低，同分较小ID。
复用可与shadow训练/验收并行；先通过复用会取消shadow。新生创建本身不与pending复用并行。有槽无有效接入计划则记录skip，不消费slot。
窗口不能免费评价休眠专家；这部分preview计算必须单列，不能宣称整个控制器零额外推理。

## 7. 固定预算与启动门控

仅四条完整新科学序列，每条5968×16。同一真实源，0新流/额外seed/F/donor训练/额外臂/永久删除/从零重跑。
U沿用旧042上限：352 live+16 shadow=368 optimizer steps；部署5616、reuse2048、shadow资格32专家区间forward。
每个R/A上限：704 live专家steps+32 shadow=736；每due16至多2 live+1 shadow；部署11232、reuse3072、shadow资格64专家区间forward。训练forward单列，不能混成部署调用；预算计专家调用，即使向量化也不合并抹去。
全协议梯度上限2576=368+3×736；部署上限39312，reuse11264，shadow资格224，三类合计50800。这是旧revision1的替代总预算，不再追加其736步。实际调用少则如实减少，不为“公平用满”训练。
实际内存须计参数、各自Adam、活动图、资格缓存、窗口统计、临时序列化副本；B历史成本与新成本分开，不宣称端到端提速或等算力击败C。

真实工程prefix最多一次<=256区间、零梯度、E0前；所有更新/生命周期预检用合成流。科学开始前固定runtime Python3.8.18/Torch2.4.1+cpu/NumPy1.24.4/单CPU线程、execution SHA、源码hash、schema、fixture与注册hash。采用revision2专属分支codex/protocol-042-windowed-utility-20261003与workflow protocol042-windowed-utility.yml，仅workflow_dispatch；本次不创建workflow。
跨run预算key=(protocol042, revision2, arm)。先持久化started再首次梯度；恢复只允许精确继续，不能重置槽。事务包含所有参数/Adam/RNG、shadow计数/ready、next_operation、环形队列/累计和/正负计数/epoch/streak、接入K/victim、预算/action ID、资格内容及终末结算进度。

U必须与W_041的预测、资格、实际batch/参数/Adam/离散事件一致，沿用旧schema和1e-12指标容差，仅预登记审计字段可排除。U失败/缺失/空/字符串true均阻止R/A；workflow及每个R/A进程入口独立验证布尔报告、非空fixture实证、输入和execution hash及预算。不接受缺失返回值的OR条件。
U成功后，R/A间不以质量好坏门控；禁止根据负结果临时换窗。任何工程失败保全停止；科学负结果正常交付。

## 8. 必须先通过的生产入口合成测试

测试必须真的运行更新/事件、从磁盘销毁重建并续跑；不能只写断点名称或all_pass：
1. 零残差精确复制背景、delta相加正确；单专家退化为原检测公式；去掉i只减其delta，其他输出不重归一；正/负贡献金标准。
2. 固定窗[ m-127,m ]边界、旧数据恰好到期、hist累计与win首次128完全一致；故障A→B切换中旧正贡献最终退出win而hist保留；不足样本返回null；不把最后128次调用当128全局区间；休眠不补零。
3. 标签延迟、未来扰动不改变前缀；历史issued与当前参数分离；正/负保护以及同事件多host不伪造独立证据；三个重叠检查非独立。
4. 两个有用专家时A无victim；一有用一有害时只选后者；分数同值ID规则；epoch/成熟期重置；最低正分不能被A淘汰；A空位加入但R替换。
5. live多专家同时从更新前图取梯度、各Adam/clip独立；shadow只收到自身梯度、detach背景；新生零初始化不重复父函数；shadow只16次、冻结后不训练；复用同ID Adam恢复。
6. 验收32未来区间不泄露；拟替换者后来恢复有用则A拒绝替换；资格不够不扩窗；reuse/shadow竞争、拓扑取消、容量满、冷却、两次机会耗尽、尾部censored。
7. 磁盘恢复覆盖出生/多专家live更新中间/shadow第1/15/16次/ready未决策/接受/拒绝/双活动/休眠/复用/窗口元素出队/终末两次settle之间。记录真实事件、action、前后实step、checkpoint hash、恢复后更新与控制（终末除外）；对齐连续运行输出、参数、Adam、所有RNG、窗口、事件和账本。
8. 崩溃注入：梯度已step但外部计数未提交、原子保存前后、ready及拓扑事务。恢复一致或ambiguous_step停止，禁止不明状态重放；插桩核验实际optimizer调用。
9. 门控正负例：缺失/假报告、错误源hash、错误revision、超预算、U非零都阻止新臂；比较器必须能抓到修改过的参数/batch/决策。
10. 独立指标和评分复算：只从封存issued margin/delta/labels重算每个控制点、窗口成员、支持/streak/victim，与在线累计一致；复算不调用专家模型。

合成fixture不得用真实science标签调阈值。合成不能覆盖的关键事件属于engineering_incomplete，不能用真实新增运行“顺便测试”。评分代码的有限算术容差事先固定1e-10，阈值两侧fixture不得靠容差改变判定。

## 9. 预登记结果与结论

主结果固定A_win128，不能从三新臂挑最好者替换。所有臂报告相对C/B/D_keep/W_041及两项机制对照：
- A_win128-R_win128：接入策略包效果；实际计算量一起报告。
- A_win128-A_hist：控制评分窗口效果；在首次模型轨迹分叉前须预测、参数、Adam及离散控制一致（登记的评分数值/缓存结构允许不同），分叉必须对应窗口差异引发的合法控制事件，不能预设第一处差异时刻。
- 窗口时间范围不可能影响仍同样的前128个epoch观测；如果不发生不同选择，记window_mechanism_not_exercised，即使主臂优于C也不能宣称窗口有效。
两机制开发信号均要求full AP差>=0、six均值AP差>=0.0005、至少4/6窗口为正、late4>=0且沿用相对该对照护栏。阈值只是操作性开发标准，不是显著性。

D>C及累计保持完全保留旧041/042标准：full/six AP各>=+0.002对C、至少4/6为正、late4>=0、prefix32>=-0.005；对D_keep full/six/late4>=-0.001、prefix32>=-0.005；所有原护栏。具体窗口/护栏数值见plan.analysis（阶段标签仅事后分析）。
分别报告validity、D_over_C、cumulative_preservation、addition_policy_signal、window_policy_signal、pool_exercised、multi_active_exercised、victim_selection_exercised。不得用总分掩盖失败项。
pool_exercised沿用至少两个接受ID各部署>=64且某同ID休眠>=64后再连续部署>=64；multi_active要求两个活动ID共同连续预测>=64；victim_selection要求在两个活动ID中至少实际按有效评分选择并休眠一个，或完成有支持的替换转换；同时分别报告sleep_selection和replacement_selection，不混称。A可能先休眠腾出空位再新增，不强求满位原地替换。未触发如实报告，不加跑补事件。
主开发目标=validity AND A_win128 D_over_C AND cumulative_preservation AND pool_exercised；窗口/接入机制另判，不自动等同总体成功。

诊断必须包括：
- 每专家全程S/S_pos/S_neg、支持、上次观测时间、状态、休眠/复用原因及每次victim候选表。
- 原始128评分及只读64/256、hist、mean(abs(delta))排名差异；这些诊断不代表其他窗口闭环表现，不能事后选参数。
- 真实阶段切换后的第一次合格证据、停止使用/恢复延迟、仍有正贡献时退出比例、故障回归表现；没有观测支持则null/censored。事后“稳定段误休眠”用接下来32已记录行描述，不假装知道休眠专家的未观测反事实。
- 接受时同一未来窗口的live、proposed、B及各自BCE/AP/FPR/recall；新增是否在保留旧专家后提供增量；拒绝原因按每条护栏列出。
- 全程/六回归窗/late4/prefix32、64以及每事件前64后32/64/128，标注跨epoch、其他转换和支持。BCE改善不替代AP和误报召回指标。
- 评分更新CPU时间、controller总时间、训练/preview/部署专家forward、内存和I/O分列；零额外评分forward不等于整体零开销。可对封存标量评分重算做计时，不得为计时重跑科学训练。
- 数据仍是已见seed3601；不能宣称独立确认、永久删除有效、复用因果优势或整体部署加速。

## 10. 文献依据与本轮改造边界

- Han等，ICML2024，Model Assessment and Selection under Temporal Distribution Shift：https://proceedings.mlr.press/v235/han24b.html 。自适应滚动窗口评估/比较模型；支持关注时间分布变化，不直接证明本轮固定128或在线训练残差评分的最优性。
- Bifet与Gavalda，SDM2007，ADWIN：https://epubs.siam.org/doi/10.1137/1.9781611972771.42 。按变化调整窗口；本轮仅列未来方法，不实现或额外试跑。
- Kolter与Maloof，JMLR2007，Dynamic Weighted Majority：https://www.jmlr.org/beta/papers/v8/kolter07a.html 。基于在线表现增删专家；其独立分类器评分不能原样替代本轮残差贡献。
- Cerebras等，ICLR2026，REAP：https://github.com/CerebrasResearch/reap ，https://arxiv.org/html/2510.13999v2#S4 。使用路由加权激活、按调用条件平均，原为离线剪枝；不直接处理本轮时间漂移。这里只读记录其幅度类代理。
本轮加性移除损失、128窗与退出规则是本项目预登记设计，不冒称复现论文或享有其理论保证。

## 11. 交付和停止

先封存raw manifest并上传artifact（>=90天），后只读分析。输出docs/PROTOCOL042_RESULTS.md及artifacts/ftmoe_online/protocol_042/runs/run_<id>/，都标revision=2。包括源码bundle、输入/注册/执行hash、四臂issued贡献/预测/资格、评分窗口和候选表、所有更新/生命周期/机会日志、fixture及断点实证、预算与成本、独立复算、artifact_index与科学/分析/发布三种status。
精确恢复不改源码/参数/窗口；发布/分析失败只处理原产物，不重训。结果commit用[skip ci]，非force推送，同步main/执行分支指针。四条序列完成或阻塞即停止；不增加候选次数、窗口扫描臂、额外seed、困难加权、永久删除、F或043。
