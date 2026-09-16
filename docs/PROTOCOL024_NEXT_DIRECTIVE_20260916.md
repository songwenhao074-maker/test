# Protocol-024 v2c 与后续情形指示（2026-09-16）

## 目标和本轮范围

目标不变：在现实合理、可以公开解释的特定在线情形下，让动态增删专家 D 相对明确对照体现优势。允许 D 使用额外后台训练和有限记忆存储，也允许固定方法受到现实更新/计算预算约束；须记录差异，不把失败改写成成功。

先读 [v2结果审阅](PROTOCOL024_REVIEW_20260916.md)。本次下一执行任务是 **v2c：修复birth替换语义，完成一次有限诊断重放**。不继续扫描p99/验收阈值，不重跑已有模拟数据。v2c若仍无明显收益，直接准备下面的Protocol-025新情形，不再把当前三law反复复现作为唯一研究方向。

只用replay seed700/model seed1。确认701–703、测试201–205继续封存。输出 `artifacts/ftmoe_online/protocol_024/next_round_v2c/`；不同尝试独立run_id，拒绝覆盖。旧v1/v2的negative development result保留。

## 1. 固定输入并取回实际事件

代码基点为 `c2ada4446199c0c9a660ac096d2a92cdb224262e` 的v2实现，本次审阅提交只增加报告、指示、证据与复现脚本。

- stream：4980 scored+1 guard，SHA256 `468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`；finalizer run34953715809。
- v2a结果：run34989082187，artifact10405894781。
- v2b结果：run35051303612，artifact10429441883；原ZIP SHA256 `525409e74634b4e3a361ced00400b292b395a3ca9914e228caded1c7852b07a2`。
- 将comparison、各arm summary、lifecycle、candidate pairs、checkpoint恢复审计和数值配置保存为可长期读取的轻量文件；大文件附哈希及取回入口。不要只复制17行结果说明。
- 固定raw_next_fault、t+2标签成熟、u4、lr1e-4、batch32、replay64、anchor .25、distill .10及v2 train/guard划分；保留所有标签和response-law参数。
- shadow预算使用已运行的buffer128_4x4，不新增lr、训练步数或acceptance网格。

在读取候选收益用于进一步设计前，提交v2c registration。说明它根据v2开发结果发现的实现问题修订，不是独立确认。

## 2. 修复真实部署语义，不只修一个函数

引入统一的“目标部署模型”forward，供以下三处共同使用：

1. shadow训练的loss；
2. prospective候选验证的完整模型输出；
3. 独立old-knowledge guard输出。

v2c birth的最终拓扑为四个generalist加新specialist，旧specialist不参与最终候选模型。live对照仍为当前四个generalist加旧specialist。三处不能继续直接调用“全部active+shadow”的旧preview。

保留现有softmax/log-ramp归一化语义、8个预测interval的crossfade以及各验收阈值。无需为了制造收益改成更宽松验收。训练时只优化shadow；候选在prospective验证期间冻结，live通用专家继续按共同规则学习。预览必须使用当时同一组generalist参数，不能拿验收结束时的参数回算过去预测。

必须验证：

- 固定所有张量、不做在线更新时，candidate preview与“接受、crossfade结束、旧专家retire”后的logits/概率一致，atol/rtol=2e-6。
- 无旧specialist、存在旧specialist、复用休眠专家三种路径均覆盖；使用非零、不同专家输出及极端router logits。
- 验证guard与训练使用同一目标拓扑；不能只修candidate验证。
- 原审阅脚本中的反例确实展示旧路径不一致，新路径应消除该差异。
- 冻结specialist的参数、router、optimizer archive在普通live update和shadow训练前后保持一致；只检查requires_grad标志不够。

若采用其他明确的“保留旧专家共同部署”结构，必须另开算法版本；不能称为同一replacement修复。

## 3. 补足时间线、计数和恢复

每到达一个理论reuse/birth时点，先记账，再根据忙碌/无记忆/匹配不足/容量拒绝等原因决定是否执行。所有状态下均记账；不补发原策略已经错过的机会，本次只修审计，不暗改调度。

输出互斥且守恒的计数：due=started+skipped_busy+skipped_no_memory+skipped_no_match+skipped_capacity（没有意义的类别记0）；另列not_yet_eligible。验证末尾created=accepted+rejected+pending，pending标为stream_end_censored，不虚构未来验证结果。

逐候选输出created_cursor、training_indices、validation_indices、每条prediction_cursor/label_published_cursor、accepted_cursor、first_influence_cursor、full_ramp_cursor、路由质量和完整验收原因。birth与reuse分别记；候选丢弃不计已部署专家purge。

恢复验证从真实birth/retirement/reactivation路径checkpoint进行。与不中断参考分支比较：
- 下一预测；
- 下一完整live update后的模型张量、Adam状态和RNG；
- **该更新后的下一预测**；
- 下一非空生命周期事件的全部语义字段，以及active/dormant IDs、ramp、候选状态。
耗时、输出路径等非语义字段可排除。没有出现事件时记录未覆盖，不能因为events_advanced有数就标PASS。

不要为归档方便把四个源码fragment和拼接文件维护成两个独立实现。允许继续loader，但记录拼接顺序及assembled源码SHA，并让测试导入生产同一路径。

## 4. 跑一次纠正后的v2c，并回答收益在哪里

从t=0运行C-u4和D-v2c。只有配置、anchor split、代码路径和输入均确认一致，才可复用v2a的C预测；保留可复核指纹。D必须独立重放。

沿用九个first100窗口和六次复现等权主指标，不换窗口、不改+0.03参考线。保留full AP、onset AP、resource F1、FPR/Recall与可部署pressure基线。v2b旧曲线保留为带缺陷版本，不覆盖。

限定的因果诊断：

- 对每个真实接受birth，比较“旧增量preview”“纠正replacement preview”“实际部署后模型”在相同预测时刻的输出；旧preview只离线审计，不参与新决策。
- 专家ID4/5的训练、验证与birth区间对应哪个law，只在分析中读取audit law。核实是否均来自R1_first；不能给它们事后随意命名为R1/R2/R3专家。
- 按每个复现窗口，记录已有记忆、提议等待、匹配拒绝、验证等待、crossfade及实际有效预测比例，区分“没有合适记忆”“有记忆没找到”“找到太晚”“部署仍无效”。
- 对已保存记忆做一次只读反事实评分：使用每时刻当时存在的专家与当时generalist，考察被匹配拒绝的记忆是否原本有利。未来标签只能用于事后诊断；绝不用于生成在线选择器或把oracle曲线当D结果。
- 不以1%混合loss验收自动推出AP改善；单独报告detection CE、resource CE、ranking、正例数、AP变化与效应持续时间。

只允许一个附加诊断arm：**D-v2c-no-reuse**，禁用reuse，其他出生、训练、冻结及容量规则保持相同，仍从t=0运行。记录关闭reuse造成的后续事件变化；不得伪称完全相同轨迹的单步消融。这项诊断不再以“先AP赢3个百分点”为前提，因为目前正需要定位弱收益。

达到旧开发信号后，补跑C-u1/总预算接近的C、C-fixed8和适用消融；未达到时保留全部结果，不读取确认种子，也不再加一轮相似阈值微调。

## 5. 若当前流收益仍小：转向Protocol-025现实有利情形

以下是新开发任务，不能回写为v2预注册结果。v2c交付后若仍缺乏实际幅度的收益，下一模型应实施该新情形，而不是重复“必须证明C永远学不会”才准做D。

**现实设定：多租户边缘设备的服务进入、退出和复现；活动计算有限，少量旧专家可常驻，全部历史服务无法同时驻留。** 服务启动时可后台训练候选，之后短时复现时希望快速复用。这个设置自然有利于D，且不会依赖未来标签。

首版只注册一个场景：

- 6种物理响应服务，累计种类超过D可保留的专用记忆槽；同时只存在少量主要服务。保留CPU短脉冲、RAM增长、IO写回三类，再增加三种有明确物理解释、可从短历史辨别的响应（例如缓存预热后衰减、周期性工作集释放、写回与CPU竞争）。具体形状/峰值/滞后在生成前写入配置。
- 固定时间线：F0=300；S1、S2、S3首次各600；S1/S3/S2复现各180；S4、S5、S6首次各600，伴随旧服务退出；S4/S2/S6/S1/S5/S3复现各180。共5520 scored+1 guard。旧服务退出并不意味着在线系统预知它永远不再出现，purge只能用历史usage/acceptance信息。
- 当前负载相近时，未来风险应受可观察的增长、积压、衰减历史影响。标签仍由物理容量超限生成，不人工按service ID换标签；不能仅把同一曲线换6个名字。
- phase/service-law ID、未来切换时间只用于生成与审计。若确有部署时可见的任务元数据，必须给C/D相同输入并明确可得时点，不能把生成器隐藏law ID当现实元数据。
- 第一次接触较长、复现较短是有意偏向记忆复用的现实条件，必须在论文/报告明确说明，不声称任意流都占优。

**容量与对照：** 首版保持4个可学习generalist，D最多1个live specialist，resident上限8，shadow同样占容量。purge只删除dormant specialist。活动专家和resident专家两种预算分开报告。新场景中至少比较固定4、固定5；固定8-dense作为较高推理成本参照；若主张活动预算优势，再加入固定8、top5路由的合法竞争者，不能只把dense8排除后宣称全面胜利。

D可使用更多后台训练，但给出累计优化次数、训练interval重复次数、实际CPU时间、峰值内存、resident字节、p95推理延迟。旧“0.5GiB可用RAM紧急下限”是运行环境保护，不等于模型resident预算，不可混淆。

**先补齐共同预测基线，再解释动态收益：** 新目标下旧冻结teacher不是强预测基线。保留同信息pressure分数；为C/D共同提供同一因果可见的归一化压力/变化特征，若增加共同pressure prior或轻量共享预测头，在运行前固定其结构和训练规则，两者完全相同。只使用开发历史前缀确定该共同部分，隔离后续评估，不凭最终D−C选共同基线。不要让D独享一个强特征后把收益全归因于增删。

数据先进行一次无模型结果的数据审计，检查采集因果顺序、每阶段正负样本、事件长度和参数物理含义；允许最多一次事先登记的数据修订。无需额外证明梯度冲突或固定C不可能拟合。生成采用可恢复检查点和不可变分块，避免环境失败后重复几小时工作；严禁事后拼接旧npz而忽略跨边界历史、调度和物理状态。

Protocol-025先声明主张类型：
- 若主张预测优势：主指标为全部注册复现first100的等权AP差，同步报告FPR/Recall与强基线。
- 若主张受资源限制的优势：提前固定硬资源上限与精度容忍差（建议AP非劣容忍0.01作为待注册设计值），比较满足约束的完整方法集合；不得看到AP不赢才临时改成成本主张。
- 硬删除的收益必须有真实容量阻塞、实际purge及no-purge对照；没有这些事件，就只报告增长/复用，不能声称证明了删除必要性。

本指示推荐先以**同一活动预算下的复现AP**作为Protocol-025主张，成本作为完整副指标。结果仍不保证D胜出，必须保留全部注册服务、窗口、种子和失败配置。

## 6. 交付与停止点

本次首个交付必须是v2c，包含：

- registration.json、reviewed_source_manifest.json、preview_deployment_equivalence.json；
- 完整predictions、candidate_pairs、lifecycle、summary和checkpoint恢复审计；
- opportunity_accounting.json、memory_training_intervals.json、recurrence_latency.json；
- comparison.json、cost_profile.json、status.json；
- docs/PROTOCOL024_V2C_RESULTS.md，更新两个LATEST入口。

status包含verified_fix、candidate_created/accepted/rejected/pending、birth/retirement/reactivation/purge、due/skipped分类、development_signal、confirmation_run=false。如果只完成修复验证，不能写“性能实验完成”。

v2c若未达开发参考，完成上述有限诊断后，提交Protocol-025数据/方法注册并推进新情形。新情形不把0.03当作统计显著性的替代品。最后只有在完整配置、比较对象、主指标冻结后，才单独启动701–703确认；本轮不自动启动。
