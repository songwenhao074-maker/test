# Protocol-024 next_round_v2 执行指示（2026-09-15）

## 任务与边界

在合理有利于动态专家的现实情形中，取得 D 相对明确对照的可复现优势。部署故事为：相同设备反复承接不同服务，首次服务持续较长、复现较短；在线更新机会有限，后台可训练候选，内存允许保存有限专家。D 可获得额外后台计算和专家存储，但必须记账，并设置对应容量/预算对照。

本指示替代“继续反复调 p99”与旧的普遍 D 禁入门禁。先读 [本次结果审阅](PROTOCOL024_REVIEW_20260915.md)。next_round_v1 是完成且为阴性的开发轮，不能覆盖。新版本已经利用 seed700 的旧结果进行设计，属于明确的开发集复用，不是新的独立验证。

只使用 replay seed700 / model seed1。701–703 和 201–205 不读取、不生成。不要运行三小时模拟器重建已经存在的数据；先取得并核验原始 stream。

## 0. 取回证据，冻结本轮设计

工作代码以 `protocol-024-next-round-gpt56@b2410d74bccb44889521549da7c3b7c7b47c0a9e` 的实现为基础；本次审阅提交只加入文档、证据摘录和入口。新输出根为 `artifacts/ftmoe_online/protocol_024/next_round_v2/`，每次尝试独立 run_id，拒绝覆盖已完成目录。

- 正式结果 run 34953810201，artifact 10391150168。
- 已收尾数据 run 34953715809，artifact 10389922674。
- stream SHA256 必须是 `468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`。
- 数据 4981×16，4980 scored + 1 guard；原标签、容量、phase/law 审计数组、时间线及 response-law 参数全部不改。
- 原学习率 1e-4、共同在线 u4、batch32、replay64、1 step、anchor .25、distill .10、raw_next_fault 及 t+2 成熟规则不改。不同策略的额外 shadow 优化单独计量。
- 运行前提交 registration.json，列明下面 v2a/v2b 的顺序、配置、晋级规则及最多两次候选训练预算版本。任何根据结果作出的后续修改新增版本，不能假称预先未见结果。

将 v1 comparison/status/lifecycle_config 数值版、lifecycle.jsonl、各 arm summary、data manifest/audit、probe 和 budget JSON 归档到 Git 的轻量证据目录。大 NPZ/checkpoint 可保存为明确命名的长期附件或独立数据资产，并附 SHA256 和取回脚本；不得只留会过期的 Actions ID。本次审阅目录的 comparison 是 stdout 提取，不替代原 NPZ。

## 1. 先修数据可得时点与诊断输出

这些工作与 D 机制开发并行，不作为无限延迟 D 的新门禁。

1. 为 host_features[t]、demands[t]、schedules[t]、raw[t]、ratio[t]、raw[t+1] 建立 availability 表。预测发生在当步 simulationStep 之前，训练目标不早于 t+2。
2. 保留旧 `post_step_pressure_reference` / `post_step_fault_reference`。新增可部署基线：从当时可见 host_features、容量或可见 demands+已决定 schedules 构造压力。须按真实字段核验 CPU/RAM/Disk 的维度与单位，不能假设前 3 列就是三类资源。延迟标签 persistence 使用实际已发布标签；若最晚为 raw[t-1]，明确记录该偏移。
3. 可得性断言不仅检查字段名称，还检查读取的时间索引。可通过审计 wrapper 捕获禁止索引；共享冻结主干、C/D 特征必须经过同样检查。
4. 输出每 8 个成熟 interval 的最近 32 步 detection CE、resource CE、ranking、正例数、正常概率、可见特征变化。记录 trigger_reason、候选训练开始/结束、每条验证预测时间和标签发布时间。
5. 修正零事件说明：分别输出 trigger_not_reached、candidate_created_but_rejected、capacity_blocked、accepted、retired、reactivated、purged；计数为 0 时不能写“已训练但未通过”。
6. AP 按同一窗口、同一目标、同一有效行集合比较。全正常窗口按既定 null 规则保留并报告覆盖率，不能为改善结果丢弃窗口。

## 2. v2a：先检验“可预算的主动候选”是否有效

不要先继续搜索更低阈值。部署存在固定后台维护机会，因此**正式注册周期性候选提议**，使动态方法不必等到持续高损失才尝试改进。这是可部署的计算预算策略，必须与 debug 强制事件分开。

- 前 600 个预测成熟前不提议；此后每积累 256 个新成熟 interval 获得一次提议机会，按 `matured_count=600+256k`（k≥0）定义。时间表不读取 phase/law/switch；所有流和确认种子使用同一公式。
- 最多一个 shadow；正在训练/验证或容量不足时跳过该次机会并记录原因，不补发连环候选。不得在事后发现有利的 switch 时刻另加提议。
- v1 的 p99 监控只作诊断，v2a 不用其结果增加提议。候选创建与接受明确分开：周期只决定“尝试”，不强制“成功”。
- 首版沿用当前父专家规则、shadow lr=1e-4、64 个不同成熟 interval 各一次训练、随后 32 个全新 interval 的预先预测验证；保留 1% 相对 supervised loss 改善要求与旧平均正常概率保护。
- 补充验证窗口固定阈值 0.5 的 FPR delta≤0.01，以及独立旧知识保护。旧知识 guard 来自共同、已经观察到的历史 anchor，预先按时间规则划出禁止任何训练的子集；候选不能用该子集优化。guard supervised loss 相对增加≤2%，正常 FPR delta≤0.01；无所需类别时输出 unavailable，并按注册保守规则暂不部署。
- 测试接受后的 ramp 对真实非零残差输出连续；报告从 created 到 accepted、首次影响预测、ramp=1 的延迟。v2a 暂用原 10 次 live update ramp，不能隐藏其速度代价。
- 跑 A、C-u4、D-v2a；同时输出修正的可部署压力基线。可复用 A/C 旧预测的前提是所有输入/目标/更新/anchor 均字节及配置一致；若 guard 划分改变训练 anchor，则必须重跑全部对应 arm。

候选诊断必须包含 shadow 与完整 live 的成对损失、normal FPR、guard 结果、当前候选路由质量和参数变化。若创建>0但全部拒绝，不能写“D无用”，先确定训练是否有效、路由是否让候选产生影响、是否无正例、或保护条件冲突。

**限定的第二预算版本：** 只有首版候选全部未接受时，允许新 run_id 使用最近 64 个不同成熟 interval 的 buffer，minibatch32、每到 4 个成熟 interval 做 4 次优化，共 128 个训练 interval，然后用后续 32 个 interval 验证。其他配置与父专家规则保持不变。该设置预先声明为额外后台计算版本，必须报告实际次数/样本重用/耗时；不声称仍是“只训练64步”的同预算结果。两版均保留，不继续同时扫 lr、验收阈值和场景。

交付要求为至少一次完整候选训练和因果验证，以及真实接受/拒绝原因；接受与 D 胜出不能人为保证。若所有候选仍拒绝，提交上述诊断与代码路径证据，结束本轮指定搜索，不消耗确认种子。

## 3. v2b：有可用候选后，加入保留、退役与快速复用

只增加参数还不足以解释动态记忆优势。v2a 有至少一个通过因果验收的专家后执行 v2b，使用同一个 stream 从 t=0 独立重放；不能从看到未来后挑选的 checkpoint 开始。

建议实施一个清晰、有限的结构，而不是无边界扩容：

- 四个初始通用专家继续按共同预算学习；额外专家形成有限 specialist bank。最多一个 specialist 在 live 中发挥作用，旧 specialist 退役到冻结存储。总 resident 上限仍为 8，包含四个通用专家、所有 active/dormant specialist 与 shadow。
- 候选独立学习完成并通过验证后，保存其专家和路由参数为记忆。冻结 specialist 的记忆参数；通用专家照常在线学习。明确记录这种 D 专用的保留机制，保留 D-v2a 作为“仅增长、全 active 继续学习”的对照。
- 每 32 个成熟 interval 可以提议一次已有 specialist 的复用。候选选择只能依据最近 32 步可见 z 的表示与已存 centroid，最多一个；没有合适记忆就跳过。不能用真实 mode_id 查表。
- 复用预览冻结记忆，不重新执行 64 步训练；在随后 16 个全新 interval 记录完整替换模型与 live 的预标签预测。验收继续采用相对损失改善≥1%、正常 FPR 与旧知识 guard。相同时间点的 live 通用模型可继续按规则学习，但成对预测须同时记录。
- 验收后用 8 个**预测 interval** 完成 specialist 间交叉渐变，旧 specialist 降至0再退役；新 specialist 从0增加。检查路由归一化、极端 logits 与不变的通用分支权重语义，防止把“参数连续”误当“输出无扰动”。
- 复用与周期性新生提议共用一个候选槽。忙碌时跳过并记账；通过旧专家验收不等于新 birth，ID 必须保持一致。
- 容量不足时，仅可 purge 已 dormant 的 specialist；按最近一次因果通过验收的时间最旧优先，ID 作为固定 tie-break。不得删除通用/active/验证中的专家。purge 必须释放参数、optimizer archive 和私有统计；记录真实字节。拒绝 shadow 只叫 candidate_discard，不计作“删除一个已部署专家”。
- 至少分别记录 birth、retirement、reactivation、purge 的真实发生数。没有发生某类事件，就不能声称该机制已证明收益。三个 law 未必形成 resident 压力，允许 purge=0；此时结论仅涉及退役/复用，不涉及硬删除收益。

完成真实 birth/retirement/reactivation 路径的 checkpoint 恢复：下一预测、下一 live update、下一事件及内存专家 ID 一致。已有 v1 工程检查作为基础，只为新增链路补充有意义测试。

## 4. 比较与晋级规则

本轮继续报告原注册的九个 first100 窗口，首次暴露与六次复现分组；不能换成对 D 最有利的 window。主要部署收益仍为六次复现 first100 AP 的等权 D−C。报告全程 AP、raw onset AP、positive-only resource F1、固定阈值 FPR/Recall、有效事件数及不确定性。

开发信号参考保持：六次复现均值 D−C≥0.03、至少4/6为正、同口径正常 FPR delta≤0.01；这只是是否值得确认的开发依据，不是统计证明。没有达到也必须完整提交。

有信号后，在 seed700 补充以下指定对照，然后才讨论冻结确认：

1. C-u1 与实际总计算接近 D 的 C-budget：明确总 wall time、更新次数、训练样本、峰值内存，不能用“时延有限”代替预算。CPU 干扰大时先固定更新机会，再报告时间误差。
2. C-fixed8：区分增长机制与单纯增容。固定8初始化规则在运行前提交，不能从 D 的未来专家倒灌初始化。
3. D-no-memory / D-no-reactivation：与 v2b 相同预算框架，分别禁用记忆冻结/复用，定位收益。
4. D-no-purge 仅在真的发生容量压力时运行并讨论硬删除贡献。没有 purge 就不做空消融。

当前 A 是冻结旧目标模型的参考，不是已优化的新目标预测基线。若要写“D优于所有其他方法”，必须先列清“其他”的方法集合，迁移项目中拟纳入的方法到相同目标并跑完；只有 A/C 结果时，结论只涉及 A/C。

## 5. 何时才改变情形

v2a/v2b 先复用 v1 数据，成本主要是几分钟级 replay，避免再次投入数小时生成。如果动态机制实际运行但仍无益，根据已记录证据只选择一个后续场景版本：

- 若 C 保持全部旧知识：增加真实服务种类、服务退出、长时间间隔和复现；使用可观测短历史区分物理响应，不能随机重映射标签。
- 若 birth 训练太慢而 reuse 有效：保留较长首次驻留、较短复现；本轮已有1200/180结构，不需要为了让 D 赢再只截取有利片段。
- 若需证明硬删除：注册总累计服务类型超过 resident capacity、部分永久退出且新服务继续进入的长流；报告 resident/峰值字节与新服务接纳失败率。不是简单把 dormant 改名“删除”。

任何场景更改明确标为新开发协议，公开 D 获得的后台计算、内存和先验，给 C 相应清晰的现实限制。别引入秘密 mode ID、未来切换时间或未来标签。确认种子只在最终配置与比较集合冻结后一次性运行；不得按结果筛选种子。

## 6. 必交付文件与停止点

提交代码及以下可读产物，更新根目录最新入口，并附实际可运行命令：

- registration.json、availability_audit.json、immutable_stream_manifest.json；
- runs/<run_id>/<arm>/predictions.npz、summary.json、lifecycle.jsonl、checkpoint_manifest.json；
- candidate_pairs.jsonl（预测时点、标签成熟时点、训练/验证索引、验收理由）；
- comparison.json、cost_profile.json、status.json；
- docs/PROTOCOL024_V2_RESULTS.md，逐项回答触发、训练、验收、保留、复用、删除是否发生，收益出现在哪些窗口，限制是什么。

status 必含 candidate_created/accepted/rejected、birth/retirement/reactivation/purge、lifecycle_exercised、development_signal、confirmation_run=false、unused_confirmation_seeds=[701,702,703]。既不能把 workflow success 当科学成功，也不能因 development_signal=false 丢掉这一轮。取得开发信号后先完成指定对照，不自动启动确认。
