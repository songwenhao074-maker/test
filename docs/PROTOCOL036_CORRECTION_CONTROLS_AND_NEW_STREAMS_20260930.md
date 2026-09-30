# Protocol-036：校正机制对照与新随机流稳健性

状态：指示已登记，未实施、未生成新流、未启动科学运行。当前任务仅发布方案；后续模型在用户要求执行本指示时获得下述有限预算。禁止发布即自动开跑、失败后自动扩展预算或偷偷改变模型。

## 目的与执行基线

检验 035 的校正收益能否在新随机流持续出现，以及 C 分数校准或线性特征校正是否已足够。035 结果保留为 recurrence_only。036 是新实验，不能覆盖 035 登记，也不能将 035 的开发数据纳入新流均值。

代码基线：`84282143b849dd3b5684e7f879964a2ab485fab0`。执行分支 `codex/protocol-036-correction-controls-20260930` 从该基线建立；main 不含全部科学实现，不能从 main 重新拼出历史算法。实施在新文件/036目录中完成，不修改历史 033/034/035 算法、原始预测与 plan。

先读 035 独立复核及机器登记 `artifacts/ftmoe_online/protocol_036/plan.json`。实现、所有分支公式、两阶段分析脚本和 guardrails 必须在查看任何 036 控制臂或新流结果前冻结，并记录实际 checkout SHA、文件哈希及 plan SHA。阶段 A 结果不得用于修改阶段 B。

## 固定四个方法

C_ref：完整保留 C_fixed5 的原有训练、初始化、分类与检测路径。每流仅训练一次并生成 issued feature/logit tape，其他三个分支共享这一条 C 轨迹。D_off 直接复制 C，不训练。

设 issued z∈R^73，m=C_detection_logits[1]−C_detection_logits[0]。所有检测分支采用 `[C0−δ/2, C1+δ/2]`，分类精确复制 C。

| 方法 | 唯一新增部分 | 参数数 | 初始化 |
|---|---|---:|---|
| D_cal | a=0.9*tanh(θ)，δ=2*tanh((a*m+b)/2) | 2 | θ=b=0 |
| D_lin | δ=2*tanh((w·z+b)/2) | 74 | w=b=0 |
| D_corr | 035 原 73→32→ReLU→1，δ=2*tanh(raw/2) | 2401 | seed3501；首层按035，末层全零 |

D_cal 只使用 C 当时的分数，不接收 z、host ID 或阶段 ID。其单个时刻的变换严格单调，因为导数 1+a*sech²≥0.1；在线参数随时间变化可改变整段 pooled AP，所以不能误称“校准绝不会影响此实验 AP”。D_lin 与 D_corr 使用相同原始 73d 输入，不另做归一化、重采样或特征筛选。这些是功能与简化对照，不是同参数量/同算力证明。

三个分支均用 mean BCEWithLogits(m+δ,y>0)+0.001*mean(δ²)，对批次全部 host 行取均值。独立 AdamW(lr=1e-4, weight_decay=1e-4, betas=(0.9,0.999), eps=1e-8)，clip_grad_norm=1。每 16 步一个更新，recent64 成熟 interval 的 C 实际抽样批次，最多32 interval，保留重复索引和原顺序。分支不得向 C 回传梯度或影响 C RNG。保持 branch seed3501、C model seed1，不按流更换初始化，不为控制臂调学习率。原 D_corr 必须逐项保持035公式与数值类型。

## 两阶段和硬预算

A，已观察开发流700：复用035 C tape及D_corr预测，只新训练 D_cal、D_lin 各一次（2条）。已知 D_corr 不重训；不生成 seed700，不追加阈值扫描。报告三个校正与C、D_corr与两个控制的全部差值，明确这是开发归因。

B，三条新随机流：replay seeds **3601、3602、3603**，按此顺序，每流固定 C_ref+D_cal+D_lin+D_corr 各一次（12条）。三条全部报告，不依据第一条表现停止、补充种子或择优汇报。模型种子保持1、分支种子保持3501。若发现这些完全相同流已被用于模型选择，应记录其暴露并停止称其新流；不可擅自搜索替代种子。

总预算：**14条新科学训练序列，3条新原始流，0次科学超参扫描**。预算单位为一个方法在一个完整流上的训练遍历，即使只需几秒仍计一次。C tape读取/纯计分不算训练。最多2次真实前缀联合工程执行，每次≤256步；它们包括C与三个分支，用来做连续/断点成对验证。纯合成测试不受此限。不得重训旧动态D或评估新缓存路由。每个 sequence 的开始、完成、恢复、异常均入不可覆盖 ledger。

## 数据与因果合同

A 输入：035 run36585058415，artifact11041705829，ZIP SHA256 `849a8367727e00833a3ae92fda873a3989121d2f7ee207e66f9789357d538421`。tape SHA256 `60f3f898a70dc886cd9e5284c1c1e62f3e086aac52e4e829e9fb9232c69dc988`，update_batches SHA256 `030114f1ff48bff0124d50b1bd4c9d23987ce62431502ae8c603bc6f4f39afc5`。核对内部 manifest，缺失或哈希不符则失败关闭，不重建替代。

B 复用033生成器的机制、服务映射、故障定义、事件概率0.30、16hosts、5969原始行/5968预测、200步chunk、最终169步chunk、总30chunks/流与原timeline，仅 replay seed 改为登记值。保留 baseline[0,300)、U_first[300,1300)、V_first[1300,2300)、W_long[2300,3300)，六个128步复现窗口起点3300/3808/4316/4824/5332/5840，中间W间隔380。新增036生成/校验入口，不能把新流强行送入硬编码700与旧stream SHA的033校验器，也不能改写033冻结manifest。

固定物理参数、基础checkpoint、输入顺序、标签规则、特征与归一化算法；列出冻结文件hash及数据派生量的来源。禁止从新流未来标签/全流统计拟合输入预处理。若既有归一化依赖全流统计，必须在生成/训练前披露并登记修订，不能一边宣称因果一边偷偷沿用或改成另一套。每条新流生成一次，分段恢复保留同一模拟器/RNG状态；生成完成后先冻结 stream hash和输入审计，再允许C训练。不得看结果后重生成更有利的流。

目标为同一host的raw[t+1]；cursor t先预测，再发布i=t−2标签，再按周期更新；预测只允许i+2<t，更新允许i+2≤t。所有z和C logits必须是预测时 issued 值，不能用未来C权重重新编码旧样本。末尾cursor5968/5969仅结算计分，不训练。把全量tape载入内存不意味着可读取未来行；用统一访问器和实际访问日志证明时序。

## 报告与预登记决策

保持035并列分数组AP定义、阈值0.5、六窗等权、后四窗、前32/64、24个不重叠32步块、16hosts、六W块及全程/pooled recall与FPR。AP只有一种类别时记null并报告支持量，不能填0、删掉种子或修改样本规则；主指标无法定义则相应门槛未成立。

新流主结论只用B，先每流成对求差，再对三流等权平均。3条流是3个重复，hosts/窗口不是独立样本。必须同时给每流数值与全部18个复现窗口，不只给均值，不做显著性或广泛泛化承诺。

`cross_stream_gain_signal` 需全部成立：

- 三流平均全程ΔAP≥0.005；平均六窗ΔAP≥0.005；平均late4ΔAP≥0；平均prefix32ΔAP≥−0.005。
- 至少2/3流同时全程ΔAP>0且六窗ΔAP>0；任一流全程或六窗ΔAP不得低于−0.002。
- 每流通过035共同护栏：全程和回归pooled FPR增量≤0.01、recall增量≥−0.01、W六块等权AP增量≥−0.02。
- 新增局部护栏（仅036前瞻应用）：任一复现窗口recall增量≥−0.02，任一复现窗口前32步recall增量≥−0.03。完整报告这些格子的正负样本数、TP/FP/FN/TN。035已发生的下降不能用来改写其原门槛。
- 所有输入、因果、预算、真实实现恢复与C隔离审计通过。

两个解释性比较分别给出信号，不拼出“结构胜利”：D_corr相对D_cal、相对D_lin在B上平均六窗ΔAP≥0.002且平均全程ΔAP≥0，记为对应的增量证据；同时报告每流方向与全部护栏。若控制与MLP相当，结论为复杂性尚无充分支持，不能为了赢控制补扫超参。它们不同容量与优化几何，不能据此宣称严格因果分解。

后续分支：新流D_corr稳定且优于控制→保持该底座，下一提案再设计同预算对照与遗忘/记忆需求测试；简单控制相当或更好→优先简化；总体增益不稳定→定位流/阶段失败，不加复杂结构掩盖；AP通过但局部召回失败→单列operating_point_risk，下一提案才考虑预登记因果校准，036不改阈值。所有分支仅报告建议，不自动执行新实验。

## 开跑前必须修复的工程条件

1. manifest分别记录 `workflow_ref_sha`、`checkout_sha=git rev-parse HEAD`、plan hash、输入hash、依赖版本。不要把GITHUB_SHA直接称实际算法commit。
2. C和各分支每512预测及末尾保存可继续的完整状态：模型/优化器、RNG、cursor、buffer、已成熟标签状态、issued tape/预测前缀、版本、更新记录、输入hash及预算ledger。实现真正的resume入口。已完成sequence绝不从零再跑；中断只能从验证过的同一状态继续，无法恢复则登记incomplete并停下该序列，不能把失败试验擦掉。
3. 用真实runner/序列化入口的连续与断点执行比较预测/状态，而非只调用另一个简化synth函数。冻结前必须有：未来标签/输入扰动前缀不变、成熟索引访问审计、末尾结算无optimizer.step、分支梯度隔离、D_off精确复制、真实批次一致、AP ties测试。布尔结论必须链接原始证据，不能硬编码true。
4. 预算与时间上限先登记；生成阶段分段推进，同一科学序列的精确恢复不计新初始化。分段预算不足时持久化状态再恢复，不能因 Actions 超时增加流数。并发锁固定协议；不为日志/同步失败自动重训。
5. 原始artifact先上传，保存artifact ID/URL、ZIP及逐文件hash、大小、有效期；再发布紧凑报告。更新main用最新HEAD重建仅结果文件的commit，非快进时至多3次重新读取/重建，保留他人launch receipt；仍失败仅标记publication_pending并输出已有artifact位置，训练保持completed。禁止force push。三个LATEST文件整体一致改写，历史正文不混入当前状态。

## 启动合同与最终交付

实现模型先冻结代码与plan，再启动独立的 `.github/workflows/protocol036-correction-controls.yml`。正式入口仅workflow_dispatch；dispatch所在ref与实际execution_ref必须分开记录。若只有GitHub连接器、没有dispatch工具，允许沿035启动修订采用一次性push helper，用GITHUB_TOKEN向正式workflow POST一次；helper无训练，只校验预期执行SHA、plan hash、无既有同批次run，然后dispatch。网络返回不明确时先查runs，禁止盲重试。helper启动后清除。外层main、内层execution_ref为036执行分支名，另设expected_execution_sha锁定实执行代码。不要复用或启动035workflow，不加入自动workflow_run训练链。

提交应含实现源码、冻结plan与场景登记、机器校验器、fixtures、逐流输入锁、完整预测/feature tape/更新日志/checkpoints、训练与生成budget ledger、per-stream及聚合comparison、因果与隔离审计、cost_profile、artifact_index、status、中文结果文档。终态分别记录scientific_status、publication_status、registered_signals，不能把部署/上传成功当成科学门槛通过。

最终用一句话明确：是否跨新流优于C、复杂MLP是否优于简单控制、是否有局部召回风险、下一步证据还缺什么。不要把本试验称为动态D记忆系统成功。
