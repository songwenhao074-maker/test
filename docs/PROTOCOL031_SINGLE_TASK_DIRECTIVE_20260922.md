# Protocol-031 revision003：构建稀有业务回归场景并完成一次D/C试跑
状态：planned_not_implemented_not_run。用户已明确授权替换尚未执行的旧031目标；当前只执行revision003。

## 唯一任务
**构建并冻结“长驻业务覆盖＋稀有旧业务短暂回归”场景，在其中运行C_fixed5与D_nonblocking_reuse各一次，检验D是否具有条件性优势，交付后停止。** 数据构建、必要实现和两组回放组成一个任务，不是自动开启一系列实验。

目标不是D在所有场景获胜，也不是预先把结果判为D最佳。本次比较范围仅C/D，A/B与其他固定结构未测。允许D更多驻留记忆/后台计算，按实际消耗披露。此试验同时改变场景、公共更新频率和D提议预算，不能解释为单因素消融。

## 当前唯一配置
读取[场景规范](PROTOCOL031_RARE_RECURRENCE_SCENARIO.md)、[复用规范](PROTOCOL031_NONBLOCKING_REUSE_SPEC.md)、[机器登记](../artifacts/ftmoe_online/protocol_031/scenario_registration.json)与[plan.json](../artifacts/ftmoe_online/protocol_031/plan.json)。

F0=300，U/V首次各1600，W长驻1600；之后U/V交替回归六次，每次128，中间W各800，总9868计分步＋1保护步，共9869步。U/V/W固定映射现有S1/S3/S4；采用既有物理需求规律，事件概率0.30。两组seed700/model1、共同73维因果输入、label t+2、训练重放64步、常规更新每16步。D新生从成熟数600开始，每1600步一次；复用每32步探测、一位候选、16个未来区间验收，并行主birth，不用业务ID控制。

相似度只用于排序；部署仍要求相对损失改善≥1%、normal概率增量≤0.01、验证FPR增量≤0.01、F0正常NLL≤live×1.02+1e-6及FPR增量≤0.01。容量仍最多8含shadow、满时保护已验收记忆。并发状态/取消规则按复用规范。

## 数据构建与执行顺序
使用独立031注册、collector、audit、data verifier与输出目录。复用模拟器/工作负载基础设施，但不要将新timeline写回025/027注册，不覆盖旧流、旧结果或历史门禁。新场景ID=protocol031_rare_recurrence_v2，新数据revision=protocol031_data_revision_002；旧“不重新模拟/只用revision002/九窗first100”仅适用于已替代的计划。

本版总长9869（9868计分＋1保护），不得裁剪旧长流充当新流。续跑必须匹配本版注册哈希，不能复用revision002计划的chunks/data_lock/模型检查点。collector、audit、verifier、窗口汇总和工作流均读取同一登记，不保留旧15468/15469硬编码。

先实现配置与小型必要检查，登记最终代码commit和源资产哈希，再连续生成一个新流。采用200步不可变chunk与完整simulator/workload/scheduler/RNG断点；中断可按同注册同seed恢复，不能拼接旧实验片段。每次生成作业在时限前保存最新断点并上传，后续作业恢复同一流；源数据、模型checkpoint缺失则记录阻塞，不换合成来源。总预算一个完整数据流，断点恢复不是新场景或新seed。

完成模型无关审计后生成data_lock，填写真实stream/chunk/事件/注册SHA并冻结；登记中的预生成expected_stream_sha256=null不是免验哈希，模型入口必须核验冻结data_lock且禁止传空哈希。审计只要求U/V/W和本次六回归，不套用旧六业务/九窗口或旧stream固定SHA。未满足[场景数据门禁](PROTOCOL031_RARE_RECURRENCE_SCENARIO.md)就输出data_audit_failed并停止模型部分，不自动调物理参数或重抽seed。

随后在同一模型job、相同依赖与硬件下，独立进程顺序运行C与D各一次。使用030单线程确定性CPU配置，Python3.8、torch2.4.1 CPU、dgl1.1.3、dill0.3.8；两组完全相同。C沿用固定5专家结构与初始化，正常训练所有专家和路由；D的共同前4专家初始化相同，额外shadow预算计入成本。不要带入030全体专家旁路/逐步重型内容哈希作为日常方法路径。

生成审计与比较应读新注册，不能调用硬编码protocol027旧哈希/5520步/九窗的父入口冒充031。旧历史注册中的seed封存、数据revision次数限制不用于阻止本次用户已授权的新场景；保持其历史文件原样。

## 防止再出现报告故障
开跑前用小型fixture验证NumPy标量/数组/索引转换、非有限值处理、六窗口汇总、status和finalizer；不要用default=str隐藏类型问题。先保存每组原始预测与日志，再比较。最终Git产物清单在报告定稿后生成，原始artifact和恢复报告各自记录哈希；后处理失败只从已有产物恢复，不追加全流训练。

实现新.github/workflows/protocol031-rare-recurrence.yml，专用分支protocol-031-rare-recurrence-20260922，push仅匹配该workflow文件并保留workflow_dispatch。允许为同一数据流恢复作业，禁止普通文档/结果提交自动再训练。旧protocol031-nonblocking-reuse工作流/分支不作为当前启动入口。此次计划提交不启动任何实验，由接手模型实现并启动。

## 判定、交付、停止
唯一主指标改为六回归first128等权AP差D-C；初步开发参考为6/6有效、均值≥0.03、至少4/6正差、合并正常FPR差≤0.01、W各阶段等权AP差≥-0.02。次要报告前32/64步、每次回归、W、全程AP/FPR及CPU/墙钟/RSS/驻留参数/优化器/p95/额外训练成本，不能用旧C结果与新D比较。

另报告初学后与每次回归前的专家来源/驻留情况、真实复用与首次影响游标、完整验收数值及取消新生已消耗预算。没有形成或保留U/V有效记忆、未复用、D落后都如实交付；不得延长训练到通过、预装专家、按阶段ID切专家或事后挑获胜窗口。旧九窗口和029/030记录留作历史，不改写失败。

交付docs/PROTOCOL031_RESULTS.md，独立方法registration、数据冻结manifest/data_lock/audit、run_id隔离的comparison/status/lifecycle/reuse/cost及完整artifact链接/SHA。结果必须标出scenario_id与plan_revision=3。回写main，同步README、AGENTS、NEXT、PROJECT_CONTEXT、当前handoff。完成一个新流和一对C/D（或明确阻塞）后停止；不自动加A/B、种子、阈值搜索、其他场景或后续协议。
