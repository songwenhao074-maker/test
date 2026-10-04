# Protocol-044 revision 2：持久化恢复与正式入口修复指示

日期：2026-10-05（Asia/Shanghai）。状态：**已预登记，尚未执行**。本次提交只发布指示，不创建或触发实验 workflow。

## 1. 目标、继承关系与授权

目标仍是：结构逻辑合理的完整 D 优于 C；不追求最优结构，不要求 D 优于 D_no_gc 或 D_lin。保留“当前窗口证明无用才休眠/删除，缺少合适专家才新增”，不改成强制替换。

这是 044 的工程恢复修订，不是新科学方案或自动启动 045。科学设定完整继承 [revision 1 指示](PROTOCOL044_CLOSEOUT_AND_CAPACITY_20261004.md)。有效执行合同为：
1. 本文及 `artifacts/ftmoe_online/protocol_044/revision_002/plan.json` 控制恢复授权、工程门控和编排；
2. 原根目录 `artifacts/ftmoe_online/protocol_044/plan.json`、`plan.sha256`、`scenario_registration.json` **保留原字节**，继续作为科学配置/数据配方；
3. 与 r1 的冲突仅限本文明确列出的运行恢复例外。模型、数据配方、阈值、分析和科学预算不变。

用户此次要求分析中断并修改指示，授权发布这份恢复方案。执行模型收到用户交接后，完成修复→验收→注册内生成与四序列→分析→发布，无需每一步再询问。不要现在自动 dispatch，也不要等待另一个科学计划。

r1 结果 [PROTOCOL044_RESULTS.md](PROTOCOL044_RESULTS.md) 及历史 033–043 永久只读，不改写成成功。r2 单独交付 `docs/PROTOCOL044_REVISION002_RESULTS.md`。

## 2. 已核查问题与证据边界

### 2.1 这次中断的直接原因

- 第一条 Stage S run [37178041408](https://github.com/songwenhao074-maker/test/actions/runs/37178041408)，在初始化前因 registration path 为 None 触发 TypeError。零生成、零训练；随后修复过，不是后续 350 分钟取消的原因。
- 正式 run [37184889231](https://github.com/songwenhao074-maker/test/actions/runs/37184889231)，执行代码为 `0fb751738718d3d06aeeaa32cfcb593436fd7c87`，workflow HEAD 为 `500e3060c2f7871495d21bf8db18eda50656c6a4`。
- 该 workflow 在**同一 job 的一个 shell 步骤**内循环 30 次，每次最多 200 行；并没有在各段之间上传检查点。
- 2026-10-04 12:33:13Z 最后记录完整 `next_t=5200`；12:58:47Z 取消，与 07:08:30Z 开始、350 分钟配置吻合。目标 5953 raw rows，已观察完整 5200 行（约 87.35%），余 753 行不能视为已生成。
- `Preserve exact interrupted state` 的条件是 `if: failure()`，本次 cancelled 时被跳过。Actions API 当前列出该 run 的 artifact 数量为 **0**。
- 数据锁未建立，C_ref、D_lin、D_no_gc、D_bounded 全部未启动，科学 optimizer.step=0。没有这轮 D 与 C 的性能结果。
- 后半程明显变慢：从 4800 到 5000 约 26 分钟，5000 到 5200 约 27 分钟。不要用初段速度估算全程。日志不足以确定底层复杂度，**本轮不修改模拟器语义来提速**。

上一版指示虽要求可恢复，但没有把“每段完成后远端持久化且回读通过，才准推进”规定为硬门控。运行实现只有本地分段恢复能力，缺少 runner 消失后的恢复能力。仅加长超时或将 failure 改成 always 都不足以解决此问题。

### 2.2 尚未运行到、但必须提前修复的入口问题

已静态核查同一冻结 SHA 的 `run_ftmoe_protocol044_science.py`：

1. 第 98 行 `make_c(data,out,run_id)` 接受三个参数，第 125 行调用 `make_c(a.data,a.input_lock,out,a.run_id)` 传四个。原样进入 C 会触发 TypeError。统一为显式 `make_c(data,input_lock,out,run_id)` 并真正使用传入 lock 验证数据，不隐式猜路径。
2. `validate_gate` 第 64 行无条件拒绝 `started=true`；C、D_lin、dynamic 都先调用它，再处理 resume 参数。故实际命令行恢复路径不可达。必须区分 new/resume/completed：new 只允许未开始，resume 必须已有已验证的完整状态且未完成，completed 一律拒绝；不能通过清除 started 或重建 ledger 绕过。
3. 原预算总数在 `complete_seq` 时汇总，不能据此证明中断中途没有用过 optimizer。修订运行包装层必须持久化中途实际计数和状态，不能用“尚未 complete”当作零消耗。

这些是静态发现的后续阻断风险，不是本次取消日志中的异常。E_gate=true 证明了当时被测核心与 case，不代表未覆盖的实际生产入口已全通过。

## 3. 有限恢复例外：不得伪称从 5200 精确恢复

先只读检查 r1 相关 run/artifact/已登记保存位置，记录 `recovery_inventory.json`。完整可验证的 simulator/RNG/chunks 状态若确实存在，优先恢复；不得将日志中的 next_t 当成检查点。

现有证据显示 r1 完整状态不可得。故**本修订明确授权一次、且仅一次** seed4401 的同配方从零重建，记为 `generation_key=protocol044_revision2_seed4401_reconstruction1`，关联被中断的 r1 key。该例外理由为 runner 丢失状态且正式模型全未启动，不是性能不佳。

- r1 的已消耗生成时间/部分流如实保留；r2 重建不是“零成本继续”，也不能证明与丢失的 5200 行逐字相同。
- 若找到可恢复状态，使用该状态，本次从零重建额度不使用；两条分支互斥。
- 种子 4401、5953 行、阶段顺序、服务映射、物理参数、scheduler 权重、模型种子、分析指标均不变。禁止换 seed、生成多份选优、看指标后重建、拼接旧流、强制触发 GC。
- r2 第一次初始化后，先把 **next_t=0 的完整初始化状态**远端上传并回读，再允许第一行生成。恢复这个 t=0 状态不是再次初始化。初始化状态在远端确认前不得进入循环。
- 从 r2 第一个远端状态开始，只能从最新已提交状态续跑；不得再次从随机种子重新初始化。缺失/损坏无法恢复则停止，不自动申请第三次重建。
- 共只完成一份用于科学比较的流。四条科学序列和总梯度预算来自原未用额度，不新增四条重复对照。原 science_key 保持 `protocol044_revision1_stageS_seed4401_sequence`，r2 receipts 绑定此 key 防止双开。

## 4. 先做范围有限的工程修复，不重做科学实验

从 E 已通过的代码 SHA `0fb751738718d3d06aeeaa32cfcb593436fd7c87` 恢复实现；当前 main 已清理的 workflow 不等于源码丢失。在独立执行分支 `codex/protocol-044-revision2-durable-recovery-20261005` 工作。

允许修改：生成 init-only / 分段出口 / soft deadline；训练入口绑定与 resume 分支；检查点包装、预算收据、远端上传下载/路径重定位、workflow 和结果发布。不得修改学习公式、梯度时序、RNG 消耗次序、样本批次、专家控制/GC 决策、模拟器物理规律或判定阈值。

双重 source lock：
- 科学合同 r1 原字节 plan SHA256 `a27a6cca6abde40b6395876eccd3db5f550a044e952a701fe63aa2e4306a5f51`；scenario SHA256 `6e6bd03efc0d84f5e885eacf36f1ac80e770198703463cc051216ad1a741683f`。
- r2 recovery plan 另存 sidecar 并验证自己的 SHA；生产入口同时验证 r1 科学合同与 r2 恢复许可。旧科学 payload 的 revision=1 明确标作 science_config_revision，外层 execution_revision=2；不要全局替换数字 1，也不要关闭 plan hash 校验。
- 冻结 `science_source_sha` 与新的 `execution_sha`、所有实际执行文件 hash、依赖、分析源码。新增 `E_compatibility.json` 逐文件列出 inherited/changed、理由及覆盖测试。
- r1 E artifact：id `11296492568`，run `37184789603`，ZIP SHA256 `90b6bab6a149c46da0aa19e551c02f76b5665a27a83e97d7de192e21e08b6591`。下载、核 hash、读真实 gate。不可只引用报告代替验证。
- 原 E JSON 保持原 SHA/内容，不把新 execution SHA 填进旧 E 冒充验收。新的 `E_recovery_gate.json` 绑定旧 artifact、逐文件兼容性、新 SHA 和本节测试。生产入口检查这两层 gate。
- 未修改核心的原 19 cases、043固定轨迹复核、内存测试可继承。改到某项覆盖区域则只重验受影响合成 case；若核心科学语义变了，应停止重新登记，不能悄悄扩成结构实验。

### 4.1 正式生成前的必须通过项

测试用小型合成 fixture；不加载 043 真实样本进行模型前向/梯度，不启动 seed4401 正式生成，不形成第五条科学训练序列。复用真实 CLI/handler/validator/checkpoint 路径；禁止仅构造一个假 gate 后声称端到端通过。fixture 模式必须显式标注且被生产入口拒绝，不能将其混入正式预算/数据。

1. **入口合同**：实际调用 C handler 至 session 构造及首个合成 step，覆盖 make_c 参数绑定与显式 input lock；不能 mock 掉 make_c 来绕过签名问题。D_lin 与两个 dynamic arm 同样覆盖。
2. **真实恢复分支**：四条入口分别 start→有状态 checkpoint→退出进程→换目录新进程 resume→至少一个下一步→正常结束；已 started+合法状态可继续，started+无状态/错 arm/错 hash/错预算均在任何 optimizer 前拒绝。
3. **连续/分段等价**：每条 arm 用同一合成 fixture 的连续与分段版本，比逐步输出、参数、Adam、RNG、游标、实际批次与事件日志；不是仅比较最终 AP。只改变分段位置，不改变在线时序。
4. **新 runner 的生成恢复**：使用小型工程 fixture 覆盖 init-only→0 checkpoint 上传→另一 job 下载→部分 segment→远端保存→再次下载续行；覆盖跨服务切换和最后不足 200 行。数据/RNG/state 等价；零额外模型梯度。
5. **上传事务**：上传失败/远端 hash 损坏/错误 parent 都不得开始下一段；上传已成功但 receipt 未写时，可通过唯一 run/attempt/key/cursor 查回并验证已有 artifact，不重新生成。
6. **取消与软截止**：实际 Actions fixture 在已提交一段后取消/故障，使用**另一 job/run**从 artifact 恢复；不能只在同一进程 reload。不依赖被取消 job 的末尾清理来保存唯一副本。
7. **预算和幂等**：重复 dispatch、重复完成、旧 cursor、completed 序列、过期 receipt、负预算、计数与 checkpoint 不一致均拒绝。两次并发请求只有一个持有执行权；另一个不初始化/不训练。
8. **发布**：发布工具不得覆盖 r1 results；断在分析/发布后仅继续无训练步骤；恢复时不把 source_code_lock 的旧 SHA 当作新代码通过证明。

通过后冻结代码、workflow、analysis 与 `E_recovery_gate`；本轮工程验收不依赖正式 D/C 表现。fixture 失败可修复再测受影响项，记录次数；不能趁机调模型阈值。

## 5. 正式数据生成：远端提交先于推进

### 5.1 具体编排

- workflow_dispatch only；禁止 push/schedule 自动开始实验，`concurrency` 按全局 science_key 串行，`cancel-in-progress: false`。不创建自动循环重试或自动新协议。
- 生成拆成 init job + **每 job 最多 200 个新增 raw intervals**；完整流为 29 个 200 行块和末块 153 行。每一 job 至多推进一个数据块，不能恢复原来单 job 循环 30 段。
- 建议预定义顺序 job/reusable workflow 链（后继 needs 前继成功），同一 run 的 artifact 立即可供后继取；中断后用户交接给执行模型继续既定链，不需要新科学批准。若用多次 dispatch，receipt 必须跨 run 可发现且严格串行。
- job 硬超时 90 分钟，工作进程 soft deadline 60 分钟，剩余至少 20 分钟用于保存、上传及回读。开始前与每个安全 interval 边界检查 deadline；到时形成合法 transient checkpoint，后继从该 cursor 续到当前 200 行块末端。未完成块不能伪报成功完成整个流。
- 不假定每段等速；记录每段耗时、RSS、artifact 大小、next_t。若单 interval 无法在安全余量内完成，停止报告，不扩大窗口、跳行或改变模拟器。不要把全部新流总耗时限制在原 350 分钟单 job。
- 初始化或恢复完成必须核验注册路径非 None，snapshot hash、036 源码/权重与实际 scheduler、所有 RNG/creation IDs。固定依赖与相同绝对 source layout；如 pickle/dill 引用旧路径需受测的路径映射。

### 5.2 durable receipt 的必要内容

每个状态包含完整 simulator/workload/scheduler/stats/RNG、next_t、已完成 chunk 清单及 SHA、未满块 transient、registration snapshot、source/runtime locks、generation ledger。不能只上传 NPZ 或 resume_manifest。

保存为唯一且不可覆盖的 artifact（key + next_t + run_id + attempt）。必须：
1. 本地完整性校验；
2. 上传 state 与数据依赖，保留至少 90 天；
3. 用 artifact ID 下载到干净目录，验证 ZIP digest、内部全部 SHA 和覆盖连续性，新进程加载成功；
4. 发布 compact receipt，含 execution_revision、科学/数据配方哈希、generation_key、science_key、parent receipt SHA、run/attempt、artifact ID/name/digest/size/expiry、next_t、chunk end、model_runs_started=0；
5. 后继只认已确认 receipt，先下载并复验，才能生成下一行。

同一个 artifact 可以自包含，或引用不可变 chunk artifacts；后者必须校验完整依赖图并保留所有被引用版本，不能清理唯一依赖。小型 receipts 写回执行分支 `artifacts/ftmoe_online/protocol_044/revision_002/progress/`（[skip ci]、非强推），便于跨 run 找到最新一致状态；已上传未写 receipt 可按唯一身份幂等补写，不把分支写入失败误当作“可再生成”。

每个全局 cursor 只允许一个已提交状态，不覆盖旧 artifact，不靠 cache 作为唯一检查点。末尾 `if: always()` 上传仅作补充，不能是持久性的唯一机制；不得用其结果覆盖失败状态。

### 5.3 中断后的有限重试

- 生成过程在未提交区间被强杀：从最后已确认 simulator/RNG 状态重算**未提交后缀**，不重做已提交段，不重新 seed 初始化；记录 wasted intervals/wall time。每个 parent receipt 最多两次这种中断重试，仍失败则交付阻塞原因。
- 状态 hash、配方、代码或连续性不符立即停止，不从更老状态碰运气。完整数据的模型无关 gate 不通过也停止，禁止再次生成选优。
- 真正完成 5953 行后，在新进程 assemble/audit，沿用全部 r1 支持度/标签/因果/物理检查。远端上传并回读 data artifact、data_lock 与 hashes，之后才准训练。

## 6. 四条正式科学序列：保持原预算，另做可恢复编排

顺序固定 C_ref→D_lin→D_no_gc→D_bounded。共享同一数据、C issued feature tape、真实 batch 序列及 B 输出；不重新编码历史，不重做任何已完成序列。D_no_gc 不理想不阻止运行 D_bounded。

- C_ref ≤372 optimizer steps；D_lin ≤372；每个 dynamic live≤744、shadow≤64、合计≤808；总≤2360。专家数/forward/候选/复用等预算完整继承 r1。工程合成计算单独记账，不能伪称没有工程开销。
- 每条序列单独 job，训练 runner 提供安全 pause/resume 出口。至多每 **512 个已发出预测 interval** 在所有该时刻原子事件及更新完成后，保存完整模型/Adam/RNG/在线窗口/延迟标签/统计/历史输出/审计游标/预算 checkpoint，立即上传回读，才推进下一段。不能只在 arm 结束上传。
- job 硬超时 300 分钟，工作进程 soft deadline 240 分钟，预留≥30 分钟；截止时只做安全 checkpoint、远端确认和退出，下一 job 继续同一 arm。禁止四个 arm 挤在一个生成 job 后面。
- 首次正式 optimizer 前持久化 `started` 与初始状态；每段开始将 segment id/cursor、可用预算、状态 hash 与 in-flight intent 持久化，完成时原子发布实际计数和新状态。旧 `complete_seq` 不能成为唯一计数来源。
- **不承诺强杀后可无限回滚训练**：如果可能已有正式 optimizer 调用但最新远端 checkpoint/事务日志不能证明精确状态与计数，则 fail-closed 停止，不把这部分当零，不从旧模型重跑来突破预算。恢复只接受完全确认状态；重复提交完成 receipt 应幂等。
- 同一 job pause/resume 也经过真实入口验证，不能删 started、重写预算、关闭 strict_journal。完成某 arm 后远端 seal predictions/update logs/feature tape，后继固定 artifact ID/hash。
- 若只剩 analysis/publication 未完成，直接读取已 seal raw 继续；禁止再做 forward/gradient 以“补结果”。

## 7. 科学判定不改，结果必须如实收口

全部 r1 analysis 与守护阈值保留，包括全流 AP 增益≥0.002、四个返回128窗口平均增益≥0.002、至少3/4正增益以及误报/召回守护。新增、窗口休眠/复用和真正 GC→候选录用→至少64活跃 interval 的证据要求不变。不能为了完成而强制触发 GC 或删除有用专家。

r2 运行结束分开报告：
- engineering_recovery_pass；
- stream_complete / model_free_data_gate；
- 各 arm started/paused/completed、实际预算；
- D_over_C、lifecycle/reuse/resource、GC_closed_loop；
- analysis_complete / publication_complete。

无 GC 就写未覆盖；不达 D>C 就写未达到；工程中断就写未完成。三者不能混称“D失败”或“实验完成”。本轮未授权追加 seed、消融、阈值搜索或045。

## 8. 交付清单和执行顺序

先实现并发布：`recovery_inventory.json`、`E_compatibility.json`、`E_recovery_gate.json`、冻结源码/依赖/分析清单、合成入口/跨 runner 恢复证据、workflow 编排。
再执行：恢复可用旧状态或登记的一次重建→逐段 durable receipts→5953行 input/data seal→四序列逐段 seal→独立分析→结果发布→停止。

交付路径：
- 当前计划：`artifacts/ftmoe_online/protocol_044/revision_002/plan.json` 与 `plan.sha256`。
- 运行证据：`artifacts/ftmoe_online/protocol_044/revision_002/runs/run_<id>/`。
- 进度索引：`artifacts/ftmoe_online/protocol_044/revision_002/progress/`。
- 新报告：`docs/PROTOCOL044_REVISION002_RESULTS.md`。
- raw artifacts/依赖至少保留90天，compact登记全部ID/hash/下载位置；先raw后analysis。
- 同步 README、AGENTS、NEXT_EXPERIMENT_LATEST、PROJECT_CONTEXT_LATEST、GITHUB_EXPERIMENT_HANDOFF 为真实状态；所有结果提交 [skip ci]，不强推，不覆盖 r1 历史结果。

不将这份“待执行指示”写成已修复/已恢复/已通过。当前唯一确认的新事实是本次日志、artifact清单与静态入口问题核查。
