> 历史指令：run35716480519已完成，paired_audit_valid=true，后处理已从artifact恢复，无需重跑。以下为原登记正文。当前唯一任务为[031非阻塞复用试跑](PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)，见[030分析](PROTOCOL030_ANALYSIS_20260922.md)。

# Protocol-030：同机成对验证旁路无干扰
日期：2026-09-22。状态：planned_not_implemented_not_run。基点：871d062519149e3a7b24265060da39963869ded5。

## 唯一任务
**确认029旁路是否改变在线轨迹，交付一份有效或明确失败的记忆效用诊断。** 在同一个Actions job、同一个CPU和依赖环境中，两个独立进程顺序运行原D：audit_off一次、audit_on一次。两次执行组成一个配对复现实验；不加入其他训练方法，不重跑C，不追加科学实验。

028原结果和029无效记录永久保留。030采用新的、事前登记的同机比较基准，不把029原门禁改成通过；相对历史028的匹配程度单独报告。

## 固定算法与执行环境
- 数据仍为protocol027_data_revision_002，seed700/model1；学习率、更新频率、训练样本、birth/reuse、相似度阈值、全部验收门槛、最大8驻留专家均继承028。不得用五个已知时点/专家9/阶段标签决定实际行为。
- 仅为本次复现固定CPU执行：OMP/MKL/OPENBLAS线程数统一为1；每个子进程计算开始前设置torch intra-op/inter-op线程为1、torch.use_deterministic_algorithms(True, warn_only=False)。固定PYTHONHASHSEED=1；初始化沿用原Python/NumPy/Torch种子逻辑，不在在线过程中重复重置随机状态。两侧同设置，不搜索不同线程/算子配置。
- 保持Python3.8、torch2.4.1 CPU、dgl1.1.3、dill0.3.8及仓库依赖；同job只安装一次，保存pip freeze、CPU信息、torch配置、线程/确定性设置、代码和注册哈希。不要为本任务升级框架。
- audit_off必须调用原Protocol028DynamicSession在线策略；audit_on继承同一策略，只允许修正旁路状态隔离、比对与汇报。两侧初始化参数/优化器/RNG内容应一致。单线程改变数值执行条件，不意味着重现了历史三线程逐元素结果；报告此区别，不能拿新耗时声称D变快。

数据来源：run35682811782/artifact10675401651，名称protocol027-frozen-data-35682811782，ZIP SHA256=111a4c5fc5c508a9e169fdbffe823ee36dc66c037c1775a6aa1abdbaba31c0a1；stream SHA256=fdea84306ac752611e4d0b1b4cd2300d0e0dcbc62ace07ac94096d904b310761。不重新模拟或修改源数据注册。

历史028证据：run35705211072/artifact10684617851，ZIP SHA256=e3be1c97d0f629c9ed40f61cfd4561b6b04e37b68357a2c92a23ac8f14e2e27e。029证据：run35711135230/artifact10688265540，ZIP SHA256=56c5cbe760cea17abeb62b9bbb3d30116ecef338f2fcaa9afda3fe0eaf80bb16。实际读取artifact元数据核验ID、run、digest及解包内容；不能仅把期望哈希抄入JSON并声称已验证。

## 必要修正和验收
1. 旁路保持标签前预测、未来16步验证及原guard，不反馈线上决策。检查参数、优化器、RNG、模型buffers/train-eval模式、requires_grad、拓扑及会被线上后续步骤读取的缓存。当前版本仅用tensor版本计数和部分状态快照，不足以覆盖全部缓存；补充必要的内容检查/恢复。不要无依据重写主训练路径。
2. 在两个进程生成按稳定名称索引的初始化与在线更新状态摘要；可记录逐步轻量索引、关键节点内容哈希。不同进程的Python对象id不用于比较。发生差异时保存首个不同的预测/更新/生命周期位置及字段；排除时间、文件路径、run_id和纯诊断计数，不排除算法状态。
3. 030配对门禁同时要求：检测概率和分类概率最大绝对差均≤1e-6；labels/raw_labels完全相同；九个回归first100各窗AP、九窗等权主AP、全程AP差均≤1e-6；离散生命周期事件、验收/拒绝原因及最终拓扑一致；旁路状态违规0。报告精确相等比例及最大差位置，不能靠四舍五入、降低保存精度或只比汇总AP过关。原容差不扩大。
4. 另输出audit_off对历史028、audit_on对历史028的完整比较，明确historical_028_match。若同机配对通过但历史匹配未过，只能声明“030固定环境下，旁路在已登记容差内不干扰”；不能声明修复了029原轨迹或重现了历史028逐元素结果。若同机配对不通过，记paired_audit_valid=false并给出首个分歧证据后停止；不临时调容差、再试种子或追加完整回放。必要的小型隔离检查属于本任务，但正式全程回放预算固定为两次。

## 诊断汇报
audit_on沿用029全体休眠专家旁路，不只重看那五次。按首次业务/回归阶段分别报告通过验收的候选机会、busy与相似度交集、同一16步上候选和四通用专家参照的损失；单独列出物理无记忆与控制器no_memory事件。结果页同时显示检测/分类误差与九窗主AP门禁，主发现前明确有效性。

九个回归窗口报告全部覆盖情况、同期live/四通用专家/每个休眠专家的AP和FPR；历史C只能是注明来源的参照，不作同期成本比较。事后最优候选继续标注post_hoc_oracle_diagnostic，不得当作在线D>C；没有通过候选也应正常交付。

## 执行、产物与停止
执行模型先登记独立030配置、实现入口和最小必要检查，再用.github/workflows/protocol030-paired-diagnostic.yml、分支protocol-030-paired-diagnostic-20260922发起一次明确push启动；push只针对该workflow文件，保留workflow_dispatch。文档/结果提交不得启动训练；不要修改旧029工作流来冒充030。设置足够单线程运行时限，不在失败后自动追加完整重跑。

交付docs/PROTOCOL030_RESULTS.md及run_id隔离的status、paired_consistency、historical_comparison、first_divergence、environment、diagnostic_summary、opportunity_audit和完整预测artifact链接/SHA。两侧执行耗时与旁路开销单列。结果写回main，同步README/AGENTS/NEXT/PROJECT_CONTEXT/当前handoff，然后停止；不自动修改D、匹配门槛、场景或添加A/B/其他种子。
