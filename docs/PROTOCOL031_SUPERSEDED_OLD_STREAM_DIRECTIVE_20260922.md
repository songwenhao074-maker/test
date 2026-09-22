> 已替代的历史计划（revision001，未执行）。用户已授权revision002新场景；当前执行入口为[031修订指示](PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)。以下原文仅用于追溯。

# Protocol-031：一次非阻塞复用D/C开发试跑
日期：2026-09-22。状态：planned_not_implemented_not_run。基点：09560d348a94af356e1d6b7e5fe0f08fcbf5aba9。030已完成，不再重跑配对诊断。

## 唯一任务
实现一个“非阻塞、经因果验证再激活”的复用策略包，在既有冻结业务流上运行C_fixed5与D_nonblocking_reuse各一次，判断能否产生实际复用并提高九回归窗口AP。共两次全流回放，完成后停止。

这是联合策略试验：**相似度改为排序用途＋复用验证不再被新生训练占用的主phase阻塞**。不是单因素消融。背景验证计算是D的额外权限，必须计入成本。不得把030事后候选直接作为已验收专家。

## 固定输入与对照
C沿用Protocol025FixedSession的固定5残差专家；D继承028的四通用专家＋一个活动专用专家、最多8驻留（含shadow），正常8步crossfade可短暂6个活动专家。保持已验收专家冻结、满容量保护旧记忆、birth周期/128步训练/32步验收、在线学习率/预算、初始化、全部标签时序和guard。不改主干、特征、样本、模型种子或数据。

两组同一个Actions job、同一依赖环境、独立进程顺序执行，均使用030单线程确定性CPU设置（线程1、torch2.4.1 CPU、Python3.8、dgl1.1.3、dill0.3.8，seed700/model1）。不得运行030的全体专家旁路与逐步重型内容哈希作为日常方法路径；保留必要因果/拓扑检查，并计入实际开销。无需新增旧D对照，028/030仅作注明来源的历史参照。

冻结数据：protocol027_data_revision_002；run35682811782/artifact10675401651，名称protocol027-frozen-data-35682811782，ZIP SHA256=111a4c5fc5c508a9e169fdbffe823ee36dc66c037c1775a6aa1abdbaba31c0a1；stream SHA256=fdea84306ac752611e4d0b1b4cd2300d0e0dcbc62ace07ac94096d904b310761。核验真实artifact元数据及文件，不重新模拟、不改027数据注册快照。031方法配置单独登记。

## 唯一新策略的精确定义
1. 沿用成熟数≥600且每32步到期的reuse时钟。至少32个历史可见z时，在已验收休眠专家中选择余弦相似度最高的一位，数值相同按较小ID；原q10阈值仅记录，不再据此拒绝启动。每次仅一位候选，不用未来标签挑选，不固定专家ID或业务阶段。
2. 复用使用独立pending槽，不把birth主phase改成reuse_validation。monitoring、candidate_training、candidate_validation、cooldown均可启动/推进；已有pending槽时记skip_pending，crossfade期间记skip_transition，其他无记忆/历史不足分别记录。正常在线更新与shadow训练继续执行；监测不复制专家参数、不增加第九个驻留专家。
3. 槽创建后取之后16个新预测区间：先产生真实live与“四通用专家＋候选、替换旧专用专家”的预测，再等原延迟标签成熟评分。候选/旧活动专家ID及角色需保持稳定；若birth先验收导致活动专用专家变化或进入crossfade，立即取消旧槽，记cancelled_topology_changed，不沿用旧验证样本。
4. 满16个成熟配对后，必须沿用原全部验收：监督损失相对改善≥1%；正常样本平均异常概率增量≤0.01；验证FPR增量≤0.01；F0正常guard的candidate NLL≤live NLL×1.02＋1e-6、FPR增量≤0.01。任何不可用条件都不能算通过。guard只验收、不训练；不降低这些预测质量门槛。
5. 同一成熟事件内，先完成已有reuse槽的到期验收，再执行原birth训练/验收；若reuse通过，先取消尚未验收的shadow（如有），释放其私有优化器、训练/验证缓存，再按原8步crossfade激活旧专家。取消shadow单列cancelled_by_reuse，不伪装为质量拒绝，不删除任何已验收记忆；旧活动专家按原规则退休。禁止同一点再接受birth或开始第二次切换。若reuse失败，仅清空该槽，birth及其cooldown状态不被重置。
6. 当步已有reuse与birth处理完成后，再按reuse到期标志尝试开启新槽；若此时处于transition则跳过。birth仍按原到期规则，只把主phase和实际shadow状态用于忙碌判断，不因pending复用槽而停训。新birth与新reuse可同点开始；不积攒/追补错过的到期次数。每个成熟标签、计数与birth机会只能处理一次。
7. 流尾不足16个成熟配对记censored；因拓扑改变取消与质量拒绝分开。保存pending槽以支持正确检查点恢复。新生日志满足created=accepted+rejected+cancelled+pending；reuse日志满足started=accepted+rejected+cancelled+censored+pending（同一记录只占一类；最终把未完成槽转为censored，pending=0），机会计数守恒。

现实解释：已有专家缓存可在新专家训练期间做低成本试用验证；相似度负责排序，真实延迟反馈决定是否接管。其风险是更多无效验证与取消新生训练造成的浪费，必须报告，不能预先宣称更省资源。

## 执行前必要检查
独立031实现，不修改旧协议已完成行为。关闭继承的阻塞式reuse启动路径，避免新旧逻辑双跑；原_decide_reuse拒绝分支会写主phase=monitoring，不能原样套用后破坏正在进行的birth。检查：birth训练期间可验证reuse；低相似度可启动但质量门槛仍能拒绝；预测严格早于标签；失败不改变birth状态；成功取消未验收shadow且不超驻留/活动限制；birth先接管会取消旧槽；同点接管只有一次；计数和检查点恢复正确。

修复031产物序列化：numpy整数/浮点/数组与最大误差索引转换为JSON原生类型，非有限值附原因，不以default=str隐藏类型问题。先用小型结果fixture跑通汇总、status、索引及finalizer。最终索引在报告定稿后生成；原artifact索引与恢复文件索引分开。后处理失败时使用已保存产物恢复，不重跑模型。

实现.github/workflows/protocol031-nonblocking-reuse.yml，专用分支protocol-031-nonblocking-reuse-20260922，push仅匹配该workflow文件并保留workflow_dispatch；明确启动一次。普通文档/结果提交使用skip ci，不触发训练。运行器先持久化两组原始预测/日志，再进行比较与序列化。

## 固定判断和交付
主指标仍为全部九个回归first100窗口等权AP差D-C；报告9/9有效性、每窗差、正差窗口数、合并正常FPR差及全程AP。开发参考保持均值差≥0.03、至少6/9正差、合并FPR差≤0.01。数值领先但未达参考应如实标注；单条已反复查看的开发轨迹不能称统计确认。

分别回答：是否真实reactivation>0；复用后性能是否提高；九窗主指标是否D>C。S6_first只是次要结果，不能用它替代九窗，也不要求复现原五次候选时点。对每次真实接管记录selection/validation/decision/first-influence游标、候选ID、全部验收数值、被取消birth的已消耗训练量。记录birth/retirement/purge（预期仍0）及容量保护，不声称本实验验证了硬删除收益。

输出两组墙钟、CPU、RSS、参数/优化器驻留字节、预测p95、候选/复用额外开销；明确计时范围。交付docs/PROTOCOL031_RESULTS.md、独立registration与run_id目录内comparison/status/逐次reuse/candidate/opportunity/cost紧凑JSON，以及完整artifact链接与SHA。结果回写main并同步所有当前入口后停止；不加A/B、额外种子、阈值搜索或场景重做。
