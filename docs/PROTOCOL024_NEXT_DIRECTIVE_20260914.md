# Protocol-024 下一轮实验指示：修复实际链路，交付首个动态 pilot

日期：2026-09-14。基于 `674c6267dc9e3aec268c1da186278f77d45224f5` 及 [本轮审阅](PROTOCOL024_REVIEW_20260914.md)。

**本文件是当前下一轮执行指示。目标是在合理有利情形下实测 D 的优势。完成必要工程修复后，应继续产出一次生命周期开启的 A/C/D 开发结果；不要又只提交容器单测和一份“尚未开始”的计划。** 若运行环境确实阻塞，报告精确命令、错误和已有产物，不虚构性能。

已有成果复用：15 项单测通过；lifecycle-off C/D 在 1140 步、各 285 更新、825 个异常 host-step 上容差内等价。不要重建另一套与 C 不同的旧 EAGate D。旧强负梯度、capacity plateau、C 必须遗忘等条件不再是实施 D 的必要门禁。

## 0. 工作范围与不可覆盖项

- 以包含本指示的最新 main/continuation 为基点建新工作分支。
- 新产物统一放 `artifacts/ftmoe_online/protocol_024/next_round_v1/`，已有 run 目录拒绝覆盖；重试用新 run_id。
- `protocol_023/**`、既有 `protocol_024/continuation_20260914/**` 与 `review_20260914/**` 全部保留；注册基点及旧输入/checkpoint SHA。
- replay seed700 为开发；model seed1 为最小 pilot。701～703 保留确认用途，本轮不使用；201～205 不生成。
- mode/phase/event ID 只能用于离线分段审计，不能进入模型、触发器、路由或候选资格判断。共享时钟与已观测遥测可用，预知切换时刻不可用。
- 同时只运行一个模型/模拟器进程，CPU 3 线程、interop 1，沿用交接中的 RAM/磁盘保护。

## 1. 先修实际 runner 与输出边界（必须完成）

### 1.1 隔离协议与时间线

覆写/重构 `save_checkpoint`、`phase_metrics`、`probe_scores` 及其依赖，输出路径只从 P24 run 配置派生，时间线从本次 manifest 读取。不要让 P24 再使用 P23 的全局 CKPT_DIR、CHECKPOINT_NAMES、PHASES 或硬编码 DEV_STREAM 作为本次实验定义。

验收：在独立临时输出目录中实际保存 A/C/D 产物，检查 protocol、arm、run_id、stream SHA 与路径；检查 023 文件集合及 SHA 前后不变。测试不允许真的覆盖旧目录后再恢复。

### 1.2 指标只有一份实现，必须检查最终 JSON

统一调用 `ftmoe_protocol024_eval` 中的二维时间轴 onset 和 positive-only resource F1；删除/转发 session 中重复实现。未知当前/未来标签排除；缺少未来的尾行排除；任何非法 shape/NaN 应显式报错。

固定回归例：t=301/host0 唯一 onset，t=300/host0 最高分，实际保存阶段 JSON 的 onset AP 必须为 1.0，而非本轮复现的 0.0625。加入 −1 标签、h>1、阶段尾部和无正例窗口。无正例/无负例 AP 返回 null 并记录原因，不填 0。

完整 run 的 `finish()` 后断言所有 scored 行已结算、无 −1，并核对 raw/tolerance/未来目标的下标。不要仅测试最后两行已置 True；还要故意制造较早未结算行，确认完整性检查失败。

已有 AP 实现对相同分数按输入次序打破平局；正式新指标应按相同 score 成组累计，或统一使用已验证的 average_precision_score，并测试全相同分数得到 prevalence、样本排列不改变 AP。旧已存指标不追改。

## 2. 生命周期必须能训练、恢复并回收

### 2.1 状态与优化器

以稳定的 `(expert_id, tensor_name)` 映射保存 optimizer moments；出生时只为新增 live 参数注册状态，不清空已有专家的 Adam moments。shadow optimizer 与 live optimizer 分离；激活时明确是否迁移候选 moments，固定一种规则并记录。

active、dormant、shadow 参数可训练性按角色设置，避免调用通用 `set_trainability()` 把所有 learner.* 又变成可训练。休眠参数不得继续被梯度或 weight decay 更新；清理残留 grad，保留用于再激活的状态。

### 2.2 恢复合法 ID 空缺

恢复时直接构造 snapshot 里实际存在的集合，不从 0 生长到 next_id。next_id 单调递增且不回收，不要求连续。

必测序列：创建4→拒绝4→创建5→激活5→保存/恢复；多个连续拒绝；混合 active/dormant/shadow；partial ramp；删除后再出生。检查 tensor、顺序、ramp 与行为 hash 一致。

### 2.3 行为 hash 与完整 resume

行为 hash 至少包含 live tensors、active ID 有序列表、ramp、拓扑版本；训练恢复另包含 dormant/shadow、两类 optimizer、buffer、cursor、pending predictions/raw_seen、RNG、loss/novelty EMA、专家年龄、候选 ledger 和预算计数。

用不中断与中断恢复两条相同轨迹比较**下一次预测、下一次更新后的 tensor、后续事件决策及指标**。CPU 确定性路径争取完全一致；其他路径如只能容差比较，事先写清容差，不比较仅一个 forward 就宣称 full resume。

### 2.4 明确 retire 与 purge

- `retire`：逐渐退到 ramp=0 后休眠，保留同 ID 的知识；不称总内存释放。
- `reactivate`：同 ID 恢复，ramp 从 0 开始。
- `purge`：从 dormant 真正删除专家、router 及 optimizer state；该专家不可再“原样唤醒”。

容量定义显式包含 active+dormant+shadow，总数上限初始取 8。满额时先比较休眠复用；无匹配且确需新专家时才 purge 符合规则的 dormant 专家。检查 purge 后重新获得槽位，长期出现超过 8 个模式时 next_id 仍正常增长。

退休和删除决策必须基于已观测使用率、年龄及因果验证；不能读取“这个模式今后不会再来”的生成器信息。只支持休眠的版本可以先参与最小 pilot，但结论必须标注 `hard_delete_enabled=false`，删除机制留待后续退出阶段验证。

### 2.5 连续与稳定路由

保留当前 ramp 加权归一化 mixture 路线，先解决数值稳定性：ramp=0 的专家在 softmax 前 mask，ramp>0 在 log 空间加 log(ramp)。至少保留一个正 ramp 活动专家。

测试训练后的非零专家，含极大候选 logit、退役/再激活和整个 ramp 过程。报告每次拓扑动作前后的 logit jump，不能只在零输出初始专家上证明连续。若改成另一个加法结构，必须重新做 C/D lifecycle-off 等价性，且固定容量对照采用同一新路由。

## 3. 最小因果策略：先能产生可审计的事件

本轮不做复杂多触发器大搜索。优先用 matured supervised-loss 上升触发诊断，结合已观测 z 特征距离选择休眠复用或新候选；router entropy/margin 先作诊断项，避免把 softmax 高熵直接等同于新机制。

### 3.1 校准与冻结

在开发场景的平稳训练前缀上校准，包含正常与异常样本；不要只用零故障 F0 定义低损失参考。候选起点：32 个**不同 interval** 的损失窗口、连续两个异常窗口、平稳损失分布的 99% 分位作为阈值。这些是开发默认值，不是已验证规律。

z 统计按训练前缀归一化；休眠匹配使用有足够支持的路由加权 z centroid，先固定距离规则；router weight row 不是天然的输入 centroid。阈值从平稳与重复业务开发前缀校准，写入 `lifecycle_config.json` 后不在同次 pilot 中修改。最多两轮预先登记的校准配置，第二轮另开 run_id，不能从最终 D-C 曲线不断倒调阈值。

### 3.2 候选训练与 qualification

- 同时最多一个 shadow，shadow 完全不进入 live forward。
- 训练样本必须已物理成熟。将 minimum sample 明确为 **64 个不同 interval**，不是 64 个高度相关 host-step；若不足，只延后，不伪造计数。
- 完成候选训练后冻结候选，使用随后 **32 个不同 interval** 做资格审查。每个验证 interval 在揭示目标前，记录当时 live 和“候选部署后的完整模型”的配对预测。
- 验证样本不得反流入该候选训练；候选不能先看标签再补预测。live learner 可按共同在线规则继续更新，但比较必须使用该时刻事先记录的两个模型版本。
- 资格不是比较裸专家 vs 完整模型，也不是在 ramp=0 时比较两个相同输出。预演注册的完整部署配置；默认比较 paired supervised-loss 相对改善至少 1%，同时检查独立旧业务校验样本上的退化及正常误报上限。
- 资格阈值、最大等待长度和 cooldown 在运行前注册；无足够正例时报告无法资格判断，按预设超时拒绝，不强制激活。
- candidate training、preview inference、validation、birth/retire/purge/reactivate 全部计数/计时。C 的基本更新机会与 D live 完全一致；D 额外开销单列，后续补 C-budget。

先用受控 replay/toy 测试主动制造事件，验证 ledger、shadow 隔离、参数更新和恢复；这种强制事件测试标为 test-only，不能用于性能结果。再在原 023 流上做短的 lifecycle-on 运行调试因果链，允许没有优势，不据此宣称新场景成功。

## 4. 首个新场景只做三个 response laws

不要第一轮就收集完整 6～8 模式大流。三个候选业务律：

| 模式 | 资源响应 | 要保留的可观察线索 |
|---|---|---|
| R1 短计算任务 | CPU 短脉冲后快速释放，RAM/Disk 变化较小 | 过去若干步的脉冲长度、下降趋势 |
| R2 持续推理/工作集增长 | CPU 持续、RAM 滞后积累 | 历史上升斜率与 CPU/RAM 的滞后关系 |
| R3 后台写回 | Disk/队列逐步积累，随后释放，CPU/RAM 有可见伴随变化 | 写入积压与排空的历史形状 |

使用真实 Bitbrain 底流与模拟器资源逻辑生成，不直接按模式改标签。接纳首步需求保持可放置；变化不能靠大规模拒绝部署制造类别差异。记录部署/迁移拒绝率、真实事件数、正例率、长度与峰值。

**模型可見信息相同却随机反转标签不构成可学场景。** 需要从不含 audit ID 的原始 12-step 遥测/调度上下文辨别结局；同当前负载附近，历史应提供额外信息。

开发初始时间线固定为：F0=300，R1/R2/R3 首次各1200，然后 R1→R3→R2→R1→R2→R3 各180，总4980 scored intervals，另加未来目标需要的 guard 行。首次长、复现短是本轮有意采用的、对知识复用有利的部署情形；不代表普遍工作负载。

## 5. 新场景的目标必须与 response law 对齐

**新性能 pilot 推荐并注册 `target_mode=raw_next_fault`：** 用 t 时刻可观察输入预测同一 host 的 `raw_label[t+1] > 0`，资源分类目标为该未来 raw_label 的 CPU/RAM/Disk。C/D 同步改监督目标，初始 backbone 保持同一冻结权重；所有臂在同一预测目标上评分。

这是新预测任务，不能与旧 tol1 当前检测 AP 直接比较。旧 1140-step lifecycle-off 检查继续使用旧目标，仅用于回归。预测→揭示→结算顺序维持因果；在保留一 interval 标签发布延迟的约定下，t 的未来目标最早在 t+2 用于训练。明确区分内部数组存在与实际发布可用时间，controller 不得直接访问 raw_label[t+1]。

onset 评分仍用 probability[t]、原始 raw_label[t] 与 raw_label[t+1] 构造，不能对已经前移的训练目标再次 shift。时间切分的 probe 训练/验证窗口之间至少留 12+h 个 interval 的隔离，避免共享输入历史或未来标签；分段交界与最终 guard 的跨界评分规则在 manifest 固定。

anchor 也需要同一目标，必须有可验证的相邻未来标签/guard。不要把旧 tol1 anchor 标签冒充未来标签；若旧 anchor 无法重建，使用所有方法共享的新场景开发前缀/独立已观测历史，记录来源与无泄漏索引。固定主干蒸馏可保留为正则并标明旧模型是当前检测 teacher，不把它当未来真值。

先用同一按时间划分的前缀训练/后缀验证，比较 allowed raw/history、冻结 64-D z 与简单 persistence/current-pressure baseline。按同一个未来目标评估，不用 mode classification 的准确率替代预测可学性；报告不使用时间历史的消融。

若 raw/history 有稳定增益而 z 丢失信息，允许给 C/D 同时增加相同的轻量时序特征。不得只给 D；该变更单独登记并重跑 lifecycle-off 等价检查。若 raw/history 本身不可学，最多修订一次 response law/可观察前兆并重新生成开发流，不扫描确认种子寻找有利结果。

## 6. 更新预算与首个 A/C/D pilot

默认 live optimizer AdamW、lr=1e-4、batch=32、replay=64 intervals、gradient step/opportunity=1。初次长驻留已用于解决冷启动，不再沿用 420 步首次块。

先对 C 做两个有界开发预算：update_every=4 与 update_every=1，其他设置不变；检测预算是否足以在首遇时适应，并测总耗时/p95。按预先记录的实际部署预算选择可执行配置，用同一配置比较 C 与 D live。这里是在确定合理优化强度，不是要求 C 先遗忘才允许 D。anchor/distill 暂保留0.25/0.10；不同时搜索学习率、保护权重、场景和触发阈值。

A 冻结、C 固定4专家、D 起始同4专家/总上限8，在相同新流上严格预序贯执行。A 在新目标下是原冻结部署参照，不冒称它是经过适配的最强预测基线。D 额外 shadow/验证预算显式单列，首轮允许合理不对称，不声称严格等算力胜出。

**本轮必须产出生命周期开启的真实运行，而不只交付 `D_off`。** 记录：trigger 次数、candidate 数、训练/资格支持数、接受/拒绝原因、birth/retire/reactivate/purge 数、每步活动/存储专家数、模型版本和预算。不要求为了达事件数强行触发；若全程零出生/零复用，明确写 `lifecycle_not_exercised` 并按日志区分原因。

## 7. 预先固定主指标和解读

- 主指标：新目标下，每个注册切换后前 W=100 intervals 的 future-fault AP，按切换等权平均；同时给每次配对 D−C。分别列首次接触与六次再现，不能只报最有利的一组。
- 补充：全程 AP、最差模式、raw same-host onset AP（当前正常 host 子集）、future-positive resource macro-F1，以及在冻结阈值下的 FPR/召回。
- 预设无正例窗口报告 null；记录有效切换覆盖率。若任一臂同一窗口不可评，所有臂按同一规则处理，不单独删除不利窗口。预注册按相同模式合并再现窗口的辅助读数，避免看到结果后扩窗口。
- 报告 persistence/current-pressure 基线，避免把已有故障持续性当成新预警知识。
- 开发信号参考：六次再现 mean(D−C)≥0.03、至少4/6为正、正常误报不恶化超过0.01；这是进入更大消融的操作标准，不是统计证明。正例不足时不能硬判达标；按新开发 run 扩充时间/重复次数并公开修订。
- CI 用按时间/事件块的配对重采样，不把16台主机每个时刻当独立重复。单 replay/model seed 只能给开发证据。

决策：有优势→固定8专家与 C-budget 以及 D-no-birth/no-reactivation/no-retirement；生命周期不工作→修触发/资格的具体缺口；生命周期工作但 D≈C→看在同一测试群体上的专业化差异，不无限提高容量；D更差→先查实际部署前后损失、路由突变、选择性更新与额外预算。所有失败结果保留。

删除机制后续另加短期业务退出与超过8个累计模式，验证 purge 对空间/更新开销的作用；本轮三模式不足以证明删除必要。确认种子701～703留到方法、目标、生成器和超参数全部冻结以后。

## 8. 本轮交付清单

下面为**需要实现/生成的新文件约定**，不是声称现有仓库已存在这些 runner：

```text
artifacts/ftmoe_online/protocol_024/next_round_v1/
  registration.json             # base commit, targets, units, seeds, thresholds, timeline
  integrity_gates.json           # actual JSON metrics, namespace, rejection-gap restore, resume
  data/manifest.json             # arrays/hashes, response laws, observed inputs, guard policy
  probes/learnability.json       # common targets/splits, raw/history/z, persistence
  budget/comparison.json
  lifecycle_config.json
  runs/<run_id>/arm_A|arm_C|arm_D/
    predictions.npz
    summary.json
    checkpoints/                # resumable P24 states, never P23 global paths
    lifecycle.jsonl              # D only; cause/effect and pre-label validation ledger
  comparison.json
  status.json
docs/PROTOCOL024_NEXT_ROUND_RESULTS.md
```

status 必须区分 `engineering_passed`、`new_stream_generated`、`lifecycle_on_run_completed`、`lifecycle_exercised`、`development_signal`、`confirmation_run=false`。不能用15项单测通过替代后三个实验状态。

结果报告先回答：D 是否真的出生/复用/回收过；相对 C 哪些切换改善/变差；代价多少；下一步是扩大确认还是修哪个具体原因。正文保持简短，证据放上述紧凑产物中。更新根目录 `NEXT_EXPERIMENT_LATEST.md` 与 main 的当前入口，避免只有分支有新结果但默认入口仍指向旧计划。
