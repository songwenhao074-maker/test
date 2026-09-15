# Protocol-024 最新结果审阅（2026-09-15）

## 结论

本轮完成了真实 seed700 / model seed1 的 A/C/D 在线 pilot，已经超出上轮的工程冒烟阶段；但仍未检验动态专家的性能收益。D 的候选创建数、接受数和拒绝数均为 0，最终仍是初始四专家，D 与 C 的已报告预测指标完全相同。应保留此轮负开发结果，进入新版本的机制开发，不进入确认种子。

目标仍是寻找现实可能出现、合理有利于 D 的部署情形。允许 D 额外进行后台候选训练、保存专家记忆，也允许固定 C 受部署更新预算约束；这些差异必须明确记录。当前不需要继续证明“C 永远学不会”或“梯度必须冲突”，也不能把没有启动的 D 当成已经否定了动态方法。

## 核验范围与来源

- 结果分支：`protocol-024-next-round-gpt56@b2410d74bccb44889521549da7c3b7c7b47c0a9e`。
- 正式运行：[Actions 34953810201](https://github.com/songwenhao074-maker/FT-MoE/actions/runs/34953810201)，执行代码 `bae6f53707c1ed7d92f28e99dd3be418f99153de`；job `104330899449` 的所有步骤成功。
- 从该 job stdout 提取完整 comparison JSON，保存于本次审阅的 evidence 目录。只删除混入 JSON 的两种明确 PyTorch warning 行，不重新计算或改写数值。
- 原始 ZIP artifact：`10391150168`。连接器返回下载引用，但本地获取该引用遭遇 HTTP 403；本次未独立重算 NPZ，也未逐行读取完整 lifecycle ledger。136 个窗口中的两个异常位置来自已提交的 problem_log，不冒充本次原始日志重算。
- stream SHA256：`468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`。4980 scored intervals + 1 guard；原生成约 11039 秒，后续仅修复收尾审计，已有不可变数据应继续复用。
- resume 分支 `5669087` 与结果分支已分叉；其恢复文档是旁支历史，不是比正式成功 pilot 更新的性能证据。

## 数值说明

| 同一 raw_next_fault 目标 | A | C | D |
|---|---:|---:|---:|
| 全程 AP | 0.510442 | 0.667584 | 0.667584 |
| 全程 Recall | 0.498050 | 0.678463 | 0.678463 |
| 全程 FPR | 0.039240 | 0.038141 | 0.038141 |
| future-positive resource macro-F1 | 0.764456 | 0.839168 | 0.839168 |
| raw same-host onset AP | 0.105217 | 0.160114 | 0.160114 |

C 相对 A 全程 AP 增加 0.157142；六个复现窗口的等权 C AP 均值为 0.656544。D−C 为 0，正增益窗口 0/6。R1 是 C/D 最弱 response law，AP=0.326094。阈值指标和 AP 不可混为一个目标。

固定 C 的 u1 全程 AP=0.720371，高于 u4=0.667584；但首次暴露 first100 均值 u1 比 u4 低 0.007019，因此依旧按已注册规则选择 u4。该选择对本轮有效，却不意味着 u4 在所有部署目标上最好。“p95 有限”也不等于满足一个真实时延预算。后续宣称超越其他方法时，应保留 C-u1 / C-budget 对照。

## 新发现：压力与 persistence 基线具有更晚的信息

comparison 中 current_pressure 全程 AP=0.798478，高于 C 0.130894，且九个 first100 窗口都高于 C；六个复现窗口等权均值 0.777349。

**不能直接据此断言 C 输给同信息部署基线。** 代码时间线不一致：

1. `prepare_ftmoe_protocol024_stream.py:249` 起采集当步模拟前的 host_features/demands；随后形成 schedules。
2. `:292` 执行 simulationStep；`:304–316` 才生成 post_totals、overload_ratio 和 raw_labels。
3. `LifecycleProtocol024Session.step` 先预测 p[t]，之后才读取 raw[t]；预测 raw[t+1]。
4. pilot 和 probe 的 current_pressure 直接使用 overload_ratio[t]，persistence 直接使用 raw[t]。两者取到了该预测时点尚未揭示的当步结果。

因此这些数值应保留为“额外获得当步模拟结果的参照”，新加同一预测时点可得的压力分数和延迟标签 persistence。不要静默把原始 0.798478 改名为可部署基线。还须核查 schedules[t] 是否只是当时已作出的调度决策，而非执行结果。

raw-history logistic AP=0.866506、frozen-z AP=0.837763 说明可用输入有预测信号，不能证明在线有限样本更新已经充分，也不能证明三个 response law 必须分别用三个专家。frozen-z probe 迭代达到 300，stdout 有收敛警告；这是诊断结果，不是表征性能上界。

## 动态机制为什么没有启动

冻结阈值为 0.7022458374191777，监控 136 个非重叠 32 步窗口。仓库 problem_log 记录只有 cursor 1529、2169 越界，均未连续两次，因此根本未创建候选。这不是“候选训练失败”或“验收太严格”的证据。

更深的限制：

- 校准用前 600 步、stride=1 的重叠窗口；运行时 stride=32，并要求连续两个越界。569 个高度重叠校准窗口不是 569 次独立观测。前缀还混合全正常 F0 与正在适应的 R1，绝对 p99 可能吸收启动误差；需分解损失才能确认原因。
- 即使切换后马上出现持续异常，两个完整窗口 64 步、训练 64 步、验证 32 步已约 160 个成熟 interval；还有标签延迟、窗口对齐，以及 10 次 live update 的 ramp。u4 时 ramp 约 40 个 interval。该时间预算与 180 步复现、first100 主指标不匹配。这是代码推导，不是本轮测得的激活延迟。
- 触发依据是 detection CE、positive-only resource CE 与 ranking 的混合损失；正常/故障比例变化也能改变它。它既没有显式 novelty 决策，也没有真正调用休眠专家复用。
- 当前退休和硬删除关闭；接受后所有 active 专家继续共同更新。即使下轮只增加专家，也未必保存了可在复现时快速调用的旧专长。
- 验收的 normal_probability_allowance 限制正常样本平均异常概率，不能保证固定阈值 FPR 不增加；也没有独立旧知识回归验收。
- 父专家按累计全历史路由质量选择，可能偏向长期占优专家，未必适合当前模式；这只是需诊断的选择偏差，不能推断已经造成失败。
- D 用时约 125.31 秒、C 约 80.09 秒（约 +56%），却没有 shadow compute。controller 每步多次深拷贝全部 history/events，值得 profiling；不能未经测量就认定全部开销来自深拷贝。

## 工程与交接问题

此前 evaluator 接线、ID 有间隔恢复、optimizer 保留、目录隔离等已有工程修复证据；下一轮对修改链路回归验证即可，不从头重复所有旧门禁。

status/comparison 中“candidates were causally trained/qualified, but no candidate passed”与 candidate_created=0 矛盾。新运行器必须按事件计数输出准确原因；本轮文件保持原样并由本审阅纠正解释。

main 原先仍停在 952b693，根目录入口还写“尚无动态性能结果”。本次将默认入口推进到最新结果及新指示；保留旧分支、旧数据、旧结论。新完整产物主要在 Actions，必须尽快归档，不能让下一模型只能依赖会过期的 ZIP 和口头摘要。

## 下一步

执行 [next_round_v2 指示](PROTOCOL024_NEXT_DIRECTIVE_20260915.md)。先复用同一 stream 做可预算的主动候选提议与准确机制记录，再实现专家保留、退役和快速复用；以可观测事件决定晋级，不靠人为切换时刻强行激活。确认种子 701–703 和保留测试种子 201–205继续封存。
