# 当前交接：Protocol-040 最小双专家动态池

2026-10-03。**仅方案已登记，科学运行未开始。** 本次发布不启动Actions。执行模型收到用户明确交接后，实施[完整指示](PROTOCOL040_TWO_EXPERT_POOL_20261003.md)，运行并发布一条D_pool2结果后停止。

先读：
1. [040指示](PROTOCOL040_TWO_EXPERT_POOL_20261003.md)及[plan.json](../artifacts/ftmoe_online/protocol_040/plan.json)，校验plan.sha256与instructions_sha256。
2. [039结果](PROTOCOL039_RESULTS.md)：单专家一次生命周期已完成，不能重跑。
3. [028失败分析](PROTOCOL028_ANALYSIS_20260922.md)与[034结果](PROTOCOL034_RESULTS.md)：保存记忆不自动产生有用能力；不要恢复旧阻塞复用或只凭相似度决定部署。

## 代码与输入

方案前main=3518e3d5f9c2cca78d0d0467ed45a04f610c851d；039真实科学checkout=600ebb0a1691c72fe7f451bae2c8e8357ad4add4。main没有完整039入口，从含040登记的main创建codex/protocol-040-two-expert-pool-20261003，再从固定039 checkout读取必要内核/依赖并核对plan的Git blob。新增040入口，保留历史实现，禁止从陈旧本地024 checkout重写基线。

036 run36831958978/artifact11147931152提供seed3601的73维issued tape、C实际成熟batch及manifest；039 run36982845152/raw artifact11216467452提供缓存C/B/D_keep/D_039预测。ZIP/逐文件hash均在plan和固定源manifest中。缺源就保全blocked_input，不换数据或重训控制。不加载F。

## 一个任务与交付

唯一新科学序列D_pool2：动态专家总槽2，active最多1、shadow最多1；live更新<=352、shadow<=16、全池总梯度<=368；只有一次第二候选尝试。reuse验证与shadow训练可并行，接受旧专家优先并取消尚未接受shadow。算法细节、取消规则和因果时序以完整指示为准，不在实现时自行调阈值。

先实现生产入口合成测试、一次<=256步零梯度真实前缀，再冻结代码启动一次。C/B/D_keep/D_039缓存重评分不算训练。质量保持以D_keep为固定累计参照，真实两专家运行与局部贡献单列。未触发/候选失败必须交付，不追加试验。

结果写docs/PROTOCOL040_RESULTS.md和artifacts/ftmoe_online/protocol_040/runs/run_<id>/；先上传原始artifact及hash/有效期，再提交紧凑JSON/中文结果到执行分支和main。五个入口README、AGENTS、NEXT_EXPERIMENT_LATEST、PROJECT_CONTEXT_LATEST、本交接页必须指向同一run与终态。结果应可供下一分析模型直接定位到原始预测、完整生命周期、候选验收表和实际成本。

科学/分析/发布状态分开；认证或推送失败只记publication_pending，不重新训练。不得force push。完成本轮或明确无法恢复即停止。历史033–039预算关闭；F、永久删除、新seed/新流、参数搜索及自动下一协议均不在本轮。

可直接交给执行模型：
> 请读取仓库main的AGENTS.md、docs/GITHUB_EXPERIMENT_HANDOFF.md、docs/PROTOCOL040_TWO_EXPERT_POOL_20261003.md及protocol_040/plan.json，按冻结方案实现并执行Protocol-040唯一一条D_pool2实验，校验和分析结果后发布到GitHub main，更新所有交接入口并停止。保留原始证据与失败结果，不扩大预算，不重跑历史控制。
