# 历史实验索引

当前唯一入口：[Protocol-031 revision003短版新场景](PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)及[当前交接](GITHUB_EXPERIMENT_HANDOFF.md)。历史指令不自动构成当前任务。

| 内容 | 历史用途及边界 |
| --- | --- |
| 023 D_eligible/STOP及024各版 | 保留早期设计、资格检查及生命周期证据；旧Python模块可能仍是依赖 |
| 025/026 | 物理业务规律、初始化、方法与旧场景来源；旧登记不覆盖新031场景 |
| 027 | revision002冻结旧数据与normal-only guard；数据和失败记录不可改写 |
| 028 | 记忆保护D/C试跑；旧场景D未胜C |
| 029 | 旁路诊断未通过跨运行一致性门禁，历史判定保留 |
| 030 | 同机旁路无干扰通过，产物离线恢复；不追溯修改029 |
| 031 revision001 | 未执行的旧数据非阻塞复用方案，已被用户新要求替代，见[归档](PROTOCOL031_SUPERSEDED_OLD_STREAM_DIRECTIVE_20260922.md) |
| 031 revision002 | 历史长版计划（总15469步），已由短版替代；见[修订记录](PROTOCOL031_PLAN_REVISION_002_20260922.md) |
| 031 revision003 | 当前唯一任务：总9869步稀有回归场景＋C/D各一次；见[修订记录](PROTOCOL031_PLAN_REVISION_003_20260926.md) |

禁止按协议编号批量删除旧代码、源数据、checkpoint或恢复工具。新实验使用031专用入口/注册/数据目录，旧实验只在明确复现时运行。
