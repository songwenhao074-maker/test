# 上传计划任务书（交给其他模型审阅/制定计划）

**任务**：把本地 `F:\PreGANPlus-master` 的 Protocol 022 工作上传到 GitHub 仓库 `songwenhao074-maker/FT-MoE`。
**执行者**：当前 DSH agent（只执行，不自行改方案）。
**日期**：2026-09-11

---

## 0. 安全事项（最高优先级，需先处理）

用户在对话中**明文提供了 Personal Access Token**（`ghp_...`）。执行 agent 已：
- 只用它做只读 API 查询与 git 认证；
- **未**写入任何文件、提交、日志或输出；shell 历史中不落盘。

**需要用户立即执行**：到 GitHub → Settings → Developer settings → Personal access tokens 撤销该 token 并重新生成。若该 token 出现在任何聊天记录/工单/截图中，应视为已泄露。
**建议**：改用 SSH key 或 `git credential-manager`，不要每次明文粘贴 token。

---

## 1. 远端仓库现状（已用 token 实测）

| 项 | 值 |
|---|---|
| 仓库 | `songwenhao074-maker/FT-MoE` |
| 可见性 | **private** |
| 默认分支 | `main` |
| 大小 | 132 804 KB |
| 最后推送 | 2026-09-10 14:39 UTC |
| 远端分支 | `main` `2bf244740343`、`protocol-019` `57e856d2a8e7`、`protocol-020` `b56d63f901a7`、`protocol-021` `f0abb5580d92` |

`main` 上只有 3 个提交（2026-09-06）：

```text
2bf2447403  Add online fine-tuning branch inputs (registered artifacts)
d284de56bd  Add FT-MoE core modules and research input data
43650a965f  Initial commit: FT-MoE research codebase (source, docs, assets; experiment artifacts excluded)
```

### 1.1 本地与远端一致性（关键结论）

| 分支 | 本地 SHA | 远端 SHA | 一致? |
|---|---|---|---|
| `main` | `2bf24474034369c5…` | `2bf244740343` | ✅ 一致 |
| `protocol-019` | `57e856d2a8e7a4cc…` | `57e856d2a8e7` | ✅ 一致 |
| `protocol-020` | `b56d63f901a79cf9…` | `b56d63f901a7` | ✅ 一致 |
| `protocol-021` | `f0abb5580d92730e…` | `f0abb5580d92` | ✅ 一致 |
| **`protocol-022`** | `f0abb5580d92730e…`（仅本地） | **远端不存在** | ❌ 需新建 |

**因此本地不需要"补push历史"**：protocol-022 是从 `protocol-021` 的 `f0abb558` 分出的，该点已在远端。

### 1.2 远端 `artifacts/ftmoe_online` 已有目录（protocol-021 分支）

```text
adapted_bwgd2_016/  pilot/  protocol_019/  protocol_020/  protocol_021/  streams/
implementation_verification.json   protocol_015_online.json   status.json
```

→ 上传 `protocol_022/` 与该布局一致，**不冲突、不覆盖**。

### 1.3 `指令/` 目录在各远端分支的文件数

```text
main: 无（404）   protocol-019: 2   protocol-020: 11   protocol-021: 14
```

本地 `指令/` 已跟踪 **14** 个文件（截至 `protocol-021`），Protocol 022 新增 2 个尚未提交：

```text
指令/FTMOE_PROTOCOL022_EXECUTION_PLAN_20260910.md      （方案，上游，不得改写）
指令/FTMOE_PROTOCOL022_PROBLEM_LOG.md                 （问题日志）
```

---

## 2. 待上传内容清单（实测 78 个文件，合计 9.49 MB）

| 分组 | 文件数 | 说明 |
|---|---:|---|
| `artifacts/ftmoe_online/protocol_022/**` | 59 | 证据链（见 §2.1） |
| 根目录 `*_protocol022_*.py` 等 | 10 | Protocol 022 源码（见 §2.2） |
| `test_ftmoe_protocol022_*.py` | 4 | 94 项测试 |
| `simulator/workload/BitbrainWorkloadProtocol022.py` | 1 | cascade_v2 生成器 |
| `stats/Stats.py` | 1 | **修改**（内存缺陷修复） |
| `docs/FTMOE_ONLINE_PROTOCOL_022.md` | 1 | 阶段报告 |
| `指令/FTMOE_PROTOCOL022_*.md` | 2 | 方案 + 问题日志 |

最大单文件 664 KB（`stream.npz`），无大文件风险。

### 2.1 `artifacts/ftmoe_online/protocol_022/` 明细（59）

```text
protocol.json  bootstrap_state.json  source_sha256_initial.json
gate_status.json  problem_log.jsonl  FINAL_REPORT.md
unseen_registry/cascade_v2.json
audit_v2/{offline_reference_v2.json, offline_delta_calibration.py,
          candidate_p015.json, candidate_p025.json, candidate_p035.json, selected.json}
pilot_streams/p0{15,25,35}_seed600_steps1200/{stream.npz, manifest.json, events.json,
          unseen_data_audit.json, task_timeline.npz, task_event_index.json, run_provenance.json}
development_streams/dev_seed600_steps2380/{同上}
learnability_v2/{learnability_p025_seed600_steps1200.json, learnability_summary.json}
fixed_c/s5_capacity_gate.json
fixed_c/s5_capacity_diagnostic/{fixed_capacity_diagnostic.json, gradient_diagnostic.jsonl, run_provenance.json}
fixed_c/budget_sweep/{budget_sweep.json, fixed_capacity_diagnostic.json,
          gradient_diagnostic.jsonl, run_provenance.json}
failed_attempts/{dev_guard_abort_20260910, dev_guard_abort_2400steps_20260911,
          guard_abort_20260911_dev_seed600_steps2381, offbyone_step_semantics_20260911}/
          failure.json 及各失败的 manifest/stream
```

### 2.2 根目录源码（10）

```text
ftmoe_protocol022_core.py                    # task 级审计仪器（U4-v2）
prepare_ftmoe_protocol022.py                 # bootstrap
prepare_ftmoe_protocol022_unseen.py          # 试点采集（3 个概率）
prepare_ftmoe_protocol022_development.py     # 开发流采集（相位结构）
register_ftmoe_protocol022_unseen.py         # cascade_v2 登记
run_ftmoe_protocol022_pilot.py               # 真实 subprocess 退出码 runner
run_ftmoe_protocol022_s5.py                  # S5 容量诊断（5 变体 + 预算扫描）
analyze_ftmoe_protocol022_unseen.py          # U4-v2 门禁
analyze_ftmoe_protocol022_s5.py              # S5 门禁（§9.4）
probe_ftmoe_protocol022_learnability.py      # 可学习性 v2
```

---

## 3. 已排除的内容（并说明理由）

| 排除项 | 理由 |
|---|---|
| `artifacts/ftmoe_online/protocol_022/fixed_c/_smoke/failure.json` | 临时 smoke 运行失败记录，**已删除**，非证据 |
| `artifacts/ftmoe_online/protocol_022/_rss_probe.jsonl` | 内存排障临时文件，**已删除** |
| `artifacts/ftmoe_online/protocol_019|020|021/**` | 冻结历史，本次不改动（仍在各分支上） |
| `artifacts/ftmoe_end_to_end/**` | 同上，已被 `.gitignore` 精确控制 |
| 任何 `*.pt` checkpoint | `.gitignore` 第 224–225 行已排除 protocol_019/020 的权重；P22 未训练新 checkpoint |
| 本地 5 次失败尝试中体积大的中间产物 | 仅保留 `failure.json` 与对应 manifest，便于审计 |

**注意**：`.gitignore` 中有 `artifacts/ftmoe_online/failed_attempts/` 规则（第 159 行）。它只匹配**紧邻 `ftmoe_online` 的那一层**，因此 `protocol_022/failed_attempts/` **不受影响、会被上传**（dry-run 已确认）。若审阅者认为不该上传失败尝试，需明确指示——但协议 §3/§23 要求失败可审计，执行 agent 倾向保留。

---

## 4. 需要审阅者决策的问题

1. **分支策略**：新建远端分支 `protocol-022`（推荐，与既有 `protocol-0NN` 惯例一致），还是并入 `main`？
2. **提交粒度**：单个提交（推荐，消息 `protocol-022: task-level unseen audit, persistence-controlled learnability, fixed-C capacity diagnostic`），还是按阶段 S0/S1/S2-S3/S4/S5 拆多个提交？
3. **是否同时更新 `main` 上的入口文档**？本地 `PROJECT_CONTEXT_LATEST.md` / `README.md` / `docs/README.md` 仍描述协议 020 为当前阶段。执行 agent **未**改动它们（协议 022 由父流程维护入口文档）。审阅者可决定是否单独提交一次入口文档更新。
4. **是否上传 `failed_attempts/`**（见 §3 说明）。
5. **`instructions/` 目录名**：远端用中文目录名 `指令/`（已存在 14 文件），本地一致，无需变动。
6. **是否需要打 tag**（如 `protocol-022-round1`）？
7. **凭据方式**：撤销旧 token 后改用 SSH 还是新 token？

---

## 5. 执行 agent 认为的约束（不得违反）

1. **不改写历史**：不 force push；不改动 `main` / `protocol-019/020/021` 的任何已有提交。
2. **不上传凭据**：token 不得进入任何被提交的文件（执行 agent 会 grep 校验后再提交）。
3. **不修改冻结产物**：`artifacts/ftmoe_online/protocol_019|020|021/**`、原始 v4、P19/P20 checkpoint 一律不动。
4. **行尾**：`.gitattributes` 生效时 git 会把 LF 转 CRLF（dry-run 已提示 22 个文件）。协议内登记的源码 SHA256 是按**本地字节**算的，因此**工作区文件不会被改动**，但远端 blob 行尾将不同——若审阅者要求远端字节级可复现，需要先统一 `.gitattributes` 策略。**建议**：本次照现状提交，并在提交说明中记录该差异，不改哈希登记。
5. **测试门槛**：提交前 4 个测试套件必须全绿（31 + 22 + 25 + 16 = 94），且受保护文件哈希零变更。
6. **push 前先 `git fetch` 并确认远端 `protocol-022` 不存在**（当前实测不存在，需在 push 时复查一次，避免覆盖他人推送）。

---

## 6. 执行 agent 的默认执行草案（待审阅者确认或修改）

```text
1. 复查：git fetch origin；确认 origin/protocol-022 不存在
2. 清理：已移除 _smoke / _rss_probe 临时产物
3. 校验：4 个测试套件全绿；受保护产物哈希 0 变更
4. 校验：git add --all 后 grep 暂存区，确认无 token/无 *.pt/无意外大文件
5. 提交：单提交（或按审阅者指定的粒度）
6. 推送：git push -u origin protocol-022
7. 核验：GitHub API 读回分支与文件数，与清单比对
8. 汇报：给出提交 SHA、远端分支 URL、文件数、排除项
```

**请审阅者就 §4 的 7 个问题给出决定，特别是第 1、2、3、4 项。收到决定后执行 agent 立即执行。**
