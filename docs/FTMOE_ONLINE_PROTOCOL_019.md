# Protocol 019 — Common-Input Repair + Same-Domain Initialization + Drift-Aware Online FT-MoE

> 状态：**已收尾（2026-09-06，用户拍板接受 S7 null 结论，不再执行 S8）**
> 依据：`指令/FTMOE_ONLINE_TUNING_REVIEW_AND_SOLUTION_PLAN.md`（下称 *The Plan*）
> 仓库：`songwenhao074-maker/FT-MoE` @ `F:\PreGANPlus-master`（git main）

## 0. 协议身份与版本登记

```json
{
  "protocol": "019",
  "base_checkpoint": "protocol014_v4_seed1_epoch21",
  "input_contract_version": 2,
  "normalization_version": 2,
  "graph_semantics_version": 2,
  "online_optimizer_version": 2,
  "dynamic_expert_version": 2
}
```

登记文件：`artifacts/ftmoe_online/protocol_019/protocol.json`

## 1. 不可修改的既有资产（immutable）

- `backup/ftmoe_protocol014_accepted_20260905/`
- `artifacts/ftmoe_online/pilot/`（Protocol 015 全部 A/B/C/D runs）
- `artifacts/ftmoe_online/adapted_bwgd2_016/`（Protocol 016）
- `artifacts/ftmoe_online/distribution_audit_015/`
- `artifacts/ftmoe_online/capacity_017/`（Protocol 017，未达完成态：无 result.json）
- Protocol 018（google2011）按 The Plan §15 定位为 real-trace-driven simulated overload；
  当前仓库内未见 `artifacts/ftmoe_online/google2011_018/` 与 `result.json` → **不得称 017/018 已完成**。

014 v4 checkpoint（仅读取）：

- `artifacts/ftmoe_end_to_end/runs/physical_lr0003_e30/v4_seed1/checkpoints_by_epoch/epoch021.pt`
  sha256 `e3513575ec26fc94a9f3c1d877a7119b21bacc97d7001fa02d18278aa316720b`
- 解析入口：`run_ftmoe_online.py::resolve_checkpoint`（seed ∈ {1,2,6} 读
  `diagnosis_reassessment/physical_lr0003_e30/v4_seed{seed}.json` 的 `best_checkpoint`，
  再与 `comparison_014_complete.json["source_sha256"]` 核对）

## 2. 阶段流程（The Plan §3，任一阶段失败即停止后续阶段）

```text
P19-S0 归档与协议登记            <- 本文档
P19-S1 共同输入修复 (normalization v2 + graph identity v2 + 测试)   [完成]
P19-S2 Cold-start compatibility（data-only audit → Frozen A gate）  [完成，gate FAIL → 进 S3]
P19-S3 Same-domain common offline adaptation（必要时，warm-start 014 v4）  [完成：adapted_v4 产出；gate 仍 FAIL → 停止点 P08，用户批准以 adapted 起点继续]
P19-S4 Online optimizer stabilization（A/B/C）  [完成：S4 scheme 实现+测试通过；gate PASS @ lr=1e-5（P09）]
P19-S5 Safe dynamic expert（D；主比较 D−C）  [完成 v2 gate + Test 7–9（5/5）；dev303 D−C=0、专家不增长]
P19-S6 Stationary long-stream validation  [完成：seed304 × 2000 interval，D 恒 4 expert、D−C=0、无遗忘 → PASS]
P19-S7 Natural / controlled drift validation  [完成（用户授权 controlled drift）：seed305×2000 相位流；漂移真实但 D−C=0 → dynamic gate FAIL（P12），机制不可触发原因已归档]
P19-S8 5 seeds × 3 streams final confirmation  [未开始（P12 后需用户方向）]
```

## 3. 阶段状态

| 阶段 | 状态 | 产出 | 日期 |
|---|---|---|---|
| S0 | 完成 | 本文档 + protocol.json + 基线 hash 快照 | 2026-09-06 |
| S1 | 完成 | input contract v2、normalization v2、graph identity v2、8 项测试通过、legacy 回归 9/9 | 2026-09-06 |
| S2 | 完成（gate FAIL） | dev stream seed303 + distribution.json + Frozen A run + 判定 | 2026-09-06 |
| S3 | 完成（adapted_v4 产出；重测 gate 仍 FAIL → 停止点 P08） | adaptation_data v1 + s3/adapted_v4_seed1 + S3_adaptation_result.json | 2026-09-06 |
| S4 | 完成（scheme s4 测试 4/4；A/B/C 对照；gate PASS @ lr=1e-5，P09） | runs_s4/{A,B,C}_model1_replay303_adapted(+_lr1e-5) + S4_online_stability_result.json | 2026-09-06 |
| S5 | 完成（gate v2 实现，Test 7–9 通过 5/5；dev303 D−C=0、专家数不增长；S5_dynamic_expert_v2_result.json） | ftmoe_online_s5.py + test_ftmoe_protocol019_s5.py + runs_s4/{C,D}_..._gatev2_lr1e-5 | 2026-09-06 |
| S6 | 完成（stationary seed304 × 2000：D 恒 4 expert、D−C=0、C/D 不劣于 A 且无遗忘 → PASS；S6_stationary_result.json） | runs_s6/{A,C,D}_... + stationary_streams/seed304_steps2000 | 2026-09-06 |
| S7 | 完成（controlled drift seed305×2000：相位漂移真实存在但 D−C=0、专家恒 4、unmatched=0 → dynamic gate FAIL（P12）；S7_drift_result.json） | PhaseAdaptedBWGD2 + drift_streams/seed305_steps2000 + runs_s7/{A,C,D} + s7_analysis.json | 2026-09-06 |
| S8 | 未开始（P12 后需用户方向：(a) 接受 null 结论收尾 / (b) 机制 v3 重设计后重跑 S7 / (c) 仍跑 5×3 全矩阵） | — | — |

## 4. S1 工作项（The Plan §5–§6、§18、§19 Tests 1–4）

1. 输入契约：新增 `recovery/PreGANSrc/src/ftmoe_input_contract.py`
   （HostFeature 7 列语义 + manifest 字段约定）
2. 归一化 v2：新增 `recovery/PreGANSrc/src/ftmoe_normalization.py`
   - `safe_scale()`：training scale ≈ 0 或低覆盖 → 同硬件组同特征统计兜底
   - 硬件组：host 0–7（RPi 4GB）/ host 8–15（RPi 8GB）
   - 覆盖判据：`count_nonzero < 12` 或 `max < 1e-4 * same_group_p95`
   - 禁止使用 online label / future statistics
3. 图语义 v2：`ftmoe_ablation.py::ScheduleGraphEncoder` identity-aware migration
   - 迁移边只由 `creation_ids` 同身份 + `before_placement → argmax(proposed_schedule)` 构造
   - occupancy 只累计 valid slot（`creation_ids >= 0`）
   - 兼容路径：`graph_context=None` 走 legacy
4. 数据接口：Replay.window 增加 identity/before/valid 输出（新 runner 实现，不改旧 runner 语义）
5. 测试：Test 1（归一化零覆盖）、Test 2（slot 替换 ≠ 迁移）、Test 3（真迁移 = 迁移）、
   Test 4（空 slot 不计 occupancy）必须通过；Test 5–10 随 S4/S5 功能补齐
6. 新开发流（300 interval）+ data-only 审计 + Frozen A

## 5. Gate 定义（The Plan §21，预先固定）

- Data gate：normal ≥5000 host-samples、CPU/RAM/Disk ≥100（长流）
- Cold-start gate：Frozen A PR-AUC ≥ 0.60 且 FPR ≤ 0.15（否则进 S3，禁止反向调 scenario）
- Online stability gate：anchor F1 drop ≤ 0.03；second-half F1 不低于 Frozen A 超过 0.03
- Dynamic expert gate：D−C mean > 0（15 paired runs）；drift 场景 D adaptation lag < C

## 6. 基线源码 hash 快照（S1 修改前，2026-09-06 采集）

```json
{
  "run_ftmoe_online.py": "a306d439646067d4e3034588186e45fe233ff8d1ba03657fbaa302cc1ed28e3a",
  "analyze_ftmoe_online.py": "0b509b05d96402995d9347d22d09f21fa4b542f9f5b42ecf432cbf7c1d0696f4",
  "train_ftmoe_ablation_existing.py": "84c2b384ff18fbea02ce29f43e0ad1f4086b75bc2e6233cbff09483bb3622e1d",
  "train_ftmoe_end_to_end.py": "b6c9c09d990ba02d31fa0c13326f7de45844c6926fcc097a40e8a0bcc723863a",
  "recovery/PreGANSrc/src/ftmoe_online.py": "9c180412230c7d6e7983e4ddfd701b046731cd7fa05c3be0155bee64e786001e",
  "recovery/PreGANSrc/src/ftmoe_ablation.py": "673ac6212a8a7b59771d86013e0a1fe9036551e1011a4078e3e9c0194b21c7c0",
  "recovery/PreGANSrc/src/ftmoe_end_to_end.py": "44b4f48979bf228994f9a088e18ccc0e534d1d97e37c28c17adde0417a65f6a0",
  "prepare_ftmoe_scenario.py": "fcc4b1baa3c0489fa78b49d5f1a3b1871a2f29afc9ac7d28c2d2da2774cfc3dd",
  "prepare_ftmoe_online.py": "2bdc9d5bf187bc87d9cf7301a70c2f7e23c686f924123044e05f265c1f575de4"
}
```

## 7. Execution Log

- 2026-09-06 S0：登记本文档 + artifacts registry + 指令/FTMOE_PROTOCOL019_PROBLEM_LOG.md。
  发现 017/018 无 result.json、无 google2011_018 产物目录、所有可用 python 环境 torch 均为 CPU 版，
  详见问题日志。
- 2026-09-06 S1：新增 `ftmoe_input_contract.py`（7 列语义 + host group 契约 v2）、
  `ftmoe_normalization.py`（zero-clamp ≤1e-8 / count_nonzero<12 / max<1e-4·group_p95 → 同组中位兜底；
  归一化绝对值报警 50）。实测 016/019 流 v1 归一化 abs max 5e11→6e11 → v2 ≤3.0（alarm off）。
  `ftmoe_ablation.py::ScheduleGraphEncoder` 增加 `graph_context={'creation_ids'}` 路径：
  migration 只计同 identity、occupancy 只计 valid slot；`graph_migration_and_occupancy()` 供测试/审计；
  `FTMoEAblation.forward`/`OnlineFTMoE.predict_online` 增加 `graph_context=None`（legacy 逐位不变）。
  测试 `test_ftmoe_protocol019_s1.py` 8/8 通过（Tests 1–4 + 健康窗 legacy/v2 parity + 模型接口）；
  `test_ftmoe_online.py` 9/9 通过（无回归）。代码 hash 快照：`artifacts/ftmoe_online/protocol_019/source_sha256_S1.json`。
- 2026-09-06 S1/S2：新 dev stream `dev_streams/seed303_steps300`（adapted-BWGD2 016 契约、seed 303、
  stream_sha256 `ff0ea85e…`，RSS 0.61 GiB）。data-only audit → `distribution.json`；
  Frozen A（run_ftmoe_protocol019.py，A_model1_replay303）完成。
- 2026-09-06 S2 verdict（`S2_frozen_A_result.json`）：**冷启动 gate FAIL** —
  full F1 0.2340 / PR-AUC 0.1863 / FPR 0.1753；second-half F1 0.1341 / PR-AUC 0.1370；
  ROC-AUC 0.9054（排序尚可，校准/阈值失配为辅因）；离线 anchor F1 0.9051 全程不变。
  → **停止 scenario 调参（P06），进入 S3：same-domain common offline adaptation**。
- 2026-09-06 S5：v2 dynamic gate（EMA gate / centroid+parent-clone birth / ramp / retire+reactivate）
  实现于 `ftmoe_online_s5.py`；Test 7–9 + clone/smoke 5/5 通过（P10 记录两个开发期缺陷及修复）；
  dev303 上 C-v2≡C-v1（fixed-mode 等价）、D≡C 且专家数不变 → 平稳期不增长验证通过
  （`S5_dynamic_expert_v2_result.json`）。
- 2026-09-06 S6：stationary stream seed304 × 2000（`stationary_streams/`）；A/C/D 全跑
  （2000 steps，C/D 200 updates，exposure 上限 3 无违规）；A anchor 恒定 0.90486；
  C/D anchor ≤+0.005 无遗忘；C/D second-half F1 0.2687 > A 0.2632；D 恒 4 experts、D−C=0
  → **S6 gate PASS**（`S6_stationary_result.json`）。
- 2026-09-06 S7 待用户方向（natural drift = Google2011 数据集下载授权；controlled drift =
  VM 组相位切换需 simulator 改动批准），见 §8 S7 备注。

## 9. 最终结论（Protocol 019 Close-out，2026-09-06）

用户决策：(a) 接受 S7 null 结论（D−C=0），Protocol 019 收尾；S8（5 seeds × 3 streams 全矩阵）不执行。

| 阶段 | 结果 | Gate | 证据产物 |
|---|---|---|---|
| S0 | 登记完成，旧协议零改动 | — | protocol.json / source_sha256 快照 |
| S1 | input contract v2 + normalization v2 + graph identity v2 | Tests 1–4 8/8；legacy 回归 9/9 | source_sha256_S1.json、normalization_v2_time_scale.json |
| S2 | dev stream 303 + audit + Frozen A | 冷启动 gate FAIL（PR-AUC 0.186，P06） | S2_frozen_A_result.json、distribution.json |
| S3 | 同域适配 adapted_v4_seed1（15ep/lr1e-4） | 重测仍 FAIL（PR-AUC 0.340，P08）；用户批准以 adapted 起点继续 | S3_adaptation_result.json、adaptation_data/v1 |
| S4 | S4 更新方案（Recent/Anchor/曝光上限/anchor+distill/分组 LR） | Tests 5/6/10 4/4；稳定性 gate PASS @lr1e-5（P09） | S4_online_stability_result.json、runs_s4/ |
| S5 | dynamic expert v2（EMA gate/clone birth/ramp/retire+reactivate） | Tests 7–9 5/5；dev303 D−C=0、不误增长 | S5_dynamic_expert_v2_result.json、ftmoe_online_s5.py |
| S6 | stationary 2000-step（seed304） | PASS：D 恒 4 experts、无遗忘、D−C=0、C/D ≥ A | S6_stationary_result.json、runs_s6/ |
| S7 | controlled drift 2000-step（seed305 相位流） | **FAIL（P12）**：漂移真实（相位 F1 0.53↔0.32）但 unmatched≡0 → v2 触发架构性不可达，D−C=0 | S7_drift_result.json、s7_analysis.json、runs_s7/ |
| S8 | 按用户决策不执行 | — | — |

**正式结论**：

1. **输入修复有效且必要**：normalization v2 把归一化绝对值从 ~1e11 修复到 ≤3（alarm off）；
   graph identity v2 使 slot 替换不再被编码为迁移（015 型流 312/513 例假迁移的根因修复）。
2. **014 checkpoint 对该域冷启动不兼容**（修复输入后 PR-AUC 仅 0.186）；**同域适配显著改善**：
   Frozen A second-half F1 0.134→0.342、FPR 0.175→0.035、ECE 0.036、ROC-AUC 0.95；
   但 0.60 PR-AUC 门槛在该 1.5% 事件率流上不可达 → 场景-模型组合的可研究性限制被如实记录。
3. **S4 在线更新方案稳定**：B/C 无参考遗忘（anchor ≤ +0.005）、second-half 不劣于 Frozen A；
   online 更新在长流上略优于冻结（S6：0.2687 vs 0.2632）。
4. **动态专家（v2）机制正确性已验证**（birth 连续、clone、ramp、retire/reactivate、Adam 保持），
   但在本域**无增益**：宽阈值余弦路由使 unmatched≈0（EMA birth 门不可达）、4 experts 全覆盖使
   retire 不可达，漂移被 C 的可训路由吸收 → **D−C = 0.0（null），作为正式负面结论**。
   机制性触发条件（unmatched-only）对"路由始终有匹配"的架构不敏感，是未来 v3 设计方向
   （困惑度/滚动 loss 触发、更窄路由）——不在本协议范围内。

所有失败、停止条件、配置与 hash 均已保留（问题日志 P01–P12、protocol.json、各阶段 result JSON）。

## 10. 关键资产索引（artifacts/ftmoe_online/protocol_019/）

- 注册：protocol.json、dev_stream_config.json、s7_vm_groups.json、s6_stream_config.json、s7_drift_config.json
- 归一化：normalization_coverage_train.json、normalization_v2_time_scale.json
- 流：dev_streams/seed303_steps300、adaptation_data/raw+…/v1、stationary_streams/seed304_steps2000、drift_streams/seed305_steps2000
- 运行：runs/（S2 Frozen A）、runs_s4/（S4/S5 A–D）、runs_s6/、runs_s7/、runs/…/superseded、runs_s4/superseded、failed_attempts（P05/P07 现场）
- 结果：S2/S3/S4/S5/S6/S7 *_result.json、drift_streams/seed305_steps2000/s7_analysis.json
- 代码快照：source_sha256_S1.json、source_sha256_final.json

S3 已执行完毕（登记/数据/训练/重测细节见 §8.0 及 指令/FTMOE_PROTOCOL019_PROBLEM_LOG.md P07/P08）：

- 数据 v1：episodes 401–408（train 401–405 / dev 406–408 × 400 interval），
  `adaptation_data/v1/`（含 creation_ids 与 normalization v2 元数据；tolerance labels 与 runner 对拍一致）。
- 训练：014 v4 seed1 → `s3/adapted_v4_seed1/best.pt`（epoch7 by dev score，sha `670c56fe…`），
  15 epochs / lr 1e-4 / batch 32，125 s，全参数 warm-start。
- **Frozen A 重测（dev303，阈值 0.5）**：full F1 0.234→0.414、PR-AUC 0.186→0.340、
  FPR 0.175→0.035（✓ ≤0.15）、ROC-AUC 0.905→0.950、ECE 0.167→0.036；
  second-half F1 0.134→0.342。
- **Gate 判定：PR-AUC 0.340 < 0.60 → FAIL（P08，正式停止点）**。
  佐证：所有 checkpoint × 所有 BWGD2 系流的 Frozen PR-AUC 均 <0.60（015: 0.316 @11.6% 异常率、
  016: 0.135、019: 0.186→0.340），adapted 模型 ROC-AUC 0.95 + ECE 0.036 说明瓶颈是超低事件率下
  固定阈值的 precision 上限，而非表征失效。
- **后续（二选一，需用户拍板，未拍板前不擅自推进 S4）**：
  (a) 严守 0.60 门槛 → 019 停于 S3，报告 adapted 起点仍未达冷启动可用线；
  (b) 接受 adapted_v4 为合理 common starting point → 进入 S4：在 adapted 起点比较 A/B/C
  （B/C 稳定前不得启动 D；D−C 为主比较）。

## 8.1 S3 执行登记（数据/训练/验收约定，已完成）

按 The Plan §8（warm-start 路线，禁止 from-scratch 选择）。

### S3 登记（2026-09-06）

- **数据（S3-data-v1）**：adapted-BWGD2（016 契约：cpu clip [2,1860]、ram×2、io=1、
  synthetic disk law 同源、CPU_CAP_SCALE .8 / DISK_CAP_SCALE .25）**新注册 replay seeds**：
  train 401–405 / dev 406–408，每段 400 scored interval（+1 guard），
  与 S2 dev stream（seed 303）来源隔离；每 episode 记录 manifest（含 selected VM cohorts、hash）。
  episode 采集：`prepare_ftmoe_protocol019_s3_data.py` → `adaptation_data/raw/seed{seed}_steps400/`
  组装：`build_ftmoe_protocol019_adaptation_dataset.py` → `adaptation_data/v1/`
  （manifest.json + normalization.json + time/container/schedule/creation_ids/labels npy；
  labels = 物理过载 ±1 tolerance，与 online runner 语义逐位一致——已用 5 组随机输入对拍
  `run_ftmoe_online.tolerance_label` 验证；开发中曾发现并修复 tolerance 向量化移位 bug，
  未产生错误数据）。
- **归一化 v2**：模型输入 = host_features / time_scale_v2（protocol_019 注册工件，
  已含 host13 同组兜底）、demands / graph_scale（014 注册值）；capacity buffer =
  host_capacities / graph_scale[[0,1,4]]（数值与 014 训练时一致）。
- **训练（S3 warm-start 预算，The Plan §8.3）**：014 v4 seed_i checkpoint →
  `adapted_v4_seed_i`；15 epochs / lr 1e-4 / AdamW / batch 32 / 全参数；
  EAGate temperature 恒 1.0（不重启退火）；graph_context（creation ids）全程传入。
  dev 评估每 epoch（validation seeds 406–408，batch 64），best.pt = dev mean score 最优；
  末尾做 protocol_004 legacy-path anchor 探针（无 creation ids，仅记录不参与选择）。
  训练器：`train_ftmoe_protocol019_s3.py` → `s3/adapted_v4_seed{seed}/`。
- **验收（S2 同一 Frozen A 流程）**：`run_ftmoe_protocol019.py --checkpoint-path
  s3/adapted_v4_seed1/best.pt --method A` 在 dev stream seed303 上重测冷启动 gate
  （PR-AUC ≥ 0.60 且 FPR ≤ 0.15）。通过 → S4；失败 → 记录停止条件（P07），
  不改 scenario、不按结果调参。
