# Protocol 021 P21-S1 — 离线学习链 Exposure Ledger（完整暴露台账）

> **审计性质**：只读审计。未修改、移动或删除任何既有文件；未运行任何模型 / 训练 / 模拟器；未创建 git commit。
> 本步唯一新增文件为 `artifacts/ftmoe_online/protocol_021/offline_coverage_audit/` 下的三份产物。

> **审计目的**：021 必须证明新 runtime regime 未被产出「在线起点 checkpoint」的**整条离线学习链**覆盖。
> 「没有进梯度」不等于「没有影响实验设计」，因此每个数据源都按 exposure type 分类。

## 0. 起点 checkpoint

| 项 | 值 |
|---|---|
| 路径 | `artifacts\ftmoe_online\protocol_020\s6\adapted_v4_seed1\best.pt` |
| sha256（本次自行计算） | `10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b` |
| 交叉核对 | `artifacts/ftmoe_online/protocol_020/revision_20260909/r1/A_dev500_m1/configuration.json` → `source_checkpoint.sha256`，逐位一致；另与 `artifacts/ftmoe_online/protocol_020/continuation/protected_checkpoint_hashes.json` 的 expected/actual 一致 |
| 文件大小 / 修改时间 | 1024772 字节 / 2026-09-08 15:37:54 |
| 说明 | `mode = "s6_adapted"`、`epoch = 4`（见 R1 全部 `configuration.json`） |

链条（每跳都有 sha256 证据）：

```text
P014 physical_lr0003_e30/v4_seed1/checkpoints_by_epoch/epoch021.pt   e3513575ec26fc94a9f3c1d877a7119b21bacc97d7001fa02d18278aa316720b
   └─(P019 S3 warm_start, 15 epochs, lr 1e-4)─> protocol_019/s3/adapted_v4_seed1/best.pt   670c56fe94e738bd8a0f65dcacb0ca836960c1ce86dce6c1601a8c74f5078e3b
        └─(P020 S6 warm_start, 15 epochs, lr 1e-4)─> protocol_020/s6/adapted_v4_seed1/best.pt   10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b  ← 在线起点
```

证据：`artifacts/ftmoe_online/protocol_019/protocol.json`（base_checkpoint）、`artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/configuration.json`（warm_start）、`artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/configuration.json`（warm_start）、`artifacts/ftmoe_online/protocol_020/continuation/protected_checkpoint_hashes.json`。三处 sha256 均由本次审计自行重算并逐位匹配。

## 1. exposure type 词表

`gradient_training`（梯度训练）、`checkpoint_selection`（checkpoint 选择）、`normalization_fit`（归一化拟合）、`generator_fit`（生成器拟合）、`scenario_selection`（场景选择）、`anchor_regularization`（锚点正则）、`teacher_distillation`（教师蒸馏）、`development_evaluation`（开发评估）。

## 2. Stage protocol014（7 个数据源）

*冻结基础 v4 seed1 epoch021 = P019/P020 链条的根；本阶段产物同时是 normalization/graph capacity 与 disk-law 的物理原型来源*

| # | source | exposure types | evidence 文件（及文件内证据） | verified |
|---|---|---|---|---|
| 1 | `artifacts/ftmoe_end_to_end/data/protocol_004_physical/manifest.json` | 梯度训练、checkpoint 选择、归一化拟合、生成器拟合、开发评估 | seeds=[42,1,6,17,23,31,101,102]；train_blocks=[0..4]（202 步/块）、validation_blocks=[5,6,7]、block_size=202；host_capacities 为 8 行 train/val + 8 行 8192.0 RAM 的 16 行表；source_sha256 指向物理 replay seed42/1/6/17/23/31/101/102；array_sha256 给出 time_series/container_demand_series/schedule_series/labels 的 sha256。train_blocks 进梯度，validation_blocks 进 checkpoint/开发评估，数组同时是 P19/P20 normalization 与 disk-law 的原型来源。 | true |
| 2 | `artifacts/ftmoe_end_to_end/data/protocol_004_physical/{time_series.npy,container_demand_series.npy,schedule_series.npy,labels.npy}` | 梯度训练、checkpoint 选择、归一化拟合、生成器拟合、开发评估 | manifest.array_sha256 与各 .npy 的 sha256 一一对应；container_demand_series 同时是 disk_law_p20_ls/lm/v1/v2.json 的 source（见 disk_law 文件的 source / source_sha256 字段）。 | true |
| 3 | `artifacts/ftmoe_end_to_end/physical_replays/seed{42,1,6,17,23,31,101,102}/replay_log.npz` | 生成器拟合、梯度训练、开发评估 | protocol_004_physical/manifest.json 的 source_sha256 逐条登记这 8 个 replay_log.npz（如 seed42=7ad54633bf252f3499fe0d379964cf3559c2d6881fa2b7b4c03e65c0c66d0d40）；它们是 P014 数据集的上游 simulator replay 记录。 | true |
| 4 | `artifacts/ftmoe_end_to_end/runs/physical_lr0003_e30/configuration.json` | 梯度训练、checkpoint 选择 | arguments.data = artifacts/ftmoe_end_to_end/data/protocol_004_physical；epochs=30、learning_rate=0.0003、save_all_epochs=true、variants=5；manifest_sha256 与上条 manifest 的 self-computed sha256 相等。 | true |
| 5 | `artifacts/ftmoe_end_to_end/runs/physical_lr0003_e30/v4_seed1/` | checkpoint 选择、开发评估 | checkpoints_by_epoch/ 保留 epoch001..epoch030 全部 30 个 checkpoint；last.pt / best.pt 与 summary.json 并存，说明存在 validation 选择过程。epoch021.pt 的 self-computed sha256 = e3513575ec26fc94a9f3c1d877a7119b21bacc97d7001fa02d18278aa316720b，与 protocol_019/protocol.json base_checkpoint.sha256 逐位相等。 | true |
| 6 | `artifacts/ftmoe_end_to_end/comparison_014_complete.json` | checkpoint 选择、开发评估 | 含 registration_sha256、逐文件 source_sha256（含 v0_seed1/summary.json 与 v0_seed1/checkpoints_by_epoch/epoch024.pt 等）、primary_protocol、original_three/new_two/pooled_five、initialization_check_supportive、eligible_for_frozen_test；即 v4 谱系的开发期模型选择证据。 | true |
| 7 | `artifacts/ftmoe_end_to_end/diagnosis_reassessment/physical_lr0003_e30/{v0..v4}_seed{1,2,6}.json (+ v4_seed1_replay{31,101,102}_predictions.npz)` | checkpoint 选择、开发评估 | 已实际打开 v4_seed1.json：variant='v4'、seed=1、best_epoch=21、best_checkpoint='F:\PreGANPlus-master\artifacts\ftmoe_end_to_end\runs\physical_lr0003_e30\v4_seed1\checkpoints_by_epoch\epoch021.pt'、best_sha256=e3513575ec26fc94a9f3c1d877a7119b21bacc97d7001fa02d18278aa316720b，并含 A_last / B_best / epochs 逐 replay(31/101/102) 指标；protocol_019/protocol.json base_checkpoint.resolve 明确写着 resolve 依据 'diagnosis_reassessment best_checkpoint + comparison_014_complete provenance'。目录内 v0..v4 × seed1/2/6 共 15 个评估 JSON 与 3 个 replay31/101/102 预测文件在位（目录列举已核实）。 | true |

## 3. Stage protocol019（16 个数据源）

*在 P014 v4_seed1/epoch021 上做同域适配，产出 P020 S6 的 warm_start*

| # | source | exposure types | evidence 文件（及文件内证据） | verified |
|---|---|---|---|---|
| 1 | `artifacts/ftmoe_online/protocol_019/adaptation_data/raw/seed{401..408}_steps400/manifest.json (8 个)` | 梯度训练、checkpoint 选择 | 每个 manifest 的 selected_vm_indices 都是同一组 42 个 VM： [9,21,28,29,31,34,106,211,219,226,228,236,256,261,267,268,269,272,274,276,280,286,311,324,332,333,375,380,383,384,385,392,407,408,411,413,420,440,488,493,494,495]；steps=400、hosts=16；raw_class_counts_scored 每 seed 均有 CPU/RAM/Disk 三类正类。 | true |
| 2 | `artifacts/ftmoe_online/protocol_019/adaptation_data/v1/manifest.json` | 梯度训练、checkpoint 选择、归一化拟合 | train_seeds=[401,402,403,404,405]、validation_seeds=[406,407,408]（分块 train_blocks=[0..4] / validation_blocks=[5,6,7]，block_size=400）；class_counts 逐 seed 登记 4 类计数；array_sha256 含 creation_ids；registration.status='protocol019 S3 same-domain adaptation data; seeds disjoint from S2 dev stream (303)'。 | true |
| 3 | `artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/configuration.json` | 梯度训练、checkpoint 选择 | warm_start.path = artifacts/ftmoe_end_to_end/runs/physical_lr0003_e30/v4_seed1/checkpoints_by_epoch/epoch021.pt（sha256 e3513575…）；epochs=15、learning_rate=0.0001、batch_size=32、optimizer=AdamW、weight_decay=0.0001；data=protocol_019/adaptation_data/v1。 | true |
| 4 | `artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/{best.pt,last.pt,checkpoints_by_epoch/epoch001..015.pt}` | checkpoint 选择 | best.pt 的 self-computed sha256 = 670c56fe94e738bd8a0f65dcacb0ca836960c1ce86dce6c1601a8c74f5078e3b，与 protocol_020/s6/adapted_v4_seed1/configuration.json warm_start.sha256 及 protocol_020/continuation/protected_checkpoint_hashes.json 的 expected/actual 逐位相等；15 个 epoch checkpoint 全保留。 | true |
| 5 | `artifacts/ftmoe_online/protocol_019/normalization_v2_time_scale.json` | 归一化拟合 | 含 time_scale_v2_16x7 / time_scale_v1_16x7 / graph_scale_014 / fallback_columns=[] / source_checkpoint_sha256=e3513575…（即 P014 v4_seed1/epoch021）；被 protocol_020/adaptation_data/v1/manifest.json 的 normalization.source_019_artifact 直接引用为 v2 基座。 | true |
| 6 | `artifacts/ftmoe_online/protocol_019/normalization_coverage_train.json` | 归一化拟合、开发评估 | source 字段 = 'F:\PreGANPlus-master\artifacts\ftmoe_end_to_end\data\protocol_004_physical'，train_blocks=[0,1,2,3,4]，rules={EPS, LOW_COVERAGE_COUNT:12, LOW_COVERAGE_MAX_FRACTION:0.0001, fallback:'median of healthy peer maxima in same hardware group'}；hosts 数组长 16。注意：该文件自称只覆盖 train_blocks，但 P014 的 train_blocks 只有 5×202=1010 行，而 hosts[0].cpu_demand.count_nonzero=404（约 1/3），其覆盖范围与「仅 train_blocks」的口径存在张力，见 md 的“矛盾”一节。 | true |
| 7 | `artifacts/ftmoe_online/protocol_019/dev_stream_config.json` | 场景选择、开发评估 | phase='S2'、replay_seed=303、steps=300、methods=['A']、learning_rate=0.0、capacity_scales={CPU_CAP_SCALE:0.8, DISK_CAP_SCALE:0.25}、cpu_positive_clip=[2.0,1860.0]、ram_multiplier=2.0、io_per_container=1.0、synthetic_dynamic_disk=true、artificial_time_drift=false、disk_law_sha256=98f4e104…（= adapted_bwgd2_016/disk_law.json）；cold_start_gate='Frozen A PR-AUC >= 0.60 and FPR <= 0.15'。learning_rate=0.0 → 不更新梯度，但该流的选择（容量/适配器/分布目标）本身是场景设计暴露。 | true |
| 8 | `artifacts/ftmoe_online/protocol_019/stationary_streams/s6_stream_config.json` | 场景选择、开发评估 | phase='S6'、replay_seed=304、steps=2000、workload='adapted_BWGD2_protocol019'、capacity_scales={0.8, 0.25}、artificial_drift=false、synthetic_dynamic_disk=true；execution_boundary 明确 'A must stay stable, B/C must not forget, D must not grow experts…'。 | true |
| 9 | `artifacts/ftmoe_online/protocol_019/drift_streams/s7_drift_config.json` | 场景选择、开发评估 | phase='S7'、replay_seed=305、steps=2000、phase_len=500、schedule=['cpu','mixed','ram','cpu']（4 段，第 4 段为 cpu recurrence）、capacity_scales={0.8,0.25}、groups_source='training/dev workload statistics only (s7_vm_groups.json)'、groups_sha256=1e0dd93a…。 | true |
| 10 | `artifacts/ftmoe_online/protocol_019/s7_vm_groups.json` | 场景选择、开发评估 | source='training/dev workload statistics (episodes 401-408, dev 303, stationary 304 selected_vm_indices union)'；groups.cpu/mixed/ram 各 14 个 VM（cpu=[274,21,29,31,272,440,106,228,256,276,408,286,9,375] 等）；per_vm 42 条统计。即 P019 drift 的分组是按统计量划分的，属场景选择暴露。 | true |
| 11 | `artifacts/ftmoe_online/protocol_019/S2_frozen_A_result.json` | 开发评估 | keys 含 verdict / gate / metrics_full / metrics_second_half / anchor_reference；冷启动门禁（PR-AUC≥0.60、FPR≤0.15）的判定产物。 | true |
| 12 | `artifacts/ftmoe_online/protocol_019/S3_adaptation_result.json` | 开发评估、checkpoint 选择 | keys 含 adapted_checkpoint / s3_training / frozen_A_retest_dev303 / baseline_protocol014_frozen_A / gate_rule / verdict；即 S3 适配 checkpoint 的验收证据。 | true |
| 13 | `artifacts/ftmoe_online/protocol_019/S4_online_stability_result.json` | 开发评估、场景选择 | replay_seed=303、start_checkpoint='artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/best.pt'、scheme='s4 (recent128/anchor384, exposure<=3, lambda_anchor 0.25, lambda_distill 0.10, grouped LR for B)'；rows 覆盖 A/B/C × lr{3e-5,1e-5}；gate.pass='lr=1e-5'、detail 记录 3e-5 被 second-half 判据否决。→ 在线超参（lr/anchor 系数）由该 dev 流选定。 | true |
| 14 | `artifacts/ftmoe_online/protocol_019/{S5_dynamic_expert_v2_result.json,S6_stationary_result.json,S7_drift_result.json}` | 开发评估、场景选择 | S5：replay_seed=303、gate_version='v2'、design 明确 'plan §10: EMA gate (samples>=128, ema>=0.05, 3 windows), centroid k=1 birth via parent clone + threshold q70 + routing ramp (0->1 over 4 updates…)'、observation 记录网络未增长（D 与 C 逐位相同）。S6：replay_seed=304、stream='stationary_streams/seed304_steps2000 (2000 intervals, raw positives 251/32000 = 0.78%)'、gates.stationary_sanity='PASS'、detail 含 'D equals C bit-for-bit on this stationary stream (D-C = 0.0)'。S7：键含 stream / stream_sha256 / phase_metrics / expert_count / d_minus_c / verdict / gate_rule。三者共同构成 P019 的开发期门禁、超参与结论记录。 | true |
| 15 | `anchor：artifacts/ftmoe_online/protocol_019/adaptation_data/v1（train 分块 401-405）` | 锚点正则 | P019 自身的锚点/蒸馏同样取自 adaptation_data/v1 的 train 分块：artifacts/ftmoe_online/protocol_019/S4_online_stability_result.json 的 scheme='s4 (recent128/anchor384, exposure<=3, lambda_anchor 0.25, lambda_distill 0.10, grouped LR for B)' 且 ref_f1_start=0.904858870906466（锚点参考）。P020 侧则把锚点窗口显式固定为 'train episodes rows 272-399'：recovery/PreGANSrc/src/ftmoe_online_s7.py:12-14 注释 '- Anchor memory: rows 272-399 of every protocol-020 adaptation train episode (12 episodes x 128 rows = 1536 windows), graph-semantics v3'，并由 run_ftmoe_protocol020.py:291 与 run_ftmoe_protocol020_r1.py:363 写入配置。 | true |
| 16 | `teacher/distillation：artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/best.pt（及 P020 起点 best.pt）` | 教师蒸馏 | artifacts/ftmoe_online/protocol_020/runs/C_model1_seed500_drift/configuration.json 的 s7_memory.teacher = 'frozen starting checkpoint'，且同文件的 s7_memory.lambda_distill=0.1、source_checkpoint 指向 s6/adapted_v4_seed1/best.pt。→ 起点 checkpoint 本身充当冻结教师。 | true |

## 4. Stage protocol020_s6（20 个数据源）

*产出在线起点 s6/adapted_v4_seed1/best.pt；并含 R1 保护阈值/方法选择所用 dev 流*

| # | source | exposure types | evidence 文件（及文件内证据） | verified |
|---|---|---|---|---|
| 1 | `artifacts/ftmoe_online/protocol_020/adaptation_data/raw/{train,dev}/{baseline,cpu_fault,ram_fault,disk_fault}/seed40X_steps400/manifest.json (20 个)` | 梯度训练、checkpoint 选择、场景选择 | train 12 个 episode（seed401-403 × 4 profile，cohort='train'，cohort_vm_ids 277 个，与 vm_split.json cohorts.train 完全一致）+ dev 8 个 episode（seed404-405 × 4 profile，cohort='dev'，cohort_vm_ids 86 个，与 cohorts.dev 完全一致）；实际 profile：baseline(1.00,1.00,0.90,ru1400) / cpu_fault(0.65,1.00,0.90,ru1400) / ram_fault(1.00,0.55,0.90,ru2400) / disk_fault(1.00,1.00,0.35,adapter={})；全部 disk_law_relative='artifacts/ftmoe_online/protocol_020/disk_law_p20_ls.json'；scenario_adapter_sha256=94c209fe…。 | true |
| 2 | `artifacts/ftmoe_online/protocol_020/adaptation_data/v1/manifest.json` | 梯度训练、checkpoint 选择、归一化拟合、场景选择 | train_episodes 12 个 / dev_episodes 8 个；block_size=400；label_policy='actual physical overload, dominant ratio; +/-1 tolerance (guard row matures the last row)'；episode_hashes[*].profile 记录的就是上条的实际 profile；normalization.registered_artifact='artifacts/ftmoe_online/protocol_020/normalization_v2_time_scale.json'、source_019_artifact 指向 P019 同名片。 | true |
| 3 | `artifacts/ftmoe_online/protocol_020/normalization_v2_time_scale.json` | 归一化拟合 | name='protocol-020 normalization v2 (019 base + 020 train-cohort coverage)'；含 time_scale_v2_16x7、graph_scale_014、coverage_stats_train（逐 host 的 cpu_demand/ram_space 等分位统计）、normalized_abs_max_report_train/dev、source_019_artifact、source_sha256。即 020 归一化是在 019 基座上用 020 train cohort 重新覆盖拟合。 | true |
| 4 | `artifacts/ftmoe_online/protocol_020/disk_law_p20_ls.json` | 生成器拟合 | source='F:\PreGANPlus-master\artifacts\ftmoe_end_to_end\data\protocol_004_physical\container_demand_series.npy'、source_sha256=99cf6fd9…（与 P014 manifest.array_sha256.container_demand_series 逐位相等）、train_blocks=[0,1,2,3,4]；limitation='per-task marginal and first-order persistence only; cross-resource and cross-task correlation not preserved'；derived_from='artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json'（sha256 98f4e104…）。→ 磁盘生成律直接拟合自 P014 训练数据。 | true |
| 5 | `artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json` | 生成器拟合、场景选择 | schema_version=3、arrival_mean=1.0、arrival_sigma=1.5、adapter={cpu_lower:2.0,cpu_upper:1860.0,cpu_mult:1.0,ram_mult:2.0,ram_upper:null,disk_mult:1.0}、disk_law_relative='artifacts/ftmoe_online/protocol_020/disk_law_p20_ls.json'；registration 记录 v4(2026-09-08) 允许 per-phase override 且基础 profile 为 016 契约 + p20-ls。 | true |
| 6 | `artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/configuration.json` | 梯度训练、checkpoint 选择、归一化拟合 | warm_start.path=protocol_019/s3/adapted_v4_seed1/best.pt（sha256 670c56fe…）；data=protocol_020/adaptation_data/v1；epochs=15、learning_rate=0.0001、batch_size=32、optimizer=AdamW、weight_decay=0.0001；selection_rule='0.6*pr_auc + 0.4*resource_macro_f1 on dev episodes with positives (pre-registered)'；normalization 内嵌 time_scale（112 项）、graph_scale、graph_host_capacity（前 8 行 RAM=0.6519…、后 8 行 RAM=1.2433…）、normalization_version=2。 | true |
| 7 | `artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/{best.pt,last.pt,checkpoints_by_epoch/epoch001..015.pt,epochs.csv,summary.json}` | checkpoint 选择 | best.pt 的 self-computed sha256=10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b，与 R1 全部 configuration.json 的 source_checkpoint.sha256(epoch=4, mode='s6_adapted') 及 continuation/protected_checkpoint_hashes.json 一致；15 个 epoch checkpoint 全保留，选择规则见上条。 | true |
| 8 | `artifacts/ftmoe_online/protocol_020/vm_split.json` | 场景选择、梯度训练、开发评估 | scanned_vm_count=500、eligible_vm_count=457；split_rule="stratum=(cpu_ips_mean_tercile,ram_units_mean_tercile); bucket=sha256('p20:v1:<vm_id>') mod 10; bucket 0-5 train / 6-7 dev / 8-9 online"；cohorts={train:277, dev:86, online:94}（我自行核算：三集合两两交集为空、并集=457）。该划分同时决定训练/开发/在线三套流的 VM 来源。 | true |
| 9 | `artifacts/ftmoe_online/protocol_020/capacity_scan/ (79 个 manifest.json, seed=410, 400 步, cohort=train)` | 场景选择、生成器拟合 | capacity_scan_report.json（report 覆盖其中 31 个非 smoke/400 步候选）记录 gate_rules={normal_share:[0.8,0.95], fault_hoststep_floor_reference:150, event_floor_reference:30, reference_horizon_host_steps:32000, phase_horizon_host_steps:8000, phase_dominant_floor:100, deployment_rejection_max:0.2, migration_rejection_max:0.4} 与 gate_pass=false（31/31）。79 个 manifest 的实际 profile/adapter 取值全部读自文件：cpu 单因子 0.4–1.0、ram 单因子 0.1–1.0、disk 单因子 0.12–0.45、phase 组合若干；adapter 变体含 cpu_mult=1.5、cpu_upper=2200/2600、disk_mult=1.5/2.0、ram_mult=2.5/3.0、ram_upper=1400/1900；disk law 变体 disk_law.json(2)、disk_law_p20_ls(23)、lm(1)、v1(2)、v2(3)。 | true |
| 10 | `artifacts/ftmoe_online/protocol_020/capacity_scan_dev/ (9 个 manifest.json, seed=500, 400 步, cohort=train 来源但流名 phase_*)` | 场景选择 | 9 个候选：ram 轴 0.4/0.5/0.55/0.6/0.75 与 ram_upper∈{None,2400,2500} 的组合，全部 disk law=disk_law_p20_ls.json；cohort_vm_ids=86（= vm_split dev 组）。注意 manifest 的 cohort 字段写 'train' 而 VM 列表是 dev 组（见 md 的“矛盾”一节）。 | true |
| 11 | `artifacts/ftmoe_online/protocol_020/drift/drift_config.json` | 场景选择 | kind='drift'、phase_len=400、5 个 phase（baseline 1/1/0.9 ru1400、cpu_fault 0.65/1/0.9 ru1400、ram_fault 1/0.55/0.9 ru2400、disk_fault 1/1/0.35 adapter={}、cpu_recurrence 0.65/1/0.9 ru1400）；registration 记录 v4(2026-09-08, P22) 将 RAM phase 定为 (1,0.55,0.9)+ru2400。 | true |
| 12 | `artifacts/ftmoe_online/protocol_020/stationary/stationary_config.json` | 场景选择、开发评估 | cohort='dev'、phase_len=2000、单一 baseline phase (1.0,1.0,0.9, ru1400)；registration='baseline condition = drift-config baseline phase (disk 0.9, ram_upper 1400) so stationary and drift streams share the same domain contract'。 | true |
| 13 | `artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed500_steps2000/manifest.json` | 开发评估、场景选择 | seed=500、steps=2000、cohort='dev'、cohort_vm_ids 86 个（= vm_split dev 组，与 train/online 交集为空）；config_sha256=drift_config_sha256=790daa32…；stream_sha256=4a275b3bc3e90b991780f0ea2de1f0841380d1978fe64f483523b0761762793b；per_phase 记录 baseline 3 / cpu 43 / ram 178(173 RAM dominant) / disk 250(226 Disk) / recurrence 45 个 anomalous host-steps；deployment_rejection_rate=0.1868、migration_rejection_rate=0.3087。 | true |
| 14 | `artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed501_steps2000/manifest.json` | 开发评估、场景选择 | seed=501、steps=2000、cohort='dev'、phase_len=2000、单一 baseline phase (1.0,1.0,0.9,ru1400)；stream_sha256=59e81053bb3136dba9c1bc7616fc5d2393bc05dcb9f85c4422cb836793a63323；deployment_rejection_rate=0.0621、migration=0.1186。 | true |
| 15 | `artifacts/ftmoe_online/protocol_020/S7_drift_AC_result.json / S7_stationary_result.json` | 开发评估 | S7_drift_AC_result.json 明确 stream='dev_seed500_steps2000'，逐 phase 给出 A/C 的 f1/pr_auc 与 lags（boundary 400/800/1200/1600），verdict.C_measurable_adaptation='WEAK/NEGATIVE'；S7_stationary_result.json 明确 stream='dev_seed501_steps2000'，包含 A/B/C 的 anchor_last_f1 与 exposure/exposure_violations，gates={C_pr_auc_ge_A_minus_0.02:true, anchor_drop_within_0.03:true}。 | true |
| 16 | `artifacts/ftmoe_online/protocol_020/continuation/dynamic_v3.json` | 开发评估、场景选择 | calibration.stream='…\drift_streams\dev_seed501_steps2000'、calibration.stream_sha256=59e81053…、calibration.checkpoint_sha256=10c44bdb…；即动态专家 v3 的 novelty 阈值（entropy_p95/margin_p05/novelty_threshold=0.1625/loss_baseline）是在 dev501 上标定的。 | true |
| 17 | `artifacts/ftmoe_online/protocol_020/continuation/protected_checkpoint_hashes.json` | checkpoint 选择、开发评估 | 保护清单逐条给出 expected/actual/unchanged=true 的三个 checkpoint：P014 epoch021.pt(e3513575…)、P020 s6/adapted_v4_seed1/best.pt(10c44bdb…)、P019 s3/adapted_v4_seed1/best.pt(670c56fe…)。我自行重算三者 sha256 与文件内记录逐位一致。 | true |
| 18 | `artifacts/ftmoe_online/protocol_020/runs/* (A/B/C, seed500/501) 与 runs_continuation/* (B/C/D_v3 × lr1e4/1e5/w1)` | 开发评估、场景选择 | runs 5 个 run 的 configuration.json：method∈{A,B,C}、model_seed=1、replay_seed∈{500,501}、lr∈{0.0,1e-5}、source_checkpoint=s6/adapted_v4_seed1/best.pt(10c44bdb…)、anchor_source='protocol-020 train episodes rows 272-399'；runs_continuation 12 个 run：method∈{B,C,D}、lr∈{1e-4,1e-5}、replay_seed∈{500,501}。continuation/verification.json 记录 new_completed_runs=12、tests_passed=23。 | true |
| 19 | `artifacts/ftmoe_online/protocol_020/revision_20260909/r1/ (A / C-legacy / C-residual-off / C-residual-on × dev500/dev501)` | 开发评估、场景选择 | revision_20260909/r1/execute_registered_grid_primary.py:45-58 冻结网格 `for seed in (500, 501): for method in ("A","C-legacy","C-residual-off","C-residual-on")`，stream 为 drift_streams/dev_seed{500,501}_steps2000，checkpoint 为 s6/adapted_v4_seed1/best.pt；final_runs_20260909_1435/ 下 8 个 run 的 configuration.json 全部 model_seed=1、replay_seed∈{500,501}、start_rng_seed∈{8419,8420}、normalization_version=2、graph_semantics_version=3、capacity_control_version=1；protection_enabled=false（A/C-legacy/C-residual-off）与 true（C-residual-on）。→ 保护阈值/方法选择流 = dev500 + dev501。 | true |
| 20 | `artifacts/ftmoe_online/protocol_020/continuation/audit_final.json` | 开发评估 | 文件存在（253894 字节）；作为 P20 S8/S9 的开发验收汇总被 continuation/verification.json 与 docs/FTMOE_ONLINE_PROTOCOL_020.md §5 引用。本次仅核实其存在与规模，未逐字段解构。 | true |

## 5. 覆盖结论

### 5.1 已在离线链中暴露的容量 profile / 适配器 / 磁盘律 / VM cohort / seed

**(a-1) 容量 profile（实际进入 P020 S6 训练与 S5/S7 流的，逐字读自文件）**

| profile | cpu_scale | ram_scale | disk_scale | ram_upper | 证据 |
|---|---|---|---|---|---|
| `baseline` | 1.0 | 1.0 | 0.9 | 1400 | `artifacts/ftmoe_online/protocol_020/drift/drift_config.json` phases[]；`artifacts/ftmoe_online/protocol_020/adaptation_data/raw/train/baseline/seed401_steps400/manifest.json` profile 字段 |
| `cpu_fault` | 0.65 | 1.0 | 0.9 | 1400 | `artifacts/ftmoe_online/protocol_020/drift/drift_config.json` phases[]；`artifacts/ftmoe_online/protocol_020/adaptation_data/raw/train/cpu_fault/seed401_steps400/manifest.json` profile 字段 |
| `cpu_recurrence` | 0.65 | 1.0 | 0.9 | 1400 | `artifacts/ftmoe_online/protocol_020/drift/drift_config.json` phases[]；`artifacts/ftmoe_online/protocol_020/adaptation_data/raw/train/cpu_recurrence/seed401_steps400/manifest.json` profile 字段 |
| `ram_fault` | 1.0 | 0.55 | 0.9 | 2400 | `artifacts/ftmoe_online/protocol_020/drift/drift_config.json` phases[]；`artifacts/ftmoe_online/protocol_020/adaptation_data/raw/train/ram_fault/seed401_steps400/manifest.json` profile 字段 |
| `disk_fault` | 1.0 | 1.0 | 0.35 | null | `artifacts/ftmoe_online/protocol_020/drift/drift_config.json` phases[]；`artifacts/ftmoe_online/protocol_020/adaptation_data/raw/train/disk_fault/seed401_steps400/manifest.json` profile 字段 |

> P021 计划 §1.1 列出的四个「真实训练 episode 已含 profile」与上表逐位一致（baseline 1.00/1.00/0.90 RAM upper=1400；cpu_fault 0.65/1.00/0.90 RAM upper=1400；ram_fault 1.00/0.55/0.90 RAM upper=2400；disk_fault 1.00/1.00/0.35）。

**(a-2) 被数据层扫描过、因此不再是「未见容量倍率」的候选（S4 data-only scan，seed=410 / seed=500）**

- cpu 单因子：`0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0`
- ram 单因子：`0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0`
- disk 单因子：`0.12, 0.15, 0.17, 0.1875, 0.2, 0.22, 0.25, 0.3, 0.35, 0.45`
- phase 组合候选：CPU∈{0.6,0.65}、RAM∈{0.4,0.6}、Disk∈{0.35,0.5,0.6,0.9} 的 12 组（`capacity_scan/phase_*/`）
- dev 侧 RAM 候选：`1.0/0.4/0.9 ru2400`、`1.0/0.5/0.9 (ru null|2400)`、`1.0/0.55/0.9 (ru 2400|2500)`、`1.0/0.6/0.9 (ru null|2500)`、`1.0/0.75/0.9 (ru null|2500)`（9 个，`capacity_scan_dev/`）
- 证据：`artifacts/ftmoe_online/protocol_020/capacity_scan/capacity_scan_report.json`（`gate_rules` + 31 条 `candidates[].profile`）与 79 个 `capacity_scan/*/seed410_steps400/manifest.json`、9 个 `capacity_scan_dev/*/seed500_steps400/manifest.json` 的 `profile` / `adapter` 字段。

**(a-3) adapter 旋钮取值**

| 旋钮 | 实际进入 S5/S6/S7 的值 | 仅在扫描中出现、未进入最终配置的值 |
|---|---|---|
| `cpu_lower` | `2.0` | — |
| `cpu_upper` | `1860.0` | `2200`、`2600` |
| `cpu_mult` | `1.0` | `1.5` |
| `ram_mult` | `2.0` | `2.5`、`3.0` |
| `ram_upper` | `1400`、`2400`、`null` | `1900`、`2500` |
| `disk_mult` | `1.0` | `1.5`、`2.0` |
| `arrival_mean` / `arrival_sigma` | `1.0` / `1.5` | — |

证据：实际值来自 `artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json`、`artifacts/ftmoe_online/protocol_020/drift/drift_config.json` phases[].adapter、20 个 adaptation raw manifest 的 `scenario_adapter.adapter`；扫描值来自 `capacity_scan/*/seed410_steps400/manifest.json` 的 `adapter` 字段（本次逐文件统计）。`scenario_adapter.json` 自算 sha256 = `94c209feb4ada03522cadf79517c9dbb5198a78628160642cf2335d3713d939b`，与其在 drift/adaptation manifest 中登记的 `scenario_adapter_sha256` 一致。

**(a-4) 磁盘律（disk laws）**

| law | sha256（自算） | p20_growth | 是否进入实际配置 |
|---|---|---|---|
| `artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json` | `98f4e104a31834d6b3841769a06ee689505e3b3d59395f823a949b01ef06a35c` | — | 是（P014/P019 时代契约律；019/020 非 scan 文件中 37 处引用） |
| `artifacts/ftmoe_online/protocol_020/disk_law_p20_ls.json` | `cd4679c594ba6a77290a025429d295699890c9619602c892be698af3fa80efd9` | stay .65 / up1 .30 / up2 .05 / up3 0 | **是（P020 唯一实际使用）**；45 处引用 |
| `artifacts/ftmoe_online/protocol_020/disk_law_p20_lm.json` | `d45491bb6e0d84df09355a4988ae674178d4dcee7784ed57f46d52824acc3d0e` | stay .45 / up1 .40 / up2 .15 / up3 0 | 否（仅 `capacity_scan/disk_0.3_lawlm`） |
| `artifacts/ftmoe_online/protocol_020/disk_law_p20_v1.json` | `7e8abf07fb0f3752686281c809d5a075e59a4c6c8507d3fe62fcd56145aa6c58` | stay .30 / up1 .40 / up2 .20 / up3 .10 | 否（仅 `capacity_scan/disk_*.25_lawv1`、`disk_0.2_lawv1`） |
| `artifacts/ftmoe_online/protocol_020/disk_law_p20_v2.json` | `35a0cb7b3f915dba509af4555fc82a37499425f8129ff7ebc369f07954110a68` | stay .15 / up1 .45 / up2 .30 / up3 .10 | 否（仅 `capacity_scan/disk_*.22/.25/.2_lawv2`） |

`p20_ls` 的 `source` 直接指向 `artifacts/ftmoe_end_to_end/data/protocol_004_physical/container_demand_series.npy`（`source_sha256 = 99cf6fd9…`），且 `limitation = "per-task marginal and first-order persistence only; cross-resource and cross-task correlation not preserved"`。

**(a-5) VM cohort**

| cohort | 数量 | 证据 |
|---|---|---|
| P020 train | 277 | `artifacts/ftmoe_online/protocol_020/vm_split.json` `cohorts.train`（与 `adaptation_data/raw/train/*/manifest.json` 的 `cohort_vm_ids` 逐元素一致） |
| P020 dev | 86 | `artifacts/ftmoe_online/protocol_020/vm_split.json` `cohorts.dev`（与 drift 流 manifest 及 dev adaptation manifest 一致） |
| P020 online | 94 | `artifacts/ftmoe_online/protocol_020/vm_split.json` `cohorts.online`（本次审计未在任何 019/020 run 中见到使用；登记为禁用集） |
| P019 selected | 42 | `artifacts/ftmoe_online/protocol_019/adaptation_data/raw/seed{401..408}_steps400/manifest.json` `selected_vm_indices`；P019 的 dev/stationary/drift 三条流同为这 42 个 |

本次自行核算：三组两两交集为空、并集 = 457 = `eligible_vm_count`（文件内写 457，`scanned_vm_count` = 500）。

P019 的 42 个 VM **全部**落在 P20 三组之内：∩train = 28 个、∩dev = 5 个 `[236, 267, 274, 280, 420]`、∩online = 9 个。

**(a-6) seed**

| 类别 | 值 | 证据 |
|---|---|---|
| model seed | P014 `[1, 2, 6]` + confirmation `[17, 42]`；P019 `1`；P020 `1` | `artifacts/ftmoe_end_to_end/protocol_004_physical.json`；`.../protocol_019/s3/adapted_v4_seed1/configuration.json`；`.../protocol_020/s6/adapted_v4_seed1/configuration.json`、`.../r1/final_runs_20260909_1435/*/configuration.json` |
| replay seed（模型运行） | P014 train `[42, 1, 6, 17, 23]`、val `[31, 101, 102]`、保留测试 `[201..205]`（`protocol_014_historical_confirmation.json` 的 `test_rule` 声明保持未开启） | `artifacts/ftmoe_end_to_end/data/protocol_004_physical/manifest.json` |
| replay seed（数据生成） | P019 `[401..408]`（train 401-405 / val 406-408）+ `303`（dev）/`304`（stationary）/`305`（drift）；P020 `[401, 402, 403]`（train）/`[404, 405]`（dev）/`410`（capacity scan）/`500`（capacity_scan_dev & drift dev）/`501`（stationary dev） | `protocol_019/adaptation_data/v1/manifest.json`、`protocol_019/{dev_stream_config.json,stationary_streams/s6_stream_config.json,drift_streams/s7_drift_config.json}`；`protocol_020/adaptation_data/v1/manifest.json`、`capacity_scan/*/manifest.json`、`capacity_scan_dev/*/manifest.json`、`drift_streams/*/manifest.json` |
| run RNG seed | `8419`（replay500）、`8420`（replay501） | `protocol_020/revision_20260909/r1/final_runs_20260909_1435/*/configuration.json` `start_rng_seed` |

### 5.2 (b) 链条中是否出现「时间耦合规则」：CPU burst → 滞后 RAM 抬升 → 滞后 Disk 累积

**结论：未发现（`not_found`）。** 在 014/019/020 的模拟器代码与全部配置/产物中，不存在任何 CPU→RAM→Disk 的时间滞后耦合机制；工作量生成只有**静态** capacity scale / demand adapter 旋钮 + **每任务独立的一阶 Markov 磁盘占用律**，相位切换是离散跳变。

本次实际运行的命令（PowerShell 7，`Set-Location F:\PreGANPlus-master`）：

```powershell
$paths = @('simulator\workload','simulator\environment','recovery\PreGANSrc\src',
           'artifacts\ftmoe_end_to_end','artifacts\ftmoe_online\protocol_019','artifacts\ftmoe_online\protocol_020')
$files = Get-ChildItem -Path $paths -Recurse -File -Include *.py,*.json,*.md |
         Where-Object { $_.Name -notlike '*Protocol021*' }
# -> 命中 1003 个文件
foreach($k in 'cascade','cpu_to_ram','delayed','mechanism_seed','hysteresis','feedback',
               'lag','ram_upper','retention','temporal','ramp','accumulat'){
  $files | Select-String -Pattern $k -SimpleMatch -AllMatches | Measure-Object   # 计数见下表
}
```

| keyword | 命中行数 | 命中文件数 | 判定 |
|---|---|---|---|
| `cascade` | **0** | **0** | 空 |
| `cpu_to_ram` | **0** | **0** | 空 |
| `delayed` | **0** | **0** | 空 |
| `mechanism_seed` | **0** | **0** | 空 |
| `hysteresis` | **0** | **0** | 空 |
| `feedback` | **0** | **0** | 空 |
| `retention` | 9 | 2 | 8 次为 `r1_analysis.json` 的 `anchor_retention`（锚点保持率）指标键；1 次为 `ftmoe_online_s7.py:273` 注释 “Measure retention on P20 anchors…”。**非 RAM retention 机制** |
| `lag` | 109 | 10 | 全部为 adaptation lag 评估指标（`boundary` / `lag_interval` / `target_median_f1`），散见 `protocol_020/continuation/audit_final.json`(61)、`protocol_019/drift_streams/seed305_steps2000/s7_analysis.json`(12)、`protocol_020/S7_drift_AC_result.json`(11) 等。**非 CPU→RAM/Disk 时间滞后** |
| `temporal` | 104 | 28 | 为 `train_ftmoe_ablation_existing.py` 及其 `source_snapshot` 副本中的时序编码器命名/注释。**非跨资源时间耦合** |
| `ramp` | 93 | 30 | 为动态专家 Top-k / expert birth-ramp（`ftmoe_online_s5.py`、`ftmoe_online_s8.py`、`revision_20260909/r0/ramp_diagnostic.json`）。**非 RAM ramp 机制** |
| `ram_upper` | 102 | 43 | 为 demand adapter 的**静态**上限旋钮。**非滞后项** |
| `accumulat` | 2 | 2 | `ftmoe_dynamic_expert_v3.py:37` “Stateless-ish accumulator for the v3 birth gate”；`ftmoe_online_s8.py:1044` “statistics accumulated by deployment predictions” |

**排除性佐证（非仅“没搜到”）**：

1. `artifacts/ftmoe_online/protocol_020/disk_law_p20_ls.json` 的 `limitation` 明文写明：`per-task marginal and first-order persistence only; cross-resource and cross-task correlation not preserved` —— 磁盘生成律**只保留单任务边缘分布与一阶持续概率，明确不保留跨资源相关性**，因此结构上不可能携带 CPU→Disk 的滞后传递。
2. `artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed500_steps2000/manifest.json` 的 `transitions[]` 记录的是 `capacity_before` / `capacity_after` 的**离散跳变**（interval 400/800/1200/1600），没有任何跨资源滞后参数。
3. 全仓库唯一含 `cascade` / `cpu_to_ram` / `mechanism_seed` 的文件是 `simulator/workload/BitbrainWorkloadProtocol021.py`（排除 021 后分别 58 / 4 / 10 次命中，均为该文件内），它是 **P21-S2 新增**的生成器，`git status --short` 显示为未跟踪文件 `?? simulator/workload/BitbrainWorkloadProtocol021.py`，**不属于本台账审计的离线学习链**。
   同批被排除的还有：`?? test_ftmoe_protocol021_unseen.py`（15292 字节）、`?? prepare_ftmoe_protocol021_unseen.py`（26081 字节）、`?? artifacts/ftmoe_online/protocol_021/`（含并发生成的 `pilot_streams/_smoke/p025_seed600_steps40/`）、`?? 指令/FTMOE_PROTOCOL021_UNSEEN_REGIME_EXECUTION_PLAN_FINAL.md`。

> **并发写入提示**：审计期间同一 021 流水线的其他步骤正在写入上述文件（mtime 21:52–21:55）。本报告记录的是 **2026-09-10 21:55 左右的排除清单快照**；排除这些文件后，014/019/020 范围内的检索计数在两次独立运行中完全一致（0/0），结论不因并发写入改变。

### 5.3 (c) 仍无法核实（UNVERIFIED）

- **P014 数据集的上游 VM / trace 选择集合** —— 原因：artifacts/ftmoe_end_to_end/data/protocol_004_physical/manifest.json 只登记 replay seed 与数组 sha256，未登记任何 VM id；physical_replays/seed*/replay_log.npz 为二进制，本次按「不运行模型/不新增中间产物」纪律未解包。
- **artifacts/ftmoe_end_to_end/comparison_014_complete.json 内全部 48k 字符的逐条对比数值** —— 原因：仅核实键结构（registration_sha256 / source_sha256 / primary_protocol / original_three / new_two / pooled_five / initialization_check_supportive / eligible_for_frozen_test）与文件规模，未逐条复算。
- **artifacts/ftmoe_online/protocol_020/continuation/audit_final.json 的逐字段结论** —— 原因：仅核实文件存在（253894 字节）并确认被 continuation/verification.json 与 docs/FTMOE_ONLINE_PROTOCOL_020.md §5 引用，未逐字段解构。
- **artifacts/ftmoe_online/protocol_020/capacity_scan/ 下 79 个 manifest 与 capacity_scan_report.json 的 31 条候选之间的差额（48 个）** —— 原因：报告由 analyze_ftmoe_protocol020_capacity_scan.py 的 main() 枚举 SCAN.iterdir() 下每个子目录的 manifest.json（本次未运行该脚本）。报告 31 条候选的 profile 全部是 disk_scale=0.3 的 cpu/ram/disk 单因子值；而磁盘上另有 48 个 phase_* 候选、lawlm/lawv1/lawv2 变体与 *_am2 / dm / cm1.5 / au / rm 变体不在报告中。因此报告相对目录集是陈旧的（见 md 矛盾 C6），但本次未逐条复原其生成时点。
- **P019 s7_vm_groups.json 的 42 个 VM 究竟是 401-408/303/304 的并集还是子集** —— 原因：groups_source 文本称 'selected_vm_indices union'，但我核对 8 个 P019 raw manifest 的 selected_vm_indices 完全同构（同一 42 个），故 union 与任一单集相等，无法区分两种解释。

另：`source_scope.relative_to_full_014_019 = "unverified"`。
P014 数据集的上游 VM / trace 选择集合未在任何被审产物中登记（`artifacts/ftmoe_end_to_end/data/protocol_004_physical/manifest.json` 只登记 replay seed 与数组 sha256，无 VM 字段），因此**不能宣称“绝对未见”**；按 021 计划 §4 必须写：

```text
source-unseen relative to full historical chain: UNVERIFIED
```

`source_scope.relative_to_p20_s6 = "source-disjoint"`（VM 集合级，证据见 5.1(a-5)）。

## 6. 与既有文档的对照 / 发现的矛盾（全部保留，不做修正）

### C1. `adaptation_data/v1/manifest.json` 顶层 `profiles` 仍然是陈旧值（P28 / P36 记录的错配**现在依然存在**）

| 位置 | baseline | cpu_fault | ram_fault | disk_fault |
|---|---|---|---|---|
| `protocol_020/adaptation_data/v1/manifest.json` 顶层 `profiles` | 1.0 / 1.0 / **0.30** | **0.75** / 1.0 / **0.30** | 1.0 / **0.45** / **0.30** | 1.0 / 1.0 / **0.20** |
| 同文件 `episode_hashes[*].profile`（= 实际） | 1.0 / 1.0 / **0.90** (ru1400) | **0.65** / 1.0 / **0.90** (ru1400) | 1.0 / **0.55** / **0.90** (ru2400) | 1.0 / 1.0 / **0.35** (adapter={}) |
| `adaptation_data/raw/*/seed40X_steps400/manifest.json` `profile` | 1.0 / 1.0 / 0.90 | 0.65 / 1.0 / 0.90 | 1.0 / 0.55 / 0.90 | 1.0 / 1.0 / 0.35 |

同样陈旧的还有 `artifacts/ftmoe_online/protocol_020/adaptation/adaptation_profiles.json`（其 `registration` 自称 “INITIAL CANDIDATES from plan §12.2 … final scales will be replaced by the S4 data-only scan selections”——但文件从未被替换，仍为 0.75 / 0.45 / 0.30 / 0.20）。

**确认结论：错配现在依然存在。** 与 `指令/FTMOE_PROTOCOL020_PROBLEM_LOG.md` P36（2026-09-09 顶部条目，原文：“实际 `episode_hashes[*].profile` 与 dev500 主要容量及 RAM 适配配置一致……顶层候选 `profiles` 陈旧，不能据此误判未见性”）与 P28 行（“适配 manifest 顶层 profiles 为旧配置”）完全一致。**P021 必须以 `episode_hashes[*].profile` / raw manifest 为准，禁止引用顶层 `profiles`。**

### C2. `capacity_scan_dev/*/manifest.json` 的 `cohort` 字段与实际 VM 列表不一致

9 个 dev 候选 manifest 的 `cohort` 写 `"train"`，但其 `cohort_vm_ids` 是 86 个 **dev 组** VM（`[11, 26, 27, 37, 45, 47, …]`，与 `vm_split.json` `cohorts.dev` 逐元素一致）。目录命名（`phase_*`，seed=500）与 `capacity_scan/`（seed=410，cohort=train）也确实不同。此处 `cohort` 字段不可信，应以 `cohort_vm_ids` 为准。

### C3. `normalization_coverage_train.json` 的覆盖范围与「仅 train_blocks」口径存在张力

`artifacts/ftmoe_online/protocol_019/normalization_coverage_train.json` 的 `source` 指向 `artifacts/ftmoe_end_to_end/data/protocol_004_physical`、`train_blocks=[0,1,2,3,4]`、`rules.LOW_COVERAGE_COUNT=12`、`fallback='median of healthy peer maxima in same hardware group'`；该文件的 `hosts` 为 16 行。

其登记的可复核事实为：`hosts[0].cpu_demand.count_nonzero = 404`（`zero_ratio = 0.6`）、`hosts[0].ram_space.count_nonzero = 212`；而 P014 的 `train_blocks` 为 5×202 = 1010 行、八条 replay 全量为 8×202 = 1616 行（均取自 `artifacts/ftmoe_end_to_end/data/protocol_004_physical/manifest.json` 的 `train_blocks`/`block_size`/`seeds`）。**本审计无法据此复现 404 / 212 这两个计数**：该文件没有登记任何切片哈希或逐 host 行数，而 declaration（仅 train_blocks）与其自报计数之间缺少可复核的换算链。**因此仅记录这一张力（所有数值逐字取自文件），不判定孰是孰非**；P021 若要以「归一化仅拟合 train 分片」作为隔离论据，应先重新生成一份显式登记切片与哈希的 coverage 文件。

### C4. `docs/FTMOE_ONLINE_PROTOCOL_020.md` 的阶段表曾被订正

该文件 §5 表格后的注记写明：原表停在 2026-09-07 的 “S4 停止、S5–S10 未开始”，与文件头部 2026-09-09 状态行及 `artifacts/ftmoe_online/protocol_020/` 产物矛盾，已于 2026-09-10 依产物订正。本次审计核实的产物状态与该订正后的表格一致（S5 PASS、S6 完成、S7 完成、S8 完成、S10 未开始）。

### C5. P020 stage 记录为 S6，但 R1 保护阈值/方法选择属于同一链条

`artifacts/ftmoe_online/protocol_020/revision_20260909/r1/execute_registered_grid_primary.py:45-58` 的冻结网格使用 `drift_streams/dev_seed{500,501}_steps2000`（即 dev cohort 的两个流）来比较 A / C-legacy / C-residual-off / C-residual-on，其中 `C-residual-on` 的 `protection_enabled = true`。因此 **dev500 与 dev501 同时在“保护阈值/方法选择”和“开发评估”两个意义上被消耗**；021 的新 regime 必须同时避开这两个流所代表的分布与 VM 组。

### C6. `capacity_scan_report.json`（31 条候选）相对磁盘上的 `capacity_scan/`（79 个 manifest）是陈旧的

`artifacts/ftmoe_online/protocol_020/capacity_scan/capacity_scan_report.json` 的 `candidates` 共 31 条，其 `profile` 全部是 `disk_scale = 0.3` 的 cpu / ram / disk 单因子取值。而 `capacity_scan/` 下实际存在 79 个 `*/seed410_steps400/manifest.json`，其中 48 个（全部 `phase_*` 候选、`disk_*_lawlm|lawv1|lawv2` 变体、`cpu_*_am2|au|cm1.5`、`ram_*_rm*|ram_1`、`disk_0.35_dm*|disk_0.45_dm2`、`cpu_1`、`disk_0.2` 等）不在报告内。报告生成脚本 `analyze_ftmoe_protocol020_capacity_scan.py:174-199` 会枚举 `SCAN.iterdir()` 下所有含 `manifest.json` 的子目录（只在 `manifest.smoke` 或 `steps != 400` 时跳过，而本次统计 79 个 manifest 的 `smoke` 全为假、`steps` 全为 400），因此**用当前代码重跑会得到多于 31 条**。→ 报告是较早时点的快照；引用 S4 候选集合时必须区分「报告内 31 条」与「目录内 79 条」。

## 7. 产物

```text
artifacts/ftmoe_online/protocol_021/offline_coverage_audit/
    exposure_ledger.json      # 学习链逐源 exposure type + mechanism_exclusion + source_scope
    exclusion_registry.json   # 禁用复用集合（VM/时间窗/容量/适配器/磁盘律/相位/时间耦合/seed）
    exposure_ledger.md        # 本文件
```

数据源计数：protocol014 = 7、protocol019 = 16、protocol020_s6 = 20，合计 **43**。
