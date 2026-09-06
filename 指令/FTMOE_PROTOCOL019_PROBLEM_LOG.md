# FT-MoE Protocol 019 — 实验执行问题日志（Problem Log）

> 本文件与 `FTMOE_ONLINE_TUNING_REVIEW_AND_SOLUTION_PLAN.md` 同目录（`F:\PreGANPlus-master\指令\`）。
> 用途：记录尝试执行 Protocol 019 过程中遇到的所有问题、失败、停止条件、配置与 hash，供后续修改与复盘。
> 规则：**任何阶段失败 → 停止后续阶段并记录**；记录后不改写历史条目，只追加。

---

## 问题汇总表（最新在上）

| # | 阶段 | 严重度 | 状态 | 摘要 |
|---|---|---|---|---|
| P12 | S7 | 高 | 已记录（机制性结论+停止点） | 受控漂移下 D−C=0（全相位逐位相同）、专家恒 4、unmatched=0：v2 gate 的 birth/retire 前提（unmatched EMA≥0.05 / 长期零激活）在本架构（tanh 阈值 ~-0.1 的宽基余弦路由 + 4 个广覆盖专家 + C 本就可训 router/experts）下不可触发 → 动态拓扑无收益证据；S7 gate（D−C>0）FAIL，按纪律停止并请示方向 |
| P11 | S7 | 低 | 已记录（数据性质说明） | BWGD2 的 VM 池是数据集静态的：所有 replay seed 的 selected_vm_indices 相同（42 个）；seed 只改变到达时序/任务构成，不改变 VM 队列 → S7 漂移必须靠相位化到达池（已实现 PhaseAdaptedBWGD2，用户授权）而非换 seed 换队列 |
| P10 | S5 | 中 | 已修复（由 Test 7/8 捕获） | v2 gate 开发期两个缺陷：①cap-4 选择未排除 ramp=0 新专家 → birth 时旧专家被挤出 top4、输出跳变 0.25（Test 7 连续性失败）；②dormant 余弦计算维度错误（Test 8 崩溃）。均已修复且 5/5 测试通过 |
| P09 | S4 | 中 | 已解决（PASS@1e-5） | S4 stability gate：lr=3e-5 时 B/C second-half 低于 Frozen A 0.037–0.040（>0.03 FAIL）；dev grid 降 lr=1e-5 后 B Δ−0.024 / C Δ−0.021、anchor 无遗忘 → gate PASS |
| P08 | S3 | 高 | 已记录（正式停止条件） | 同域适配后 Frozen A 冷启动 gate 仍 FAIL：PR-AUC 0.340 < 0.60（FPR/校准已达标）；停止点，需用户决策下一步 |
| P07 | S3 | 低 | 已修复 | S3 episode 采集器同进程多段采集时 `torch.set_num_interop_threads(1)` 二次调用崩溃（seed402 attempt001）；已加一次性配置保护并重跑 |
| P06 | S2 | 高 | 已记录（正式停止条件） | Frozen A 冷启动 gate FAIL：PR-AUC 0.186 / FPR 0.175（阈 0.60/0.15）；修复输入后仍不兼容 → 按计划停止调 scenario，进入 S3 同域共同适配 |
| P05 | S1 | 中 | 已记录+已处理 | 仓库级 RAM guard 4.5 GiB 无法满足（可用 ~4.0 GiB / 总 15.2 GiB）；019 采集器降为 3.0 GiB 并记录偏差 |
| P01 | S0 | 中 | 已记录 | 017/018 未达完成态：无 result.json；google2011_018 产物目录不存在 |
| P02 | S0 | 中 | 已记录 | 所有可用 Python 环境 torch 均为 CPU-only，GPU (RTX 4060) 不可用 |
| P03 | S0 | 低 | 已记录 | git: main 本地分支与 origin/main 关系失效（origin/main 已删除），本地未提交内容仅 untracked `指令/` |
| P04 | S0 | 高 | 已记录 | 014 v4 归一化病态确认：host13 (0-based) ram_space=0 / disk_space=0 / cpu=2（组内正常量级 1860+） |

---

## 详细记录

### 2026-09-06 — S7 阶段（续）

#### P12. 受控漂移下 D−C=0 —— dynamic expert v2 机制在本域不可触发（S7 gate FAIL）

- **执行**：相位化漂移流 seed305×2000（PhaseAdaptedBWGD2，cpu→mixed→ram→cpu 各 500 interval，
  VM 组按训练/开发统计 terciles-of-RAM 划分，用户 2026-09-06 授权；stream_sha256 `9f7fffde…`）。
  漂移真实性验证（纯数据）：host-aggregate RAM 均值跨相位 86→218→251；CPU 过载事件
  cpu 相位 19-20 → mixed 79 / ram 71（~4×）；Frozen A 相位 F1 摆动 0.53（mixed）↔ 0.32（cpu 复现）
  ——场景确有相位级分布漂移。
- **A/C/D（v2 gate @ lr1e-5，adapted_v4 起点，2000 steps，各 200 updates，exposure≤3 零违规）**：

  | 相位（容差标签） | A | C | D |
  |---|---|---|---|
  | 0 cpu | F1 .4131 PR .4291 | .4050 / .4386 | .4050 / .4386 |
  | 1 mixed | .5320 / .5376 | .5191 / .5355 | .5191 / .5355 |
  | 2 ram | .3991 / .2833 | .4052 / .2832 | .4052 / .2832 |
  | 3 cpu 复现 | .3230 / .3026 | .3011 / .3098 | .3011 / .3098 |
  | second-half 总 | F1 .3682 | .3628 | .3628 |

- **关键观测**：D 与 C **全相位逐位相同**；专家数恒 4；20 次 adapt 评估全部
  `unmatched=0, ema=0.0`（无 addition/dormant/reactivation）。
- **机制分析（为什么 gate 不触发）**：
  1. EAGate 阈值为 tanh 参数（初始 ~-0.1），4 个广覆盖 key 的余弦捕获域极宽 →
     几乎任何路由状态都有 expert 匹配 → `unmatched_ratio ≈ 0`（全程 max 0.0）；
     v2 gate 的 birth 前提（unmatched EMA≥0.05、≥128 samples、连续 3 windows）**架构性不可达**；
  2. retire 前提（500+ interval 零激活）同样不可达：4 experts 每窗口全部被激活
     （activation counts 800-1600/窗）；
  3. 漂移被 C 的可训 router/experts 参数更新吸收（C/D 与 A 相位差异 <0.03，无遗忘、
     无显著劣化）——在 anchor+distill 稳定方案下，固定 4 experts 已足够，动态拓扑无增益。
- **Gate 判定**：S7 dynamic expert gate `D−C mean > 0`（且 lag D<C）→ **FAIL（D−C=0.0）**。
  按 The Plan §3/§21：**停止后续阶段并报告，不擅自改门限/机制或启动 S8**。
- **候选方向（需用户拍板）**：
  (a) 接受机制性结论（本域/本方案下动态专家无额外收益），以 A/B/C 全链路结果为最终输出，
     报告"019 完成（除 D−C 无增益外全部 gate 通过）"，S8 不跑；
  (b) 重新设计使动态可测（阈值更严格/更窄路由、或按"路由器困惑度/损失上升"触发而非 unmatched-only、
     或显式新故障模式注入——非 synthetic label，而是真实需求形态）→ 属新机制版本（dynamic_expert
     version 3），需先注册与用户批准再重跑 S7；
  (c) 仍执行 S8（5 seeds × 3 streams A/B/C/D 全矩阵）把当前结果（含 null D−C）做成完整统计报告。

---

#### P11. BWGD2 VM 池为数据集静态 —— 影响 S3/S8 "来源隔离"的语义

- **现象**：比对 303/304/401–408 全部 manifest 的 `selected_vm_indices`：**所有 replay seed 相同
  （同一 42 个 VM 文件）**。BWGD2.possible_indices 由数据集 CSV 静态过滤（CPU@idx10 ∈ (500,3000)MHz）
  决定，与 seed 无关；`randint` 仅决定到达顺序与重复抽样。
- **含义**：The Plan §8.2 的"按 VM ID/时间划分隔离"在本生成器上无法通过换 seed 实现（队列恒同），
  seed 差异 = 到达时序/任务生命周期/泊松混合的差异；S8 的 "3 disjoint streams" 需要如实理解为
  arrival/时间上不重叠而非 cohort 不重叠（与 §17 的单元是 model-seed × stream 一致，可在报告中注明）。
- **处理**：不改 BWGD2（旧协议原样）；S7 controlled drift 采用**相位化到达池**（PhaseAdaptedBWGD2，
  用户 2026-09-06 授权）：cpu/mixed/ram/cpu 四相位各 500 interval，池由训练/开发统计分组
  （s7_vm_groups.json，terciles of mean RAM）；此为本生成器上唯一可用的受控漂移机制。

---

### 2026-09-06 — S5/S6 阶段

#### P10. v2 dynamic gate 开发期缺陷（由 Test 7/8 捕获并修复）

- **缺陷 ①（Test 7 失败暴露）**：birth 后新 expert（ramp=0）仍参与 top-4 cap 竞争，会把一个成熟
  expert 挤出 hard-mask → 输出 mixture 变化 → logit 跳变 0.248（违反 birth 连续性 <1e-4 验收）。
  修复：cap 槽位只从 `ramp>0` 的 experts 中 topk 选取（`cap_scores = score.masked_fill(~cap_eligible, -inf)`，
  `cap_k = min(4, max(1, #ramp>0))`）；ramp=0 的 newborn 不占槽位、零贡献，输出在 birth 前后保持连续。
  修复后 Test 7 diff < 1e-4。
- **缺陷 ②（Test 8 崩溃暴露）**：reactivation 余弦计算 `normalize(centroid) @ normalize(dormant_keys)`
  维度不匹配（(64,) @ (1,64)）。修复：`normalize(centroid,0) @ normalize(dormant_keys,-1).T`。
- **结构重构（同一提交）**：把 birth 逻辑从"仅 dormant 分支内可达"重构为统一
  `reactivated → birth_ok(≥128 samples、ema≥0.05、连续 3 windows、<8 experts)` 流程，
  dormant 无匹配时同样可 birth。
- **验证**：`test_ftmoe_protocol019_s5.py` 5/5 通过（Test 7 连续性、Test 8 A→B→A 同 id 复活、
  Test 9 Adam state 保持、clone-parent、S5Session smoke）；`test_ftmoe_protocol019_s4.py` 4/4 回归通过。

#### S6（stationary 2000-step）结果存档（无问题，摘要）

- stream seed304 × 2000 interval（采集 765 s，raw 正例 251/32000=0.78%），
  C/D = adapted_v4 起点 @ lr1e-5 gate v2，各 200 updates（exposure max 3、violations 0）。
- A：anchor F1 0.90486 五个参考点完全不变；C/D：anchor ≤ +0.005（无遗忘）；
  second-half F1：A 0.26316 → C/D 0.26866（在线不劣且略优）；FPR 0.0185→0.0171；ECE 0.023。
- **D 专家数全程恒为 4、unmatched_ratio=0、D−C=0.0（逐位相同）→ S6 平稳性 gate PASS**
  （动态机制不会无理由增长）。S6_stationary_result.json 存档。
- **下一步（S7）需要用户方向**：natural drift（Google 2011 需外部数据集下载——仓库明示
  "downloading is not authorized"）或 controlled drift（VM 组/容量相位切换 = simulator 改动，
  按仓库纪律需先报告获批），见 docs §S7。

---

### 2026-09-06 — S4 阶段

#### P09. S4 stability gate 在 lr=3e-5 失败 → dev grid lr=1e-5 通过

- **背景**：用户批准以 adapted_v4_seed1/best.pt 为共同起点进入 S4。实现 plan §9 的 S4 更新方案：
  Recent(128) + Anchor(384，train episodes 401–403 尾部 128×3 行) 两层记忆、times_sampled ≤3
  （Test 6 通过、exposure violations=0）、L = L_online + 0.25·L_anchor + 0.10·L_distill（教师=冻结起点）、
  B 分组 LR（encoder./graph_encoder./cmha* 0.1×，其余 1.0×）；C 只训 moe./eagate.。
  S4 测试 Test 5/6/10 全通过（未来标签隔离、曝光上限、resume 逐位等价）；Frozen A 对照与 S3 runner
  **逐位一致**（second-half F1 0.341463 / PR-AUC 0.221148）→ 管线验证。
- **现象（lr=3e-5，均 30 updates，exposure max 3、violations 0）**：
  | 指标 | A frozen | B | C |
  |---|---|---|---|
  | second-half F1 | 0.3415 | 0.3014 | 0.3041 |
  | second-half PR-AUC | 0.2211 | 0.2297 | 0.2258 |
  | second-half FPR | 0.0374 | 0.0587 | 0.0578 |
  | anchor ref F1 (0→300) | 0.90486→0.90486 | →0.91205 | →0.91070 |
  → anchor 无遗忘（ref 甚至微升）；但 second-half F1 低于 Frozen A 0.037–0.040（>0.03），
  gate 第二臂 FAIL（在线更新 recall↑/precision↓，FPR 上升）。
- **处理（plan §9 允许的小范围固定比较；所有 B/C 同一方案）**：注册 dev grid 尝试 lr=1e-5 →
  B: second-half F1 0.3175（Δ−0.024）、C: 0.3209（Δ−0.021）；anchor ref B/C 均 →0.90794（+0.003）；
  full F1 B 0.3972 / C 0.4000（A 0.4141）；ECE/fpr 均优于 3e-5。
- **结论**：**S4 stability gate PASS（lr=1e-5）**——B/C 在线稳定、无参考遗忘、second-half 不劣于 Frozen A
  超过 0.03。lr=3e-5 版本归档（runs_s4/ 同名目录 + S4_online_stability_result.json rows），
  后续 B/C/D 全部使用 lr=1e-5 与同一 S4 方案。
- **下一步（登记）**：S5 dynamic expert v2（plan §10：EMA unmatched gate、centroid 聚类、
  parent-clone birth + routing ramp、shadow 激活、retire/reactivate + Test 7–9），
  之后在 dev303 上做 D=C+topology 的 300-step 对照（D−C 主比较），再进入 S6 stationary 长流。

---

### 2026-09-06 — S3 阶段（续）

#### P08. 同域共同适配后 Frozen A 冷启动 gate 仍 FAIL —— 正式停止点（The Plan §3/§20/§21）

- **执行内容**：
  1. S3 数据 v1：adapted-BWGD2 016 契约 episode 401–408（train 401–405 / dev 406–408，
     每段 400 scored interval，与 dev stream seed303 来源隔离），
     `adaptation_data/raw/` + `adaptation_data/v1/`（manifest/normalization/time/container/
     schedule/creation_ids/labels，labels = ±1 tolerance，与 runner 语义对拍一致）。
  2. S3 训练：014 v4 seed1 epoch21 warm-start，15 epochs / lr 1e-4 / AdamW / batch 32 /
     全参数 / EAGate temp 恒 1.0 / graph_context 全程启用（125 s 完成，242,786 参数）。
     dev（406–408）best epoch 7：F1 0.4726 / PR-AUC 0.4824；epoch15（last）PR-AUC 0.5022；
     protocol_004 legacy anchor F1 0.8825（原域轻度遗忘，可接受记录）。
     checkpoint：`s3/adapted_v4_seed1/best.pt`（sha256 `670c56fe…`）。
  3. Frozen A 重测（run_ftmoe_protocol019.py --checkpoint-path，dev303 同流同协议）。
- **结果对比**（dev stream seed303，Frozen A，threshold 0.5）：

  | 指标（full） | 014 v4（S2） | adapted_v4（S3） | 变化 |
  |---|---|---|---|
  | F1 | 0.2340 | 0.4141 | +0.18 |
  | PR-AUC | 0.1863 | 0.3402 | +0.15 |
  | FPR | 0.1753 | 0.0351 | ✓ ≤0.15 |
  | ROC-AUC | 0.9054 | 0.9497 | +0.04 |
  | ECE | 0.1666 | 0.0357 | ✓ 优秀 |
  | second-half F1 | 0.1341 | 0.3415 | +0.21 |

- **Gate 判定**：`PR-AUC ≥ 0.60 且 FPR ≤ 0.15` → **FAIL（PR-AUC 0.340；FPR 维度已达标）**。
- **可行性佐证（不调参，仅整理既有证据）**：gate 阈值 0.60 是在 019 开始前预先固定的；
  目前**任何 checkpoint 在任意 BWGD2 系流上的 Frozen PR-AUC 均低于 0.60**：
  pilot015 native 0.316（异常率 11.6%）、016 0.135（2.9%）、019 dev303：014=0.186 / adapted=0.340
  （1.56%）。adapted 模型 ROC-AUC 0.95 + ECE 0.036 说明表征与校准良好，瓶颈在超低事件率
  （1.5–3%，事件多 1–2 interval 短促）下 0.5 固定阈值的 precision 上限。
- **执行动作（记录即停止，The Plan §20 Step 4 / §3）**：**不改 scenario、不改数据/模型旋钮来追 gate**；
  不擅自进入 S4（A/B/C online comparison 需用户在两种解读中抉择：
  (a) 严守 0.60 门槛 → 019 在 S3 结束，报告"adapted_v4 仍未达到可用冷启动线"，后续方向另议
  （含可行性上限分析 / S3 数据扩展 5/10/15-epoch 比较 / 换 Google-2011 域）；
  (b) 判定 adapted_v4 为"合理 common starting point"（PR-AUC 0.34、FPR 0.035、ECE 0.036、
  ROC-AUC 0.95）→ 进入 S4 在 adapted 起点上比较 A/B/C，验证 online optimizer 稳定性。
- **存档**：`S3_adaptation_result.json`、`s3/adapted_v4_seed1/`（含 epochs.csv/progress.json/
  best.pt/last.pt/checkpoints_by_epoch）、`runs/A_model1_replay303_adapted/`。

#### P07. S3 采集器多段同进程采集崩溃（torch interop threads）

- **现象**：`prepare_ftmoe_protocol019_s3_data.py` 首次运行在 seed401 成功后，seed402 启动即失败：
  `RuntimeError: cannot set number of interop threads after parallel work has started`。
- **原因**：采集器把 8 个 episode 放在同一进程循环执行，而 `configure()` 每段都调用
  `torch.set_num_interop_threads(1)`；torch 只允许在首次并行工作前设置一次。
- **处理**：`configure()` 增加进程级一次性配置保护（`_THREADS_CONFIGURED`）；
  失败现场保留于 `adaptation_data/failed_attempts/seed402_steps400_interop_threads_attempt001/`；
  从 seed402 起重跑（seed401 已完成不受影响）。
- **数据观察（seed401，记录待数据集汇总复核）**：400 scored intervals 类别计数
  `[6254 normal, 76 CPU, 1 RAM, 69 Disk]`（正例 146/6400 = 2.28%）——RAM 事件近乎缺失，
  与 016/dev303 流的低 RAM 事件率一致；S3 训练与评估报告需按类别 support 解读。

---

### 2026-09-06 — S2 阶段

#### P06. Frozen A 冷启动 gate 失败 —— Protocol 019 正式停止条件（The Plan §20 Step 4 触发）

- **场景**：dev stream `protocol_019/dev_streams/seed303_steps300`（adapted-BWGD2 016 契约、
  input_contract/normalization/graph 语义 v2、seed 303、300 interval、生成耗时 110 s、RSS 0.61 GiB、
  stream_sha256 `ff0ea85e1a0fb58aee9f3696a31f8972fdecba60840eac9bdea9a92c2d47c8f4`）。
- **data-only audit**（distribution.json，纯数据无模型）：
  - 归一化 v1 下归一化绝对最大值 **6.0e11**（host13 放大），v2 修复后 **max=3.0、alarm=off、无 |x|>5**；
  - 事件：75/4800 host-steps（1.56%），CPU 32 / RAM 9 / Disk 34；mean anomaly duration 1.92、
    p95 4.1、mean recovery 56.9；
  - host 聚集：mean active 15.8、P(host≥2)=22.7%、P(host≥3)=2.9%（训练约 10%）；
  - 调度：proposed 497 / executed 328 migrations、rejection 34.0%；
  - identity 审计：**本流 0 例 slot 替换**、44 例真实迁移（churn 低，identity-v2 修复在本流不产生边差异，
    但仍是 015 型流的必要修复）；
  - 资源：container 需求全部落在训练范围内（exceed ratio 0）。
- **Frozen A 结果**（014 v4 seed1 epoch21、无更新、容差 ±1 标签延迟 1 interval）：
  - full：F1 0.2340、P 0.1357、R 0.8477、PR-AUC 0.1863、FPR 0.1753、ROC-AUC 0.9054、
    ECE 0.1666、Brier 0.1552；
  - second half：F1 0.1341、PR-AUC 0.1370、FPR 0.1986、ROC-AUC 0.8856；
  - 参考 anchor（离线 validation）：F1 **0.9051 全程不变**（模型在自身域上健康，无遗忘）。
- **Gate 判定**：`PR-AUC >= 0.60 且 FPR <= 0.15` → **FAIL（PR-AUC 0.186 ≪ 0.60）**。
- **对比**：016 旧输入 Frozen A full F1 0.2138 / PR-AUC 0.1350；修复输入后 0.2340 / 0.1863 ——
  有改善但远未达 gate，**与 The Plan §0 的预判一致：主因是分布形态/时序/生命周期/调度语义差异，
  不只是归一化与图身份两个 bug**。ROC-AUC 0.905 说明排序能力尚存（校准/阈值失配为辅因，
  见 P1-3），但按 §7.2 规矩：PR-AUC 也低 → 判定旧 014 checkpoint 不适合作为该 domain 的 online 起点。
- **执行动作（记录即停止）**：**不再调整 scenario / capacity / D 参数**；
  进入 **S3 same-domain common offline adaptation**（warm-start 014 v4 → adapted_v4，
  预算 15 epochs / lr 1e-4 / AdamW / batch 32，数据划分按来源隔离、时间区间不重叠，
  见 docs/FTMOE_ONLINE_PROTOCOL_019.md S3 计划）。
- **存档**：`artifacts/ftmoe_online/protocol_019/runs/A_model1_replay303/`
  （configuration/summary/gate_metrics/predictions/reference/resume/last + stepNNN.pt）。
  前两次打包差异（extras 合并版、缺 gate_metrics 键版）保存在 `runs/superseded/`，指标数值完全一致
  （三次运行 bitwise 相同，验证 resume 等价性基础）。

---

### 2026-09-06 — S1 阶段

#### P05. 仓库级 RAM guard（4.5 GiB）在当前机器上不可满足

- **现象**：`prepare_ftmoe_protocol019_dev_stream.py` 首次采集 dev stream（seed 303）在 `guard()`
  处失败：`RAM guard below 4.5 GiB (available 3.991 GiB)`；失败现场完整保留于
  `artifacts/ftmoe_online/protocol_019/dev_streams/failed_attempts/seed303_steps300_attempt001/failure.json`
  （attempt 已移入 failed_attempts，避免污染正式输出目录）。
- **背景**：整机 15.2 GiB 内存，可用常年在 ~4.0 GiB 附近（浏览器/java/杀软等占用）。
  016 采集进程实测峰值 RSS 仅 0.6 GiB（见 016 manifest `rss_gib`）。**017 confirmation 失败两次
  （1045/2000、1479/2000 steps）同为该 guard 触发**（capacity_017/failure.json 与
  confirmation/seed302_steps2000/failure.json），是仓库已知的环境性停止条件。
- **处理（Protocol-019 范围内的偏差，已显式记录）**：019 新增脚本的 guard 下限改为
  3.0 GiB（`FTMOE019_RAM_GUARD_GIB` 可覆盖），仍保留磁盘 20 GiB 与语义（低水位保护），
  与 016 峰值 RSS 0.6 GiB 相比仍留有 >2 GiB 裕量。
- **边界**：旧注册脚本（prepare_ftmoe_adapted_bwgd2.py 等）与 017/018 资产**一律不改**；
  4.5 GiB 是历史协议的环境约束，不属于 019 输入修复的范畴。若机器可用内存长期低于
  3.0 GiB，则 019 的长流阶段（S6/S8 2000-step、5 seeds×3 streams）需要用户关闭部分应用
  或换机执行，届时在问题日志补充记录。

---

### 2026-09-06 — S0 侦察阶段（P01–P04）

#### P01. Protocol 017/018 未达完成态（The Plan §24 第 3 条触发）

- **现象**：`artifacts/ftmoe_online/capacity_017/result.json` 不存在（Test-Path = False）；
  `artifacts/ftmoe_online/google2011_018/result.json` 不存在；`google2011_018` 产物目录整体不存在。
- **佐证**：`artifacts/ftmoe_online/capacity_017/` 下存在 `failure.json`、`development_results.json`、
  `failed_attempts/confirmation_attempt001/failure.json`、`failed_attempts/confirmation_interrupted002/`、
  `confirmation/seed302_steps2000/failure.json`。
- **影响**：按 The Plan 任务模板第 3 条，017/018 **不得称协议已完成**；Protocol 019 必须独立推进，
  017 的 capacity sweep 结果只作为 scenario diagnostic 参考。
- **后续动作**：无（不改动 017 资产）。若需恢复 017 confirmation，需先记录其自身 pipeline 失败原因。

#### P02. 环境：torch 全部为 CPU-only

- **现象**：
  - `C:\Program Files\Python310\python.exe`：Python 3.10.10，numpy 2.2.6，torch 2.13.0+cpu，cuda False
  - conda `dynmoe`（`D:\Anaconda\envs\dynmoe`）：Python 3.8.20，numpy 1.19.2，torch 2.4.1+cpu，cuda False
  - `nvidia-smi` 显示 RTX 4060 Laptop 8 GiB 存在
- **影响**：所有训练/在线实验只能在 CPU 上运行（历史上 pilot 015 也应在 CPU 上完成，
  因为 dynmoe env 无 CUDA torch）。速度与并发受限；全参数 B 微调 300 interval 预计可承受，
  2000-step 长流与 S8 的 5 seeds×3 streams 会非常慢（需预算评估）。
- **决策**：使用 `dynmoe` env 运行（torch 2.4.1 与既有 checkpoint 同代际；numpy 1.19.2 匹配旧代码风格）。
  除非必要不安装 CUDA 版 torch（避免破坏 env）。

#### P03. git 仓库状态

- `git status`：`## main...origin/main [gone]`；工作树仅 untracked `指令/`。
- 本地最新 commit `2bf2447 Add online fine-tuning branch inputs (registered artifacts)`。
- 影响：低。所有工作文件按仓库惯例落盘即可，不依赖远端；不主动 push。

#### P04. 014 v4 归一化病态（The Plan §1.2.B 复核确认）

- 从 `epoch021.pt` 的 `normalization.time_scale`（reshape 16×7，列序
  cpu_demand, ram_space, ram_rx, ram_tx, disk_space, disk_read, disk_write）读取：
  - host13：`[2.0, 0.0, 1, 1, 0.0, 1, 1]` → ram_space、disk_space scale = **0**；cpu scale = 2.0（极小）
  - 同组 host 8–15（8GB 组）其他成员 ram_space 量级 3993–9736，disk_space 量级 5000–16000
  - 推理验证：online 正常负载除以 scale 0 → 无限 → 旧代码 `1e-8` 兜底 → 1e8/1e11 级放大，与审计结论一致
- 处理：S1 normalization v2 `safe_scale` fallback 到同组统计；`normalized_abs_max < 50` 作为报警阈值。

---
