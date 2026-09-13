# FT-MoE / FTIoT 当前项目状态（唯一默认入口）

> **给后续模型的读取规则：默认只读本文件和根目录 `NEXT_EXPERIMENT_LATEST.md`。不要递归读取 `artifacts/`、`指令/`、历史 Protocol 文档或 problem logs。只有当本文件明确指向某个证据文件，且需要核查具体数值/bug 时，才读取那个文件。**

## 1. 项目目标

本项目研究 IIoT/边缘计算中的在线故障预测与容错。16 台 Raspberry Pi 组成边缘集群，Edge Broker 负责调度/迁移。每个 host 的输入资源特征为 7 维：CPU、RAMSize/RAMRead/RAMWrite、DiskSize/DiskRead/DiskWrite；同时使用调度/迁移图。

模型 FTIoT / FT-MoE 为双路径结构：
- MoE 路径学习异构资源故障模式；
- GATv2 路径学习调度/迁移依赖；
- Cross-Attention 融合；
- 输出 host-level fault detection 与 CPU/RAM/Disk fault classification。

在线实验最终目标只有一个：

> **在合理的 recurring heterogeneous non-stationarity 场景下，证明 D（动态新增/休眠/再激活专家）相对于 Frozen A、Full-online B、固定专家 C，以及参数/训练预算公平对照，具有更好的在线适应与复现恢复能力。**

不是目标：证明 D 在所有静态/单一场景中都优于 C。

## 2. 当前 Git 状态

- 当前工作分支：`protocol-023`
- 本次精简前 HEAD：`288fca0b470951a4e28955589ae5af31bd3ba834`
- 开发 replay seed：`700`
- 最终确认 seeds：`701, 702, 703`，**目前未使用，继续冻结**。
- 冻结基础 checkpoint：`artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt`
- Protocol023 校准开发流：`dev_seed700_steps2880_calibrated`
- 开发流 SHA256：`54d155260a0cac71ff5d383f9c3b7270ff9ac96c4209881c2dbd1bd63bffd0f2`

## 3. 当前实验场景

Protocol023 使用三个 recurring regime：

1. **A / compute-first**：CPU → RAM → Disk。继承 Protocol022 cascade_v2，A 不再修改。
2. **B / memory-first**：RAM → Disk → CPU。
3. **C / io-first**：Disk → CPU → RAM。

开发时间线：

```text
F0 300 -> A1 420 -> B1 420 -> C1 420 -> F1 240 -> A2 360 -> C2 360 -> B2 360
```

模型输入禁止使用 `phase_id/regime_id/mechanism_id/cascade flag/future label`。

数据层已经足够继续模型实验：
- H1-original：FAIL，永久保留，不重写；原因是原 full-window follow-up 定义与 task 生命周期不匹配。
- H1-v2：PASS。
- H2-v2 specialization：PASS。
- 校准后 prevalence：A 6.80%、B 5.45%、C 6.88%。
- primary-window follow-up：A 185、B 405、C 428。
- independent fault events：A 410、B 478、C 500。
- io-first h=1 specialist probe 已从 Round1 的极少样本改善到可用，但仍属于较弱第三机制，确认阶段必须重新验证。

**从现在开始冻结 Protocol023 calibrated generator，不再为了模型结果继续调 A/B/C 场景。**

## 4. 当前 Fixed-C 配置与结果

Round2A 的固定 C：4 个 residual experts；基础网络冻结。在线配置：

```text
update every 4 intervals
batch = 32
1 gradient step / opportunity
LR = 1e-4
AdamW
replay window = 64 intervals
anchor weight = 0.25
distillation = 0.1
detection/resource/ranking weights = 0.7 / 0.3 / 0.5
```

总计 2880 intervals、720 次 update；prequential integrity PASS，预测都发生在对应 label/update 之前；基础参数未改变。

但该配置**没有真正学会新 regime**。first-exposure late-100 PR-AUC 相对 Frozen A：

```text
compute-first  +0.0049
memory-first   +0.0020
io-first       +0.0203
```

预注册学习门槛为 `+0.03`，0/3 通过。因此：

```text
STOP-NO-LEARN
D_eligible = false
```

因为 C 尚未显著学习，所以不能把 probe 变化解释成 catastrophic forgetting。

## 5. 成熟梯度结论

Round1 的梯度实验因 residual 为 zero-output 初始状态而不可用；Round2A 已在 `C_after_A1/B1/C1` 成熟 checkpoint 上重新测量，所有参数组梯度均有效。

结果仍没有发现预注册的负梯度冲突：主要 regime pair 的 mean cosine 约 `+0.71 ~ +0.95`，negative-pair fraction 远低于 30%。因此旧假设“D 必须通过解决强负梯度冲突获益”目前**不被支持**。

保留该负结果，不降低门槛，不重写为 PASS。

## 6. 对 D 的当前科学判断

现在不能实现/宣称 D 有效，原因不是数据层失败，而是：

1. 当前 Fixed-C 在线更新太弱，几乎没有形成可测的新知识；
2. 既没有显著学习，也没有显著遗忘；
3. 成熟梯度总体同向，没有证据支持“强梯度冲突”这一 D 理由。

但这不等于动态专家没有价值。下一步将把科学问题改为更直接的两步：

1. **先找到一个在有限部署预算内确实能学习新 regime 的强 Fixed-C（C\*）。**
2. **再检查 regime-specific specialist 是否显著优于共享 C\*。** 如果存在稳定 specialist gap，则动态专家可以被解释为“在线按需形成并复用 specialist”，而不要求负梯度 cosine。

这个新假设必须在 D 实现前验证。

## 7. 历史上必须记住的两件事

- Protocol020 旧 Dynamic-D 的 normalized Top-k/newborn 方案存在结构性不连续，成熟边界曾出现约 3.47 的 logit jump。**以后 D 只能使用 additive residual + continuous ramp，禁止复用旧方案。**
- Protocol022 单一 unseen regime 中，固定 C 随训练预算增加仍持续提高，因此单一长期 regime 不能证明动态专家必要。Protocol023 的 recurring multi-regime 场景因此保留。

## 8. 唯一下一步

读取根目录：

`NEXT_EXPERIMENT_LATEST.md`

它是当前唯一有效的实验执行文档。历史 `指令/FTMOE_PROTOCOL0xx_*.md` 与旧 problem logs 只作为审计归档，不再作为默认执行依据。

## 9. 只有需要核查时才读的证据

最多读取下列单文件，不要扫描目录：

- 当前 Gate：`artifacts/ftmoe_online/protocol_023/round2a/gate_status_round2a.json`
- C 在线配置/完整性：`artifacts/ftmoe_online/protocol_023/round2a/fixed_c_prequential/arm_C/summary.json`
- C 学习/遗忘诊断：`artifacts/ftmoe_online/protocol_023/round2a/fixed_c_prequential/forgetting_v2.json`
- 成熟梯度：`artifacts/ftmoe_online/protocol_023/round2a/mature_gradient/summary.json`
- H1-v2 数据：`artifacts/ftmoe_online/protocol_023/round2a/data_calibration/data_gate_v2.json`

除调试具体异常外，不需要读取更早 Protocol 的详细日志。