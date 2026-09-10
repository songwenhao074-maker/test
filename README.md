<h1 align="center">PreGAN+</h1>

> **本地 FT-MoE 实验入口：** [项目最新上下文](PROJECT_CONTEXT_LATEST.md) · [文档索引](docs/README.md)。协议 014 消融已由用户验收并保存为[回档基线](docs/BASELINE_ACCEPTED_20260905.md)，原筛选结果保留。在线阶段已进入协议 020：R0-A 因果基础通过、R0-B 动态部署未通过；R1（冻结基础＋固定在线修正）完成 8×2000 步开发对照，减轻旧 C 漂移退化但尚未稳定超过 A。下一步为[离线覆盖审计与新模式注册](指令/FTMOE_PROTOCOL020_UNSEEN_REGIME_PLAN_20260909.md)，尚未开始；R2–R6 与 S10 未实施。历史在线 015/016 小试未达标（016 冻结 A 后半程 F1=0.143911，B/C/D 未运行），预留测试未使用。

<div align="center">
  <a href="https://github.com/imperial-qore/PreGAN/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
   <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FPreGANPlus&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
   <a href="https://github.com/imperial-qore/PreGAN/actions">
    <img src="https://github.com/imperial-qore/COSCO/workflows/DeFog-Benchmarks/badge.svg" alt="Actions Status">
  </a>
 <br>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo" alt="Docker pulls yolo">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx" alt="Docker pulls pocketsphinx">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas" alt="Docker pulls aeneas">
  </a>
</div>

Typical mobile edge computing infrastructures have to contend with unreliable computing devices at their end-points. The limited resource capacities of mobile edge devices gives rise to frequent contentions, node overloads or failures. This is exacerbated by the strict deadlines of modern applications. To avoid failures, fault-tolerant approaches utilize preemptive migration to transfer active tasks across nodes and prevent nodes running at capacity. However, prior work struggles to dynamically adapt in settings with highly volatile workloads or even accurately detect and diagnose anomalies for optimal remediation. To meet the strict service level objectives of contemporary workloads, there is a need for dynamic fault-tolerant methods that can quickly adapt to changes in edge environments while having parsimonious remediation in the form of preemptive migration to avoid stressing the system network. This work proposes PreGAN, featuring a Generative Adversarial Network (GAN) based approach to predict contentions, pinpoint specific resource types with high chance of overload, and generate migration decisions to proactively avoid system downtime. PreGAN leverages coupled-simulations to train the GAN model at run-time and a few-shot fault classifier to update decisions of an underpinning scheduler. We also extend it to PreGAN+ that also periodically tunes the decision model using semi-supervised training and a Transformer based neural network for low tuning time, albeit with higher memory overheads.  Experiments on a Raspberry-Pi based edge environment demonstrate that both models outperform state-of-the-art baselines in fault detection and diagnosis scores by up to 12.5% and 31.2% respectively. This also translates in improvements in Quality of Service against baseline approaches.

## Quick Test
Clone repo.
```console
git clone https://github.com/imperial-qore/PreGANPlus.git
cd PreGAN/
```
Install dependencies.
```console
sudo apt -y update
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
python3 -m pip install "torch>=1.11" "dgl>=1.1"
export PATH=$PATH:~/.local/bin
```
The default recovery method is `FTMoERecovery`, a local implementation inspired by the accompanying FT-MoE paper. It extends PreGAN+ with schedule-aware graph encoding, adaptive Top-any MoE routing, cross-attention fusion, and MoE-only online tuning. Dataset, metric and ablation differences are documented in the current project context; this is not a claim of an exact paper reproduction. Use `-r` to select a baseline (`preganplus`, `pregan`, `pcft`, `dftm`, `eclb`, or `cmodlb`).

```bash
python main.py -r ftmoe
```

The bundled dataset is retained for compatibility with PreGAN+. Current offline full-model experiments use `artifacts/ftmoe_end_to_end/data/protocol_004_physical` and `train_ftmoe_end_to_end.py`; the simulator quick test above is a separate workflow. The paper's edge-fault dataset is not bundled, so local scores are not directly comparable with its reported results.

## External Links
| Items | Contents | 
| --- | --- |
| **Pre-print** | (coming soon) |
| **Video** | https://youtu.be/Pp82aZu5dJw |
| **Contact**| Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))  |
| **Funding**| Imperial President's scholarship |


## License

BSD-3-Clause. 
Copyright (c) 2022, Shreshth Tuli.
All rights reserved.

See License file for more details.
