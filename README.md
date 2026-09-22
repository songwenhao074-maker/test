# PreGAN+ / FT-MoE 在线实验

当前目标：在合理的业务更替与复现场景中检验**动态残差专家D相对固定残差专家C的预测优势**。先研究D/C，再由用户决定A/B。

**当前唯一任务：[Protocol-027：一次C_fixed5/D_dynamic开发试跑](docs/PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)。每份指示只规划一个任务，交付后停止。**

## 从这里继续

1. 阅读 [当前交接说明](docs/GITHUB_EXPERIMENT_HANDOFF.md)。
2. 查看 [最新执行状态](NEXT_EXPERIMENT_LATEST.md) 与 [结果/阻塞](docs/PROTOCOL027_RESULTS.md)。
3. 从main手动运行 **Protocol-027 single D/C pilot**：`run_models=true`，首次`frozen_data_run_id`留空；工作流准备现存数据、审计、完整归档后只训练C/D。已有完整归档则填其run ID复用。

```console
git clone --depth 1 https://github.com/songwenhao074-maker/test.git FT-MoE
cd FT-MoE
```

工作流仅手动触发，推送不会自动准备数据或训练。不要从旧协议分支、上游 `main.py` 或025五方法入口启动当前任务。

## 已知事实

- 原027失败是特征错误和旧登记文件恢复问题，没有模型结果。特征已修复；2026-09-22用户授权独立数据revision002，直接复用已有数据并完整冻结。[修订说明](docs/PROTOCOL027_DATA_REVISION_002_20260922.md)。
- 最新已完成性能结果仍为024 v2c：C AP=0.669545、D AP=0.667100，成功复用0；不能声称D已经胜出。
- 027复用025六业务数据，5520步、replay seed700/model1，9个回归窗口。D允许额外后台计算和有限记忆；不声称严格同总成本。
- 历史B代表全量微调；当前C/D均冻结主干、训练残差专家。旧协议数值不能跨场景直接排名。

## 历史和依赖

[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [状态历史](docs/PROJECT_STATUS_AND_HISTORY.md) / [原交接与资产清单](docs/GITHUB_EXPERIMENT_HANDOFF_HISTORY_20260921.md)。历史STOP与D_eligible不构成027门禁，历史失败不改写为通过。

保留 `recovery/`、模拟器、调度器、Bitbrain数据、冻结checkpoint和被当前代码import的旧模块。实验输出按run_id隔离；大产物保存在有保留期的GitHub Actions artifacts，Git记录链接及哈希。

本仓库基于PreGAN+，附加FT-MoE研究线。上游许可：[BSD-3-Clause](LICENSE)。
