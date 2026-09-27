# 当前唯一任务：Protocol-032正常NLL容忍额试验
状态：planned_not_run。

[执行指示](docs/PROTOCOL032_SINGLE_TASK_DIRECTIVE_20260927.md) / [分析](docs/PROTOCOL031_INTERIM6800_ANALYSIS_20260927.md) / [计划JSON](artifacts/ftmoe_online/protocol_032/plan.json)。

031的6800行前缀已跑完C/D；两回归平均AP差D-C=-0.002791，复用0次，D回放耗时约为C的18倍。9869完整计划未完成，不继续生成。

032只检验一个因素：F0正常NLL容忍额增加固定0.01 nats绝对下限，保留FPR和其他验收门槛。在现有6799计分＋1保护行上完成一次C_fixed5/D_guard_budget试验。先补齐输入资格核验，不调数据，不降低其他门槛，不优化其他代码路径。交付或明确阻塞后停止。

实现取实验分支4e7df7e5f7abca2e2470633864b35440858bd4bf，同步本指示后可直接复用该分支；main此时保存最新计划与紧凑证据。
