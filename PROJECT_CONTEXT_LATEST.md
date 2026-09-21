# 当前上下文：Protocol-027

研究目标：合理业务更替场景下，仅检验动态残差专家D_dynamic与固定残差专家C_fixed5，replay seed700/model1；不扩展A/B、其他C、追加种子、消融或调参。

9维因果特征的slope4补齐错误已经修复，并在当前可恢复的5520步数据上通过原5e-5容差。当前尚无027模型性能结果，原因是物理数据恢复哈希仍未满足登记条件。

最新恢复证据：
- 登记来源run35254809015 / job105357470988：stream `46b1dbdd...`，final chunk `fc3e9887...`。
- preservation run35292728416 artifact10526683176中的state_5520直接来自原overlay-5520 artifact10516563749。
- main维护run35604239834恢复为 `fdea8430.../21cd7335...`。
- 旧成功run35596448071使用相同旧commit/workflow原样重跑attempt2后，也恢复为同一 `fdea.../21cd...`，因此main特征修复和维护workflow不是原因。
- run35114741391 artifact10469412410中的9/16孤立final chunk与当前恢复final chunk逐字节相同，21个NPZ数组全部完全一致。
- 已记录的GitHub runner image、Python、torch、numpy、DGL、dill、psutil版本在登记/成功复现/当前失败恢复之间一致。剩余最可能是最后一个模拟/调度interval的底层运行时或硬件敏感数值差异，但由于原登记完整文件未保存，尚不能做登记数据的数组级反向对比。

当前按协议处于数据恢复硬阻塞，C_fixed5和D_dynamic均未运行。见[当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)、[结果页](docs/PROTOCOL027_RESULTS.md)以及 `artifacts/ftmoe_online/protocol_027/maintenance_20260921/recovery_reproduction_blocker.json`。
