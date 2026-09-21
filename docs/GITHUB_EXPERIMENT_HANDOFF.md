# 当前实验交接：Protocol-027

以默认分支 `main` 为准。其他协议分支和历史文档用于查证，不是当前执行入口。

唯一任务：[六业务更替下的一次C_fixed5/D_dynamic试跑](PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)。每份指示只安排一个任务；交付后停止。A/B、其他C、消融、额外种子和调参不在范围内。

## 本次失败与修复

run35596448071已恢复正确物理数据，但9维特征审计失败，两组模型均未运行。原报告遗漏具体阻塞，不能理解为D性能失败。

补齐窗口的四步斜率现按生成器定义处理：历史不足四个lag时为0。预测、回放训练、候选和正常guard使用相同历史长度元数据；它仅用于补齐处理，不作为可学习输入。不放宽5e-5容差，不改物理数据及原失败证据。

## 当前剩余阻塞：恢复文件哈希

维护run35604239834在恢复完整文件后未匹配原登记SHA，C/D均未运行。不要直接重试训练，也不要将当前恢复文件的新SHA替换登记值。

已核验当前恢复数组的前5520行与27个保全chunk加transient完全一致；Windows与Linux两次恢复的完整数组也一致。但原登记完整文件没有留存可下载副本，因此尚不能确认最后一行及整体与原登记数据等价。跨系统ZIP元数据解释两次新副本之间的文件SHA差异，不能据此解释原登记SHA不符。

这仍是027单任务的前置恢复问题，不增加实验。接手时先寻找原登记完整数据副本，或定位最后一步/序列化的差异；以原登记stream和final chunk SHA实际通过为结束条件。未解决就提交明确阻塞并停止，不能开展两组训练。详细证据见 [整理记录](REPOSITORY_CLEANUP_20260921.md)。

## 2026-09-21进一步恢复复现结论

已对旧成功工作流做原样重跑：run `35596448071` attempt 2、job `106392757871` 使用同一head `30d44d9f...`、同一旧workflow和同一保全artifact，但恢复得到stream `fdea84306ac752611e4d0b1b4cd2300d0e0dcbc62ace07ac94096d904b310761`、final chunk `21cd73359a4837b7c2d58456e4c85da63e0023c9088bab5dc87e87dccf529d48`，在哈希断言处停止，未进入模型审计或训练。attempt 1曾在完全相同head/workflow下得到登记 `46b1.../fc3e...`。两次日志中的runner image、Python、torch、numpy、DGL、dill、psutil版本一致。

保全artifact `10526683176` 的manifest确认，其 `state_5520` 直接保存自原登记run `35254809015` 的 `protocol025-r1-overlay-5520`（artifact `10516563749`），不是后续重建状态。另将run `35114741391` 尚未过期的revision1进度artifact `10469412410` 中孤立的 `chunk_005400_005521.npz` 与维护run `35604239834` 的恢复尾chunk比较：两者54,820字节逐字节相同，21个NPZ数组全部完全相等，最大数值差为0；二者SHA均为 `21cd7335...`。

因此当前可以排除main特征修复、维护workflow改动、错误checkpoint替换以及已记录软件镜像/依赖版本变化。现有证据最符合“最后一个模拟/调度interval存在底层运行时或硬件敏感的数值执行差异”；但由于原登记完整stream/final chunk字节未保留，无法证明与 `fc3e.../46b1...` 的数组级差异。按本协议规则，这仍是硬阻塞：不得替换登记SHA，不得启动C/D。详细机器可读证据见 `artifacts/ftmoe_online/protocol_027/maintenance_20260921/recovery_reproduction_blocker.json`。

## 唯一运行入口

GitHub Actions选择 **Protocol-027 single D/C pilot**，分支选择 `main`：
- `run_models=false`：仅恢复/审计数据及验证代码，不训练模型。
- `run_models=true`：审计通过后，只运行登记的C_fixed5和D_dynamic。

推送相关修复默认只触发维护审计；不会自动运行科学实验。环境、恢复、审计、两组进程与结果上传都由 `.github/workflows/protocol027-pilot.yml` 管理。不要执行025的五方法runner或历史工作流。

本地入口仍为 `audit_ftmoe_protocol027.py` 与 `run_ftmoe_protocol027_pilot.py`，参数见各自 `--help`。使用Python3.8、torch2.4.1 CPU及 `requirements-online.txt`；不要通过上游 `main.py` 启动本协议。

## 保留与限制

- 物理stream SHA：`46b1dbdd885683bd45c146ffc3cbe68dd151bf60a10c1663eda45b68f12d7c42`。
- 025原两类guard审计仍为失败；027独立检查正常guard。历史失败不是新协议通用STOP。
- 027资格仍须真实通过物理标签、时间因果、特征一致性和九窗口覆盖检查。
- 原始检查点归档run35292728416仍可下载；不代表任意恢复都能复现登记SHA。工作流另保存完整恢复stream，避免证据只保留模型输出。
- 所有旧结果、checkpoint、模拟器数据与被import的023—025模块保留。旧文件名不代表旧计划仍生效。
- 确认701—703、测试201—205封存。D额外后台计算和记忆公开；不预先声称省算力或必胜。

历史资产说明：[原交接快照](GITHUB_EXPERIMENT_HANDOFF_HISTORY_20260921.md)。最新执行状态见 [结果页](PROTOCOL027_RESULTS.md) 和根目录LATEST文件。
