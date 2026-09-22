# 当前执行修订（2026-09-22）

用户已授权采用独立登记的protocol027_data_revision_002。下文“必须复现旧SHA/不得替换”的停止规则仅记录2026-09-21当时状态，已由[明确数据修订](PROTOCOL027_DATA_REVISION_002_20260922.md)替代；原失败事实保留。当前仍无C/D性能结果。

---

# Protocol-027 当前结果与阻塞

**C_fixed5和D_dynamic均未完成训练；暂无027性能比较。**

- 首次run35596448071 attempt 1：恢复得到登记stream/final chunk哈希，但当时9维特征一致性检查失败；该slope4补齐错误现已修复。
- 维护run35604239834：仅审计模式，19项测试通过；恢复得到stream `fdea84306ac752611e4d0b1b4cd2300d0e0dcbc62ace07ac94096d904b310761`、final chunk `21cd73359a4837b7c2d58456e4c85da63e0023c9088bab5dc87e87dccf529d48`，与登记 `46b1.../fc3e...` 不符，因此在C/D之前停止。
- 原样重跑旧成功环境：run35596448071 attempt 2 / job106392757871使用同一旧commit、旧workflow和保全checkpoint，再次稳定得到 `fdea.../21cd...`，在哈希断言处失败，模型步骤被跳过。
- 保全artifact 10526683176的manifest证明 `state_5520` 直接来自原登记run35254809015的overlay-5520 artifact 10516563749；不是后续重建checkpoint。
- run35114741391的revision1进度artifact 10469412410中孤立final chunk与run35604239834恢复final chunk **54,820字节完全相同**；21个NPZ数组全部array_equal，最大数值差为0，SHA均为 `21cd7335...`。
- 原登记job105357470988和上述重跑记录的runner image与Python/torch/numpy/DGL/dill/psutil版本一致。现有证据因此更支持最后一个模拟/调度interval存在底层运行时/硬件敏感的数值执行差异，而不是main代码或checkpoint来源变化。
- 修正后的5520步9D特征检查仍通过原5e-5容差；当前唯一硬阻塞是无法从保全的精确5520状态重新得到登记的完整stream/final chunk字节，而且原登记完整文件没有可下载副本。

机器可读证据：`artifacts/ftmoe_online/protocol_027/maintenance_20260921/recovery_reproduction_blocker.json`。

按Protocol-027规则，不修改登记SHA，不以当前可复现的新哈希代替，不运行C/D，也不自动追加A/B、种子、消融或调参。
