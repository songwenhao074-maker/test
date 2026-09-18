# 下一步实验入口（2026-09-18）

当前指示：[Protocol-026：复用数据，完成五方法比较](docs/PROTOCOL026_NEXT_DIRECTIVE_20260918.md)。
先读 [最新审阅](docs/PROTOCOL025_REVIEW_20260918.md)；方法修订已登记于 [registration.json](artifacts/ftmoe_online/protocol_026/registration.json)。

v2c工程修复通过，但复现AP平均D−C=-0.00009536。Protocol-025数据生成完成，唯一强化审计失败项为F0保护集960正常、0故障，不是D性能失败；尚无Protocol-025模型比较。

保留Protocol-025 audit_pass=false。新Protocol-026使用同一物理数据，将F0用于正常行为回归保护，不要求正类；不能回写旧协议为通过。先恢复并核验完整stream，然后完成五方法开发运行。

恢复材料已归档：run35292728416，artifact10526683176，保留至2026-12-17。包含5520状态与不可变chunk，必要时只恢复最后1个guard interval，不重新生成全流。

仓库当前名songwenhao074-maker/test，ID1358898099未变。仅seed700/model1，确认701–703及测试201–205封存。本次只完成审阅、注册与证据归档，未运行新性能实验。
