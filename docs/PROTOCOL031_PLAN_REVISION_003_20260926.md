# Protocol-031计划revision003：缩短为9869步

用户于2026-09-26明确要求将GitHub设定改为9869步。本次只修订下一轮计划、场景登记及交接；不生成数据、不运行模型、不启动Actions。

## 当前唯一任务
按[执行指示](PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md)构建并冻结一个短版稀有回归流，运行C_fixed5与D_nonblocking_reuse各一次，报告全部结果后停止。当前plan_revision=3，scenario_id=protocol031_rare_recurrence_v2，data_revision=protocol031_data_revision_002。

| 项目 | 历史revision002 | 当前revision003 |
| --- | ---: | ---: |
| F0 | 300 | 300 |
| U/V首次 | 各1600 | 各1600 |
| W_long | 3200 | 1600 |
| 五个W_gap | 各1600 | 各800 |
| 六次U/V回归 | 各128 | 各128 |
| 计分区间 | 15468 | 9868 |
| 末尾保护行 | 1 | 1 |
| 总步数 | 15469 | 9869 |

减少5600步，约36.2%；不承诺同比例墙钟加速。800步间隔仍大于64步训练重放，但缩短干扰可能减少C遗忘，D优势仍须实测。六窗AP等权主指标、FPR/W退化约束、种子、学习率、在线更新16步及全部预测验收门槛保持。

## 专家时点与数据一致性
D新生仍按成熟数600起每1600步，早期600/2200/3800机会分别落在U/V初学与W_long内部，留有训练128＋验证32及标签成熟/切换余量。不得按阶段ID触发或强制验收；实际创建、驻留与有效复用另行报告，失败也应交付。

collector、audit、verifier、回放与窗口汇总必须读取本版登记的连续边界。保护行不纳入9868计分窗。旧版chunks/data_lock/模型检查点不能直接续跑本版，不可裁剪旧流冒充新生成；同版恢复需匹配注册哈希和完整RNG/模拟器状态。

[历史plan revision002](../artifacts/ftmoe_online/protocol_031/history/plan_revision_002.json)及[历史场景登记](../artifacts/ftmoe_online/protocol_031/history/scenario_registration_revision_002.json)原样归档。旧数据和结果不改写。保留既定专用分支与工作流名称；接手模型必须先同步main的revision003，再实现并启动。执行指示文件名保留20260922以保持链接有效，内容以revision003为准。
