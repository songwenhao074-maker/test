# 下一实验：Protocol-037，只接入一次动态出生

用户要求一点一点尝试动态增删。当前执行步骤仅037，后续休眠/唤醒/永久删除只有路线图，没有自动执行授权。

阅读docs/PROTOCOL037_SINGLE_DYNAMIC_BIRTH_20261001.md与artifacts/ftmoe_online/protocol_037/plan.json；先运行python maintenance/validate_protocol037_plan.py。执行分支codex/protocol-037-single-dynamic-birth-20261001，基于036结果876172e1a16ebcaa321f3ca713bb06ee6ba31b87，main不作为科学代码重建起点。

共同底座B=C+D_lin，复用036 seed3601 issued tape。B_ref不训练；新训练F_extra（从头固定额外专家）和D_birth（仅成熟B误差触发一次出生），每个专家同为74参数。零新流、零扫描、最多一次出生、两条科学序列；不休眠、不删专家、不按阶段强制出生。触发规则未激活也必须交付，不准改阈值重跑。

先冻结代码与分析、验证生产状态机恢复/因果性，再按登记运行。报告相对B收益、相对固定F的动态增量、召回护栏和真实资源账目。无论成功失败，都停止分析后再决定下一协议。当前仅发布指示，未启动037。

仅有GitHub连接器时允许一次性push helper调用037正式workflow_dispatch，按文档锁SHA/plan、查重、只POST一次；helper不得训练。发布结果必须使用干净checkout或Git tree API，更新status后同步哈希清单。
