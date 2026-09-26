# Protocol-031非阻塞复用规范（计划revision003）
本文件仅规定D复用机制，配合[新场景](PROTOCOL031_RARE_RECURRENCE_SCENARIO.md)使用。常规在线更新为每16步；birth从成熟数600开始，每1600步一次。其余未变的shadow训练/验收与正常guard取当前固定配置。不要重新启用旧阻塞reuse路径。

## 唯一新策略的精确定义
1. 沿用成熟数≥600且每32步到期的reuse时钟。至少32个历史可见z时，在已验收休眠专家中选择余弦相似度最高的一位，数值相同按较小ID；原q10阈值仅记录，不再据此拒绝启动。每次仅一位候选，不用未来标签挑选，不固定专家ID或业务阶段。
2. 复用使用独立pending槽，不把birth主phase改成reuse_validation。monitoring、candidate_training、candidate_validation、cooldown均可启动/推进；已有pending槽时记skip_pending，crossfade期间记skip_transition，其他无记忆/历史不足分别记录。正常在线更新与shadow训练继续执行；监测不复制专家参数、不增加第九个驻留专家。
3. 槽创建后取之后16个新预测区间：先产生真实live与“四通用专家＋候选、替换旧专用专家”的预测，再等原延迟标签成熟评分。候选/旧活动专家ID及角色需保持稳定；若birth先验收导致活动专用专家变化或进入crossfade，立即取消旧槽，记cancelled_topology_changed，不沿用旧验证样本。
4. 满16个成熟配对后，必须沿用原全部验收：监督损失相对改善≥1%；正常样本平均异常概率增量≤0.01；验证FPR增量≤0.01；F0正常guard的candidate NLL≤live NLL×1.02＋1e-6、FPR增量≤0.01。任何不可用条件都不能算通过。guard只验收、不训练；不降低这些预测质量门槛。
5. 同一成熟事件内，先完成已有reuse槽的到期验收，再执行原birth训练/验收；若reuse通过，先取消尚未验收的shadow（如有），释放其私有优化器、训练/验证缓存，再按原8步crossfade激活旧专家。取消shadow单列cancelled_by_reuse，不伪装为质量拒绝，不删除任何已验收记忆；旧活动专家按原规则退休。禁止同一点再接受birth或开始第二次切换。若reuse失败，仅清空该槽，birth及其cooldown状态不被重置。
6. 当步已有reuse与birth处理完成后，再按reuse到期标志尝试开启新槽；若此时处于transition则跳过。birth按本次从成熟数600开始每1600步的到期规则，只把主phase和实际shadow状态用于忙碌判断，不因pending复用槽而停训。新birth与新reuse可同点开始；不积攒/追补错过的到期次数。每个成熟标签、计数与birth机会只能处理一次。
7. 流尾不足16个成熟配对记censored；因拓扑改变取消与质量拒绝分开。保存pending槽以支持正确检查点恢复。新生日志满足created=accepted+rejected+cancelled+pending；reuse日志满足started=accepted+rejected+cancelled+censored+pending（同一记录只占一类；最终把未完成槽转为censored，pending=0），机会计数守恒。


## 实现边界
关闭继承的旧reuse入口，防止双跑。原_decide_reuse失败会写phase=monitoring，不能原样调用而破坏birth状态。保持guard评估不改变训练参数/RNG及之后会读取的模型缓存；不增加候选训练或读取未来标签。

必要检查覆盖：并发birth/复用，低相似度准入但质量拒绝，失败不改birth，同点唯一接管，未验收shadow取消与缓存释放，拓扑改变取消pending，标签时序、容量和计数守恒、检查点恢复。检查只服务这一个试跑，不扩展为其他实验。
