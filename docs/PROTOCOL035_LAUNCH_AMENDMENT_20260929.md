# Protocol-035 启动方式修订：连接器无 dispatch 接口时的执行授权

日期：2026-09-29；修订ID：launch_20260929_v1。用户明确要求修改指示，解决执行模型无法首次启动的问题。本次提交只修改指示及提供未激活模板，不直接启动实验。

## 适用范围与优先级

本修订优先于035原指示中“不得新增push启动助手或连锁触发”的工程限制。允许执行模型在现有用户授权范围内选择下列启动方式，无需因连接器缺少特定接口再次询问是否允许首次启动。

科学工作流仍由 **workflow_dispatch** 运行；数据、模型、损失、判据、2条训练序列预算与禁止从头重训规则均保持不变。不可改动冻结 `plan.json` 或其校验摘要来迁就启动方式。工程例外单列在 [launch_amendment.json](../artifacts/ftmoe_online/protocol_035/launch_amendment.json)，与不可变科学登记一起归档。

当前已有实现代码（修订前执行分支head为 `1f069330703f8ad2e56ebb3357022a2f9ef222a0`，未在本修订中重新审计其科学正确性）。main已存在正式workflow，id=370221116，路径 `.github/workflows/protocol035-c-preserving-correction.yml`。不得仅因原入口仍写“未实现”就覆盖其他模型已经提交的实现。启动前读取最新分支和工作流实际内容。

**特别注意：正式workflow在main，执行代码在035分支。** dispatch外层 `ref` 应为 `main`；`inputs.execution_ref` 应为执行分支名。不要将其混淆，也不要把执行SHA作为 `execution_ref`，因为当前workflow还使用这个值推送结果到分支。

## 路径一：直接手动派发

优先使用真实可用的dispatch工具；若连接器未提供但有终端/HTTP能力，使用 REST 或 GitHub CLI。连接器不能dispatch，不等于GitHub没有接口。

REST请求为：

```
POST /repos/songwenhao074-maker/test/actions/workflows/protocol035-c-preserving-correction.yml/dispatches
Accept: application/vnd.github+json
Authorization: Bearer <运行时可用凭据，不写进文件>
X-GitHub-Api-Version: 2026-03-10

{"ref":"main","inputs":{"execution_ref":"codex/protocol-035-c-preserving-correction-20260929"}}
```

GitHub CLI等价命令：

```bash
gh workflow run protocol035-c-preserving-correction.yml \
  --repo songwenhao074-maker/test \
  --ref main \
  -f execution_ref=codex/protocol-035-c-preserving-correction-20260929
```

凭据通过现有登录态/运行环境提供，不打印、不提交、不把用户此前明文令牌复制到仓库。细粒度凭据需要仓库Actions写权限。首次调用前检查是否已经有035正式run；已有run则跟踪它，不能把“重跑已有run”当首次dispatch替代品。

成功响应优先记录返回的workflow_run_id/html_url；兼容旧API的204无响应体，再按workflow与发起时间查询run。HTTP超时或响应不明时先查run，不能盲目重复POST。官方说明：[dispatch REST API](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event)、[gh workflow run](https://cli.github.com/manual/gh_workflow_run)。

## 路径二：只有仓库写入连接器时，允许一次性启动助手

**这正是本修订新增的例外。** 若模型没有可用终端/HTTP/dispatch工具，但能创建或更新仓库文件，可以直接执行以下步骤，不再以“原文禁止push助手”为阻塞理由：

1. 检查正式工作流与已有run；确认没有已派发的035科学run。读取最新执行分支SHA和冻结plan hash。若已有run，保存其run_id并按预算规则继续检查，不创建新科学run。
2. 读取 [未激活模板](templates/protocol035-dispatch-once.yml.example)，将 `REPLACE_WITH_VERIFIED_EXECUTION_HEAD_SHA` 换成**刚核验的**035执行分支40位SHA。其余算法、目标workflow、分支、权限与次数限制不变；确保科学fixture和准备工作已完成或正式workflow会先完成这些检查。
3. 用仓库写入连接器将替换后的模板创建为 main 上 `.github/workflows/protocol035-dispatch-once.yml`。此文件只监听main及自己的路径，且job要求提交消息包含 `[p035-dispatch-once]`。
4. **这一次激活提交不得带 `[skip ci]`**，否则push助手不会运行。推荐消息 `ops: launch registered Protocol035 once [p035-dispatch-once]`。这是原skip-ci规则唯一新增例外；普通文档、结果与清理提交仍使用skip-ci。必须通过外部连接器/正常用户凭据写入，不能指望另一workflow的GITHUB_TOKEN push递归触发这个push助手。
5. 助手只做只读核验、查重和一次dispatch请求。使用`${{ github.token }}`及 `contents: read, actions: write`；不需要用户新增PAT secret。不得checkout并执行训练，不新生成数据，不调用旧协议，不添加workflow_run/schedule/repository_dispatch链。
6. 执行分支在核验到正式job完成checkout期间保持不变；核实正式job实际checkout的SHA与预期一致。正式workflow自身继续沿用原训练与预算检查。
7. 用已有Actions读取工具获取正式run_id，确认它的event是workflow_dispatch、workflow路径正确，并区分助手run和科学run。日志显示dispatch成功或助手success，并不等于科学实验完成。
8. 识别正式run后删除助手文件，提交用`[skip ci]`；保留本修订、模板和启动凭据/参数的**非秘密**审计记录。删除助手不会取消已启动的正式run。随后通过日志/artifact跟踪正式运行。

模板的固定并发组仅序列化启动助手；正式workflow自身也必须保留已有并发控制。并发控制不是科学预算计数器，所以模板遇到任何已有035正式run就拒绝再次dispatch。助手自身网络失败或超时后，先读取科学run状态，禁止自动重试POST；不得因为助手重跑功能可用就反复点击。

GitHub允许使用GITHUB_TOKEN创建workflow_dispatch事件并启动下游工作流，属于其递归触发限制的例外，见[官方触发说明](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#triggering-a-workflow-from-a-workflow)。本修订明确登记这一个启动桥接，不泛化允许其他连锁实验。

## 权限、失败及停止

若401/403或组织策略禁止Actions写入，给出具体接口、状态与所缺权限；本修订不能凭空赋予平台权限。不得仅凭工具列表推断缺权限，也不得从无关服务提取凭据。若文件写权限、Actions权限或分支保护确实阻止上述路径，提交已完成的启动材料和具体阻塞。

已有正式run失败时，先检查预算账本：如果科学训练已启动，不允许从头重跑；只能按035原规则精确恢复。若尚未消耗科学预算，也要保留已用真实前缀fixture次数，不能当成全新预算。一次性助手不能用来绕过这些规则。

启动记录保存 `launch_method`、修订ID、helper commit/run_id（如有）、科学workflow id/run_id、main工作流版本、执行分支及预期/实际SHA、发起时间、清理提交、已消费预算。当前这次“修改指令”提交没有激活助手或调用dispatch；执行模型随后按本修订自行启动。
