"""Finalize and compact a Protocol-027 GitHub Actions run for repository handoff."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

COMPARATORS=("C_fixed5","D_dynamic")

def ensure_status(root, run_id):
    """Persist a useful status even when eligibility prevents the runner starting."""
    root = Path(root)
    existing = read_json(root / "status.json")
    if existing is not None:
        return existing
    eligibility = read_json(root / "eligibility.json", {})
    execution = read_json(root / "workflow_execution.json", {})
    failed = [k for k, value in eligibility.get("gates", {}).items() if not value]
    eligible = eligibility.get("protocol027_data_eligible") is True
    audit_only = execution.get("run_models") is False
    status = {
        "protocol": "027", "github_run_id": str(run_id), "completed": False,
        "audit_only": audit_only, "eligibility_verified": eligible,
        "completed_comparators": [], "failed_comparators": [],
        "failed_eligibility_gates": failed, "development_signal": None,
        "confirmation_run": False, "confirmation_seeds_used": [], "test_seeds_used": [],
        "automatic_followups_started": [],
        "blocker": ("data_eligibility_failed: " + ", ".join(failed)) if failed else
                   (None if eligible and audit_only else "execution_stopped_before_runner_status"),
        "state": "ready_for_two_arm_pilot" if eligible and audit_only else "blocked",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf8")
    return status

def read_json(path, default=None):
    p=Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else default

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",required=True)
    ap.add_argument("--git-dest",required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--repository",required=True)
    ap.add_argument("--artifact-id",default="")
    ap.add_argument("--artifact-url",default="")
    ap.add_argument("--status-only", action="store_true")
    args=ap.parse_args()

    root=Path(args.run_root)
    dest=Path(args.git_dest)
    dest.mkdir(parents=True,exist_ok=True)

    status=ensure_status(root, args.run_id)
    if args.status_only:
        return
    cmp=read_json(root/"comparison.json")
    life=(cmp or {}).get("lifecycle") or {}

    lines=[
        "# Protocol-027 Results","",
        f"GitHub Actions run: {args.run_id}.",
        f"Completed: {status.get('completed',False)}.",
        f"Completed comparators: {status.get('completed_comparators',[])}.",
        f"Failed comparators: {status.get('failed_comparators',[])}.","",
    ]
    if cmp:
        p=cmp["primary"]; full=cmp["full_stream"]
        lines += [
            "## Registered D/C answer","",
            f"- Nine-window valid count: {p['valid_windows']}/9.",
            f"- Equal-weight recurrence-first100 AP delta (D-C): {p['equal_weight_mean_D_minus_C_fixed5_ap']}.",
            f"- Positive recurrence windows: {p['positive_windows']}/9.",
            f"- Pooled normal-FPR delta (D-C): {p['pooled_normal_fpr_delta_D_minus_C']}.",
            f"- Development signal: {p['development_signal']}.",
            f"- Full-stream AP delta (D-C): {full['D_minus_C_fixed5_ap']}.","",
            "## Dynamic lifecycle","",
            f"- Births: {life.get('births',0)}.",
            f"- Retirements: {life.get('retirements',0)}.",
            f"- Reactivations: {life.get('reactivations',0)}.",
            f"- Purges: {life.get('purges',0)}.",
            f"- Reuse observed: {life.get('reactivations',0)>0}; causal reuse benefit is not established by event counts alone.","",
            "## Main limitation","",
            "This is one registered development trajectory (replay seed700/model1), and D is allowed additional background training and resident memory; it is not a statistical confirmation or an equal-total-cost comparison.","",
            "No follow-up experiment was started automatically.",
        ]
    elif status.get("audit_only") and status.get("eligibility_verified"):
        lines += ["## Maintenance verification", "",
                  "The full data eligibility audit passed. Neither comparator was run.",
                  "The repository is ready for the registered two-arm pilot; no performance claim is made."]
    else:
        lines += [
            "## Blocker","",
            str(status.get("blocker") or "See workflow logs and archived evidence."),"",
            "No additional comparator, seed, ablation, or parameter search was started.",
        ]
    Path("docs/PROTOCOL027_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")

    keep=["eligibility.json","data_lock.json","guard_manifest.json","normal_guard_audit.json",
          "comparison.json","cost_profile.json","status.json","workflow_execution.json"]
    for name in keep:
        src=root/name
        if src.is_file(): shutil.copy2(src,dest/name)
    for arm in COMPARATORS:
        for name in ["summary.json","lifecycle_summary.json","candidate_records.json",
                     "opportunity_accounting.json","failure.json"]:
            src=root/arm/name
            if src.is_file():
                out=dest/arm/name; out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,out)

    files={}
    for pth in sorted(dest.rglob("*")):
        if pth.is_file():
            files[str(pth.relative_to(dest))]={"sha256":sha256(pth),"bytes":pth.stat().st_size}
    index={
        "protocol":"027","github_run_id":int(args.run_id),
        "workflow_url":f"https://github.com/{args.repository}/actions/runs/{args.run_id}",
        "artifact_id":args.artifact_id or None,
        "artifact_url":args.artifact_url or None,
        "artifact_name":f"protocol027-single-pilot-{args.run_id}",
        "retention_days":90,
        "files_committed_to_git":files,
        "large_evidence_location":"GitHub Actions artifact; predictions and logs are intentionally not committed to git",
    }
    (dest/"ARTIFACT_INDEX.json").write_text(json.dumps(index,indent=2)+"\n",encoding="utf8")

    if status.get("audit_only") and status.get("eligibility_verified"):
        next_lines = ["# 当前实验入口", "",
            "Protocol-027全流数据资格检查已通过；C/D模型尚未运行。", "",
            "唯一待执行任务：[Protocol-027两组试跑](docs/PROTOCOL027_SINGLE_TASK_DIRECTIVE_20260921.md)。",
            "在main手动启动Protocol-027工作流，run_models=true；只运行C_fixed5与D_dynamic，交付后停止。",
            "每份指示只规划一个任务；不自动追加A/B、其他C、消融或种子。", "",
            f"本次仅维护验证：run {args.run_id}；[记录](docs/PROTOCOL027_RESULTS.md)。", ""]
    elif status.get("completed") and cmp:
        p=cmp["primary"]; full=cmp["full_stream"]
        next_lines=[
            "# 下一步实验入口（2026-09-21）","",
            "Protocol-027 单任务已执行并停止。结果见 [Protocol-027 Results](docs/PROTOCOL027_RESULTS.md)。","",
            "本次仅运行 C_fixed5 与 D_dynamic，replay seed700/model1；未启动 A/B、其他固定拓扑、消融、确认种子、测试种子或参数搜索。","",
            f"九个业务回归前100步等权 AP 差 D-C = {p['equal_weight_mean_D_minus_C_fixed5_ap']}；正差窗口 {p['positive_windows']}/9；development_signal={str(p['development_signal']).lower()}。全程 AP 差 D-C = {full['D_minus_C_fixed5_ap']}。D 实际 reactivation={life.get('reactivations',0)}，purge={life.get('purges',0)}。","",
            f"完整大文件证据位于 GitHub Actions run {args.run_id} 的 artifact protocol027-single-pilot-{args.run_id}。任务已到登记结束点，不自动安排下一实验。","",
        ]
    else:
        next_lines=[
            "# 下一步实验入口（2026-09-21）","",
            "Protocol-027 已启动执行但未形成两组完整结果，当前为 documented blocker。详情见 [Protocol-027 Results](docs/PROTOCOL027_RESULTS.md)。","",
            f"GitHub Actions run: {args.run_id}。紧凑证据：artifacts/ftmoe_online/protocol_027/runs/run_{args.run_id}/。","",
            "未自动启动额外方法、种子、消融或参数搜索；需要先分析本次确定性工程/数据阻塞证据。","",
        ]
    Path("NEXT_EXPERIMENT_LATEST.md").write_text("\n".join(next_lines),encoding="utf8")

    context_lines=[
        "# 当前项目上下文（2026-09-21）","",
        f"当前最新任务为 Protocol-027 单次 D/C 开发试跑。GitHub Actions run {args.run_id}；completed={status.get('completed',False)}。","",
        f"结果与问题：docs/PROTOCOL027_RESULTS.md。紧凑证据：artifacts/ftmoe_online/protocol_027/runs/run_{args.run_id}/。大文件预测与日志：artifact protocol027-single-pilot-{args.run_id}。","",
        "本协议只允许 C_fixed5 与 D_dynamic、replay seed700/model1。没有自动追加实验；后续由用户基于本次结果决定。","",
    ]
    Path("PROJECT_CONTEXT_LATEST.md").write_text("\n".join(context_lines),encoding="utf8")

if __name__=="__main__":
    main()
