"""Finalize and compact a Protocol-028 GitHub Actions run for repository handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from ftmoe_protocol027_data import REVISION_ID


COMPARATORS = ("C_fixed5", "D_memory_protected")
SOURCE_ARTIFACT = {
    "run_id": 35682811782,
    "artifact_id": 10675401651,
    "artifact_name": "protocol027-frozen-data-35682811782",
    "artifact_digest": "sha256:111a4c5fc5c508a9e169fdbffe823ee36dc66c037c1775a6aa1abdbaba31c0a1",
}
LATE_RECURRENCE = (
    "S4_rec1", "S2_rec2", "S6_rec1", "S1_rec2", "S5_rec1", "S3_rec2"
)


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else default


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_status(root, run_id):
    """Persist a useful status if the model runner never wrote one."""
    root = Path(root)
    existing = read_json(root / "status.json")
    if existing is not None:
        return existing

    source = read_json(root / "source_frozen_artifact.json", {})
    eligibility = read_json(root / "eligibility.json", {})
    execution = read_json(root / "workflow_execution.json", {})
    failed_gates = [
        key for key, value in eligibility.get("gates", {}).items() if not value
    ]
    source_ok = bool(
        source.get("verified") is True
        and all(source.get(key) == value for key, value in SOURCE_ARTIFACT.items())
    )
    eligible = bool(
        eligibility.get("protocol027_data_eligible") is True
        and not failed_gates
    )
    run_models = execution.get(
        "run_models", os.environ.get("RUN_MODELS") == "true"
    )
    blocker = None
    if not source_ok:
        blocker = "source_frozen_artifact_identity_failed"
    elif not eligible:
        blocker = "inherited_data_eligibility_failed: " + ", ".join(failed_gates)
    elif not run_models:
        blocker = "Protocol028 model execution was not requested"
    else:
        blocker = "execution_stopped_before_runner_status"

    status = {
        "protocol": "028",
        "data_revision": REVISION_ID,
        "github_run_id": str(run_id),
        "completed": False,
        "source_frozen_artifact": source,
        "same_frozen_data_verified": source_ok,
        "eligibility_verified": eligible,
        "completed_comparators": [],
        "failed_comparators": [],
        "failed_eligibility_gates": failed_gates,
        "development_signal": None,
        "confirmation_run": False,
        "test_run": False,
        "automatic_followups_started": [],
        "blocker": blocker,
        "state": "blocked",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf8"
    )
    return status


def _fmt(value, digits=6):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _late_table(lifecycle):
    late = (lifecycle or {}).get("late_recurrence_reuse") or {}
    lines = [
        "| 后期回归窗口 | 匹配进入验收 | 接受复用 | 拒绝复用 | 无记忆 | 无匹配 | 满容量保护跳过birth | 专家ID |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in LATE_RECURRENCE:
        row = late.get(name, {})
        lines.append(
            "| {name} | {matched} | {accepted} | {rejected} | {nomem} | {nomatch} | {skip} | {ids} |".format(
                name=name,
                matched=row.get("matched", 0),
                accepted=row.get("reuse_accepted", 0),
                rejected=row.get("reuse_rejected", 0),
                nomem=row.get("reuse_skipped_no_memory", 0),
                nomatch=row.get("reuse_skipped_no_match", 0),
                skip=row.get("capacity_preserve_birth_skips", 0),
                ids=", ".join(row.get("expert_ids", [])) or "-",
            )
        )
    return lines


def _reuse_lines(lifecycle):
    decisions = (lifecycle or {}).get("reuse_decisions") or []
    if not decisions:
        return ["没有进入完整 reuse 验收决策的候选。"]
    lines = []
    for i, row in enumerate(decisions, 1):
        reason = ",".join(row.get("reject_reasons") or []) or "accepted"
        lines.append(
            "- #{i} phase={phase}, expert={expert}, accepted={accepted}, "
            "loss_improvement={imp}, normal_ok={normal_ok}, "
            "validation_FPR_delta={fpr}, guard_loss_delta={gloss}, "
            "guard_FPR_delta={gfpr}, reason={reason}.".format(
                i=i,
                phase=row.get("phase"),
                expert=row.get("expert_id"),
                accepted=row.get("accepted"),
                imp=_fmt(row.get("relative_improvement")),
                normal_ok=row.get("normal_safety_ok"),
                fpr=_fmt(row.get("validation_fpr_delta_0p5")),
                gloss=_fmt(row.get("guard_relative_loss_increase")),
                gfpr=_fmt(row.get("guard_fpr_delta_0p5")),
                reason=reason,
            )
        )
    return lines


def _cost_table(cost):
    lines = [
        "| 指标 | C_fixed5 | D_memory_protected |",
        "|---|---:|---:|",
    ]
    c = (cost or {}).get("C_fixed5", {})
    d = (cost or {}).get("D_memory_protected", {})
    rows = (
        ("wall_seconds", "模型循环墙钟秒"),
        ("cpu_seconds", "模型循环CPU秒"),
        ("max_rss_bytes", "峰值RSS字节"),
        ("p95_inference_seconds", "预测p95秒"),
        ("candidate_measured_overhead_seconds", "候选测得额外秒"),
        ("reuse_measured_overhead_seconds", "复用测得额外秒"),
    )
    for key, label in rows:
        lines.append(f"| {label} | {_fmt(c.get(key))} | {_fmt(d.get(key))} |")
    return lines


def write_result_docs(root, status, artifact_url, artifact_digest):
    comparison = read_json(root / "comparison.json")
    lifecycle = (comparison or {}).get("lifecycle") or {}
    cost = (comparison or {}).get("cost_profile") or {}

    lines = [
        "# Protocol-028 Results",
        "",
        f"GitHub Actions run: {status.get('run_id') or status.get('github_run_id')}.",
        f"Data revision: {REVISION_ID}; source frozen artifact is Protocol-027 run {SOURCE_ARTIFACT['run_id']} / artifact {SOURCE_ARTIFACT['artifact_id']}.",
        f"Completed: {status.get('completed', False)}.",
        f"Completed comparators: {status.get('completed_comparators', [])}.",
        f"Failed comparators: {status.get('failed_comparators', [])}.",
        "",
    ]

    if comparison is not None and status.get("completed"):
        primary = comparison["primary"]
        full = comparison["full_stream"]
        preserved = lifecycle.get("would_have_purged_expert_ids_preserved", [])
        lines += [
            "## Registered D/C answer",
            "",
            f"- Nine-window valid count: {primary['valid_windows']}/9.",
            f"- Equal-weight recurrence-first100 AP delta (D-C): {primary['equal_weight_mean_D_minus_C_fixed5_ap']}.",
            f"- Positive recurrence windows: {primary['positive_windows']}/9.",
            f"- Pooled normal-FPR delta (D-C): {primary['pooled_normal_fpr_delta_D_minus_C']}.",
            f"- Development signal: {primary['development_signal']}.",
            f"- Full-stream AP delta (D-C): {full['D_minus_C_fixed5_ap']}.",
            "",
            "## Memory-protection mechanism",
            "",
            f"- Full-capacity birth skips preserving accepted memory: {lifecycle.get('capacity_preserve_skips', 0)}.",
            f"- Experts that the old oldest-dormant purge rule would have selected, but were preserved: {preserved}.",
            f"- Actual purges: {lifecycle.get('purges', 0)}.",
            f"- Actual reactivations: {lifecycle.get('reactivations', 0)}.",
            f"- Final resident memory IDs: {lifecycle.get('resident_memory_ids', [])}.",
            "",
            "## Six late recurrence windows",
            "",
        ]
        lines += _late_table(lifecycle)
        lines += ["", "## Reuse decisions", ""]
        lines += _reuse_lines(lifecycle)
        lines += ["", "## Measured cost", ""]
        lines += _cost_table(cost)
        timing = (cost.get("D_memory_protected") or {}).get("timing_scope", {})
        lines += [
            "",
            f"Timing included: {timing.get('included', 'see compact evidence')}.",
            f"Timing excluded: {timing.get('excluded', 'see compact evidence')}.",
            "",
            "## Main limitation",
            "",
            "This is one development trajectory (replay seed700/model1), not a statistical confirmation. "
            "Protecting a full specialist memory can also block learning a genuinely new service; "
            "D retains extra background-compute and resident-memory privileges, so this is not an equal-total-cost comparison.",
            "",
            "Increased reuse or absence of purge is not by itself evidence of improved AP or a causal advantage.",
        ]
    else:
        lines += [
            "## Blocker", "",
            str(status.get("blocker") or "See workflow logs and archived evidence."), "",
            "No additional comparator, seed, ablation, or parameter search was started.",
        ]

    lines += [
        "", "## Evidence", "",
        f"- Artifact: {artifact_url or 'unavailable'}",
        f"- Artifact SHA-256 digest: {artifact_digest or 'unavailable'}",
        f"- Frozen source artifact SHA-256 digest: {SOURCE_ARTIFACT['artifact_digest']}",
        "",
    ]
    Path("docs/PROTOCOL028_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf8"
    )

    if comparison is not None and status.get("completed"):
        p = comparison["primary"]
        next_lines = [
            "# 当前下一步：Protocol-028已完成，停止自动实验", "",
            "Protocol-028 单任务已经完成；结果见 [Protocol-028 Results](docs/PROTOCOL028_RESULTS.md)。", "",
            f"九回归first100等权AP差 D_memory_protected-C_fixed5 = {p['equal_weight_mean_D_minus_C_fixed5_ap']}；"
            f"正差窗口 {p['positive_windows']}/9；development_signal={str(p['development_signal']).lower()}。",
            f"reactivation={lifecycle.get('reactivations', 0)}，purge={lifecycle.get('purges', 0)}，"
            f"capacity-preserve birth skips={lifecycle.get('capacity_preserve_skips', 0)}。",
            "",
            "当前没有已登记的下一项科学实验。不要自动追加A/B、其他固定拓扑、消融、确认种子或参数搜索；等待新的明确指示。",
            "",
        ]
        context_lines = [
            "# 当前项目上下文：Protocol-028已完成", "",
            "Protocol-027 已完成，随后按唯一指示执行了 Protocol-028：C_fixed5 vs D_memory_protected，seed700/model1。",
            "Protocol-028 只改变满驻留容量时的 birth 准入：保护已验收 memory，跳过未验证 shadow 的 birth，不 purge dormant specialist。",
            f"结果与完整解释：docs/PROTOCOL028_RESULTS.md；run {status.get('run_id') or status.get('github_run_id')}。",
            "当前任务到达停止点，没有自动启动后续实验。", "",
        ]
        handoff_lines = [
            "# 当前交接：Protocol-028已完成", "",
            "以 main 为准。Protocol-028 单任务已经执行并交付，当前没有已登记的后续实验。", "",
            "结果：docs/PROTOCOL028_RESULTS.md。",
            f"GitHub Actions run：{status.get('run_id') or status.get('github_run_id')}。", "",
            "不要自动追加方法、种子、消融或调参。新的科学实验必须等待新的明确指示。", "",
            "Protocol-027 与 Protocol-028 的注册、冻结数据来源和历史结果均保留，不覆盖。", "",
        ]
        agents_lines = [
            "# Current task: Protocol-028 completed", "",
            "Read NEXT_EXPERIMENT_LATEST.md and docs/GITHUB_EXPERIMENT_HANDOFF.md.",
            "Protocol-027 and the single registered Protocol-028 memory-protected D/C pilot are completed.",
            "There is no currently registered next scientific experiment.", "",
            "Do not automatically launch additional methods, seeds, ablations, confirmation runs or parameter searches.",
            "Preserve all registrations, frozen-data provenance, results and historical evidence. Wait for an explicit new directive before starting another scientific trial.",
            "",
        ]
    else:
        next_lines = [
            "# 当前下一步：Protocol-028 documented blocker", "",
            "Protocol-028 已启动但未形成两组完整结果。详情见 [Protocol-028 Results](docs/PROTOCOL028_RESULTS.md)。",
            f"GitHub Actions run: {status.get('run_id') or status.get('github_run_id')}。",
            "只允许在同一科学配置下修复确定性工程问题并恢复；不得改变数据、阈值、容量、种子、方法或增加实验。",
            "",
        ]
        context_lines = [
            "# 当前项目上下文：Protocol-028 blocker", "",
            "Protocol-028 当前未完成。只允许同配置工程修复/恢复，不允许科学参数变化或额外实验。",
            "详情：docs/PROTOCOL028_RESULTS.md。", "",
        ]
        handoff_lines = [
            "# 当前交接：Protocol-028 blocker", "",
            "Protocol-028 当前未完成；以 docs/PROTOCOL028_RESULTS.md 和紧凑 evidence 为准。",
            "仅允许同一注册配置的确定性工程修复/恢复。不要增加方法、种子、消融、调参或重新模拟冻结数据。",
            "",
        ]
        agents_lines = [
            "# Current task: Protocol-028 blocked", "",
            "Read docs/PROTOCOL028_RESULTS.md and NEXT_EXPERIMENT_LATEST.md.",
            "Only deterministic engineering repair/resume of the same registered Protocol-028 configuration is allowed.",
            "Do not change scientific settings or start extra experiments.", "",
        ]

    Path("NEXT_EXPERIMENT_LATEST.md").write_text("\n".join(next_lines), encoding="utf8")
    Path("PROJECT_CONTEXT_LATEST.md").write_text("\n".join(context_lines), encoding="utf8")
    Path("docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text("\n".join(handoff_lines), encoding="utf8")
    Path("AGENTS.md").write_text("\n".join(agents_lines), encoding="utf8")


def copy_compact_evidence(root, dest):
    keep = [
        "source_frozen_artifact.json", "eligibility.json", "data_lock.json",
        "guard_manifest.json", "normal_guard_audit.json", "comparison.json",
        "cost_profile.json", "status.json", "workflow_execution.json",
    ]
    for name in keep:
        src = root / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for arm in COMPARATORS:
        for name in [
            "summary.json", "lifecycle_summary.json", "candidate_records.json",
            "opportunity_accounting.json", "reuse_decisions.json",
            "late_recurrence_reuse.json", "failure.json",
        ]:
            src = root / arm / name
            if src.is_file():
                out = dest / arm / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--git-dest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--artifact-id", default="")
    parser.add_argument("--artifact-url", default="")
    parser.add_argument("--artifact-digest", default="")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.run_root)
    dest = Path(args.git_dest)
    dest.mkdir(parents=True, exist_ok=True)
    status = ensure_status(root, args.run_id)
    if args.status_only:
        return

    copy_compact_evidence(root, dest)
    files = {}
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(dest))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size
            }
    index = {
        "protocol": "028",
        "github_run_id": int(args.run_id),
        "workflow_url": f"https://github.com/{args.repository}/actions/runs/{args.run_id}",
        "artifact_id": int(args.artifact_id) if args.artifact_id else None,
        "artifact_url": args.artifact_url or None,
        "artifact_digest": args.artifact_digest or None,
        "artifact_name": f"protocol028-single-pilot-{args.run_id}",
        "retention_days": 90,
        "source_frozen_artifact": SOURCE_ARTIFACT,
        "files_committed_to_git": files,
        "large_evidence_location": "GitHub Actions artifact; predictions and logs are intentionally not committed to git",
    }
    (dest / "ARTIFACT_INDEX.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf8"
    )
    write_result_docs(root, status, args.artifact_url, args.artifact_digest)


if __name__ == "__main__":
    main()
