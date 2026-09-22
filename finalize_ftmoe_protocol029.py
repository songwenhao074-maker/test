"""Finalize Protocol-029 passive memory-utility diagnostic for repository handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else default


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_status(root, run_id):
    root = Path(root)
    status = read_json(root / "status.json")
    if status is not None:
        return status
    execution = read_json(root / "workflow_execution.json", {})
    status = {
        "protocol": "029",
        "run_id": str(run_id),
        "completed": False,
        "diagnostic_valid": False,
        "new_training_arms": 0,
        "C_rerun": False,
        "automatic_followups_started": [],
        "blocker": "execution_stopped_before_protocol029_status",
        "workflow_execution": execution,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf8")
    return status


def fmt(value, digits=6):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_results(summary, status, artifact_url, artifact_digest):
    lines = [
        "# Protocol-029 Results", "",
        f"GitHub Actions run: {status.get('run_id')}.",
        "Task: passive dormant-memory utility audit on one unchanged Protocol-028 D_memory_protected replay.",
        f"Diagnostic valid: {status.get('diagnostic_valid', False)}.",
        "C rerun: False. New training arms: 0.", "",
    ]
    if not summary:
        lines += ["## Blocker", "", str(status.get("blocker") or "Diagnostic summary missing."), ""]
        return lines

    consistency = summary["consistency"]
    opp = summary["opportunities"]
    passing = int(summary.get("all_gates_passing_candidate_count", 0))
    lines += [
        "## Source-trajectory consistency", "",
        f"- Probability max absolute error vs Protocol-028: {fmt(consistency.get('probability_max_abs_error'), 9)} (required <=1e-6).",
        f"- Full-stream AP absolute difference: {fmt(consistency.get('full_ap_absolute_difference'), 9)} (required <=1e-6).",
        f"- Discrete lifecycle events exact: {consistency.get('discrete_lifecycle_events_exact')}.",
        f"- Passive state checks/failures: {consistency.get('passive_state_checks')}/{consistency.get('passive_state_check_failures')} failures.", "",
        "## Opportunity audit", "",
        f"- Reuse due opportunities: {opp.get('reuse_due_opportunities')} (Protocol-028 source: {opp.get('source_protocol028_reuse_due')}).",
        f"- No-memory opportunities: {opp.get('no_memory_opportunities')}.",
        f"- Stream-end censored opportunities: {opp.get('stream_end_censored_opportunities')}.",
        f"- Opportunities with at least one candidate satisfying every original reuse gate: {opp.get('opportunities_with_any_all_gates_candidate')}.",
        f"- Such opportunities blocked by busy state: {opp.get('passing_candidate_opportunities_blocked_by_busy')}.",
        f"- Such opportunities blocked by similarity/rank: {opp.get('passing_candidate_opportunities_blocked_by_similarity_or_rank')}.",
        f"- Original selected candidate also passed all gates: {opp.get('original_selected_candidate_passed')}.", "",
    ]
    if passing:
        finding = (
            f"Passive audit found {passing} candidate-opportunity records that satisfy all original gates. "
            "They are post_hoc_oracle_diagnostic evidence of missed utility opportunities only; they do not establish an online D>C result."
        )
    else:
        finding = (
            "No audited dormant-memory candidate satisfied all original reuse gates on this unchanged trajectory. "
            "This supports insufficient reusable utility in the observed memories/times more than a pure selector-miss explanation, without proving general impossibility."
        )
    lines += ["## Main finding", "", finding, "", "## Nine recurrence first100 windows", ""]
    lines += ["| Window | Live AP | 4-generalist AP | Full-window dormant oracle | Notes |", "|---|---:|---:|---|---|"]
    for row in summary.get("nine_recurrence_windows", []):
        live_ap = (row.get("live") or {}).get("ap")
        gen_ap = (row.get("generalists_only") or {}).get("ap")
        oracle = row.get("full_window_post_hoc_best_expert")
        oracle_text = "NA" if not oracle else f"expert {oracle['expert_id']} / AP {fmt(oracle.get('candidate_ap'))}"
        partial = sum(1 for x in row.get("experts", []) if x.get("coverage_intervals", 0) and not x.get("direct_C_comparison_allowed"))
        note = "post-hoc oracle; not online policy" if oracle else (f"{partial} expert(s) partial coverage; not compared with full C" if partial else "no dormant full-window coverage")
        lines.append(f"| {row['phase']} | {fmt(live_ap)} | {fmt(gen_ap)} | {oracle_text} | {note} |")
    cost = summary.get("cost", {})
    lines += [
        "", "## Diagnostic overhead", "",
        f"- Diagnostic replay wall seconds: {fmt(cost.get('diagnostic_replay_wall_seconds'))}.",
        f"- Diagnostic replay CPU seconds: {fmt(cost.get('diagnostic_replay_cpu_seconds'))}.",
        f"- Passive prediction overhead seconds: {fmt(cost.get('passive_prediction_overhead_seconds'))}.",
        f"- Passive guard overhead seconds: {fmt(cost.get('passive_guard_overhead_seconds'))}.",
        "- This is diagnostic overhead and is not a new method-cost comparison against historical C.", "",
        "## Interpretation boundary", "",
        "All best-candidate choices in this report are post_hoc_oracle_diagnostic. Busy counterfactuals and passive candidates are not deployable policy results and are not formal D>C evidence.", "",
        "## Evidence", "",
        f"- Artifact: {artifact_url or 'see workflow run artifact'}",
        f"- Artifact SHA-256 digest: {artifact_digest or 'NA'}",
    ]
    return lines


def update_entrypoints(status, summary):
    valid = bool(status.get("diagnostic_valid"))
    passing = int((summary or {}).get("all_gates_passing_candidate_count", 0))
    if valid:
        if passing:
            finding = f"029发现{passing}条通过全部原门槛的被动候选记录，但仅属post-hoc oracle诊断，不是线上D>C证据。"
        else:
            finding = "029未发现任何休眠候选在原门槛下通过全部验收，当前轨迹更支持可复用收益不足而非单纯匹配漏检。"
        state = "Protocol-029已完成"
    else:
        finding = f"029诊断未通过028同轨迹一致性门禁；blocker={status.get('blocker')}。"
        state = "Protocol-029形成documented blocker"

    Path("AGENTS.md").write_text(
        "# Current task: Protocol-029 completed\n\n"
        "Read NEXT_EXPERIMENT_LATEST.md and docs/GITHUB_EXPERIMENT_HANDOFF.md.\n"
        f"{state}. There is no registered follow-up scientific experiment.\n\n"
        "Preserve Protocol-027/028/029 registrations, frozen-data provenance, raw artifacts and negative results. "
        "Do not automatically launch methods, seeds, ablations, confirmation runs, scenario changes or parameter search. "
        "Post-hoc oracle diagnostics are not online D>C evidence. Wait for an explicit new directive.\n",
        encoding="utf8",
    )
    Path("NEXT_EXPERIMENT_LATEST.md").write_text(
        "# 当前下一步：Protocol-029已完成，停止自动实验\n\n"
        f"{finding}\n\n"
        "结果见[Protocol-029 Results](docs/PROTOCOL029_RESULTS.md)。当前没有已登记的下一项科学实验。"
        "不要自动追加A/B、算法修改、种子、消融、确认实验、场景重做或调参；等待新的明确指示。\n",
        encoding="utf8",
    )
    Path("PROJECT_CONTEXT_LATEST.md").write_text(
        "# 当前项目上下文：Protocol-029已完成\n\n"
        "027和028均已完成；029在保持028线上D_memory_protected轨迹不变的条件下完成一次休眠记忆旁路效用诊断。\n"
        f"{finding}\n\n"
        "029没有重跑C，没有新增训练组，oracle候选只用于诊断。当前达到停止点；结果与证据见docs/PROTOCOL029_RESULTS.md。\n",
        encoding="utf8",
    )
    Path("docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
        "# 当前交接：Protocol-029已完成\n\n"
        "以main为准。Protocol-029单任务已经执行并交付；当前没有已登记的后续实验。\n\n"
        "结果：docs/PROTOCOL029_RESULTS.md。历史027/028/029注册、冻结数据来源和原始artifact均保留。\n\n"
        "不要自动追加方法、种子、消融、调参、确认实验或场景修改。新的科学实验必须等待新的明确指示。\n",
        encoding="utf8",
    )
    Path("README.md").write_text(
        "# PreGAN+ / FT-MoE 在线实验\n\n"
        "当前目标：在合理业务更替与复现场景中检验动态残差专家D相对固定残差专家C的预测优势，并保留所有负结果与诊断边界。\n\n"
        "**当前状态：Protocol-029一次休眠记忆效用诊断已完成，暂无登记的下一实验。**\n\n"
        "## 最新结果\n\n"
        f"{finding} 详见[Protocol-029 Results](docs/PROTOCOL029_RESULTS.md)。\n\n"
        "029保持028线上算法、数据、训练、阈值和生命周期决策不变，只增加标签前只读旁路。它没有重跑C或新增训练组；"
        "事后最好专家是post_hoc_oracle_diagnostic，不能作为线上D>C结论。\n\n"
        "## 历史\n\n"
        "[Protocol-028结果](docs/PROTOCOL028_RESULTS.md) / [Protocol-028分析](docs/PROTOCOL028_ANALYSIS_20260922.md) / "
        "[历史索引](docs/HISTORICAL_EXPERIMENTS.md) / [当前交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)。\n\n"
        "保留冻结数据、注册、历史失败和大artifact。当前没有自动后续任务。\n",
        encoding="utf8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--git-dest", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repository", required=True)
    ap.add_argument("--artifact-id", default="")
    ap.add_argument("--artifact-url", default="")
    ap.add_argument("--artifact-digest", default="")
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.run_root)
    dest = Path(args.git_dest)
    dest.mkdir(parents=True, exist_ok=True)
    status = ensure_status(root, args.run_id)
    if args.status_only:
        return
    summary = read_json(root / "diagnostic_summary.json")
    lines = build_results(summary, status, args.artifact_url, args.artifact_digest)
    Path("docs/PROTOCOL029_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    update_entrypoints(status, summary)

    keep = [
        "diagnostic_summary.json", "opportunity_audit.json", "consistency.json",
        "status.json", "cost_profile.json", "window_memory_utility.json",
        "workflow_execution.json", "source_artifacts.json", "eligibility.json",
        "data_lock.json", "guard_manifest.json",
    ]
    for name in keep:
        src = root / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    files = {}
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(dest))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    index = {
        "protocol": "029",
        "github_run_id": int(args.run_id),
        "workflow_url": f"https://github.com/{args.repository}/actions/runs/{args.run_id}",
        "artifact_id": args.artifact_id or None,
        "artifact_url": args.artifact_url or None,
        "artifact_digest": args.artifact_digest or None,
        "artifact_name": f"protocol029-memory-utility-{args.run_id}",
        "retention_days": 90,
        "files_committed_to_git": files,
        "raw_evidence": "GitHub Actions artifact contains passive_predictions.npz and opportunity_audit_full.json",
    }
    (dest / "ARTIFACT_INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
