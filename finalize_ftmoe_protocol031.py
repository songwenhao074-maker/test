"""Finalize Protocol-031 revision003 rare-recurrence development pilot."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else default


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value, digits=6):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def ensure_status(root, run_id):
    root = Path(root); status = read_json(root / "status.json")
    if status is not None:
        return status
    status = {
        "protocol": "031", "plan_revision": 3,
        "scenario_id": "protocol031_rare_recurrence_v2",
        "data_revision": "protocol031_data_revision_002",
        "run_id": str(run_id), "completed": False,
        "full_model_replays_started": 0, "full_model_replay_budget": 2,
        "A_B_runs": 0, "automatic_followups_started": [],
        "blocker": "execution_stopped_before_protocol031_status",
    }
    root.mkdir(parents=True, exist_ok=True); write_json(root / "status.json", status)
    return status


def build_results(status, comparison, generation, eligibility, lock, artifact_url, artifact_digest):
    lines = [
        "# Protocol-031 Results", "",
        "Scenario: `protocol031_rare_recurrence_v2` (plan revision003).",
        f"GitHub Actions terminal run: {status.get('run_id')}.",
        f"Completed registered C/D pair: {bool(status.get('completed'))}.",
        "Scope: one new seed700 simulator stream, model seed1, C_fixed5 versus D_nonblocking_reuse only.", "",
        "## Data freeze", "",
        f"- Generation complete: {bool((generation or {}).get('complete'))}.",
        f"- Model-free eligibility: {bool((eligibility or {}).get('protocol031_data_eligible'))}.",
        f"- Data locked before model runs: {bool((lock or {}).get('locked'))}.",
        f"- Frozen stream SHA-256: {(lock or {}).get('stream_sha256', 'NA')}.",
        f"- Scored intervals / guard rows: {(lock or {}).get('steps', 'NA')} / {(lock or {}).get('guard_rows', 'NA')}.", "",
    ]
    if not status.get("completed") or comparison is None:
        lines += [
            "## Blocker", "", str(status.get("blocker") or "Protocol031 stopped before the registered C/D pair completed."), "",
            "No reroll, extra seed, extra scenario, A/B run, threshold search, ablation or tuning was started.", "",
            "## Evidence", "", f"- Artifact: {artifact_url or 'see workflow run artifact'}",
            f"- Artifact digest: {artifact_digest or 'NA'}",
        ]
        return lines

    p = comparison["primary"]
    lines += [
        "## Registered primary result", "",
        f"- Valid recurrence windows: {p.get('valid_windows')}/6.",
        f"- Equal-weight mean AP delta (D-C): {fmt(p.get('equal_weight_mean_D_minus_C_ap'), 9)}.",
        f"- Positive recurrence windows: {p.get('positive_windows')}/6.",
        f"- Pooled recurrence normal FPR delta (D-C): {fmt(p.get('pooled_recurrence_normal_fpr_delta_D_minus_C'), 9)}.",
        f"- Equal-weight mean W-block AP delta (D-C): {fmt(p.get('equal_weight_mean_W_block_D_minus_C_ap'), 9)}.",
        f"- Registered development reference satisfied: {bool(p.get('development_signal'))}.", "",
        "| Recurrence | C AP | D AP | D-C AP | C FPR | D FPR | D-C FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.get("recurrence128", []):
        c, d = row["C_fixed5"], row["D_nonblocking_reuse"]
        lines.append(
            f"| {row['phase']} | {fmt(c.get('ap'),9)} | {fmt(d.get('ap'),9)} | {fmt(row.get('D_minus_C_ap'),9)} | "
            f"{fmt(c.get('fpr'),9)} | {fmt(d.get('fpr'),9)} | {fmt(row.get('D_minus_C_fpr'),9)} |"
        )
    life = comparison.get("lifecycle") or {}; outcomes = life.get("reuse_outcomes") or {}
    lines += [
        "", "## Dynamic-memory behavior", "",
        f"- Birth created/accepted/rejected/cancelled/pending: {life.get('birth_candidates_created')}/{life.get('birth_candidates_accepted')}/{life.get('birth_candidates_rejected')}/{life.get('birth_candidates_cancelled')}/{life.get('birth_candidates_pending')}.",
        f"- Reuse accepted/rejected/cancelled/censored: {outcomes.get('accepted',0)}/{outcomes.get('rejected',0)}/{outcomes.get('cancelled',0)}/{outcomes.get('censored',0)}.",
        f"- Reactivations / retirements / purges: {life.get('reactivations')} / {life.get('retirements')} / {life.get('purges')}.",
        f"- Full-capacity memory-preserving birth skips: {life.get('capacity_preserve_skips')}.",
        f"- First accepted-reuse influence cursor: {life.get('reuse_first_influence_cursor')}.", "",
    ]
    if int(outcomes.get("accepted", 0) or 0) == 0:
        lines += ["No dormant specialist was causally reactivated on this trajectory; this is retained as a negative memory-reuse outcome and is not repaired by rerolling or tuning.", ""]
    lines += [
        "## Interpretation boundary", "",
        "This is one preregistered development trajectory, not statistical confirmation. D is allowed extra resident memory and background shadow/reuse compute; those costs are reported rather than normalized away. The result applies only to C_fixed5 versus D_nonblocking_reuse in this registered rare-recurrence scenario and does not imply A/B results, global superiority, or a hard-delete benefit.", "",
        "No extra scenario, seed, method, threshold adjustment, A/B experiment, reroll or automatic scientific follow-up was started.", "",
        "## Evidence", "", f"- Artifact: {artifact_url or 'see workflow run artifact'}", f"- Artifact digest: {artifact_digest or 'NA'}",
    ]
    return lines


def update_entrypoints(status, comparison):
    if status.get("completed") and comparison is not None:
        p = comparison["primary"]
        finding = (
            "Protocol-031 revision003已完成登记的一次rare-recurrence C/D开发性对比。"
            f"六个recurrence128窗口的等权AP差D-C={fmt(p.get('equal_weight_mean_D_minus_C_ap'),6)}，"
            f"正差窗口={p.get('positive_windows')}/6，开发参考门槛是否全部满足={bool(p.get('development_signal'))}。"
        )
    else:
        finding = "Protocol-031 revision003已停止于显式blocker：" + str(status.get("blocker") or "unknown") + "。未追加reroll、seed、场景、A/B、阈值搜索或调参。"
    Path("AGENTS.md").write_text(
        "# Current task: Protocol-031 revision003 reached stop point\n\nRead NEXT_EXPERIMENT_LATEST.md, docs/GITHUB_EXPERIMENT_HANDOFF.md and docs/PROTOCOL031_RESULTS.md.\n\n" + finding +
        "\n\nDo not automatically launch more simulator streams, model replays, seeds, A/B arms, threshold changes, scenario redesign, ablations or tuning. Wait for an explicit new directive.\n",
        encoding="utf8")
    Path("NEXT_EXPERIMENT_LATEST.md").write_text(
        "# 当前状态：Protocol-031 revision003已到停止点\n\n" + finding +
        "\n\n结果与证据见[Protocol-031 Results](docs/PROTOCOL031_RESULTS.md)。当前没有登记的后续实验；等待新的明确指示。\n", encoding="utf8")
    Path("PROJECT_CONTEXT_LATEST.md").write_text(
        "# 当前项目上下文：Protocol-031 revision003\n\n" + finding +
        "\n\n本轮只允许一个新冻结数据流以及C_fixed5/D_nonblocking_reuse各一次完整回放。所有数据门禁、生命周期、成本与解释边界见docs/PROTOCOL031_RESULTS.md。\n", encoding="utf8")
    Path("docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
        "# 当前交接：Protocol-031 revision003已到停止点\n\n" + finding +
        "\n\n结果：docs/PROTOCOL031_RESULTS.md。保留scenario registration、method registration、data_lock、原始predictions/lifecycle日志与Actions artifact。不要自动追加完整回放、算法修改、额外seed/场景、A/B、消融或调参。\n", encoding="utf8")
    readme = Path("README.md").read_text(encoding="utf8") if Path("README.md").is_file() else "# FT-MoE experiments\n"
    if "\n## Protocol-031 latest\n" in readme:
        readme = readme.split("\n## Protocol-031 latest\n", 1)[0].rstrip() + "\n"
    Path("README.md").write_text(readme.rstrip() + "\n\n## Protocol-031 latest\n\n" + finding + " 详见[Protocol-031 Results](docs/PROTOCOL031_RESULTS.md)。\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True); ap.add_argument("--git-dest", required=True)
    ap.add_argument("--run-id", required=True); ap.add_argument("--repository", required=True)
    ap.add_argument("--artifact-id", default=""); ap.add_argument("--artifact-url", default=""); ap.add_argument("--artifact-digest", default="")
    ap.add_argument("--status-only", action="store_true"); a = ap.parse_args()
    root = Path(a.run_root); dest = Path(a.git_dest); dest.mkdir(parents=True, exist_ok=True)
    status = ensure_status(root, a.run_id)
    if a.status_only:
        return
    comparison = read_json(root / "comparison.json"); generation = read_json(root / "generation_status.json")
    eligibility = read_json(root / "eligibility.json"); lock = read_json(root / "data_lock.json")
    Path("docs/PROTOCOL031_RESULTS.md").write_text(
        "\n".join(build_results(status, comparison, generation, eligibility, lock, a.artifact_url, a.artifact_digest)) + "\n", encoding="utf8")
    update_entrypoints(status, comparison)
    keep = ["status.json","comparison.json","cost_profile.json","eligibility.json","data_lock.json","generation_status.json","scenario_registration_frozen.json","pre_generation_receipt.json","environment.json","workflow_execution.json","frozen_data_archive.json"]
    for name in keep:
        src = root / name
        if src.is_file(): shutil.copy2(src, dest / name)
    for arm in ("C_fixed5","D_nonblocking_reuse"):
        for name in ("summary.json","initialization.json","lifecycle_summary.json","reuse_records.json","opportunity_accounting.json","expert_snapshots.json"):
            src = root / arm / name
            if src.is_file():
                target = dest / arm / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, target)
    files = {}
    for path in sorted(dest.rglob("*")):
        if path.is_file(): files[str(path.relative_to(dest))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(dest / "ARTIFACT_INDEX.json", {
        "protocol":"031","plan_revision":2,"github_run_id":int(a.run_id),
        "workflow_url":f"https://github.com/{a.repository}/actions/runs/{a.run_id}",
        "artifact_id":a.artifact_id or None,"artifact_url":a.artifact_url or None,"artifact_digest":a.artifact_digest or None,
        "artifact_name":f"protocol031-rare-recurrence-{a.run_id}","retention_days":90,"files_committed_to_git":files,
        "raw_evidence":"Actions artifact contains frozen-data evidence plus both full predictions, lifecycle/reuse logs and runner logs."
    })


if __name__ == "__main__":
    main()
