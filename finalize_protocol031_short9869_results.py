"""Write compact terminal handoff for the Protocol-031 short9869 amendment."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def read(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else default


def fmt(v, n=6):
    return "NA" if v is None else f"{float(v):.{n}f}"


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--evidence-root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repository", required=True)
    a = ap.parse_args()
    root = Path(a.run_root); ev = Path(a.evidence_root)
    comp = read(root / "comparison.json")
    c = read(root / "C_fixed5/summary.json")
    d = read(root / "D_nonblocking_reuse/summary.json")
    eligible = read(ev / "eligibility.json", {})
    lock = read(ev / "data_lock.json", {})
    gen = read(ev / "generation_status.json", {})
    ok = comp is not None and c is not None and d is not None
    status = {
        "protocol": "031", "plan_revision": 2, "run_id": str(a.run_id),
        "completed": bool(ok), "post_start_user_amendment": True,
        "short_horizon_rows": 9869, "short_scored_intervals": 9868,
        "full_model_replays_started": 2 if ok else int(c is not None) + int(d is not None),
        "full_model_replay_budget": 2,
        "completed_comparators": ["C_fixed5", "D_nonblocking_reuse"] if ok else [],
        "A_B_runs": 0, "extra_seed_runs": 0, "extra_scenarios": 0,
        "development_signal": None if comp is None else bool(comp["primary"]["development_signal"]),
        "blocker": None if ok else "short9869_model_pair_incomplete",
    }
    write_json(root / "status.json", status)

    lines = [
        "# Protocol-031 short9869 amendment results", "",
        "**Important:** this is a user-requested post-start shortening of the original revision002 horizon. It freezes the deterministic prefix at 9869 rows (9868 scored + 1 guard) and is **not** the original six-recurrence preregistered result.", "",
        f"- Terminal Actions run: `{a.run_id}`.",
        f"- Frozen prefix rows: {lock.get('short_horizon_rows', 9869)}.",
        f"- Source checkpoint reached: {gen.get('source_checkpoint_next_t', 'NA')}.",
        f"- Model-free eligibility passed: {eligible.get('protocol031_data_eligible', False)}.",
        f"- Frozen stream SHA-256: `{lock.get('stream_sha256', 'NA')}`.",
        "- Available recurrence windows before the shortened horizon: `U_rec1`, `V_rec1`.", "",
    ]
    if comp is not None:
        p = comp["primary"]
        lines += [
            "## Short-horizon C/D result", "",
            f"- Valid recurrence windows: {p.get('valid_windows')}/{p.get('required_valid_windows')}.",
            f"- Equal-weight mean AP delta (D-C): {fmt(p.get('equal_weight_mean_D_minus_C_ap'), 9)}.",
            f"- Positive recurrence windows: {p.get('positive_windows')}/2.",
            f"- Pooled recurrence normal FPR delta (D-C): {fmt(p.get('pooled_recurrence_normal_fpr_delta_D_minus_C'), 9)}.",
            f"- Mean available W-block AP delta (D-C): {fmt(p.get('equal_weight_mean_W_block_D_minus_C_ap'), 9)}.",
            f"- Short-horizon development reference satisfied: {bool(p.get('development_signal'))}.", "",
            "| Window | C AP | D AP | D-C AP |",
            "|---|---:|---:|---:|",
        ]
        for row in comp.get("recurrence128", []):
            lines.append(
                f"| {row['phase']} | {fmt(row['C_fixed5'].get('ap'), 9)} | {fmt(row['D_nonblocking_reuse'].get('ap'), 9)} | {fmt(row.get('D_minus_C_ap'), 9)} |"
            )
        lines += [
            "", "## Interpretation boundary", "",
            "This shortened run is development-only. It may be used to inspect whether the mechanism shows a signal on the early recurrence pair, but it must not be reported as satisfying or failing the original six-window Protocol-031 primary endpoint. No extra seed, reroll, A/B arm, threshold search, or tuning was added.", "",
        ]
    else:
        lines += ["## Blocker", "", status["blocker"], ""]
    Path("docs/PROTOCOL031_SHORT9869_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    finding = (
        "Protocol-031已按用户明确指示改为9869行短周期版本。该版本复用原seed700确定性数据流前缀，"
        "仅评估9869之前可用的U_rec1/V_rec1两个recurrence窗口；它不是原六窗口预注册主结果。"
    )
    Path("AGENTS.md").write_text(
        "# Current task: Protocol-031 short9869 amendment reached stop point\n\n"
        "Read NEXT_EXPERIMENT_LATEST.md, docs/GITHUB_EXPERIMENT_HANDOFF.md and docs/PROTOCOL031_SHORT9869_RESULTS.md.\n\n" + finding +
        "\n\nDo not automatically add seeds, scenarios, A/B arms, rerolls, threshold changes, ablations or tuning. Wait for an explicit new directive.\n",
        encoding="utf8",
    )
    Path("NEXT_EXPERIMENT_LATEST.md").write_text(
        "# 当前状态：Protocol-031 short9869 amendment已到停止点\n\n" + finding +
        "\n\n结果见 docs/PROTOCOL031_SHORT9869_RESULTS.md。等待新的明确指示。\n",
        encoding="utf8",
    )
    Path("docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
        "# 当前交接：Protocol-031 short9869 amendment\n\n" + finding +
        "\n\n保留 frozen short9869 data、C/D predictions、lifecycle/reuse logs、comparison 与 Actions artifacts。不要把该结果描述为原revision002六窗口主结果。\n",
        encoding="utf8",
    )

    dest = Path("artifacts/ftmoe_online/protocol_031/short9869_latest")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("status.json", "comparison.json", "cost_profile.json"):
        src = root / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for name in ("eligibility.json", "data_lock.json", "generation_status.json", "scenario_registration_frozen.json", "short_horizon_amendment.json"):
        src = ev / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for arm in ("C_fixed5", "D_nonblocking_reuse"):
        src = root / arm / "summary.json"
        if src.is_file():
            (dest / arm).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / arm / "summary.json")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
