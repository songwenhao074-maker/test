"""Protocol 022 P22-S0 bootstrap — create the protocol-022 artifact scaffold.

Records, BEFORE any P22 data or code is produced:

    * the checked-out branch and HEAD (must, or must not, equal the plan's
      registered parent SHA ``f0abb5580d92730eecdc1db4326382c81a5f1e31``;
      if it differs the plan requires recording the actual HEAD and auditing
      the difference instead of force-resetting);
    * the SHA256 source snapshot of every file this protocol reads or writes
      (frozen P019/P020/P021 generators and collectors included) - the
      evidence chain must not silently drift;
    * the frozen online start checkpoint hash;
    * an empty gate table and problem log so later stages can only append.

No experiment is run here and no simulator/model file is touched.

Usage:
    python prepare_ftmoe_protocol022.py
    python prepare_ftmoe_protocol022.py --check
"""
import argparse
import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ftmoe_online/protocol_022"

PARENT_SHA = "f0abb5580d92730eecdc1db4326382c81a5f1e31"
PARENT_BRANCH = "protocol-021"
BRANCH = "protocol-022"
PLAN = "指令/FTMOE_PROTOCOL022_EXECUTION_PLAN_20260910.md"

# Unified online start checkpoint, hash registered by the plan (unchanged from
# protocol 020 S6; this protocol must never retrain or overwrite it).
START_CHECKPOINT = {
    "path": "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt",
    "sha256": "10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b",
}
# Protected frozen assets (plan §3): hashes are recorded so any later stage can
# prove it did not overwrite them.
PROTECTED = [
    "artifacts/ftmoe_online/protocol_019/**",
    "artifacts/ftmoe_online/protocol_020/**",
    "artifacts/ftmoe_online/protocol_021/**",
]
PROTECTED_FILES = [
    "artifacts/ftmoe_end_to_end/data/protocol_004_physical/time_series.npy",
    "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt",
    "artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/best.pt",
]
SOURCE_GLOBS = [
    "*.py",
    "simulator/**/*.py",
    "scheduler/**/*.py",
    "recovery/**/*.py",
    "stats/**/*.py",
    "metrics/**/*.py",
    "utils/**/*.py",
]
SOURCE_FILES = [
    PLAN,
    "artifacts/ftmoe_online/protocol_020/vm_split.json",
    "artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json",
    "artifacts/ftmoe_online/protocol_020/drift/drift_config.json",
    "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
    "artifacts/ftmoe_online/protocol_021/offline_coverage_audit/exposure_ledger.json",
    "artifacts/ftmoe_online/protocol_021/offline_coverage_audit/exclusion_registry.json",
    "artifacts/ftmoe_online/protocol_021/unseen_registry/cascade_v1.json",
]

GATES = [
    ("H0", "audit_correctness", "S0",
     "task-level audit really works at task scale (T-AUDIT-01..06)"),
    ("H1", "u3_u4v2_unseen_validity", "S3",
     "mechanism-level U3 + task/event-level distributional U4-v2 novelty"),
    ("H2", "learnability_beyond_persistence", "S4",
     "onset learnability beats persistence/current-ratio/host-prior baselines"),
]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(*args):
    out = subprocess.run(["git"] + list(args), cwd=str(ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return out.returncode, out.stdout.decode("utf8", "replace").strip()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf8")
    temporary.replace(path)


def source_snapshot():
    files = set()
    for pattern in SOURCE_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path)
    for rel in SOURCE_FILES + [START_CHECKPOINT["path"]]:
        path = ROOT / rel
        if path.is_file():
            files.add(path)
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
            for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report state without writing anything")
    args = parser.parse_args()

    code, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    code, head = git("rev-parse", "HEAD")
    code, dirty = git("status", "--porcelain")

    checkpoint = ROOT / START_CHECKPOINT["path"]
    checkpoint_ok = checkpoint.is_file() and sha(checkpoint) == START_CHECKPOINT["sha256"]

    state = {
        "protocol": "022",
        "created_at": _dt.date.today().isoformat(),
        "branch_checked_out": branch,
        "head": head,
        "parent_branch": PARENT_BRANCH,
        "parent_sha_registered": PARENT_SHA,
        "head_matches_registered_parent": head == PARENT_SHA,
        "rebase_or_reset_performed": False,
        "worktree_dirty_paths": [line for line in dirty.splitlines() if line.strip()],
        "start_checkpoint": dict(START_CHECKPOINT, present=checkpoint.is_file(),
                                 sha256_matches=checkpoint_ok),
        "protected_asset_globs": PROTECTED,
    }
    if head != PARENT_SHA:
        state["head_mismatch_action"] = (
            "plan §3: no force reset; the actual HEAD is recorded here and the "
            "difference must be audited before formal data generation")

    print(json.dumps({k: state[k] for k in
                      ("branch_checked_out", "head", "head_matches_registered_parent",
                       "worktree_dirty_paths")}, ensure_ascii=False, indent=2))
    print(json.dumps({"start_checkpoint_sha256_matches": checkpoint_ok},
                     ensure_ascii=False))
    if not checkpoint_ok:
        raise SystemExit("refusing to bootstrap: start checkpoint hash mismatch")

    if args.check:
        return

    for sub in ("audit_v2", "pilot_streams", "learnability_v2", "fixed_c",
                "dynamic_d", "confirmation"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    write_json(OUT / "bootstrap_state.json", state)

    sources = source_snapshot()
    write_json(OUT / "source_sha256_initial.json", {
        "protocol": "022",
        "captured_at": _dt.date.today().isoformat(),
        "head": head,
        "n_sources": len(sources),
        "note": ("P22 sources are captured before/after writing them via "
                 "register_ftmoe_protocol022.py; frozen P019/P020/P021 generator "
                 "and collector hashes are recorded here so the audit can prove "
                 "the offline reference was built from unmodified assets"),
        "sources": sources,
    })

    protocol = {
        "protocol": "022",
        "title": "Mechanism-level unseen audit and persistence-controlled learnability",
        "plan": PLAN,
        "plan_sha256": sha(ROOT / PLAN),
        "branch": BRANCH,
        "parent_branch": PARENT_BRANCH,
        "parent_sha": PARENT_SHA,
        "bootstrap_head": head,
        "gates": [{"id": gid, "name": name, "stage": stage, "rule": rule,
                   "status": "pending"} for gid, name, stage, rule in GATES],
        "stop_table": {
            "STOP-0": "task-level audit correctness FAIL -> no formal data generation",
            "STOP-A2": "U3/U4-v2 FAIL -> no unseen model experiment",
            "STOP-B2": "onset learnability / permutation FAIL -> no C/D tuning",
            "STOP-CAP": "fixed residual upper bound shows no gain -> no online grid, no D",
            "STOP-C2": "strict online C not better than A -> no D",
            "STOP-D0": "additive continuity FAIL -> no D effectiveness run",
            "STOP-D1": "D has no independent benefit -> cannot claim dynamic experts work",
            "STOP-R": "recurrence reuse FAIL -> add-only claim only",
        },
        "round1_scope": {
            "allowed": ["S0 task-level audit", "S1 cascade_v2 registration",
                        "S1 data-only pilot", "S2 U4-v2 instrumentation",
                        "S3 U4-v2 gate", "S4 learnability-v2"],
            "forbidden": ["full C run", "implement or run D",
                          "modify the FT-MoE backbone",
                          "retrain offline v4 / P19 / P20 checkpoints"],
        },
        "registered_seeds": {
            "mechanism_seed_dev": 22022,
            "mechanism_seed_confirm": 22023,
            "replay_seed_pilot": 600,
            "note": ("frozen before the first formal generation; the confirm "
                     "seed is not used in round 1"),
        },
        "frozen_candidate_probabilities": [0.15, 0.25, 0.35],
        "registered_horizon_steps": 1200,
        "guard_steps": 1,
        "cohort": "online",
        "resource_discipline": {
            "python": "3.8 (D:\\Anaconda\\envs\\dynmoe\\python.exe)",
            "single_process": True,
            "ram_guard_gib": 3.0,
            "disk_guard_gib": 20.0,
            "torch_threads": 3,
            "exit_code_source": "subprocess returncode, never a PowerShell wrapper",
        },
        "honesty_rules": [
            "STOP outcomes are experimental results and must be reported with evidence.",
            "No gate threshold may be redefined after seeing candidate results.",
            "Offline percentiles are locked before any candidate is read.",
            "Model scores may never influence which probability passes U4-v2.",
            "History is never rewritten: P21 STOP-A stays FAIL.",
        ],
    }
    write_json(OUT / "protocol.json", protocol)

    write_json(OUT / "gate_status.json", {
        "protocol": "022",
        "updated_at": _dt.date.today().isoformat(),
        "gates": {gid: {"stage": stage, "status": "pending", "rule": rule,
                        "evidence": None}
                  for gid, _, stage, rule in GATES},
        "stop_conditions_hit": [],
        "note": "round-1 scope is S0-S4; C/D stages are out of scope for this commit",
    })

    log = OUT / "problem_log.jsonl"
    if not log.is_file():
        with log.open("w", encoding="utf8") as stream:
            stream.write(json.dumps({
                "id": "P22-00",
                "date": _dt.date.today().isoformat(),
                "stage": "S0",
                "severity": "info",
                "status": "open",
                "summary": ("protocol 022 bootstrapped on branch %s from registered "
                            "parent %s" % (branch, PARENT_SHA)),
                "detail": ("round-1 scope S0-S4; cascade_v2 registered with a new "
                           "mechanism seed; task-level audit replaces the "
                           "host-aggregate U4 instrument (P21-09)"),
                "evidence": ["artifacts/ftmoe_online/protocol_022/protocol.json",
                             "artifacts/ftmoe_online/protocol_022/bootstrap_state.json"],
            }, ensure_ascii=False) + "\n")

    print(json.dumps({"written": str(OUT), "n_sources": len(sources),
                      "protocol_sha256": sha(OUT / "protocol.json")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
