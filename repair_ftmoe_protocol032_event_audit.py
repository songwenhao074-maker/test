"""Repair one Protocol-032 retrospective audit implementation mistake.

Run 36310071267 required a fresh active service event inside every service phase.
That was stricter than the registered requirement: phase/service annotations must
match exactly, U/V/W must each have observed events somewhere in the frozen
prefix, while recurrence windows have their own >=32 positive/negative coverage
gate.  This script repairs *only* that exact blocker and refuses all others.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED_STREAM_SHA = "1e8b6bde1fa3f906586030547777428b28fca23c43e06868077dc0e1e46e4d1a"
REQUIRED_GATE = "UVW_event_annotations"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--source-run-id", default="36310071267")
    args = ap.parse_args()

    bundle = Path(args.bundle_root)
    evidence = bundle / "evidence"
    audit_path = evidence / "input_audit.json"
    lock_path = evidence / "input_lock.json"
    stream_path = bundle / "data" / "stream.npz"
    if not (audit_path.is_file() and lock_path.is_file() and stream_path.is_file()):
        raise SystemExit("Protocol032 repair requires the preserved failed-audit frozen bundle")

    audit = json.loads(audit_path.read_text(encoding="utf8"))
    lock = json.loads(lock_path.read_text(encoding="utf8"))
    if audit.get("protocol") != "032" or int(audit.get("plan_revision", -1)) != 1:
        raise SystemExit("unexpected Protocol032 audit identity")
    if audit.get("kind") != "retrospective_input_eligibility_audit_after_031_results_seen":
        raise SystemExit("unexpected audit kind")
    if audit.get("model_results_seen_before_audit") is not True:
        raise SystemExit("retrospective provenance missing")
    if sha(stream_path) != EXPECTED_STREAM_SHA or lock.get("stream_sha256") != EXPECTED_STREAM_SHA:
        raise SystemExit("frozen stream identity mismatch")

    required = list(audit.get("required_gate_names") or [])
    gates = audit.get("gates") or {}
    blocked = [name for name in required if (gates.get(name) or {}).get("status") != "pass"]
    if blocked != [REQUIRED_GATE]:
        raise SystemExit("refusing repair: blocker set is not exactly UVW_event_annotations: %r" % blocked)

    old = gates[REQUIRED_GATE]
    ev = old.get("evidence") or {}
    phases = ev.get("phase_checks") or {}
    logical = ev.get("logical_services_seen") or {}
    if not phases:
        raise SystemExit("missing phase-level event audit evidence")
    if not all(bool(row.get("phase_id_exact")) and bool(row.get("service_id_exact"))
               for row in phases.values()):
        raise SystemExit("phase/service annotation mismatch is a real blocker; refusing repair")
    if not all(bool(logical.get(name)) for name in ("U", "V", "W")):
        raise SystemExit("U/V/W was not each observed with an active event; refusing repair")

    coverage = gates.get("nonF0_and_recurrence_class_coverage") or {}
    if coverage.get("status") != "pass":
        raise SystemExit("recurrence/non-F0 class coverage did not pass")

    original = evidence / "input_audit_original_overstrict.json"
    if not original.exists():
        shutil.copy2(audit_path, original)

    gates[REQUIRED_GATE] = {
        "status": "pass",
        "evidence": {
            **ev,
            "corrected_registered_criterion": (
                "phase_id and service_id annotations exact in every observed phase; "
                "U, V and W each have >=1 active event somewhere in frozen prefix; "
                "recurrence class coverage is enforced independently"
            ),
            "per_phase_new_event_required": False,
            "overstrict_original_gate_preserved_at": str(original),
        },
        "reason": None,
    }
    audit["gates"] = gates
    audit["audit_pass"] = all((gates.get(name) or {}).get("status") == "pass" for name in required)
    audit["audit_repair"] = {
        "kind": "engineering_criterion_correction_only",
        "source_run_id": str(args.source_run_id),
        "changed_gate": REQUIRED_GATE,
        "scientific_data_changed": False,
        "model_or_threshold_changed": False,
        "windows_changed": False,
        "reason": (
            "original audit incorrectly required a fresh active event inside every "
            "service phase; registered evidence requirement is exact annotations plus "
            "global U/V/W event presence, with recurrence class coverage separate"
        ),
    }
    if audit["audit_pass"] is not True:
        raise SystemExit("repaired audit still does not pass")
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8")

    lock["locked"] = True
    lock["input_audit_passed"] = True
    lock["audit_repair"] = audit["audit_repair"]
    lock_path.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n", encoding="utf8")

    print(json.dumps({
        "protocol": "032",
        "audit_pass": True,
        "stream_sha256": EXPECTED_STREAM_SHA,
        "repaired_gate": REQUIRED_GATE,
        "all_phase_service_annotations_exact": True,
        "logical_services_with_events": logical,
        "model_replays_consumed_before_repair": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
