"""Protocol 023 — consolidate the S3 verdicts into the gate status (S3).

Reads the three analyzer/probe outputs and writes
``artifacts/ftmoe_online/protocol_023/gate_status.json``:

    H0  generator registration          <- verify_ftmoe_protocol023_generator.py
    H1  data gate + marginal matching   <- data_audit/DATA_GATE.md's verdict file
    H2  specialization opportunity      <- specialization/cross_regime_probe.json
    H3  gradient interference           <- specialization/gradient_interference.json
    H4  sequential forgetting           <- the same probe's train/test cells

A gate is only written with the verdict the artifact carries.  Where an artifact
marks its own result inadmissible (H3 at an untrained residual bank), the gate is
written ``inadmissible`` and the registered outcome is preserved next to it —
never promoted to PASS or FAIL.

Usage:
    python summarize_ftmoe_protocol023_s3.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/ftmoe_online/protocol_023"
PROBE = OUT / "specialization/cross_regime_probe.json"
GRADIENT = OUT / "specialization/gradient_interference.json"
DATA_GATE = OUT / "data_audit/marginal_match.json"
REGIMES = ("regime_A.json", "regime_B.json", "regime_C.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8")) if Path(path).is_file() else None


def generator_verdict():
    """Re-run the generator verifier and read its final status line."""
    try:
        out = subprocess.run(
            [sys.executable, str(ROOT / "verify_ftmoe_protocol023_generator.py")],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        text = out.stdout.decode("utf-8", "replace").strip().splitlines()
        last = json.loads(text[-1]) if text else {}
        return {"status": "pass" if last.get("status") == "PASS" else "fail",
                "failed_checks": last.get("failed_checks"),
                "evidence": "verify_ftmoe_protocol023_generator.py",
                "exit_code": out.returncode}
    except Exception as exc:                     # never invent a PASS
        return {"status": "unknown", "error": "%s: %s" % (type(exc).__name__, exc),
                "evidence": "verify_ftmoe_protocol023_generator.py"}


def data_gate_verdict():
    payload = read_json(DATA_GATE)
    if payload is None:
        return {"status": "pending", "reason": "no marginal_match.json"}
    gate = payload.get("gate_stats") or {}
    per_regime = {}
    for name in REGIMES:
        item = read_json(OUT / "data_audit" / name)
        if item is None:
            continue
        verdict = item.get("verdict") or item.get("gate") or {}
        per_regime[item.get("regime", name)] = {
            "passed": verdict.get("passed"),
            "checks": verdict.get("checks"),
            "metrics": verdict.get("metrics"),
        }
    return {
        "status": ("pass" if gate.get("gate_passed") else "fail"),
        "failed_checks": gate.get("failed_checks") or gate.get("failed_on"),
        "thresholds_admissible": gate.get("thresholds_admissible"),
        "prevalence_spread_pp": gate.get("prevalence_spread_pp"),
        "per_regime": per_regime,
        "definition_question": (
            "the plan's 'valid onset/follow-up >= 50' is gated under the strict "
            "reading (whole registered response window observed). The "
            "P22-compatible reading is reported next to it and would pass; the "
            "choice is the reviewer's and the analyzer never switches it "
            "silently"),
        "evidence": "artifacts/ftmoe_online/protocol_023/data_audit/",
    }


def specialization_verdict():
    payload = read_json(PROBE)
    if payload is None:
        return {"status": "pending", "reason": "no cross_regime_probe.json"}
    gate = payload.get("specialization_gate") or {}
    cells = payload.get("cells") or []
    per_regime = []
    for entry in gate.get("per_regime", []):
        if entry.get("horizon") != (payload.get("horizons") or [1])[0]:
            continue
        per_regime.append(entry)
    # H4 (sequential forgetting): for every ordered regime pair, how much does a
    # learner trained on the first lose on the second's own regime.
    forgetting = []
    for train in sorted({c.get("train_regime") for c in cells
                         if c.get("train_regime")}):
        within = [c for c in cells
                  if c.get("train_regime") == train and c.get("within_regime")
                  and c.get("horizon") == (payload.get("horizons") or [1])[0]]
        base = (within[0].get("metrics") or {}).get("ap") if within else None
        for other in sorted({c.get("test_regime") for c in cells
                             if c.get("test_regime")} - {train}):
            cell = [c for c in cells
                    if c.get("train_regime") == train
                    and c.get("test_regime") == other
                    and c.get("horizon") == (payload.get("horizons") or [1])[0]]
            value = (cell[0].get("metrics") or {}).get("ap") if cell else None
            forgetting.append({
                "trained_on": train, "evaluated_on": other,
                "same_regime_ap": base, "cross_regime_ap": value,
                "drop": (None if base is None or value is None
                         else float(base - value)),
                "test_positives": cell[0].get("positives") if cell else None,
            })
    return {
        "status": ("pass" if gate.get("passed") else "fail"),
        "n_regimes_passing": gate.get("n_regimes_passing"),
        "verdict": gate.get("verdict"),
        "per_regime": per_regime,
        "sequential_forgetting": forgetting,
        "evidence": "artifacts/ftmoe_online/protocol_023/specialization/cross_regime_probe.json",
    }


def gradient_verdict():
    payload = read_json(GRADIENT)
    if payload is None:
        return {"status": "pending", "reason": "no gradient_interference.json"}
    learner = (payload.get("learner") or {})
    verdict = payload.get("verdict") or {}
    return {
        "status": ("inadmissible"
                   if verdict.get("verdict") == "INADMISSIBLE"
                   else ("pass" if verdict.get("conflict_present") else "fail")),
        "registered_outcome": verdict.get("registered_gate_at_this_state")
                              or verdict.get("verdict"),
        "conflict_present": verdict.get("conflict_present"),
        "mean_cosine": verdict.get("mean_cosine") or payload.get("mean_cosine"),
        "negative_fraction": (verdict.get("negative_fraction")
                              or payload.get("negative_fraction")),
        "degenerate_groups": verdict.get("degenerate_groups"),
        "admissibility": verdict.get("admissibility"),
        "learner_state": learner.get("initialization") or learner,
        "evidence": "artifacts/ftmoe_online/protocol_023/specialization/gradient_interference.json",
    }


def main():
    status_path = OUT / "gate_status.json"
    status = read_json(status_path) or {"protocol": "023", "gates": []}
    verdicts = {
        "H0": generator_verdict(),
        "H1": data_gate_verdict(),
        "H2": specialization_verdict(),
        "H3": gradient_verdict(),
    }
    verdicts["H4"] = {
        "status": "measured" if verdicts["H2"].get("sequential_forgetting")
                  else "pending",
        "rule": ("fixed C trained on one regime loses performance on the "
                 "others; recurrence requires re-adaptation"),
        "evidence": verdicts["H2"].get("evidence"),
        "reading": ("reported as the cross-regime cells of the §13 probe: this "
                    "is a probe-level measurement, not a fixed-C online run, so "
                    "it bounds the conflict without replacing the S4 prequential "
                    "comparison"),
    }
    for gate in status.get("gates", []):
        if gate["id"] in verdicts:
            gate["status"] = verdicts[gate["id"]]["status"]
            gate["evidence"] = verdicts[gate["id"]].get("evidence")
    status["stage"] = "S3-complete (round 1: S0-S3)"
    status["verdicts"] = verdicts
    status["streams"] = {
        "development": {"tag": "dev_seed700_steps2880", "status": "collected",
                        "steps": 2880},
        "single_compute_first": {
            "tag": "single_compute_first_seed700_steps1200", "status": "collected"},
        "single_memory_first": {
            "tag": "single_memory_first_seed700_steps1200", "status": "collected"},
        "single_io_first": {
            "tag": "single_io_first_seed700_steps1200", "status": "collected"},
    }
    status["d_implemented"] = False
    status["next_decision"] = (
        "reviewer: (1) the §9 follow-up definition, (2) whether H3 must be "
        "evaluated on a mature fixed-C learner, (3) whether the "
        "instrument-local marginal contrasts become gated")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf8")
    print(json.dumps({g: verdicts[g].get("status") for g in sorted(verdicts)},
                     ensure_ascii=False))
    print(json.dumps({"written": str(status_path.relative_to(ROOT)).replace("\\", "/")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
