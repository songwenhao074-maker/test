"""S4 runner smoke test on an already-collected stream (no registered output).

Exercises the strict-prequential loop, the settlement rule, the update path, the
checkpoint provenance and the probe scoring on a SHORT horizon, so the mechanics
are proven before the calibrated stream exists.  Everything it writes goes to a
scratch directory, never to the registered artifact tree.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import run_ftmoe_protocol023_s4 as s4

SCRATCH = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/_s4_smoke"
STREAM = (ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
                 "/dev_seed700_steps2880")


def run_once(arm, intervals=24):
    budget = json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))
    frozen = budget["frozen_configuration"]
    bundle = s4.build_replay(STREAM)
    anchor = s4.load_anchor_pool(bundle["replay"].time_scale)
    out = SCRATCH / ("arm_%s" % arm)
    session = s4.PrequentialS4(arm, 1, bundle, frozen, out, probe_paths=None,
                              anchor=anchor, learning_rate=1e-4)
    while session.cursor < intervals:
        session.step()
        if session.update_log and session.cursor % 4 == 0:
            pass
    integrity = s4.prequential_integrity(session)
    # This smoke intentionally stops before the full stream's guard row.
    # Keep the full-run audit visible, but test the observed prefix explicitly.
    # Do not weaken prequential_integrity() or claim finalization was tested.
    prefix_checks = {key: value for key, value in integrity.items()
                     if key not in ("passed", "final_interval_settled_on_the_stream_guard_row")}
    prefix_checks["all_mature_prefix_rows_settled"] = bool(np.array_equal(
        np.flatnonzero(session.predictions["settled_at"] >= 0),
        np.arange(max(0, intervals - 2))))
    prefix_checks["observed_predictions_finite"] = bool(np.isfinite(
        session.predictions["probability"][:intervals]).all())
    prefix_checks["unobserved_predictions_unwritten"] = bool(np.isnan(
        session.predictions["probability"][intervals:]).all())
    prefix_checks["updates_match_partial_budget"] = (
        session.updates == (intervals // 4 if arm == "C" else 0))
    prefix_checks["expected_learner_change"] = (
        (session.predictions["learner_hash"][0] != session.learner_hash) == (arm == "C"))
    prefix_checks["passed"] = all(prefix_checks.values())
    # settlement rule: label index must be strictly older than the read index
    settled = session.predictions["settled_at"]
    idx = np.arange(session.steps)
    lag = (settled[settled >= 0] - idx[settled >= 0])
    return {
        "arm": arm, "intervals": session.cursor, "updates": session.updates,
        "integrity": integrity,
        "smoke_integrity": prefix_checks,
        "scope": "Partial prefix only; final guard-row settlement is not tested here.",
        "settlement_lag_min": int(lag.min()) if lag.size else None,
        "settlement_lag_max": int(lag.max()) if lag.size else None,
        "first_learner_hash": session.predictions["learner_hash"][0][:16],
        "last_learner_hash": session.learner_hash[:16],
        "learner_changed": (session.predictions["learner_hash"][0]
                            != session.learner_hash),
        "frozen_unchanged": session.model.frozen_hash() == session.frozen_hash,
        "update_losses": (session.update_log[-1]["losses"]
                          if session.update_log else None),
        "update_mean_seconds": (float(np.mean([u["seconds"]
                                               for u in session.update_log]))
                                if session.update_log else None),
        "prediction_mean_seconds": float(
            session.predictions["prediction_seconds"][:session.cursor].mean()),
    }


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    results = [run_once("A"), run_once("C")]
    (SCRATCH / "smoke.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    ok = all(r["smoke_integrity"]["passed"] and r["frozen_unchanged"] for r in results)
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
