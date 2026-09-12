"""Protocol 023 round 2A — assemble gate_status_round2a.json (directive §18).

The eight entry conditions for D are read from the round-2A artifacts and
written as one boolean per condition plus the overall ``D_eligible``.  No
condition is inferred: a missing artifact yields ``null`` and ``D_eligible``
false, because "not measured" is not "passed".

Usage
-----
    python assemble_ftmoe_protocol023_round2a_status.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUND2A = ROOT / "artifacts/ftmoe_online/protocol_023/round2a"
P23 = ROOT / "artifacts/ftmoe_online/protocol_023"

CONDITIONS = (
    "h1_v2_pass",
    "h2_v2_pass",
    "c_learns_two_regimes",
    "h3_v2_mature_gradient_conflict",
    "h4_v2_actual_forgetting",
    "recurrence_relearning",
    "online_budget_frozen",
    "prequential_integrity",
)


def read(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROUND2A / "gate_status_round2a.json")
    args = parser.parse_args()

    data_gate = read(ROUND2A / "data_calibration/data_gate_v2.json")
    h2 = read(ROUND2A / "specialization_v2/cross_regime_probe_v2.json")
    forgetting = read(ROUND2A / "fixed_c_prequential/forgetting_v2.json")
    recurrence = read(ROUND2A / "fixed_c_prequential/recurrence_metrics.json")
    gradient = read(ROUND2A / "mature_gradient/summary.json")
    budget = read(ROUND2A / "amendments/online_budget.json")
    selected = read(ROUND2A / "data_calibration/selected_generator.json")
    summaries = {arm: read(ROUND2A / ("fixed_c_prequential/arm_%s/summary.json"
                                      % arm)) for arm in ("A", "C")}

    integrity = {}
    for arm, summary in summaries.items():
        integrity[arm] = (None if summary is None
                          else (summary.get("prequential_integrity") or {})
                          .get("passed"))

    conditions = {
        "h1_v2_pass": (None if data_gate is None
                       else bool(data_gate["gate"]["passed"])),
        "h2_v2_pass": (None if h2 is None else bool(h2["gate"]["passed"])),
        "c_learns_two_regimes": (
            None if forgetting is None
            else bool(forgetting["step_1_learning"]["learned"])),
        "h3_v2_mature_gradient_conflict": (
            None if gradient is None else bool(gradient["passed"])),
        "h4_v2_actual_forgetting": (
            None if forgetting is None
            else bool(forgetting["step_2_forgetting"]["forgets"])),
        "recurrence_relearning": (
            None if recurrence is None
            else bool(recurrence["passed"]
                      and recurrence.get("recommended_for_d", True))),
        "online_budget_frozen": bool(
            budget is not None
            and budget.get("status") == "frozen"
            and bool(budget.get("frozen_configuration"))),
        "prequential_integrity": (
            None if any(value is None for value in integrity.values())
            else all(bool(value) for value in integrity.values())),
    }
    d_eligible = bool(all(conditions[name] is True for name in CONDITIONS))

    verdicts = {
        "H0_generator_correctness": "PASS (round 1; the A arm is still "
                                    "byte-identical to cascade_v2)",
        "H1_original": "FAIL (preserved permanently; never rewritten)",
        "H1_v2": (None if data_gate is None
                  else ("PASS" if data_gate["gate"]["passed"] else "FAIL")),
        "H2_v2": (None if h2 is None
                  else ("PASS" if h2["gate"]["passed"] else "STOP-MR")),
        "H3_v2": (None if gradient is None else gradient["verdict"]),
        "H4_v2": (None if forgetting is None else forgetting["verdict"]),
        "recurrence": (None if recurrence is None else recurrence["verdict"]),
    }
    payload = {
        "protocol": "023", "round": "2A",
        "stage": "round-2A complete" if all(v is not None
                                            for v in conditions.values())
                 else "round-2A partial",
        "directive": "指令/FTMOE_PROTOCOL023_ROUND2A_DIRECTIVE_20260912.md",
        "d_implemented": False,
        "forbidden_in_this_round": ["D", "expert birth/retire/reactivate",
                                    "C-wide", "C-budget",
                                    "seeds 701/702/703"],
        "seeds": {"development": 700, "confirmation_untouched": [701, 702, 703]},
        "conditions": conditions,
        "condition_meanings": {
            "h1_v2_pass": "directive §3 H1-v2 gate (primary-window follow-up "
                          ">= 80/regime plus the seven other checks)",
            "h2_v2_pass": "directive §7 H2-v2 (gap >= 0.05 and within positives "
                          ">= 30 in >= 2/3 regimes)",
            "c_learns_two_regimes": "directive §14 step 1: C beats A by >= 0.03 "
                                    "PR-AUC in >= 2/3 first-exposure regimes",
            "h3_v2_mature_gradient_conflict": "directive §16/§17 on a mature "
                                              "learner state",
            "h4_v2_actual_forgetting": "directive §14 step 2: previous-regime "
                                       "probe drop >= 0.03",
            "recurrence_relearning": "directive §15: relearning_gap >= 0.03 AND "
                                     "the gap must exceed the frozen baseline's "
                                     "own gap on the same transition (the raw "
                                     "§15 rule alone is also reported; the "
                                     "baseline shows a larger gap on every "
                                     "transition, so the attributable reading "
                                     "is what the condition uses)",
            "online_budget_frozen": "directive §10: online_budget.json frozen "
                                    "before the run and unchanged",
            "prequential_integrity": "directive §11/§21: prediction-before-"
                                     "update audit passed on both arms",
        },
        "verdicts": verdicts,
        "d_eligible": d_eligible,
        "d_eligibility_statement": (
            "D_eligible = true" if d_eligible else "D_eligible = false"),
        "failed_conditions": sorted(name for name in CONDITIONS
                                    if conditions[name] is not True),
        "prequential_integrity_by_arm": integrity,
        "calibration": {
            "selected_generator": (None if selected is None
                                   else selected.get("regimes")),
            "frozen_regimes": ["compute_first"],
            "data_only": True,
        },
        "evidence": {
            "h1_v2": "round2a/data_calibration/data_gate_v2.json",
            "marginal_match_v2": "round2a/data_calibration/marginal_match_v2.json",
            "h2_v2": "round2a/specialization_v2/cross_regime_probe_v2.json",
            "s4_A": "round2a/fixed_c_prequential/arm_A/summary.json",
            "s4_C": "round2a/fixed_c_prequential/arm_C/summary.json",
            "probe_matrix": "round2a/fixed_c_prequential/probe_matrix.json",
            "h4_v2": "round2a/fixed_c_prequential/forgetting_v2.json",
            "recurrence": "round2a/fixed_c_prequential/recurrence_metrics.json",
            "h3_v2": "round2a/mature_gradient/summary.json",
            "budget": "round2a/amendments/online_budget.json",
            "h1_v2_definition": "round2a/amendments/h1_v2_definition.json",
        },
        "next_round_rule": ("the next round may implement D's additive "
                            "birth/retire/reactivate only if every condition "
                            "above is true"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf8")
    print(json.dumps({"written": str(args.out.relative_to(ROOT)),
                      "conditions": conditions, "d_eligible": d_eligible,
                      "failed": payload["failed_conditions"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
