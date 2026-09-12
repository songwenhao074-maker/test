"""Protocol 023 round 2A — H4-v2 forgetting, recurrence relearning, H3-v2 matrix.

Reads the S4 outputs (both arms) plus the three mature-gradient reports and
writes the three round-2A verdict artifacts:

    fixed_c_prequential/probe_matrix.json     checkpoint x probe table (§13)
    fixed_c_prequential/forgetting_v2.json    H4-v2 gate (§14)
    fixed_c_prequential/recurrence_metrics.json  relearning gate (§15)
    mature_gradient/summary.json              H3-v2 checkpoint x pair matrix (§16-17)

The rules are applied literally and a STOP condition is written as a STOP
condition:

    H4-v2 learn-first   >= 2/3 first-exposure regimes with
                        late-phase PR-AUC(C) >= A + 0.03
    H4-v2 forgetting    >= 1 transition with previous-regime probe PR-AUC drop
                        >= 0.03 (2 is better)
    conflict            both of the above
    STOP-NO-LEARN       forgetting without learning
    §15 relearning_gap  previous exposure end score - recurrence first-100 score
                        >= 0.03 in >= 1 recurrence (2 is better)

Usage
-----
    python analyze_ftmoe_protocol023_round2a_gates.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ROUND2A = ROOT / "artifacts/ftmoe_online/protocol_023/round2a"

ARMS = ("A", "C")
PHASES = ("F0_baseline", "A1_compute", "B1_memory", "C1_io", "F1_baseline",
          "A2_recur", "C2_recur", "B2_recur")
FIRST_EXPOSURE = {"compute_first": "A1_compute", "memory_first": "B1_memory",
                  "io_first": "C1_io"}
LATE_PHASE = FIRST_EXPOSURE          # the phase's last scored interval is the
#                                       "late-phase" reading of that exposure
TRANSITIONS = (("A1_compute", "B1_memory", "compute_first", "memory_first"),
               ("B1_memory", "C1_io", "memory_first", "io_first"),
               ("C1_io", "A2_recur", "io_first", "compute_first"),
               ("A2_recur", "C2_recur", "compute_first", "io_first"),
               ("C2_recur", "B2_recur", "io_first", "memory_first"))
RECURRENCE = (("A2_recur", "A1_compute", "compute_first"),
              ("C2_recur", "C1_io", "io_first"),
              ("B2_recur", "B1_memory", "memory_first"))
CHECKPOINT_ORDER = ("start", "after_A1", "after_B1", "after_C1", "after_F1",
                    "after_A2", "after_C2", "after_B2")
THRESHOLDS = {
    "learn_gain_min": 0.03,
    "learn_regimes_min": 2,
    "forgetting_drop_min": 0.03,
    "forgetting_transitions_min": 1,
    "forgetting_transitions_ideal": 2,
    "relearning_gap_min": 0.03,
    "relearning_transitions_min": 1,
    "relearning_transitions_ideal": 2,
    "gradient_mean_cosine_max": -0.05,
    "gradient_negative_fraction_min": 0.30,
    "min_events_per_regime": 30,
    "recommended_events_per_regime": 64,
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def phase_metrics(arm):
    path = ROUND2A / ("fixed_c_prequential/arm_%s/phase_metrics.json" % arm)
    if not path.is_file():
        return None
    return load_json(path)


def probe_cells(arm):
    path = ROUND2A / ("fixed_c_prequential/arm_%s/probe_matrix.json" % arm)
    if not path.is_file():
        return None
    return load_json(path)["cells"]


def pr_auc(entry):
    if entry is None:
        return None
    detection = entry.get("detection") or {}
    return detection.get("pr_auc")


def probe_table():
    """checkpoint x probe matrix, the table directive §13 asks for."""
    arms = {arm: probe_cells(arm) for arm in ARMS}
    if any(value is None for value in arms.values()):
        return None
    checkpoints = []
    for cell in arms["C"]:
        if cell["checkpoint"] not in checkpoints:
            checkpoints.append(cell["checkpoint"])
    checkpoints = [name for name in CHECKPOINT_ORDER if name in checkpoints] + \
        [name for name in checkpoints if name not in CHECKPOINT_ORDER]
    rows = []
    for name in checkpoints:
        row = {"checkpoint": name}
        for arm in ARMS:
            for cell in arms[arm]:
                if cell["checkpoint"] != name:
                    continue
                row["%s_%s" % (arm, cell["probe"])] = {
                    "pr_auc": pr_auc(cell),
                    "macro_f1": (cell.get("diagnosis") or {}).get("macro_f1"),
                    "onset_ap": (cell.get("onset") or {}).get("ap"),
                    "positives": (cell.get("detection") or {}).get("positives"),
                    "model_version": cell.get("model_version"),
                    "learner_hash": cell.get("learner_hash"),
                }
        rows.append(row)
    return {"kind": "round2a_probe_matrix", "protocol": "023",
            "round": "2A",
            "metric_primary": "PR-AUC",
            "metric_secondary": "resource macro-F1",
            "probes": ["probe_%s" % r for r in
                       ("compute_first", "memory_first", "io_first")],
            "rows": rows,
            "note": ("every row is one checkpoint evaluated on all three fixed "
                     "held-out probe slices; the probes never train, never "
                     "enter replay and never tune a threshold")}


def forgetting_gate():
    arms = {arm: phase_metrics(arm) for arm in ARMS}
    if any(value is None for value in arms.values()):
        return None
    by_arm = {arm: {entry["phase"]: entry for entry in arms[arm]["phases"]}
              for arm in ARMS}

    # --- 1. does C learn the regime it is in? --------------------------
    # "late phase" is read in two ways and both are reported: the last 100
    # scored intervals of the exposure (the directive's wording) and the whole
    # exposure (the stricter reading, which the reference arm is known to win on
    # because it needs no learning at all).
    learning = []
    for regime, phase in FIRST_EXPOSURE.items():
        readings = {}
        for label, key in (("late_100", "late"), ("whole_phase", "whole")):
            late_c = pr_auc((by_arm["C"].get(phase) or {}).get(key))
            late_a = pr_auc((by_arm["A"].get(phase) or {}).get(key))
            gain = (None if late_c is None or late_a is None
                    else float(late_c) - float(late_a))
            readings[label] = {
                "pr_auc_c": late_c, "pr_auc_a": late_a, "gain_vs_a": gain,
                "meets_gain": bool(gain is not None
                                   and gain
                                   >= THRESHOLDS["learn_gain_min"]),
                "window": ((by_arm["C"].get(phase) or {}).get(key) or {})
                .get("intervals")}
        primary = readings["late_100"]
        learning.append({
            "regime": regime, "phase": phase,
            "primary_reading": "late_100",
            "pr_auc_c": primary["pr_auc_c"], "pr_auc_a": primary["pr_auc_a"],
            "gain_vs_a": primary["gain_vs_a"],
            "meets_gain": primary["meets_gain"],
            "readings": readings,
        })
    n_learn = sum(1 for row in learning if row["meets_gain"])
    learned = n_learn >= THRESHOLDS["learn_regimes_min"]

    # --- 2. does C lose the previous regime? ---------------------------
    probe_series = {}
    cells = {arm: probe_cells(arm) for arm in ARMS}
    if all(value is not None for value in cells.values()):
        for arm in ARMS:
            series = {}
            for cell in cells[arm]:
                series.setdefault(cell["probe"], {})[cell["checkpoint"]] = \
                    pr_auc(cell)
            probe_series[arm] = series

    forgetting = []
    if "C" in probe_series:
        checkpoints = [name for name in CHECKPOINT_ORDER
                       if name in probe_series["C"][
                           "compute_first"]]
        for before, after, previous, current in TRANSITIONS:
            cp_before = checkpoint_name(before)
            cp_after = checkpoint_name(after)
            if cp_before not in checkpoints or cp_after not in checkpoints:
                continue
            previous_series = probe_series["C"].get(previous, {})
            start = previous_series.get(cp_before)
            end = previous_series.get(cp_after)
            drop = (None if start is None or end is None
                    else float(start) - float(end))
            base_drop = None
            if "A" in probe_series:
                a_series = probe_series["A"].get(previous, {})
                a_start, a_end = a_series.get(cp_before), a_series.get(cp_after)
                if a_start is not None and a_end is not None:
                    base_drop = float(a_start) - float(a_end)
            forgetting.append({
                "from_phase": before, "to_phase": after,
                "previous_regime": previous, "current_regime": current,
                "checkpoint_from": cp_before, "checkpoint_to": cp_after,
                "previous_probe_pr_auc_before": start,
                "previous_probe_pr_auc_after": end,
                "drop": drop,
                "drop_arm_a": base_drop,
                "excess_drop_over_a": (None if drop is None or base_drop is None
                                       else drop - base_drop),
                "meets_drop": bool(drop is not None
                                   and drop
                                   >= THRESHOLDS["forgetting_drop_min"]),
            })
    n_forget = sum(1 for row in forgetting if row["meets_drop"])
    # The directive registers forgetting as the SECOND step: it is only
    # evaluated once C has been shown to learn the current regime.  When step 1
    # does not hold the forgetting verdict is null ("not evaluated"), not false,
    # and the transition measurements are still reported as diagnostics.
    if not learned:
        forgets = None
        evaluated = False
        stop = "STOP-NO-LEARN"
    else:
        forgets = n_forget >= THRESHOLDS["forgetting_transitions_min"]
        evaluated = True
        stop = None if forgets else "STOP-NO-CONFLICT"
    conflict = bool(learned and forgets)
    if conflict:
        verdict = "stability-plasticity conflict present"
    elif not learned:
        verdict = ("STOP-NO-LEARN: C does not beat A by the registered margin "
                   "in any first-exposure regime, so fixed C cannot be said to "
                   "learn the new regime; forgetting is therefore not evaluated")
    else:
        verdict = "no conflict"
    return {
        "kind": "round2a_h4_v2_forgetting", "protocol": "023", "round": "2A",
        "thresholds": dict(THRESHOLDS),
        "step_1_learning": {
            "rule": ("late-phase PR-AUC(C) >= A + %.2f in >= %d of %d "
                     "first-exposure regimes"
                     % (THRESHOLDS["learn_gain_min"],
                        THRESHOLDS["learn_regimes_min"], len(FIRST_EXPOSURE))),
            "per_regime": learning, "n_meeting": n_learn,
            "learned": bool(learned),
        },
        "step_2_forgetting": {
            "rule": ("previous-regime probe PR-AUC drop >= %.2f in >= %d "
                     "transition(s); %d is better"
                     % (THRESHOLDS["forgetting_drop_min"],
                        THRESHOLDS["forgetting_transitions_min"],
                        THRESHOLDS["forgetting_transitions_ideal"])),
            "transitions": forgetting, "n_meeting": n_forget,
            "forgets": forgets if evaluated else None,
            "diagnostic_n_meeting": n_forget,
            "evaluated_only_because_step_1_held": bool(evaluated),
            "note": ("directive §14 evaluates forgetting only after C has been "
                     "shown to learn the current regime; with step 1 failing "
                     "the verdict is null, and the transition drops are still "
                     "reported as diagnostics") if not evaluated else None,
        },
        "stability_plasticity_conflict": conflict,
        "verdict": verdict,
        "stop_condition": stop,
        "note": ("a conflict requires learning AND forgetting; forgetting "
                 "without learning is STOP-NO-LEARN, not a conflict"),
    }


def checkpoint_name(phase):
    return {"A1_compute": "after_A1", "B1_memory": "after_B1",
            "C1_io": "after_C1", "F1_baseline": "after_F1",
            "A2_recur": "after_A2", "C2_recur": "after_C2",
            "B2_recur": "after_B2"}.get(phase, phase)


def recurrence_gate():
    arms = {arm: phase_metrics(arm) for arm in ARMS}
    if any(value is None for value in arms.values()):
        return None
    out = {"kind": "round2a_recurrence_relearning", "protocol": "023",
           "round": "2A", "thresholds": dict(THRESHOLDS), "per_arm": {}}
    n_meeting = None
    for arm in ARMS:
        path = ROUND2A / ("fixed_c_prequential/arm_%s/recurrence_metrics.json"
                          % arm)
        if not path.is_file():
            continue
        block = load_json(path)["recurrence"]
        rows = []
        for entry in block:
            row = dict(entry)
            gap = row.get("relearning_gap")
            row["meets_gap"] = bool(gap is not None
                                    and gap
                                    >= THRESHOLDS["relearning_gap_min"])
            rows.append(row)
        out["per_arm"][arm] = {
            "rows": rows,
            "n_meeting": sum(1 for row in rows if row["meets_gap"]),
            "rule": ("previous exposure end score - recurrence first-100 score "
                     ">= %.2f in >= %d recurrence(s); %d is better"
                     % (THRESHOLDS["relearning_gap_min"],
                        THRESHOLDS["relearning_transitions_min"],
                        THRESHOLDS["relearning_transitions_ideal"])),
        }
        if arm == "C":
            n_meeting = out["per_arm"][arm]["n_meeting"]
    passed = bool(n_meeting is not None
                  and n_meeting >= THRESHOLDS["relearning_transitions_min"])
    # Confound control: the same gap must be smaller for the frozen baseline,
    # otherwise "relearning" is a property of the environment (a new regime's
    # first 100 intervals being harder to score) rather than of C.
    a_rows = {row["recurrence_phase"]: row
              for row in (out["per_arm"].get("A") or {}).get("rows", [])}
    c_rows = {row["recurrence_phase"]: row
              for row in (out["per_arm"].get("C") or {}).get("rows", [])}
    attribution = []
    for phase, row in sorted(c_rows.items()):
        baseline = a_rows.get(phase)
        c_gap = row.get("relearning_gap")
        a_gap = None if baseline is None else baseline.get("relearning_gap")
        attribution.append({
            "recurrence_phase": phase,
            "relearning_gap_c": c_gap,
            "relearning_gap_a": a_gap,
            "excess_over_a": (None if c_gap is None or a_gap is None
                              else c_gap - a_gap),
            "c_specific": bool(c_gap is not None and a_gap is not None
                               and c_gap > a_gap
                               and row.get("meets_gap")),
        })
    out["attribution"] = {
        "rows": attribution,
        "n_transitions_where_c_gap_exceeds_a": sum(
            1 for row in attribution if row["c_specific"]),
        "note": ("a relearning gap that the frozen baseline shows just as "
                 "strongly is a property of the timeline, not evidence that "
                 "fixed C had to relearn; both are reported"),
    }
    out["n_meeting_arm_c"] = n_meeting
    out["passed"] = passed
    out["verdict"] = ("recurrence requires relearning" if passed
                      else "no relearning gap measured")
    out["recommended_for_d"] = bool(
        passed and out["attribution"]["n_transitions_where_c_gap_exceeds_a"] >= 1)
    return out


def gradient_matrix():
    """checkpoint x regime-pair matrix from the three H3-v2 runs."""
    directory = ROUND2A / "mature_gradient"
    rows = []
    for name in ("after_A1", "after_B1", "after_C1"):
        path = directory / ("%s.json" % name)
        if not path.is_file():
            continue
        report = load_json(path)
        # the probe puts everything the gate reads under "verdict"
        verdict = report.get("verdict") or {}
        rows.append({
            "checkpoint": name,
            "file": path.name,
            "learner": report.get("learner"),
            "learner_state": ((report.get("learner") or {}).get("loaded_state")
                              or (report.get("learner") or {})
                              .get("initialization")),
            "admissibility": verdict.get("admissibility"),
            "verdict": verdict.get("verdict"),
            "verdict_text": verdict.get("verdict_text"),
            "conflict_present": verdict.get("conflict_present"),
            "conflict_present_any_group": verdict.get(
                "conflict_present_any_group"),
            "mean_cosine": verdict.get("mean_cosine"),
            "negative_fraction": verdict.get("negative_fraction"),
            "n_pairs": verdict.get("n_pairs"),
            "n_pairs_defined": verdict.get("n_pairs_defined"),
            "undefined_pairs": verdict.get("undefined_pairs"),
            "groups_with_conflict": verdict.get("groups_with_conflict"),
            "registered_gate_at_this_state": verdict.get(
                "registered_gate_at_this_state"),
            "n_events_per_regime": report.get("n_events_per_regime"),
            "degenerate_groups": report.get("degenerate_groups"),
            "parameter_integrity_unchanged": (report.get(
                "parameter_integrity") or {}).get("unchanged"),
        })
    if not rows:
        return None
    cps = [row for row in rows if row["conflict_present"]]
    admissible = [row for row in rows if not (
        (row["admissibility"] or {}).get("scientifically_admissible") is False)]
    return {
        "kind": "round2a_mature_gradient_summary", "protocol": "023",
        "round": "2A", "thresholds": dict(THRESHOLDS),
        "rows": rows,
        "n_checkpoints": len(rows),
        "n_checkpoints_with_conflict": len(cps),
        "n_admissible_checkpoints": len(admissible),
        "admissible_checkpoints": [row["checkpoint"] for row in admissible],
        "passed": bool(cps),
        "verdict": ("mature gradient conflict present" if cps else
                    "no mature gradient conflict at any tested checkpoint"),
        "stop_condition": None if cps else "STOP-GI",
        "note": ("the registered rule is unchanged from round 1: at least one "
                 "regime pair with mean cosine <= %.2f OR negative-pair "
                 "fraction >= %.2f; a checkpoint whose groups are still "
                 "degenerate is reported INADMISSIBLE and cannot carry the "
                 "verdict (directive §16)" %
                 (THRESHOLDS["gradient_mean_cosine_max"],
                  THRESHOLDS["gradient_negative_fraction_min"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROUND2A)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    # The per-arm S4 outputs live under fixed_c_prequential/arm_<A|C>/, because
    # the directive's flat names (predictions.npz, settlements.jsonl,
    # update_log.jsonl, phase_metrics.json) would otherwise collide between the
    # two arms.  This index states where each one is, so a reader looking for
    # the directive's layout finds it immediately instead of guessing.
    index = {
        "kind": "round2a_fixed_c_artifact_index", "protocol": "023",
        "round": "2A",
        "note": ("the directive's fixed_c_prequential/* filenames are the "
                 "per-arm ones below; both arms produce them, so they are "
                 "namespaced by arm rather than overwriting each other"),
        "arms": {},
        "combined": {},
    }
    for arm in ARMS:
        directory = ROUND2A / ("fixed_c_prequential/arm_%s" % arm)
        files = {}
        for name in ("predictions.npz", "settlements.jsonl", "update_log.jsonl",
                     "phase_metrics.json", "probe_matrix.json",
                     "recurrence_metrics.json", "checkpoints.json",
                     "summary.json"):
            path = directory / name
            files[name] = {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "present": path.is_file(),
                "bytes": (path.stat().st_size if path.is_file() else None)}
        index["arms"][arm] = {"directory":
                              str(directory.relative_to(ROOT)).replace("\\", "/"),
                              "files": files}
    for name in ("probe_matrix.json", "probe_matrix.md", "forgetting_v2.json",
                 "recurrence_metrics.json"):
        path = ROUND2A / ("fixed_c_prequential/%s" % name)
        index["combined"][name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "present": path.is_file()}
    index["checkpoints_directory"] = str(
        (ROUND2A / "checkpoints").relative_to(ROOT)).replace("\\", "/")
    write(args.out / "fixed_c_prequential/ARTIFACT_INDEX.json", index)

    table = probe_table()
    if table is not None:
        write(args.out / "fixed_c_prequential/probe_matrix.json", table)
        # the per-arm raw cells are preserved next to the combined table
        print("probe matrix: %d checkpoints x %d probes"
              % (len(table["rows"]), len(table["probes"])))
        # forgetting table as a markdown artifact too
        lines = ["# Protocol 023 round 2A — fixed-C probe matrix", "",
                 "| checkpoint | Probe A (PR-AUC) | Probe B | Probe C |",
                 "|---|---:|---:|---:|"]
        for row in table["rows"]:
            lines.append("| %s | %s | %s | %s |"
                         % (row["checkpoint"],
                            fmt(_cell(row, "C_compute_first")),
                            fmt(_cell(row, "C_memory_first")),
                            fmt(_cell(row, "C_io_first"))))
        (args.out / "fixed_c_prequential/probe_matrix.md").write_text(
            "\n".join(lines) + "\n", encoding="utf8")
    results["probe_matrix"] = "written" if table else "missing S4 outputs"

    forgetting = forgetting_gate()
    if forgetting is not None:
        write(args.out / "fixed_c_prequential/forgetting_v2.json", forgetting)
        results["h4_v2"] = forgetting["verdict"]
        results["conflict"] = forgetting["stability_plasticity_conflict"]
    else:
        results["h4_v2"] = "missing S4 outputs"

    recurrence = recurrence_gate()
    if recurrence is not None:
        write(args.out / "fixed_c_prequential/recurrence_metrics.json",
              recurrence)
        results["recurrence"] = recurrence["verdict"]
    else:
        results["recurrence"] = "missing S4 outputs"

    gradient = gradient_matrix()
    if gradient is not None:
        write(args.out / "mature_gradient/summary.json", gradient)
        results["h3_v2"] = gradient["verdict"]
    else:
        results["h3_v2"] = "missing mature-gradient reports"

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _cell(row, arm_probe):
    value = row.get(arm_probe)
    return None if value is None else value.get("pr_auc")


def fmt(value):
    return "—" if value is None else "%.4f" % float(value)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf8")
    return path


if __name__ == "__main__":
    sys.exit(main())
