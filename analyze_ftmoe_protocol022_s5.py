"""Protocol 022 P22-S5 — evaluate the capacity Gate (plan §9.4).

    (future PR-AUC(A -> best fixed C) >= +0.03 absolute
     OR future onset AP >= +0.05 absolute)
    AND anchor/familiar F1 drop <= 0.03 absolute
    else STOP-CAP

"future" means the windows the residual never trained on: unseen late test and
unseen recurrence.  The familiar segments bracket the unseen block, so
familiar_2 (after the unseen exposure) is the retention check and familiar_1 is
the anchor reference.

Usage:
    python analyze_ftmoe_protocol022_s5.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P22 = ROOT / "artifacts/ftmoe_online/protocol_022"
DIAG = P22 / "fixed_c/s5_capacity_diagnostic/fixed_capacity_diagnostic.json"
OUT = P22 / "fixed_c/s5_capacity_gate.json"

FUTURE = ("unseen_late_test", "unseen_recurrence")
FAMILIAR = ("familiar_2", "familiar_1")
GATE = {"pr_auc_gain_min": 0.03, "onset_ap_gain_min": 0.05,
        "familiar_f1_drop_max": 0.03, "anchor_f1_drop_max": 0.03}


def value(payload, split, kind, field):
    block = payload["post_training"][split]
    if kind == "detection":
        return block["detection"][field]
    if kind == "onset":
        return block["onset"][field]
    if kind == "diagnosis":
        return block["diagnosis"][field] if field == "macro_f1" \
            else block["diagnosis"]["per_class"][str(field)]["f1"]
    raise ValueError(kind)


def main():
    results = json.loads(DIAG.read_text(encoding="utf8"))
    a = results["A"]
    rows = {}
    for name, payload in results.items():
        rows[name] = {
            "trainable_parameters": payload["trainable_parameters"],
            "updates": payload["optimization_updates"],
            "late_pr_auc": value(payload, "unseen_late_test", "detection", "pr_auc"),
            "late_f1": value(payload, "unseen_late_test", "detection", "f1"),
            "late_onset_ap": value(payload, "unseen_late_test", "onset", "ap"),
            "recur_pr_auc": value(payload, "unseen_recurrence", "detection", "pr_auc"),
            "recur_onset_ap": value(payload, "unseen_recurrence", "onset", "ap"),
            "familiar_2_f1": value(payload, "familiar_2", "detection", "f1"),
            "familiar_1_f1": value(payload, "familiar_1", "detection", "f1"),
            "familiar_2_pr_auc": value(payload, "familiar_2", "detection", "pr_auc"),
            "mid_pr_auc": value(payload, "unseen_mid_validation", "detection", "pr_auc"),
            "delay_late_pr_auc": payload["post_training"]["unseen_late_test"]["detection"]["pr_auc"]
            - payload["pre_training"]["unseen_late_test"]["detection"]["pr_auc"],
        }
    base = rows["A"]
    summary = {"protocol": "022", "step": "P22-S5", "gate_rule": GATE,
               "stream": json.loads(
                   (P22 / "fixed_c/s5_capacity_diagnostic/run_provenance.json")
                   .read_text(encoding="utf8"))["stream"],
               "baseline_A": base, "variants": {}}
    for name, row in rows.items():
        if name == "A":
            continue
        pr_gain = row["late_pr_auc"] - base["late_pr_auc"]
        recur_gain = row["recur_pr_auc"] - base["recur_pr_auc"]
        onset_gain = row["late_onset_ap"] - base["late_onset_ap"]
        fam_drop = base["familiar_2_f1"] - row["familiar_2_f1"]
        anchor_drop = base["familiar_1_f1"] - row["familiar_1_f1"]
        capacity = (pr_gain >= GATE["pr_auc_gain_min"]
                    or onset_gain >= GATE["onset_ap_gain_min"])
        protection = (fam_drop <= GATE["familiar_f1_drop_max"]
                      and anchor_drop <= GATE["anchor_f1_drop_max"])
        summary["variants"][name] = {
            "metrics": row,
            "late_pr_auc_gain_vs_A": pr_gain,
            "recurrence_pr_auc_gain_vs_A": recur_gain,
            "late_onset_ap_gain_vs_A": onset_gain,
            "familiar_2_f1_drop_vs_A": fam_drop,
            "familiar_1_f1_drop_vs_A": anchor_drop,
            "capacity_check_passed": capacity,
            "protection_check_passed": protection,
            "variant_passed": capacity and protection,
        }
    best = max(summary["variants"], key=lambda k: summary["variants"][k]
               ["late_pr_auc_gain_vs_A"])
    summary["best_variant_by_late_pr_auc"] = best
    summary["gate_passed"] = any(v["variant_passed"]
                                 for v in summary["variants"].values())
    summary["stop_conditions_hit"] = [] if summary["gate_passed"] else ["STOP-CAP"]
    summary["conclusion"] = (
        "capacity gain exists on the future unseen windows; the fixed residual is "
        "trainable and the representation is not the bottleneck"
        if summary["gate_passed"] else
        "no fixed residual variant produced a capacity gain: fix the "
        "representation/feature path before any online grid or D (plan §9.4)")
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf8")

    print("%-11s %8s %8s %9s %9s %8s %8s" % ("variant", "params", "updates",
                                             "latePR", "recurPR", "lateOnAP",
                                             "fam2F1"))
    for name, row in rows.items():
        print("%-11s %8d %8d %9.4f %9.4f %8.4f %8.4f"
              % (name, row["trainable_parameters"], row["updates"],
                 row["late_pr_auc"], row["recur_pr_auc"], row["late_onset_ap"],
                 row["familiar_2_f1"]))
    print()
    for name, entry in summary["variants"].items():
        print("%-11s latePR %+.4f  recurPR %+.4f  onsetAP %+.4f  "
              "fam2 drop %+.4f  fam1 drop %+.4f  -> capacity %s protection %s"
              % (name, entry["late_pr_auc_gain_vs_A"],
                 entry["recurrence_pr_auc_gain_vs_A"],
                 entry["late_onset_ap_gain_vs_A"],
                 entry["familiar_2_f1_drop_vs_A"],
                 entry["familiar_1_f1_drop_vs_A"],
                 entry["capacity_check_passed"], entry["protection_check_passed"]))
    print()
    print("Gate passed:", summary["gate_passed"],
          "| best:", best, "| STOP:", summary["stop_conditions_hit"] or "none")
    print("written:", OUT)


if __name__ == "__main__":
    main()
