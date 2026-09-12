"""Round 2A — freeze the selected S2.5 calibration into selected_generator.json.

Reads the candidate table (data-only calibration runs on seed 700), picks the
registered B/C parameter set, and writes the file the collector consumes with
``--calibration-parameters``.  The selection is recorded with the evidence that
produced it and the directives' targets, so the choice is auditable instead of
implicit.

A is never a candidate: it must stay byte-identical to Protocol 022's
``cascade_v2``.

Usage
-----
    python select_ftmoe_protocol023_s25_candidate.py \
        --io-first c_prob45_peak24dur --memory-first b_prob60
    python select_ftmoe_protocol023_s25_candidate.py --list
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import calibrate_ftmoe_protocol023_s25 as s25

OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/data_calibration"
SELECTED = OUT / "selected_generator.json"

TARGETS = {
    "event_count_relative_diff_max": 0.20,
    "duration_relative_diff_max": 0.25,
    "median_peak_ratio_relative_diff_max": 0.30,
    "prevalence_absolute_diff_max": 0.03,
    "h1_positives_recommended": 50,
    "h1_positives_minimum": 30,
    "prevalence_in_range": [0.03, 0.12],
    "migration_rejection_max": 0.40,
    "deployment_rejection_max": 0.25,
}


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_row(name, table):
    for entry in table["candidates"]:
        if entry.get("candidate") == name:
            return entry
    raise SystemExit("candidate %r is not in the candidate table (%s)"
                     % (name, sorted(c["candidate"] for c in table["candidates"])))


def require_full_length(row):
    if int(row.get("steps", 0)) != 1200:
        raise SystemExit(
            "candidate %s was measured on %s intervals; a calibration "
            "selection must cite the registered 1200-interval single-stream "
            "shape (the same shape the H1-v2 gate reads)"
            % (row["candidate"], row.get("steps")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io-first", default=None)
    parser.add_argument("--memory-first", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", type=Path, default=SELECTED)
    parser.add_argument("--allow-short", action="store_true",
                        help="accept a candidate measured on fewer than 1200 "
                             "intervals (records the deficit in the file)")
    args = parser.parse_args()

    table_path = OUT / "candidate_table.json"
    if not table_path.is_file():
        raise SystemExit("no candidate table at %s" % table_path)
    table = json.loads(table_path.read_text(encoding="utf8"))

    if args.list:
        for entry in sorted(table["candidates"],
                            key=lambda e: (e.get("regime", ""),
                                           e.get("candidate", ""))):
            if "error" in entry:
                print("%-28s ERROR %s" % (entry["candidate"], entry["error"]))
                continue
            print("%-28s %-13s steps=%4d prev=%.4f h1pos=%3d cascade=%4d "
                  "migrej=%.3f deprej=%.3f"
                  % (entry["candidate"], entry["regime"], entry["steps"],
                     entry["prevalence"], entry["h1_onset_positives"],
                     entry["cascade_events_in_stream"],
                     entry["migrate_rejection_rate"],
                     entry["deploy_rejection_rate"]))
        return 0

    if not args.io_first or not args.memory_first:
        raise SystemExit("pass --io-first NAME --memory-first NAME")
    chosen = {}
    for regime, name in (("io_first", args.io_first),
                         ("memory_first", args.memory_first)):
        row = candidate_row(name, table)
        if row.get("regime") != regime:
            raise SystemExit("candidate %s is a %s candidate, not %s"
                             % (name, row.get("regime"), regime))
        if "error" in row:
            raise SystemExit("candidate %s failed to run: %s"
                             % (name, row["error"]))
        if not args.allow_short:
            require_full_length(row)
        params = dict(row["params"])
        if not params:
            raise SystemExit("candidate %s is the registered baseline; a "
                             "calibration selection must change something"
                             % name)
        chosen[regime] = {
            "candidate": name, "params": params,
            "measured": {
                "steps": row["steps"], "window": row["window"],
                "prevalence": row["prevalence"],
                "positive_hoststeps": row["positive_hoststeps"],
                "h1_onset_positives": row["h1_onset_positives"],
                "cascade_events_in_stream": row["cascade_events_in_stream"],
                "label_class_counts": row["label_class_counts"],
                "deployment_rejection_rate": row["deploy_rejection_rate"],
                "migration_rejection_rate": row["migrate_rejection_rate"],
                "stream": row["stream"],
            },
            "screening_targets": dict(TARGETS),
        }

    payload = {
        "kind": "round2a_selected_generator",
        "protocol": "023", "round": "2A",
        "family": "cascade_v3",
        "generator_module": "simulator/workload/BitbrainWorkloadProtocol023",
        "generator_source_sha256": sha(
            ROOT / "simulator/workload/BitbrainWorkloadProtocol023.py"),
        "generator_git_parent": ("d7bbc8c75221ed8707bee80ab30fc83adbecc500"
                                 " (round-1 review HEAD)"),
        "calibration_kind": "data_only_marginal_calibration",
        "why": [
            "round 1 gave io-first only 5 h=1 within-regime positives "
            "(directive §5) and separated the A/B/C marginals beyond the plan "
            "§9 tolerances (directive §4)",
            "the io-first deficit was measured to be structural: a single "
            "io-first disk phase reaches at most ~0.83 of the registered disk "
            "capacity, so a disk cell needs two co-located disk phases "
            "(diagnostics/disk_onset_deficit.json)",
            "the two levers that were measured to move the io-first positive "
            "budget without touching the registered law are the cascade "
            "probability and the registered disk retained peak (16000 -> the "
            "registered cap 24000)",
        ],
        "no_model_metric_was_read": True,
        "seeds": {"used": [700], "frozen": [701, 702, 703],
                  "note": "calibration is seed 700 only (directive §4)"},
        "regimes": {regime: entry["params"]
                    for regime, entry in chosen.items()},
        "evidence": chosen,
        "frozen_regimes": {
            "compute_first": {
                "frozen": True,
                "reason": ("it must stay byte-identical to Protocol 022's "
                           "cascade_v2, so any A-vs-B/C difference stays "
                           "attributable to the resource order"),
            }
        },
        "confirmation_seeds_untouched": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf8")
    print(json.dumps({"written": str(args.out.relative_to(ROOT)),
                      "regimes": payload["regimes"],
                      "evidence": {r: {"h1_onset_positives":
                                       e["measured"]["h1_onset_positives"],
                                       "prevalence":
                                       e["measured"]["prevalence"]}
                                   for r, e in chosen.items()}},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
