"""Protocol 023 round 2A — score every calibration candidate against the toolchain.

The S2.5 harness records raw label/cohort numbers while it collects.  This script
takes those collected candidate streams and runs the *real* instruments over
them -- the round-1 analyzer's own per-regime recomputation and this round's
H1-v2 follow-up counting -- so a candidate can be selected on the quantities the
gates actually read:

    onset event count            (marginal match vs A, directive §4)
    independent fault events     (H1-v2 >= 80)
    n_primary_window_followup    (H1-v2 >= 80)
    prevalence                   (H1-v2 3%-12%, spread <= 4 pp)
    deployment / migration rejection
    h=1 onset positives in the regime window (directive §5 >= 30)

Nothing is collected here; this is read-only scoring of already-collected
candidate streams.

Usage
-----
    python score_ftmoe_protocol023_s25_candidates.py
    python score_ftmoe_protocol023_s25_candidates.py --write selected_table.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_ftmoe_protocol023_round2a as r2a
import analyze_ftmoe_protocol023_s2 as base

CALIB = (ROOT / "artifacts/ftmoe_online/protocol_023/round2a/data_calibration"
                "/calibration_streams")
ROUND1 = (ROOT / "artifacts/ftmoe_online/protocol_023/development_streams")
OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/data_calibration"
CANONICAL = {"compute_first": "A", "memory_first": "B", "io_first": "C"}
RESOURCE = {"compute_first": "cpu", "memory_first": "ram", "io_first": "disk"}


def score_stream(stream_dir, canonical):
    """Every gate-relevant quantity of one collected candidate stream."""
    measured = base.measure_stream(stream_dir)
    block = measured["per_regime"][CANONICAL[canonical]]
    followup = r2a.measure_followup(stream_dir, canonical, stream_dir.name)
    budget = r2a.h1_positive_budget(stream_dir, canonical)
    return {
        "stream": stream_dir.name,
        "event_count": int(block["gate_stats"]["event_count"]),
        "independent_fault_events": int(
            block["gate_stats"]["independent_fault_events"]),
        "prevalence": float(block["gate_stats"]["prevalence"]),
        "primary_window_followup": followup["n_primary_window_followup"],
        "full_window_followup": followup["n_full_window_followup"],
        "any_window_followup": followup["n_any_window_followup"],
        "primary_response_positive": followup["primary_response_positive"],
        "deployment_rejection_rate": block["gate_stats"][
            "deployment_rejection_rate"],
        "migration_rejection_rate": block["gate_stats"][
            "migration_rejection_rate"],
        "mean_duration": block["gate_stats"]["mean_duration"],
        "peak_ratio": block["gate_stats"]["peak_ratio"],
        "worst_event_share": block["gate_stats"]["worst_event_share"],
        "h1_positive_budget": budget,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--write", default="candidate_scores.json")
    args = parser.parse_args()

    reference = {}
    for canonical in ("compute_first", "memory_first", "io_first"):
        path = ROUND1 / ("single_%s_seed700_steps1200" % canonical)
        reference[canonical] = score_stream(path, canonical)

    rows = []
    for regime_dir in sorted(CALIB.iterdir()) if CALIB.is_dir() else []:
        if not regime_dir.is_dir():
            continue
        name = regime_dir.name
        canonical = None
        for candidate, resources in (("compute_first", "compute_first"),
                                     ("memory_first", "memory_first"),
                                     ("io_first", "io_first")):
            if "_%s_" % resources in name:
                canonical = candidate
        if canonical is None:
            continue
        params = {}
        table_path = OUT / "candidate_table.json"
        if table_path.is_file():
            table = json.loads(table_path.read_text(encoding="utf8"))
            for entry in table["candidates"]:
                if entry.get("candidate") == name.split("_")[0] or \
                        name.startswith(entry.get("candidate", "\0")):
                    params = entry.get("params") or {}
        row = score_stream(regime_dir, canonical)
        row["regime"] = canonical
        row["candidate"] = name[: -len("_%s_%s" % (canonical,
                                                   name.split("_")[-1]))] \
            if name.endswith(name.split("_")[-1]) else name
        row["params"] = params
        rows.append(row)

    verdict = {
        "kind": "round2a_s25_candidate_scores",
        "reference_round1": reference,
        "candidates": rows,
        "note": ("scored with the same instruments the gates use; the "
                 "round-1 streams in reference_round1 are the frozen A/B/C "
                 "baseline"),
    }
    path = args.out / args.write
    path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf8")
    header = ("%-28s %-13s %6s %6s %5s %6s %6s %6s %6s" %
              ("candidate", "regime", "events", "indep", "prev", "primFU",
               "h1pos", "regrej", "migrej"))
    print(header)
    print("-" * len(header))
    for canonical in ("compute_first", "memory_first", "io_first"):
        r = reference[canonical]
        print("%-28s %-13s %6d %6d %5.3f %6d %6s %6.3f %6.3f" %
              ("ROUND1-" + r["stream"].split("_")[1], canonical,
               r["event_count"], r["independent_fault_events"],
               r["prevalence"],
               r["primary_window_followup"],
               max(v["positives"] for v in r["h1_positive_budget"].values()),
               r["deployment_rejection_rate"], r["migration_rejection_rate"]))
    for row in rows:
        print("%-28s %-13s %6d %6d %5.3f %6d %6d %6.3f %6.3f" %
              (row["candidate"], row["regime"], row["event_count"],
               row["independent_fault_events"], row["prevalence"],
               row["primary_window_followup"],
               max(v["positives"] for v in row["h1_positive_budget"].values()),
               row["deployment_rejection_rate"], row["migration_rejection_rate"]))
    print("written:", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
