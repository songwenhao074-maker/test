"""Advance exactly the registered Protocol-033 stream by at most 200 rows.

The simulator loop is the frozen Protocol-031 collector.  Only its protocol
contract/checkpoint/finalize hooks are replaced with Protocol-033-owned hooks;
this preserves physical semantics while keeping 031 evidence immutable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_ftmoe_protocol033_stream as P
import maintenance.run_protocol031_segment as BASE

# Patch only the protocol-facing hooks used by the frozen collector.  The
# simulationStep loop, scheduler calls, workload laws and collection order are
# left byte-for-byte in the frozen implementation module.
BASE.P = P
BASE.write_checkpoint = P.write_checkpoint
BASE.read_checkpoint = P.read_checkpoint


def _rewrite_status(output, result):
    output = Path(output)
    reg = P.registration()
    status = dict(result or {})
    status.update({
        "protocol": "033", "plan_revision": P.PLAN_REVISION,
        "scenario_id": P.SCENARIO_ID, "data_revision": P.DATA_REVISION,
        "row_target": int(reg["total_intervals"]),
        "scored_target": int(reg["scored_intervals"]),
        "model_runs_started": 0,
    })
    if bool(status.get("complete")):
        status["stream_sha256"] = None
        status["input_audit_pending"] = True
        status["model_free_eligibility_passed"] = None
        status["assembly_pending"] = True
        status.pop("audit_pass", None)
        status.pop("gates", None)
    (output / "generation_status.json").write_text(
        json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return status


def collect_segment(output, max_intervals):
    output = Path(output)
    max_intervals = int(max_intervals)
    if max_intervals <= 0 or max_intervals > 200:
        raise ValueError("Protocol033 generation segment must be in [1,200]")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "registration_snapshot.json"
    if snapshot.is_file():
        if snapshot.read_bytes() != P.REGISTRATION_PATH.read_bytes():
            raise AssertionError("Protocol033 registration snapshot changed during same-stream resume")
    else:
        snapshot.write_bytes(P.REGISTRATION_PATH.read_bytes())
    P.init_memory_monitor(output)
    try:
        result = BASE.collect_segment(output, max_intervals)
        status = _rewrite_status(output, result)
        P.write_memory_report(output)
        return status
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "protocol": "033", "plan_revision": P.PLAN_REVISION,
            "scenario_id": P.SCENARIO_ID, "data_revision": P.DATA_REVISION,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "resume_available": P._checkpoint_paths(output)[0].is_file(),
            "no_reroll": True,
            "next_action": "repair deterministic engineering issue and resume only this registered stream from a verified checkpoint",
        }
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf8")
        P.write_memory_report(output)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-intervals", type=int, default=200)
    args = ap.parse_args()
    result = collect_segment(args.output, args.max_intervals)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
