"""Protocol 023 S2 — serial stage runner for the four registered streams.

Plan §21 forbids the pattern where a wrapper turned a child's stderr warning into
an apparent exit code.  Every S2 stream therefore runs as a REAL subprocess
launched with ``sys.executable``, and the parent records the true
``subprocess.CompletedProcess.returncode`` -- never an inferred one
(``run_ftmoe_protocol022_pilot.py`` is the reference for that discipline).  The
verifier is launched the same way and its exit code is recorded next to the
stream it judged.

Registered order (one process at a time, strictly sequential):

    1. --mode single --regime compute_first
    2. --mode single --regime memory_first
    3. --mode single --regime io_first
    4. --mode dev

After every stream the independent verifier is run on the produced directory and
its JSON verdict is folded into the report (per-regime gate verdict, per-regime
onset counts, the cross-hit confound count and the measured threshold
admissibility).

Report: ``artifacts/ftmoe_online/protocol_023/s2_run_report.json`` -- per stream
the exact command line, start/end UTC, elapsed seconds, the real child exit
code, the output directory, the stream sha256, the verifier exit code and the
per-regime gate verdict.

Usage:
    python run_ftmoe_protocol023_s2.py
    python run_ftmoe_protocol023_s2.py --streams single_io_first,dev
    python run_ftmoe_protocol023_s2.py --resume
"""
import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_ftmoe_protocol023_stream import (       # noqa: E402
    OUT, REGISTERED_REGIME_IDS, registered_steps, stream_tag)

PROTOCOL = "023"
STAGE = "S2"
COLLECTOR = ROOT / "prepare_ftmoe_protocol023_stream.py"
VERIFIER = ROOT / "verify_ftmoe_protocol023_stream.py"
REPORT = ROOT / "artifacts/ftmoe_online/protocol_023/s2_run_report.json"
PYTHON = sys.executable

#: The registered stream order.  ``--streams`` selects a subset but the
#: registered order is always preserved.
STREAM_SPECS = (
    {"stream": "single_compute_first", "mode": "single", "regime": "compute_first"},
    {"stream": "single_memory_first", "mode": "single", "regime": "memory_first"},
    {"stream": "single_io_first", "mode": "single", "regime": "io_first"},
    {"stream": "dev", "mode": "dev", "regime": None},
)
#: Accepted aliases for ``--streams`` (bare regime names, ``single:<regime>``).
STREAM_ALIASES = {"compute_first": "single_compute_first",
                  "memory_first": "single_memory_first",
                  "io_first": "single_io_first"}
for _regime in REGISTERED_REGIME_IDS:
    STREAM_ALIASES["single:%s" % _regime] = "single_%s" % _regime
    STREAM_ALIASES["single_%s" % _regime] = "single_%s" % _regime


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def utcnow():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf8")
    temporary.replace(path)


def resolve_streams(requested):
    """Registered specs in registered order for a ``--streams`` selection."""
    names = [name.strip() for name in requested.split(",") if name.strip()]
    resolved, unknown = [], []
    for name in names:
        canonical = STREAM_ALIASES.get(name, name)
        if canonical not in [spec["stream"] for spec in STREAM_SPECS]:
            unknown.append(name)
        elif canonical not in resolved:
            resolved.append(canonical)
    if unknown:
        raise SystemExit("Unregistered stream name(s) %s; registered: %s"
                         % (unknown, [s["stream"] for s in STREAM_SPECS]))
    if not names:
        raise SystemExit("--streams was empty; registered: %s"
                         % ([s["stream"] for s in STREAM_SPECS],))
    return [spec for spec in STREAM_SPECS if spec["stream"] in resolved]


def tail(text, lines=8):
    return text.strip().splitlines()[-lines:] if text and text.strip() else []


def child_environment():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf8"
    return env


def run_child(command):
    """Run one child with a REAL return code and captured output."""
    started = time.perf_counter()
    started_at = utcnow()
    completed = subprocess.run(command, cwd=str(ROOT), env=child_environment(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - started
    return {
        "command": [str(part) for part in command],
        "command_line": subprocess.list2cmdline([str(p) for p in command]),
        "exit_code": int(completed.returncode),
        "exit_code_source": "subprocess.CompletedProcess.returncode (real child code)",
        "started_at": started_at,
        "finished_at": utcnow(),
        "elapsed_seconds": elapsed,
        "stdout": completed.stdout.decode("utf8", "replace"),
        "stderr": completed.stderr.decode("utf8", "replace"),
    }


def parse_verdict(stdout):
    """The verifier prints one JSON document; take the last parseable line."""
    for line in reversed([l for l in stdout.splitlines() if l.strip()]):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except Exception:
        return None


def audit_gate(path):
    audit = load_json(Path(path) / "unseen_data_audit.json")
    if not audit:
        return None
    return {canonical: {"passed": bool((block.get("gate") or {}).get("passed")),
                        "checks": (block.get("gate") or {}).get("checks")}
            for canonical, block in (audit.get("per_regime") or {}).items()}


def manifest_matches_request(directory, spec):
    """Does the stream's own manifest declare the requested stream?"""
    manifest = load_json(Path(directory) / "manifest.json") or {}
    observed = {"mode": manifest.get("mode"), "regime": manifest.get("regime"),
                "steps": manifest.get("steps")}
    expected = {"mode": spec["mode"], "regime": spec["regime"],
                "steps": registered_steps(spec["mode"], spec["regime"])}
    return manifest, observed, expected, observed == expected


def verify_dir(directory):
    """Run the independent verifier on one stream directory."""
    child = run_child([PYTHON, str(VERIFIER), str(directory)])
    verdict = parse_verdict(child["stdout"])
    return child, verdict


def stream_entry(spec, output_root, resume):
    directory = Path(output_root) / stream_tag(spec["mode"], spec["regime"])
    relative = str(directory.relative_to(ROOT)).replace("\\", "/") \
        if str(directory).startswith(str(ROOT)) else str(directory)
    entry = {
        "stream": spec["stream"], "mode": spec["mode"], "regime": spec["regime"],
        "output_dir": relative, "output_dir_abs": str(directory),
        "registered_steps": registered_steps(spec["mode"], spec["regime"]),
        "status": "failed", "notes": [],
        "command": None, "command_line": None, "exit_code": None,
        "exit_code_source": None, "started_at": None, "finished_at": None,
        "elapsed_seconds": None, "stream_sha256": None,
        "verifier_command": None, "verifier_exit_code": None,
        "per_regime_gate": None, "audit_per_regime_gate": None,
        "gate_agreement": None, "verdict": None,
    }
    if directory.exists():
        if not resume:
            entry["notes"].append(
                "output directory already exists; refusing to overwrite it.  "
                "Re-run with --resume to accept it if the verifier passes, or "
                "move the directory aside (a failed attempt must be reviewed, "
                "not deleted) before a fresh collection.")
            return entry
        child, verdict = verify_dir(directory)
        entry["verifier_command"] = child["command"]
        entry["verifier_command_line"] = child["command_line"]
        entry["verifier_exit_code"] = child["exit_code"]
        entry["verifier_started_at"] = child["started_at"]
        entry["verifier_finished_at"] = child["finished_at"]
        entry["verifier_elapsed_seconds"] = child["elapsed_seconds"]
        entry["verifier_stdout_tail"] = tail(child["stdout"], 3)
        entry["verifier_stderr_tail"] = tail(child["stderr"])
        entry["verdict"] = verdict
        stream_file = directory / "stream.npz"
        if stream_file.is_file():
            entry["stream_sha256"] = sha(stream_file)
        manifest, observed, expected, matches = manifest_matches_request(directory,
                                                                        spec)
        entry["manifest_declares_requested_stream"] = matches
        entry["manifest_observed"] = observed
        entry["manifest_expected"] = expected
        if child["exit_code"] == 0 and matches:
            entry["status"] = "skipped_verified"
            entry["exit_code"] = 0
            entry["exit_code_source"] = "resume: existing verified stream (no child launched)"
            entry["elapsed_seconds"] = 0.0
            entry["notes"].append(
                "resume: the existing stream verifies, so it was not re-collected")
        elif not matches:
            entry["notes"].append(
                "resume: the existing directory holds a DIFFERENT stream than "
                "the one requested (manifest says %s, requested %s); nothing was "
                "collected and the directory was left untouched"
                % (observed, expected))
        else:
            entry["notes"].append(
                "resume: the existing directory does NOT verify (verifier exit "
                "%d); it was left untouched for review and nothing was "
                "collected" % child["exit_code"])
    else:
        command = [PYTHON, "-u", str(COLLECTOR), "--mode", spec["mode"]]
        if spec["regime"] is not None:
            command += ["--regime", spec["regime"]]
        child = run_child(command)
        entry.update({"command": child["command"],
                      "command_line": child["command_line"],
                      "exit_code": child["exit_code"],
                      "exit_code_source": child["exit_code_source"],
                      "started_at": child["started_at"],
                      "finished_at": child["finished_at"],
                      "elapsed_seconds": child["elapsed_seconds"],
                      "stdout_tail": tail(child["stdout"], 5),
                      "stderr_tail": tail(child["stderr"], 10),
                      "stderr_note": ("torch/numpy warnings on stderr are "
                                      "recorded verbatim and are NOT interpreted "
                                      "as failure; the exit code above is the "
                                      "child's real return code")})
        stream_file = directory / "stream.npz"
        if stream_file.is_file():
            entry["stream_sha256"] = sha(stream_file)
        if child["exit_code"] != 0:
            entry["notes"].append(
                "the collector failed; see failure.json / generation.log in the "
                "output directory")
            entry["audit_per_regime_gate"] = audit_gate(directory)
            return entry
        child, verdict = verify_dir(directory)
        entry["verifier_command"] = child["command"]
        entry["verifier_command_line"] = child["command_line"]
        entry["verifier_exit_code"] = child["exit_code"]
        entry["verifier_started_at"] = child["started_at"]
        entry["verifier_finished_at"] = child["finished_at"]
        entry["verifier_elapsed_seconds"] = child["elapsed_seconds"]
        entry["verifier_stdout_tail"] = tail(child["stdout"], 3)
        entry["verifier_stderr_tail"] = tail(child["stderr"])
        entry["verdict"] = verdict
        manifest, observed, expected, matches = manifest_matches_request(directory,
                                                                         spec)
        entry["manifest_declares_requested_stream"] = matches
        entry["manifest_observed"] = observed
        entry["manifest_expected"] = expected
        entry["status"] = ("ok" if (child["exit_code"] == 0 and matches)
                           else "collected_unverified")
        if child["exit_code"] != 0:
            entry["notes"].append(
                "the stream was collected but the verifier rejected it; the "
                "broken checks are in verdict.failed_checks")

    verdict = entry.get("verdict") or {}
    manifest = load_json(directory / "manifest.json") or {}
    if verdict.get("per_regime_gate"):
        entry["per_regime_gate"] = verdict["per_regime_gate"]
    entry["audit_per_regime_gate"] = audit_gate(directory)
    if entry["per_regime_gate"] and entry["audit_per_regime_gate"]:
        entry["gate_agreement"] = all(
            bool(entry["per_regime_gate"].get(canonical, {}).get("passed"))
            == bool(block.get("passed"))
            for canonical, block in entry["audit_per_regime_gate"].items())
    if manifest.get("stream_sha256") and entry.get("stream_sha256"):
        entry["stream_sha256_matches_manifest"] = \
            manifest["stream_sha256"] == entry["stream_sha256"]
    if verdict:
        entry["verdict_summary"] = {
            "passed": verdict.get("passed"),
            "n_failed_checks": verdict.get("n_failed_checks"),
            "failed_checks": verdict.get("failed_checks"),
            "admissibility_ok": verdict.get("admissibility_ok"),
            "inadmissible_thresholds": verdict.get("inadmissible_thresholds"),
            "cross_hits": {canonical: block.get("cross_hits") for canonical, block
                           in (verdict.get("cross_regime_confound") or {}).items()},
            "per_regime_onsets": {canonical: block.get("n_onsets_with_task_map")
                                  for canonical, block
                                  in (verdict.get("per_regime_onsets") or {}).items()},
        }
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", default=None,
                        help="comma list of %s (registered order is preserved)"
                             % ([spec["stream"] for spec in STREAM_SPECS],))
    parser.add_argument("--resume", action="store_true",
                        help="skip a stream whose manifest exists AND verifies")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    output_root = Path(args.output_root) if args.output_root else OUT
    specs = (resolve_streams(args.streams) if args.streams
             else list(STREAM_SPECS))
    started_at = utcnow()
    clock = time.perf_counter()
    entries = []
    for spec in specs:                       # strictly one child at a time
        print(json.dumps({"starting": spec["stream"], "mode": spec["mode"],
                          "regime": spec["regime"]}, ensure_ascii=False),
              flush=True)
        entry = stream_entry(spec, output_root, args.resume)
        entries.append(entry)
        print(json.dumps({"stream": entry["stream"], "status": entry["status"],
                          "exit_code": entry["exit_code"],
                          "verifier_exit_code": entry["verifier_exit_code"],
                          "elapsed_seconds": (None if entry["elapsed_seconds"] is None
                                              else round(entry["elapsed_seconds"], 1)),
                          "per_regime_gate": {canonical: block.get("passed")
                                              for canonical, block
                                              in (entry["per_regime_gate"] or {}).items()}},
                         ensure_ascii=False), flush=True)
    failed_streams = [entry["stream"] for entry in entries
                      if entry["status"] == "failed"]
    unverified = [entry["stream"] for entry in entries
                  if entry["status"] == "collected_unverified"]
    verified_ok = [entry["stream"] for entry in entries
                   if entry["status"] == "ok"]
    skipped = [entry["stream"] for entry in entries
               if entry["status"] == "skipped_verified"]
    any_failed = bool(failed_streams or unverified)
    report = {
        "protocol": PROTOCOL, "stage": STAGE,
        "runner": "run_ftmoe_protocol023_s2.py",
        "python": PYTHON, "root": str(ROOT),
        "started_at": started_at, "finished_at": utcnow(),
        "elapsed_seconds": time.perf_counter() - clock,
        "resume": bool(args.resume),
        "requested_streams": args.streams,
        "stream_order": [spec["stream"] for spec in specs],
        "registered_stream_order": [spec["stream"] for spec in STREAM_SPECS],
        "children_at_a_time": 1,
        "sequential": True,
        "collector_sha256": sha(COLLECTOR),
        "verifier_sha256": sha(VERIFIER),
        "core022_sha256": sha(ROOT / "ftmoe_protocol022_core.py"),
        "core023_sha256": sha(ROOT / "ftmoe_protocol023_core.py"),
        "generator_sha256": sha(ROOT / "simulator/workload/BitbrainWorkloadProtocol023.py"),
        "output_root": str(output_root),
        "streams": entries,
        "n_streams": len(entries),
        "n_ok": len(verified_ok),
        "n_collected_unverified": len(unverified),
        "n_skipped_verified": len(skipped),
        "n_failed": len(failed_streams),
        "failed_streams": failed_streams,
        "unverified_streams": unverified,
        "any_child_failed": any_failed,
        "exit_code": 1 if any_failed else 0,
        "next_action": ("review the failed stream's failure.json / generation.log "
                        "and the verifier's failed_checks; a collected but "
                        "unverified stream must be reviewed before it is used"
                        if any_failed else
                        "read s2_run_report.json: every stream verified"),
    }
    write_json(args.report, report)
    print(json.dumps({"protocol": PROTOCOL, "stage": STAGE,
                      "streams": len(entries), "ok": len(verified_ok),
                      "skipped_verified": len(skipped), "failed": len(failed_streams),
                      "collected_unverified": len(unverified),
                      "exit_codes": [entry["exit_code"] for entry in entries],
                      "verifier_exit_codes": [entry["verifier_exit_code"]
                                              for entry in entries],
                      "per_regime_gate": {entry["stream"]: {
                          canonical: block.get("passed") for canonical, block
                          in (entry["per_regime_gate"] or {}).items()}
                          for entry in entries},
                      "any_child_failed": any_failed,
                      "report": str(args.report),
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)},
                     ensure_ascii=False), flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
