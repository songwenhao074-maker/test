"""Deferred, sequential smoke verifier for Protocol-020 R0 prequential runs.

The verifier is intentionally a harness, not an experiment launcher for this
turn.  When invoked by the main agent it runs the registered Protocol-020
runner exactly three times, one process at a time:

* D / prequential_v1 to cursor 20 in one uninterrupted stop-after-20 run;
* D / prequential_v1 to cursor 10 in a fresh output;
* the same output resumed to cursor 20.

Every child run gets a fresh output directory below the dated R0 directory.
The harness records stdout, stderr and return code, then compares the two
cursor-20 results after recursively removing timing fields.  Candidate rows
are allowed to be empty; an empty candidate ledger is explicitly reported and
is never treated as evidence that a candidate is correct.
"""

from __future__ import annotations

from datetime import datetime
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_ftmoe_protocol020.py"
CHECKPOINT = ROOT / "artifacts" / "ftmoe_online" / "protocol_020" / \
    "s6" / "adapted_v4_seed1" / "best.pt"
STREAM = ROOT / "artifacts" / "ftmoe_online" / "protocol_020" / \
    "drift_streams" / "dev_seed500_steps2000"
DYNAMIC_CONFIG = ROOT / "artifacts" / "ftmoe_online" / "protocol_020" / \
    "continuation" / "dynamic_v3.json"
R0_ROOT = ROOT / "artifacts" / "ftmoe_online" / "protocol_020" / \
    "revision_20260909" / "r0"
RAM_GUARD_GIB = 3.0
CURSOR = 20
TIMING_JSONL = (
    "seconds", "elapsed_seconds", "prediction_seconds", "update_seconds",
    "teacher_forward_seconds", "candidate_forward_seconds",
    "shadow_forward_seconds", "runtime_seconds", "duration_seconds",
    "latency_seconds", "timestamp", "started_at", "finished_at",
)
TIMING_SUFFIXES = ("_seconds", "_milliseconds", "_ms", "_duration", "_latency")
JSONL_FILES = (
    "predictions.jsonl",
    "candidate_predictions.jsonl",
    "qualification.jsonl",
    "causal_audit.jsonl",
)


def available_ram_gib() -> float:
    """Return available physical RAM without allocating model memory."""
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.ullAvailPhys / (1024 ** 3)
    return float(psutil.virtual_memory().available) / (1024 ** 3)


def model_processes() -> list[dict[str, Any]]:
    """Find likely FT-MoE model processes, excluding this verifier."""
    found = []
    own_pid = os.getpid()
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            if process.info["pid"] == own_pid:
                continue
            cmdline = process.info.get("cmdline") or []
            command = " ".join(str(part) for part in cmdline).lower()
            is_model = (
                "run_ftmoe_protocol020.py" in command or
                "run_ftmoe_protocol019" in command or
                "train_ftmoe_" in command or
                "evaluate_ftmoe_" in command
            )
            if is_model:
                found.append({
                    "pid": process.info["pid"],
                    "name": process.info.get("name"),
                    "cmdline": cmdline,
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def require_paths() -> None:
    for path in (RUNNER, CHECKPOINT, STREAM / "manifest.json",
                 STREAM / "stream.npz", DYNAMIC_CONFIG):
        if not path.exists():
            raise FileNotFoundError(path)


def timing_key(key: Any) -> bool:
    name = str(key).lower()
    return (
        name in TIMING_JSONL or
        name.endswith(TIMING_SUFFIXES) or
        "elapsed" in name or
        "duration" in name or
        "latency" in name or
        "timestamp" in name
    )


def _normalise_float(value: float) -> Any:
    if math.isnan(value):
        return {"__nan__": True}
    if math.isinf(value):
        return {"__inf__": 1 if value > 0 else -1}
    return value


def _array_bytes(array: np.ndarray) -> bytes:
    array = np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        # Canonicalize NaN payloads so untouched placeholders compare equal.
        array = array.copy()
        array[np.isnan(array)] = 0
    return array.tobytes()


def canonical(value: Any) -> Any:
    """JSON-like, timing-free representation with NaN-safe tensor digests."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return {
            "__tensor__": {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(_array_bytes(array)).hexdigest(),
            }
        }
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(_array_bytes(value)).hexdigest(),
            }
        }
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, dict):
        return {
            str(key): canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not timing_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        return _normalise_float(value)
    return value


def state_model_hash(model_state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(model_state):
        value = model_state[name]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        array = value.detach().cpu().numpy()
        digest.update(str(name).encode("utf8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(_array_bytes(array))
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def new_output_root() -> Path:
    R0_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output = R0_ROOT / f"runner_smoke_{stamp}"
    suffix = 2
    while output.exists():
        output = R0_ROOT / f"runner_smoke_{stamp}_{suffix}"
        suffix += 1
    output.mkdir()
    return output


def child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["FTMOE020_RAM_GUARD_GIB"] = f"{RAM_GUARD_GIB:.1f}"
    environment["PYTHONHASHSEED"] = "0"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return environment


def run_child(name: str, output: Path, extra: list[str], log_dir: Path) -> dict[str, Any]:
    available = available_ram_gib()
    if available < RAM_GUARD_GIB:
        raise RuntimeError(
            f"RAM guard failed before {name}: {available:.3f} GiB available"
        )
    existing = model_processes()
    if existing:
        raise RuntimeError(f"Other model process detected before {name}: {existing}")

    is_resume = "--resume" in extra
    if is_resume:
        if not output.exists():
            raise FileNotFoundError(
                f"Resume output does not exist for {name}: {output}"
            )
    elif output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing runner output for {name}: {output}"
        )
    log_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, str(RUNNER),
        "--method", "D",
        "--phase", "R0-smoke",
        "--model-seed", "1",
        "--checkpoint-path", str(CHECKPOINT),
        "--stream", str(STREAM),
        "--output", str(output),
        "--base-lr", "1e-4",
        "--dynamic-config", str(DYNAMIC_CONFIG),
        "--validation-protocol", "prequential_v1",
        *extra,
    ]
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf8"
    )
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            env=child_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = int(result.returncode)
    except Exception as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        returncode = -1
    (log_dir / "stdout.txt").write_text(stdout, encoding="utf8")
    (log_dir / "stderr.txt").write_text(stderr, encoding="utf8")
    (log_dir / "returncode.txt").write_text(str(returncode), encoding="ascii")
    return {
        "name": name,
        "output": str(output),
        "command": command,
        "ram_available_gib_before": available,
        "returncode": returncode,
        "log_dir": str(log_dir),
        "stdout_file": str(log_dir / "stdout.txt"),
        "stderr_file": str(log_dir / "stderr.txt"),
        "returncode_file": str(log_dir / "returncode.txt"),
    }


def assertion(
    report: dict[str, Any], name: str, passed: bool, detail: Any = None
) -> bool:
    row = {"name": name, "passed": bool(passed)}
    if detail is not None:
        row["detail"] = detail
    report.setdefault("assertions", []).append(row)
    return bool(passed)


def compare_jsonl(
    report: dict[str, Any], continuous: Path, resumed: Path
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for relative in JSONL_FILES:
        first = read_jsonl(continuous / relative)
        second = read_jsonl(resumed / relative)
        first_canonical = canonical(first)
        second_canonical = canonical(second)
        equal = first_canonical == second_canonical
        comparison[relative] = {
            "continuous_count": len(first),
            "resumed_count": len(second),
            "canonical_equal_ignoring_timing": equal,
        }
        assertion(
            report,
            f"jsonl_equal:{relative}",
            equal,
            {"continuous_count": len(first), "resumed_count": len(second)},
        )

    predictions = read_jsonl(continuous / "predictions.jsonl")
    qualifications = read_jsonl(continuous / "qualification.jsonl")
    candidates = read_jsonl(continuous / "candidate_predictions.jsonl")
    audits = read_jsonl(continuous / "causal_audit.jsonl")
    steps = [int(row.get("step", -1)) for row in predictions]
    assertion(report, "continuous_has_20_predictions", len(predictions) == 20, len(predictions))
    assertion(report, "prediction_steps_are_1_to_20", steps == list(range(1, 21)), steps)
    qualification_indices = [int(row.get("window_index", -1)) for row in qualifications]
    assertion(
        report,
        "continuous_has_19_matured_records",
        len(qualifications) == 19 and qualification_indices == list(range(19)),
        {"count": len(qualifications), "window_indices": qualification_indices},
    )
    matured_flags = [bool(row.get("label_matured")) for row in qualifications]
    assertion(report, "all_qualification_records_matured", all(matured_flags), matured_flags)

    prediction_teachers = []
    for row in predictions:
        view = row.get("r0_prediction")
        prediction_teachers.append(
            isinstance(view, dict) and isinstance(view.get("teacher"), dict)
        )
    qualification_teachers = [isinstance(row.get("teacher"), dict) for row in qualifications]
    candidate_teachers = [isinstance(row.get("teacher"), dict) for row in candidates]
    teacher_ok = (
        len(predictions) == 20 and all(prediction_teachers) and
        len(qualifications) == 19 and all(qualification_teachers) and
        all(candidate_teachers)
    )
    assertion(
        report,
        "teacher_information_present",
        teacher_ok,
        {"prediction_rows": sum(prediction_teachers),
         "qualification_rows": sum(qualification_teachers),
         "candidate_rows": sum(candidate_teachers)},
    )
    assertion(report, "causal_audit_has_records", len(audits) > 0, len(audits))
    comparison["record_counts"] = {
        "predictions": len(predictions),
        "matured_qualifications": len(qualifications),
        "candidate_predictions": len(candidates),
        "causal_audit": len(audits),
    }
    comparison["candidate_correctness_claim"] = False
    comparison["candidate_note"] = (
        "candidate ledger may be empty at 20 steps; this smoke does not establish "
        "candidate correctness. Use the real candidate unit/integration tests."
    )
    return comparison


def compare_prefix_arrays(
    report: dict[str, Any], first: dict[str, Any], second: dict[str, Any]
) -> None:
    first_predictions = first.get("predictions", {})
    second_predictions = second.get("predictions", {})
    keys = (
        "probability", "class_probability", "labels", "raw_labels",
        "model_version", "expert_count", "mean_active", "unmatched_ratio",
    )
    prefix_results = {}
    for key in keys:
        if key not in first_predictions or key not in second_predictions:
            prefix_results[key] = False
            continue
        first_array = np.asarray(first_predictions[key])
        second_array = np.asarray(second_predictions[key])
        prefix_length = CURSOR + 1 if key == "raw_labels" else CURSOR
        first_prefix = first_array[:prefix_length]
        second_prefix = second_array[:prefix_length]
        prefix_results[key] = bool(
            first_prefix.shape == second_prefix.shape and
            np.array_equal(first_prefix, second_prefix, equal_nan=True)
        )
    assertion(report, "prediction_prefixes_equal_nan_safe", all(prefix_results.values()), prefix_results)


def compare_resume_state(
    report: dict[str, Any], continuous: Path, resumed: Path
) -> dict[str, Any]:
    first_outer = torch.load(continuous / "resume.pt", map_location="cpu", weights_only=False)
    second_outer = torch.load(resumed / "resume.pt", map_location="cpu", weights_only=False)
    first = first_outer["session"]
    second = second_outer["session"]
    first_model = first["model"]
    second_model = second["model"]
    model_keys_equal = list(first_model.keys()) == list(second_model.keys())
    tensor_mismatches = []
    if model_keys_equal:
        for key in first_model:
            left = first_model[key]
            right = second_model[key]
            try:
                torch.testing.assert_close(
                    left, right, rtol=0.0, atol=0.0, equal_nan=True
                )
            except (AssertionError, RuntimeError) as exc:
                tensor_mismatches.append({"key": key, "error": str(exc)})
    model_equal = model_keys_equal and not tensor_mismatches
    first_hash = state_model_hash(first_model)
    second_hash = state_model_hash(second_model)
    assertion(report, "resume_model_state_equal", model_equal, tensor_mismatches[:3])
    assertion(report, "resume_model_hash_equal", first_hash == second_hash,
              {"continuous": first_hash, "resumed": second_hash})

    state_comparison: dict[str, Any] = {
        "model_hash_continuous": first_hash,
        "model_hash_resumed": second_hash,
        "model_tensor_mismatches": tensor_mismatches,
    }
    for key in ("optimizer", "exposure", "rare_exposure", "update_number"):
        equal = canonical(first.get(key)) == canonical(second.get(key))
        state_comparison[f"{key}_equal"] = equal
        assertion(report, f"resume_{key}_equal", equal)

    update_count_equal = (
        int(first.get("update_number", -1)) == int(second.get("update_number", -1)) and
        len(first.get("updates", [])) == len(second.get("updates", [])) and
        len(first.get("updates", [])) == int(first.get("update_number", -1)) and
        len(second.get("updates", [])) == int(second.get("update_number", -1))
    )
    state_comparison["update_count"] = {
        "continuous": int(first.get("update_number", -1)),
        "resumed": int(second.get("update_number", -1)),
    }
    assertion(report, "resume_update_count_consistent", update_count_equal,
              state_comparison["update_count"])

    first_dynamic = first.get("dynamic", {})
    second_dynamic = second.get("dynamic", {})
    dynamic_fields = (
        "pending_predictions", "qualification_records", "matured_indices",
        "shadow_validation", "shadow_validation_indices", "causal_audit",
        "dynamic_events", "trigger", "candidate_vectors", "pending_observations",
        "topology_version", "ramp_version", "shadow_started",
        "shadow_training_steps", "cooldown_until", "shadow_optimizer",
    )
    dynamic_results = {}
    for key in dynamic_fields:
        equal = canonical(first_dynamic.get(key)) == canonical(second_dynamic.get(key))
        dynamic_results[key] = equal
        assertion(report, f"resume_dynamic:{key}_equal", equal)
    state_comparison["dynamic_fields_equal"] = dynamic_results

    pending_first = first_dynamic.get("pending_predictions", {})
    pending_second = second_dynamic.get("pending_predictions", {})
    qualification_first = first_dynamic.get("qualification_records", [])
    qualification_second = second_dynamic.get("qualification_records", [])
    pending_count_ok = len(pending_first) == 1 and len(pending_second) == 1
    pending_keys_ok = {int(key) for key in pending_first} == {19} and \
        {int(key) for key in pending_second} == {19}
    qualification_count_ok = len(qualification_first) == 19 and \
        len(qualification_second) == 19
    matured_first = {int(index) for index in first_dynamic.get("matured_indices", [])}
    matured_second = {int(index) for index in second_dynamic.get("matured_indices", [])}
    matured_ok = matured_first == set(range(19)) and matured_second == set(range(19))
    state_comparison["pending_and_qualification_view"] = {
        "pending_counts": [len(pending_first), len(pending_second)],
        "pending_keys": [sorted(pending_first), sorted(pending_second)],
        "qualification_counts": [len(qualification_first), len(qualification_second)],
        "matured_indices": [sorted(matured_first), sorted(matured_second)],
    }
    assertion(report, "one_unmatured_pending_record", pending_count_ok and pending_keys_ok,
              state_comparison["pending_and_qualification_view"])
    assertion(report, "nineteen_qualification_records_in_resume", qualification_count_ok)
    assertion(report, "nineteen_matured_indices_in_resume", matured_ok)

    violation_counts = [
        int(first_dynamic.get("audit_violation_count", -1)),
        int(second_dynamic.get("audit_violation_count", -1)),
    ]
    assertion(report, "causal_audit_violation_count_zero", violation_counts == [0, 0], violation_counts)
    compare_prefix_arrays(report, first, second)
    state_comparison["outer_timing_free_equal"] = canonical(first_outer) == canonical(second_outer)
    assertion(report, "resume_state_equal_ignoring_timing", state_comparison["outer_timing_free_equal"])
    return state_comparison


def main() -> int:
    # This is the deferred harness entry point.  No child is started until the
    # script is explicitly invoked by the main agent after production freeze.
    require_paths()
    output_root = new_output_root()
    continuous = output_root / "uninterrupted20"
    resumed = output_root / "split10_resume20"
    report: dict[str, Any] = {
        "verification": "ftmoe_protocol020_r0_runner_smoke",
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.executable,
        "runner": str(RUNNER),
        "fixed_inputs": {
            "method": "D",
            "model_seed": 1,
            "checkpoint": str(CHECKPOINT),
            "stream": str(STREAM),
            "base_lr": 1e-4,
            "dynamic_config": str(DYNAMIC_CONFIG),
            "validation_protocol": "prequential_v1",
            "ram_guard_gib": RAM_GUARD_GIB,
        },
        "output_root": str(output_root),
        "runs": [],
        "assertions": [],
    }
    failure: Exception | None = None
    try:
        report["runs"].append(run_child(
            "uninterrupted20", continuous, ["--stop-after", "20"],
            output_root / "harness_logs" / "uninterrupted20"
        ))
        if report["runs"][-1]["returncode"] != 0:
            raise RuntimeError("uninterrupted20 runner failed")
        report["runs"].append(run_child(
            "split10", resumed, ["--stop-after", "10"],
            output_root / "harness_logs" / "split10"
        ))
        if report["runs"][-1]["returncode"] != 0:
            raise RuntimeError("split10 runner failed")
        report["runs"].append(run_child(
            "resume20", resumed, ["--resume", "--stop-after", "20"],
            output_root / "harness_logs" / "resume20"
        ))
        if report["runs"][-1]["returncode"] != 0:
            raise RuntimeError("resume20 runner failed")
        report["jsonl_comparison"] = compare_jsonl(report, continuous, resumed)
        report["resume_comparison"] = compare_resume_state(report, continuous, resumed)
        report["status"] = "passed" if all(
            row["passed"] for row in report["assertions"]
        ) else "failed"
    except Exception as exc:
        failure = exc
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (output_root / "verification.json").write_text(
            json.dumps(report, indent=2, allow_nan=True), encoding="utf8"
        )
    if failure is not None:
        raise failure
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
