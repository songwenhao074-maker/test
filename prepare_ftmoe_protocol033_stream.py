"""Protocol-033 5969-row rare-recurrence stream contract.

This module deliberately reuses the frozen Protocol-031 simulator plumbing while
owning a separate Protocol-033 registration, checkpoint identity, timeline and
RAM telemetry.  Generation never assembles the full stream here: the last
segment only seals immutable chunks and exits; assembly/audit run later in a
fresh process.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

import dill
import numpy as np
import psutil

import prepare_ftmoe_protocol031_stream as P31

ROOT = Path(__file__).resolve().parent
REGISTRATION_PATH = ROOT / "artifacts/ftmoe_online/protocol_033/scenario_registration.json"
SCENARIO_ID = "protocol033_rare_recurrence_5969_v1"
DATA_REVISION = "protocol033_data_revision_001"
PLAN_REVISION = 1
REGISTERED_SEED = 700
SOURCE_SERVICE = {"U": "S1", "V": "S3", "W": "S4"}
LOGICAL_BY_SOURCE = {v: k for k, v in SOURCE_SERVICE.items()}
COMMON_FEATURE_ORDER = list(P31.COMMON_FEATURE_ORDER)

# Frozen simulator support.  Historical Protocol-031 files are never modified.
ROOT25 = P31.ROOT25
configure = P31.configure
guard = P31.guard
assert_no_other_experiment = P31.assert_no_other_experiment
SCENARIO_PATH = P31.SCENARIO_PATH
DRIFT_CONFIG = P31.DRIFT_CONFIG
FAMILIAR_PHASE = P31.FAMILIAR_PHASE
COHORT = P31.COHORT
Protocol025ServiceTurnoverBWGD2 = P31.Protocol025ServiceTurnoverBWGD2
SERVICE_LAWS = P31.SERVICE_LAWS
SERVICE_MECHANISM_IDS = P31.SERVICE_MECHANISM_IDS
FORBIDDEN_MODEL_INPUTS = P31.FORBIDDEN_MODEL_INPUTS
_allocate = P31._allocate
_chunk_name = P31._chunk_name
_save_chunk = P31._save_chunk
_verify_chunks = P31._verify_chunks
_checkpoint_paths = P31._checkpoint_paths
_common_features = P31._common_features
sha = P31.sha

_MEMORY = None


def json_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration():
    reg = json.loads(REGISTRATION_PATH.read_text(encoding="utf8"))
    if reg.get("protocol") != "033" or int(reg.get("plan_revision", -1)) != PLAN_REVISION:
        raise AssertionError("Protocol033 registration revision mismatch")
    if reg.get("scenario_id") != SCENARIO_ID or reg.get("data_revision") != DATA_REVISION:
        raise AssertionError("Protocol033 scenario/data revision mismatch")
    if int(reg.get("replay_seed", -1)) != 700 or int(reg.get("model_seed", -1)) != 1:
        raise AssertionError("Protocol033 seed mismatch")
    if int(reg.get("scored_intervals", -1)) != 5968 or int(reg.get("guard_intervals", -1)) != 1:
        raise AssertionError("Protocol033 registered scored/guard length mismatch")
    if int(reg.get("total_intervals", -1)) != 5969:
        raise AssertionError("Protocol033 total row count mismatch")
    gen = reg["generation"]
    if float(gen["event_probability"]) != 0.30:
        raise AssertionError("Protocol033 event probability changed")
    if int(gen["chunk_intervals"]) != 200 or int(gen["segment_max_intervals"]) != 200:
        raise AssertionError("Protocol033 chunk/segment size changed")
    if int(gen["expected_chunk_count"]) != 30 or int(gen["final_chunk_intervals"]) != 169:
        raise AssertionError("Protocol033 chunk geometry changed")
    if int(reg["D_capacity_and_birth"]["birth_start_matured"]) != 600:
        raise AssertionError("Protocol033 birth start changed")
    if int(reg["D_capacity_and_birth"]["birth_every_matured"]) != 1000:
        raise AssertionError("Protocol033 birth cadence changed")
    return reg


def phase_table(reg=None):
    reg = registration() if reg is None else reg
    rows = []
    cursor = 0
    for item in reg["timeline"]:
        start, end = int(item["start"]), int(item["end"])
        logical = item["service"]
        if start != cursor or end - start != int(item["length"]):
            raise AssertionError("Protocol033 timeline is not contiguous")
        source = None if logical == "baseline" else SOURCE_SERVICE[str(logical)]
        rows.append({
            "name": str(item["name"]), "start": start, "end": end,
            "length": end - start,
            "logical_service": None if logical == "baseline" else str(logical),
            "service": source,
            "event_probability": 0.0 if source is None else float(reg["generation"]["event_probability"]),
            "kind": "baseline" if source is None else "service_response",
        })
        cursor = end
    if cursor != int(reg["scored_intervals"]):
        raise AssertionError("Protocol033 timeline sum mismatch")
    return rows


def _cgroup_limit_bytes():
    candidates = [Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")]
    values = []
    for p in candidates:
        try:
            text = p.read_text().strip()
            if text and text != "max":
                value = int(text)
                if 0 < value < (1 << 60):
                    values.append(value)
        except Exception:
            pass
    return min(values) if values else None


def _tree_rss_bytes():
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return int(total)


def init_memory_monitor(output):
    global _MEMORY
    vm = psutil.virtual_memory()
    cg = _cgroup_limit_bytes()
    effective = min(int(vm.total), int(cg)) if cg else int(vm.total)
    _MEMORY = {
        "protocol": "033", "stage": "generation",
        "effective_memory_limit_bytes": effective,
        "cgroup_memory_limit_bytes": cg,
        "system_total_bytes": int(vm.total),
        "soft_limit_fraction": 0.8,
        "soft_limit_bytes": int(effective * 0.8),
        "sample_every_intervals": 20,
        "samples": [], "soft_limit_crossed": False,
        "started_at_unix": time.time(),
        "output": str(Path(output)),
    }
    sample_memory("generation_start", 0)
    return _MEMORY


def sample_memory(label, step=None):
    if _MEMORY is None:
        return None
    vm = psutil.virtual_memory()
    rss = _tree_rss_bytes()
    row = {
        "label": str(label), "step": None if step is None else int(step),
        "time_unix": time.time(), "process_tree_rss_bytes": rss,
        "system_available_bytes": int(vm.available),
    }
    _MEMORY["samples"].append(row)
    if rss >= int(_MEMORY["soft_limit_bytes"]):
        _MEMORY["soft_limit_crossed"] = True
    return row


def write_memory_report(output):
    if _MEMORY is None:
        return
    samples = _MEMORY["samples"]
    report = dict(_MEMORY)
    report.pop("output", None)
    report["peak_process_tree_rss_bytes"] = max((x["process_tree_rss_bytes"] for x in samples), default=0)
    report["minimum_system_available_bytes"] = min((x["system_available_bytes"] for x in samples), default=None)
    p = Path(output) / "memory_profile_generation.json"
    p.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")


def phase_at(t, phases):
    # The collector calls this exactly once per generated row, giving us the
    # required interval-indexed RAM sampling without touching simulator logic.
    if int(t) % 20 == 0:
        sample_memory("interval", int(t))
    for i, p in enumerate(phases):
        if int(p["start"]) <= int(t) < int(p["end"]):
            return i, p
    return len(phases) - 1, phases[-1]


def source_identity():
    files = [
        REGISTRATION_PATH,
        ROOT / "prepare_ftmoe_protocol033_stream.py",
        ROOT / "maintenance/run_protocol033_segment.py",
        ROOT / "simulator/workload/BitbrainWorkloadProtocol025.py",
        SCENARIO_PATH,
        DRIFT_CONFIG,
    ]
    return {str(Path(p).relative_to(ROOT)): sha(p) for p in files if Path(p).is_file()}


def write_checkpoint(output, reg_sha, next_t, sim_state, chunks, arrays,
                     active_service, active_probability, applied_switches):
    output = Path(output)
    sample_memory("checkpoint_before", next_t)
    state_path, manifest_path = _checkpoint_paths(output)
    tmp = state_path.with_suffix(".tmp")
    identity = source_identity()
    payload = {
        "protocol": "033", "plan_revision": PLAN_REVISION,
        "scenario_id": SCENARIO_ID, "data_revision": DATA_REVISION,
        "registration_sha256": reg_sha, "source_sha256": identity,
        "implementation_source_commit": registration()["source_code_commit"],
        "next_t": int(next_t), "sim_state": sim_state,
        "active_service": active_service, "active_probability": float(active_probability),
        "applied_switches": list(applied_switches),
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": sim_state["torch"].get_rng_state(),
        },
    }
    with tmp.open("wb") as f:
        dill.dump(payload, f, protocol=dill.HIGHEST_PROTOCOL)
    os.replace(tmp, state_path)
    chunk_end = int(_verify_chunks(output, chunks))
    next_t = int(next_t)
    if next_t < chunk_end:
        raise AssertionError("Protocol033 transient checkpoint precedes immutable chunk end")
    transient_path = output / "transient_partial.npz"
    transient = None
    if next_t > chunk_end:
        np.savez_compressed(transient_path, **{k: v[chunk_end:next_t] for k, v in arrays.items()})
        transient = {"file": transient_path.name, "start": chunk_end, "end": next_t, "sha256": sha(transient_path)}
    elif transient_path.exists():
        transient_path.unlink()
    manifest = {
        "protocol": "033", "plan_revision": PLAN_REVISION,
        "scenario_id": SCENARIO_ID, "data_revision": DATA_REVISION,
        "registration_sha256": reg_sha, "source_sha256": identity,
        "next_t": next_t, "state_file": state_path.name, "state_sha256": sha(state_path),
        "chunks": list(chunks), "engineering_transient_checkpoint": transient,
        "registered_chunk_semantics_unchanged": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    sample_memory("checkpoint_after", next_t)
    try:
        manifest["checkpoint_size_bytes"] = int(state_path.stat().st_size)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    finally:
        write_memory_report(output)
    return manifest


def read_checkpoint(output, reg_sha, arrays):
    output = Path(output)
    state_path, manifest_path = _checkpoint_paths(output)
    if not (state_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError("Protocol033 resume checkpoint absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("protocol") != "033" or manifest.get("scenario_id") != SCENARIO_ID:
        raise AssertionError("wrong Protocol033 checkpoint identity")
    if manifest.get("registration_sha256") != reg_sha:
        raise AssertionError("Protocol033 registration changed since checkpoint")
    if manifest.get("source_sha256") != source_identity():
        raise AssertionError("Protocol033 implementation/source hash changed since checkpoint")
    if sha(state_path) != manifest.get("state_sha256"):
        raise AssertionError("Protocol033 resume state digest mismatch")
    chunks = list(manifest["chunks"])
    chunk_end = int(_verify_chunks(output, chunks))
    next_t = int(manifest["next_t"])
    transient = manifest.get("engineering_transient_checkpoint")
    if transient is None:
        if chunk_end != next_t:
            raise AssertionError("Protocol033 resume state/chunk boundary mismatch")
    else:
        start, end = int(transient["start"]), int(transient["end"])
        path = output / transient["file"]
        if start != chunk_end or end != next_t or end <= start:
            raise AssertionError("invalid Protocol033 transient coverage")
        if not path.is_file() or sha(path) != transient["sha256"]:
            raise AssertionError("Protocol033 transient digest mismatch")
        with np.load(path) as d:
            for key in arrays:
                arrays[key][start:end] = d[key]
    with state_path.open("rb") as f:
        payload = dill.load(f)
    if payload.get("protocol") != "033" or payload.get("registration_sha256") != reg_sha:
        raise AssertionError("wrong Protocol033 resume payload")
    if payload.get("source_sha256") != source_identity() or int(payload["next_t"]) != next_t:
        raise AssertionError("Protocol033 resume source/position mismatch")
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    payload["sim_state"]["torch"].set_rng_state(payload["rng"]["torch"])
    return payload, chunks, chunk_end


def finalize(output, reg, phases, arrays, workload, chunks, applied_switches,
             scheduler_weight, started, reg_sha):
    """Seal generation evidence only; never assemble the full stream here."""
    output = Path(output)
    count = int(reg["total_intervals"])
    end = int(_verify_chunks(output, chunks))
    if end != count:
        raise AssertionError("cannot seal incomplete Protocol033 chunk sequence")
    if len(chunks) != 30 or int(chunks[-1]["end"]) - int(chunks[-1]["start"]) != 169:
        raise AssertionError("Protocol033 final chunk geometry mismatch")
    (output / "events_final.json").write_text(
        json.dumps(workload.response_events, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (output / "applied_switches.json").write_text(
        json.dumps(applied_switches, indent=2, allow_nan=False) + "\n", encoding="utf8")
    generation = {
        "protocol": "033", "plan_revision": PLAN_REVISION,
        "scenario_id": SCENARIO_ID, "data_revision": DATA_REVISION,
        "complete": True, "row_target": count, "scored_target": int(reg["scored_intervals"]),
        "chunk_count": len(chunks), "last_chunk_rows": 169,
        "chunks": list(chunks), "registration_sha256": reg_sha,
        "source_sha256": source_identity(),
        "scheduler_checkpoint": str(scheduler_weight),
        "scheduler_checkpoint_sha256": sha(scheduler_weight),
        "generation_elapsed_seconds_this_process": float(time.perf_counter() - started),
        "assembly_pending": True, "input_audit_pending": True,
        "no_stream_npz_created_in_generation_process": True,
    }
    (output / "generation_manifest.json").write_text(
        json.dumps(generation, indent=2, allow_nan=False) + "\n", encoding="utf8")
    sample_memory("generation_complete", count)
    write_memory_report(output)
    # Compatibility with the inherited collector's terminal return shape only.
    # The wrapper rewrites generation_status.json to make audit-pending explicit.
    return {"stream_sha256": None, **generation}, {"audit_pass": False, "gates": {"input_audit_pending": True}}
