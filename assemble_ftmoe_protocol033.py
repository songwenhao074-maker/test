"""Assemble Protocol-033 immutable chunks in a fresh process.

The generation process has already exited.  This script verifies chunk hashes,
materializes per-field .npy memmaps, then writes the compatibility stream.npz.
It does not make an eligibility decision; audit_ftmoe_protocol033.py owns that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil

import prepare_ftmoe_protocol033_stream as P


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def process_tree_rss():
    p = psutil.Process()
    total = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            total += c.memory_info().rss
        except Exception:
            pass
    return int(total)


def sample(samples, label, chunk=None):
    vm = psutil.virtual_memory()
    samples.append({
        "label": str(label), "chunk": chunk, "time_unix": time.time(),
        "process_tree_rss_bytes": process_tree_rss(),
        "system_available_bytes": int(vm.available),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    src, out = args.generation_root, args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Protocol033 assembly output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    field_dir = out / "fields"
    field_dir.mkdir()
    samples = []
    sample(samples, "assembly_start")

    reg = P.registration()
    reg_sha = P.json_sha(P.REGISTRATION_PATH)
    gen = J(src / "generation_manifest.json")
    resume = J(src / "resume_manifest.json")
    if gen.get("protocol") != "033" or resume.get("protocol") != "033":
        raise AssertionError("Protocol033 generation identity missing")
    if gen.get("registration_sha256") != reg_sha or resume.get("registration_sha256") != reg_sha:
        raise AssertionError("Protocol033 registration hash mismatch at assembly")
    if resume.get("source_sha256") != P.source_identity():
        raise AssertionError("Protocol033 source hash mismatch at assembly")
    chunks = list(resume["chunks"])
    if P._verify_chunks(src, chunks) != int(reg["total_intervals"]):
        raise AssertionError("Protocol033 chunks do not cover 5969 rows")
    if len(chunks) != 30:
        raise AssertionError("Protocol033 expected exactly 30 immutable chunks")
    if int(chunks[-1]["end"]) - int(chunks[-1]["start"]) != 169:
        raise AssertionError("Protocol033 final chunk is not 169 rows")

    # Infer the exact field schema from the first immutable chunk.
    with np.load(src / "chunks" / chunks[0]["file"], allow_pickle=False) as first:
        keys = list(first.files)
        schema = {k: {"dtype": str(first[k].dtype), "tail_shape": list(first[k].shape[1:])} for k in keys}
        memmaps = {
            k: np.lib.format.open_memmap(
                field_dir / (k + ".npy"), mode="w+", dtype=first[k].dtype,
                shape=(int(reg["total_intervals"]),) + first[k].shape[1:])
            for k in keys
        }

    for ci, rec in enumerate(chunks):
        a, b = int(rec["start"]), int(rec["end"])
        with np.load(src / "chunks" / rec["file"], allow_pickle=False) as z:
            if set(z.files) != set(keys):
                raise AssertionError("Protocol033 chunk field schema changed")
            for k in keys:
                expected = (b - a,) + tuple(schema[k]["tail_shape"])
                if z[k].shape != expected or str(z[k].dtype) != schema[k]["dtype"]:
                    raise AssertionError("Protocol033 chunk field shape/dtype mismatch: " + k)
                memmaps[k][a:b] = z[k]
        for mm in memmaps.values():
            mm.flush()
        sample(samples, "chunk_materialized", ci)

    # Build causal common features after the simulator object graph is gone.
    # At 5969 rows this is small relative to the simulator state, while still
    # keeping the source arrays memory-mapped instead of duplicating the stream.
    common = P._common_features(memmaps["host_features"], memmaps["capacities"])
    np.save(out / "common_observable_features.npy", common)
    del common
    sample(samples, "common_features_written")

    # Preserve the historical stream schema; overload_mask is small (~0.3 MB)
    # and can be created in this fresh process without duplicating simulator state.
    overload_mask = (memmaps["overload_ratio"] > 1.0).astype(np.uint8)
    stream = out / "stream.npz"
    np.savez_compressed(stream, **memmaps, overload_mask=overload_mask)
    del overload_mask
    stream_sha = P.sha(stream)
    sample(samples, "stream_npz_written")

    phases = P.phase_table(reg)
    manifest = {
        "protocol": "033", "plan_revision": P.PLAN_REVISION,
        "scenario_id": P.SCENARIO_ID, "data_revision": P.DATA_REVISION,
        "seed": int(reg["replay_seed"]),
        "steps": int(reg["scored_intervals"]), "guard_rows": int(reg["guard_intervals"]),
        "total_rows": int(reg["total_intervals"]),
        "stream_file": "stream.npz", "stream_sha256": stream_sha,
        "registration_sha256": reg_sha, "timeline": phases,
        "logical_to_source_service": P.SOURCE_SERVICE,
        "event_probability": float(reg["generation"]["event_probability"]),
        "forbidden_model_inputs": list(P.FORBIDDEN_MODEL_INPUTS),
        "audit_only_npz_keys": [k for k in keys if k.startswith("audit_")] + ["overload_mask"],
        "model_input_keys": ["host_features", "demands", "schedules", "capacities", "creation_ids", "before_placement"],
        "common_observable_features_file": "common_observable_features.npy",
        "common_observable_feature_order": P.COMMON_FEATURE_ORDER,
        "label_key": "raw_labels",
        "label_rule": reg["causality"]["label_rule"],
        "target": reg["causality"]["target"],
        "target_maturity": reg["causality"]["target_maturity"],
        "chunk_manifest": chunks,
        "field_schema": schema,
        "assembled_in_fresh_process": True,
        "audit_pass": None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    for name in ("events_final.json", "applied_switches.json", "memory_profile_generation.json", "generation_manifest.json", "resume_manifest.json", "registration_snapshot.json"):
        source = src / name
        if source.is_file():
            (out / name).write_bytes(source.read_bytes())

    vm = psutil.virtual_memory()
    effective = min(int(vm.total), int(P._cgroup_limit_bytes())) if P._cgroup_limit_bytes() else int(vm.total)
    report = {
        "protocol": "033", "stage": "assembly",
        "effective_memory_limit_bytes": effective,
        "soft_limit_bytes": int(effective * 0.8),
        "samples": samples,
        "peak_process_tree_rss_bytes": max(x["process_tree_rss_bytes"] for x in samples),
        "minimum_system_available_bytes": min(x["system_available_bytes"] for x in samples),
    }
    (out / "memory_profile_assembly.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"protocol": "033", "assembled": True, "rows": int(reg["total_intervals"]), "chunks": len(chunks), "stream_sha256": stream_sha}, indent=2))


if __name__ == "__main__":
    main()
