"""Collect the registered Protocol-024 next-round seed700 response-law stream.

The collector uses the real Bitbrain online cohort, the existing scheduler,
RPiEdge capacities and simulator execution path.  R1/R2/R3 alter task demand
trajectories only.  Labels are computed after simulator execution from physical
aggregate CPU/RAM/Disk demand divided by per-host capacity; no response-law ID
is consulted by the label function.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_ftmoe_protocol023_stream import (  # resource/process guards only
    configure, guard, assert_no_other_experiment, sha,
    SCENARIO_PATH, DRIFT_CONFIG, FAMILIAR_PHASE)
from simulator.workload.BitbrainWorkloadProtocol024 import (
    Protocol024ResponseLawBWGD2, RESPONSE_LAWS, LAW_IDS,
    LAW_MECHANISM_IDS, EVENT_PROBABILITY, FORBIDDEN_MODEL_INPUTS)


REGISTERED_SEED = 700
COHORT = "online"
SCORDED_STEPS = 4980
GUARD_ROWS = 1
TIMELINE = (
    ("F0", 300, None),
    ("R1_first", 1200, "R1"),
    ("R2_first", 1200, "R2"),
    ("R3_first", 1200, "R3"),
    ("R1_rec1", 180, "R1"),
    ("R3_rec1", 180, "R3"),
    ("R2_rec1", 180, "R2"),
    ("R1_rec2", 180, "R1"),
    ("R2_rec2", 180, "R2"),
    ("R3_rec2", 180, "R3"),
)


def phase_table():
    cursor, rows = 0, []
    for name, length, law in TIMELINE:
        rows.append({"name": name, "start": cursor, "end": cursor + length,
                     "length": length, "response_law": law,
                     "event_probability": 0.0 if law is None else EVENT_PROBABILITY,
                     "kind": "familiar" if law is None else "response_law"})
        cursor += length
    if cursor != SCORDED_STEPS:
        raise AssertionError("registered timeline does not sum to 4980")
    return rows


def phase_at(t, phases):
    for i, phase in enumerate(phases):
        if phase["start"] <= t < phase["end"]:
            return i, phase
    return len(phases) - 1, phases[-1]  # guard row inherits final phase


def _runs(labels, steps, start=0, end=None):
    end = steps if end is None else min(int(end), steps)
    start = max(0, int(start))
    out = []
    for h in range(labels.shape[1]):
        klass, run_start = 0, None
        for t in range(start, end):
            value = int(labels[t, h])
            if value == klass and value > 0:
                continue
            if klass > 0:
                out.append({"host": h, "class": klass, "start": run_start,
                            "end": t - 1, "length": t - run_start})
            klass = value
            run_start = t if value > 0 else None
        if klass > 0:
            out.append({"host": h, "class": klass, "start": run_start,
                        "end": end - 1, "length": end - run_start})
    return out


def _summary_block(labels, ratio, host_event_any, deploy_attempts,
                   deploy_rejected, migrate_attempts, migrate_rejected,
                   start, end):
    y = labels[start:end]
    r = ratio[start:end]
    event = host_event_any[start:end]
    hoststeps = int(y.size)
    positive = int((y > 0).sum())
    runs = _runs(labels, labels.shape[0] - 1, start, end)
    lengths = [x["length"] for x in runs]
    class_counts = {str(k): int((y == k).sum()) for k in range(4)}
    da = int(deploy_attempts[start:end].sum())
    dr = int(deploy_rejected[start:end].sum())
    ma = int(migrate_attempts[start:end].sum())
    mr = int(migrate_rejected[start:end].sum())
    return {
        "intervals": [int(start), int(end)],
        "host_steps": hoststeps,
        "positive_host_steps": positive,
        "positive_rate": positive / float(hoststeps) if hoststeps else None,
        "class_counts": class_counts,
        "fault_runs": len(runs),
        "run_length_mean": float(np.mean(lengths)) if lengths else None,
        "run_length_p95": float(np.percentile(lengths, 95)) if lengths else None,
        "run_length_max": int(max(lengths)) if lengths else None,
        "event_related_positive_host_steps": int(((y > 0) & (event > 0)).sum()),
        "peak_overload_ratio": float(r.max()) if r.size else None,
        "deployment_attempts": da,
        "deployment_rejected": dr,
        "deployment_rejection_rate": dr / float(max(da, 1)),
        "migration_attempts": ma,
        "migration_rejected": mr,
        "migration_rejection_rate": mr / float(max(ma, 1)),
    }


def collect(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite Protocol-024 data dir %s" % output)
    output.mkdir(parents=True)
    phases = phase_table()
    process_guard = None
    started = time.perf_counter()
    try:
        configure()
        guard()
        process_guard = assert_no_other_experiment()
        os.chdir(ROOT)
        import torch
        from simulator.Simulator import Simulator
        from simulator.environment.RPiEdge import RPiEdge
        from simulator.environment.RPiCapacity import RPiCapacity
        from scheduler.GOBI import GOBIScheduler
        from recovery.Recovery import Recovery
        from stats.Stats import Stats
        from src.constants import MODEL_SAVE_PATH

        bitbrain = ROOT / "simulator/workload/datasets/bitbrain/rnd"
        if not all((bitbrain / ("%d.csv" % i)).is_file() for i in range(1, 500)):
            raise FileNotFoundError("local Bitbrain rnd cohort is incomplete")
        scheduler_candidates = [
            ROOT / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt",
            ROOT / "scheduler/BaGTI" / str(MODEL_SAVE_PATH) /
            "energy_latency_16_Trained.ckpt"]
        scheduler_weight = next((p.resolve() for p in scheduler_candidates if p.is_file()), None)
        if scheduler_weight is None:
            raise FileNotFoundError("GOBI trained checkpoint missing")

        random.seed(REGISTERED_SEED)
        np.random.seed(REGISTERED_SEED)
        torch.manual_seed(REGISTERED_SEED)
        scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf8"))
        drift = json.loads(DRIFT_CONFIG.read_text(encoding="utf8"))
        familiar = next(p for p in drift["phases"] if p["name"] == FAMILIAR_PHASE)
        adapter = dict(scenario.get("adapter") or {})
        adapter.update(familiar.get("adapter") or {})
        law_rel = scenario.get("disk_law_relative")
        disk_law_path = (ROOT / law_rel).resolve() if law_rel else None

        with (output / "generation.log").open("w", encoding="utf8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            dc = RPiEdge(16)
            workload = Protocol024ResponseLawBWGD2(
                float(scenario.get("arrival_mean", 1.0)),
                float(scenario.get("arrival_sigma", 1.5)), REGISTERED_SEED,
                cohort=COHORT, adapter=adapter, disk_law_path=disk_law_path,
                active_law=None, event_probability=EVENT_PROBABILITY)
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            stats.history_limit = 64
            stats.series_tail = 96
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300,
                            dc.generateHosts())
            capacity = RPiCapacity(env.hostlist)
            capacity.apply(familiar["cpu_scale"], familiar["ram_scale"],
                           familiar["disk_scale"])

            # Initialise exactly as prior collectors do.
            initial = workload.generateNewContainers(env.interval)
            deployed = env.addContainersInit(initial)
            decision = scheduler.placement(deployed)
            migrations = env.allocateInit(decision)
            workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
            stats.saveStats(deployed, migrations, [], deployed, decision, 0)

            count, slots = SCORDED_STEPS + GUARD_ROWS, 16
            host = np.zeros((count, slots, 7), np.float32)
            demands = np.zeros_like(host)
            schedules = np.zeros((count, slots, slots), np.float32)
            totals = np.zeros((count, slots, 3), np.float64)
            caps = np.zeros((count, slots, 3), np.float64)
            ratio = np.zeros((count, slots, 3), np.float64)
            labels = np.zeros((count, slots), np.int64)
            before = np.full((count, slots), -1, np.int64)
            after = np.full((count, slots), -1, np.int64)
            creation = np.full((count, slots), -1, np.int64)
            intervals = np.zeros(count, np.int64)
            phase_ids = np.zeros(count, np.int64)
            phase_law = np.full(count, -1, np.int64)
            event_ids = np.full((count, slots), -1, np.int64)
            event_laws = np.full((count, slots), -1, np.int64)
            event_active = np.zeros((count, slots), np.uint8)
            host_event_any = np.zeros((count, slots), np.uint8)
            deploy_attempts = np.zeros(count, np.int64)
            deploy_rejected = np.zeros(count, np.int64)
            migrate_attempts = np.zeros(count, np.int64)
            migrate_rejected = np.zeros(count, np.int64)
            applied_switches = []
            active_law = None
            active_probability = 0.0

            for t in range(count):
                if t % 50 == 0:
                    guard()
                phase_index, phase = phase_at(t, phases)
                law_id = phase["response_law"]
                probability = phase["event_probability"]
                if law_id != active_law or probability != active_probability:
                    workload.set_active_law(law_id, probability=probability)
                    active_law, active_probability = law_id, probability
                    if t < SCORDED_STEPS:
                        applied_switches.append({"interval": int(t),
                                                 "phase": phase["name"],
                                                 "response_law": law_id,
                                                 "event_probability": probability})
                phase_ids[t] = phase_index
                phase_law[t] = -1 if law_id is None else LAW_MECHANISM_IDS[law_id]
                caps[t] = capacity.current()
                new = workload.generateNewContainers(env.interval)
                deployed, destroyed = env.addContainers(new)
                intervals[t] = env.interval

                for slot, container in enumerate(env.containerlist):
                    if container is None:
                        continue
                    if container.id != slot or not container.active:
                        raise AssertionError("invalid live slot identity")
                    ram = container.getRAM()
                    disk = container.getDisk()
                    values = np.asarray([container.getBaseIPS(), *ram, *disk],
                                        dtype=np.float64)
                    demands[t, slot] = values
                    creation[t, slot] = container.creationID
                    hid = container.getHostID()
                    before[t, slot] = hid
                    if hid >= 0:
                        host[t, hid] += values
                    eid = workload.response_event_id(container.creationID)
                    if eid is not None:
                        event = workload.response_events[eid]
                        event_ids[t, slot] = int(eid)
                        event_laws[t, slot] = int(event["mechanism_id"])
                        age = int(env.interval - container.startAt)
                        spec = RESPONSE_LAWS[event["law_id"]]
                        shapes = (spec["cpu_shape"], spec["ram_shape"], spec["disk_shape"])
                        active = any(0 <= age < len(shape) and float(shape[age]) > 0.0
                                     for shape in shapes)
                        event_active[t, slot] = 1 if active else 0
                        if hid >= 0 and active:
                            host_event_any[t, hid] = 1

                selected = scheduler.selection()
                decision = scheduler.filter_placement(
                    scheduler.placement(selected + deployed))
                schedules[t] = np.asarray(scheduler.result_cache)
                np.testing.assert_allclose(schedules[t].sum(-1), 1.0, atol=1e-5)
                for cid, hid in decision:
                    c = env.containerlist[cid] if 0 <= cid < len(env.containerlist) else None
                    if c is None:
                        continue
                    if c.getHostID() == -1:
                        deploy_attempts[t] += 1
                    else:
                        migrate_attempts[t] += 1

                executed = env.simulationStep(recovery.run_model(stats.time_series, decision))
                executed_set = {(cid, int(hid)) for cid, hid in executed}
                for cid, hid in decision:
                    if (cid, int(hid)) in executed_set:
                        continue
                    c = env.containerlist[cid] if 0 <= cid < len(env.containerlist) else None
                    if c is None or c.getHostID() == -1:
                        deploy_rejected[t] += 1
                    else:
                        migrate_rejected[t] += 1
                workload.updateDeployedContainers(env.getCreationIDs(executed, deployed))

                for slot, container in enumerate(env.containerlist):
                    if container is None:
                        continue
                    hid = container.getHostID()
                    after[t, slot] = hid
                    if hid >= 0:
                        totals[t, hid] += [container.getBaseIPS(),
                                           container.getRAM()[0],
                                           container.getDisk()[0]]
                ratio[t] = totals[t] / caps[t]
                # IMPORTANT: response law/audit arrays are not referenced here.
                labels[t] = np.where((ratio[t] > 1.0).any(-1),
                                     ratio[t].argmax(-1) + 1, 0)
                stats.saveStats(deployed, migrations, destroyed, selected, decision, 0)

        stream_path = output / "stream.npz"
        np.savez_compressed(
            stream_path,
            host_features=host, demands=demands, schedules=schedules,
            raw_labels=labels, capacities=caps, post_totals=totals,
            overload_ratio=ratio, overload_mask=(ratio > 1.0).astype(np.uint8),
            before_placement=before, after_placement=after,
            creation_ids=creation, intervals=intervals,
            audit_phase_ids=phase_ids, audit_response_law_ids=phase_law,
            audit_event_ids=event_ids, audit_event_law_ids=event_laws,
            audit_event_active=event_active, audit_host_event_any=host_event_any,
            audit_deploy_attempts=deploy_attempts,
            audit_deploy_rejected=deploy_rejected,
            audit_migrate_attempts=migrate_attempts,
            audit_migrate_rejected=migrate_rejected)
        digest = sha(stream_path)

        by_phase = {}
        for phase in phases:
            by_phase[phase["name"]] = dict(
                _summary_block(labels, ratio, host_event_any,
                               deploy_attempts, deploy_rejected,
                               migrate_attempts, migrate_rejected,
                               phase["start"], phase["end"]),
                response_law=phase["response_law"])
        by_law = {}
        for law in LAW_IDS:
            blocks = [p for p in phases if p["response_law"] == law]
            mask = np.zeros(SCORED_STEPS, dtype=bool)
            for p in blocks:
                mask[p["start"]:p["end"]] = True
            y = labels[:SCORED_STEPS][mask]
            by_law[law] = {
                "scored_intervals": int(mask.sum()),
                "positive_host_steps": int((y > 0).sum()),
                "positive_rate": float((y > 0).mean()) if y.size else None,
                "class_counts": {str(k): int((y == k).sum()) for k in range(4)},
                "task_events": int(sum(1 for e in workload.response_events
                                       if e["law_id"] == law)),
            }

        audit = {
            "protocol": "024",
            "kind": "response_law_v1_stream_audit",
            "labels_source": "post-simulator aggregate demand / physical host capacity",
            "direct_label_assignment_from_response_law": False,
            "whole": _summary_block(labels, ratio, host_event_any,
                                    deploy_attempts, deploy_rejected,
                                    migrate_attempts, migrate_rejected,
                                    0, SCORDED_STEPS),
            "per_phase": by_phase,
            "per_response_law": by_law,
            "task_events_total": len(workload.response_events),
            "short_trace_skips": dict(workload.short_trace_skips),
            "applied_switches": applied_switches,
        }
        (output / "audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8")
        (output / "events.json").write_text(
            json.dumps(workload.response_events, indent=2, allow_nan=False) + "\n",
            encoding="utf8")

        manifest = {
            "protocol": "024",
            "round": "next_round_v1",
            "family": "protocol024_response_law_v1",
            "seed": REGISTERED_SEED,
            "steps": SCORDED_STEPS,
            "guard_rows": GUARD_ROWS,
            "stream_file": "stream.npz",
            "stream_sha256": digest,
            "timeline": phases,
            "response_laws": RESPONSE_LAWS,
            "event_probability": EVENT_PROBABILITY,
            "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
            "audit_only_npz_keys": [
                "audit_phase_ids", "audit_response_law_ids", "audit_event_ids",
                "audit_event_law_ids", "audit_event_active", "audit_host_event_any",
                "audit_deploy_attempts", "audit_deploy_rejected",
                "audit_migrate_attempts", "audit_migrate_rejected"],
            "model_input_keys": ["host_features", "demands", "schedules",
                                 "capacities", "creation_ids", "before_placement"],
            "label_key": "raw_labels",
            "label_rule": "argmax(CPU,RAM,Disk aggregate/capacity) when any ratio>1 else 0",
            "admission_rule": "all response-law shapes are exactly zero at task age 0",
            "scheduler_checkpoint": str(scheduler_weight),
            "scheduler_checkpoint_sha256": sha(scheduler_weight),
            "scenario_adapter_sha256": sha(SCENARIO_PATH),
            "drift_config_sha256": sha(DRIFT_CONFIG),
            "generation_elapsed_seconds": time.perf_counter() - started,
            "audit_file": "audit.json",
            "events_file": "events.json",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
        print(json.dumps({"stream_sha256": digest, "audit": audit},
                         indent=2, allow_nan=False), flush=True)
        return manifest
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(),
                   "next_action": "fix the concrete generator/collector failure; do not reinterpret it as a model result"}
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf8")
        raise
    finally:
        # P23 process guard returns a psutil process only as an audit token; no
        # lock file is held, so there is nothing destructive to release here.
        _ = process_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.output)


if __name__ == "__main__":
    main()
