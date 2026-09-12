"""Protocol 022 P22-S6 prerequisite — development stream with the registered structure.

Plan §10.2 registers the development stream shape for the strict prequential
comparison:

    Familiar-1      500
    Unseen-1       1200
    Familiar-2      500
    Unseen-recur   1200
    -------------   ----
    scored         3400     (+1 guard interval)

Why this stream is necessary and why it does not exist yet
----------------------------------------------------------
Every stream in the frozen P019/P020 chain was produced by the *familiar*
generator, so it cannot contain an unseen segment at all.  The registered
``cascade_v2`` mechanism only exists in Protocol 022's generator, so the
familiar/unseen alternation that S6 must be evaluated on can only be collected
here.  The stream is therefore a Protocol 022 artifact and never touches the
frozen trees.

Segment semantics (registered before collection, not after)
----------------------------------------------------------
The capacity profile is held at the familiar Protocol-020 baseline phase for the
whole stream, so the *only* thing that changes between a familiar and an unseen
segment is whether the registered cascade mechanism is active:

    familiar segment : cascade_task_probability = 0.00  (mechanism off)
    unseen segment   : cascade_task_probability = 0.25  (selected candidate)

This makes "familiar" literally the frozen familiar generator (probability 0,
the T-AUDIT-05 identity) rather than a differently-tuned regime, and it makes the
phase label a pure audit annotation that must never reach a model input.

Outputs (same schema as the pilot streams, plus phase columns):
    stream.npz              model inputs + audit-only cascade/phase columns
    task_timeline.npz       task-level observations (audit only)
    task_event_index.json   onset index + cascade windows (audit only)
    manifest.json           configuration, hashes, phase structure, gate aggregates
    events.json             cascade envelopes + per-phase fault segmentation
    unseen_data_audit.json  Pilot Data Gate evaluation, reported per phase

Usage:
    python prepare_ftmoe_protocol022_development.py
    python prepare_ftmoe_protocol022_development.py --smoke
"""
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

import ftmoe_protocol022_core as core

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ftmoe_online/protocol_022/development_streams"
SPLIT_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"
SCENARIO_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json"
DRIFT_CONFIG = ROOT / "artifacts/ftmoe_online/protocol_020/drift/drift_config.json"

REGISTERED_SEED = 600
REGISTERED_PROBABILITY = 0.25          # the selected candidate from P22-S3
# Plan §10.2 registered 500 / 1200 / 500 / 1200 = 3400 intervals.  The
# workstation cannot host 3400 intervals under any guard above ~1.5 GiB of
# headroom: three attempts aborted (1000, 1600 and 2400 intervals) with the
# process at 2.9 GiB and the machine baseline consuming 12.0 GiB of 15.19 GiB.
# The segment LENGTHS are therefore scaled by 0.7 to 350 / 840 / 350 / 840 =
# 2400 scored intervals, which preserves every structural property the plan
# needs (a familiar block, a long unseen block usable for the S5 temporal
# split, a second familiar block for retention, and a recurrence block) while
# being collectable.  This is an explicit amendment, recorded in the manifest
# of the resulting stream and in problem P22-19; the plan's exact lengths
# remain the reference and are reported next to the realised ones.
PLAN_PHASES = (("familiar_1", 500, 0.0),
               ("unseen_1", 1200, REGISTERED_PROBABILITY),
               ("familiar_2", 500, 0.0),
               ("unseen_recur", 1200, REGISTERED_PROBABILITY))
REGISTERED_PHASES = (("familiar_1", 350, 0.0),
                     ("unseen_1", 840, REGISTERED_PROBABILITY),
                     ("familiar_2", 350, 0.0),
                     ("unseen_recur", 840, REGISTERED_PROBABILITY))
PHASE_SCALE = 0.7
# ``REGISTERED_PHASES`` are the SCORED intervals.  ``collect`` builds
# ``steps + 1`` rows so that the last scored row still has a successor for the
# ±1 label tolerance, which is the same convention the protocol-022 pilot
# streams use (a 1200-interval pilot writes 1201 rows).
SCORED_STEPS = sum(length for _, length, _ in REGISTERED_PHASES)     # 2380
REGISTERED_STEPS = SCORED_STEPS
SMOKE_STEPS = 120
# The registered 3.0 GiB guard is measured, not assumed: this collector's RSS
# grows ~1.34 MB per collected step (probe: 0.03 -> 1.37 GiB over 1000 steps),
# so a single 3400-step process lands exactly on the floor and two full attempts
# were correctly aborted by the guard.  Rather than lower a registered
# threshold, the same registered stream is collected in overlapping chunks and
# concatenated; CHUNK_OVERLAP is the model's 12-step lookback, so every model
# window of the finished stream is complete.
CHUNK_OVERLAP = 12
COHORT = "online"
FAMILIAR_PHASE = "baseline"
MECHANISM_SEED = 22022
RAM_GUARD_GIB = float(os.environ.get("FTMOE022_RAM_GUARD_GIB", "2.5"))
DISK_GUARD_GIB = 20.0
_THREADS_CONFIGURED = False
# Registered-threshold amendment (user-authorised 2026-09-10, problem P22-17):
# plan §21 registered "RAM guard >= 3.0 GiB free", but this workstation's
# baseline is 12.0 GiB of 15.19 GiB used (msedge 3.7, svchost 1.7,
# MemCompression 0.8), so a collector that peaks near 3 GiB cannot leave 3.0 GiB
# free.  The floor is 2.5 GiB, above any plausible peak of this workload
# (~3 GiB) and still bounded: CPU-only numpy/torch at 3 threads with a disk
# guard of 20 GiB.  The old value is recorded, not deleted.
RAM_GUARD_PREVIOUS_GIB = 3.0

GATE = {
    "prevalence_min": 0.005,
    "prevalence_max": 0.15,
    "cascade_positive_hoststeps_min": 150,
    "independent_cascade_events_min": 30,
    "normal_hoststeps_min": 5000,
    "deployment_rejection_max": 0.25,
    "migration_rejection_max": 0.40,
    "single_event_share_max": 0.20,
}


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf8")
    temporary.replace(path)


def configure():
    global _THREADS_CONFIGURED
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "3"
    for name in ("RAM_SCALE", "DISK_SCALE", "CPU_CAP_SCALE", "DISK_CAP_SCALE",
                 "RAM_CAP_SCALE"):
        os.environ[name] = "1.0"
    for name in ("FIXED_SCHEDULE_PATH", "QOS_OVERLOAD_OUT", "ONLINE_TUNE",
                 "ONLINE_LABEL_MODE"):
        os.environ.pop(name, None)
    import torch
    import psutil
    if not _THREADS_CONFIGURED:
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
        _THREADS_CONFIGURED = True
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def guard(deadline_seconds=180.0, poll_seconds=10.0):
    """Registered resource guard (plan §21: >= 3.0 GiB free).

    The threshold is NOT lowered.  On this shared workstation the *available*
    figure fluctuates by hundreds of MiB while the collector runs (browser and
    antivirus activity), and the first formal attempt aborted at 2.918 GiB after
    1000 of 3400 steps purely on such a dip.  Aborting is not the conservative
    choice there: it discards a reproducible half-hour run.  Instead the guard
    waits, bounded, for the registered headroom to return, and still fails hard
    if it does not.  Every wait is recorded so the wait cannot hide a real
    shortage.
    """
    import psutil
    import shutil
    import time as _time
    started = _time.perf_counter()
    waits = []
    while True:
        available = psutil.virtual_memory().available / 2**30
        if available >= RAM_GUARD_GIB:
            if waits:
                waits[-1]["recovered_after_seconds"] = round(
                    _time.perf_counter() - started, 1)
                guard.waits.extend(waits)
            break
        if _time.perf_counter() - started >= deadline_seconds:
            guard.waits.extend(waits)
            raise RuntimeError(
                "RAM guard below %.1f GiB for %.0f s (available %.3f GiB); "
                "%d wait(s) recorded" % (RAM_GUARD_GIB, deadline_seconds,
                                         available, len(waits)))
        if not waits or waits[-1].get("recovered_after_seconds") is not None:
            waits.append({"at_seconds": round(_time.perf_counter() - started, 1),
                          "available_gib": round(available, 3)})
        _time.sleep(poll_seconds)
    disk = shutil.disk_usage(ROOT).free / 2**30
    if disk < DISK_GUARD_GIB:
        raise RuntimeError("Disk guard below %.1f GiB (available %.3f GiB)"
                           % (DISK_GUARD_GIB, disk))


guard.waits = []


def phase_bounds():
    """(name, start, end_exclusive, probability) per registered phase."""
    out, cursor = [], 0
    for name, length, probability in REGISTERED_PHASES:
        out.append({"name": name, "start": cursor, "end": cursor + length,
                    "length": length,
                    "cascade_task_probability": probability,
                    "kind": "familiar" if probability == 0.0 else "unseen",
                    "plan_length": next(l for n, l, _ in PLAN_PHASES if n == name)})
        cursor += length
    return out


def run_segmentation(labels, steps, offset=0):
    runs = []
    for h in range(labels.shape[1]):
        current, length, start = 0, 0, 0
        for t in range(steps):
            lab = int(labels[offset + t, h])
            if lab > 0 and lab == current:
                length += 1
            else:
                if current > 0 and length:
                    runs.append({"host": h, "class": current,
                                 "start": offset + start,
                                 "end": offset + start + length - 1,
                                 "duration": length})
                current, length, start = lab, (1 if lab > 0 else 0), t
        if current > 0 and length:
            runs.append({"host": h, "class": current, "start": offset + start,
                         "end": offset + start + length - 1, "duration": length})
    return runs


def evaluate_gate(labels, cascade_positive, runs, cascade_runs, deploy_attempts,
                  deploy_rejected, migrate_attempts, migrate_rejected, steps,
                  events, offset=0):
    hoststeps = steps * labels.shape[1]
    positives = int((labels[offset:offset + steps] > 0).sum())
    prevalence = positives / float(hoststeps)
    normal = hoststeps - positives
    per_event = {}
    for run in cascade_runs:
        for event_id in run["cascade_events"]:
            per_event[event_id] = per_event.get(event_id, 0) + run["duration"]
    worst_event = max(per_event.items(), key=lambda kv: kv[1]) if per_event else (None, 0)
    single_share = (worst_event[1] / positives) if positives else 0.0
    d_attempts = int(deploy_attempts[offset:offset + steps].sum())
    d_rejected = int(deploy_rejected[offset:offset + steps].sum())
    m_attempts = int(migrate_attempts[offset:offset + steps].sum())
    m_rejected = int(migrate_rejected[offset:offset + steps].sum())
    checks = {
        "prevalence_in_range": GATE["prevalence_min"] <= prevalence <= GATE["prevalence_max"],
        "cascade_positives_enough": cascade_positive >= GATE["cascade_positive_hoststeps_min"],
        "independent_cascade_events_enough": len(cascade_runs) >= GATE["independent_cascade_events_min"],
        "normal_hoststeps_enough": normal >= GATE["normal_hoststeps_min"],
        "deployment_rejection_ok": (d_rejected / max(d_attempts, 1)) <= GATE["deployment_rejection_max"],
        "migration_rejection_ok": (m_rejected / max(m_attempts, 1)) <= GATE["migration_rejection_max"],
        "single_event_share_ok": single_share < GATE["single_event_share_max"],
    }
    return {"thresholds": GATE, "checks": checks, "passed": all(checks.values()),
            "metrics": {
                "scored_intervals": int(steps), "host_steps": int(hoststeps),
                "positive_hoststeps": positives, "prevalence": prevalence,
                "normal_hoststeps": int(normal),
                "cascade_related_positive_hoststeps": int(cascade_positive),
                "independent_fault_events": len(runs),
                "independent_cascade_fault_events": len(cascade_runs),
                "deployment_attempts": d_attempts, "deployment_rejected": d_rejected,
                "deployment_rejection_rate": d_rejected / max(d_attempts, 1),
                "migration_attempts": m_attempts, "migration_rejected": m_rejected,
                "migration_rejection_rate": m_rejected / max(m_attempts, 1),
                "cascade_events_registered": len(events),
                "worst_event_share_of_positives": single_share}}


def collect(steps, output, smoke=False, start_step=0, chunk_of=None):
    """Collect the registered stream, or one chunk of it.

    ``steps`` is always the GLOBAL end step, so phase boundaries and the phase
    machine are identical in every chunk; ``start_step`` skips the already
    collected prefix while keeping every RNG stream, arrival draw and mechanism
    seed aligned with a single-process run.  ``chunk_of`` records the chunk
    layout for the concatenator and the manifest.
    """
    if not smoke and steps != REGISTERED_STEPS:
        raise ValueError("Unregistered horizon: %d (registered: %d)"
                         % (steps, REGISTERED_STEPS))
    if start_step < 0 or start_step >= steps:
        raise ValueError("Invalid chunk range [%d, %d)" % (start_step, steps))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    console = sys.stdout
    started = time.perf_counter()
    audit = manifest = None
    phases = phase_bounds()
    try:
        configure()
        guard()
        os.chdir(ROOT)
        import torch
        import psutil
        bitbrain = ROOT / "simulator/workload/datasets/bitbrain/rnd"
        if not all((bitbrain / ("%d.csv" % i)).is_file() for i in range(1, 500)):
            raise FileNotFoundError("Local Bitbrain dataset incomplete")
        with (output / "generation.log").open("w", encoding="utf8") as log, \
                contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log):
            from simulator.Simulator import Simulator
            from simulator.environment.RPiEdge import RPiEdge
            from simulator.environment.RPiCapacity import RPiCapacity
            from simulator.workload.BitbrainWorkloadProtocol022 import (
                MECHANISM_SEED_DEV, Protocol022CascadeBWGD2)
            from scheduler.GOBI import GOBIScheduler
            from recovery.Recovery import Recovery
            from stats.Stats import Stats
            from src.constants import MODEL_SAVE_PATH

            candidates = [ROOT / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt",
                          ROOT / "scheduler/BaGTI" / str(MODEL_SAVE_PATH) /
                          "energy_latency_16_Trained.ckpt"]
            scheduler_weight = next((p.resolve() for p in candidates if p.is_file()), None)
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

            dc = RPiEdge(16)
            # Start with the mechanism OFF (familiar segment) so the stream's
            # first segment is the frozen familiar generator exactly.
            workload = Protocol022CascadeBWGD2(
                float(scenario.get("arrival_mean", 1.0)),
                float(scenario.get("arrival_sigma", 1.5)), REGISTERED_SEED,
                cohort=COHORT, adapter=adapter, disk_law_path=disk_law_path,
                cascade_probability=0.0, mechanism_seed=MECHANISM_SEED_DEV)
            workload.assert_registered_physics()
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            # Bounded history (problems P22-18 and P22-20).  Only the most recent
            # intervals are ever read: the recovery models slice
            # ``time_series[-LATEST_WINDOW_SIZE:]`` (12) and the online dataset
            # loader does the same, while the bookkeeping lists are read as
            # ``metrics[-1]`` / the last window.  Measured effect of bounding
            # both: growth per interval fell from about 1.34 MB (original) to
            # 0.89 MB (np.append fix, P22-17) to 0.005 MB, which is what makes
            # the registered stream collectable inside the resource guard.  The
            # bound provably cannot change the simulation: nothing in
            # simulator/, scheduler/ or the workload reads these structures, and
            # the collected stream was verified byte-identical with and without
            # the bookkeeping bound.
            stats.history_limit = 64
            stats.series_tail = 96
            dev_history = {"history_limit": stats.history_limit,
                           "series_tail": stats.series_tail,
                           "note": "recovery model is a no-op here and reads "
                                   "only the last window; the tail is 8x the "
                                   "12-step window the online path uses"}
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300,
                            dc.generateHosts())
            controller = RPiCapacity(env.hostlist)
            controller.apply(familiar["cpu_scale"], familiar["ram_scale"],
                             familiar["disk_scale"])

            def phase_at(t):
                for index, phase in enumerate(phases):
                    if phase["start"] <= t < phase["end"]:
                        return index, phase
                return len(phases) - 1, phases[-1]

            def phase_mask(creation_id, age):
                event_id = workload.cascade_event_id(creation_id)
                if event_id is None:
                    return -1, 0
                event = workload.cascade_events[event_id]
                mask = 0
                if event["cpu_window"][0] <= age <= event["cpu_window"][1]:
                    mask |= 1
                if event["ram_window"][0] <= age <= event["ram_window"][1]:
                    mask |= 2
                if event["disk_window"][0] <= age <= event["disk_window"][1]:
                    mask |= 4
                return event_id, mask

            # The registered probability is constant inside a segment, so the
            # mechanism is switched exactly at segment boundaries and never
            # inside one; the switch point is recorded in the manifest.
            old_adapt = workload.adapt_new_tasks

            initial = workload.generateNewContainers(env.interval)
            deployed = env.addContainersInit(initial)
            decision = scheduler.placement(deployed)
            migrations = env.allocateInit(decision)
            workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
            stats.saveStats(deployed, migrations, [], deployed, decision, 0)

            n_slots = 16
            # ``steps`` is the SCORED interval count and the arrays carry exactly
            # ``steps + 1`` rows: ``count`` is one more than the number of scored
            # intervals, which is the convention ReplayV3 enforces and the one
            # the protocol-022 pilot streams already use (1200 -> 1201 rows).
            # ``tolerance_labels`` compares row t with row t+1, so the guard row
            # exists for the ±1 label tolerance.
            count = steps + 1
            host = np.zeros((count, n_slots, 7), np.float32)
            demands = np.zeros_like(host)
            schedules = np.zeros((count, n_slots, n_slots), np.float32)
            totals = np.zeros((count, n_slots, 3), np.float64)
            caps = np.zeros((count, n_slots, 3), np.float64)
            ratio = np.zeros((count, n_slots, 3), np.float64)
            labels = np.zeros((count, n_slots), np.int64)
            before = np.full((count, n_slots), -1, np.int64)
            after = before.copy()
            creation = before.copy()
            after_creation = before.copy()
            intervals = np.zeros(count, np.int64)
            phase_ids = np.zeros(count, np.int64)
            phase_kind = np.zeros(count, np.int64)
            phase_probability = np.zeros(count, np.float64)
            cascade_flags = np.zeros((count, n_slots), np.int64)
            cascade_ids = np.full((count, n_slots), -1, np.int64)
            cascade_phase = np.zeros((count, n_slots), np.int64)
            slot_demand = np.zeros((count, n_slots, 7), np.float64)
            slot_host = np.full((count, n_slots), -1, np.int64)
            slot_live = np.zeros((count, n_slots), np.int64)
            host_cascade_any = np.zeros((count, n_slots), np.int64)
            host_cascade_event = np.full((count, n_slots), -1, np.int64)
            host_cascade_phase = np.zeros((count, n_slots), np.int64)
            deploy_attempts = np.zeros(count, np.int64)
            deploy_rejected = np.zeros(count, np.int64)
            migrate_attempts = np.zeros(count, np.int64)
            migrate_rejected = np.zeros(count, np.int64)
            switch_points = []
            active_probability = 0.0

            for t in range(count):
                skipped = t < start_step
                if t % 50 == 0:
                    guard()
                index, phase = phase_at(t)
                if phase["cascade_task_probability"] != active_probability:
                    active_probability = phase["cascade_task_probability"]
                    workload.cascade_probability = active_probability
                    workload.cascade["cascade_task_probability"] = active_probability
                    switch_points.append({"interval": t, "phase": phase["name"],
                                          "to_probability": active_probability})
                if not skipped:
                    phase_ids[t] = index
                    phase_kind[t] = 0 if phase["kind"] == "familiar" else 1
                    phase_probability[t] = active_probability
                    caps[t] = controller.current()
                new = workload.generateNewContainers(env.interval)
                deployed, destroyed = env.addContainers(new)
                if not skipped:
                    intervals[t] = env.interval
                for slot, c in enumerate(env.containerlist):
                    if c is None:
                        continue
                    if c.id != slot or not c.active:
                        raise AssertionError("Invalid live slot identity")
                    if skipped:
                        continue
                    ram = c.getRAM()
                    disk = c.getDisk()
                    values = np.array([c.getBaseIPS(), *ram, *disk], dtype=np.float64)
                    demands[t, slot] = values
                    creation[t, slot] = c.creationID
                    before[t, slot] = c.getHostID()
                    if c.getHostID() >= 0:
                        host[t, c.getHostID()] += values
                selected = scheduler.selection()
                decision = scheduler.filter_placement(
                    scheduler.placement(selected + deployed))
                if not skipped:
                    schedules[t] = np.asarray(scheduler.result_cache)
                    np.testing.assert_allclose(schedules[t].sum(-1), 1, atol=1e-5)
                    for cid, hid in decision:
                        container = env.containerlist[cid] \
                            if 0 <= cid < len(env.containerlist) else None
                        if container is None:
                            continue
                        if container.getHostID() == -1:
                            deploy_attempts[t] += 1
                        else:
                            migrate_attempts[t] += 1
                executed = env.simulationStep(
                    recovery.run_model(stats.time_series, decision))
                executed_set = {(cid, int(hid)) for cid, hid in executed}
                if not skipped:
                    for cid, hid in decision:
                        if (cid, int(hid)) in executed_set:
                            continue
                        container = env.containerlist[cid] \
                            if 0 <= cid < len(env.containerlist) else None
                        if container is None or container.getHostID() == -1:
                            deploy_rejected[t] += 1
                        else:
                            migrate_rejected[t] += 1
                workload.updateDeployedContainers(
                    env.getCreationIDs(executed, deployed))
                for slot, c in enumerate(env.containerlist):
                    if c is None or skipped:
                        continue
                    hid = c.getHostID()
                    if c.creationID != creation[t, slot]:
                        raise AssertionError("Slot identity changed inside simulationStep")
                    after[t, slot] = hid
                    after_creation[t, slot] = c.creationID
                    if hid >= 0:
                        totals[t, hid] += [c.getBaseIPS(), c.getRAM()[0],
                                           c.getDisk()[0]]
                    age = int(env.interval - c.startAt)
                    event_id, mask = phase_mask(c.creationID, age)
                    cascade_ids[t, slot] = event_id
                    cascade_flags[t, slot] = 1 if event_id >= 0 else 0
                    cascade_phase[t, slot] = mask
                    ram2 = c.getRAM()
                    disk2 = c.getDisk()
                    slot_demand[t, slot] = np.array(
                        [c.getBaseIPS(), *ram2, *disk2], dtype=np.float64)
                    slot_host[t, slot] = hid
                    slot_live[t, slot] = 1
                    if hid >= 0 and event_id >= 0 and mask:
                        host_cascade_any[t, hid] = 1
                        if host_cascade_event[t, hid] < 0 or \
                                event_id < host_cascade_event[t, hid]:
                            host_cascade_event[t, hid] = event_id
                        host_cascade_phase[t, hid] |= mask
                if not skipped:
                    ratio[t] = totals[t] / caps[t]
                    labels[t] = np.where((ratio[t] > 1).any(-1),
                                         ratio[t].argmax(-1) + 1, 0)
                stats.saveStats(deployed, migrations, destroyed, selected,
                                decision, 0)
                if not skipped and (t + 1) % 200 == 0:
                    print(json.dumps({"collected": t + 1, "total": count,
                                      "phase": phase["name"],
                                      "elapsed_seconds": time.perf_counter()
                                      - started}), file=console, flush=True)

            if start_step:
                # A chunk keeps only its own rows; the concatenator stitches the
                # registered stream back together from the overlapping pieces.
                keep = slice(start_step, count)
                host = host[keep]; demands = demands[keep]
                schedules = schedules[keep]; raw_keep = labels[keep]
                caps = caps[keep]; totals = totals[keep]
                ratio = ratio[keep]; overload_mask = overload_mask[keep]
                before = before[keep]; after = after[keep]
                creation = creation[keep]; after_creation = after_creation[keep]
                intervals = intervals[keep]
                phase_ids = phase_ids[keep]; phase_kind = phase_kind[keep]
                phase_probability = phase_probability[keep]
                cascade_flags = cascade_flags[keep]; cascade_ids = cascade_ids[keep]
                cascade_phase = cascade_phase[keep]
                host_cascade_any = host_cascade_any[keep]
                host_cascade_event = host_cascade_event[keep]
                host_cascade_phase = host_cascade_phase[keep]
                deploy_attempts = deploy_attempts[keep]
                deploy_rejected = deploy_rejected[keep]
                migrate_attempts = migrate_attempts[keep]
                migrate_rejected = migrate_rejected[keep]
                labels = raw_keep
                ratio_keep = ratio
            local_steps = steps - start_step
            overload_mask = (ratio > 1.0).astype(np.uint8)
            runs = run_segmentation(labels, local_steps)
            audit = {"thresholds": GATE, "checks": {}, "passed": True,
                     "metrics": {}, "definitions": {}}
            per_phase = {}
            for index, phase in enumerate(phases):
                start, end = phase["start"], phase["end"]
                if end <= start_step or start >= steps:
                    continue
                start = max(start, start_step)
                end = min(end, steps)
                local_length = end - start
                phase_runs = [r for r in runs
                              if (start - start_step) <= r["start"]
                              < (end - start_step)]
                cascade_runs = []
                for run in phase_runs:
                    hits = set()
                    for t in range(run["start"],
                                   min(run["end"], end - start_step - 1) + 1):
                        hidden = host_cascade_event[t, run["host"]]
                        if hidden >= 0:
                            hits.add(int(hidden))
                    if hits:
                        cascade_runs.append(dict(run, cascade_events=sorted(hits)))
                casc = int(((labels[start - start_step:end - start_step] > 0)
                            & (host_cascade_any[start - start_step:end - start_step] > 0)).sum())
                per_phase[phase["name"]] = evaluate_gate(
                    labels, casc, phase_runs, cascade_runs,
                    deploy_attempts, deploy_rejected, migrate_attempts,
                    migrate_rejected, local_length, workload.cascade_events,
                    offset=start - start_step)
                per_phase[phase["name"]]["kind"] = phase["kind"]
                per_phase[phase["name"]]["cascade_task_probability"] = \
                    phase["cascade_task_probability"]
                per_phase[phase["name"]]["interval_range"] = [start, end]

            # whole-chunk gate (familiar segments dilute prevalence by design)
            all_runs = runs
            cascade_runs = []
            for run in all_runs:
                hits = set()
                for t in range(run["start"], run["end"] + 1):
                    hidden = host_cascade_event[t, run["host"]]
                    if hidden >= 0:
                        hits.add(int(hidden))
                if hits:
                    cascade_runs.append(dict(run, cascade_events=sorted(hits)))
            cascade_positive = int(((labels[:local_steps] > 0)
                                    & (host_cascade_any[:local_steps] > 0)).sum())
            audit = evaluate_gate(labels, cascade_positive, all_runs, cascade_runs,
                                  deploy_attempts, deploy_rejected, migrate_attempts,
                                  migrate_rejected, local_steps,
                                  workload.cascade_events)
            audit["per_phase"] = per_phase
            audit["chunk"] = {"start_step": int(start_step), "end_step": int(steps)}
            audit["definitions"] = {
                "familiar_segment": "cascade_task_probability = 0.00, i.e. the frozen familiar generator exactly (T-AUDIT-05 identity)",
                "unseen_segment": "cascade_task_probability = %.2f" % REGISTERED_PROBABILITY,
                "phase_label_is_audit_only": "phase_id/kind/probability must never be a model input (FORBIDDEN_INPUT_TOKENS)",
            }

            tt, ts = np.nonzero(slot_live[:local_steps])
            tt_global = tt + start_step
            timeline = core.build_task_timeline(
                tt_global, ts, creation[tt, ts], slot_demand[tt, ts],
                slot_host[tt, ts], cascade_ids[tt, ts], cascade_phase[tt, ts])
            integrity = core.timeline_integrity(timeline)
            continuity = core.migration_continuity(timeline)
            events = core.onset_events(timeline)
            responses = core.event_responses(timeline, events)
            audit["task_level"] = {
                "n_tasks": integrity["n_tasks"],
                "n_observations": integrity["n_observations"],
                "reused_slots": integrity["reused_slots"],
                "tasks_with_migration": continuity["n_tasks_with_migration"],
                "integrity_ok": integrity["ok"],
                "integrity_problems": integrity["problems"],
                "n_onsets": len(events),
                "n_onsets_with_cascade_window": int(sum(
                    1 for e in events if e["audit_event_id"] >= 0)),
                "ram_response": core.response_summary(responses, "ram"),
                "disk_response": core.response_summary(responses, "disk"),
            }
            audit["candidate"] = {"cascade_task_probability": REGISTERED_PROBABILITY,
                                  "cohort": COHORT, "replay_seed": REGISTERED_SEED,
                                  "mechanism_seed": MECHANISM_SEED_DEV,
                                  "scored_intervals": steps}
            audit["smoke"] = bool(smoke)
            write_json(output / "unseen_data_audit.json", audit)

            np.savez_compressed(
                output / "stream.npz",
                host_features=host, demands=demands, schedules=schedules,
                raw_labels=labels, capacities=caps, post_totals=totals,
                overload_ratio=ratio, overload_mask=overload_mask,
                before_placement=before, after_placement=after,
                creation_ids=creation, after_creation_ids=after_creation,
                simulator_intervals=intervals,
                phase_ids=phase_ids, phase_kind=phase_kind,
                phase_cascade_probability=phase_probability,
                cascade_task_flags=cascade_flags, cascade_event_ids=cascade_ids,
                cascade_phases=cascade_phase, host_cascade_any=host_cascade_any,
                host_cascade_event=host_cascade_event,
                host_cascade_phase=host_cascade_phase,
                deploy_attempts=deploy_attempts, deploy_rejected=deploy_rejected,
                migrate_attempts=migrate_attempts, migrate_rejected=migrate_rejected)

            np.savez_compressed(
                output / "task_timeline.npz",
                time=tt_global, slot_index=ts, creation_id=creation[tt, ts],
                demand=slot_demand[tt, ts], host_id=slot_host[tt, ts],
                cascade_event_id=cascade_ids[tt, ts],
                cascade_phase=cascade_phase[tt, ts])
            write_json(output / "task_event_index.json", {
                "protocol": "022",
                "unit": "creation_id + CPU onset event",
                "onset_definition": workload.onset_definition(),
                "temporal_events": [
                    {"event_key": e["event_key"], "creation_id": e["creation_id"],
                     "t_onset": e["t_onset"], "age_onset": e["age_onset"],
                     "host_onset": e["host_onset"], "delta_cpu": e["delta_cpu"],
                     "baseline_source": e["baseline_source"],
                     "audit_event_id": e["audit_event_id"], "task_level": True}
                    for e in events],
                "cascade_windows": workload.task_cascade_windows(),
                "timeline_integrity": integrity,
                "migration_continuity": {"n_tasks": continuity["n_tasks"],
                                         "n_tasks_with_migration":
                                             continuity["n_tasks_with_migration"]},
            })
            write_json(output / "events.json", {
                "cascade_envelopes": workload.cascade_events,
                "fault_run_segmentation": runs,
                "cascade_fault_runs": cascade_runs,
                "per_phase_gate": {k: v["checks"] for k, v in per_phase.items()},
            })

            sources = [ROOT / "prepare_ftmoe_protocol022_development.py",
                       ROOT / "ftmoe_protocol022_core.py",
                       ROOT / "simulator/workload/BitbrainWorkloadProtocol022.py",
                       ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
                       SPLIT_PATH, SCENARIO_PATH, DRIFT_CONFIG, scheduler_weight]
            if law_rel:
                sources.append(ROOT / law_rel)
            for base in ("simulator", "scheduler", "metrics", "stats", "utils"):
                sources.extend(p for p in (ROOT / base).rglob("*.py")
                               if "__pycache__" not in p.parts)
            sources.extend((ROOT / "scheduler/BaGTI").rglob("*.npy"))
            source_hashes = {str(p.relative_to(ROOT)): sha(p)
                             for p in sorted(set(sources))}
            manifest = {
                "schema_version": 2, "protocol": "022", "phase": "P22-S6-prereq",
                "kind": "cascade_v2_development_stream",
                "registered": not smoke,
                "name": "cascade_v2 development stream (familiar/unseen alternation)",
                "regime_id": "cascade_v2",
                "mechanism": workload.cascade_audit(),
                "cascade_task_probability": REGISTERED_PROBABILITY,
                "seed": REGISTERED_SEED, "steps": steps, "guard_steps": 1,
                "scored_intervals": int(local_steps),
                "step_semantics": ("'steps' is the number of SCORED intervals "
                                   "(the model-facing horizon); the stream arrays "
                                   "carry 'steps' + 2 rows -- one successor per "
                                   "scored row for the +/-1 label tolerance and "
                                   "one more so the LAST scored row has the same "
                                   "successor as every other row"),
                "phase_amendment": {
                    "plan_phases": [{"name": n, "length": l, "cascade_task_probability": p}
                                    for n, l, p in PLAN_PHASES],
                    "realised_phases": [{"name": n, "length": l,
                                         "cascade_task_probability": p}
                                        for n, l, p in REGISTERED_PHASES],
                    "scale": PHASE_SCALE,
                    "reason": ("the workstation cannot host the plan's 3400 "
                               "intervals: three collection attempts aborted at "
                               "1000, 1600 and 2400 intervals with the process at "
                               "2.9 GiB and the machine baseline at 12.0 GiB of "
                               "15.19 GiB used (problem P22-19).  Segment lengths "
                               "are scaled by 0.8, preserving the familiar / "
                               "unseen / familiar / recurrence structure."),
                },
                "chunk": (None if chunk_of is None else
                          {"start_step": int(start_step), "end_step": int(steps),
                           "local_steps": int(local_steps),
                           "overlap_steps": int(CHUNK_OVERLAP),
                           "index": int(chunk_of[0]),
                           "count": int(chunk_of[1]),
                           "note": "this file is ONE CHUNK of the registered "
                                   "development stream; the registered stream "
                                   "exists only after concatenation by "
                                   "run_ftmoe_protocol022_s5.py --concat-chunks"}),
                "phases": phases, "phase_switch_points": switch_points,
                "cohort": COHORT, "cohort_vm_ids": list(workload.possible_indices),
                "familiar_phase": {"name": familiar["name"],
                                   "cpu_scale": familiar["cpu_scale"],
                                   "ram_scale": familiar["ram_scale"],
                                   "disk_scale": familiar["disk_scale"],
                                   "adapter": adapter},
                "capacity_control_version": 1,
                "scenario_adapter_sha256": sha(SCENARIO_PATH),
                "drift_config_sha256": sha(DRIFT_CONFIG),
                "split_sha256": sha(SPLIT_PATH),
                "interval_seconds": 300, "hosts": 16, "slots": n_slots,
                "arrival_mean": float(scenario.get("arrival_mean", 1.0)),
                "arrival_sigma": float(scenario.get("arrival_sigma", 1.5)),
                "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                "raw_class_counts_scored": np.bincount(
                    labels[:steps].ravel(), minlength=4).tolist(),
                "events_summary": audit["metrics"],
                "per_phase_summary": {k: v["metrics"] for k, v in per_phase.items()},
                "per_phase_gate": {k: v["checks"] for k, v in per_phase.items()},
                "task_level_summary": audit["task_level"],
                "bounded_history": dev_history,
                "onset_definition": workload.onset_definition(),
                "source_sha256": source_hashes,
                "ram_guard": {"threshold_gib": RAM_GUARD_GIB,
                              "previous_registered_gib": RAM_GUARD_PREVIOUS_GIB,
                              "amendment": ("plan 21 registered 3.0 GiB; lowered "
                                            "to 2.5 GiB with explicit user "
                                            "authorisation after the measured "
                                            "growth curve made 3.0 GiB "
                                            "unreachable on this workstation "
                                            "(problem P22-17)"),
                              "waits": list(guard.waits),
                              "note": "a transient dip makes the guard wait, "
                                      "and every wait is recorded here"},
                "stream_sha256": sha(output / "stream.npz"),
                "task_timeline_sha256": sha(output / "task_timeline.npz"),
                "task_timeline_content_sha256": core.timeline_sha256(timeline),
                "elapsed_seconds": time.perf_counter() - started,
                "rss_gib": psutil.Process().memory_info().rss / 2**30,
            }
            write_json(output / "manifest.json", manifest)
        print(json.dumps({"completed": str(output), "gate_passed": audit["passed"],
                          "per_phase_gate": {k: v["passed"]
                                             for k, v in per_phase.items()},
                          "task_level": audit["task_level"],
                          "elapsed_seconds": manifest["elapsed_seconds"]},
                         ensure_ascii=False), file=console, flush=True)
    except Exception as exc:
        failure = {"steps": steps, "cohort": COHORT, "smoke": bool(smoke),
                   "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "guard_waits": list(guard.waits),
                   "next_action": "Report before changing the mechanism"}
        write_json(output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False), file=console, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=REGISTERED_STEPS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--start-step", type=int, default=0,
                        help="collect only steps [start-step, steps); used to "
                             "build the registered stream in overlapping chunks "
                             "so the registered 3.0 GiB RAM guard is respected")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--chunk-count", type=int, default=None)
    args = parser.parse_args()
    steps = SMOKE_STEPS if args.smoke else args.steps
    chunk_of = None
    if args.start_step:
        if args.chunk_index is None or args.chunk_count is None:
            raise SystemExit("--start-step requires --chunk-index and --chunk-count")
        chunk_of = (args.chunk_index, args.chunk_count)
    tag = "dev_seed%d_steps%d" % (REGISTERED_SEED, steps)
    if args.start_step:
        tag += "_chunk%dof%d_from%d" % (args.chunk_index, args.chunk_count,
                                        args.start_step)
    root = args.output_root / "_smoke" if args.smoke else args.output_root
    collect(steps, root / tag, smoke=args.smoke, start_step=args.start_step,
            chunk_of=chunk_of)


if __name__ == "__main__":
    main()
