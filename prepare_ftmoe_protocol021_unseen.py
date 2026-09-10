"""Protocol 021 P21-S3 — data-only pilot collector for the cascade regime.

Collects ONE candidate stream on a source-disjoint VM cohort at the *familiar*
baseline capacity profile (cpu/ram/disk scales 1.0/1.0/0.9 and the baseline
demand adapter incl. ram_upper=1400 — identical to the familiar Protocol-020
baseline phase), with the registered Temporal Resource Cascade mechanism
switched on at a registered cascade probability.

No model is loaded: this is a data-only run (protocol §7 explicitly forbids
loading A/B/C/D here).  Candidate selection may only use data physics and
event structure, never model scores.

Registered candidates (§6):  cascade_task_probability in {0.15, 0.25, 0.35}
Registered horizon  (§7):   1200 scored intervals + 1 guard interval
Registered cohort/seed:     online cohort (disjoint from P20 S6 train and from
                            the dev cohort used by P20 S5/S7/R1), replay seed 600

Outputs per stream directory:
    stream.npz              model inputs + audit-only cascade columns
    manifest.json           configuration, hashes, gate-relevant aggregates
    events.json             cascade envelopes + fault-run segmentation
    unseen_data_audit.json  Pilot Data Gate evaluation (§7.1)

Usage:
    python prepare_ftmoe_protocol021_unseen.py --probability 0.25
    python prepare_ftmoe_protocol021_unseen.py --probability 0.25 --smoke
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ftmoe_online/protocol_021/pilot_streams"
SPLIT_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"
SCENARIO_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json"
DRIFT_CONFIG = ROOT / "artifacts/ftmoe_online/protocol_020/drift/drift_config.json"

REGISTERED_PROBABILITIES = (0.15, 0.25, 0.35)
REGISTERED_SEED = 600
REGISTERED_STEPS = 1200
SMOKE_STEPS = 40
COHORT = "online"                     # source-disjoint from S6 train and P20 dev
FAMILIAR_PHASE = "baseline"           # familiar capacity profile (drift config phase 0)
RAM_GUARD_GIB = float(os.environ.get("FTMOE021_RAM_GUARD_GIB", "3.0"))
DISK_GUARD_GIB = 20.0
_THREADS_CONFIGURED = False

# Pilot Data Gate thresholds (protocol §7.1, pre-registered)
GATE = {
    "prevalence_min": 0.01,
    "prevalence_max": 0.15,
    "cascade_positive_hoststeps_min": 150,
    "independent_cascade_events_min": 30,
    "normal_hoststeps_min": 5000,
    "deployment_rejection_max": 0.25,
    "migration_rejection_max": 0.40,
    "single_event_share_max": 0.20,
}


def sha(path):
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


def guard():
    import psutil
    import shutil
    available = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage(ROOT).free / 2**30
    if available < RAM_GUARD_GIB:
        raise RuntimeError("RAM guard below %.1f GiB (available %.3f GiB)"
                           % (RAM_GUARD_GIB, available))
    if disk < DISK_GUARD_GIB:
        raise RuntimeError("Disk guard below %.1f GiB (available %.3f GiB)"
                           % (DISK_GUARD_GIB, disk))


def run_segmentation(labels, steps):
    """Per-host runs of a constant non-zero label = one independent fault event.

    A sustained fault on one host counts once (protocol §11: 20 consecutive
    host fault steps are one event, not twenty independent feedbacks).
    """
    runs = []
    for h in range(labels.shape[1]):
        current, length, start = 0, 0, 0
        for t in range(steps):
            lab = int(labels[t, h])
            if lab > 0 and lab == current:
                length += 1
            else:
                if current > 0 and length:
                    runs.append({"host": h, "class": current, "start": start,
                                 "end": start + length - 1, "duration": length})
                current, length, start = lab, (1 if lab > 0 else 0), t
        if current > 0 and length:
            runs.append({"host": h, "class": current, "start": start,
                         "end": start + length - 1, "duration": length})
    return runs


def evaluate_gate(labels, ratio, cascade_positive, runs, cascade_runs,
                  deploy_attempts, deploy_rejected, migrate_attempts,
                  migrate_rejected, steps, events):
    hoststeps = steps * labels.shape[1]
    positives = int((labels[:steps] > 0).sum())
    prevalence = positives / float(hoststeps)
    normal = hoststeps - positives
    # attribution: positive host-steps attributable to each cascade event
    per_event = {}
    for run in cascade_runs:
        for event_id in run["cascade_events"]:
            per_event[event_id] = per_event.get(event_id, 0) + run["duration"]
    worst_event = max(per_event.items(), key=lambda kv: kv[1]) if per_event else (None, 0)
    single_share = (worst_event[1] / positives) if positives else 0.0
    d_attempts = int(deploy_attempts[:steps].sum())
    d_rejected = int(deploy_rejected[:steps].sum())
    m_attempts = int(migrate_attempts[:steps].sum())
    m_rejected = int(migrate_rejected[:steps].sum())
    checks = {
        "prevalence_in_range": GATE["prevalence_min"] <= prevalence <= GATE["prevalence_max"],
        "cascade_positives_enough": cascade_positive >= GATE["cascade_positive_hoststeps_min"],
        "independent_cascade_events_enough": len(cascade_runs) >= GATE["independent_cascade_events_min"],
        "normal_hoststeps_enough": normal >= GATE["normal_hoststeps_min"],
        "deployment_rejection_ok": (d_rejected / max(d_attempts, 1)) <= GATE["deployment_rejection_max"],
        "migration_rejection_ok": (m_rejected / max(m_attempts, 1)) <= GATE["migration_rejection_max"],
        "single_event_share_ok": single_share < GATE["single_event_share_max"],
    }
    return {
        "thresholds": GATE,
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "scored_intervals": int(steps),
            "host_steps": int(hoststeps),
            "positive_hoststeps": positives,
            "prevalence": prevalence,
            "normal_hoststeps": int(normal),
            "cascade_related_positive_hoststeps": int(cascade_positive),
            "independent_fault_events": len(runs),
            "independent_cascade_fault_events": len(cascade_runs),
            "deployment_attempts": d_attempts,
            "deployment_rejected": d_rejected,
            "deployment_rejection_rate": d_rejected / max(d_attempts, 1),
            "migration_attempts": m_attempts,
            "migration_rejected": m_rejected,
            "migration_rejection_rate": m_rejected / max(m_attempts, 1),
            "cascade_events_registered": len(events),
            "positive_hoststeps_per_cascade_event": per_event,
            "worst_event": worst_event[0],
            "worst_event_share_of_positives": single_share,
        },
        "definitions": {
            "cascade_related_positive_hoststep": "label>0 on a host that carries a live cascade task whose age is inside one of its registered cascade windows",
            "independent_fault_event": "maximal run of a constant non-zero label on one host (protocol §11)",
            "independent_cascade_fault_event": "independent fault event whose interval range overlaps a live cascade window on that host",
            "selection_rule": "candidates may only be compared on these data physics / event structure metrics, never on model scores",
        },
    }


def collect(probability, steps, output, smoke=False):
    if not smoke and probability not in REGISTERED_PROBABILITIES:
        raise ValueError("Unregistered cascade probability: %r" % probability)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    console = sys.stdout
    started = time.perf_counter()
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
            from simulator.workload.BitbrainWorkloadProtocol021 import \
                Protocol021CascadeBWGD2
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
            workload = Protocol021CascadeBWGD2(
                float(scenario.get("arrival_mean", 1.0)),
                float(scenario.get("arrival_sigma", 1.5)), REGISTERED_SEED,
                cohort=COHORT, adapter=adapter, disk_law_path=disk_law_path,
                cascade_probability=probability)
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300,
                            dc.generateHosts())
            controller = RPiCapacity(env.hostlist)
            controller.apply(familiar["cpu_scale"], familiar["ram_scale"],
                             familiar["disk_scale"])

            def phase_mask(creation_id, age):
                """Audit-only: which registered cascade windows cover this age.

                ``cascade_events`` is append-only and ``event_id`` is its index,
                so the lookup stays valid as new events are drawn during the
                collection loop.
                """
                event_id = workload.cascade_event_id(creation_id)
                if event_id is None:
                    return -1, 0
                event = workload.cascade_events[event_id]
                cpu_w, ram_w, disk_w = (event["cpu_window"], event["ram_window"],
                                        event["disk_window"])
                mask = 0
                if cpu_w[0] <= age <= cpu_w[1]:
                    mask |= 1
                if ram_w[0] <= age <= ram_w[1]:
                    mask |= 2
                if disk_w[0] <= age <= disk_w[1]:
                    mask |= 4
                return event_id, mask

            initial = workload.generateNewContainers(env.interval)
            deployed = env.addContainersInit(initial)
            decision = scheduler.placement(deployed)
            migrations = env.allocateInit(decision)
            workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
            stats.saveStats(deployed, migrations, [], deployed, decision, 0)

            count = steps + 1
            host = np.zeros((count, 16, 7), np.float32)
            demands = np.zeros_like(host)
            schedules = np.zeros((count, 16, 16), np.float32)
            totals = np.zeros((count, 16, 3), np.float64)
            caps = np.zeros((count, 16, 3), np.float64)
            ratio = np.zeros((count, 16, 3), np.float64)
            labels = np.zeros((count, 16), np.int64)
            before = np.full((count, 16), -1, np.int64)
            after = before.copy()
            creation = before.copy()
            after_creation = before.copy()
            intervals = np.zeros(count, np.int64)
            cascade_flags = np.zeros((count, 16), np.int64)
            cascade_ids = np.full((count, 16), -1, np.int64)
            host_cascade_any = np.zeros((count, 16), np.int64)
            host_cascade_event = np.full((count, 16), -1, np.int64)
            host_cascade_phase = np.zeros((count, 16), np.int64)
            deploy_attempts = np.zeros(count, np.int64)
            deploy_rejected = np.zeros(count, np.int64)
            migrate_attempts = np.zeros(count, np.int64)
            migrate_rejected = np.zeros(count, np.int64)

            for t in range(count):
                if t % 50 == 0:
                    guard()
                caps[t] = controller.current()
                new = workload.generateNewContainers(env.interval)
                deployed, destroyed = env.addContainers(new)
                intervals[t] = env.interval
                for slot, c in enumerate(env.containerlist):
                    if c is None:
                        continue
                    if c.id != slot or not c.active:
                        raise AssertionError("Invalid live slot identity")
                    ram = c.getRAM()
                    disk = c.getDisk()
                    values = np.array([c.getBaseIPS(), *ram, *disk],
                                      dtype=np.float64)
                    demands[t, slot] = values
                    creation[t, slot] = c.creationID
                    before[t, slot] = c.getHostID()
                    if c.getHostID() >= 0:
                        host[t, c.getHostID()] += values
                selected = scheduler.selection()
                decision = scheduler.filter_placement(
                    scheduler.placement(selected + deployed))
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
                    if c is None:
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
                    if hid >= 0 and event_id >= 0 and mask:
                        host_cascade_any[t, hid] = 1
                        if host_cascade_event[t, hid] < 0 or \
                                event_id < host_cascade_event[t, hid]:
                            host_cascade_event[t, hid] = event_id
                        host_cascade_phase[t, hid] |= mask
                ratio[t] = totals[t] / caps[t]
                labels[t] = np.where((ratio[t] > 1).any(-1),
                                     ratio[t].argmax(-1) + 1, 0)
                stats.saveStats(deployed, migrations, destroyed, selected,
                                decision, 0)
                if (t + 1) % 200 == 0:
                    print(json.dumps({"probability": probability,
                                      "collected": t + 1, "total": count,
                                      "elapsed_seconds": time.perf_counter()
                                      - started}), file=console, flush=True)

            overload_mask = (ratio > 1.0).astype(np.uint8)
            scored_counts = np.bincount(labels[:steps].ravel(),
                                        minlength=4).tolist()
            runs = run_segmentation(labels, steps)
            cascade_positive = int(((labels[:steps] > 0)
                                    & (host_cascade_any[:steps] > 0)).sum())
            cascade_runs = []
            for run in runs:
                hits = set()
                for t in range(run["start"], run["end"] + 1):
                    hidden = host_cascade_event[t, run["host"]]
                    if hidden >= 0:
                        hits.add(int(hidden))
                if hits:
                    cascade_runs.append(dict(run, cascade_events=sorted(hits)))
            audit = evaluate_gate(labels, ratio, cascade_positive, runs,
                                  cascade_runs, deploy_attempts,
                                  deploy_rejected, migrate_attempts,
                                  migrate_rejected, steps,
                                  workload.cascade_events)
            audit["candidate"] = {"cascade_task_probability": probability,
                                  "cohort": COHORT, "replay_seed": REGISTERED_SEED,
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
                cascade_task_flags=cascade_flags,
                cascade_event_ids=cascade_ids,
                host_cascade_any=host_cascade_any,
                host_cascade_event=host_cascade_event,
                host_cascade_phase=host_cascade_phase,
                deploy_attempts=deploy_attempts,
                deploy_rejected=deploy_rejected,
                migrate_attempts=migrate_attempts,
                migrate_rejected=migrate_rejected)

            audit_summary = {
                "gate": audit, "runs": runs, "cascade_runs": cascade_runs}
            write_json(output / "events.json", {
                "cascade_envelopes": workload.cascade_events,
                "fault_run_segmentation": runs,
                "cascade_fault_runs": cascade_runs,
                "gate": audit["checks"],
            })

            sources = [ROOT / "prepare_ftmoe_protocol021_unseen.py",
                       ROOT / "simulator/workload/BitbrainWorkloadProtocol021.py",
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
                "schema_version": 1, "protocol": "021", "phase": "P21-S3",
                "kind": "unseen_temporal_resource_cascade_pilot",
                "registered": not smoke,
                "name": "cascade_v1 data-only pilot (probability %.2f)" % probability,
                "regime_id": "cascade_v1",
                "mechanism": workload.cascade_audit(),
                "cascade_task_probability": probability,
                "seed": REGISTERED_SEED, "steps": steps, "guard_steps": 1,
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
                "interval_seconds": 300, "hosts": 16, "containers": 16,
                "arrival_mean": float(scenario.get("arrival_mean", 1.0)),
                "arrival_sigma": float(scenario.get("arrival_sigma", 1.5)),
                "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                "raw_class_counts_scored": scored_counts,
                "events_summary": audit_summary["gate"]["metrics"],
                "source_sha256": source_hashes,
                "stream_sha256": sha(output / "stream.npz"),
                "elapsed_seconds": time.perf_counter() - started,
                "rss_gib": psutil.Process().memory_info().rss / 2**30,
            }
            write_json(output / "manifest.json", manifest)
        print(json.dumps({"completed": str(output),
                          "gate_passed": audit["passed"],
                          "metrics": audit["metrics"],
                          "elapsed_seconds": manifest["elapsed_seconds"]},
                         ensure_ascii=False), file=console, flush=True)
    except Exception as exc:
        failure = {"probability": probability, "steps": steps, "cohort": COHORT,
                   "smoke": bool(smoke),
                   "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "next_action": "Report before changing the mechanism"}
        write_json(output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False), file=console, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--steps", type=int, default=REGISTERED_STEPS)
    parser.add_argument("--smoke", action="store_true",
                        help="engineering check only: unregistered horizon, "
                             "marked registered=false, written under _smoke/")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    steps = SMOKE_STEPS if args.smoke else args.steps
    if not args.smoke and steps != REGISTERED_STEPS:
        raise ValueError("Unregistered horizon: %d (registered: %d)"
                         % (steps, REGISTERED_STEPS))
    tag = "p%03d_seed%d_steps%d" % (round(args.probability * 100),
                                    REGISTERED_SEED, steps)
    root = args.output_root / "_smoke" if args.smoke else args.output_root
    collect(args.probability, steps, root / tag, smoke=args.smoke)


if __name__ == "__main__":
    main()
