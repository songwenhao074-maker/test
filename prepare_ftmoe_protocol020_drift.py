"""Protocol 020 S5 — fault-mode drift stream collector (capacity phases).

Collects a single 5-phase stream on one source-disjoint VM cohort:

    Phase 0  baseline      (registered from the S4 data-only scan)
    Phase 1  cpu_fault
    Phase 2  ram_fault
    Phase 3  disk_fault
    Phase 4  cpu_recurrence (same scales as Phase 1)

At every phase boundary the CapacityController rewrites the live host
capacity fields (effective-quota interpretation, plan §5.1/§14-A).  The
change is visible to the scheduler (Scheme B1) and recorded in the stream
(per-interval capacities + transition events with capacity before/after,
plan §42).  No model is involved.

Phase scales come from the registered drift config
(artifacts/ftmoe_online/protocol_020/drift/drift_config.json), whose values
are chosen strictly from the S4 data-only scan report (never from model
scores).  The per-phase dominance gate (§13) is evaluated by
analyze_ftmoe_protocol020_drift.py on this stream.

Usage:
    python prepare_ftmoe_protocol020_drift.py --seed 500 --cohort dev \
        [--steps 2000] [--output-root artifacts/ftmoe_online/protocol_020/drift_streams]
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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ftmoe_online/protocol_020/drift_streams"
CONFIG_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/drift/drift_config.json"
ALLOWED_SEEDS = {500, 501}
ALLOWED_STEPS = {2000}
RAM_GUARD_GIB = float(os.environ.get("FTMOE020_RAM_GUARD_GIB", "3.0"))
DISK_GUARD_GIB = 20.0
_THREADS_CONFIGURED = False  # torch allows set_num_interop_threads only once per process


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
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
    if not _THREADS_CONFIGURED:  # torch 只允许进程级设置一次（019 P07 同类）
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


def event_summary(labels, steps):
    scored = labels[:steps]
    durations = {r: [] for r in (1, 2, 3)}
    for h in range(16):
        run_class, run_len = 0, 0
        for t in range(steps):
            lab = int(scored[t, h])
            if lab > 0 and lab == run_class:
                run_len += 1
            else:
                if run_class > 0 and run_len:
                    durations[run_class].append(run_len)
                run_class, run_len = lab, (1 if lab > 0 else 0)
        if run_class > 0 and run_len:
            durations[run_class].append(run_len)
    out = {}
    for r, values in durations.items():
        out[str(r)] = {"count": len(values),
                       "mean_duration": float(sum(values) / len(values))
                       if values else 0.0}
    return out


def collect(seed, steps, cohort, output, config_path=CONFIG_PATH):
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Unregistered drift seed: {seed}")
    if steps not in ALLOWED_STEPS:
        raise ValueError(f"Unregistered drift horizon: {steps}")
    if cohort not in ("train", "dev", "online"):
        raise ValueError(f"Unknown cohort: {cohort}")
    if not Path(config_path).is_file():
        raise FileNotFoundError("Registered config missing: %s" % config_path)
    config = json.loads(Path(config_path).read_text(encoding="utf8"))
    kind = config.get("kind", "drift")
    if kind not in ("drift", "stationary"):
        raise ValueError("config kind must be drift|stationary")
    phase_len = int(config["phase_len"])
    phases = config["phases"]
    schedule_len = len(phases)
    if phase_len * schedule_len != steps:
        raise ValueError("config phase_len*len(phases) must equal steps")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    console = sys.stdout
    started = time.perf_counter()
    try:
        configure()
        guard()
        os.chdir(ROOT)
        import numpy as np
        import torch
        import psutil
        bitbrain = ROOT / "simulator/workload/datasets/bitbrain/rnd"
        if not all((bitbrain / f"{i}.csv").is_file() for i in range(1, 500)):
            raise FileNotFoundError("Local Bitbrain dataset incomplete")
        with (output / "generation.log").open("w", encoding="utf8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from simulator.Simulator import Simulator
            from simulator.environment.RPiEdge import RPiEdge
            from simulator.environment.RPiCapacity import RPiCapacity
            from simulator.workload.BitbrainWorkloadProtocol020 import \
                Protocol020AdaptedBWGD2
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
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            dc = RPiEdge(16)
            workload = Protocol020AdaptedBWGD2(1, 1.5, seed, cohort=cohort)
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300,
                            dc.generateHosts())
            controller = RPiCapacity(env.hostlist)
            controller.apply(phases[0]["cpu_scale"], phases[0]["ram_scale"],
                             phases[0]["disk_scale"])
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
            phase_ids = np.zeros(count, np.int64)
            deploy_attempts = np.zeros(count, np.int64)
            deploy_rejected = np.zeros(count, np.int64)
            migrate_attempts = np.zeros(count, np.int64)
            migrate_rejected = np.zeros(count, np.int64)
            transitions = []
            current_phase = 0
            current_scales = tuple(controller.current_scales)
            for t in range(count):
                if t % 50 == 0:
                    guard()
                phase = min(int(t) // phase_len, schedule_len - 1)
                if phase != current_phase:
                    transitions.append({
                        "interval": t, "from": phases[current_phase]["name"],
                        "to": phases[phase]["name"],
                        "capacity_before": [current_scales, caps[t - 1].tolist()],
                    })
                    current_phase = phase
                    scales = (phases[phase]["cpu_scale"], phases[phase]["ram_scale"],
                              phases[phase]["disk_scale"])
                    controller.apply(*scales)
                    current_scales = tuple(controller.current_scales)
                    transitions[-1]["capacity_after"] = [list(current_scales),
                                                         controller.current().tolist()]
                phase_ids[t] = phase
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
                ratio[t] = totals[t] / caps[t]
                labels[t] = np.where((ratio[t] > 1).any(-1),
                                     ratio[t].argmax(-1) + 1, 0)
                stats.saveStats(deployed, migrations, destroyed, selected,
                                decision, 0)
                if (t + 1) % 250 == 0:
                    print(json.dumps({"seed": seed, "cohort": cohort,
                                      "collected": t + 1, "total": count,
                                      "elapsed_seconds": time.perf_counter()
                                      - started}), file=console, flush=True)
            overload_mask = (ratio > 1.0).astype(np.uint8)
            scored_counts = np.bincount(labels[:steps].ravel(),
                                        minlength=4).tolist()
            events = event_summary(labels, steps)
            # per-phase scored class counts (phase p covers
            # [p*phase_len, (p+1)*phase_len) of the scored horizon)
            per_phase = []
            for p, ph in enumerate(phases):
                seg = labels[p * phase_len:(p + 1) * phase_len]
                counts = np.bincount(seg.ravel(), minlength=4).tolist()
                anomalous = int(seg[seg > 0].size)
                per_phase.append({
                    "phase": ph["name"], "scales": {k: ph[k] for k in
                                                    ("cpu_scale", "ram_scale",
                                                     "disk_scale")},
                    "raw_class_counts": counts,
                    "anomalous_hoststeps": anomalous,
                    "target_share_of_anomalous":
                        float(counts[[0] + [1, 2, 3][["cpu_fault", "ram_fault",
                                                     "disk_fault"].index(ph["name"])]
                              if ph["name"] in ("cpu_fault", "ram_fault",
                                                "disk_fault") else 0] /
                              max(anomalous, 1)) if anomalous else 0.0,
                })
            np.savez_compressed(output / "stream.npz",
                                host_features=host, demands=demands,
                                schedules=schedules, raw_labels=labels,
                                capacities=caps, post_totals=totals,
                                overload_ratio=ratio, overload_mask=overload_mask,
                                before_placement=before,
                                after_placement=after, creation_ids=creation,
                                after_creation_ids=after_creation,
                                simulator_intervals=intervals,
                                phase_ids=phase_ids,
                                deploy_attempts=deploy_attempts,
                                deploy_rejected=deploy_rejected,
                                migrate_attempts=migrate_attempts,
                                migrate_rejected=migrate_rejected)
            sources = [ROOT / "prepare_ftmoe_protocol020_drift.py",
                       ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
                       ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json",
                       Path(config_path), scheduler_weight]
            for base in ("simulator", "scheduler", "metrics", "stats", "utils"):
                sources.extend(p for p in (ROOT / base).rglob("*.py")
                               if "__pycache__" not in p.parts)
            sources.extend((ROOT / "scheduler/BaGTI").rglob("*.npy"))
            source_hashes = {str(p.relative_to(ROOT)): sha(p)
                             for p in sorted(set(sources))}
            manifest = {"schema_version": 1, "protocol": "020", "phase": "S5",
                        "name": ("capacity-driven fault-mode drift stream" if kind == "drift"
                                 else "stationary same-domain stream"),
                        "kind": kind,
                        "seed": seed, "steps": steps, "guard_steps": 1,
                        "cohort": cohort, "cohort_vm_ids": list(workload.possible_indices),
                        "phase_len": phase_len,
                        "phases": [{"name": p["name"],
                                    "cpu_scale": p["cpu_scale"],
                                    "ram_scale": p["ram_scale"],
                                    "disk_scale": p["disk_scale"]}
                                   for p in phases],
                        "capacity_control_version": 1,
                        "interval_seconds": 300, "hosts": 16, "containers": 16,
                        "arrival_mean": 1, "arrival_sigma": 1.5,
                        "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                        "config_sha256": sha(config_path),
                        "drift_config_sha256": sha(config_path),
                        "raw_class_counts_scored": scored_counts,
                        "per_phase": per_phase,
                        "events": events,
                        "transitions": transitions,
                        "deployment_attempts": int(deploy_attempts.sum()),
                        "deployment_rejected": int(deploy_rejected.sum()),
                        "deployment_rejection_rate":
                            float(deploy_rejected.sum() / max(deploy_attempts.sum(), 1)),
                        "migration_attempts": int(migrate_attempts.sum()),
                        "migration_rejected": int(migrate_rejected.sum()),
                        "migration_rejection_rate":
                            float(migrate_rejected.sum() / max(migrate_attempts.sum(), 1)),
                        "source_sha256": source_hashes,
                        "stream_sha256": sha(output / "stream.npz"),
                        "elapsed_seconds": time.perf_counter() - started,
                        "rss_gib": psutil.Process().memory_info().rss / 2**30}
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"completed": str(output),
                          "stream_sha256": manifest["stream_sha256"],
                          "per_phase": per_phase,
                          "elapsed_seconds": manifest["elapsed_seconds"]}),
              file=console, flush=True)
    except Exception as exc:
        failure = {"seed": seed, "steps": steps, "cohort": cohort,
                   "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "next_action": "Report before any scenario change"}
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf8")
        print(json.dumps(failure), file=console, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--cohort", default="dev")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_root / f"{args.cohort}_seed{args.seed}_steps{args.steps}"
    collect(args.seed, args.steps, args.cohort, output, args.config)


if __name__ == "__main__":
    main()
