"""Protocol 020 S4 — data-only capacity scan collector (no model is loaded).

Runs one single-factor capacity axis (cpu / ram / disk) over its
pre-registered candidate grid (plan §8.4 / §46) on the source-disjoint
*training* VM cohort and saves, per candidate:

- stream.npz: host_features, demands, schedules, creation_ids,
  before/after_placement, after_creation_ids, per-interval capacities
  [T+1,16,3], post_totals, overload_ratio [T+1,16,3], overload_mask
  [T+1,16,3] (plan §10), raw dominant labels, per-interval deployment /
  migration attempt & rejection counts, simulator intervals;
- manifest.json: profile, cohort, sources + hashes, stream hash,
  scored class counts, rejection rates, event summary.

Candidate grids (registered 2026-09-07 before results were inspected;
Extension 1 added 2026-09-07 after the original grid's measured shortfall,
see problem log P15 — both selections remain data-only):
    cpu:  RAM=1.00 Disk=0.30  CPU in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                     0.70, 0.75, 0.80, 0.90, 1.00]
    ram:  CPU=1.00 Disk=0.30  RAM in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                                     0.40, 0.45, 0.50, 0.60, 0.75, 1.00]
    disk: CPU=1.00 RAM =1.00  DISK in [0.12, 0.15, 0.17, 0.1875, 0.20,
                                       0.22, 0.25, 0.30]

The cpu/ram corner candidate (1.00,1.00,0.30) is identical in both families;
it is collected once under the cpu axis and referenced by the analyzer.

Usage:
    python prepare_ftmoe_protocol020_capacity_scan.py --axis cpu [--steps 400]
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ftmoe_online/protocol_020/capacity_scan"
ALLOWED_STEPS = {400, 60}  # 60 only via --smoke (validation, excluded from gates)
ALLOWED_SEEDS = {410}
COHORT = "train"

GRIDS = {
    # Extension 1 (registered 2026-09-07, problem log P15): the original
    # plan-§46 grid produced only 0-25 CPU-dominant host-steps per 6400 on the
    # P20 train cohort (phase projection <=31/8000 vs the 100 floor).  Deeper
    # scales are data-only probes; the final capacity choice still follows the
    # pre-registered §11/§13 gates.  Already-completed candidates are skipped
    # via their manifest.
    "cpu": {"cpu": [0.70, 0.75, 0.80, 0.90, 1.00, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
            "fixed": {"ram": 1.00, "disk": 0.30}},
    "ram": {"ram": [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00,
                    0.25, 0.20, 0.15, 0.10],
            "fixed": {"cpu": 1.00, "disk": 0.30}},
    "disk": {"disk": [0.17, 0.1875, 0.20, 0.22, 0.25, 0.30, 0.15, 0.12],
             "fixed": {"cpu": 1.00, "ram": 1.00}},
}

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
    """Host-level run-length events of identical dominant fault classes."""
    scored = labels[:steps]  # [T,16]
    events, durations = {}, {}
    for r in (1, 2, 3):
        durations[r] = []
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
    for r, values in durations.items():
        events[r] = {"count": len(values),
                     "mean_duration": float(sum(values) / len(values))
                     if values else 0.0,
                     "p95_duration": float(sorted(values)[
                         min(len(values) - 1, int(0.95 * len(values)))])
                     if values else 0.0}
    return events


def collect(axis, value, steps, seed, output):
    if steps not in ALLOWED_STEPS or seed not in ALLOWED_SEEDS:
        raise ValueError("Unregistered scan steps/seed: %d/%d" % (steps, seed))
    profile = dict(GRIDS[axis]["fixed"])
    profile[axis] = float(value)
    smoke = steps == 60
    if output.exists():
        if (output / "failure.json").is_file():
            shutil.rmtree(output)  # stale failed attempt: clean retry
        else:
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
            workload = Protocol020AdaptedBWGD2(1, 1.5, seed, cohort=COHORT)
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300,
                            dc.generateHosts())
            controller = RPiCapacity(env.hostlist)
            controller.apply(profile["cpu"], profile["ram"], profile["disk"])
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
                    if container is None:
                        # released slot == rejected initial deployment
                        deploy_rejected[t] += 1
                    elif container.getHostID() == -1:
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
                if (t + 1) % 100 == 0:
                    print(json.dumps({"candidate": output.name,
                                      "collected": t + 1, "total": count,
                                      "elapsed_seconds": time.perf_counter()
                                      - started}), file=console, flush=True)
            overload_mask = (ratio > 1.0).astype(np.uint8)
            scored_counts = np.bincount(labels[:steps].ravel(),
                                        minlength=4).tolist()
            events = event_summary(labels, steps)
            attempted_deploy = int(deploy_attempts.sum())
            attempted_migrate = int(migrate_attempts.sum())
            deployment_rejection_rate = float(deploy_rejected.sum() /
                                              max(attempted_deploy, 1))
            migration_rejection_rate = float(migrate_rejected.sum() /
                                             max(attempted_migrate, 1))
            np.savez_compressed(output / "stream.npz",
                                host_features=host, demands=demands,
                                schedules=schedules, raw_labels=labels,
                                capacities=caps, post_totals=totals,
                                overload_ratio=ratio, overload_mask=overload_mask,
                                before_placement=before,
                                after_placement=after, creation_ids=creation,
                                after_creation_ids=after_creation,
                                simulator_intervals=intervals,
                                deploy_attempts=deploy_attempts,
                                deploy_rejected=deploy_rejected,
                                migrate_attempts=migrate_attempts,
                                migrate_rejected=migrate_rejected)
            sources = [ROOT / "prepare_ftmoe_protocol020_capacity_scan.py",
                       ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
                       ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json",
                       scheduler_weight]
            for base in ("simulator", "scheduler", "metrics", "stats", "utils"):
                sources.extend(p for p in (ROOT / base).rglob("*.py")
                               if "__pycache__" not in p.parts)
            sources.extend((ROOT / "scheduler/BaGTI").rglob("*.npy"))
            source_hashes = {str(p.relative_to(ROOT)): sha(p)
                             for p in sorted(set(sources))}
            manifest = {"schema_version": 1, "protocol": "020", "phase": "S4",
                        "name": "data-only capacity scan candidate",
                        "seed": seed, "steps": steps, "guard_steps": 1,
                        "smoke": smoke,
                        "workload": "protocol020_adapted_BWGD2",
                        "cohort": COHORT,
                        "cohort_vm_ids": list(workload.possible_indices),
                        "profile": profile, "axis": axis,
                        "capacity_control_version": 1,
                        "controller": "RPiCapacity(apply scales to live hosts)",
                        "interval_seconds": 300, "hosts": 16, "containers": 16,
                        "arrival_mean": 1, "arrival_sigma": 1.5,
                        "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                        "capacities_row0": caps[0].tolist(),
                        "selected_vm_indices": workload.possible_indices,
                        "raw_class_counts_scored": scored_counts,
                        "events": events,
                        "deployment_attempts": attempted_deploy,
                        "deployment_rejected": int(deploy_rejected.sum()),
                        "deployment_rejection_rate": deployment_rejection_rate,
                        "migration_attempts": attempted_migrate,
                        "migration_rejected": int(migrate_rejected.sum()),
                        "migration_rejection_rate": migration_rejection_rate,
                        "source_sha256": source_hashes,
                        "stream_sha256": sha(output / "stream.npz"),
                        "elapsed_seconds": time.perf_counter() - started,
                        "rss_gib": psutil.Process().memory_info().rss / 2**30}
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"candidate": output.name, "profile": profile,
                          "stream_sha256": manifest["stream_sha256"],
                          "class_counts": scored_counts,
                          "events": events,
                          "deployment_rejection_rate": deployment_rejection_rate,
                          "migration_rejection_rate": migration_rejection_rate,
                          "elapsed_seconds": manifest["elapsed_seconds"]}),
              file=console, flush=True)
    except Exception as exc:
        failure = {"candidate": output.name, "profile": profile,
                   "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "next_action": "Report before any scenario change"}
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf8")
        print(json.dumps(failure), file=console, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=sorted(GRIDS), required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--smoke", action="store_true",
                        help="validation run with 60 steps (excluded from gates)")
    parser.add_argument("--value", type=float, default=None,
                        help="single candidate value (only with --smoke)")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    steps = 60 if args.smoke else args.steps
    for value in GRIDS[args.axis][args.axis]:
        if args.smoke:
            if args.value is None or abs(value - args.value) > 1e-9:
                continue
        label = "%s_%.4g" % (args.axis, value)
        if args.axis == "disk" and abs(value - 0.1875) < 1e-9:
            label = "disk_0.1875"
        output = args.output_root / label / f"seed{args.seed}_steps{steps}"
        if (output / "manifest.json").exists():
            print(json.dumps({"skip": str(output)}), flush=True)
            continue
        collect(args.axis, value, steps, args.seed, output)


if __name__ == "__main__":
    main()
