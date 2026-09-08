"""Protocol 020 S6 — same-domain adaptation episode collector.

Collects fixed-profile episodes (400 scored intervals + guard) from the
source-disjoint training cohort (train episodes) and dev cohort (validation
episodes).  Capacity profiles (baseline / cpu_fault / ram_fault / disk_fault)
are read from the registered adaptation-profiles JSON, whose scales are
chosen exclusively from the S4 data-only scan report.

Episode directories reuse the full S4 stream schema (per-interval
capacities, overload_ratio/mask, before/after placement, rejection counts),
so the S6 trainer can consume them directly with graph semantics v3 and
per-sample capacities.

Registered layout:
    artifacts/ftmoe_online/protocol_020/adaptation_data/raw/
        train/<profile>/seed<seed>_steps400/
        dev/  <profile>/seed<seed>_steps400/

Usage:
    python prepare_ftmoe_protocol020_adaptation_episodes.py --split train \
        [--profile-key baseline] [--seeds 401,402,403]
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
OUT = ROOT / "artifacts/ftmoe_online/protocol_020/adaptation_data/raw"
PROFILES_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/adaptation/adaptation_profiles.json"
SCENARIO_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json"
SPLITS = {"train": "train", "dev": "dev"}
ALLOWED_STEPS = {400}
ALLOWED_TRAIN_SEEDS = {401, 402, 403}
ALLOWED_DEV_SEEDS = {404, 405}
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
    return {str(r): {"count": len(values)} for r, values in durations.items()}


def collect_episode(split, profile_key, profile, seed, steps, output):
    if steps not in ALLOWED_STEPS:
        raise ValueError("Unregistered episode steps: %d" % steps)
    if output.exists():
        if (output / "failure.json").is_file():
            import shutil
            shutil.rmtree(output)
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
            scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf8"))
            arrival_mean = float(scenario.get("arrival_mean", 1.0))
            law_rel = scenario.get("disk_law_relative")
            disk_law_path = (ROOT / law_rel).resolve() if law_rel else None
            workload = Protocol020AdaptedBWGD2(arrival_mean, 1.5, seed,
                                               cohort=SPLITS[split],
                                               adapter=scenario.get("adapter"),
                                               disk_law_path=disk_law_path)
            scenario_effective = {"arrival_mean": arrival_mean,
                                  "adapter": dict(workload.adapter),
                                  "disk_law_relative": law_rel}
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300,
                            dc.generateHosts())
            controller = RPiCapacity(env.hostlist)
            controller.apply(profile["cpu_scale"], profile["ram_scale"],
                             profile["disk_scale"])
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
                if (t + 1) % 100 == 0:
                    print(json.dumps({"episode": output.name,
                                      "collected": t + 1, "total": count,
                                      "elapsed_seconds": time.perf_counter()
                                      - started}), file=console, flush=True)
            np.savez_compressed(output / "stream.npz",
                                host_features=host, demands=demands,
                                schedules=schedules, raw_labels=labels,
                                capacities=caps, post_totals=totals,
                                overload_ratio=ratio,
                                overload_mask=(ratio > 1.0).astype(np.uint8),
                                before_placement=before,
                                after_placement=after, creation_ids=creation,
                                after_creation_ids=after_creation,
                                simulator_intervals=intervals,
                                deploy_attempts=deploy_attempts,
                                deploy_rejected=deploy_rejected,
                                migrate_attempts=migrate_attempts,
                                migrate_rejected=migrate_rejected)
            scored_counts = np.bincount(labels[:steps].ravel(),
                                        minlength=4).tolist()
            sources = [ROOT / "prepare_ftmoe_protocol020_adaptation_episodes.py",
                       ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
                       ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json",
                       PROFILES_PATH, SCENARIO_PATH, scheduler_weight]
            if law_rel:
                sources.append(ROOT / law_rel)
            for base in ("simulator", "scheduler", "metrics", "stats", "utils"):
                sources.extend(p for p in (ROOT / base).rglob("*.py")
                               if "__pycache__" not in p.parts)
            sources.extend((ROOT / "scheduler/BaGTI").rglob("*.npy"))
            source_hashes = {str(p.relative_to(ROOT)): sha(p)
                             for p in sorted(set(sources))}
            manifest = {"schema_version": 1, "protocol": "020", "phase": "S6",
                        "name": "same-domain adaptation episode",
                        "split": split, "cohort": SPLITS[split],
                        "profile_key": profile_key, "profile": profile,
                        "seed": seed, "steps": steps, "guard_steps": 1,
                        "capacity_control_version": 1,
                        "scenario_adapter": scenario_effective,
                        "scenario_adapter_sha256": sha(SCENARIO_PATH),
                        "interval_seconds": 300, "hosts": 16, "containers": 16,
                        "arrival_mean": arrival_mean, "arrival_sigma": 1.5,
                        "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                        "cohort_vm_ids": list(workload.possible_indices),
                        "raw_class_counts_scored": scored_counts,
                        "events": event_summary(labels, steps),
                        "source_sha256": source_hashes,
                        "stream_sha256": sha(output / "stream.npz"),
                        "elapsed_seconds": time.perf_counter() - started,
                        "rss_gib": psutil.Process().memory_info().rss / 2**30}
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"completed": output.name,
                          "stream_sha256": manifest["stream_sha256"],
                          "class_counts": scored_counts,
                          "elapsed_seconds": manifest["elapsed_seconds"]}),
              file=console, flush=True)
    except Exception as exc:
        failure = {"split": split, "profile_key": profile_key, "seed": seed,
                   "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "next_action": "Report before any scenario change"}
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf8")
        print(json.dumps(failure), file=console, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--profile-key", default="all",
                        help="single profile key or 'all'")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if not PROFILES_PATH.is_file():
        raise FileNotFoundError("Registered adaptation profiles missing: %s"
                                % PROFILES_PATH)
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf8"))
    allowed = ALLOWED_TRAIN_SEEDS if args.split == "train" else ALLOWED_DEV_SEEDS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(allowed)
    for seed in seeds:
        if seed not in allowed:
            raise ValueError("Unregistered %s seed: %d" % (args.split, seed))
    keys = [args.profile_key] if args.profile_key != "all" else \
        list(profiles["profiles"])
    for key in keys:
        if key not in profiles["profiles"]:
            raise ValueError("Unknown profile key %r (have %s)"
                             % (key, sorted(profiles["profiles"])))
        profile = profiles["profiles"][key]
        for seed in seeds:
            output = OUT / args.split / key / f"seed{seed}_steps{args.steps}"
            if (output / "manifest.json").exists():
                print(json.dumps({"skip": str(output)}), flush=True)
                continue
            collect_episode(args.split, key, profile, seed, args.steps, output)


if __name__ == "__main__":
    main()
