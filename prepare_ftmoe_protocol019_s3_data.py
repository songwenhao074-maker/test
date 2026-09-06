"""Protocol 019 S3 — same-domain adaptation data collector.

Collects adapted-BWGD2 (protocol-016 contract) episodes under newly
registered S3 replay seeds for the common offline adaptation dataset
(train seeds 401-405, dev seeds 406-408; horizon 400 scored intervals).

Registration is S3-scoped and does not alter the S2 dev-stream registration
(seed 303 / steps 300) nor any protocol 015/016 script.
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
OUT = ROOT / "artifacts/ftmoe_online/protocol_019/adaptation_data"
TRAIN_SEEDS = [401, 402, 403, 404, 405]
DEV_SEEDS = [406, 407, 408]
ALLOWED_STEPS = {400}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


_THREADS_CONFIGURED = False


def configure():
    global _THREADS_CONFIGURED
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "3"
    for name in ("RAM_SCALE", "DISK_SCALE", "CPU_CAP_SCALE", "DISK_CAP_SCALE"):
        os.environ[name] = "1.0"
    os.environ["CPU_CAP_SCALE"] = "0.8"
    os.environ["DISK_CAP_SCALE"] = "0.25"
    for name in ("FIXED_SCHEDULE_PATH", "QOS_OVERLOAD_OUT", "ONLINE_TUNE", "ONLINE_LABEL_MODE"):
        os.environ.pop(name, None)
    import torch
    import psutil
    if not _THREADS_CONFIGURED:
        # torch refuses a second set_num_interop_threads call in one process.
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
        _THREADS_CONFIGURED = True
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def guard():
    import psutil
    import shutil
    floor = float(os.environ.get("FTMOE019_RAM_GUARD_GIB", "3.0"))
    available = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage(ROOT).free / 2**30
    if available < floor:
        raise RuntimeError("RAM guard below %.1f GiB (available %.3f GiB)" % (floor, available))
    if disk < 20:
        raise RuntimeError("Disk guard below 20 GiB (available %.3f GiB)" % disk)


def episode(seed, steps, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    console = sys.stdout
    try:
        configure()
        guard()
        os.chdir(ROOT)
        import numpy as np
        import torch
        import psutil
        bitbrain = ROOT / "simulator/workload/datasets/bitbrain/rnd"
        if not all((bitbrain / f"{i}.csv").is_file() for i in range(1, 500)):
            raise FileNotFoundError("Local Bitbrain dataset incomplete; downloading is not authorized")
        started = time.perf_counter()
        with (output / "generation.log").open("w", encoding="utf8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from simulator.Simulator import Simulator
            from simulator.environment.RPiEdge import RPiEdge
            from simulator.workload.BitbrainWorkloadAdapted import AdaptedBWGD2
            from scheduler.GOBI import GOBIScheduler
            from recovery.Recovery import Recovery
            from stats.Stats import Stats
            from src.constants import MODEL_SAVE_PATH
            candidates = [ROOT / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt",
                          ROOT / "scheduler/BaGTI" / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt"]
            scheduler_weight = next((p.resolve() for p in candidates if p.is_file()), None)
            if scheduler_weight is None:
                raise FileNotFoundError("GOBI trained checkpoint missing")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            dc = RPiEdge(16)
            workload = AdaptedBWGD2(1, 1.5, seed)
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300, dc.generateHosts())
            initial = workload.generateNewContainers(env.interval)
            deployed = env.addContainersInit(initial)
            decision = scheduler.placement(deployed)
            migrations = env.allocateInit(decision)
            workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
            stats.saveStats(deployed, migrations, [], deployed, decision, 0)
            capacities = np.array([[h.ipsCap, h.ramCap.size, h.diskCap.size] for h in env.hostlist],
                                  dtype=np.float64)
            count = steps + 1
            host = np.zeros((count, 16, 7), np.float32)
            demands = np.zeros_like(host)
            schedules = np.zeros((count, 16, 16), np.float32)
            totals = np.zeros((count, 16, 3), np.float64)
            labels = np.zeros((count, 16), np.int64)
            before = np.full((count, 16), -1, np.int64)
            after = before.copy()
            creation = before.copy()
            after_creation = before.copy()
            intervals = np.zeros(count, np.int64)
            for t in range(count):
                guard()
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
                    values = np.array([c.getBaseIPS(), *ram, *disk], dtype=np.float64)
                    demands[t, slot] = values
                    creation[t, slot] = c.creationID
                    before[t, slot] = c.getHostID()
                    if c.getHostID() >= 0:
                        host[t, c.getHostID()] += values
                selected = scheduler.selection()
                decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
                schedules[t] = np.asarray(scheduler.result_cache)
                np.testing.assert_allclose(schedules[t].sum(-1), 1, atol=1e-5)
                migrations = env.simulationStep(recovery.run_model(stats.time_series, decision))
                workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
                for slot, c in enumerate(env.containerlist):
                    if c is None:
                        continue
                    hid = c.getHostID()
                    if c.creationID != creation[t, slot]:
                        raise AssertionError("Slot identity changed inside simulationStep")
                    after[t, slot] = hid
                    after_creation[t, slot] = c.creationID
                    totals[t, hid] += [c.getBaseIPS(), c.getRAM()[0], c.getDisk()[0]]
                ratio = totals[t] / capacities
                labels[t] = np.where((ratio > 1).any(-1), ratio.argmax(-1) + 1, 0)
                stats.saveStats(deployed, migrations, destroyed, selected, decision, 0)
                if (t + 1) % 100 == 0:
                    print(json.dumps({"collected": t + 1, "total": count,
                                      "elapsed_seconds": time.perf_counter() - started}),
                          file=console, flush=True)
            np.savez_compressed(output / "stream.npz", host_features=host, demands=demands,
                                schedules=schedules, raw_labels=labels, capacities=capacities,
                                post_totals=totals, before_placement=before, after_placement=after,
                                creation_ids=creation, after_creation_ids=after_creation,
                                simulator_intervals=intervals)
            sources = [ROOT / "prepare_ftmoe_protocol019_s3_data.py",
                       ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
                       scheduler_weight]
            for base in ("simulator", "scheduler", "metrics", "stats", "utils"):
                sources.extend(p for p in (ROOT / base).rglob("*.py") if "__pycache__" not in p.parts)
            sources.extend((ROOT / "scheduler/BaGTI").rglob("*.npy"))
            sources.extend(sorted(bitbrain.glob("*.csv")))
            source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))}
            manifest = {"schema_version": 1, "protocol": "019", "phase": "S3-data",
                        "split": "train" if seed in TRAIN_SEEDS else "dev",
                        "seed": seed, "steps": steps, "guard_steps": 1,
                        "workload": "adapted_BWGD2_protocol019", "interval_seconds": 300,
                        "hosts": 16, "containers": 16, "arrival_mean": 1, "arrival_sigma": 1.5,
                        "capacity_scales": {"CPU_CAP_SCALE": 0.8, "DISK_CAP_SCALE": 0.25},
                        "artificial_drift": False, "synthetic_dynamic_disk": True,
                        "demand_adapter": {"cpu_positive_clip": [2, 1860], "ram_multiplier": 2,
                                           "io_constant": 1},
                        "input_contract_version": 2, "normalization_version": 2,
                        "graph_semantics_version": 2,
                        "disk_law_sha256": sha(ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json"),
                        "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                        "capacities": capacities.tolist(),
                        "selected_vm_indices": workload.possible_indices,
                        "source_sha256": source_hashes,
                        "stream_sha256": sha(output / "stream.npz"),
                        "raw_class_counts_scored": np.bincount(labels[:steps].ravel(), minlength=4).tolist(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "rss_gib": psutil.Process().memory_info().rss / 2**30}
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"seed": seed, "steps": steps, "split": manifest["split"],
                          "stream_sha256": manifest["stream_sha256"],
                          "class_counts": manifest["raw_class_counts_scored"],
                          "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    except Exception as exc:
        failure = {"seed": seed, "steps": steps, "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "simulator_behavior_modified": False,
                   "next_action": "Report before any scenario change"}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf8")
        print(json.dumps(failure), file=console, flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.steps not in ALLOWED_STEPS:
        raise SystemExit(f"Unregistered horizon {args.steps}; allowed {sorted(ALLOWED_STEPS)}")
    seeds = args.seeds or (TRAIN_SEEDS + DEV_SEEDS)
    for seed in seeds:
        if seed not in TRAIN_SEEDS + DEV_SEEDS:
            raise SystemExit(f"Unregistered S3 seed {seed}")
    for seed in seeds:
        episode(seed, args.steps, OUT / f"raw/seed{seed}_steps{args.steps}")
