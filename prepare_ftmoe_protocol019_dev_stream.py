"""Protocol 019 — development stream collector (S2 cold-start gate).

Reuses the protocol-016 adapted-BWGD2 contract verbatim (same demand
adapter, same training-derived synthetic disk law read-only, same simulator
stack) under a *new registered replay seed*, so the stream is independent of
the 016 pilot stream.

This is a new top-level script on purpose: prepare_ftmoe_adapted_bwgd2.py is
registered to seed 301 only and must stay untouched.

Registered here (dev only):
    replay seed 303, steps 300, method A (frozen), model seed 1

The scenario is frozen after collection; any scenario change after the
data-only audit is a Protocol-019 violation (see
docs/FTMOE_ONLINE_PROTOCOL_019.md gates and
指令/FTMOE_ONLINE_TUNING_REVIEW_AND_SOLUTION_PLAN.md §20).
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
OUT = ROOT / "artifacts/ftmoe_online/protocol_019"
ALLOWED_SEEDS = {303}
ALLOWED_STEPS = {300}


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


def configure():
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
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def guard():
    """Protocol-019 scoped resource floor (deviation P05, see 指令/
    FTMOE_PROTOCOL019_PROBLEM_LOG.md): host only has ~15.2 GiB total RAM and
    the repo-wide 4.5 GiB floor is regularly unsatisfiable (it already killed
    the 017 confirmation twice).  This collector's recorded peak RSS is
    0.6 GiB (016 manifest), so the floor is 3.0 GiB, still leaving >2 GiB
    slack; override with FTMOE019_RAM_GUARD_GIB."""
    import psutil
    import shutil
    floor = float(os.environ.get("FTMOE019_RAM_GUARD_GIB", "3.0"))
    available = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage(ROOT).free / 2**30
    if available < floor:
        raise RuntimeError("RAM guard below %.1f GiB (available %.3f GiB)" % (floor, available))
    if disk < 20:
        raise RuntimeError("Disk guard below 20 GiB (available %.3f GiB)" % disk)


def dev_config(seed, steps):
    law = ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json"
    config = {
        "schema_version": 1,
        "protocol": "019",
        "phase": "S2",
        "name": "development stream for cold-start compatibility",
        "registered_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_seed": 1,
        "replay_seed": seed,
        "steps": steps,
        "guard_steps": 1,
        "methods": ["A"],
        "learning_rate": 0.0,
        "input_contract_version": 2,
        "normalization_version": 2,
        "graph_semantics_version": 2,
        "cpu_positive_clip": [2.0, 1860.0],
        "ram_multiplier": 2.0,
        "io_per_container": 1.0,
        "capacity_scales": {"CPU_CAP_SCALE": 0.8, "DISK_CAP_SCALE": 0.25},
        "native_arrivals_and_vm_selection": True,
        "synthetic_dynamic_disk": True,
        "artificial_time_drift": False,
        "disk_law_sha256": sha(law),
        "distribution_target": {
            "main_positive_fraction": [0.05, 0.25],
            "cpu_dominant": True,
            "all_three_raw_resources_present": True,
        },
        "execution_boundary": (
            "Protocol 019 S2 development stream only. The scenario is frozen after "
            "the data-only audit; tuning capacities or workloads from model results "
            "is forbidden. If the cold-start gate fails, proceed to S3 common "
            "offline adaptation instead."
        ),
        "cold_start_gate": "Frozen A PR-AUC >= 0.60 and FPR <= 0.15",
    }
    return config


def register(seed, steps):
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Unregistered Protocol-019 development seed: {seed} (allowed {sorted(ALLOWED_SEEDS)})")
    if steps not in ALLOWED_STEPS:
        raise ValueError(f"Unregistered Protocol-019 horizon: {steps} (allowed {sorted(ALLOWED_STEPS)})")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "dev_stream_config.json"
    if not path.exists():
        write_json(path, dev_config(seed, steps))
        print(json.dumps({"registered": str(path), "seed": seed, "steps": steps}))
    return json.loads(path.read_text(encoding="utf8"))


def collect(seed, steps, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    console = sys.stdout
    config = register(seed, steps)
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
            events = []
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
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise AssertionError("Invalid pre-action demand")
                    demands[t, slot] = values
                    creation[t, slot] = c.creationID
                    before[t, slot] = c.getHostID()
                    if c.getHostID() >= 0:
                        host[t, c.getHostID()] += values
                selected = scheduler.selection()
                decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
                schedules[t] = np.asarray(scheduler.result_cache)
                if not np.isfinite(schedules[t]).all():
                    raise AssertionError("Nonfinite proposed schedule")
                np.testing.assert_allclose(schedules[t].sum(-1), 1, atol=1e-5)
                if (schedules[t] < -1e-6).any():
                    raise AssertionError("Negative proposed schedule")
                migrations = env.simulationStep(recovery.run_model(stats.time_series, decision))
                workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
                for slot, c in enumerate(env.containerlist):
                    if c is None:
                        continue
                    hid = c.getHostID()
                    if not 0 <= hid < 16:
                        raise AssertionError("Unallocated container survived simulationStep")
                    if c.creationID != creation[t, slot]:
                        raise AssertionError("Slot identity changed inside simulationStep")
                    after[t, slot] = hid
                    after_creation[t, slot] = c.creationID
                    totals[t, hid] += [c.getBaseIPS(), c.getRAM()[0], c.getDisk()[0]]
                ratio = totals[t] / capacities
                labels[t] = np.where((ratio > 1).any(-1), ratio.argmax(-1) + 1, 0)
                if not np.isfinite(totals[t]).all() or (totals[t] < 0).any():
                    raise AssertionError("Invalid post-action totals")
                events.append({"step": t + 1, "simulator_interval": int(env.interval),
                               "proposed_decision": [[int(x), int(y)] for x, y in decision],
                               "actual_migrations": [[int(x), int(y)] for x, y in migrations]})
                stats.saveStats(deployed, migrations, destroyed, selected, decision, 0)
                if (t + 1) % 50 == 0:
                    print(json.dumps({"collected": t + 1, "total": count,
                                      "elapsed_seconds": time.perf_counter() - started}),
                          file=console, flush=True)
            np.savez_compressed(output / "stream.npz", host_features=host, demands=demands,
                                schedules=schedules, raw_labels=labels, capacities=capacities,
                                post_totals=totals, before_placement=before, after_placement=after,
                                creation_ids=creation, after_creation_ids=after_creation,
                                simulator_intervals=intervals)
            (output / "events.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf8")
            sources = [ROOT / "prepare_ftmoe_protocol019_dev_stream.py",
                       ROOT / "recovery/PreGANSrc/src/ftmoe_normalization.py",
                       ROOT / "recovery/PreGANSrc/src/ftmoe_input_contract.py",
                       ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
                       scheduler_weight]
            for base in ("simulator", "scheduler", "metrics", "stats", "utils"):
                sources.extend(p for p in (ROOT / base).rglob("*.py") if "__pycache__" not in p.parts)
            sources.extend((ROOT / "scheduler/BaGTI").rglob("*.npy"))
            sources.extend(sorted(bitbrain.glob("*.csv")))
            source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))}
            manifest = {"schema_version": 1, "protocol": "019", "phase": "S2",
                        "seed": seed, "steps": steps, "guard_steps": 1,
                        "workload": "adapted_BWGD2_protocol019_dev",
                        "interval_seconds": 300, "hosts": 16, "containers": 16,
                        "arrival_mean": 1, "arrival_sigma": 1.5,
                        "capacity_scales": {"CPU_CAP_SCALE": 0.8, "DISK_CAP_SCALE": 0.25},
                        "artificial_drift": False, "synthetic_dynamic_disk": True,
                        "demand_adapter": {"cpu_positive_clip": [2, 1860], "ram_multiplier": 2,
                                           "io_constant": 1},
                        "input_contract_version": config["input_contract_version"],
                        "normalization_version": config["normalization_version"],
                        "graph_semantics_version": config["graph_semantics_version"],
                        "dev_config_sha256": sha(OUT / "dev_stream_config.json"),
                        "disk_law_sha256": config["disk_law_sha256"],
                        "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                        "capacities": capacities.tolist(),
                        "selected_vm_indices": workload.possible_indices,
                        "source_sha256": source_hashes,
                        "stream_sha256": sha(output / "stream.npz"),
                        "events_sha256": sha(output / "events.json"),
                        "raw_class_counts_scored": np.bincount(labels[:steps].ravel(), minlength=4).tolist(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "rss_gib": psutil.Process().memory_info().rss / 2**30,
                        "ordering": ("new tasks, addContainers, pre-action features, proposed GOBI "
                                     "schedule, simulationStep, actual physical labels"),
                        "replay_protocol": ("record once; consumers predict before exposing each step "
                                            "label; tolerance matures one interval later")}
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        print(json.dumps({k: v for k, v in manifest.items()
                          if k not in ("source_sha256", "selected_vm_indices")}, indent=2), flush=True)
    except Exception as exc:
        failure = {"seed": seed, "steps": steps, "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "completed_steps": len(locals().get("events", [])),
                   "simulator_behavior_modified": False,
                   "next_action": "Report before any scenario change"}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf8")
        print(json.dumps(failure), file=console, flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    collect(args.seed, args.steps,
            args.output or OUT / f"dev_streams/seed{args.seed}_steps{args.steps}")
