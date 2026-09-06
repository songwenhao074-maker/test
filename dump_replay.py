"""Dump a full-trace deterministic gobi replay: per-step physical-host
overload labels + the LIVE GOBI schedule (result_cache), aligned with the
trace rows.  This is the ground-truth material for overload-based anomaly
labels (replacing the p98 rule).

Writes (deterministic given seed):
  data/qos_overload/time_series.npy      = copy of the qos trace (202x112)
  data/qos_overload/schedule_series.npy  = GOBI result_cache one-hot (202x16x16)
  data/qos_overload/labels_overload.npy  = per-step per-physical-host overload
                                           (202x16 int: 1 = base_ips>cap OR ram>cap)
  data/qos_overload/replay_log.npz       = extra diagnostics:
                                           placement (T,16) argmax host per slot
                                           n_containers (T,16)
                                           total_base_ips (T,16)

Usage: python dump_replay.py [seed]
"""
import os
import sys
import random
import numpy as np
import torch
import warnings
import logging
import psutil
import shutil

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from simulator.Simulator import Simulator
from simulator.environment.RPiEdge import RPiEdge
from simulator.workload.OfflineTraceWorkloadV2 import (
    OfflineTraceWorkloadV2 as OfflineTraceWorkload, _trace_columns, EPS_IPS)
from scheduler.GOBI import GOBIScheduler
from recovery.Recovery import Recovery
from stats.Stats import Stats

HOSTS = 16
CONTAINERS = 16
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300
START_CONTAINERS = 12
LOAD_SCALE = 1.5
OUT = os.environ.get('QOS_OVERLOAD_OUT', 'recovery/PreGANSrc/data/qos_overload')

TRACE = np.load('recovery/PreGANSrc/data/qos/time_series.npy')
T = TRACE.shape[0]
FIXED_SCHEDULE_PATH = os.environ.get('FIXED_SCHEDULE_PATH', '')


def main():
    os.environ['OMP_NUM_THREADS'] = '3'
    os.environ['MKL_NUM_THREADS'] = '3'
    os.environ['OPENBLAS_NUM_THREADS'] = '3'
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    datacenter = RPiEdge(HOSTS)
    workload = OfflineTraceWorkload(0)
    scheduler = GOBIScheduler('energy_latency_' + str(HOSTS))
    recovery = Recovery()
    stats = Stats(workload, datacenter, scheduler)
    stats.feats_per_host = 7
    hostlist = datacenter.generateHosts()
    env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, recovery, stats,
                    CONTAINERS, INTERVAL_TIME, hostlist)

    if START_CONTAINERS > 0:
        for h in range(START_CONTAINERS):
            cid, container = workload.make_task(env, h, 0, LOAD_SCALE)
            env.addContainerInit(cid, 0, container.ipsmodel, container.rammodel,
                                 container.diskmodel)
            workload.createdContainers.append(
                (cid, 0, container.ipsmodel, container.rammodel, container.diskmodel))
            workload.deployedContainers.append(False)
    deployed = env.addContainerListInit([])
    decision = scheduler.placement(deployed)
    migrations = env.allocateInit(decision)
    workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
    stats.saveStats(deployed, migrations, [], deployed, decision, 0)
    workload.set_containers(env.containerlist)

    schedules = np.zeros((T, CONTAINERS, HOSTS))
    fixed_schedules = (np.load(FIXED_SCHEDULE_PATH)
                       if FIXED_SCHEDULE_PATH else None)
    if fixed_schedules is not None and fixed_schedules.shape != schedules.shape:
        raise ValueError('FIXED_SCHEDULE_PATH must contain (202,16,16)')
    placements = np.full((T, CONTAINERS), -1, dtype=int)
    overload = np.zeros((T, HOSTS), dtype=int)
    n_containers = np.zeros((T, HOSTS), dtype=int)
    total_base = np.zeros((T, HOSTS), dtype=float)
    total_ram = np.zeros((T, HOSTS), dtype=float)
    total_disk = np.zeros((T, HOSTS), dtype=float)
    container_demands = np.zeros((T, CONTAINERS, 7), dtype=float)
    causal_host_features = np.zeros((T, HOSTS, 7), dtype=float)

    for step in range(T):
        if psutil.virtual_memory().available / 1024**3 < 4.5:
            raise RuntimeError('Replay RAM guard: available memory below 4.5 GiB')
        if shutil.disk_usage(os.getcwd()).free / 1024**3 < 20:
            raise RuntimeError('Replay disk guard: free space below 20 GiB')
        workload.set_containers(env.containerlist)
        workload.apply_trace_row(step, LOAD_SCALE)
        newcontainerinfos = workload.generateNewContainers(env.interval)
        deployed, destroyed = env.addContainers(newcontainerinfos)
        for spawned in workload.create_next_tasks(env, step, LOAD_SCALE):
            deployed.append(spawned)
        workload.update_slot_hosts(env)
        # Physical-host state before the proposed scheduling action.  Slot
        # demands and physical-host telemetry have different identities.
        for hid in range(HOSTS):
            for cid in env.getContainersOfHost(hid):
                container = env.getContainerByID(cid)
                ram_size, ram_read, ram_write = container.getRAM()
                disk_size, disk_read, disk_write = container.getDisk()
                causal_host_features[step, hid] += [container.getBaseIPS(), ram_size,
                    ram_read, ram_write, disk_size, disk_read, disk_write]
        selected = scheduler.selection()
        if fixed_schedules is None:
            decision = scheduler.filter_placement(
                scheduler.placement(selected + deployed))
            rc = np.array(scheduler.result_cache)
        else:
            rc = fixed_schedules[step]
            decision = workload.replay_decision(rc, env)
        recovered = recovery.run_model(stats.time_series, decision)

        # record schedule BEFORE simulationStep (the decision for this step)
        schedules[step] = rc
        for cid in range(CONTAINERS):
            container = env.containerlist[cid]
            if container is not None:
                row = rc[cid]
                placements[step, cid] = int(np.argmax(row))
                ram_size, ram_read, ram_write = container.getRAM()
                disk_size, disk_read, disk_write = container.getDisk()
                container_demands[step, cid] = [
                    container.getBaseIPS(), ram_size, ram_read, ram_write,
                    disk_size, disk_read, disk_write,
                ]
                if os.environ.get('TRACE_BINDING_AUDIT', '0') == '1':
                    expected = _trace_columns(workload.trace[step], cid, LOAD_SCALE)
                    expected = (max(expected[0], EPS_IPS), expected[1], expected[2])
                    actual = (container.getBaseIPS(), ram_size, disk_size)
                    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-5):
                        raise AssertionError(f'Live trace mismatch step={step} slot={cid}: {actual} vs {expected}')

        migrations = env.simulationStep(recovered)
        workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))

        # record overload AFTER the step
        for h in range(HOSTS):
            cids = env.getContainersOfHost(h)
            base = sum(env.getContainerByID(c).getBaseIPS() for c in cids)
            ram = sum(env.getContainerByID(c).getRAM()[0] for c in cids)
            disk = sum(env.getContainerByID(c).getDisk()[0] for c in cids)
            host = env.getHostByID(h)
            n_containers[step, h] = len(cids)
            total_base[step, h] = base
            total_ram[step, h] = ram
            total_disk[step, h] = disk
            if (base > host.ipsCap or ram > host.ramCap.size or
                    disk > host.diskCap.size):
                overload[step, h] = 1

        stats.saveStats(deployed, migrations, destroyed, selected, decision, 0)
        stats.time_series[-1] = workload.trace[step]

    # class labels: the resource dimension that pushed the host over capacity
    # (ratio argmax among cpu/ram/disk), matching the p98 class convention
    # 1=cpu 2=ram 3=disk.  Deterministic from the recorded totals.
    caps = np.array([h.ipsCap for h in env.hostlist])
    ramcaps = np.array([h.ramCap.size for h in env.hostlist])
    diskcaps = np.array([h.diskCap.size for h in env.hostlist])
    ratio = np.stack([total_base / caps, total_ram / ramcaps,
                      total_disk / diskcaps], axis=2)          # (T,H,3)
    cls = np.zeros((T, HOSTS), dtype=int)
    for t in range(T):
        for h in range(HOSTS):
            if overload[t, h]:
                cls[t, h] = int(np.argmax(ratio[t, h])) + 1

    os.makedirs(OUT, exist_ok=True)
    exported_time = (causal_host_features.reshape(T, -1)
                     if os.environ.get('EXPORT_CAUSAL_HOST_SERIES', '0') == '1' else TRACE)
    np.save(os.path.join(OUT, 'time_series.npy'), exported_time)
    np.save(os.path.join(OUT, 'schedule_series.npy'), schedules)
    np.save(os.path.join(OUT, 'labels_overload.npy'), overload)
    np.save(os.path.join(OUT, 'labels_overload_class.npy'), cls)
    np.savez(os.path.join(OUT, 'replay_log.npz'),
             placement=placements, n_containers=n_containers,
             total_base_ips=total_base, total_ram=total_ram,
             total_disk=total_disk, container_demands=container_demands,
             causal_host_features=causal_host_features,
             host_cpu_capacity=caps, host_ram_capacity=ramcaps,
             host_disk_capacity=diskcaps)
    print(f'[dump] seed={seed}: wrote {OUT}')
    print(f'[dump] overload host-steps: {overload.sum()}/{overload.size} '
          f'({100 * overload.mean():.2f}%), steps with overload: '
          f'{(overload.sum(axis=1) > 0).sum()}/{T}')
    print(f'[dump] class counts: '
          f'1(cpu)={int((cls == 1).sum())} 2(ram)={int((cls == 2).sum())} '
          f'3(disk)={int((cls == 3).sum())}')
    print(f'[dump] overload in 140:200 segment: '
          f'{overload[140:200].sum()} host-steps '
          f'({(overload[140:200].sum(axis=1) > 0).sum()}/60 steps)')


if __name__ == '__main__':
    main()
