"""Model QoS probe: run a chain model (v0..v4c1) through the QoS replay and
record, per step, detection trigger vs p98 label vs real overload.

Usage: python probe_model.py <mode> [steps] [seed]
  mode: v0|v1b|v2c2|v3b2|v4c1
  env QOS_PROBE_WARMUP (default 40), QOS_PROBE_OUT (default logs/qos_probe_model.csv)
"""
import os
import sys
import csv
import time
import random
import numpy as np
import torch
import warnings
import logging

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from simulator.Simulator import Simulator
from simulator.environment.RPiEdge import RPiEdge
from simulator.workload.OfflineTraceWorkloadV2 import (
    OfflineTraceWorkloadV2 as OfflineTraceWorkload)
from scheduler.GOBI import GOBIScheduler
from recovery.FTMoEChain import FTMoEChainRecovery
from stats.Stats import Stats

HOSTS = 16
CONTAINERS = 16
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300
TRACE_OFFSET = int(os.environ.get('QOS_TRACE_OFFSET', '140'))
WARMUP_OFFSET = int(os.environ.get('QOS_WARMUP_OFFSET', '100'))
START_CONTAINERS = 12
LOAD_SCALE = 1.5

TRACE = np.load('recovery/PreGANSrc/data/qos/time_series.npy')
P98 = np.percentile(TRACE, 98, axis=0)
ROW_LABEL = (TRACE > P98).any(axis=1)
HOST_LABEL = (TRACE[:, :, None].reshape(TRACE.shape[0], 16, 7) >
              P98.reshape(16, 7)).any(axis=2)


def remove_gan_ckpt(folder, prefix):
    import glob
    seed = os.environ.get('FTMoE_CKPT_SEED', '')
    seed_suffix = f'_s{seed}' if seed else ''
    for stem in ('qos_', 'simulator_'):
        for pat in (f'{stem}{prefix}_Gen_16{seed_suffix}.ckpt',
                    f'{stem}{prefix}_Disc_16{seed_suffix}.ckpt'):
            for p in glob.glob(os.path.join(folder, pat)):
                os.remove(p)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'v4c1'
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    warmup = int(os.environ.get('QOS_PROBE_WARMUP', '40'))
    out_path = os.environ.get('QOS_PROBE_OUT', 'logs/qos_probe_model.csv')

    os.environ['FTMoE_MODEL'] = mode
    os.environ['FTMoE_CKPT_SEED'] = str(seed)
    if os.environ.get('GAN_FRESH', '1') == '1':
        remove_gan_ckpt('recovery/PreGANSrc/checkpointsplus/', 'FTMoE')
    os.environ['V2_MOE_EXPERTS'] = os.environ.get('V2_MOE_EXPERTS', '12')
    os.environ['V2_DROPOUT'] = os.environ.get('V2_DROPOUT', '0')
    os.environ['V2_GRAPH_DROPOUT'] = os.environ.get('V2_GRAPH_DROPOUT', '0.4')
    os.environ['V2_MOE_HEAD'] = os.environ.get('V2_MOE_HEAD', '1')

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    datacenter = RPiEdge(HOSTS)
    workload = OfflineTraceWorkload(TRACE_OFFSET)
    scheduler = GOBIScheduler('energy_latency_' + str(HOSTS))
    recovery = FTMoEChainRecovery(HOSTS, '', training=True)
    stats = Stats(workload, datacenter, scheduler)
    stats.feats_per_host = 7
    hostlist = datacenter.generateHosts()
    env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, recovery, stats,
                    CONTAINERS, INTERVAL_TIME, hostlist)

    if START_CONTAINERS > 0:
        for h in range(START_CONTAINERS):
            cid, container = workload.make_task(env, h, TRACE_OFFSET, LOAD_SCALE)
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

    for step in range(warmup):
        row = WARMUP_OFFSET + step
        workload.apply_trace_row(row, LOAD_SCALE)
        newcontainerinfos = workload.generateNewContainers(env.interval)
        deployed, destroyed = env.addContainers(newcontainerinfos)
        for spawned in workload.create_next_tasks(env, row, LOAD_SCALE):
            deployed.append(spawned)
        workload.update_slot_hosts(env)
        selected = scheduler.selection()
        decision = scheduler.filter_placement(
            scheduler.placement(selected + deployed))
        recovered = recovery.run_model(stats.time_series, decision)
        migrations = env.simulationStep(recovered)
        workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
        stats.saveStats(deployed, migrations, destroyed, selected, decision, 0)
        stats.time_series[-1] = workload.trace[row]
    stats.hostinfo = []
    stats.metrics = []
    stats.workloadinfo = []
    stats.activecontainerinfo = []
    stats.allcontainerinfo = []
    stats.schedulerinfo = []

    rows = []
    t0 = time.time()
    for step in range(num_steps):
        row_idx = TRACE_OFFSET + step
        workload.apply_trace_row(row_idx, LOAD_SCALE)
        newcontainerinfos = workload.generateNewContainers(env.interval)
        deployed, destroyed = env.addContainers(newcontainerinfos)
        for spawned in workload.create_next_tasks(env, row_idx, LOAD_SCALE):
            deployed.append(spawned)
        workload.update_slot_hosts(env)
        selected = scheduler.selection()
        decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
        recovered = recovery.run_model(stats.time_series, decision)
        detected = int(getattr(recovery, 'last_detected', 0))
        migrations = env.simulationStep(recovered)
        workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
        stats.saveStats(deployed, migrations, destroyed, selected, decision, 0)

        per_host = []
        for h in range(HOSTS):
            cids = env.getContainersOfHost(h)
            base = sum(env.getContainerByID(c).getBaseIPS() for c in cids)
            ram = sum(env.getContainerByID(c).getRAM()[0] for c in cids)
            host = env.getHostByID(h)
            per_host.append(dict(n=len(cids), ocpu=float(base > host.ipsCap),
                                 oram=float(ram > host.ramCap.size)))
        m = stats.metrics[-1]
        rows.append(dict(
            step=step, row=row_idx,
            label=int(ROW_LABEL[row_idx]),
            detected=detected,
            n_overload=sum(p['ocpu'] or p['oram'] for p in per_host),
            destroyed=m['numdestroyed'], resp=m['avgresponsetime'],
            migrations=len(migrations),
        ))
        stats.time_series[-1] = workload.trace[row_idx]

    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lab = np.array([r['label'] for r in rows])
    det = np.array([r['detected'] for r in rows])
    ov = np.array([r['n_overload'] for r in rows])
    dest = np.array([r['destroyed'] for r in rows])

    def conf(a, b):
        tp = int(((a == 1) & (b == 1)).sum()); fp = int(((a == 1) & (b == 0)).sum())
        fn = int(((a == 0) & (b == 1)).sum()); tn = int(((a == 0) & (b == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=round(prec, 3),
                    rec=round(rec, 3))

    print(f'=== {mode} s{seed}: {num_steps} steps, elapsed {elapsed:.0f}s ===')
    print(f'triggers: {det.sum()}/{num_steps} | labeled: {lab.sum()} | '
          f'overloaded steps: {(ov > 0).sum()} | destroyed: {dest.sum()}')
    for name, ev in [('label', lab), ('overload', ov > 0), ('destroy', dest > 0)]:
        c = conf(det, ev)
        print(f'  det vs {name:9s}: prec={c["prec"]} rec={c["rec"]} '
              f'tp={c["tp"]} fp={c["fp"]} fn={c["fn"]}')
    c = conf(lab, ov > 0)
    print(f'  label vs overload (baseline): prec={c["prec"]} rec={c["rec"]}')
    print(f'csv -> {out_path}')


if __name__ == '__main__':
    main()
