"""112-column offline-trace workload (expand_old / version2 7-feature layout).

Same finite-task design as OfflineTraceWorkload, but reads the version2
7-feature-per-host trace layout [cpu, ram, ram_read, ram_write, disk,
disk_read, disk_write] x 16 hosts (112 columns) that the final-chain v2
models (FTMoE_v*_16_v2) are trained on.

CPU: per-host column i*7 + 0 (percent demand -> base IPS via IPS_RATIO).
RAM: per-host column i*7 + 1 (MB demand, RAM_FACTOR).
Disk: per-host column i*7 + 4 (size demand, clamped to MAX_DISK).
"""
import os
import numpy as np
from .Workload import Workload
from simulator.container.IPSModels.IPSMConstant import IPSMConstant
from simulator.container.RAMModels.RMConstant import RMConstant
from simulator.container.DiskModels.DMConstant import DMConstant

TRACE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'recovery', 'PreGANSrc', 'data')
TIME_SERIES_PATH = os.path.join(TRACE_ROOT, 'qos', 'time_series.npy')
SCHEDULE_SERIES_PATH = os.path.join(TRACE_ROOT, 'qos', 'schedule_series.npy')

IPS_RATIO = 18.6        # (requested cpu %) -> base IPS demand on a 4029-IPS RPi host
RAM_FACTOR = 1.4        # offline ram (MB) -> container ram demand (MB) on 4 GB hosts
# RAM_SCALE: extra multiplier on the ram demand (env-tunable).  The default
# 1.0 keeps legacy behavior; higher values make RAM a second real bottleneck
# in the simulator (multi-class overload labels: cpu AND ram faults), which
# the QoS-overload labeling needs for non-trivial fault-classification
# metrics.  Must be the same during label generation (dump_replay.py),
# training-data generation and online replay (run_qos_chain/probe).
RAM_SCALE = float(os.environ.get('RAM_SCALE', '1.0'))
DISK_SCALE = float(os.environ.get('DISK_SCALE', '1.0'))
EPS_IPS = 2.0           # floor so a 0-demand container never reads as finished/destroyed
MAX_CPU = 100.0
MAX_DISK = 9.0          # keep sum(disk) <= RPiEdge host disk capacity (~32212)
TASK_DURATION = 30      # intervals of work per task (at its creation demand)
FEATS_PER_HOST = 7


class _TraceContainerCPU(IPSMConstant):
    """Per-interval IPS demand read from the offline trace, finite task."""

    def set_trace_row(self, cpu_req):
        self.constant_ips = max(cpu_req, EPS_IPS)
        self.max_ips = max(self.constant_ips * 1.15, 4.0)

    def start_task(self, cpu_req, duration):
        self.set_trace_row(cpu_req)
        self.duration = max(duration, 1)
        self.totalInstructions = self.constant_ips * self.duration * self.container.env.intervaltime
        self.completedInstructions = 0

    def getIPS(self):
        if self.completedInstructions < self.totalInstructions:
            return self.constant_ips
        return 0


class _TraceContainerRAM(RMConstant):
    """Constant per-interval RAM demand read from the offline trace."""

    def set_trace_row(self, ram_req):
        self.size = ram_req
        self.read = 1.0
        self.write = 1.0


class _TraceContainerDisk(DMConstant):
    """Constant per-interval disk demand read from the offline trace."""

    def set_trace_row(self, disk_req):
        self.constant_size = disk_req
        self.constant_read = 1.0
        self.constant_write = 1.0


def _trace_columns(row, host_slot, load_scale=1.0):
    """Map a 112-col trace row to (cpu, ram, disk) demands for one host."""
    cpu = np.clip(row[host_slot * FEATS_PER_HOST + 0] * load_scale,
                  0.0, MAX_CPU) * IPS_RATIO
    ram = row[host_slot * FEATS_PER_HOST + 1] * RAM_FACTOR * RAM_SCALE
    disk = (np.clip(row[host_slot * FEATS_PER_HOST + 4], 0.0, MAX_DISK) *
            DISK_SCALE)
    return cpu, ram, disk


class OfflineTraceWorkloadV2(Workload):
    """Workload that replays the 112-col trace as a sequence of finite tasks.

    Same protocol as OfflineTraceWorkload (apply_trace_row / make_task /
    create_next_tasks / load_replay_schedule / replay_decision).
    """

    def __init__(self, trace_offset=0):
        super().__init__()
        self.trace = np.load(TIME_SERIES_PATH)
        self.trace_offset = trace_offset
        n_hosts = 16
        if self.trace.shape[1] != n_hosts * FEATS_PER_HOST:
            raise ValueError(
                f'expected 16-host {FEATS_PER_HOST}-col trace (112), '
                f'got {self.trace.shape}')
        self._container_reqs = {}
        self._slot_hosts = [-1] * n_hosts

    def set_containers(self, containerlist):
        # Keep the live slot list, not only the models that existed at startup.
        # Completed tasks are replaced in-place by create_next_tasks.
        self._live_containers = containerlist
        self._container_reqs = {
            c.id: (c.ipsmodel, c.rammodel, c.diskmodel)
            for c in containerlist if c is not None and c.active}

    def update_slot_hosts(self, env):
        for c in env.containerlist:
            if c and c.active:
                while len(self._slot_hosts) <= c.id:
                    self._slot_hosts.append(-1)
                self._slot_hosts[c.id] = c.getHostID()

    def apply_trace_row(self, step, load_scale=1.0):
        if hasattr(self, '_live_containers'):
            self.set_containers(self._live_containers)
        row = self.trace[step]
        for slot, (ips, rmod, dmod) in self._container_reqs.items():
            cpu, ram, disk = _trace_columns(row, slot, load_scale)
            ips.set_trace_row(cpu)
            rmod.set_trace_row(ram)
            dmod.set_trace_row(disk)

    def make_task(self, env, host_slot, step, load_scale=1.0, cid=None, slot=None):
        from simulator.container.Container import Container
        row = self.trace[step]
        cpu, ram, disk = _trace_columns(row, host_slot, load_scale)
        ipsm = _TraceContainerCPU(cpu, max(cpu * 1.15, 4.0),
                                  TASK_DURATION, TASK_DURATION)
        rmm = _TraceContainerRAM(ram, 1.0, 1.0)
        dmm = _TraceContainerDisk(disk, 1.0, 1.0)
        if cid is None:
            cid = self.creation_id
            self.creation_id += 1
        if slot is None:
            slot = len(env.containerlist)
        container = Container(slot, cid, step, ipsm, rmm, dmm, env, HostID=-1)
        ipsm.start_task(cpu, TASK_DURATION)
        return cid, container

    def create_next_tasks(self, env, step, load_scale=1.0):
        spawned = []
        for i, c in enumerate(env.containerlist):
            if c is None or not c.active:
                host = self._slot_hosts[i] if i < len(self._slot_hosts) else -1
                if host < 0 or host >= len(env.hostlist):
                    host = i % len(env.hostlist)
                # A workload slot retains its trace identity after migration.
                # Physical placement must not select another slot's demand.
                cid, container = self.make_task(env, i, step, load_scale,
                                                slot=i)
                container.hostid = host
                env.containerlist[i] = container
                self._slot_hosts[i] = host
                self.createdContainers.append(
                    (cid, step, container.ipsmodel, container.rammodel,
                     container.diskmodel))
                self.deployedContainers.append(False)
                spawned.append(container)
        self.set_containers(env.containerlist)
        return spawned

    def generateNewContainers(self, interval):
        return []

    @classmethod
    def load_replay_schedule(cls):
        return np.load(SCHEDULE_SERIES_PATH)

    @classmethod
    def replay_decision(cls, schedule, env, container_alloc=None):
        if container_alloc is None:
            container_alloc = [-1] * len(env.containerlist)
            for c in env.containerlist:
                if c and c.getHostID() != -1:
                    container_alloc[c.id] = c.getHostID()
        decision = []
        for cid in range(len(container_alloc)):
            if container_alloc[cid] == -1:
                continue
            new_host = int(np.argmax(schedule[cid]))
            if new_host != container_alloc[cid]:
                decision.append((cid, new_host))
        return decision
