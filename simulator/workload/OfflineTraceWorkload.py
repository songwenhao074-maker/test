"""Offline-trace-driven workload for QoS experiments (Direction A).

Replays the host-level metric columns of the offline training trace
(recovery/PreGANSrc/data/simulator/time_series.npy) as the demands of
containers.  Each host slot runs a sequence of FINITE tasks: a task's
total work is the demand at its creation times TASK_DURATION intervals,
and the container is destroyed (completed) once that work is executed.
The workload then immediately creates the next task for the slot with
the current trace demand.  This gives the response-time metric real
signal (faster apparent IPS -> shorter completion) and keeps host load
dynamic, unlike the fully-persistent design where every host always had
one container and all QoS metrics collapsed.
"""
import os
import numpy as np
from .Workload import Workload
from simulator.container.IPSModels.IPSMConstant import IPSMConstant
from simulator.container.RAMModels.RMConstant import RMConstant
from simulator.container.DiskModels.DMConstant import DMConstant

TIME_SERIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'recovery', 'PreGANSrc', 'data', 'simulator', 'time_series.npy')
SCHEDULE_SERIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'recovery', 'PreGANSrc', 'data', 'simulator', 'schedule_series.npy')

IPS_RATIO = 18.6        # (requested cpu %) -> base IPS demand on a 4029-IPS RPi host
RAM_FACTOR = 1.4        # offline ram (MB) -> container ram demand (MB) on 4 GB hosts
EPS_IPS = 2.0           # floor so a 0-demand container never reads as finished/destroyed
MAX_CPU = 100.0
MAX_DISK = 9.0          # keep sum(disk) <= RPiEdge host disk capacity (~32212)
TASK_DURATION = 30      # intervals of work per task (at its creation demand)


class _TraceContainerCPU(IPSMConstant):
    """Per-interval IPS demand read from the offline trace, finite task.

    Each task has a fixed amount of work: totalInstructions =
    demand_IPS * TASK_DURATION intervals.  The container consumes work at
    its apparent IPS; on a sparse host the apparent IPS is much higher than
    the demand, so the task finishes early, the container is destroyed and
    the host slot idles (0 CPU) until a replacement task is spawned.
    """

    def set_trace_row(self, cpu_req):
        self.constant_ips = max(cpu_req, EPS_IPS)
        self.max_ips = max(self.constant_ips * 1.15, 4.0)

    def start_task(self, cpu_req, duration):
        """Begin a new finite task: total work = demand * duration intervals."""
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


class OfflineTraceWorkload(Workload):
    """Workload that replays the offline trace as a sequence of tasks.

    apply_trace_row() updates each active container's demand to the current
    trace row; create_next_tasks() spawns replacement tasks on slots whose
    container just completed (called after env.addContainers destroys them).
    """

    def __init__(self, trace_offset=0):
        super().__init__()
        self.trace = np.load(TIME_SERIES_PATH)
        self.trace_offset = trace_offset
        if self.trace.shape[1] != 48:
            raise ValueError(f'expected 16-host (48-col) trace, got {self.trace.shape}')
        self._container_reqs = {}
        self._slot_hosts = [-1] * 16   # last known host of each container slot

    def set_containers(self, containerlist):
        """Register the env.containerlist to push demands into."""
        self._container_reqs = {
            c.creationID: (c.ipsmodel, c.rammodel, c.diskmodel)
            for c in containerlist if c is not None}

    def update_slot_hosts(self, env):
        """Record each active slot's current host (migrations move them)."""
        for c in env.containerlist:
            if c and c.active:
                while len(self._slot_hosts) <= c.id:
                    self._slot_hosts.append(-1)
                self._slot_hosts[c.id] = c.getHostID()

    def apply_trace_row(self, step, load_scale=1.0):
        """Push trace[step] host metrics into the active containers."""
        row = self.trace[step]
        cpu = np.clip(row[0::3] * load_scale, 0.0, MAX_CPU) * IPS_RATIO
        ram = row[1::3] * RAM_FACTOR
        disk = np.clip(row[2::3], 0.0, MAX_DISK)
        for i, (ips, rmod, dmod) in enumerate(self._container_reqs.values()):
            ips.set_trace_row(cpu[i])
            rmod.set_trace_row(ram[i])
            dmod.set_trace_row(disk[i])

    def make_task(self, env, host_slot, step, load_scale=1.0, cid=None, slot=None):
        """Build a fresh task container for host_slot at trace row `step`."""
        from simulator.container.Container import Container
        row = self.trace[step]
        cpu = np.clip(row[0::3] * load_scale, 0.0, MAX_CPU) * IPS_RATIO
        ram = row[1::3] * RAM_FACTOR
        disk = np.clip(row[2::3], 0.0, MAX_DISK)
        ipsm = _TraceContainerCPU(cpu[host_slot], max(cpu[host_slot] * 1.15, 4.0), TASK_DURATION, TASK_DURATION)
        rmm = _TraceContainerRAM(ram[host_slot], 1.0, 1.0)
        dmm = _TraceContainerDisk(disk[host_slot], 1.0, 1.0)
        if cid is None:
            cid = self.creation_id
            self.creation_id += 1
        if slot is None:
            slot = len(env.containerlist)
        container = Container(slot, cid, step, ipsm, rmm, dmm, env, HostID=-1)
        ipsm.start_task(cpu[host_slot], TASK_DURATION)  # needs container.env set
        return cid, container

    def create_next_tasks(self, env, step, load_scale=1.0):
        """Refill container slots whose task just completed (slot is None).

        The replacement task is placed directly on the slot's last known
        host (self._slot_hosts), so the host's load stays continuous; the
        scheduler may then migrate it.  Returns the list of new containers.
        """
        spawned = []
        for i, c in enumerate(env.containerlist):
            if c is None or not c.active:
                host = self._slot_hosts[i] if i < len(self._slot_hosts) else -1
                if host < 0 or host >= len(env.hostlist):
                    host = i % len(env.hostlist)
                cid, container = self.make_task(env, host, step, load_scale, slot=i)
                container.hostid = host
                env.containerlist[i] = container
                self._slot_hosts[i] = host
                self.createdContainers.append((cid, step, container.ipsmodel,
                                               container.rammodel, container.diskmodel))
                self.deployedContainers.append(False)
                spawned.append(container)
        return spawned

    def generateNewContainers(self, interval):
        # Replacement tasks are created by the driver loop (it knows which
        # slots completed); nothing is spawned here.
        return []

    # --- replay-scheduling helpers -------------------------------------

    @classmethod
    def load_replay_schedule(cls):
        return np.load(SCHEDULE_SERIES_PATH)

    @classmethod
    def replay_decision(cls, schedule, env, container_alloc=None):
        """(cid, host) migration decisions from the argmax of schedule[idx]."""
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
