"""Protocol-025 service-turnover workload with six preregistered physical responses.

The six service laws operate only on task demand trajectories. Service IDs,
phase IDs and future switch times are audit-only. Fault labels remain generated
downstream from executed aggregate demand divided by physical/effective host
capacity. Age zero is always byte-identical to the familiar adapted Bitbrain
demand, so admission behavior is unchanged.

Revision data_revision_001 (registered before any Protocol-025 model result)
fixes two source/registration mismatches only: S4 now applies its registered
CPU decay floor, and S5 truly repeats its registered 36-interval release cycle
with release_fraction=0.42 affecting the trajectory.
"""
from __future__ import annotations

import math
import numpy as np

from .BitbrainWorkloadProtocol020 import Protocol020AdaptedBWGD2, SPLIT_PATH, _ScaledMarkovDisk
from simulator.environment.RPiCapacity import PHYSICAL_CPU, PHYSICAL_RAM, PHYSICAL_DISK

SERVICE_IDS = ("S1", "S2", "S3", "S4", "S5", "S6")
SERVICE_MECHANISM_IDS = {name: i for i, name in enumerate(SERVICE_IDS)}
SERVICE_SEEDS = {name: 25011 + i for i, name in enumerate(SERVICE_IDS)}
EVENT_PROBABILITY = 0.30
OBSERVABLE_HISTORY = 12
DATA_REVISION_ID = "data_revision_001"
FORBIDDEN_MODEL_INPUTS = (
    "service_id", "service_law_id", "mode_id", "phase_id", "event_id",
    "known_future_switch_time", "future_label", "future_demand", "future_capacity",
)


def _rise_hold_decay(rise, hold, decay):
    rise = int(rise); hold = int(hold); decay = int(decay)
    up = np.linspace(0.0, 1.0, rise + 1, dtype=np.float64)[:-1]
    mid = np.ones(hold, dtype=np.float64)
    down = np.linspace(1.0, 0.0, decay + 1, dtype=np.float64)[1:]
    out = np.concatenate([up, mid, down]); out[0] = 0.0
    return out.tolist()


def _growth_plateau_release(growth, plateau, release):
    growth = int(growth); plateau = int(plateau); release = int(release)
    up = np.linspace(0.0, 1.0, growth + 1, dtype=np.float64)[:-1]
    mid = np.ones(plateau, dtype=np.float64)
    down = np.linspace(1.0, 0.0, release + 1, dtype=np.float64)[1:]
    out = np.concatenate([up, mid, down]); out[0] = 0.0
    return out.tolist()


def _accumulate_flush(accumulate, flush):
    up = np.linspace(0.0, 1.0, int(accumulate) + 1, dtype=np.float64)[:-1]
    down = np.linspace(1.0, 0.0, int(flush) + 1, dtype=np.float64)[1:]
    out = np.concatenate([up, down]); out[0] = 0.0
    return out.tolist()


def _warmup_decay(warmup, half_life, tail_half_lives=3):
    warmup = int(warmup); half_life = int(half_life)
    up = np.linspace(0.0, 1.0, warmup + 1, dtype=np.float64)[:-1]
    tail_len = half_life * int(tail_half_lives)
    tail = np.asarray([2.0 ** (-k / float(half_life)) for k in range(tail_len)], dtype=np.float64)
    out = np.concatenate([up, tail]); out[0] = 0.0
    return out.tolist()


def _periodic_growth_release(length, period, growth, release, release_fraction):
    """Repeated causal S5 excess shape; release_fraction is excess removed.

    The first cycle starts from zero excess at age0.  At each release 42% of
    the induced peak excess is removed, so 58% remains for the next cycle.
    This interpretation is frozen in data_revision_001.
    """
    length = int(length); period = int(period); growth = int(growth); release = int(release)
    release_fraction = float(release_fraction)
    if length < 1 or period < 1 or growth < 1 or release < 1:
        raise ValueError("invalid periodic response geometry")
    if growth + release > period or not 0.0 <= release_fraction <= 1.0:
        raise ValueError("invalid periodic release parameters")
    hold = period - growth - release
    remain = 1.0 - release_fraction
    values = []
    cycle = 0
    while len(values) < length:
        start = 0.0 if cycle == 0 else remain
        up = np.linspace(start, 1.0, growth + 1, dtype=np.float64)[:-1]
        mid = np.ones(hold, dtype=np.float64)
        down = np.linspace(1.0, remain, release + 1, dtype=np.float64)[1:]
        values.extend(np.concatenate([up, mid, down]).tolist())
        cycle += 1
    out = np.asarray(values[:length], dtype=np.float64)
    out[0] = 0.0
    return out.tolist()


def _competition(backlog_build, competition, coupling):
    build = int(backlog_build); comp = int(competition); coupling = float(coupling)
    backlog = np.linspace(0.0, 1.0, build + 1, dtype=np.float64)[:-1]
    backlog_tail = np.linspace(1.0, 0.25, comp, dtype=np.float64)
    disk = np.concatenate([backlog, backlog_tail]); disk[0] = 0.0
    cpu_head = np.linspace(0.0, coupling, build + 1, dtype=np.float64)[:-1]
    cpu_tail = np.linspace(max(coupling, 0.45), 1.0, max(2, comp // 2), dtype=np.float64)
    cpu_decay = np.linspace(1.0, 0.0, comp - cpu_tail.size, dtype=np.float64) if comp - cpu_tail.size > 0 else np.zeros(0)
    cpu = np.concatenate([cpu_head, cpu_tail, cpu_decay]); cpu[0] = 0.0
    return cpu.tolist(), disk.tolist()


def s4_cpu_target_fraction(age):
    """Registered S4 target fraction after the 16-interval warmup."""
    age = int(age)
    warmup = 16; half_life = 12; peak = 1.16; floor = 0.55
    if age < warmup:
        return None
    k = age - warmup
    return float(floor + (peak - floor) * (2.0 ** (-k / float(half_life))))


S1_CPU = _rise_hold_decay(6, 10, 8)
S2_RAM = _growth_plateau_release(24, 12, 10)
S3_DISK = _accumulate_flush(14, 12)
S4_DECAY = _warmup_decay(16, 12, 3)
S5_FIRST_CYCLE = _periodic_growth_release(36, 36, 22, 6, 0.42)
S6_CPU, S6_DISK = _competition(12, 14, 0.35)

# Targets correspond to the preregistered capacity fractions. For RAM we use
# the 4 GiB Pi capacity as the conservative physical target because tasks are
# assigned only after admission and cannot know future host type causally.
RAM4 = float(PHYSICAL_RAM[0])
SERVICE_LAWS = {
    "S1": {
        "name": "cpu_short_pulse", "mechanism_id": 0, "seed": SERVICE_SEEDS["S1"],
        "probability": EVENT_PROBABILITY,
        "cpu_shape": S1_CPU, "cpu_floor": float(PHYSICAL_CPU * 1.18), "cpu_upper": float(PHYSICAL_CPU * 1.24),
        "ram_shape": (np.asarray(S1_CPU) * 0.10).tolist(), "ram_floor": 1800.0, "ram_upper": 2200.0,
        "disk_shape": [0.0] * len(S1_CPU), "disk_retained_peak": 0.0, "disk_cap": float(PHYSICAL_DISK * 1.25),
        "minimum_trace": len(S1_CPU), "history_signal": "positive cpu slope",
        "registered_parameters": {"rise_intervals":6,"hold_intervals":10,"decay_intervals":8,"peak_capacity_fraction":1.18},
    },
    "S2": {
        "name": "ram_growth", "mechanism_id": 1, "seed": SERVICE_SEEDS["S2"],
        "probability": EVENT_PROBABILITY,
        "cpu_shape": (np.asarray(S2_RAM) * 0.22).tolist(), "cpu_floor": 2600.0, "cpu_upper": 3400.0,
        "ram_shape": S2_RAM, "ram_floor": float(RAM4 * 1.12), "ram_upper": float(RAM4 * 1.20),
        "disk_shape": [0.0] * len(S2_RAM), "disk_retained_peak": 0.0, "disk_cap": float(PHYSICAL_DISK * 1.25),
        "minimum_trace": len(S2_RAM), "history_signal": "ram level plus positive slope",
        "registered_parameters": {"growth_intervals":24,"plateau_intervals":12,"release_intervals":10,"peak_capacity_fraction":1.12},
    },
    "S3": {
        "name": "io_writeback", "mechanism_id": 2, "seed": SERVICE_SEEDS["S3"],
        "probability": EVENT_PROBABILITY,
        "cpu_shape": (np.asarray(S3_DISK) * 0.15).tolist(), "cpu_floor": 2400.0, "cpu_upper": 3200.0,
        "ram_shape": (np.asarray(S3_DISK) * 0.12).tolist(), "ram_floor": 1900.0, "ram_upper": 2600.0,
        "disk_shape": S3_DISK, "disk_retained_peak": float(PHYSICAL_DISK * 1.20), "disk_cap": float(PHYSICAL_DISK * 1.24),
        "minimum_trace": len(S3_DISK), "history_signal": "write backlog and write slope",
        "registered_parameters": {"accumulate_intervals":14,"flush_intervals":12,"peak_capacity_fraction":1.20,"cpu_coupling":0.15},
    },
    "S4": {
        "name": "cache_warmup_decay", "mechanism_id": 3, "seed": SERVICE_SEEDS["S4"],
        "probability": EVENT_PROBABILITY,
        "cpu_shape": S4_DECAY, "cpu_floor": float(PHYSICAL_CPU * 1.16), "cpu_upper": float(PHYSICAL_CPU * 1.22),
        "ram_shape": (np.asarray(S4_DECAY) * 0.16).tolist(), "ram_floor": 2100.0, "ram_upper": 3000.0,
        "disk_shape": (np.asarray(S4_DECAY) * 0.35).tolist(), "disk_retained_peak": float(PHYSICAL_DISK * 0.55), "disk_cap": float(PHYSICAL_DISK * 1.20),
        "minimum_trace": len(S4_DECAY), "history_signal": "high pressure with negative slope",
        "registered_parameters": {"warmup_intervals":16,"decay_half_life_intervals":12,"initial_capacity_fraction":1.16,"floor_capacity_fraction":0.55},
        "semantic_revision": DATA_REVISION_ID,
    },
    "S5": {
        "name": "periodic_workingset_release", "mechanism_id": 4, "seed": SERVICE_SEEDS["S5"],
        "probability": EVENT_PROBABILITY,
        "cpu_shape": (np.asarray(S5_FIRST_CYCLE) * 0.18).tolist(), "cpu_floor": 2500.0, "cpu_upper": 3300.0,
        "ram_shape": S5_FIRST_CYCLE, "ram_floor": float(RAM4 * 1.14), "ram_upper": float(RAM4 * 1.22),
        "disk_shape": [0.0] * len(S5_FIRST_CYCLE), "disk_retained_peak": 0.0, "disk_cap": float(PHYSICAL_DISK * 1.25),
        "minimum_trace": 36, "history_signal": "phase inferred from ram level and slope",
        "registered_parameters": {"period_intervals":36,"growth_intervals":22,"release_intervals":6,"peak_capacity_fraction":1.14,"release_fraction":0.42},
        "semantic_revision": DATA_REVISION_ID,
    },
    "S6": {
        "name": "writeback_cpu_competition", "mechanism_id": 5, "seed": SERVICE_SEEDS["S6"],
        "probability": EVENT_PROBABILITY,
        "cpu_shape": S6_CPU, "cpu_floor": float(PHYSICAL_CPU * 1.10), "cpu_upper": float(PHYSICAL_CPU * 1.18),
        "ram_shape": (np.asarray(S6_DISK) * 0.10).tolist(), "ram_floor": 1900.0, "ram_upper": 2600.0,
        "disk_shape": S6_DISK, "disk_retained_peak": float(PHYSICAL_DISK * 1.13), "disk_cap": float(PHYSICAL_DISK * 1.20),
        "minimum_trace": len(S6_DISK), "history_signal": "joint cpu/write slopes and backlog",
        "registered_parameters": {"backlog_build_intervals":12,"competition_intervals":14,"disk_peak_capacity_fraction":1.13,"cpu_peak_capacity_fraction":1.10,"coupling":0.35},
    },
}


def service_shapes(service_id, length):
    """Return response shapes for the available causal task trace length."""
    service_id = str(service_id); length = int(length); spec = SERVICE_LAWS[service_id]
    if service_id == "S5":
        ram = np.asarray(_periodic_growth_release(length, 36, 22, 6, 0.42), dtype=np.float64)
        return (ram * 0.18).tolist(), ram.tolist(), [0.0] * length
    return list(spec["cpu_shape"]), list(spec["ram_shape"]), list(spec["disk_shape"])


def service_active_at(service_id, age):
    service_id = str(service_id); age = int(age)
    if age < 0: return False
    if service_id == "S5":
        return age > 0
    spec = SERVICE_LAWS[service_id]
    return any(age < len(shape) and float(shape[age]) > 0.0
               for shape in (spec["cpu_shape"], spec["ram_shape"], spec["disk_shape"]))


def _event_rng(replay_seed, creation_id, service_id):
    return np.random.default_rng(np.random.SeedSequence([int(replay_seed), int(creation_id), int(SERVICE_SEEDS[service_id])]))


def _blend_to_target(base, shape, floor, upper):
    base = np.asarray(base, dtype=np.float64); out = base.copy(); shape = np.asarray(shape, dtype=np.float64)
    n = min(base.size, shape.size)
    if n == 0: return out
    target = np.clip(np.maximum(base[:n], float(floor)), float(floor), float(upper))
    out[:n] = base[:n] + shape[:n] * (target - base[:n]); out[0] = base[0]
    return out


def _apply_s4_cpu(familiar_cpu, shape):
    """Admission-safe warmup to 1.16x CPU, then half-life decay to 0.55x floor."""
    familiar = np.asarray(familiar_cpu, dtype=np.float64); shape = np.asarray(shape, dtype=np.float64)
    out = familiar.copy(); n = min(familiar.size, shape.size); warmup = 16
    first = min(n, warmup)
    if first:
        peak = float(PHYSICAL_CPU * 1.16)
        target = np.maximum(familiar[:first], peak)
        out[:first] = familiar[:first] + shape[:first] * (target - familiar[:first])
    for age in range(warmup, n):
        target = float(PHYSICAL_CPU * s4_cpu_target_fraction(age))
        out[age] = max(float(familiar[age]), target)
    if n: out[0] = familiar[0]
    return out


class _ServiceDisk(_ScaledMarkovDisk):
    def __init__(self, law, replay_seed, creation_id, disk_mult, retained_shape, retained_peak, cap):
        super().__init__(law, replay_seed, creation_id, disk_mult)
        self.retained_shape = tuple(float(x) for x in retained_shape); self.retained_peak = float(retained_peak); self.cap = float(cap)
    def retained(self, age):
        age = int(age)
        if age < 0 or age >= len(self.retained_shape): return 0.0
        return self.retained_peak * self.retained_shape[age]
    def disk(self):
        value, read, write = super().disk(); age = self.container.env.interval - self.container.startAt; added = self.retained(age)
        if added <= 0.0: return value, read, write
        return min(float(value) + added, self.cap), read, write


class Protocol025ServiceTurnoverBWGD2(Protocol020AdaptedBWGD2):
    """Bitbrain workload with causal service-response laws S1..S6."""
    def __init__(self, mean, sigma, replay_seed, cohort="online", split_path=SPLIT_PATH,
                 adapter=None, disk_law_path=None, active_service=None,
                 event_probability=EVENT_PROBABILITY):
        super().__init__(mean, sigma, replay_seed, cohort=cohort, split_path=split_path,
                         adapter=adapter, disk_law_path=disk_law_path)
        self._active_service = None; self.event_probability = float(event_probability)
        self.response_events = []; self._event_of = {}; self.short_trace_skips = {x:0 for x in SERVICE_IDS}
        self.set_active_service(active_service, probability=event_probability)
    @property
    def active_service(self): return self._active_service
    def set_active_service(self, service_id, probability=None):
        if service_id is not None and service_id not in SERVICE_IDS: raise ValueError("unregistered service %r" % (service_id,))
        if probability is not None:
            probability = float(probability)
            if not 0.0 <= probability <= 1.0: raise ValueError("event probability must be in [0,1]")
            self.event_probability = probability
        self._active_service = service_id
        return {"service_id":service_id,"event_probability":self.event_probability}
    def response_event_id(self, creation_id): return self._event_of.get(int(creation_id))
    def response_audit(self):
        return {"family":"protocol025_service_turnover_v1r1","data_revision_id":DATA_REVISION_ID,
                "laws":SERVICE_LAWS,"event_probability":self.event_probability,
                "observable_history":OBSERVABLE_HISTORY,"forbidden_model_inputs":list(FORBIDDEN_MODEL_INPUTS),
                "short_trace_skips":dict(self.short_trace_skips),"events":list(self.response_events)}
    def _draw_event(self, cid):
        if self._active_service is None or self.event_probability <= 0.0: return None
        rng = _event_rng(self.replay_seed, cid, self._active_service); draw = float(rng.random())
        if draw >= self.event_probability: return None
        return {"service_id":self._active_service,"draw":draw}
    def _apply_plain(self, i, cid, interval, ips, ram, raw, raw_ram):
        a=self.adapter; cpu=np.where(raw>0,np.clip(raw*a["cpu_mult"],a["cpu_lower"],a["cpu_upper"]),0.0); ips.ips_list=cpu.tolist()
        raw_max=float(ips.max_ips); scaled_max=float(np.clip(raw_max*a["cpu_mult"],a["cpu_lower"],a["cpu_upper"])) if raw_max>0 else 0.0
        ips.max_ips=max(scaled_max,float(cpu.max()) if cpu.size else 0.0); ram_scaled=raw_ram*a["ram_mult"]
        if a.get("ram_upper") is not None: ram_scaled=np.minimum(ram_scaled,a["ram_upper"])
        ram.size_list=ram_scaled.tolist(); ram.read_list=[1.0]*len(ram.read_list); ram.write_list=[1.0]*len(ram.write_list)
        disk=_ScaledMarkovDisk(self.disk_law,self.replay_seed,cid,a["disk_mult"]); self.createdContainers[i]=(cid,interval,ips,ram,disk); return cpu,ram_scaled
    def _apply_event(self, i, cid, interval, ips, ram, raw, raw_ram, envelope):
        service_id=envelope["service_id"]; spec=SERVICE_LAWS[service_id]; minimum=int(spec["minimum_trace"])
        if raw.size < minimum or raw_ram.size < minimum:
            self.short_trace_skips[service_id]+=1; self._apply_plain(i,cid,interval,ips,ram,raw,raw_ram); return False
        a=self.adapter; familiar_cpu=np.where(raw>0,np.clip(raw*a["cpu_mult"],a["cpu_lower"],a["cpu_upper"]),0.0); familiar_ram=raw_ram*a["ram_mult"]
        if a.get("ram_upper") is not None: familiar_ram=np.minimum(familiar_ram,a["ram_upper"])
        shape_len=max(int(raw.size),int(raw_ram.size)); cpu_shape,ram_shape,disk_shape=service_shapes(service_id,shape_len)
        cpu=(_apply_s4_cpu(familiar_cpu,cpu_shape) if service_id=="S4" else
             _blend_to_target(familiar_cpu,cpu_shape,spec["cpu_floor"],spec["cpu_upper"]))
        ram_scaled=_blend_to_target(familiar_ram,ram_shape,spec["ram_floor"],spec["ram_upper"])
        ips.ips_list=cpu.tolist(); raw_max=float(ips.max_ips); familiar_max=float(np.clip(raw_max*a["cpu_mult"],a["cpu_lower"],a["cpu_upper"])) if raw_max>0 else 0.0
        ips.max_ips=max(familiar_max,float(cpu.max()) if cpu.size else 0.0); ram.size_list=ram_scaled.tolist(); ram.read_list=[1.0]*len(ram.read_list); ram.write_list=[1.0]*len(ram.write_list)
        disk=_ServiceDisk(self.disk_law,self.replay_seed,cid,a["disk_mult"],disk_shape,spec["disk_retained_peak"],spec["disk_cap"])
        self.createdContainers[i]=(cid,interval,ips,ram,disk)
        event_id=len(self.response_events); event={"event_id":event_id,"family":"protocol025_service_turnover_v1r1","data_revision_id":DATA_REVISION_ID,"service_id":service_id,
            "mechanism_id":int(spec["mechanism_id"]),"creation_id":int(cid),"creation_interval":int(interval),"probability_draw":float(envelope["draw"]),
            "cpu_window":[0,len(cpu_shape)-1],"ram_window":[0,len(ram_shape)-1],"disk_window":[0,len(disk_shape)-1],
            "periodic":bool(service_id=="S5"),"period_intervals":36 if service_id=="S5" else None,
            "cpu_peak":float(cpu[:minimum].max()),"ram_peak":float(ram_scaled[:minimum].max()),"disk_retained_peak":float(spec["disk_retained_peak"]),
            "history_signal":spec["history_signal"]}
        self.response_events.append(event); self._event_of[int(cid)]=event_id; return True
    def adapt_new_tasks(self, first):
        for i in range(first,len(self.createdContainers)):
            cid,interval,ips,ram,_=self.createdContainers[i]
            if ips.completedInstructions or ips.totalInstructions: raise AssertionError("Task already executed before adaptation")
            raw=np.asarray(ips.ips_list,dtype=np.float64); raw_ram=np.asarray(ram.size_list,dtype=np.float64)
            if (not np.isfinite(raw).all() or (raw<0).any() or not np.isfinite(raw_ram).all() or (raw_ram<0).any()): raise ValueError("invalid Bitbrain demand")
            envelope=self._draw_event(cid)
            if envelope is None: self._apply_plain(i,cid,interval,ips,ram,raw,raw_ram)
            else: self._apply_event(i,cid,interval,ips,ram,raw,raw_ram,envelope)
