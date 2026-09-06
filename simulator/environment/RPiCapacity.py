"""Protocol 020 — host capacity controller (capacity-control version 1).

Applies per-phase *effective resource quotas* by rewriting the live host
capacity fields that every simulator consumer reads on demand:

- Host.getIPSAvailable/getRAMAvailable/getDiskAvailable recompute
  ``capacity - current demand`` from the live container list on every call
  (simulator/host/Host.py), so writing ``host.ipsCap / host.ramCap.size /
  host.diskCap.size`` takes effect at the next placement decision without any
  cache synchronisation.
- The GOBI scheduler is deliberately NOT capacity-aware (its input features
  only contain CPU utilisation); constraints bite at execution time in
  ``getPlacementPossible``.  That is the research object of Protocol 020, not
  a bug.
- Precondition: hosts must be constructed with NO CPU/RAM/DISK_CAP_SCALE
  environment variables (physical capacities), otherwise scales would be
  applied twice.  P20 collectors assert the physical base before use.

Interpretation (plan §5.1): scales implement
    EffectiveCapacity = PhysicalCapacity x scale
reported as background-service occupancy / multi-tenant contention / resource
reservation — never as "the Pi's RAM was physically changed".
"""
import numpy as np

PHYSICAL_CPU = 4029.0
PHYSICAL_RAM = np.array([4295.0] * 8 + [8192.0] * 8, dtype=np.float64)
PHYSICAL_DISK = 32212.0


class RPiCapacity:
    def __init__(self, hostlist):
        self.hosts = list(hostlist)
        base = np.array([[float(h.ipsCap), float(h.ramCap.size),
                          float(h.diskCap.size)] for h in self.hosts])
        expected = np.stack([
            np.full(16, PHYSICAL_CPU), PHYSICAL_RAM,
            np.full(16, PHYSICAL_DISK),
        ], axis=-1)
        if not np.allclose(base, expected, rtol=0, atol=1e-9):
            raise ValueError(
                "RPiCapacity requires physically-configured hosts "
                "(no CAP_SCALE env vars); got\n%s" % base
            )
        self.base = base
        self.current_scales = (1.0, 1.0, 1.0)

    def apply(self, cpu_scale, ram_scale, disk_scale):
        for s, physical in ((cpu_scale, self.base[:, 0]),
                            (ram_scale, self.base[:, 1]),
                            (disk_scale, self.base[:, 2])):
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError("capacity scales must be finite positive: %r" % (s,))
        for h, (bips, bram, bdisk) in zip(self.hosts, self.base):
            h.ipsCap = bips * cpu_scale
            h.ramCap.size = bram * ram_scale
            h.diskCap.size = bdisk * disk_scale
        self.current_scales = (float(cpu_scale), float(ram_scale),
                               float(disk_scale))
        return self.current_scales

    def current(self):
        """Current effective capacities as a [16,3] numpy array."""
        return np.array([[h.ipsCap, h.ramCap.size, h.diskCap.size]
                         for h in self.hosts], dtype=np.float64)
