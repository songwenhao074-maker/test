"""Protocol 020 — cohort-constrained adapted BWGD2 workload.

Same demand adapter and synthetic disk law as AdaptedBWGD2 (protocol 016
contract, used by protocol 019), but the VM arrival pool is one pre-registered
source-disjoint cohort from artifacts/ftmoe_online/protocol_020/vm_split.json
(see build_ftmoe_protocol020_vm_split.py).  Each replay stream therefore draws
arrivals from a different VM cohort: a true source-disjoint split (019 problem
P11: plain BWGD2 possible_indices is dataset-static across all seeds).

The BWGD2.__init__ full-pool scan is intentionally bypassed: attributes are
initialized identically, but possible_indices comes from the cohort file.
"""
import json
from pathlib import Path

import numpy as np

from .Workload import Workload
from .BitbrainWorkloadAdapted import AdaptedBWGD2, LAW_PATH, TrainingMarkovDisk

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"

# Protocol-020 demand-adapter knobs (direction P17-b).  Defaults reproduce the
# 016/019 contract exactly (cpu clip [2,1860], ram x2, disk x1); only this
# workload subclass reads them.
DEFAULT_ADAPTER = {
    "cpu_lower": 2.0, "cpu_upper": 1860.0, "ram_mult": 2.0, "disk_mult": 1.0,
}


class _ScaledMarkovDisk(TrainingMarkovDisk):
    """TrainingMarkovDisk whose occupancy values are scaled by disk_mult."""

    def __init__(self, law, replay_seed, creation_id, disk_mult):
        super().__init__(law, replay_seed, creation_id)
        self.disk_mult = float(disk_mult)

    def disk(self):
        value, read, write = super().disk()
        return value * self.disk_mult, read, write


class Protocol020AdaptedBWGD2(AdaptedBWGD2):
    def __init__(self, mean, sigma, replay_seed, cohort="train", split_path=SPLIT_PATH,
                 adapter=None):
        # Replicate BWGD2.__init__ attribute state WITHOUT the static full-pool
        # CSV scan (range(1, 500) pandas reads); cohort ids are validated below.
        Workload.__init__(self)
        self.mean = mean
        self.sigma = sigma
        self.dataset_path = "simulator/workload/datasets/bitbrain/"
        self.disk_sizes = [1, 2, 3]
        self.meanSLA = 20
        self.sigmaSLA = 3
        self.split = json.loads(Path(split_path).read_text(encoding="utf8"))
        if self.split.get("protocol") != "020" or "cohorts" not in self.split:
            raise ValueError("Invalid protocol-020 vm_split file: %s" % split_path)
        ids = self.split["cohorts"].get(cohort)
        if not ids:
            raise ValueError("Unknown cohort %r in vm_split (have %s)"
                             % (cohort, sorted(self.split["cohorts"])))
        missing = [i for i in ids if not Path(self.dataset_path + "rnd/%d.csv" % i).is_file()]
        if missing:
            raise FileNotFoundError("Cohort VM traces missing: %s" % missing)
        self.cohort = cohort
        self.possible_indices = [int(i) for i in ids]
        self.replay_seed = replay_seed
        self.disk_law = json.loads(LAW_PATH.read_text(encoding="utf8"))
        merged = dict(DEFAULT_ADAPTER, **(adapter or {}))
        for key in ("cpu_lower", "cpu_upper", "ram_mult", "disk_mult"):
            try:
                value = float(merged[key])
            except (TypeError, ValueError):
                raise ValueError("Invalid adapter %s=%r" % (key, merged.get(key)))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("adapter %s must be finite positive: %r"
                                 % (key, value))
        self.adapter = merged

    def adapt_new_tasks(self, first):
        """Parameterized version of the 016 demand transform; at default
        adapter values the output equals the 016 transform exactly."""
        a = self.adapter
        for i in range(first, len(self.createdContainers)):
            cid, interval, ips, ram, _ = self.createdContainers[i]
            if ips.completedInstructions or ips.totalInstructions:
                raise AssertionError("Task already executed before adaptation")
            raw = np.asarray(ips.ips_list, dtype=float)
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("Invalid CPU demand")
            ips.ips_list = np.where(
                raw > 0, np.clip(raw, a["cpu_lower"], a["cpu_upper"]), 0.).tolist()
            raw_max = float(ips.max_ips)
            mapped_max = float(np.clip(raw_max, a["cpu_lower"], a["cpu_upper"])) \
                if raw_max > 0 else 0.
            ips.max_ips = max(mapped_max, max(ips.ips_list))
            ram.size_list = (np.asarray(ram.size_list, dtype=float)
                             * a["ram_mult"]).tolist()
            ram.read_list = [1.] * len(ram.read_list)
            ram.write_list = [1.] * len(ram.write_list)
            disk = _ScaledMarkovDisk(self.disk_law, self.replay_seed, cid,
                                     a["disk_mult"])
            self.createdContainers[i] = (cid, interval, ips, ram, disk)
