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

from .Workload import Workload
from .BitbrainWorkloadAdapted import AdaptedBWGD2, LAW_PATH

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"


class Protocol020AdaptedBWGD2(AdaptedBWGD2):
    def __init__(self, mean, sigma, replay_seed, cohort="train", split_path=SPLIT_PATH):
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
