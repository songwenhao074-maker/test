"""Protocol 019 S7 — controlled-drift phase workload (registered, user-approved).

Subclasses the protocol-016 adapted BWGD2 (same demand adapter / disk law)
and switches the *arrival VM pool* between pre-registered VM groups at fixed
interval boundaries:

    Phase 0  cpu   (low-memory / low-cpu-ratio VMs)
    Phase 1  mixed
    Phase 2  ram   (high-memory VMs)
    Phase 3  cpu   (recurrence of phase 0: tests expert retire/reactivate)

VM groups are defined purely from training/development workload statistics
(artifacts/ftmoe_online/protocol_019/s7_vm_groups.json) — never from model
results (The Plan §13.2).  No existing simulator file is modified; old
protocol streams keep their byte-identical provenance.
"""
import json
from pathlib import Path

from .BitbrainWorkloadAdapted import AdaptedBWGD2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUPS = ROOT / "artifacts/ftmoe_online/protocol_019/s7_vm_groups.json"
DEFAULT_SCHEDULE = ("cpu", "mixed", "ram", "cpu")


class PhaseAdaptedBWGD2(AdaptedBWGD2):
    """AdaptedBWGD2 whose arrival pool follows a registered phase schedule."""

    def __init__(self, mean, sigma, replay_seed, groups_path=DEFAULT_GROUPS,
                 schedule=DEFAULT_SCHEDULE, phase_len=500):
        super().__init__(mean, sigma, replay_seed)
        groups = json.loads(Path(groups_path).read_text(encoding="utf8"))["groups"]
        self.group_pools = {name: sorted(indices) for name, indices in groups.items()}
        for name in schedule:
            if name not in self.group_pools:
                raise ValueError(f"unknown phase group: {name}")
        self.phase_schedule = list(schedule)
        self.phase_len = int(phase_len)
        self.current_phase = 0
        self.possible_indices = list(self.group_pools[self.phase_schedule[0]])

    def apply_phase(self, interval_index):
        """Switch the arrival pool for ``interval_index`` (0-based step).

        Returns (previous_phase_name, new_phase_name) on a switch, else None.
        """
        phase = min(int(interval_index) // self.phase_len, len(self.phase_schedule) - 1)
        if phase == self.current_phase:
            return None
        previous = self.current_phase
        self.current_phase = phase
        self.possible_indices = list(self.group_pools[self.phase_schedule[phase]])
        return self.phase_schedule[previous], self.phase_schedule[phase]

    def phase_name(self, interval_index):
        phase = min(int(interval_index) // self.phase_len, len(self.phase_schedule) - 1)
        return self.phase_schedule[phase]
