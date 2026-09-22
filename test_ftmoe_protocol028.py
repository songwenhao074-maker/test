import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol027_normal_guard import Protocol027NormalGuardLifecycle
from ftmoe_protocol028_memory_protected import Protocol028MemoryProtectedLifecycle


class ReferenceExpert(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.LayerNorm(16), nn.Linear(16, 12), nn.GELU(), nn.Linear(12, 5)
        )


class ReferenceFixedBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(16, 4)
        self.experts = nn.ModuleList([ReferenceExpert() for _ in range(4)])


class FakeSession:
    def __init__(self, bank):
        self.model = SimpleNamespace(learner=bank)
        self.cursor = 4441
        self.lifecycle_state = {}
        self.lifecycle_events = []
        self.shadow_optimizer = None

    def create_shadow_optimizer(self):
        self.shadow_optimizer = object()


def bank_hash(bank):
    digest = hashlib.sha256()
    for name, value in sorted(bank.state_dict().items()):
        digest.update(name.encode("utf8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(bank.topology_manifest(), sort_keys=True).encode("utf8"))
    return digest.hexdigest()


def new_bank():
    torch.manual_seed(28028)
    return DynamicResidualBank(ReferenceFixedBank(), max_experts=8, ramp_updates=8)


def fill_with_dormant_memory(bank):
    ids = []
    for _ in range(4):
        key = bank.create_shadow("0")
        bank.activate_shadow()
        bank.retire(key)
        ids.append(key)
    return ids


class TestProtocol028MemoryProtection(unittest.TestCase):
    def test_full_capacity_preserves_dormant_memory_and_skips_shadow(self):
        bank = new_bank()
        dormant = fill_with_dormant_memory(bank)
        self.assertEqual(bank.resident_count(), 8)
        session = FakeSession(bank)
        ctrl = Protocol028MemoryProtectedLifecycle(guard_anchor={})
        ctrl.specialist_memory = {
            key: {"last_causal_accept_matured": 100 + i * 10}
            for i, key in enumerate(dormant)
        }

        before_hash = bank_hash(bank)
        before_memory = deepcopy(ctrl.specialist_memory)
        before_next_id = bank.next_id
        started = ctrl._start_candidate(session, "periodic_background_budget")

        self.assertFalse(started)
        self.assertIsNone(bank.shadow_id)
        self.assertEqual(bank.resident_count(), 8)
        self.assertEqual(bank.next_id, before_next_id)
        self.assertEqual(bank_hash(bank), before_hash)
        self.assertEqual(ctrl.specialist_memory, before_memory)
        self.assertEqual(ctrl.purges, 0)
        self.assertEqual(ctrl.capacity_blocked, 1)
        self.assertEqual(ctrl.capacity_preserve_skips, 1)
        events = [e for e in ctrl.events
                  if e["kind"] == "birth_skipped_capacity_preserve_memory"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["expert_id"], dormant[0])
        self.assertEqual(events[0]["preserved_expert_ids"], dormant)
        self.assertFalse(events[0]["shadow_created"])
        self.assertFalse(events[0]["purge_performed"])

    def test_under_capacity_birth_matches_protocol027(self):
        bank27 = new_bank()
        bank28 = new_bank()
        s27, s28 = FakeSession(bank27), FakeSession(bank28)
        c27 = Protocol027NormalGuardLifecycle(guard_anchor={})
        c28 = Protocol028MemoryProtectedLifecycle(guard_anchor={})
        c27.route_mass = {"0": 4.0, "1": 1.0}
        c28.route_mass = deepcopy(c27.route_mass)

        self.assertTrue(c27._start_candidate(s27, "periodic_background_budget"))
        self.assertTrue(c28._start_candidate(s28, "periodic_background_budget"))
        self.assertEqual(c27.candidate_id, c28.candidate_id)
        self.assertEqual(c27.candidate_parent_id, c28.candidate_parent_id)
        self.assertEqual(bank27.topology_manifest(), bank28.topology_manifest())
        for name, value in bank27.state_dict().items():
            self.assertTrue(torch.equal(value, bank28.state_dict()[name]), name)

    def test_reactivation_moves_resident_memory_without_capacity_growth(self):
        bank = new_bank()
        dormant = fill_with_dormant_memory(bank)
        before = bank.resident_count()
        bank.reactivate(dormant[0])
        self.assertEqual(before, 8)
        self.assertEqual(bank.resident_count(), before)
        self.assertLessEqual(bank.resident_count(), bank.max_experts)
        self.assertIn(dormant[0], bank.ids)
        self.assertNotIn(dormant[0], bank.dormant_experts)

    def test_registration_freezes_single_change_and_source_artifact(self):
        reg = json.loads(Path(
            "artifacts/ftmoe_online/protocol_028/registration.json"
        ).read_text(encoding="utf8"))
        self.assertEqual(reg["protocol"], "028")
        self.assertEqual(reg["comparators"], ["C_fixed5", "D_memory_protected"])
        self.assertEqual(reg["seeds"], {"replay": 700, "model": 1})
        src = reg["source_data"]
        self.assertEqual(src["frozen_run_id"], 35682811782)
        self.assertEqual(src["frozen_artifact_id"], 10675401651)
        self.assertEqual(
            src["frozen_artifact_digest"],
            "sha256:111a4c5fc5c508a9e169fdbffe823ee36dc66c037c1775a6aa1abdbaba31c0a1",
        )
        self.assertEqual(
            reg["scientific_change"]["only_change"],
            "skip_birth_at_full_resident_capacity_preserve_accepted_memory",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
