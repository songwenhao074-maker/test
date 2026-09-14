import unittest

import torch

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol024_session import export_dynamic_bank, restore_dynamic_bank
from test_ftmoe_protocol024_dynamic_residual import ReferenceFixedBank


class TestProtocol024NextRoundDynamic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24025)
        self.source = ReferenceFixedBank()
        self.bank = DynamicResidualBank(self.source, max_experts=8, ramp_updates=4)
        self.z = torch.randn(4, 16, 64)

    def test_gap_id_restore_after_rejected_shadow(self):
        rejected = self.bank.create_shadow("0")
        self.assertEqual(rejected, "4")
        self.bank.discard_shadow()
        child = self.bank.create_shadow("0")
        self.assertEqual(child, "5")
        self.bank.activate_shadow()
        self.bank.ramp_step()
        before, route_before = self.bank(self.z)
        snapshot = export_dynamic_bank(self.bank)
        restored = restore_dynamic_bank(self.source, snapshot)
        after, route_after = restored(self.z)
        self.assertEqual(restored.ids, ["0", "1", "2", "3", "5"])
        self.assertEqual(restored.next_id, 6)
        self.assertEqual(restored.topology_manifest(), self.bank.topology_manifest())
        self.assertEqual(restored.behavior_state_hash(), self.bank.behavior_state_hash())
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(route_before, route_after))

    def test_multiple_gaps_mixed_dormant_shadow_and_partial_ramp_restore(self):
        for expected in ("4", "5"):
            self.assertEqual(self.bank.create_shadow("1"), expected)
            self.bank.discard_shadow()
        child = self.bank.create_shadow("2")
        self.assertEqual(child, "6")
        self.bank.activate_shadow()
        self.bank.ramp_step()
        self.bank.retire("1")
        shadow = self.bank.create_shadow("0")
        self.assertEqual(shadow, "7")
        with torch.no_grad():
            self.bank.shadow_router_biases[shadow].add_(3.0)
        snapshot = export_dynamic_bank(self.bank)
        restored = restore_dynamic_bank(self.source, snapshot)
        self.assertEqual(restored.ids, self.bank.ids)
        self.assertEqual(list(restored.dormant_experts.keys()), ["1"])
        self.assertEqual(restored.shadow_id, "7")
        self.assertEqual(restored.ramp["6"], 0.25)
        self.assertEqual(restored.behavior_state_hash(), self.bank.behavior_state_hash())

    def test_behavior_hash_changes_when_only_ramp_changes(self):
        child = self.bank.create_shadow("0")
        self.bank.activate_shadow()
        before = self.bank.behavior_state_hash()
        self.bank.ramp_step()
        after = self.bank.behavior_state_hash()
        self.assertNotEqual(before, after)

    def test_extreme_shadow_logit_at_ramp_zero_is_masked_before_softmax(self):
        before, _ = self.bank(self.z)
        child = self.bank.create_shadow("0")
        with torch.no_grad():
            self.bank.shadow_router_weights[child].zero_()
            self.bank.shadow_router_biases[child].fill_(1000.0)
            for p in self.bank.shadow_experts[child].parameters():
                p.add_(0.2)
        self.bank.activate_shadow()
        after, route = self.bank(self.z)
        self.assertTrue(torch.isfinite(after).all())
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(route[..., -1], torch.zeros_like(route[..., -1])))

    def test_purge_frees_resident_capacity_and_next_id_stays_monotonic(self):
        for _ in range(4):
            child = self.bank.create_shadow("0")
            self.bank.activate_shadow()
            self.bank.set_ramp(child, 1.0)
        self.assertEqual(self.bank.resident_count(), 8)
        retired = self.bank.ids[-1]
        self.bank.retire(retired)
        self.assertEqual(self.bank.resident_count(), 8)
        with self.assertRaises(RuntimeError):
            self.bank.create_shadow("0")
        old_next = self.bank.next_id
        self.bank.purge(retired)
        self.assertEqual(self.bank.resident_count(), 7)
        newborn = self.bank.create_shadow("0")
        self.assertEqual(int(newborn), old_next)
        self.assertEqual(self.bank.next_id, old_next + 1)

    def test_dormant_parameters_are_frozen_and_reactivated_same_id(self):
        child = self.bank.create_shadow("0")
        self.bank.activate_shadow()
        self.bank.set_ramp(child, 1.0)
        self.bank.retire(child)
        self.assertTrue(all(not p.requires_grad
                            for p in self.bank.dormant_experts[child].parameters()))
        self.assertFalse(self.bank.dormant_router_weights[child].requires_grad)
        self.bank.reactivate(child)
        self.assertEqual(self.bank.ids[-1], child)
        self.assertEqual(self.bank.ramp[child], 0.0)
        self.assertTrue(all(p.requires_grad for p in self.bank.experts[child].parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
