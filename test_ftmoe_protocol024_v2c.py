import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol024_v2c import ReplacementConsistentLifecycle, V2C_DEFAULT


class _Source(nn.Module):
    def __init__(self, hidden=2, out=5):
        super().__init__()
        self.router = nn.Linear(hidden, 4)
        self.experts = nn.ModuleList([nn.Linear(hidden, out) for _ in range(4)])
        with torch.no_grad():
            self.router.weight.zero_(); self.router.bias.copy_(torch.tensor([40.0, -40.0, 15.0, -15.0]))
            for i, expert in enumerate(self.experts):
                expert.weight.zero_(); expert.bias.fill_(float(i - 2))


def _session(bank):
    return SimpleNamespace(model=SimpleNamespace(learner=bank), optimizer_archive={})


class TestProtocol024V2C(unittest.TestCase):
    def setUp(self):
        self.ctrl = ReplacementConsistentLifecycle(guard_anchor={}, config=V2C_DEFAULT)
        self.z = torch.tensor([[0.3, -0.2], [-1.0, 0.7]], dtype=torch.float32)

    def test_registered_policy_unchanged_except_replacement(self):
        self.assertEqual(V2C_DEFAULT['shadow_budget_version'], 'buffer128_4x4')
        self.assertEqual(V2C_DEFAULT['crossfade_prediction_intervals'], 8)
        self.assertEqual(V2C_DEFAULT['reuse_validation_intervals'], 16)
        self.assertTrue(V2C_DEFAULT['replacement_consistent_birth'])

    def test_no_old_specialist_preview_equals_final_deployment(self):
        bank = DynamicResidualBank(_Source(), max_experts=8)
        sid = bank.create_shadow('0')
        with torch.no_grad():
            bank.shadow_experts[sid].bias.fill_(3.25)
            bank.shadow_router_biases[sid].fill_(32.0)
        preview, _ = self.ctrl._birth_target_preview(_session(bank), self.z)
        actual = copy.deepcopy(bank); new = actual.activate_shadow(); actual.set_ramp(new, 1.0)
        deployed, _ = actual(self.z)
        self.assertTrue(torch.allclose(preview, deployed, atol=2e-6, rtol=2e-6))

    def test_old_specialist_is_excluded_and_replacement_matches(self):
        bank = DynamicResidualBank(_Source(), max_experts=8)
        old = bank.create_shadow('0'); bank.activate_shadow(); bank.set_ramp(old, 1.0)
        with torch.no_grad():
            bank.experts[old].bias.fill_(10.0); bank.router_biases[old].fill_(35.0)
        new = bank.create_shadow('1')
        with torch.no_grad():
            bank.shadow_experts[new].bias.fill_(1.0); bank.shadow_router_biases[new].fill_(34.0)
        legacy, _ = bank.preview_with_shadow(self.z, shadow_ramp=1.0)
        replacement, _ = self.ctrl._birth_target_preview(_session(bank), self.z)
        actual = copy.deepcopy(bank); accepted = actual.activate_shadow(); actual.set_ramp(accepted, 1.0); actual.set_ramp(old, 0.0); actual.retire(old)
        deployed, _ = actual(self.z)
        self.assertFalse(torch.allclose(legacy, deployed, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(replacement, deployed, atol=2e-6, rtol=2e-6))

    def test_dormant_reuse_preview_matches_reactivation(self):
        bank = DynamicResidualBank(_Source(), max_experts=8)
        key = bank.create_shadow('2'); bank.activate_shadow(); bank.set_ramp(key, 1.0); bank.retire(key)
        with torch.no_grad():
            bank.dormant_experts[key].bias.fill_(4.0); bank.dormant_router_biases[key].fill_(31.0)
        preview, _ = self.ctrl._preview_specialist(_session(bank), key, self.z)
        actual = copy.deepcopy(bank); actual.reactivate(key); actual.set_ramp(key, 1.0)
        deployed, _ = actual(self.z)
        self.assertTrue(torch.allclose(preview, deployed, atol=2e-6, rtol=2e-6))

    def test_due_conservation_shape(self):
        rows = self.ctrl.due_conservation()
        for name in ('birth', 'reuse'):
            self.assertEqual(rows[name]['due'], 0)
            self.assertTrue(rows[name]['conserved'])


if __name__ == '__main__':
    unittest.main()
