import unittest
import torch
from torch import nn

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank


class ReferenceExpert(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(),
                         nn.Linear(32, 5))


class ReferenceFixedBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(64, 4)
        self.experts = nn.ModuleList([ReferenceExpert() for _ in range(4)])

    def forward(self, z):
        p = torch.softmax(self.router(z), dim=-1)
        out = torch.stack([e(z) for e in self.experts], dim=-2)
        return (p.unsqueeze(-1) * out).sum(-2), p


class TestDynamicResidualBank(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24024)
        self.source = ReferenceFixedBank()
        self.dynamic = DynamicResidualBank(self.source, max_experts=8, ramp_updates=4)
        self.z = torch.randn(11, 64)

    def test_t0_is_equivalent_to_fixed_bank(self):
        expected_correction, expected_p = self.source(self.z)
        got_correction, got_p = self.dynamic(self.z)
        self.assertTrue(torch.allclose(expected_correction, got_correction,
                                       atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(expected_p, got_p, atol=1e-7, rtol=1e-6))
        self.assertEqual(self.dynamic.active_parameter_count(),
                         sum(p.numel() for p in self.source.parameters()))

    def test_shadow_is_not_deployed(self):
        before, _ = self.dynamic(self.z)
        key = self.dynamic.create_shadow("0")
        with torch.no_grad():
            for p in self.dynamic.shadow_parameters():
                p.add_(10.0)
        after, _ = self.dynamic(self.z)
        self.assertTrue(torch.allclose(before, after, atol=1e-7, rtol=1e-6))
        self.assertEqual(self.dynamic.shadow_id, key)

    def test_activation_at_ramp_zero_is_output_continuous(self):
        before, _ = self.dynamic(self.z)
        key = self.dynamic.create_shadow("0")
        with torch.no_grad():
            for p in self.dynamic.shadow_experts[key].parameters():
                p.add_(0.3)
            self.dynamic.shadow_router_biases[key].add_(2.0)
        self.dynamic.activate_shadow()
        after, _, routing = self.dynamic(self.z, return_routing=True)
        self.assertEqual(routing["ramps"][key], 0.0)
        self.assertTrue(torch.allclose(before, after, atol=2e-7, rtol=2e-6))
        self.dynamic.ramp_step()
        introduced, _ = self.dynamic(self.z)
        self.assertFalse(torch.allclose(before, introduced, atol=1e-7, rtol=1e-6))

    def test_retire_and_reactivate_keep_same_id(self):
        key = self.dynamic.create_shadow("1")
        self.dynamic.activate_shadow()
        for _ in range(4):
            self.dynamic.ramp_step()
        self.assertIn(key, self.dynamic.ids)
        self.dynamic.retire(key)
        self.assertNotIn(key, self.dynamic.ids)
        self.assertIn(key, self.dynamic.dormant_experts)
        self.dynamic.reactivate(key)
        self.assertIn(key, self.dynamic.ids)
        self.assertEqual(self.dynamic.ramp[key], 0.0)

    def test_capacity_is_enforced(self):
        for _ in range(4):
            self.dynamic.create_shadow("0")
            self.dynamic.activate_shadow()
        self.assertEqual(len(self.dynamic.ids), 8)
        with self.assertRaises(RuntimeError):
            self.dynamic.create_shadow("0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
