import unittest
import numpy as np
import torch

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol025_model import (
    RegisteredFixedResidualBank, causal_common_features,
    RESIDUAL_INPUT_DIM,
)


class TestProtocol025Model(unittest.TestCase):
    def test_shared_prefix_initialization_is_byte_identical(self):
        b4 = RegisteredFixedResidualBank(4)
        b5 = RegisteredFixedResidualBank(5)
        b8 = RegisteredFixedResidualBank(8)
        for i in range(4):
            for p4, p5, p8 in zip(b4.experts[i].parameters(), b5.experts[i].parameters(), b8.experts[i].parameters()):
                self.assertTrue(torch.equal(p4, p5))
                self.assertTrue(torch.equal(p4, p8))
            self.assertTrue(torch.equal(b4.router.weight[i], b5.router.weight[i]))
            self.assertTrue(torch.equal(b4.router.weight[i], b8.router.weight[i]))
            self.assertTrue(torch.equal(b4.router.bias[i], b5.router.bias[i]))
            self.assertTrue(torch.equal(b4.router.bias[i], b8.router.bias[i]))

    def test_top5_router_has_exactly_five_nonzero_probabilities(self):
        bank = RegisteredFixedResidualBank(8, topk=5)
        z = torch.randn(3, 16, RESIDUAL_INPUT_DIM)
        _, p = bank(z)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.equal((p > 0).sum(-1), torch.full((3,16), 5, dtype=torch.long)))

    def test_common_features_use_current_and_past_pressure_only(self):
        # Replay-style normalized host window [B,16,12,7]. Use unit scales.
        x = torch.zeros(1, 16, 12, 7)
        for t in range(12):
            x[0, :, t, 0] = float(t)       # CPU
            x[0, :, t, 1] = float(2*t)     # RAM-size
            x[0, :, t, 4] = float(3*t)     # Disk-size
            x[0, :, t, 2] = 9999.0         # RAM read: must not enter pressure
            x[0, :, t, 3] = 9999.0
            x[0, :, t, 5] = 9999.0
            x[0, :, t, 6] = 9999.0
        caps = torch.ones(1, 12, 16, 3)
        out = causal_common_features(x, {"capacities": caps}, np.ones((16,7),np.float32), np.ones(7,np.float32))
        self.assertEqual(tuple(out.shape), (1,16,9))
        # t=11: pressure [11,22,33], delta1 [1,2,3], slope4 [1,2,3].
        expected = torch.tensor([11.,22.,33.,1.,2.,3.,1.,2.,3.])
        self.assertTrue(torch.allclose(out[0,0], expected))

    def test_dynamic_bank_wraps_registered_four_expert_source(self):
        source = RegisteredFixedResidualBank(4)
        dynamic = DynamicResidualBank(source, max_experts=8, ramp_updates=8)
        self.assertEqual(dynamic.hidden, RESIDUAL_INPUT_DIM)
        self.assertEqual(dynamic.ids, ['0','1','2','3'])
        z = torch.randn(2,16,RESIDUAL_INPUT_DIM)
        corr, route = dynamic(z)
        self.assertEqual(tuple(corr.shape), (2,16,5))
        self.assertEqual(tuple(route.shape), (2,16,4))
        self.assertTrue(torch.allclose(route.sum(-1), torch.ones_like(route.sum(-1)), atol=1e-7, rtol=1e-7))


if __name__ == '__main__':
    unittest.main()
