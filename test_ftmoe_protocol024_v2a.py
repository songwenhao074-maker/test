import unittest

import numpy as np
import torch

from ftmoe_protocol024_v2a import (
    deployable_pressure,
    fixed_threshold_fpr,
    proposal_due,
    split_anchor_train_guard,
)


class TestProtocol024V2A(unittest.TestCase):
    def test_periodic_schedule_is_phase_independent(self):
        expected = [600, 856, 1112, 1368, 1624]
        found = [i for i in range(1, 1700) if proposal_due(i)]
        self.assertEqual(found[:5], expected)
        self.assertFalse(proposal_due(599))
        self.assertFalse(proposal_due(601))

    def test_anchor_split_is_disjoint_target_independent(self):
        n = 20
        anchor = {
            "x": torch.arange(n * 2).reshape(n, 2).float(),
            "schedule": torch.arange(n).reshape(n, 1).float(),
            "graph_x": torch.arange(n).reshape(n, 1).float(),
            "ids": torch.arange(n).reshape(n, 1),
            "before": torch.arange(n).reshape(n, 1),
            "caps": torch.ones(n, 1),
            "labels": torch.tensor([[i % 4] for i in range(n)]),
            "meta": {"source": "toy"},
        }
        train, guard = split_anchor_train_guard(anchor, modulo=5, guard_remainder=0)
        self.assertEqual(int(train["labels"].shape[0]), 16)
        self.assertEqual(int(guard["labels"].shape[0]), 4)
        self.assertFalse(train["meta"]["v2_target_dependent_split"])
        self.assertFalse(guard["meta"]["v2_target_dependent_split"])
        self.assertEqual(guard["x"][:, 0].tolist(), [0.0, 10.0, 20.0, 30.0])

    def test_fixed_fpr_uses_only_normal_rows(self):
        logits = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 3.0]])
        target = torch.tensor([0, 0, 2])
        fpr, rows = fixed_threshold_fpr(logits, target)
        self.assertEqual(rows, 2)
        self.assertAlmostEqual(fpr, 0.5)

    def test_deployable_pressure_maps_resource_sizes_not_first_three(self):
        host = np.zeros((1, 1, 7), dtype=np.float64)
        host[0, 0, 0] = 50.0
        host[0, 0, 1] = 200.0
        host[0, 0, 2] = 9999.0  # RAM read: must not be treated as capacity pressure.
        host[0, 0, 4] = 30.0
        caps = np.array([[[100.0, 400.0, 100.0]]])
        score = deployable_pressure(host, caps)
        self.assertAlmostEqual(float(score[0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
