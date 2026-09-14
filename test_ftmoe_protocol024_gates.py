import unittest

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol024_session import (
    Protocol024Session,
    export_dynamic_bank,
    positive_only_resource_macro_f1,
    restore_dynamic_bank,
    same_host_onset_metrics,
)
from recovery.PreGANSrc.src.ftmoe_online_r1 import FixedResidualBank


class Protocol024GateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        np.random.seed(17)

    def test_dynamic_bank_checkpoint_roundtrip_preserves_topology_and_output(self):
        source = FixedResidualBank()
        bank = DynamicResidualBank(source, max_experts=8, ramp_updates=4)
        # Exercise all topology states simultaneously: one retired expert, one
        # activated child with a partial ramp, and one live shadow candidate.
        child = bank.create_shadow("0")
        bank.activate_shadow()
        bank.ramp[child] = 0.5
        bank.retire("1")
        shadow = bank.create_shadow("2")
        self.assertEqual(shadow, "5")

        with torch.no_grad():
            bank.router_weights["0"].add_(0.0123)
            bank.dormant_router_biases["1"].add_(0.0456)
            bank.shadow_router_weights[shadow].sub_(0.007)

        z = torch.randn(3, 16, 64)
        before, routing_before, meta_before = bank(z, return_routing=True)
        snapshot = export_dynamic_bank(bank)
        restored = restore_dynamic_bank(source, snapshot)
        after, routing_after, meta_after = restored(z, return_routing=True)

        self.assertEqual(bank.topology_manifest(), restored.topology_manifest())
        self.assertEqual(meta_before, meta_after)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(routing_before, routing_after))
        self.assertFalse(restored.dormant_router_weights["1"].requires_grad)

    def test_ramp_zero_activation_is_output_continuous(self):
        source = FixedResidualBank()
        bank = DynamicResidualBank(source, max_experts=8, ramp_updates=4)
        z = torch.randn(2, 16, 64)
        before, _ = bank(z)
        bank.create_shadow("0")
        bank.activate_shadow()
        after, _ = bank(z)
        self.assertTrue(torch.equal(before, after))

    def test_same_host_onset_does_not_cross_host_boundary(self):
        # Host 0 faults at t=1. Host 1 is already faulted at t=0. Flatten-then-
        # shift can leak host-1 state into host-0; temporal same-host shifting
        # must produce only one eligible positive onset here. There are exactly
        # two eligible current-normal rows before the final incomplete horizon.
        labels = np.array([
            [0, 1],
            [1, 0],
            [1, 0],
        ], dtype=np.int64)
        probability = np.array([
            [0.9, 0.1],
            [0.8, 0.1],
            [0.7, 0.1],
        ], dtype=np.float64)
        result = same_host_onset_metrics(probability, labels, horizon=1)
        self.assertEqual(result["positives"], 1)
        self.assertEqual(result["rows"], 2)
        self.assertAlmostEqual(result["ap"], 1.0, places=12)

    def test_positive_only_resource_f1_excludes_normal_rows(self):
        labels = np.array([[0, 1, 2, 3]], dtype=np.int64)
        # Deliberately arbitrary class prediction for normal row; positives are
        # all correct and should therefore have macro-F1 1.0.
        prob = np.array([[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                        dtype=np.float64)
        result = positive_only_resource_macro_f1(prob, labels)
        self.assertEqual(result["rows"], 3)
        self.assertAlmostEqual(result["macro_f1"], 1.0, places=12)

    def test_finish_settles_both_pending_tail_records(self):
        # Construct only the state finish() touches. This makes the off-by-one
        # regression deterministic and independent of the large experiment NPZ.
        session = Protocol024Session.__new__(Protocol024Session)
        session.steps = 5
        session.cursor = 5
        session.buffer_limit = 100
        raw = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 2],
            [0, 0],
            [0, 0],  # guard
        ], dtype=np.int64)
        session.bundle = {"arrays": {"raw_labels": raw}}
        session.raw_seen = raw.copy()
        session.raw_seen[-1] = -1
        session.predictions = {
            "raw_labels": np.full((5, 2), -1, dtype=np.int64),
            "labels": np.full((5, 2), -1, dtype=np.int64),
            "settled_at": np.full(5, -1, dtype=np.int64),
        }
        session.settled = np.array([True, True, True, False, False], dtype=bool)
        session.buffer = [0, 1, 2]
        session.assert_frozen = lambda: None
        Protocol024Session.finish(session)

        self.assertTrue(session.settled[3])
        self.assertTrue(session.settled[4])
        self.assertEqual(session.buffer[-2:], [3, 4])
        self.assertTrue(np.array_equal(
            session.predictions["labels"][3],
            s4.tolerance_label(session.raw_seen, 3, 5)))
        self.assertTrue(np.array_equal(
            session.predictions["labels"][4],
            s4.tolerance_label(session.raw_seen, 4, 5)))

    def test_protocol024_session_is_literal_protocol023_session_subclass(self):
        self.assertTrue(issubclass(Protocol024Session, s4.PrequentialS4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
