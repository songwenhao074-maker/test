from __future__ import annotations

import math
import unittest

import torch

from ftmoe_protocol031_nonblocking_reuse import P031_CONFIG
from ftmoe_protocol032_guard_budget import (
    ABSOLUTE_FLOOR_NATS,
    NUMERICAL_TOLERANCE,
    RELATIVE_TOLERANCE,
    Protocol032GuardBudgetLifecycle,
    guard_budget_report,
)


def logits_for_normal_nll(loss, rows=8):
    """Two-class logits whose class-0 CE is the requested scalar loss."""
    p0 = math.exp(-float(loss))
    p1 = 1.0 - p0
    if not (0.0 < p1 < 1.0):
        raise ValueError(loss)
    row = torch.tensor([math.log(p0), math.log(p1)], dtype=torch.float32)
    return row.repeat(int(rows), 1)


class Protocol032GuardBudgetTests(unittest.TestCase):
    def report(self, live_loss, candidate_loss, rows=8):
        y = torch.zeros(rows, dtype=torch.long)
        return guard_budget_report(
            logits_for_normal_nll(live_loss, rows),
            logits_for_normal_nll(candidate_loss, rows),
            y,
        )

    def test_absolute_floor_changes_low_nll_decision(self):
        r = self.report(0.020, 0.025)
        self.assertTrue(r["available"])
        self.assertFalse(r["old_normal_nll_ok_direct"])
        self.assertTrue(r["normal_nll_ok_direct"])
        self.assertTrue(r["inherited_birth_relative_check_ok"])
        self.assertAlmostEqual(r["candidate_normal_nll_limit"], 0.030001, places=5)
        self.assertAlmostEqual(r["old_candidate_normal_nll_limit"], 0.020401, places=5)

    def test_birth_and_reuse_decisions_are_exactly_equivalent(self):
        for live in (0.005, 0.02, 0.5, 1.0):
            limit = live + max(RELATIVE_TOLERANCE * live, ABSOLUTE_FLOOR_NATS) + NUMERICAL_TOLERANCE
            for delta in (-2e-5, 2e-5):
                r = self.report(live, max(1e-8, limit + delta))
                self.assertEqual(
                    r["normal_nll_ok_direct"],
                    r["relative_loss_increase"] <= RELATIVE_TOLERANCE,
                )

    def test_old_rule_retained_for_audit(self):
        r = self.report(1.0, 1.015)
        self.assertTrue(r["old_normal_nll_ok_direct"])
        self.assertTrue(r["normal_nll_ok_direct"])
        self.assertAlmostEqual(r["relative_loss_increase_raw"], 0.015, places=3)

    def test_unavailable_guard_rejects_no_normal_rows(self):
        y = torch.ones(4, dtype=torch.long)
        r = guard_budget_report(torch.zeros(4, 2), torch.zeros(4, 2), y)
        self.assertFalse(r["available"])
        self.assertFalse(r["normal_nll_ok_direct"])
        self.assertFalse(r["old_normal_nll_ok_direct"])

    def test_unavailable_guard_rejects_nonfinite(self):
        y = torch.zeros(2, dtype=torch.long)
        live = torch.zeros(2, 2)
        cand = torch.zeros(2, 2)
        cand[0, 0] = float("nan")
        r = guard_budget_report(live, cand, y)
        self.assertFalse(r["available"])
        self.assertEqual(r["unavailable_reason"], "nonfinite_logits")

    def test_other_registered_protocol031_budget_is_unchanged(self):
        self.assertEqual(P031_CONFIG["proposal_start_matured"], 600)
        self.assertEqual(P031_CONFIG["proposal_every_matured"], 1600)
        self.assertEqual(P031_CONFIG["reuse_every_matured"], 32)
        self.assertEqual(P031_CONFIG["reuse_validation_intervals"], 16)
        self.assertEqual(P031_CONFIG["crossfade_prediction_intervals"], 8)
        self.assertEqual(P031_CONFIG["buffer_size"], 64)
        self.assertEqual(P031_CONFIG["accept_relative_loss_improvement"], 0.01)
        self.assertEqual(P031_CONFIG["normal_probability_allowance"], 0.01)
        self.assertEqual(P031_CONFIG["validation_fpr_delta_allowance"], 0.01)
        self.assertEqual(P031_CONFIG["guard_fpr_delta_allowance"], 0.01)

    def test_checkpoint_round_trip_preserves_guard_policy(self):
        first = Protocol032GuardBudgetLifecycle({}, config=P031_CONFIG)
        state = first.state_dict()
        self.assertIn("protocol032_guard_budget", state)
        second = Protocol032GuardBudgetLifecycle({}, config=P031_CONFIG)
        second.load_state_dict(state)
        self.assertEqual(second.p031_config, first.p031_config)
        self.assertEqual(
            second.state_dict()["protocol032_guard_budget"],
            first.state_dict()["protocol032_guard_budget"],
        )


if __name__ == "__main__":
    unittest.main()
