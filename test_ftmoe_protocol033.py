"""Static Protocol-033 contract tests; no simulator/model execution."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import prepare_ftmoe_protocol033_stream as p33
from maintenance.validate_protocol033_plan import validate
from ftmoe_protocol033_guard_budget import P033_CONFIG
from ftmoe_protocol032_guard_budget import (
    RELATIVE_TOLERANCE,
    ABSOLUTE_FLOOR_NATS,
    NUMERICAL_TOLERANCE,
)

ROOT = Path(__file__).resolve().parent


class Protocol033ContractTest(unittest.TestCase):
    def setUp(self):
        self.reg = p33.registration()
        self.phases = p33.phase_table(self.reg)

    def test_registered_plan_validator(self):
        report = validate(self.reg)
        self.assertTrue(report["plan_validation_passed"])
        self.assertEqual(report["total_rows"], 5969)
        self.assertEqual(report["chunk_count"], 30)
        self.assertEqual(report["last_chunk_rows"], 169)
        self.assertFalse(report["actual_dataset_generated"])

    def test_exact_geometry_and_target_support(self):
        self.assertEqual(self.reg["scored_intervals"], 5968)
        self.assertEqual(self.reg["guard_intervals"], 1)
        self.assertEqual(self.reg["total_intervals"], 5969)
        self.assertEqual(self.phases[0]["start"], 0)
        self.assertEqual(self.phases[-1]["end"], 5968)
        # Prediction window [start,end) consumes same-host targets
        # raw_labels[start+1:end+1], so the final target row is index 5968.
        self.assertEqual(self.phases[-1]["end"], self.reg["total_intervals"] - 1)
        q, r = divmod(self.reg["total_intervals"], self.reg["generation"]["chunk_intervals"])
        self.assertEqual((q, r), (29, 169))

    def test_all_six_registered_recurrence_windows(self):
        expected = ["U_rec1", "V_rec1", "U_rec2", "V_rec2", "U_rec3", "V_rec3"]
        got = [p["name"] for p in self.phases if "_rec" in p["name"]]
        self.assertEqual(got, expected)
        self.assertEqual(self.reg["evaluation"]["primary_windows"], expected)
        for p in self.phases:
            if p["name"] in expected:
                self.assertEqual(p["end"] - p["start"], 128)

    def test_registered_birth_clock_and_guard_policy(self):
        self.assertEqual(P033_CONFIG["proposal_start_matured"], 600)
        self.assertEqual(P033_CONFIG["proposal_every_matured"], 1000)
        self.assertEqual(P033_CONFIG["reuse_every_matured"], 32)
        self.assertEqual(P033_CONFIG["candidate_train_intervals"], 128)
        self.assertEqual(P033_CONFIG["validation_intervals"], 32)
        self.assertAlmostEqual(RELATIVE_TOLERANCE, 0.02)
        self.assertAlmostEqual(ABSOLUTE_FLOOR_NATS, 0.01)
        self.assertAlmostEqual(NUMERICAL_TOLERANCE, 1e-6)

    def test_physical_mapping_and_probability_are_frozen(self):
        self.assertEqual(p33.SOURCE_SERVICE, {"U": "S1", "V": "S3", "W": "S4"})
        self.assertAlmostEqual(self.reg["generation"]["event_probability"], 0.30)
        self.assertEqual(self.reg["replay_seed"], 700)
        self.assertEqual(self.reg["model_seed"], 1)
        for p in self.phases:
            if p["name"] == "F0":
                self.assertEqual(p["event_probability"], 0.0)
            else:
                self.assertEqual(p["event_probability"], 0.30)

    def test_workflow_is_dispatch_only_and_sequential(self):
        text = (ROOT / ".github/workflows/protocol033-5969.yml").read_text(encoding="utf8")
        on_block = text.split("permissions:", 1)[0]
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("push:", on_block)
        self.assertNotIn("schedule:", on_block)
        self.assertIn("Run exactly one C_fixed5 in an independent process", text)
        self.assertIn("Run exactly one D_guard_budget in an independent process", text)
        self.assertLess(text.index("Run exactly one C_fixed5"), text.index("Run exactly one D_guard_budget"))
        self.assertIn("--max-intervals 200", text)

    def test_plan_still_seals_single_stream_and_two_replays(self):
        plan = json.loads((ROOT / "artifacts/ftmoe_online/protocol_033/plan.json").read_text(encoding="utf8"))
        self.assertEqual(plan["full_generation_budget"], 1)
        self.assertEqual(plan["full_replay_budget"], 2)
        self.assertEqual(plan["arms"], ["C_fixed5", "D_guard_budget"])
        self.assertFalse(plan["confirmation_seeds_allowed"])
        self.assertFalse(plan["automatic_followups"])
        self.assertEqual(plan["workflow_trigger"], "workflow_dispatch_only")


if __name__ == "__main__":
    unittest.main()
