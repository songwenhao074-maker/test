"""Failed audits must produce durable, specific handoff status."""
import json
import tempfile
import unittest
from pathlib import Path
from finalize_ftmoe_protocol027 import ensure_status

class HandoffTests(unittest.TestCase):
    def test_failed_gate_is_persisted_not_reported_as_model_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "eligibility.json").write_text(json.dumps({"protocol027_data_eligible": False,
                "gates": {"physical_labels": True, "runtime_9d_features_match_stored_causal_features": False}}))
            result = ensure_status(p, "123")
            self.assertIn("runtime_9d_features", result["blocker"])
            self.assertEqual(result["failed_comparators"], [])
            self.assertEqual(json.loads((p / "status.json").read_text()), result)

    def test_audit_only_is_ready_without_claiming_model_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "eligibility.json").write_text(json.dumps({"protocol027_data_eligible": True, "gates": {"features": True}}))
            (p / "workflow_execution.json").write_text(json.dumps({"run_models": False}))
            result = ensure_status(p, "124")
            self.assertEqual(result["state"], "ready_for_two_arm_pilot")
            self.assertFalse(result["completed"])
            self.assertIsNone(result["blocker"])
            self.assertEqual(ensure_status(p, "125"), result)

if __name__ == "__main__":
    unittest.main()
