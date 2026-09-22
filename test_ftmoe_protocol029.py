import hashlib
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol029_memory_utility import (
    Protocol029PassiveAuditLifecycle,
    evaluate_passive_candidate,
)


class ReferenceExpert(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(16), nn.Linear(16, 12), nn.GELU(), nn.Linear(12, 5))


class ReferenceFixedBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(16, 4)
        self.experts = nn.ModuleList([ReferenceExpert() for _ in range(4)])


class FakeModel(nn.Module):
    def __init__(self, bank, z):
        super().__init__()
        self.learner = bank
        self._last_z = z


class FakeSession:
    def __init__(self, bank, z):
        self.model = FakeModel(bank, z)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.optimizer_archive = {}
        self.shadow_optimizer = None
        self.raw_seen = np.full((1, 16), -1, dtype=np.int64)
        self.cursor = 0


def bank_hash(bank):
    h = hashlib.sha256()
    for name, tensor in sorted(bank.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    h.update(json.dumps(bank.topology_manifest(), sort_keys=True).encode())
    return h.hexdigest()


class Protocol029Tests(unittest.TestCase):
    def test_passive_preview_does_not_mutate_bank_optimizer_or_rng(self):
        torch.manual_seed(29029)
        np.random.seed(29029)
        random.seed(29029)
        bank = DynamicResidualBank(ReferenceFixedBank(), max_experts=8, ramp_updates=8)
        key = bank.create_shadow("0")
        bank.activate_shadow()
        bank.retire(key)
        z = torch.randn(1, 16, 16)
        session = FakeSession(bank, z)
        ctrl = Protocol029PassiveAuditLifecycle(guard_anchor={})
        ctrl.specialist_memory = {
            key: {
                "centroid": np.ones(16, dtype=np.float64) / 4.0,
                "similarity_threshold": -1.0,
                "last_causal_accept_matured": 10,
            }
        }
        output = {
            "base_final_detection_logits": torch.zeros(1, 16, 2),
            "base_final_class_logits": torch.zeros(1, 16, 3),
        }
        before_bank = bank_hash(bank)
        before_opt = json.dumps(session.optimizer.state_dict(), default=str, sort_keys=True)
        before_torch = torch.get_rng_state().clone()
        before_np = np.random.get_state()
        before_py = random.getstate()
        dormant = ctrl._passive_prediction(session, 0, output)
        self.assertEqual(dormant, [key])
        self.assertEqual(bank_hash(bank), before_bank)
        self.assertEqual(json.dumps(session.optimizer.state_dict(), default=str, sort_keys=True), before_opt)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        after_np = np.random.get_state()
        self.assertEqual(before_np[0], after_np[0])
        self.assertTrue(np.array_equal(before_np[1], after_np[1]))
        self.assertEqual(before_np[2:], after_np[2:])
        self.assertEqual(random.getstate(), before_py)
        self.assertEqual(ctrl.passive_state_check_failures, 0)
        self.assertEqual(len(ctrl.passive_prediction_rows), 1)
        self.assertEqual(len(ctrl.generalist_prediction_rows), 1)

    def test_passive_candidate_uses_original_acceptance_gates(self):
        pairs = [{
            "live_loss": 1.0, "candidate_loss": 0.9,
            "normal_rows": 10, "live_normal_probability": 0.1,
            "candidate_normal_probability": 0.1,
            "live_fp_0p5": 0, "candidate_fp_0p5": 0,
        } for _ in range(16)]
        guard = {"available": True, "relative_loss_increase": 0.0, "fpr_delta_0p5": 0.0}
        decision = evaluate_passive_candidate(
            pairs, guard, improvement_min=0.01, normal_allowance=0.0,
            validation_fpr_allowance=0.01, guard_loss_allowance=0.02,
            guard_fpr_allowance=0.01,
        )
        self.assertTrue(decision["all_gates_pass"])
        guard["relative_loss_increase"] = 0.03
        blocked = evaluate_passive_candidate(
            pairs, guard, improvement_min=0.01, normal_allowance=0.0,
            validation_fpr_allowance=0.01, guard_loss_allowance=0.02,
            guard_fpr_allowance=0.01,
        )
        self.assertFalse(blocked["all_gates_pass"])
        self.assertIn("old_knowledge_guard_loss_regression", blocked["reject_reasons"])

    def test_registration_is_diagnostic_only(self):
        reg = json.loads(Path("artifacts/ftmoe_online/protocol_029/registration.json").read_text())
        self.assertEqual(reg["protocol"], "029")
        self.assertFalse(reg["rerun_C"])
        self.assertEqual(reg["new_training_arms"], 0)
        self.assertFalse(reg["online_method_changed"])
        self.assertEqual(reg["replay_seed"], 700)
        self.assertEqual(reg["model_seed"], 1)
        self.assertEqual(reg["source_protocol028_artifact_id"], 10684617851)


if __name__ == "__main__":
    unittest.main(verbosity=2)
