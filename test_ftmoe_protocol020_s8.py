"""Integration checks for Protocol 020 causal dynamic experts and resume."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from run_ftmoe_protocol020 import ReplayV3, load_anchor_pool_v3, load_v2_time_scale_p20
from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from recovery.PreGANSrc.src.ftmoe_online_s7 import S7Session
from recovery.PreGANSrc.src.ftmoe_online_s8 import OnlineFTMoEV3, S8Session

ART = Path("artifacts/ftmoe_online/protocol_020")


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        cls.checkpoint = torch.load(ART / "s6/adapted_v4_seed1/best.pt", map_location="cpu", weights_only=False)
        with np.load(ART / "drift_streams/dev_seed500_steps2000/stream.npz") as data:
            cls.arrays = {key: data[key][:121].copy() for key in data.files
                          if data[key].ndim and data[key].shape[0] == 2001}
        cls.arrays["capacities_per_interval"] = cls.arrays.pop("capacities")
        cls.arrays["capacities"] = cls.arrays["capacities_per_interval"][0]
        cls.scale, _ = load_v2_time_scale_p20(cls.checkpoint["normalization"])
        cls.pool, cls.balance = load_anchor_pool_v3()
        cls.pool = {k: v[:64] for k, v in cls.pool.items()}
        cls.config = json.loads((ART / "continuation/dynamic_v3.json").read_text())

    def session(self, dynamic=True):
        replay = ReplayV3(deepcopy(self.arrays), self.scale,
            self.checkpoint["normalization"]["graph_scale"], 120)
        teacher = OnlineFTMoE(self.checkpoint, "A", 1)
        kwargs = {"dynamic_config": self.config} if dynamic else {}
        return (S8Session if dynamic else S7Session)(self.checkpoint,
            "D" if dynamic else "C", 1, replay, 1e-4, 500,
            self.scale, self.pool, teacher, self.balance, **kwargs)

    def test_initial_prediction_equal_to_c(self):
        c, d = self.session(False), self.session()
        p, cls = c.step()
        q, dcls = d.step()
        np.testing.assert_allclose(p, q, atol=1e-6, rtol=0)
        np.testing.assert_allclose(cls, dcls, atol=1e-6, rtol=0)

    def test_shadow_and_ramp_zero_preserve_predictions(self):
        session = self.session()
        gate = session.model.eagate
        before = session._forward_indices([0])["detection_logits"].detach()
        gate.create_shadow(gate.key_rows["0"].detach())
        with torch.no_grad():
            gate.expert_detection_heads[gate.shadow_id].bias.add_(10.)
        shadow = session._forward_indices([0])["detection_logits"].detach()
        torch.testing.assert_close(before, shadow, rtol=0, atol=1e-6)
        gate.activate_shadow()
        after = session._forward_indices([0])["detection_logits"].detach()
        torch.testing.assert_close(before, after, rtol=0, atol=1e-6)

    def test_optimizer_preserves_surviving_moments(self):
        session = self.session()
        for _ in range(10):
            session.step()
        parameter = session.model.eagate.key_rows["0"]
        moment = session.optimizer.state[parameter]["exp_avg"].clone()
        gate = session.model.eagate
        gate.create_shadow(parameter.detach())
        gate.activate_shadow()
        session.sync_optimizer()
        torch.testing.assert_close(moment, session.optimizer.state[parameter]["exp_avg"], rtol=0, atol=0)
        all_parameters = [p for group in session.optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(all_parameters), len(set(all_parameters)))

    def test_resume_exact_c_and_d(self):
        for dynamic in (False, True):
            original = self.session(dynamic)
            for _ in range(17):
                original.step()
            saved = deepcopy(original.save())
            for _ in range(13):
                original.step()
            resumed = self.session(dynamic)
            resumed.restore(saved)
            for _ in range(13):
                resumed.step()
            self.assertEqual(original.model.state_hash(), resumed.model.state_hash())
            np.testing.assert_array_equal(original.predictions["probability"], resumed.predictions["probability"])
            self.assertEqual(original.exposure, resumed.exposure)

    def test_shadow_resume_exact(self):
        original = self.session()
        for _ in range(10):
            original.step()
        gate = original.model.eagate
        gate.create_shadow(gate.key_rows["0"].detach())
        original.shadow_optimizer = torch.optim.AdamW(gate.shadow_parameters(), lr=1e-4, weight_decay=1e-4)
        original._train_shadow(list(original.buffer))
        saved = deepcopy(original.save())
        for _ in range(20):
            original.step()
        resumed = self.session()
        resumed.restore(saved)
        for _ in range(20):
            resumed.step()
        self.assertEqual(original.model.state_hash(), resumed.model.state_hash())
        np.testing.assert_array_equal(original.predictions["probability"], resumed.predictions["probability"])

    def test_future_labels_do_not_change_predictions_or_trigger(self):
        first, second = self.session(), self.session()
        second.replay.arrays["raw_labels"][20:] = 3
        second.replay.arrays["capacities_per_interval"][20:] *= .1
        for _ in range(20):
            first.step()
            second.step()
        np.testing.assert_array_equal(first.predictions["probability"][:20], second.predictions["probability"][:20])
        self.assertEqual(first.trigger.state(), second.trigger.state())

    def test_retire_reactivate_same_identity(self):
        session = self.session()
        gate = session.model.eagate
        gate.create_shadow(gate.key_rows["0"].detach())
        key = gate.activate_shadow()
        parameter = gate.key_rows[key]
        gate._retire(key)
        gate._reactivate(key)
        self.assertIs(gate.key_rows[key], parameter)
        self.assertIn(key, gate.ids)

    def test_resume_after_birth_and_retirement(self):
        for dormant in (False, True):
            original = self.session()
            for _ in range(10):
                original.step()
            gate = original.model.eagate
            gate.create_shadow(gate.key_rows["0"].detach())
            key = gate.activate_shadow()
            if dormant:
                gate._retire(key)
                gate.freeze_dormant()
            original.sync_optimizer()
            saved = deepcopy(original.save())
            for _ in range(10):
                original.step()
            restored = self.session()
            restored.restore(saved)
            for _ in range(10):
                restored.step()
            self.assertEqual(original.model.state_hash(), restored.model.state_hash())
            np.testing.assert_array_equal(original.predictions["probability"], restored.predictions["probability"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
