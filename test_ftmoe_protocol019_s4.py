"""Protocol 019 S4 acceptance tests (The Plan §19 Tests 5/6/10).

- Test 5: no future label leakage into prediction/update <= t
- Test 6: replay exposure cap (times_sampled <= max)
- Test 10: resume exact equivalence (model hash / predictions / exposure)

Run:  python test_ftmoe_protocol019_s4.py
"""
import unittest
from pathlib import Path

import numpy as np
import torch

from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from recovery.PreGANSrc.src.ftmoe_online_s4 import (
    S4Session, load_anchor_pool, MAX_EXPOSURE,
)
from run_ftmoe_protocol019 import ReplayV2, load_v2_time_scale
from run_ftmoe_online import resolve_checkpoint

ROOT = Path(__file__).resolve().parent


class S4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
        cls.checkpoint, _ = resolve_checkpoint(1)
        cls.v2_scale, _ = load_v2_time_scale(cls.checkpoint["normalization"])
        cls.graph_scale = np.asarray(cls.checkpoint["normalization"]["graph_scale"], np.float64)
        data = ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical"
        cls.host = np.load(data / "time_series.npy")[5].reshape(-1, 16, 7)
        cls.graph = np.load(data / "container_demand_series.npy")[5].reshape(-1, 16, 7)
        cls.schedule = np.load(data / "schedule_series.npy")[5]
        cls.anchor_pool = load_anchor_pool()

    def arrays(self, n=110, creation_shift=0):
        """Small artificial replay fixture with constant creation identities."""
        labels = np.zeros((n + 1, 16), np.int64)
        labels[::3, 0] = 1
        labels[1::4, 1] = 2
        labels[2::5, 2] = 3
        creation = np.tile(np.arange(16, dtype=np.int64), (n + 1, 1))
        creation = creation + creation_shift * 1000
        return {"host_features": self.host[:n + 1].copy(),
                "demands": self.graph[:n + 1].copy(),
                "schedules": self.schedule[:n + 1].copy(),
                "raw_labels": labels,
                "capacities": np.array([[4029, 4295 if h < 8 else 8192, 32212]
                                        for h in range(16)], np.float32),
                "creation_ids": creation}

    def session(self, arrays, steps, method="C"):
        replay = ReplayV2(arrays, self.v2_scale, self.graph_scale, steps)
        teacher = OnlineFTMoE(self.checkpoint, "A", 1)
        teacher.eval()
        return S4Session(self.checkpoint, method, 1, replay, 3e-5, 301,
                         self.v2_scale, self.anchor_pool, teacher)

    def test_future_label_isolation(self):
        # Test 5: labels at t+2.. (w.r.t. the furthest matured label) must not
        # affect predictions or updates at any step <= the horizon.  Rows up
        # to and including the guard are legitimately observed during step 40.
        a = self.arrays()
        b = self.arrays()
        b["raw_labels"][42:] = 0  # corrupt every label beyond step 42
        sa, sb = self.session(a, 110), self.session(b, 110)
        for t in range(40):
            pa, _ = sa.step()
            pb, _ = sb.step()
            np.testing.assert_array_equal(pa, pb)
            self.assertEqual(sa.model.state_hash(), sb.model.state_hash())
        self.assertEqual(sa.cursor, 40)

    def test_exposure_cap(self):
        # Test 6: no interval may be sampled more than MAX_EXPOSURE times.
        session = self.session(self.arrays(), 110)
        for _ in range(50):
            session.step()
        self.assertTrue(session.exposure)
        self.assertLessEqual(max(session.exposure.values()), MAX_EXPOSURE)
        sampled = sum(session.exposure.values())
        recorded = sum(len(u["buffer_indices"]) for u in session.updates)
        self.assertEqual(sampled, recorded)

    def test_resume_exact_equivalence(self):
        # Test 10: run A == run B (save at N/2, resume, continue).
        first = self.session(self.arrays(), 110)
        for _ in range(60):
            first.step()
        second = self.session(self.arrays(), 110)
        for _ in range(30):
            second.step()
        saved = second.save()
        third = self.session(self.arrays(), 110)
        third.restore(saved)
        for _ in range(30):
            third.step()
        self.assertEqual(first.model.state_hash(), third.model.state_hash())
        for key in ("probability", "class_probability", "labels", "raw_labels"):
            np.testing.assert_array_equal(first.predictions[key], third.predictions[key])
        self.assertEqual(first.exposure, third.exposure)
        # updates carry wall-clock 'seconds'; compare everything else exactly
        def stripped(updates):
            return [{k: v for k, v in item.items() if k != "seconds"}
                    for item in updates]
        self.assertEqual(stripped(first.updates), stripped(third.updates))

    def test_anchor_and_distill_components_recorded(self):
        session = self.session(self.arrays(), 110)
        for _ in range(40):
            session.step()
        self.assertGreaterEqual(len(session.updates), 1)
        for update in session.updates:
            components = update.get("components")
            if components is None:
                continue
            for key in ("loss_online", "loss_anchor", "loss_distill"):
                self.assertIsNotNone(components[key])
                self.assertTrue(np.isfinite(components[key]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
