"""Protocol 019 S5 acceptance tests (The Plan §19 Tests 7/8/9 + smoke).

- Test 7: expert birth output continuity (max |logit_after - logit_before| < 1e-4)
- Test 8: dormant expert can reactivate (same id) instead of unbounded growth
- Test 9: optimizer (Adam) state preserved for surviving parameters
- Smoke: S5Session (method D, v2 gate) runs updates end-to-end

Run:  python test_ftmoe_protocol019_s5.py
"""
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, EAGateMoE
from recovery.PreGANSrc.src.ftmoe_online_s5 import (
    OnlineEAGateV2, OnlineFTMoEV2, S5Session,
)
from recovery.PreGANSrc.src.ftmoe_online_s4 import load_anchor_pool
from run_ftmoe_protocol019 import ReplayV2, load_v2_time_scale
from run_ftmoe_online import resolve_checkpoint

ROOT = Path(__file__).resolve().parent


def dummy_gate(seed=3):
    torch.manual_seed(seed)
    cfg = AblationConfig(experts=4)
    source = EAGateMoE(cfg).eval()
    return OnlineEAGateV2(source, seed)


def inject_window(gate, samples=1600, counts=None, unmatched=0, vectors=None,
                  ema=0.0, history=None):
    gate.routing_samples = samples
    gate.unmatched_count = unmatched
    gate.activation_counts = counts or {key: 0 for key in gate.ids}
    gate.unmatched_vectors = deque(vectors or [], maxlen=256)
    gate.unmatched_ratio_ema = ema
    gate.unmatched_ratio_history = deque(history or [], maxlen=8)


class S5Tests(unittest.TestCase):
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

    def arrays(self, n=110):
        labels = np.zeros((n + 1, 16), np.int64)
        labels[::3, 0] = 1
        labels[1::4, 1] = 2
        labels[2::5, 2] = 3
        creation = np.tile(np.arange(16, dtype=np.int64), (n + 1, 1))
        return {"host_features": self.host[:n + 1].copy(),
                "demands": self.graph[:n + 1].copy(),
                "schedules": self.schedule[:n + 1].copy(),
                "raw_labels": labels,
                "capacities": np.array([[4029, 4295 if h < 8 else 8192, 32212]
                                        for h in range(16)], np.float32),
                "creation_ids": creation}

    def test_birth_output_continuity(self):
        # Test 7: adding an expert (ramp 0) must leave logits ~unchanged.
        model = OnlineFTMoEV2(self.checkpoint, "D", 1)
        model.eval()
        gate = model.eagate
        torch.manual_seed(11)
        x = torch.randn(2, 16, 12, 7)
        s = torch.zeros(2, 12, 16, 16)
        s[:, :, :, :] = torch.eye(16).view(1, 1, 16, 16).expand(2, 12, 16, 16)
        ids = torch.tile(torch.arange(16).long(), (2, 12, 1))
        with torch.no_grad():
            before = model(x, s, x, graph_context={"creation_ids": ids})
        # force a birth (all criteria met via direct stat injection)
        parent = "0"
        centroid = F.normalize(gate.key_rows[parent].detach().clone() + 0.01, dim=0)
        vectors = [centroid.detach()] * 40
        inject_window(gate, samples=1600, counts={k: 10 for k in gate.ids},
                      unmatched=400, vectors=vectors, ema=0.3,
                      history=[0.3, 0.3, 0.3])
        gate.expert_age[parent] = 1000
        event = gate.adapt()
        self.assertIn("added", event)
        added = event["added"]
        self.assertEqual(len(added), 1)
        new_key = added[0]
        self.assertEqual(gate.ramp_steps_done[new_key], 0)
        with torch.no_grad():
            after = model(x, s, x, graph_context={"creation_ids": ids})
        for name in ("detection_logits", "class_logits"):
            diff = float((after[name] - before[name]).abs().max())
            self.assertLess(diff, 1e-4, f"{name} continuity violated: {diff}")

    def test_dormant_expert_reactivates_same_id(self):
        # Test 8: pattern A -> B -> A restores the old id instead of new experts.
        gate = dummy_gate(seed=5)
        self.assertEqual(gate.ids, ["0", "1", "2", "3"])
        # phase A: expert '0' unused long enough -> retire to dormant (>=2 kept)
        for key in gate.ids:
            gate.expert_age[key] = 0
            gate.activation_ema[key] = 0.0
            gate.low_activation_windows[key] = 0
        gate.expert_age["0"] = 600
        gate.activation_ema["0"] = 0.001
        gate.low_activation_windows["0"] = 5
        inject_window(gate, samples=1600, counts={"0": 0, "1": 600, "2": 500, "3": 500})
        event = gate.adapt()
        self.assertIn("0", event["dormant"])
        self.assertEqual(gate.ids, ["1", "2", "3"])
        self.assertEqual(list(gate.dormant_key_rows), ["0"])
        # phase B: recurrence of cluster A (centroid near dormant key '0')
        centroid = F.normalize(gate.dormant_key_rows["0"].detach().clone(), dim=0)
        vectors = [centroid + torch.randn_like(centroid) * 1e-3 for _ in range(60)]
        inject_window(gate, samples=1600, counts={"1": 600, "2": 500, "3": 500},
                      unmatched=300, vectors=vectors, ema=0.2, history=[0.2, 0.2, 0.2])
        event = gate.adapt()
        self.assertIn("0", event["reactivated"])
        self.assertIn("0", gate.ids)
        self.assertEqual(len(gate.ids), 4)
        self.assertEqual(event["added"], [])
        # no unbounded growth: further windows near A stay matched
        inject_window(gate, samples=1600, counts={"0": 600, "1": 400, "2": 300, "3": 300},
                      unmatched=0, vectors=[], ema=0.2, history=[0.2, 0.2, 0.2])
        event = gate.adapt()
        self.assertEqual(event["added"], [])
        self.assertEqual(len(gate.ids), 4)

    def test_birth_clones_parent_and_new_cluster_adds(self):
        gate = dummy_gate(seed=7)
        parent = max(gate.ids, key=lambda k: float(gate.key_rows[k].norm()))
        centroid = F.normalize(gate.key_rows[parent].detach().clone() * 0.3 +
                               torch.randn(64) * 0.1, dim=0)
        vectors = [centroid + torch.randn_like(centroid) * 2e-3 for _ in range(50)]
        inject_window(gate, samples=1600, counts={k: 100 for k in gate.ids},
                      unmatched=320, vectors=vectors, ema=0.25,
                      history=[0.25, 0.25, 0.25])
        event = gate.adapt()
        self.assertEqual(len(event["added"]), 1)
        key = event["added"][0]
        self.assertEqual(key, "4")
        # clone semantics: new expert weights == parent weights at birth
        for a, b in zip(gate.experts[key].parameters(), gate.experts[parent].parameters()):
            self.assertTrue(torch.equal(a, b))
        # threshold row is finite and initialised in tanh domain
        self.assertTrue(torch.isfinite(gate.threshold_rows[key]).all())
        # ramp starts at 0
        self.assertEqual(gate.ramp_steps_done[key], 0)
        gate.ramp_step()
        self.assertEqual(gate.ramp_steps_done[key], 1)

    def test_optimizer_state_preserved_across_topology_change(self):
        # Test 9: surviving parameter identities keep their Adam state.
        model = OnlineFTMoEV2(self.checkpoint, "D", 1)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-4)
        named = {name: parameter for name, parameter in model.named_parameters()}
        surviving = named["eagate.experts.1.0.weight"]
        live_before = [p for p in model.parameters() if p.requires_grad]
        # give every live param a non-empty Adam state
        for p in live_before:
            optimizer.state[p]["exp_avg"] = torch.randn_like(p)
            optimizer.state[p]["exp_avg_sq"] = torch.rand_like(p)
        snapshot = {id(p): (optimizer.state[p]["exp_avg"].clone(),
                            optimizer.state[p]["exp_avg_sq"].clone())
                    for p in live_before}
        gate = model.eagate
        centroid = F.normalize(gate.key_rows["0"].detach().clone() + 0.01, dim=0)
        vectors = [centroid.detach()] * 40
        inject_window(gate, samples=1600, counts={k: 10 for k in gate.ids},
                      unmatched=400, vectors=vectors, ema=0.3, history=[0.3, 0.3, 0.3])
        gate.adapt()
        model.set_trainability()
        gate.freeze_dormant()
        live_after = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(p is surviving for p in live_after))
        for p in live_after:
            if id(p) in snapshot:
                exp_avg, exp_avg_sq = snapshot[id(p)]
                self.assertTrue(torch.equal(optimizer.state[p]["exp_avg"], exp_avg))
                self.assertTrue(torch.equal(optimizer.state[p]["exp_avg_sq"], exp_avg_sq))

    def test_s5session_smoke(self):
        from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
        teacher = OnlineFTMoE(self.checkpoint, "A", 1)
        teacher.eval()
        replay = ReplayV2(self.arrays(40), self.v2_scale, self.graph_scale, 40)
        session = S5Session(self.checkpoint, "D", 1, replay, 1e-5, 301,
                            self.v2_scale, self.anchor_pool, teacher)
        for _ in range(40):
            session.step()
        self.assertEqual(session.update_number, 4)
        self.assertLessEqual(max(session.exposure.values()), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
