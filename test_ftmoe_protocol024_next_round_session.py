import unittest

import numpy as np

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_session import Protocol024Session


class TestProtocol024NextRoundSession(unittest.TestCase):
    def test_phase_metrics_uses_same_host_p24_evaluator(self):
        session = Protocol024Session.__new__(Protocol024Session)
        session.arm = "D"
        session.target_mode = "tol1_regression"
        session.steps = 800
        session.phase_defs = [{"name": "fixture", "start": 300,
                               "end": 720, "regime": "fixture"}]
        labels = np.zeros((800, 16), dtype=np.int64)
        labels[301, 0] = 1
        probability = np.full((800, 16), 0.1, dtype=np.float64)
        probability[300, 0] = 0.9
        classes = np.ones((800, 16, 3), dtype=np.float64) / 3.0
        session.predictions = {"labels": labels.copy(),
                               "raw_labels": labels.copy(),
                               "probability": probability,
                               "class_probability": classes}
        metrics = Protocol024Session.phase_metrics(session)
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics[0]["whole"]["onset"]["ap"], 1.0,
                               places=12)
        self.assertEqual(metrics[0]["whole"]["onset"]["positives"], 1)

    def test_phase_metrics_is_not_inherited_protocol023_implementation(self):
        self.assertIsNot(Protocol024Session.phase_metrics,
                         s4.PrequentialS4.phase_metrics)
        self.assertIsNot(Protocol024Session.probe_scores,
                         s4.PrequentialS4.probe_scores)
        self.assertIsNot(Protocol024Session.save_checkpoint,
                         s4.PrequentialS4.save_checkpoint)

    def test_phase_metrics_requires_explicit_or_manifest_timeline(self):
        session = Protocol024Session.__new__(Protocol024Session)
        session.phase_defs = []
        with self.assertRaises(RuntimeError):
            Protocol024Session.phase_metrics(session)


if __name__ == "__main__":
    unittest.main(verbosity=2)
