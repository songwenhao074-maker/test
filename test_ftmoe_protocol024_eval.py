import unittest
import numpy as np

from ftmoe_protocol024_eval import (
    average_precision,
    temporal_onset_metrics,
    positive_resource_macro_f1,
    finalize_two_pending,
)


def legacy_flattened_onset(probability, labels):
    """Minimal reproduction of the Protocol-023 flatten-then-shift defect."""
    p = np.asarray(probability).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    in_class = (y > 0).astype(np.float64)
    future = np.zeros_like(in_class)
    future[:-1] = in_class[1:]
    valid = in_class == 0
    return average_precision(future[valid], p[valid])


class TestProtocol024Evaluation(unittest.TestCase):
    def test_onset_is_same_host_future(self):
        labels = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int64)
        probability = np.array([[0.99, 0.01], [0.01, 0.01], [0.01, 0.01]],
                               dtype=np.float64)
        fixed = temporal_onset_metrics(probability, labels, horizon=1)
        legacy = legacy_flattened_onset(probability, labels)
        self.assertAlmostEqual(fixed["ap"], 1.0, places=12)
        self.assertAlmostEqual(legacy, 0.5, places=12)

    def test_phase_tail_is_excluded_from_onset(self):
        labels = np.array([[0], [0], [1]], dtype=np.int64)
        probability = np.array([[0.1], [0.9], [1.0]], dtype=np.float64)
        result = temporal_onset_metrics(probability, labels, horizon=1)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["positives"], 1)
        self.assertAlmostEqual(result["ap"], 1.0, places=12)

    def test_finalize_settles_both_tail_rows(self):
        steps, hosts = 6, 2
        raw_seen = np.zeros((steps + 1, hosts), dtype=np.int64)
        raw_seen[4, 0] = 1
        raw_seen[5, 1] = 2
        labels = np.full((steps, hosts), -1, dtype=np.int64)
        raw_labels = np.full((steps, hosts), -1, dtype=np.int64)
        settled = np.zeros(steps, dtype=bool)
        settled_at = np.full(steps, -1, dtype=np.int64)
        labels[:steps-2] = 0
        raw_labels[:steps-2] = 0
        settled[:steps-2] = True
        settled_at[:steps-2] = np.arange(2, steps)
        finalize_two_pending(raw_seen, labels, raw_labels, settled, settled_at, steps)
        self.assertTrue(settled.all())
        self.assertFalse((labels < 0).any())
        self.assertEqual(settled_at[steps-2], steps)
        self.assertEqual(settled_at[steps-1], steps)

    def test_resource_f1_ignores_normal_rows(self):
        labels = np.array([0, 0, 1, 2, 3], dtype=np.int64)
        cp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
        result = positive_resource_macro_f1(cp, labels)
        self.assertEqual(result["rows"], 3)
        self.assertAlmostEqual(result["macro_f1"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
