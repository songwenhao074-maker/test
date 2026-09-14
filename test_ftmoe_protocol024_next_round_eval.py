import unittest
import numpy as np

from ftmoe_protocol024_eval import (
    assert_complete_settlement,
    average_precision_report,
    binary_detection_metrics,
    finalize_two_pending,
    positive_resource_macro_f1,
    raw_next_target,
    temporal_onset_metrics,
)


class TestProtocol024NextRoundEvaluation(unittest.TestCase):
    def test_same_host_onset_and_unknown_exclusion(self):
        labels = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int64)
        probability = np.array([[0.99, 0.01], [0.01, 0.01], [0.01, 0.01]])
        fixed = temporal_onset_metrics(probability, labels, horizon=1)
        self.assertAlmostEqual(fixed["ap"], 1.0, places=12)
        unknown = temporal_onset_metrics(
            np.array([[0.9], [0.8], [0.1]]),
            np.array([[0], [-1], [1]], dtype=np.int64), horizon=1)
        self.assertEqual(unknown["rows"], 0)
        self.assertEqual(unknown["reason"], "no_valid_rows")

    def test_horizon_two_excludes_tail(self):
        labels = np.array([[0], [0], [1], [0]], dtype=np.int64)
        probability = np.array([[0.9], [0.1], [0.2], [0.3]])
        result = temporal_onset_metrics(probability, labels, horizon=2)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["positives"], 2)
        self.assertIsNone(result["ap"])
        self.assertEqual(result["reason"], "no_negative")

    def test_tied_scores_are_order_invariant_and_equal_prevalence(self):
        y = np.array([1, 0, 1, 0, 0], dtype=np.int64)
        score = np.ones(5) * 0.3
        first = average_precision_report(y, score)
        order = np.array([4, 2, 0, 3, 1])
        second = average_precision_report(y[order], score[order])
        self.assertAlmostEqual(first["ap"], 2.0 / 5.0, places=12)
        self.assertEqual(first, second)

    def test_degenerate_ap_has_reason_and_nan_is_rejected(self):
        a = average_precision_report(np.zeros(4, dtype=int), np.arange(4))
        b = average_precision_report(np.ones(4, dtype=int), np.arange(4))
        self.assertEqual(a["reason"], "no_positive")
        self.assertEqual(b["reason"], "no_negative")
        with self.assertRaises(ValueError):
            binary_detection_metrics(np.array([0.1, np.nan]), np.array([0, 1]))

    def test_earlier_settlement_hole_is_not_hidden_by_tail(self):
        steps = 5
        raw = np.zeros((steps + 1, 1), dtype=np.int64)
        labels = np.zeros((steps, 1), dtype=np.int64)
        raw_labels = np.zeros((steps, 1), dtype=np.int64)
        settled = np.ones(steps, dtype=bool)
        settled[1] = False
        settled_at = np.arange(steps)
        with self.assertRaises(AssertionError):
            finalize_two_pending(raw, labels, raw_labels, settled, settled_at, steps)

    def test_resource_f1_ignores_normal_and_unknown_rows(self):
        labels = np.array([-1, 0, 0, 1, 2, 3], dtype=np.int64)
        cp = np.array([[1.0, 0.0, 0.0], [0.2, 0.7, 0.1], [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = positive_resource_macro_f1(cp, labels)
        self.assertEqual(result["rows"], 3)
        self.assertAlmostEqual(result["macro_f1"], 1.0, places=12)

    def test_raw_next_target_obeys_publication_delay(self):
        raw = np.array([[0], [2], [0]], dtype=np.int64)
        with self.assertRaises(ValueError):
            raw_next_target(raw, 0, observed_until=1)
        self.assertEqual(raw_next_target(raw, 0, observed_until=2)[0], 2)

    def test_complete_settlement_checks_raw_and_target(self):
        with self.assertRaises(AssertionError):
            assert_complete_settlement(np.array([[0], [0]]),
                                       np.array([[0], [-1]]),
                                       np.array([True, True]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
