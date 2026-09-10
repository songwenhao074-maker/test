"""Checks for retrospective lag and duration-normalized rolling area."""
import unittest
from analyze_ftmoe_protocol020 import adaptation_lag, area_under


class AnalysisTests(unittest.TestCase):
    def test_constant_area_is_constant_metric(self):
        rows = [{"end": t, "f1": .6, "pr_auc": .7} for t in range(100, 501, 10)]
        result = area_under(rows, 100, 500)
        self.assertAlmostEqual(result["f1"], .6)
        self.assertAlmostEqual(result["pr_auc"], .7)

    def test_lag_is_duration_and_uses_complete_new_phase_windows(self):
        rows = [{"end": t, "f1": .5} for t in range(400, 801, 10)]
        lag, target = adaptation_lag(rows, 400, 800)
        self.assertEqual(lag, 100)
        self.assertEqual(target, .5)

    def test_cannot_recover_in_next_phase(self):
        rows = [{"end": t, "f1": float(t >= 790)} for t in range(400, 1201, 10)]
        lag, _ = adaptation_lag(rows, 400, 800)
        self.assertIsNone(lag)

    def test_zero_target_is_undefined(self):
        rows = [{"end": t, "f1": 0.} for t in range(400, 801, 10)]
        self.assertEqual(adaptation_lag(rows, 400, 800), (None, 0.))


if __name__ == "__main__":
    unittest.main()
