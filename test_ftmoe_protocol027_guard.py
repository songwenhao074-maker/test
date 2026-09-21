import unittest
import torch

from ftmoe_protocol027_normal_guard import normal_detection_guard_report


class TestProtocol027NormalGuard(unittest.TestCase):
    def test_all_normal_rows_are_available_without_positive_rows(self):
        live = torch.tensor([[[3.0, -2.0], [2.0, -1.0]]])
        cand = torch.tensor([[[3.1, -2.1], [2.1, -1.1]]])
        target = torch.zeros((1, 2), dtype=torch.long)
        report = normal_detection_guard_report(live, cand, target)
        self.assertTrue(report["available"])
        self.assertEqual(report["normal_rows"], 2)
        self.assertEqual(report["positive_rows"], 0)
        self.assertIsNotNone(report["relative_loss_increase"])
        self.assertIsNone(report["positive_only_metrics"]["ap"])

    def test_no_normal_rows_are_unavailable(self):
        logits = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        target = torch.tensor([[1, 2]], dtype=torch.long)
        report = normal_detection_guard_report(logits, logits, target)
        self.assertFalse(report["available"])
        self.assertEqual(report["unavailable_reason"], "no_normal_rows")

    def test_nonfinite_logits_are_unavailable(self):
        live = torch.tensor([[[0.0, float("nan")]]])
        cand = torch.tensor([[[0.0, 1.0]]])
        target = torch.zeros((1, 1), dtype=torch.long)
        report = normal_detection_guard_report(live, cand, target)
        self.assertFalse(report["available"])
        self.assertEqual(report["unavailable_reason"], "nonfinite_logits")


if __name__ == "__main__":
    unittest.main()
