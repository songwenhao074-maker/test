"""Protocol 022 round 2 — S5 capacity-diagnostic correctness tests.

These cover the pieces that would silently invalidate an S5 conclusion:

  * the causal feature extension is exactly the P22-S4 probe families, so the
    model-side and probe-side "causal capacity input" are the same object;
  * label tolerance matches the frozen P19/P20 rule;
  * detection / onset / diagnosis metrics are correct on hand-computed cases;
  * the capacity controls really do differ in parameter count (otherwise
    "C-wide" would be an accidental duplicate of "C-current");
  * a residual bank starts as an exact no-op, which is what makes the
    pre-training evaluation a valid A/C reference;
  * the registered evaluation windows are contiguous, disjoint and inside the
    registered development stream.

Run from the repository root:
    python test_ftmoe_protocol022_s5.py
"""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

import run_ftmoe_protocol022_s5 as s5

ROOT = Path(__file__).resolve().parent


class CausalFeatureTests(unittest.TestCase):
    def test_headroom_and_registered_lags(self):
        steps, hosts = 20, 2
        ratios = np.full((steps, hosts, 3), 0.5, dtype=np.float32)
        ratios[6:, 0, 1] = 0.9            # RAM rises at t=6
        ratios[10:, 1, 2] = 0.8           # Disk rises at t=10
        out = s5.causal_features(ratios, None)
        self.assertEqual(out.shape, (steps, hosts, 3))
        # cpu_headroom
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.5, places=6)
        # ram_minus_cpu_lag4 at t = ram(t) - cpu(t-4).  RAM starts rising at
        # t=6, so t=5 is the last row before any rise can appear and t=10 is
        # four steps past the rise.
        self.assertAlmostEqual(float(out[10, 0, 1]), 0.4, places=5)
        self.assertAlmostEqual(float(out[5, 0, 1]), 0.0, places=5)
        # disk_minus_cpu_lag8 at t = disk(t) - cpu(t-8).  Disk rises at t=10,
        # so t=9 is the last row before any rise can appear.
        self.assertAlmostEqual(float(out[18, 1, 2]), 0.3, places=5)
        self.assertAlmostEqual(float(out[9, 1, 2]), 0.0, places=5)

    def test_first_lag_rows_are_zero_not_nan(self):
        ratios = np.full((30, 3, 3), 0.4, dtype=np.float32)
        out = s5.causal_features(ratios, None)
        self.assertTrue(np.isfinite(out).all())


class ToleranceLabelTests(unittest.TestCase):
    def test_both_fill_directions(self):
        raw = np.zeros((10, 1), dtype=np.int64)
        raw[4, 0] = 2
        labels = s5.tolerance_labels(raw, 10)
        self.assertEqual(int(labels[3, 0]), 2, "one step back is filled")
        self.assertEqual(int(labels[4, 0]), 2)
        self.assertEqual(int(labels[5, 0]), 2, "one step forward is filled")
        self.assertEqual(int(labels[2, 0]), 0)
        self.assertEqual(int(labels[6, 0]), 0)

    def test_forward_fill_actually_writes_back(self):
        """The frozen runner's forward pass writes to a temporary view."""
        raw = np.zeros((10, 16), dtype=np.int64)
        raw[4, 3] = 1
        labels = s5.tolerance_labels(raw, 10)
        self.assertEqual(int(labels[5, 3]), 1,
                         "the next interval must inherit the fault state")
        self.assertEqual(int(labels[3, 3]), 1,
                         "the previous interval must inherit the fault state")

    def test_two_events_do_not_bleed_into_each_other(self):
        raw = np.zeros((12, 1), dtype=np.int64)
        raw[3, 0] = 2
        raw[8, 0] = 3
        labels = s5.tolerance_labels(raw, 12)
        self.assertEqual(int(labels[2, 0]), 2)
        self.assertEqual(int(labels[4, 0]), 2)
        self.assertEqual(int(labels[5, 0]), 0, "the gap stays normal")
        self.assertEqual(int(labels[7, 0]), 3)
        self.assertEqual(int(labels[9, 0]), 3)

    def test_shapes(self):
        raw = np.zeros((11, 4), dtype=np.int64)
        self.assertEqual(s5.tolerance_labels(raw, 10).shape, (10, 4))

    def test_short_horizon_is_handled(self):
        raw = np.zeros((2, 1), dtype=np.int64)
        raw[0, 0] = 1
        self.assertEqual(s5.tolerance_labels(raw, 1).shape, (1, 1))


class MetricTests(unittest.TestCase):
    def test_onset_metrics_exclude_rows_already_faulted(self):
        labels = np.array([[0], [1], [1], [0], [1]], dtype=np.int64)
        probability = np.array([[0.1], [0.9], [0.9], [0.8], [0.7]])
        out = s5.onset_metrics(probability, labels)
        # kept rows are the non-faulted ones: t=0 and t=3.  Both are followed
        # by a fault, so both are positive onsets.
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["positives"], 2)
        # and a host already faulted at t is never evaluated
        labels2 = np.array([[1], [1], [0], [0]], dtype=np.int64)
        out2 = s5.onset_metrics(np.full((4, 1), 0.5), labels2)
        self.assertEqual(out2["n"], 2)

    def test_onset_ap_is_one_for_a_perfect_ranking(self):
        labels = np.zeros((8, 1), dtype=np.int64)
        labels[3, 0] = 1
        probability = np.zeros((8, 1))
        probability[2, 0] = 0.9          # warns exactly one step ahead
        out = s5.onset_metrics(probability, labels)
        self.assertEqual(out["positives"], 1)
        self.assertAlmostEqual(out["ap"], 1.0, places=10)

    def test_detection_metrics_hand_computed(self):
        labels = np.array([0, 1, 1, 0], dtype=np.int64)
        probability = np.array([0.6, 0.7, 0.2, 0.1])
        out = s5.detection_metrics(probability, labels)
        # threshold 0.5 -> predict [1,1,0,0] against [0,1,1,0]
        self.assertEqual((out["tp"], out["fp"], out["fn"], out["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(out["precision"], 0.5)
        self.assertAlmostEqual(out["recall"], 0.5)
        self.assertAlmostEqual(out["f1"], 0.5)

    def test_pr_auc_perfect_and_inverted(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        good = s5.detection_metrics(np.array([0.1, 0.2, 0.8, 0.9]), labels)
        bad = s5.detection_metrics(np.array([0.9, 0.8, 0.2, 0.1]), labels)
        self.assertAlmostEqual(good["pr_auc"], 1.0, places=10)
        self.assertLess(bad["pr_auc"], 0.6)

    def test_resource_macro_f1_uses_positive_rows_only(self):
        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        class_probability = np.array([[0.9, 0.05, 0.05],
                                      [0.9, 0.05, 0.05],
                                      [0.05, 0.9, 0.05],
                                      [0.05, 0.05, 0.9]])
        out = s5.resource_macro_f1(class_probability, labels)
        self.assertAlmostEqual(out["macro_f1"], 1.0, places=10)
        self.assertEqual([v["support"] for v in out["per_class"].values()],
                         [1, 1, 1])

    def test_resource_macro_f1_is_none_without_positives(self):
        labels = np.zeros(4, dtype=np.int64)
        out = s5.resource_macro_f1(np.full((4, 3), 1 / 3), labels)
        self.assertIsNone(out["macro_f1"])


class CapacityControlTests(unittest.TestCase):
    def test_bank_parameter_counts_are_actually_different(self):
        current = s5.WideResidualBank(64, 32, 4).parameter_count()
        wide = s5.WideResidualBank(64, 64, 8).parameter_count()
        causal = s5.WideResidualBank(64 + 3, 32, 4).parameter_count()
        self.assertGreater(wide, current)
        self.assertGreater(causal, current)
        self.assertNotEqual(wide, current)
        print("\n  parameter counts: current=%d wide=%d causal=%d"
              % (current, wide, causal))

    def test_bank_forward_shapes(self):
        bank = s5.WideResidualBank(64, 32, 4)
        z = torch.zeros(2, 16, 64)
        correction, probabilities = bank(z)
        self.assertEqual(tuple(correction.shape), (2, 16, 5))
        self.assertEqual(tuple(probabilities.shape), (2, 16, 4))
        self.assertTrue(torch.allclose(probabilities.sum(-1),
                                       torch.ones(2, 16), atol=1e-5))

    def test_bank_starts_as_an_exact_noop(self):
        bank = s5.WideResidualBank(64, 32, 4)
        z = torch.randn(2, 5, 64)
        correction, _ = bank(z)
        self.assertTrue(torch.allclose(correction, torch.zeros_like(correction),
                                       atol=1e-7),
                        "a zero-initialised last layer must make the residual "
                        "an exact no-op so the pre-training row is a valid A/C "
                        "reference")

    def test_causal_bank_accepts_the_extension(self):
        bank = s5.WideResidualBank(64 + 3, 32, 4)
        z = torch.randn(2, 5, 64)
        extra = torch.randn(2, 5, 3)
        correction, _ = bank(z, extra)
        self.assertEqual(tuple(correction.shape), (2, 5, 5))


class SplitTests(unittest.TestCase):
    def test_splits_are_contiguous_disjoint_and_registered(self):
        bounds = [s5.SPLITS[name] for name in s5.SPLITS]
        for start, end in bounds:
            self.assertLess(start, end)
        ordered = sorted(bounds)
        self.assertEqual(ordered[0][0], 0)
        self.assertEqual(ordered[-1][1], 2380,
                         "the registered windows must cover the realised "
                         "development stream exactly")
        for (_, prev_end), (next_start, _) in zip(ordered, ordered[1:]):
            self.assertEqual(prev_end, next_start,
                             "registered windows must be contiguous")

    def test_train_split_is_disjoint_from_every_future_split(self):
        train_start, train_end = s5.SPLITS[s5.TRAIN_SPLIT]
        for name in s5.FUTURE_SPLITS:
            start, end = s5.SPLITS[name]
            self.assertTrue(end <= train_start or start >= train_end,
                            "%s overlaps the training segment" % name)

    def test_unseen_block_is_split_into_three_contiguous_parts(self):
        early = s5.SPLITS["unseen_early_train"]
        mid = s5.SPLITS["unseen_mid_validation"]
        late = s5.SPLITS["unseen_late_test"]
        self.assertEqual(early[1], mid[0])
        self.assertEqual(mid[1], late[0])
        self.assertEqual(early[0], s5.SPLITS["familiar_1"][1])
        self.assertEqual(late[1], s5.SPLITS["familiar_2"][0])

    def test_familiar_segments_bracket_the_unseen_block(self):
        # the registered windows are contiguous, so "bracket" means the
        # familiar segment ends exactly where the unseen block begins and the
        # unseen block ends exactly where the second familiar segment begins
        self.assertEqual(s5.SPLITS["familiar_1"][1],
                         s5.SPLITS["unseen_early_train"][0])
        self.assertEqual(s5.SPLITS["unseen_late_test"][1],
                         s5.SPLITS["familiar_2"][0])
        self.assertEqual(s5.SPLITS["familiar_2"][1],
                         s5.SPLITS["unseen_recurrence"][0])


class RegisteredGateTests(unittest.TestCase):
    def test_gate_thresholds_match_the_plan(self):
        self.assertEqual(s5.GATE["pr_auc_gain_min"], 0.03)
        self.assertEqual(s5.GATE["onset_ap_gain_min"], 0.05)
        self.assertEqual(s5.GATE["familiar_f1_drop_max"], 0.03)
        self.assertEqual(s5.GATE["anchor_f1_drop_max"], 0.03)

    def test_variant_list_is_exactly_the_registered_comparison(self):
        self.assertEqual(set(s5.VARIANTS),
                         {"A", "C-current", "C-wide", "C-budget", "C-causal"})

    def test_checkpoint_hash_is_the_registered_one(self):
        self.assertEqual(
            s5.CHECKPOINT_SHA,
            "10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b")


class DevelopmentStreamPresenceTests(unittest.TestCase):
    def test_manifest_declares_the_registered_phases_when_present(self):
        path = s5.STREAM / "manifest.json"
        if not path.is_file():
            self.skipTest("development stream not collected yet")
        manifest = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(manifest["protocol"], "022")
        names = [p["name"] for p in manifest["phases"]]
        self.assertEqual(names, ["familiar_1", "unseen_1", "familiar_2",
                                 "unseen_recur"])
        probabilities = [p["cascade_task_probability"] for p in manifest["phases"]]
        self.assertEqual(probabilities, [0.0, 0.25, 0.0, 0.25])
        self.assertEqual(manifest["steps"], 2380,
                         "the realised stream must match the amended geometry")
        self.assertEqual(manifest["scored_intervals"], 2380)
        with __import__("numpy").load(s5.registered_stream() / "stream.npz") as data:
            rows = data["host_features"].shape[0]
        self.assertEqual(rows, manifest["steps"] + 1,
                         "arrays carry exactly steps + 1 rows, the layout the "
                         "frozen ReplayV3 contract requires")
        # the amendment must be recorded, not silent
        amendment = manifest.get("phase_amendment")
        self.assertIsNotNone(amendment)
        self.assertEqual([p["length"] for p in amendment["plan_phases"]],
                         [500, 1200, 500, 1200])
        self.assertEqual([p["length"] for p in amendment["realised_phases"]],
                         [350, 840, 350, 840])
        lengths = [p["length"] for p in manifest["phases"]]
        self.assertEqual(lengths, [350, 840, 350, 840])
        self.assertEqual(sum(lengths), manifest["steps"])
        for phase in manifest["phases"]:
            self.assertEqual(phase["kind"],
                             "familiar" if phase["cascade_task_probability"] == 0.0
                             else "unseen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
