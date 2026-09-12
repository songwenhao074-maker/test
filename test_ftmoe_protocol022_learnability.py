"""Protocol 022 — learnability-probe correctness tests (P22-S4).

Plan §25 requires the probe to be validated before its result is believed: a
"not learnable" verdict is only meaningful if the probe can detect a signal
that is definitely there.

The decisive regression is ``NaNPoisoningTests``: a single undefined value in a
lagged feature made the standardizer produce NaN weights, which silently turned
the probe into a constant predictor and produced a *false* STOP-B2 on the real
first run.  That must never be able to happen silently again.

Run from the repository root:
    python test_ftmoe_protocol022_learnability.py
"""
import json
import unittest
from pathlib import Path

import numpy as np

import probe_ftmoe_protocol022_learnability as pb

ROOT = Path(__file__).resolve().parent
STREAMS = ROOT / "artifacts/ftmoe_online/protocol_022/pilot_streams"


def synthetic_stream(steps=400, hosts=4, seed=5, onset_probability=0.05,
                     precursor_steps=4):
    """A stream with a planted, causally recoverable CPU-onset structure.

    Each CPU stress episode is preceded by a *precursor*: the host's CPU ratio
    rises ``precursor_steps`` intervals before the fault state begins.  That is
    what makes the data genuinely learnable beyond persistence -- an onset task
    whose only precursor is the fault itself cannot be predicted from
    pre-onset features by any model, and asserting that it can would be an
    unsound test rather than a strict one.

    Shapes follow the real contract: every model-visible array has ``steps``
    rows, while ``raw_labels`` carries one extra guard row.
    """
    rng = np.random.default_rng(seed)
    features = np.zeros((steps, hosts, 7))
    capacities = np.full((steps, hosts, 3), 1000.0)
    labels = np.zeros((steps + 1, hosts), dtype=np.int64)
    for h in range(hosts):
        ratio = np.full(steps, 0.1)
        t = 3
        while t < steps - precursor_steps - 3:
            if rng.random() < onset_probability:
                ratio[t:t + precursor_steps] = 0.35          # precursor
                ratio[t + precursor_steps:t + precursor_steps + 3] = 0.9
                t += precursor_steps + 3
            else:
                t += 1
        features[:, h, 0] = ratio * capacities[:, h, 0]
        # labels[i] is the state AFTER the action in interval i; the guard row
        # (index `steps`) repeats the last interval's state
        labels[:-1, h] = (ratio > 0.5).astype(np.int64)
        labels[-1, h] = labels[steps - 1, h]
    schedules = np.full((steps, hosts, hosts), 1.0 / hosts)
    after = np.tile(np.arange(hosts), (steps, 1))
    return {"features": features, "capacities": capacities, "labels": labels,
            "schedules": schedules, "after": after, "before": after,
            "steps": steps, "manifest": {"stream_sha256": "synthetic"}}


class TargetSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_stream()

    def test_state_target_is_binary(self):
        y = pb.state_target(self.data, 1, binary=True)
        finite = y[np.isfinite(y)]
        self.assertTrue(set(np.unique(finite)).issubset({0.0, 1.0}))
        labels = self.data["labels"]
        for t in range(0, 50):
            for h in range(labels.shape[1]):
                self.assertEqual(int(y[t, h]), int(labels[t + 1, h] > 0))

    def test_onset_target_excludes_rows_already_in_the_fault(self):
        y = pb.onset_target(self.data, 1, 1)
        labels = self.data["labels"]
        steps = self.data["steps"]
        for t in range(1, steps - 1):
            for h in range(labels.shape[1]):
                if labels[t, h] == 1:
                    self.assertTrue(np.isnan(y[t, h]),
                                    "a host already in the target fault must "
                                    "not be evaluated for a new onset")

    def test_onset_target_is_one_when_the_fault_starts_next_step(self):
        y = pb.onset_target(self.data, 1, 1)
        labels = self.data["labels"]
        for t in range(1, self.data["steps"] - 1):
            for h in range(labels.shape[1]):
                if labels[t, h] != 1 and labels[t + 1, h] == 1:
                    self.assertEqual(y[t, h], 1.0)

    def test_horizon_four_looks_further_ahead(self):
        y1 = pb.onset_target(self.data, 1, 1)
        y4 = pb.onset_target(self.data, 1, 4)
        finite = np.isfinite(y1) & np.isfinite(y4)
        self.assertGreaterEqual(int(np.nansum(y4[finite] >= y1[finite])),
                                int(finite.sum()),
                                "a 4-step window must cover everything a "
                                "1-step window covers")


class ValidationGuardTests(unittest.TestCase):
    """The probe must refuse inputs it cannot model, not silently degrade."""

    def setUp(self):
        self.data = synthetic_stream()

    def test_multiclass_target_is_rejected(self):
        """A 4-class target must never reach the binary logistic fit."""
        labels = self.data["labels"].copy()
        labels[::7] = 2                       # introduce a second fault class
        self.data["labels"] = labels
        steps, hosts = self.data["steps"], labels.shape[1]
        y = labels[1:steps + 1].astype(float)   # deliberately NOT binarised
        self.assertEqual(set(np.unique(y[y > 0]).tolist()), {1.0, 2.0})
        groups = pb.build_groups(self.data)
        with self.assertRaises(ValueError):
            pb.evaluate_target(self.data, groups, pb.all_keys(groups), y, 1, 0)

    def test_mis_shaped_feature_group_is_rejected(self):
        groups = pb.build_groups(self.data)
        groups[("bad", "wrong_shape")] = np.zeros((3, 2, 2))
        with self.assertRaises(ValueError):
            pb.evaluate_target(self.data, groups, pb.all_keys(groups),
                               pb.state_target(self.data, 1), 1, 0)


class NaNPoisoningTests(unittest.TestCase):
    """Regression guard for the false STOP-B2 of the first P22 probe run."""

    def test_finite_rows_flags_nan(self):
        X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
        mask, dropped = pb.finite_rows(X)
        self.assertEqual(dropped, 1)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_lagged_feature_introduces_nan_and_is_dropped(self):
        data = synthetic_stream()
        groups = pb.build_groups(data)
        X, names = pb.assemble(groups, pb.all_keys(groups))
        lagged = [i for i, n in enumerate(names) if n.startswith("past_fault_lag1")]
        self.assertTrue(lagged, "the capacity group must carry a lagged feature")
        self.assertGreater(int(np.isnan(X[:, lagged[0]]).sum()), 0)
        entry, _ = pb.evaluate_target(data, groups, pb.all_keys(groups),
                                      pb.state_target(data, 1), 1, 0)
        self.assertGreater(entry["rows_dropped_non_finite_feature"], 0,
                           "the undefined lagged rows must be dropped, not kept")
        self.assertIsNone(entry.get("status"))

    def test_nan_features_do_not_destroy_the_fit(self):
        """With NaN present the model must still beat chance in sample."""
        data = synthetic_stream()
        groups = pb.build_groups(data)
        entry, _ = pb.evaluate_target(data, groups, pb.all_keys(groups),
                                      pb.state_target(data, 1), 1, 0)
        fit = entry["metrics"]["fit"]
        self.assertIsNotNone(fit["ap"])
        self.assertGreater(fit["ap"], 0.30,
                           "a NaN-poisoned standardizer collapses to a constant "
                           "predictor (AP == prevalence ~= 0.10)")


class SignalDetectionTests(unittest.TestCase):
    """A probe that cannot find a planted signal cannot certify its absence."""

    def setUp(self):
        self.data = synthetic_stream(onset_probability=0.08)

    def test_planted_signal_is_recovered_out_of_sample(self):
        groups = pb.build_groups(self.data)
        steps = self.data["steps"]
        planted = (self.data["labels"][:steps] > 0).astype(float)
        groups[("planted", "true_state")] = planted
        entry, _ = pb.evaluate_target(self.data, groups, set(groups), planted, 1, 0)
        self.assertGreater(entry["metrics"]["test"]["ap"], 0.95)

    def test_leaky_control_is_predictable(self):
        """The leaky control validates the pipeline, not the regime (plan §9)."""
        groups = pb.build_groups(self.data)
        steps = self.data["steps"]
        hosts = self.data["labels"].shape[1]
        ratios = pb.ratios_of(self.data)
        nxt = np.zeros((steps, hosts))
        nxt[:-1] = ratios[1:, :, 0]
        groups[("leaky_control", "next_host_cpu_ratio")] = nxt
        entry, _ = pb.evaluate_target(self.data, groups, set(groups),
                                      pb.state_target(self.data, 1), 1, 0)
        self.assertGreater(entry["metrics"]["test"]["ap"], 0.5)

    def test_onset_task_is_recoverable_in_the_synthetic_stream(self):
        groups = pb.build_groups(self.data)
        entry, _ = pb.evaluate_target(self.data, groups, pb.all_keys(groups),
                                      pb.onset_target(self.data, 1, 1), 1, 0)
        test = entry["metrics"]["test"]
        self.assertIsNotNone(test["ap"])
        self.assertGreater(test["ap"], 3.0 * test["prevalence"])


class PermutationTests(unittest.TestCase):
    def test_permutation_preserves_the_label_multiset(self):
        y = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 4)
        hosts = np.zeros(y.size, dtype=int)
        rng = np.random.default_rng(3)
        for kind in pb.PERMUTATION_KINDS:
            out = pb.permute_labels(y, np.arange(y.size), hosts, kind, rng)
            self.assertEqual(sorted(out.tolist()), sorted(y.tolist()),
                             "permutation %s changed the class balance" % kind)
            self.assertEqual(out.shape, y.shape)

    def test_within_host_permutation_stays_inside_the_host(self):
        y = np.array([1.0, 0.0] * 20)
        hosts = np.repeat([0, 1], 20)
        rng = np.random.default_rng(4)
        out = pb.permute_labels(y, np.arange(y.size), hosts, "within_host", rng)
        for h in (0, 1):
            self.assertEqual(sorted(out[hosts == h].tolist()),
                             sorted(y[hosts == h].tolist()))

    def test_block_permutation_handles_a_ragged_tail(self):
        y = np.arange(101, dtype=float)
        hosts = np.zeros(y.size, dtype=int)
        rng = np.random.default_rng(5)
        out = pb.permute_labels(y, np.arange(y.size), hosts, "block12", rng)
        self.assertEqual(out.size, y.size)
        self.assertEqual(sorted(out.tolist()), sorted(y.tolist()))


class MetricTests(unittest.TestCase):
    def test_average_precision_matches_a_hand_computed_case(self):
        y = np.array([1, 0, 1, 0])
        score = np.array([0.9, 0.8, 0.7, 0.1])
        # precision at ranks 1 and 3 -> (1/1 + 2/3) / 2
        self.assertAlmostEqual(pb.average_precision(y, score),
                               (1.0 + 2.0 / 3.0) / 2.0, places=10)

    def test_perfect_and_inverted_ranking(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(pb.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(pb.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_constant_score_gives_chance_level_roc(self):
        y = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(pb.roc_auc(y, np.full(4, 0.5)), 0.5)

    def test_ap_of_a_constant_score_equals_prevalence(self):
        y = np.array([0, 1, 0, 0, 1, 0, 0, 0])
        self.assertAlmostEqual(pb.average_precision(y, np.full(8, 0.3)),
                               float(y.mean()), places=10)


class RealStreamTests(unittest.TestCase):
    """Only run against an existing pilot candidate (no simulator here)."""

    def _candidate(self):
        if not STREAMS.is_dir():
            self.skipTest("no pilot streams collected")
        for directory in sorted(STREAMS.glob("p*_seed*_steps*")):
            if (directory / "stream.npz").is_file():
                return directory
        self.skipTest("no complete pilot candidate")

    def test_feature_names_carry_no_forbidden_token(self):
        data = pb.load_stream(self._candidate())
        groups = pb.build_groups(data)
        _, names = pb.assemble(groups, pb.all_keys(groups))
        for name in names:
            for token in pb.FORBIDDEN_TOKENS:
                self.assertNotIn(token, name)

    def test_every_feature_is_deployable_and_matches_stream_shape(self):
        data = pb.load_stream(self._candidate())
        steps, hosts = data["steps"], data["labels"].shape[1]
        for (group, name), value in pb.build_groups(data).items():
            self.assertEqual(value.shape, (steps, hosts),
                             "%s.%s has the wrong shape" % (group, name))

    def test_probe_runs_end_to_end_on_a_real_candidate(self):
        report = pb.run_candidate(self._candidate(), n_permutations=2)
        for key in ("state_task", "onset_task_h1", "onset_task_h4",
                    "learnability_v2_gate", "review_question_answers"):
            self.assertIn(key, report)
        self.assertIn("ap", report["onset_task_h1"]["metrics"]["test"])
        self.assertEqual(set(report["learnability_v2_gate"]["checks"]),
                         {"ap_at_least_floor", "roc_auc_at_least_0.75",
                          "beats_best_simple_baseline_by_0.05",
                          "observed_ap_above_all_null_p99",
                          "positives_at_least_50"})
        answers = report["review_question_answers"]
        for i in range(1, 9):
            self.assertTrue(any(k.startswith("q%d_" % i) for k in answers),
                            "review question %d is unanswered" % i)


if __name__ == "__main__":
    unittest.main(verbosity=2)
