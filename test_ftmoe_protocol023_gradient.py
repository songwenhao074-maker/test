"""Unit tests for ``probe_ftmoe_protocol023_gradient.py`` (plan §14, gate H3).

What is pinned here
-------------------
1. the cosine math (identical -> 1.0, opposite -> -1.0, orthogonal -> ~0.0);
2. the registered pass/fail rule, exactly as written in plan §14 and
   ``artifacts/ftmoe_online/protocol_023/protocol.json`` -> ``gradient_gate``,
   including the ``>= 30% of event pairs with cosine < 0`` branch and the
   inclusive boundaries at exactly -0.05 and exactly 0.30;
3. deterministic, host-balanced event sampling for a fixed seed;
4. that the probe updates nothing: every parameter/buffer tensor is
   bit-identical before and after the gradient measurement, and ``.grad`` is
   never written;
5. the NaN guard: one non-finite gradient or loss raises instead of producing
   NaN cosines (Protocol 022 lost a round to exactly that failure mode).

The registered development stream ``dev_seed700_steps2880`` is not present in
this checkout, so the end-to-end verification runs on a *synthetic tiny stream
built inside this file* (a temporary directory, never under
``artifacts/ftmoe_online/protocol_023/development_streams/``).  The synthetic
stream keeps the registered phase names and the real array layout, so the same
code path (window normalization, maturity, sampling, forward, gradient, hash
check, verdict, JSON) is exercised; only the numbers are synthetic.

The frozen checkpoint IS the registered one and must not be modified; the tests
load it read-only and assert its SHA-256.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_ftmoe_protocol023_gradient as probe   # noqa: E402

HAS_CHECKPOINT = probe.CHECKPOINT.is_file()


# --------------------------------------------------------------------------
# a synthetic tiny protocol-023 stream (array layout of the real collector)
# --------------------------------------------------------------------------
SYNTHETIC_PHASES = (("F0_baseline", 8), ("A1_compute", 12), ("B1_memory", 12),
                    ("C1_io", 12), ("F1_baseline", 8), ("A2_recur", 12),
                    ("C2_recur", 12), ("B2_recur", 12))
EXPECTED_SYNTHETIC_STEPS = sum(length for _, length in SYNTHETIC_PHASES)

#: fault class per host inside a regime phase: 16 hosts, three of them carry a
#: fault (1 = cpu, 2 = ram, 3 = disk) and the rest stay normal.
FAULT_HOSTS = {1: 1, 2: 2, 3: 3}


def synthetic_phases():
    cursor, phases = 0, []
    for name, length in SYNTHETIC_PHASES:
        phases.append({"name": name, "start": cursor, "end": cursor + length,
                       "length": length,
                       "regime_id": probe.PHASE_REGIME.get(name)})
        cursor += length
    return phases


def synthetic_raw_labels(steps, phases):
    """Mostly normal labels, with one fault run per regime phase."""
    raw = np.zeros((steps + 1, 16), dtype=np.int64)
    for phase in phases:
        regime = probe.PHASE_REGIME.get(phase["name"])
        if regime is None:
            continue
        for host, klass in FAULT_HOSTS.items():
            raw[phase["start"] + 3:phase["end"] - 3, host] = klass
    return raw


def synthetic_arrays(steps, phases, seed=17):
    rng = np.random.default_rng(seed)
    raw = synthetic_raw_labels(steps, phases)
    return {
        "host_features": (rng.random((steps + 1, 16, 7)) * 500.0 + 1.0)
                         .astype(np.float32),
        "demands": (rng.random((steps + 1, 16, 7)) * 500.0 + 1.0)
                   .astype(np.float32),
        "schedules": (rng.random((steps + 1, 16, 16)) > .5).astype(np.float32),
        "raw_labels": raw,
        "capacities": (rng.random((steps + 1, 16, 3)) * 9000.0 + 100.0),
        "creation_ids": np.repeat(np.arange(steps + 1, dtype=np.int64)[:, None],
                                  16, axis=1) % 7,
        "before_placement": np.zeros((steps + 1, 16), dtype=np.int64),
    }


def write_synthetic_stream(directory, seed=17):
    phases = synthetic_phases()
    steps = EXPECTED_SYNTHETIC_STEPS
    arrays = synthetic_arrays(steps, phases, seed=seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "stream.npz", **arrays)
    manifest = {
        "schema_version": 2,
        "protocol": "023",
        "phase": "P23-S2",
        "kind": "synthetic_tiny_stream_for_unit_tests",
        "registered": False,
        "smoke": True,
        "mode": "synthetic",
        "seed": 700,
        "steps": steps,
        "guard_steps": 1,
        "phases": phases,
        "stream_sha256": probe.sha256_file(directory / "stream.npz"),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return directory, manifest, arrays, steps


def synthetic_class_balance():
    return {"detection_weight": [1.0, 2.0], "resource_weight": [1.0, 1.0, 1.5],
            "source": "unit test"}


def tiny_events(steps, count=6):
    """A few mature host-steps spread over the synthetic regimes."""
    masks, windows = probe.regime_intervals(synthetic_phases(), steps)
    rows = np.nonzero(masks["compute_first"])[0]
    return [(int(rows[i]), 1) for i in range(0, min(count, len(rows)))]


# --------------------------------------------------------------------------
# 1. cosine math
# --------------------------------------------------------------------------
class CosineMathTest(unittest.TestCase):
    def test_identical_gradients_have_cosine_one(self):
        vector = np.array([0.5, -1.25, 3.0, 0.0])
        self.assertAlmostEqual(probe.cosine(vector, vector.copy()), 1.0, 12)

    def test_opposite_gradients_have_cosine_minus_one(self):
        vector = np.array([2.0, -3.0, 0.5])
        self.assertAlmostEqual(probe.cosine(vector, -vector), -1.0, 12)

    def test_orthogonal_gradients_have_zero_cosine(self):
        self.assertAlmostEqual(probe.cosine([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                               0.0, 12)
        self.assertAlmostEqual(probe.cosine([1.0, 1.0], [1.0, -1.0]), 0.0, 12)

    def test_cosine_is_scale_invariant(self):
        left = np.array([1.0, 2.0, -0.5])
        right = np.array([0.3, -0.7, 2.0])
        self.assertAlmostEqual(probe.cosine(left, right),
                               probe.cosine(1e6 * left, -2.0 * right) * -1.0,
                               12)

    def test_zero_norm_gradient_has_no_cosine(self):
        """A zero gradient carries no direction: null, never a fabricated 0.0."""
        self.assertIsNone(probe.cosine([0.0, 0.0], [1.0, 2.0]))
        self.assertIsNone(probe.cosine([1.0, 2.0], np.zeros(2)))

    def test_shape_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            probe.cosine([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_mean_gradient_averages_vectors(self):
        mean = probe.mean_gradient([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(mean, [2.0, 3.0])

    def test_mean_gradient_refuses_an_empty_matrix(self):
        with self.assertRaises(ValueError):
            probe.mean_gradient([])

    def test_cross_pair_cosines_uses_nan_for_zero_rows(self):
        matrix = probe.cross_pair_cosines([[1.0, 0.0], [0.0, 0.0]],
                                         [[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(matrix[0, 0], 1.0, 12)
        self.assertAlmostEqual(matrix[0, 1], 0.0, 12)
        self.assertTrue(np.isnan(matrix[1, 0]))
        self.assertTrue(np.isnan(matrix[1, 1]))

    def test_bootstrap_ci_brackets_the_mean(self):
        values = np.linspace(-0.5, 0.5, 400)
        low, high = probe.bootstrap_mean_ci(
            values, reps=200, rng=np.random.default_rng(3))
        self.assertLessEqual(low, float(values.mean()))
        self.assertGreaterEqual(high, float(values.mean()))
        self.assertLess(low, high)

    def test_bootstrap_seed_is_reproducible(self):
        values = np.linspace(-1.0, 1.0, 64)
        first = probe.bootstrap_mean_ci(values, reps=100,
                                        rng=np.random.default_rng(5))
        second = probe.bootstrap_mean_ci(values, reps=100,
                                         rng=np.random.default_rng(5))
        self.assertEqual(first, second)


# --------------------------------------------------------------------------
# 2. the registered pass/fail rule
# --------------------------------------------------------------------------
class GateRuleTest(unittest.TestCase):
    PAIRS = ("compute_first|memory_first", "compute_first|io_first",
             "memory_first|io_first")

    def rule(self, means, fractions):
        return probe.conflict_verdict(dict(zip(self.PAIRS, means)),
                                      dict(zip(self.PAIRS, fractions)))

    def test_registered_thresholds_match_protocol_json(self):
        """The gate constants must equal the registered artifact."""
        self.assertTrue(probe.PROTOCOL_JSON.is_file())
        registered = json.loads(probe.PROTOCOL_JSON.read_text(
            encoding="utf8"))["gradient_gate"]
        self.assertEqual(registered["pair_mean_cosine_max"],
                         probe.PAIR_MEAN_COSINE_MAX)
        self.assertEqual(registered["negative_pair_fraction_min"],
                         probe.NEGATIVE_PAIR_FRACTION_MIN)
        self.assertIn("<= -0.05", registered["rule"])
        self.assertIn("30%", registered["rule"])
        self.assertEqual(probe.CHECKPOINT_SHA256,
                         json.loads(probe.PROTOCOL_JSON.read_text(
                             encoding="utf8"))["frozen_checkpoint"]
                         ["sha256_expected"])

    def test_codirectional_gradients_are_not_conflict(self):
        verdict = self.rule([0.9, 0.8, 0.95], [0.0, 0.05, 0.0])
        self.assertFalse(verdict["conflict_present"])
        self.assertEqual(verdict["mean_cosine_conflict_pairs"], [])
        self.assertEqual(verdict["negative_fraction_conflict_pairs"], [])

    def test_mean_branch_boundary_at_exactly_minus_0_05(self):
        verdict = self.rule([-0.05, 0.4, 0.4], [0.0, 0.0, 0.1])
        self.assertTrue(verdict["conflict_present"])
        self.assertEqual(verdict["mean_cosine_conflict_pairs"], [self.PAIRS[0]])

    def test_mean_branch_just_above_the_boundary_is_not_conflict(self):
        verdict = self.rule([-0.0499999, 0.4, 0.4], [0.0, 0.0, 0.1])
        self.assertFalse(verdict["conflict_present"])

    def test_negative_fraction_branch_boundary_at_exactly_thirty_percent(self):
        verdict = self.rule([0.5, 0.5, 0.5], [0.0, 0.30, 0.0])
        self.assertTrue(verdict["conflict_present"])
        self.assertEqual(verdict["negative_fraction_conflict_pairs"],
                         [self.PAIRS[1]])
        self.assertEqual(verdict["mean_cosine_conflict_pairs"], [])

    def test_negative_fraction_just_below_the_boundary_is_not_conflict(self):
        verdict = self.rule([0.5, 0.5, 0.5], [0.0, 0.2999999, 0.0])
        self.assertFalse(verdict["conflict_present"])

    def test_both_branches_are_reported_separately(self):
        verdict = self.rule([-0.2, 0.6, 0.6], [0.0, 0.35, 0.0])
        self.assertTrue(verdict["conflict_present"])
        self.assertEqual(verdict["mean_cosine_conflict_pairs"], [self.PAIRS[0]])
        self.assertEqual(verdict["negative_fraction_conflict_pairs"],
                         [self.PAIRS[1]])

    def test_undefined_pair_counts_as_neither_branch(self):
        verdict = self.rule([None, None, None], [None, None, None])
        self.assertFalse(verdict["conflict_present"])
        self.assertEqual(sorted(verdict["undefined_pairs"]), sorted(self.PAIRS))

    def test_gate_verdict_names_the_stop_condition(self):
        stop = probe.gate_verdict(dict(zip(self.PAIRS, [0.9, 0.9, 0.9])),
                                  dict(zip(self.PAIRS, [0.0, 0.0, 0.0])))
        self.assertEqual(stop["verdict"], "STOP-GI")
        self.assertIn("do not implement D", stop["verdict_text"])
        self.assertEqual(stop["group"], probe.PRIMARY_GROUP)
        self.assertIn("-0.05", stop["registration"])

    def test_gate_verdict_reports_the_conflict(self):
        fired = probe.gate_verdict(dict(zip(self.PAIRS, [-0.3, 0.2, 0.2])),
                                   dict(zip(self.PAIRS, [0.1, 0.1, 0.1])))
        self.assertEqual(fired["verdict"], "conflict_present")
        self.assertTrue(fired["conflict_present"])

    def test_event_pair_statistics_count_negative_pairs(self):
        left = np.array([[1.0, 0.0], [1.0, 0.0]])          # identical rows
        right = np.array([[1.0, 0.0], [-1.0, 0.0]])        # 2 aligned, 2 flipped
        stats = probe.pair_statistics(left, right, "all", "A|B", seed=1,
                                      reps=50)
        self.assertEqual(stats["n_pairs"], 4)
        self.assertEqual(stats["n_pairs_defined"], 4)
        self.assertEqual(stats["negative_pairs"], 2)
        self.assertAlmostEqual(stats["negative_fraction"], 0.5, 12)
        self.assertAlmostEqual(stats["mean_cosine"], 0.0, 12)
        self.assertEqual(len(stats["bootstrap_ci95"]), 2)

    def test_event_pair_statistics_exclude_undefined_pairs(self):
        left = np.array([[1.0, 0.0], [0.0, 0.0]])
        right = np.array([[1.0, 0.0], [1.0, 0.0]])
        stats = probe.pair_statistics(left, right, "all", "A|B", seed=1,
                                      reps=20)
        self.assertEqual(stats["n_pairs"], 4)
        self.assertEqual(stats["n_pairs_undefined"], 2)
        self.assertEqual(stats["n_pairs_defined"], 2)
        self.assertEqual(stats["negative_pairs"], 0)
        self.assertAlmostEqual(stats["mean_cosine"], 1.0, 12)


# --------------------------------------------------------------------------
# 3. sampling
# --------------------------------------------------------------------------
class SamplingTest(unittest.TestCase):
    def candidates(self):
        return [(interval, host) for host in range(16)
                for interval in range(host, host + 40)]

    def test_fixed_seed_is_deterministic(self):
        first, description = probe.sample_events(self.candidates(), 64, seed=11)
        second, _ = probe.sample_events(self.candidates(), 64, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(description["n_sampled"], 64)

    def test_a_different_seed_changes_the_sample(self):
        first, _ = probe.sample_events(self.candidates(), 64, seed=11)
        second, _ = probe.sample_events(self.candidates(), 64, seed=12)
        self.assertNotEqual(first, second)

    def test_sample_is_balanced_across_hosts(self):
        sample, description = probe.sample_events(self.candidates(), 64,
                                                  seed=3)
        counts = description["per_host"]
        self.assertEqual(sum(counts), 64)
        self.assertEqual(len(counts), 16)
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sorted(counts), sorted([4] * 16))
        self.assertAlmostEqual(description["max_host_share"], 4 / 64.0, 12)

    def test_balance_holds_when_one_host_dominates_the_pool(self):
        """A host with thousands of candidates still cannot dominate."""
        candidates = [(interval, 0) for interval in range(5000)]
        candidates += [(interval, host) for host in range(1, 16)
                       for interval in range(0, 3)]
        sample, description = probe.sample_events(candidates, 32, seed=5)
        self.assertEqual(len(sample), 32)
        self.assertLessEqual(max(description["per_host"]) * 2, 32)

    def test_shortfall_is_reported_not_hidden(self):
        sample, description = probe.sample_events([(1, 2), (2, 2)], 64, seed=1)
        self.assertEqual(len(sample), 2)
        self.assertEqual(description["n_candidates"], 2)
        self.assertEqual(description["shortfall"], 62)

    def test_empty_pool_samples_nothing(self):
        sample, description = probe.sample_events([], 64, seed=1)
        self.assertEqual(sample, [])
        self.assertEqual(description["n_sampled"], 0)
        self.assertEqual(description["shortfall"], 64)

    def test_events_stay_inside_the_regime_phases(self):
        steps = EXPECTED_SYNTHETIC_STEPS
        arrays = synthetic_arrays(steps, synthetic_phases())
        labels, mature = probe.maturity_and_labels(arrays["raw_labels"], steps)
        masks, _ = probe.regime_intervals(synthetic_phases(), steps)
        candidates = probe.candidate_events(labels, mature, masks)
        for regime in probe.REGIME_IDS:
            self.assertTrue(candidates[regime], regime)
            for interval, host in candidates[regime]:
                self.assertTrue(masks[regime][interval])
                self.assertTrue(mature[interval, host])
                self.assertGreater(labels[interval, host], 0)


# --------------------------------------------------------------------------
# 4. labels and maturity
# --------------------------------------------------------------------------
class LabelConventionTest(unittest.TestCase):
    def test_maturity_requires_three_observed_rows(self):
        steps = 4
        raw = np.zeros((steps + 1, 3), dtype=np.int64)
        raw[:, 0] = 1
        raw[2, 1] = -1                     # host 1 is unobserved at interval 2
        labels, mature = probe.maturity_and_labels(raw, steps)
        self.assertTrue(mature[0, 1])
        self.assertFalse(mature[1, 1])     # t-1 = 0 ... t+1 = 2 unobserved
        self.assertFalse(mature[2, 1])
        self.assertFalse(mature[3, 1])
        self.assertEqual(labels[1, 1], -1)
        self.assertTrue(mature[:, 2].all())

    def test_tolerance_rule_matches_the_frozen_online_function(self):
        from run_ftmoe_online import tolerance_label

        steps = 6
        raw = np.zeros((steps + 1, 4), dtype=np.int64)
        raw[1:3, 0] = 2                    # a RAM fault run
        raw[3, 1] = 3                      # an isolated disk fault
        raw[2, 2] = 0                      # between two faults (tie -> earlier)
        raw[1, 2] = 1
        raw[3, 2] = 3
        labels, mature = probe.maturity_and_labels(raw, steps)
        for interval in range(steps):
            row = tolerance_label(raw, interval, interval + 1)
            for host in range(4):
                if not mature[interval, host]:
                    continue
                self.assertEqual(labels[interval, host], int(row[host]),
                                 "interval %d host %d" % (interval, host))
        self.assertEqual(labels[0, 0], 2)   # filled forward from interval 1
        self.assertEqual(labels[2, 1], 3)   # filled forward from interval 3
        self.assertEqual(labels[2, 2], 1)   # equal-distance tie -> earlier

    def test_label_scope_selects_faults_or_everything(self):
        steps = EXPECTED_SYNTHETIC_STEPS
        arrays = synthetic_arrays(steps, synthetic_phases())
        labels, mature = probe.maturity_and_labels(arrays["raw_labels"], steps)
        masks, _ = probe.regime_intervals(synthetic_phases(), steps)
        faults = probe.candidate_events(labels, mature, masks, "fault")
        everything = probe.candidate_events(labels, mature, masks, "all")
        for regime in probe.REGIME_IDS:
            self.assertLess(len(faults[regime]), len(everything[regime]))
            self.assertTrue(all(labels[t, h] > 0 for t, h in faults[regime]))
        with self.assertRaises(ValueError):
            probe.candidate_events(labels, mature, masks, "sometimes")


# --------------------------------------------------------------------------
# 5. regime segmentation
# --------------------------------------------------------------------------
class RegimeSegmentationTest(unittest.TestCase):
    def registered_phases(self):
        cursor, phases = 0, []
        for name, length in probe.REGISTERED_DEV_PHASES:
            phases.append({"name": name, "start": cursor, "end": cursor + length,
                           "length": length,
                           "regime_id": probe.PHASE_REGIME.get(name)})
            cursor += length
        return phases

    def test_registered_phase_names_map_to_the_registered_regimes(self):
        phases = self.registered_phases()
        self.assertEqual(len(phases), 8)
        masks, windows = probe.regime_intervals(phases,
                                               probe.REGISTERED_DEV_STEPS)
        self.assertEqual([item[0] for item in windows["compute_first"]],
                         ["A1_compute", "A2_recur"])
        self.assertEqual([item[0] for item in windows["memory_first"]],
                         ["B1_memory", "B2_recur"])
        self.assertEqual([item[0] for item in windows["io_first"]],
                         ["C1_io", "C2_recur"])
        self.assertTrue(masks["compute_first"][300])
        self.assertFalse(masks["compute_first"][720])
        self.assertTrue(masks["memory_first"][720])
        self.assertTrue(masks["io_first"][1140])
        self.assertFalse(masks["io_first"][0])       # F0_baseline is familiar
        self.assertTrue(masks["compute_first"][1800])
        self.assertTrue(masks["memory_first"][2520])
        self.assertEqual(int(masks["compute_first"].sum()), 420 + 360)
        self.assertEqual(int(masks["memory_first"].sum()), 420 + 360)
        self.assertEqual(int(masks["io_first"].sum()), 420 + 360)

    def test_registered_timeline_check_accepts_the_plan_timeline(self):
        check = probe.registered_timeline_check(self.registered_phases(),
                                                probe.REGISTERED_DEV_STEPS)
        self.assertTrue(check["matches_registered_timeline"])
        self.assertEqual(check["registered_steps"], 2880)

    def test_registered_timeline_check_rejects_another_shape(self):
        phases = synthetic_phases()
        check = probe.registered_timeline_check(phases,
                                                EXPECTED_SYNTHETIC_STEPS)
        self.assertFalse(check["matches_registered_timeline"])

    def test_a_manifest_regime_that_contradicts_the_table_is_refused(self):
        phases = synthetic_phases()
        phases[1]["regime_id"] = "io_first"          # A1_compute claims io_first
        with self.assertRaises(ValueError):
            probe.regime_intervals(phases, EXPECTED_SYNTHETIC_STEPS)

    def test_a_missing_regime_is_refused(self):
        phases = [phase for phase in synthetic_phases()
                  if probe.PHASE_REGIME.get(phase["name"]) != "memory_first"]
        with self.assertRaises(ValueError):
            probe.regime_intervals(phases, EXPECTED_SYNTHETIC_STEPS)

    def test_non_contiguous_phases_are_refused(self):
        phases = synthetic_phases()
        phases[2]["start"] += 1
        with self.assertRaises(ValueError):
            probe.normalize_phases(phases)


class TemporaryStreamMixin(object):
    """A synthetic stream in a temporary directory, removed after the test."""

    def temp_stream(self):
        root = Path(tempfile.mkdtemp(prefix="p23_gradient_test_"))
        self.addCleanup(shutil.rmtree, str(root), True)
        return write_synthetic_stream(root / "dev_synthetic")


# --------------------------------------------------------------------------
# 6. the model: residual parameter identification and "nothing is updated"
# --------------------------------------------------------------------------
@unittest.skipUnless(HAS_CHECKPOINT, "frozen checkpoint is absent")
class ResidualParameterTest(TemporaryStreamMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE

        probe.configure_runtime()
        cls.model, cls.checkpoint, _ = probe.build_model()
        cls.layout = probe.learner_layout(cls.model)
        cls.FrozenResidualFTMoE = FrozenResidualFTMoE

    def test_online_trainable_parameters_are_exactly_the_learner_bank(self):
        model = self.model
        trainable = [name for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("learner.") for name in trainable))
        bank = set(model.learner.state_dict())
        self.assertEqual(set(name[len("learner."):] for name in trainable), bank)
        # the registered R1 identification: correction_parameters -> the bank
        self.assertEqual(set(trainable),
                         set("learner." + name for name in
                             dict(model.learner.named_parameters())))
        self.assertEqual(len(model.correction_parameters("learner")),
                         len(trainable))
        self.assertEqual(bank, set(dict(model.learner.named_parameters())))

    def test_frozen_base_parameters_are_outside_every_group(self):
        self.assertFalse(self.layout["unexpected_trainable"])
        self.assertGreater(len(self.layout["frozen_names"]), 100)
        self.assertIn("encoder.input_proj.weight", self.layout["frozen_names"])
        self.assertIn("moe.router.weight", self.layout["frozen_names"])
        for group, names in self.layout["groups"].items():
            for name in names:
                self.assertTrue(name.startswith("learner."),
                                "%s leaked into %s" % (name, group))

    def test_only_the_learner_bank_is_trainable(self):
        banks = self.layout["banks"]
        self.assertEqual(banks["learner"]["n_trainable_tensors"],
                         banks["learner"]["n_tensors"])
        for bank in ("live", "assessment", "previous"):
            self.assertEqual(banks[bank]["n_trainable_tensors"], 0)
            self.assertEqual(banks[bank]["n_tensors"],
                             banks["learner"]["n_tensors"])

    def test_registered_groups_are_present_and_disjoint_inside_all(self):
        groups = self.layout["groups"]
        for group in ("all", "router", "experts", "expert0", "expert1",
                      "expert2", "expert3", "layer:learner.router",
                      "layer:learner.experts.0.3"):
            self.assertIn(group, groups)
        self.assertEqual(set(groups["router"]) | set(groups["experts"]),
                         set(groups["all"]))
        self.assertEqual(len(groups["experts"]),
                         4 * len(self.layout["groups"]["expert0"]))
        self.assertEqual(self.layout["width"],
                         sum(parameter.numel()
                             for parameter in self.model.parameters()
                             if parameter.requires_grad))
        self.assertEqual(self.layout["width"], 9752)

    def test_per_layer_decomposition_is_available(self):
        """The registered per-layer report is possible without frozen edits."""
        layers = [name for name in self.layout["groups"]
                  if name.startswith("layer:")]
        self.assertEqual(len(layers), 1 + 4 * 3)     # router + 4 x 3 modules
        self.assertEqual(sorted(name for name in layers
                                if name.startswith("layer:learner.experts.0")),
                         ["layer:learner.experts.0.0",
                          "layer:learner.experts.0.1",
                          "layer:learner.experts.0.3"])

    def test_measurement_changes_no_parameter(self):
        steps = EXPECTED_SYNTHETIC_STEPS
        _, _, replay, _, _, _ = probe.load_stream(self.temp_stream()[0])
        arrays = synthetic_arrays(steps, synthetic_phases())
        labels, _ = probe.maturity_and_labels(arrays["raw_labels"], steps)
        before = probe.state_hashes(self.model)
        digest_before = probe.digest_of(before)
        frozen_before = self.model.frozen_hash()
        learner_before = self.model.learner_state_hash()
        measured = probe.measure_events(self.model, replay, self.layout, labels,
                                        tiny_events(steps),
                                        synthetic_class_balance())
        self.assertEqual(measured["matrix"].shape, (6, self.layout["width"]))
        self.assertTrue(np.isfinite(measured["matrix"]).all())
        self.assertTrue(all(np.isfinite(value)
                            for value in measured["losses"]))
        after = probe.state_hashes(self.model)
        self.assertEqual(probe.digest_of(after), digest_before)
        self.assertEqual(after, before)
        self.assertEqual(self.model.frozen_hash(), frozen_before)
        self.assertEqual(self.model.learner_state_hash(), learner_before)
        self.assertEqual([name for name, parameter
                          in self.model.named_parameters()
                          if parameter.grad is not None], [])

    def test_learner_state_loading_touches_only_the_residual_bank(self):
        """``--learner-state`` may replace the residual, never the frozen base."""
        model = self.FrozenResidualFTMoE(self.checkpoint, "C",
                                         int(self.checkpoint["seed"]))
        state = {"learner." + name: value.clone()
                 for name, value in model.learner.state_dict().items()}
        for value in state.values():
            if value.is_floating_point():
                value.add_(0.25)
        root = Path(tempfile.mkdtemp(prefix="p23_gradient_test_"))
        self.addCleanup(shutil.rmtree, str(root), True)
        path = root / "mature_learner.pt"
        torch.save(state, path)
        frozen_before = model.frozen_hash()
        learner_before = model.learner_state_hash()
        info = probe.load_learner_state(model, path)
        self.assertEqual(info["n_tensors"], len(state))
        self.assertEqual(model.frozen_hash(), frozen_before)
        self.assertNotEqual(model.learner_state_hash(), learner_before)
        self.assertEqual(len(info["learner_sha256"]), 64)
        self.assertTrue(info["path"].endswith("mature_learner.pt"))
        trainable = [name for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        self.assertTrue(all(name.startswith("learner.") for name in trainable))

    def test_an_incomplete_learner_state_is_refused(self):
        model = self.FrozenResidualFTMoE(self.checkpoint, "C",
                                         int(self.checkpoint["seed"]))
        state = {"learner." + name: value.clone()
                 for name, value in model.learner.state_dict().items()}
        state.pop(sorted(state)[0])
        root = Path(tempfile.mkdtemp(prefix="p23_gradient_test_"))
        self.addCleanup(shutil.rmtree, str(root), True)
        path = root / "incomplete.pt"
        torch.save(state, path)
        with self.assertRaises(ValueError):
            probe.load_learner_state(model, path)

    def test_the_residual_correction_is_a_no_op_at_the_start_state(self):
        """The reason the hidden-layer gradients are structurally zero.

        ``FixedResidualExpert`` zeroes its final layer, so the bank's output is
        exactly zero at construction (ftmoe_online_r1.py:80-84).  The probe must
        see that, because it is what makes the router and hidden-layer blocks
        degenerate until the residual has been trained.
        """
        steps = EXPECTED_SYNTHETIC_STEPS
        arrays = synthetic_arrays(steps, synthetic_phases())
        _, _, replay, _, _, _ = probe.load_stream(self.temp_stream()[0])
        host_window, schedule, graph, ids, before, caps = replay.window_v3(
            tiny_events(steps)[0][0])
        with torch.no_grad():
            output = self.model(
                host_window.unsqueeze(0), schedule.unsqueeze(0),
                graph.unsqueeze(0),
                graph_context={"creation_ids": ids.unsqueeze(0),
                               "before_placement": before.unsqueeze(0),
                               "capacities": caps.unsqueeze(0)})
        self.assertEqual(float(output["correction_logits"].abs().max()), 0.0)
        self.assertEqual(float(output["base_final_detection_logits"]
                               .sub(output["detection_logits"]).abs().max()),
                         0.0)

    def test_the_start_state_residual_gradient_is_structurally_sparse(self):
        """Measured, not assumed: the R1 start bank zeroes its read-out layer.

        Only the expert read-out layers and the registered router balance term
        carry a gradient until the residual has been trained; the hidden layers
        are multiplied by a zero matrix.  The instrument must see this and
        report it as a degeneracy rather than as 'no conflict'.
        """
        steps = EXPECTED_SYNTHETIC_STEPS
        arrays = synthetic_arrays(steps, synthetic_phases())
        labels, _ = probe.maturity_and_labels(arrays["raw_labels"], steps)
        _, _, replay, _, _, _ = probe.load_stream(self.temp_stream()[0])
        measured = probe.measure_events(self.model, replay, self.layout, labels,
                                        tiny_events(steps),
                                        synthetic_class_balance())
        matrix = measured["matrix"]
        for name in ("learner.experts.0.0.weight", "learner.experts.0.1.weight"):
            start, end = self.layout["offsets"][name]
            self.assertEqual(float(np.abs(matrix[:, start:end]).max()), 0.0,
                             "%s should be structurally zero at the start "
                             "state" % name)
        start, end = self.layout["offsets"]["learner.experts.0.3.weight"]
        self.assertGreater(float(np.abs(matrix[:, start:end]).max()), 0.0)
        start, end = self.layout["offsets"]["learner.router.weight"]
        self.assertGreater(float(np.abs(matrix[:, start:end]).max()), 0.0)


# --------------------------------------------------------------------------
# 7. the NaN guard
# --------------------------------------------------------------------------
class NanGuardTest(TemporaryStreamMixin, unittest.TestCase):
    def test_cosine_refuses_nan(self):
        with self.assertRaises(probe.NonFiniteGradientError):
            probe.cosine([1.0, np.nan], [1.0, 2.0])

    def test_cosine_refuses_infinity(self):
        with self.assertRaises(probe.NonFiniteGradientError):
            probe.cosine([1.0, np.inf], [1.0, 2.0])

    def test_mean_gradient_refuses_one_bad_event(self):
        with self.assertRaises(probe.NonFiniteGradientError):
            probe.mean_gradient([[1.0, 2.0], [np.nan, 1.0], [0.5, 0.5]])

    def test_cross_pair_cosines_refuses_one_bad_event(self):
        with self.assertRaises(probe.NonFiniteGradientError):
            probe.cross_pair_cosines([[1.0, 0.0], [np.nan, 0.0]],
                                     [[1.0, 0.0]])

    def test_pair_statistics_refuses_nan(self):
        with self.assertRaises(probe.NonFiniteGradientError):
            probe.pair_statistics([[np.nan, 0.0]], [[1.0, 0.0]], "all", "A|B",
                                  seed=1, reps=10)

    def test_json_writer_refuses_non_finite_values(self):
        with self.assertRaises(probe.NonFiniteGradientError):
            probe.jsonable({"mean_cosine": float("nan")})

    @unittest.skipUnless(HAS_CHECKPOINT, "frozen checkpoint is absent")
    def test_a_nan_forward_pass_raises_instead_of_producing_nan_cosines(self):
        """End to end: poison one input cell and the probe must refuse.

        Two layers are pinned here: the replay builder already refuses a
        non-finite stream, and if a non-finite value still reaches the forward
        pass, the gradient guard raises instead of returning NaN cosines.
        """
        steps = EXPECTED_SYNTHETIC_STEPS
        directory, _, arrays, _ = self.temp_stream()
        model, _, _ = probe.build_model()
        layout = probe.learner_layout(model)
        _, _, replay, _, _, _ = probe.load_stream(directory)
        labels, _ = probe.maturity_and_labels(arrays["raw_labels"], steps)
        poisoned = dict(arrays)
        poisoned["host_features"] = arrays["host_features"].copy()
        poisoned["host_features"][7, 4, 3] = np.nan
        saved = replay.arrays["host_features"]
        try:
            replay.arrays["host_features"] = poisoned["host_features"]
            with self.assertRaises(probe.NonFiniteGradientError):
                probe.measure_events(model, replay, layout, labels,
                                     [(7, 4)], synthetic_class_balance())
        finally:
            replay.arrays["host_features"] = saved
            replay.arrays["host_features"][7, 4, 3] = arrays[
                "host_features"][7, 4, 3]
        np.savez_compressed(Path(directory) / "stream.npz", **poisoned)
        with self.assertRaises(AssertionError):
            probe.load_stream(directory)          # the stream hash guard
        manifest = json.loads((Path(directory) / "manifest.json").read_text(
            encoding="utf8"))
        manifest["stream_sha256"] = probe.sha256_file(
            Path(directory) / "stream.npz")
        (Path(directory) / "manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf8")
        with self.assertRaises(ValueError):
            probe.load_stream(directory)          # the non-finite guard


# --------------------------------------------------------------------------
# 8. resource guard
# --------------------------------------------------------------------------
class ResourceGuardTest(unittest.TestCase):
    def test_ram_guard_aborts_below_the_registered_headroom(self):
        with self.assertRaises(RuntimeError) as caught:
            probe.check_ram(10 ** 6, "unit-test", [])
        self.assertIn("RAM guard ABORT", str(caught.exception))

    def test_ram_guard_records_its_samples(self):
        samples = []
        probe.check_ram(0.0, "unit-test", samples)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["stage"], "unit-test")
        self.assertGreater(samples[0]["available_gib"], 0.0)


# --------------------------------------------------------------------------
# 9. end to end on the synthetic tiny stream
# --------------------------------------------------------------------------
@unittest.skipUnless(HAS_CHECKPOINT, "frozen checkpoint is absent")
class SyntheticEndToEndTest(unittest.TestCase):
    """The whole instrument, on a synthetic stream (the real dev stream is not
    collected in this checkout)."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.stream, cls.manifest, cls.arrays, cls.steps = \
            write_synthetic_stream(cls.root / "dev_synthetic")
        cls.report = probe.run_probe(
            stream_dir=cls.stream, out_path=cls.root / "gradient.json",
            n_events=6, seed=4242, class_balance=synthetic_class_balance(),
            registered_timeline=False, bootstrap_reps=50,
            ram_guard_gib=0.05, write=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_report_carries_the_registered_fields(self):
        report = self.report
        for key in ("registration", "stream", "checkpoint", "learner",
                    "label_convention", "loss_definition", "sampling", "groups",
                    "per_layer", "verdict", "parameter_integrity",
                    "degenerate_groups", "command_line", "elapsed_seconds",
                    "n_events_per_regime", "n_pairs", "seed"):
            self.assertIn(key, report)
        self.assertEqual(report["registration"]["pair_mean_cosine_max"], -0.05)
        self.assertEqual(report["registration"]["negative_pair_fraction_min"],
                         0.30)
        self.assertEqual(report["checkpoint"]["matches_registration"], True)
        self.assertEqual(report["sampling"]["events_per_regime_requested"], 6)
        self.assertEqual(report["sampling"]["n_events_total_sampled"], 18)
        self.assertEqual(report["sampling"]["seed"], 4242)
        self.assertEqual(report["seed"], 4242)
        self.assertEqual(report["n_events_per_regime"],
                         {regime: 6 for regime in probe.REGIME_IDS})
        self.assertEqual(sorted(report["n_pairs"]),
                         sorted(probe.pair_name(*pair)
                                for pair in probe.regime_pairs()))
        self.assertEqual(set(report["n_pairs"].values()), {36})
        self.assertEqual(report["verdict"]["n_events_per_regime"],
                         report["n_events_per_regime"])
        self.assertEqual(report["verdict"]["n_pairs"], report["n_pairs"])

    def test_every_registered_group_is_reported(self):
        groups = self.report["groups"]
        for group in ("all", "router", "experts", "expert0", "expert3",
                      "layer:learner.router", "layer:learner.experts.0.3"):
            self.assertIn(group, groups)
            self.assertEqual(len(groups[group]["pairs"]), 3)
        for entry in groups["all"]["pairs"]:
            self.assertEqual(entry["n_events_left"], 6)
            self.assertEqual(entry["n_events_right"], 6)
            self.assertEqual(entry["n_pairs"], 36)
            self.assertEqual(len(entry["bootstrap_ci95"]), 2)
            self.assertLessEqual(entry["bootstrap_ci95"][0],
                                 entry["mean_cosine"])
            self.assertLessEqual(entry["mean_cosine"],
                                 entry["bootstrap_ci95"][1])
        self.assertEqual(sorted(groups["all"]["pair_mean_cosine"]),
                         sorted([probe.pair_name(*pair)
                                 for pair in probe.regime_pairs()]))

    def test_per_layer_decomposition_is_reported(self):
        per_layer = self.report["per_layer"]
        self.assertTrue(per_layer["available"])
        self.assertEqual(per_layer["n_groups"], 13)
        self.assertIn("layer:learner.experts.3.1", per_layer["groups"])

    def test_the_verdict_matches_the_registered_rule(self):
        verdict = self.report["verdict"]
        primary = self.report["groups"][probe.PRIMARY_GROUP]
        recomputed = probe.conflict_verdict(primary["pair_mean_cosine"],
                                            primary["pair_negative_fraction"])
        self.assertEqual(verdict["conflict_present"],
                         recomputed["conflict_present"])
        # The registered start state carries an untrained residual bank (zeroed
        # read-out), so the registered rule is evaluated and reported, but the
        # verdict is marked inadmissible rather than presented as the §14
        # result for the fixed learner.
        admissibility = verdict["admissibility"]
        self.assertIn(verdict["verdict"],
                      ("conflict_present", "STOP-GI", "INADMISSIBLE"))
        if not admissibility["scientifically_admissible"]:
            self.assertEqual(verdict["verdict"], "INADMISSIBLE")
            self.assertIn(verdict["registered_gate_at_this_state"],
                          ("conflict_present", "STOP-GI"))
        else:
            self.assertIn(verdict["verdict"], ("conflict_present", "STOP-GI"))
        self.assertIsInstance(verdict["conflict_present_any_group"], bool)
        self.assertEqual(verdict["group"], probe.PRIMARY_GROUP)

    def test_an_untrained_learner_cannot_pass_the_gate(self):
        """A degenerate (start-state) measurement is never a §14 verdict."""
        verdict = self.report["verdict"]
        admissibility = verdict["admissibility"]
        if admissibility["degenerate_groups"]:
            self.assertFalse(admissibility["scientifically_admissible"])
            self.assertEqual(verdict["verdict"], "INADMISSIBLE")
            self.assertIn("mature", admissibility["what_would_make_it_admissible"])

    def test_no_parameter_or_buffer_changed(self):
        integrity = self.report["parameter_integrity"]
        self.assertTrue(integrity["unchanged"])
        self.assertEqual(integrity["state_sha256_before"],
                         integrity["state_sha256_after"])
        self.assertEqual(integrity["learner_sha256_before"],
                         integrity["learner_sha256_after"])
        self.assertEqual(integrity["frozen_base_sha256_before"],
                         integrity["frozen_base_sha256_after"])
        self.assertEqual(integrity["changed_tensors"], [])
        self.assertEqual(integrity["grad_attribute_written"], [])
        self.assertFalse(integrity["optimizer_constructed"])
        self.assertEqual(integrity["backward_calls"], 0)
        self.assertEqual(integrity["gradient_calls"], 18)
        self.assertEqual(integrity["parameter_sha256_before"],
                         integrity["parameter_sha256_after"])
        self.assertEqual(len(integrity["parameter_sha256_before"]),
                         integrity["n_state_tensors"])
        self.assertGreater(len(integrity["parameter_sha256_before"]), 150)

    def test_start_state_degeneracy_is_reported_not_hidden(self):
        """The frozen start bank is a residual no-op: say so, do not fake 0.0.

        Measured structure of the R1 start state: the expert hidden layers are
        structurally zero (multiplied by the zeroed read-out matrix), the
        read-out layer carries the signal, and the router receives only the
        registered balance term.  The instrument must report the zero groups as
        degenerate instead of reading their null cosine as 'no conflict'.
        """
        degenerate = self.report["degenerate_groups"]
        for expert in range(4):
            self.assertIn("layer:learner.experts.%d.0" % expert, degenerate)
            self.assertIn("layer:learner.experts.%d.1" % expert, degenerate)
            self.assertNotIn("layer:learner.experts.%d.3" % expert, degenerate)
        for group in ("all", "router", "experts", "layer:learner.router"):
            self.assertNotIn(group, degenerate, group)
        for name in degenerate:
            for pair, value in self.report["groups"][name][
                    "pair_mean_cosine"].items():
                self.assertIsNone(value, "%s/%s should be null" % (name, pair))
            for entry in self.report["groups"][name]["pairs"]:
                self.assertEqual(entry["n_pairs_defined"], 0)
                self.assertIsNone(entry["negative_fraction"])
                self.assertIsNone(entry["bootstrap_ci95"])
        self.assertIn("degeneracy_note", self.report)
        self.assertEqual(self.report["verdict"]["conflict_present"], False)

    def test_the_router_gradient_comes_only_from_the_balance_term(self):
        """Measured, not assumed: the read-out block dominates the router."""
        per_regime = self.report["sampling"]["gradient_norms_per_regime"]
        for regime, norms in per_regime.items():
            self.assertEqual(len(norms), 6)
            self.assertTrue(all(value > 0 for value in norms), regime)
        router = self.report["groups"]["router"]["regime_mean_norms"]
        read_out = self.report["groups"]["layer:learner.experts.0.3"][
            "regime_mean_norms"]
        for regime in probe.REGIME_IDS:
            self.assertGreater(float(router[regime]), 0.0)
            self.assertLess(float(router[regime]), float(read_out[regime]))

    def test_the_primary_group_has_defined_cosines(self):
        primary = self.report["groups"][probe.PRIMARY_GROUP]
        for pair, value in primary["pair_mean_cosine"].items():
            self.assertIsNotNone(value, pair)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)
        for entry in primary["pairs"]:
            self.assertEqual(entry["n_pairs_defined"], 36)
            self.assertEqual(entry["n_pairs_undefined"], 0)

    def test_the_written_json_is_strict_and_readable(self):
        path = Path(self.report["written"])
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf8")
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        payload = json.loads(text)
        self.assertEqual(payload["protocol"], "023")
        self.assertEqual(payload["verdict"]["conflict_present"],
                         self.report["verdict"]["conflict_present"])

    def test_the_probe_never_writes_outside_its_output_path(self):
        """The registered artifacts tree is untouched by a synthetic run."""
        self.assertEqual(Path(self.report["written"]).parent, self.root)
        self.assertFalse((self.root / "gradient.json.tmp").exists())

    def test_events_are_inside_the_regime_phases_and_mature(self):
        steps = self.steps
        arrays = self.arrays
        labels, mature = probe.maturity_and_labels(arrays["raw_labels"], steps)
        masks, _ = probe.regime_intervals(synthetic_phases(), steps)
        for regime in probe.REGIME_IDS:
            events = self.report["sampling"]["per_regime"][regime]["events"]
            self.assertEqual(len(events), 6)
            for interval, host, label in events:
                self.assertTrue(masks[regime][interval])
                self.assertTrue(mature[interval, host])
                self.assertEqual(label, int(labels[interval, host]))
                self.assertGreater(label, 0)


# --------------------------------------------------------------------------
# 10. the registered class-balanced weights
# --------------------------------------------------------------------------
class ClassBalanceTest(unittest.TestCase):
    def test_weights_come_from_the_protocol_020_training_bundle(self):
        balance = probe.registered_class_balance()
        self.assertEqual(sorted(balance), sorted(
            ["counts", "detection_weight", "resource_weight", "formula",
             "source"]))
        self.assertEqual(balance["detection_weight"][0], 1.0)
        self.assertGreaterEqual(balance["detection_weight"][1], 2.0)
        self.assertLessEqual(balance["detection_weight"][1], 10.0)
        self.assertEqual(len(balance["resource_weight"]), 3)
        self.assertIn("same-domain training bundle", balance["source"])
        self.assertIn("same-domain", balance["formula"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
