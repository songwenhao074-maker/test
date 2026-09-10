"""Protocol 021 — audit-layer tests (U4/U3/U-probe integrity, no model).

Covers the remaining data-layer rows of the protocol's required-test table
(§34) that the mechanism tests cannot reach:

    07 Window overlap      exact scoring-window hashing detects identity/sharing
    08 Event segmentation  a sustained same-host fault is ONE independent event
    09 Probe causality     probe features are deployable; the matured-label
                           feature is offset away from the target; temporal
                           split only

Run from the repository root:
    python test_ftmoe_protocol021_audit.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_ftmoe_protocol021_unseen as u4
import probe_ftmoe_protocol021_learnability as probe
from prepare_ftmoe_protocol021_unseen import run_segmentation

ROOT = Path(__file__).resolve().parent


class WindowOverlapTests(unittest.TestCase):
    def test_shared_window_is_detected(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(40, 4, 7))
        b = a.copy()
        self.assertGreater(len(u4.window_hashes(a) & u4.window_hashes(b)), 0)

    def test_disjoint_streams_have_zero_overlap(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(40, 4, 7))
        b = rng.normal(size=(40, 4, 7))
        self.assertEqual(len(u4.window_hashes(a) & u4.window_hashes(b)), 0)

    def test_shifted_copy_still_shares_windows(self):
        rng = np.random.default_rng(11)
        a = rng.normal(size=(60, 4, 7))
        b = a[5:]
        self.assertGreater(len(u4.window_hashes(a) & u4.window_hashes(b)), 0)


class EventSegmentationTests(unittest.TestCase):
    def test_sustained_fault_is_one_event(self):
        labels = np.zeros((100, 4), dtype=np.int64)
        labels[10:30, 2] = 2          # one 20-step RAM fault on host 2
        runs = run_segmentation(labels, 100)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["host"], 2)
        self.assertEqual(runs[0]["class"], 2)
        self.assertEqual(runs[0]["duration"], 20)
        self.assertEqual(runs[0]["start"], 10)

    def test_class_change_splits_events(self):
        labels = np.zeros((60, 2), dtype=np.int64)
        labels[5:10, 0] = 1
        labels[10:15, 0] = 3          # class change -> new event
        runs = run_segmentation(labels, 60)
        self.assertEqual(len(runs), 2)
        self.assertEqual([r["class"] for r in runs], [1, 3])

    def test_separated_faults_are_separate_events(self):
        labels = np.zeros((80, 3), dtype=np.int64)
        labels[5:8, 1] = 2
        labels[40:44, 1] = 2
        runs = run_segmentation(labels, 80)
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(r["host"] == 1 for r in runs))

    def test_steps_limit_is_respected(self):
        labels = np.zeros((80, 2), dtype=np.int64)
        labels[70:75, 0] = 1          # beyond the scored horizon
        self.assertEqual(run_segmentation(labels, 60), [])


class ProbeCausalityTests(unittest.TestCase):
    """Build a synthetic stream so the label-offset invariant can be asserted."""

    def _write_stream(self, path, steps=30, hosts=2):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(steps + 1, hosts, 7)) + 100.0
        capacities = np.full((steps + 1, hosts, 3), 1000.0)
        labels = np.zeros((steps + 1, hosts), dtype=np.int64)
        labels[1::2, 0] = 1           # alternating fault on host 0
        labels[::3, 1] = 2
        schedules = np.full((steps + 1, hosts, hosts), 1.0 / hosts)
        after = np.tile(np.arange(hosts), (steps + 1, 1))
        np.savez_compressed(path, host_features=features, capacities=capacities,
                            raw_labels=labels, schedules=schedules,
                            after_placement=after)
        return labels

    def test_feature_names_have_no_forbidden_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "stream.npz"
            self._write_stream(stream)
            for target in ("next", "same"):
                ds = probe.build_matrix(stream, 30, target=target)
                for name in ds["names"]:
                    for token in probe.FORBIDDEN:
                        self.assertNotIn(token, name)

    def test_matured_label_offset_matches_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "stream.npz"
            labels = self._write_stream(stream)
            steps, hosts = 30, 2
            for target, offset in (("next", 0), ("same", 1)):
                ds = probe.build_matrix(stream, steps, target=target)
                for i in range(ds["X"].shape[0]):
                    t, h = int(ds["t"][i]), int(ds["host"][i])
                    if target == "same" and t == 0:
                        continue
                    expected = labels[t - offset, h]
                    observed = int(np.argmax(ds["X"][i, -4:]))
                    self.assertEqual(observed, expected,
                                     "past-fault feature is not offset correctly")

    def test_target_is_shifted_by_one_for_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "stream.npz"
            labels = self._write_stream(stream)
            ds = probe.build_matrix(stream, 30, target="next")
            for i in range(ds["X"].shape[0]):
                t, h = int(ds["t"][i]), int(ds["host"][i])
                self.assertEqual(int(ds["y"][i]), int(labels[t + 1, h]))
                self.assertLess(t, 29)

    def test_split_is_temporal_and_never_shuffled(self):
        bounds = probe.SPLITS
        self.assertEqual([name for name, _, _ in bounds], ["fit", "dev", "test"])
        lows = [lo for _, lo, _ in bounds]
        highs = [hi for _, _, hi in bounds]
        self.assertEqual(lows, sorted(lows))
        self.assertEqual(highs, sorted(highs))
        for (_, _, hi), (_, lo, _) in zip(bounds, bounds[1:]):
            self.assertEqual(hi, lo, "split boundaries must be contiguous")


class U4ReferenceTests(unittest.TestCase):
    def test_reference_definitions_present(self):
        path = u4.AUDIT / "offline_reference.json"
        if not path.is_file():
            self.skipTest("offline reference not built yet")
        ref = json.loads(path.read_text(encoding="utf8"))
        for key in ("corpora", "cross_lag_95", "elevation_transitions", "definitions"):
            self.assertIn(key, ref)
        for pair in ("cpu_ram", "cpu_disk", "ram_disk"):
            self.assertIn(pair, ref["cross_lag_95"])
            self.assertIn("4", ref["cross_lag_95"][pair])
            self.assertIn("8", ref["cross_lag_95"][pair])

    def test_gate_requires_all_five_checks(self):
        if not (u4.AUDIT / "selected.json").is_file():
            self.skipTest("no selected.json yet")
        payload = json.loads((u4.AUDIT / "selected.json").read_text(encoding="utf8"))
        self.assertIn("selection_rule", payload)
        self.assertIn("data physics", payload["selection_rule"])
        for candidate in payload["candidates"]:
            self.assertIn("u4_gate_passed", candidate)
            self.assertIn("data_gate_passed", candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
