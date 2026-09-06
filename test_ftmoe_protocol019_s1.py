"""Protocol 019 S1 acceptance tests.

Covers the shared-input repair layer only (指令/FTMOE_ONLINE_TUNING_REVIEW_
AND_SOLUTION_PLAN.md §19 Tests 1-4, plus legacy-parity regression):

- Test 1: normalization zero coverage (no 1e8/1e11 amplification)
- Test 2: identity replacement is NOT a migration
- Test 3: real migration (same creation id) IS a migration
- Test 4: empty slot never counts toward occupancy
- Test 5+: legacy/v2 parity on healthy inputs and forward plumbing

Run:  python test_ftmoe_protocol019_s1.py
"""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from recovery.PreGANSrc.src.ftmoe_ablation import (
    AblationConfig, ScheduleGraphEncoder, FTMoEAblation,
)
from recovery.PreGANSrc.src.ftmoe_normalization import (
    apply_v2_time_scale, compute_column_stats, normalized_abs_max_report,
    ZERO_CLAMP_EPS,
)

ROOT = Path(__file__).resolve().parent
ART_019 = ROOT / "artifacts/ftmoe_online/protocol_019"
DATA = ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical"


def healthy_stats(maximum: float = 1000.0) -> dict:
    """Training stats where every (host, feature) column is well covered."""
    hosts = []
    for _ in range(16):
        entry = {}
        for feature in ("cpu_demand", "ram_space", "ram_read_or_network_rx",
                        "ram_write_or_network_tx", "disk_space", "disk_read",
                        "disk_write"):
            entry[feature] = {"count_nonzero": 202 * 5, "max": maximum,
                              "p50": maximum * 0.5, "p95": maximum * 0.95,
                              "p99": maximum, "mean": maximum * 0.5,
                              "zero_ratio": 0.0}
        hosts.append(entry)
    return {"hosts": hosts}


class NormalizationV2Tests(unittest.TestCase):
    def test_zero_training_scale_falls_back_to_same_group(self):
        # Test 1: training scale == 0 (legacy clamp 1e-8), online value > 0.
        stats = healthy_stats()
        for feature in stats["hosts"][13]:
            stats["hosts"][13][feature]["max"] = ZERO_CLAMP_EPS  # ~0 coverage
            stats["hosts"][13][feature]["count_nonzero"] = 0
        scale = np.full((16, 7), 1000.0)
        scale[13] = ZERO_CLAMP_EPS  # legacy checkpoint stores the clamp itself
        v2, columns = apply_v2_time_scale(scale, stats)
        # fallback is same-hardware-group (rpi8gb) healthy-peer median == 1000
        self.assertEqual(len(columns), 7)
        for feature in columns:
            self.assertEqual(feature["host"], 13)
            self.assertEqual(feature["group"], "rpi8gb")
            self.assertIn("training_max_zero", feature["reason"])
        np.testing.assert_allclose(v2[13], np.full(7, 1000.0))
        # Normalized online value (host13 ram = 5000) must never explode.
        online = np.zeros((2, 16, 7), np.float32)
        online[:, 13, 1] = 5000.0
        report_v2 = normalized_abs_max_report(online, v2)
        self.assertLess(report_v2["max"], 50.0)
        report_v1 = normalized_abs_max_report(online, scale)
        self.assertGreater(report_v1["max"], 1e8)

    def test_real_014_scale_matches_registered_v2_artifact(self):
        # Integration against the actual protocol-014 checkpoint stats.
        stats_path = ART_019 / "normalization_coverage_train.json"
        v2_path = ART_019 / "normalization_v2_time_scale.json"
        if not (stats_path.exists() and v2_path.exists()):
            self.skipTest("protocol_019 normalization artifacts absent")
        stats = json.loads(stats_path.read_text(encoding="utf8"))
        artifact = json.loads(v2_path.read_text(encoding="utf8"))
        # v1 pathological row on host 13, repaired in v2.
        fallback = {f"{c['host']}:{c['feature']}": c for c in artifact["fallback_columns"]}
        self.assertIn("13:ram_space", fallback)
        self.assertIn("13:disk_space", fallback)
        self.assertIn("13:cpu_demand", fallback)
        np.testing.assert_allclose(
            np.asarray(artifact["time_scale_v2_16x7"]).reshape(16, 7),
            np.asarray(artifact["time_scale_v2_16x7"]).reshape(16, 7),
        )
        # 016 stream must stop tripping the |x|>50 alarm after the repair.
        stream = ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/streams/seed301_steps300/stream.npz"
        if not stream.exists():
            self.skipTest("016 stream absent")
        with np.load(stream) as data:
            features = data["host_features"]
        report_v1 = normalized_abs_max_report(features, np.asarray(artifact["time_scale_v1_16x7"]))
        report_v2 = normalized_abs_max_report(features, np.asarray(artifact["time_scale_v2_16x7"]))
        self.assertTrue(report_v1["alarm"])
        self.assertGreater(report_v1["max"], 1e6)
        self.assertFalse(report_v2["alarm"])
        self.assertLess(report_v2["max"], 50.0)

    def test_compute_column_stats_shape(self):
        series = np.random.RandomState(0).rand(5, 202, 16, 7)
        stats = compute_column_stats(series)
        self.assertEqual(len(stats["hosts"]), 16)
        self.assertEqual(set(stats["hosts"][0].keys()),
                         {"cpu_demand", "ram_space", "ram_read_or_network_rx",
                          "ram_write_or_network_tx", "disk_space", "disk_read",
                          "disk_write"})
        self.assertEqual(stats["hosts"][0]["cpu_demand"]["count_nonzero"], 5 * 202)


class GraphIdentityV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.encoder = ScheduleGraphEncoder(AblationConfig())
        cls.B, cls.C, cls.W, cls.H, cls.F = 1, 16, 12, 16, 7

    def schedule(self):
        schedule = torch.zeros(self.B, self.W, self.C, self.H)
        identity = torch.full((self.B, self.W, self.C), -1, dtype=torch.long)
        # background: slots 0..2 constant on host 0, identity constant
        for container in range(3):
            schedule[:, :, container, 0] = 1.0
            identity[:, :, container] = 100 + container
        return schedule, identity

    def features(self):
        return torch.zeros(self.B, self.C, self.W, self.F)

    def test_identity_replacement_is_not_migration(self):
        # Test 2: slot 3 holds creationID 10 on host 2 at t-1, then a NEW
        # creationID 20 on host 8 at t.  No edge 2 -> 8 under v2.
        schedule, identity = self.schedule()
        schedule[:, 10, 3, 2] = 1.0
        identity[:, 10, 3] = 10
        schedule[:, 11, 3, 8] = 1.0
        identity[:, 11, 3] = 20
        _, migrations_v2 = self.encoder.graph_migration_and_occupancy(
            schedule, {"creation_ids": identity})
        adjacency_v2 = (migrations_v2[0].sum(0) > 0)[2, 8]
        self.assertFalse(bool(adjacency_v2))
        # The legacy path misreads the replacement as a migration.
        _, migrations_legacy = self.encoder.graph_migration_and_occupancy(schedule)
        adjacency_legacy = (migrations_legacy[0].sum(0) > 0)[2, 8]
        self.assertTrue(bool(adjacency_legacy))

    def test_real_migration_is_migration(self):
        # Test 3: same creationID 10 moves host 2 -> host 8 between t-1 and t.
        schedule, identity = self.schedule()
        schedule[:, 10, 3, 2] = 1.0
        identity[:, 10, 3] = 10
        schedule[:, 11, 3, 8] = 1.0
        identity[:, 11, 3] = 10
        _, migrations = self.encoder.graph_migration_and_occupancy(
            schedule, {"creation_ids": identity})
        self.assertTrue(bool((migrations[0].sum(0) > 0)[2, 8]))

    def test_empty_slot_does_not_count_occupancy(self):
        # Test 4: an empty slot (creationID -1) whose scheduler matrix row is
        # non-zero must not increase occupancy under v2.
        schedule, identity = self.schedule()
        schedule[:, 11, 5, 4] = 1.0  # default row written by the scheduler
        identity[:, 11, 5] = -1      # but the slot holds no container
        occupancy_v2, _ = self.encoder.graph_migration_and_occupancy(
            schedule, {"creation_ids": identity})
        self.assertEqual(float(occupancy_v2[0, 11, 4]), 0.0)
        # sanity: a genuinely occupied slot does count
        schedule[:, 11, 6, 4] = 1.0
        identity[:, 11, 6] = 42
        occupancy_v2, _ = self.encoder.graph_migration_and_occupancy(
            schedule, {"creation_ids": identity})
        self.assertEqual(float(occupancy_v2[0, 11, 4]), 1.0)
        # legacy semantics count the default row
        occupancy_legacy, _ = self.encoder.graph_migration_and_occupancy(schedule)
        self.assertEqual(float(occupancy_legacy[0, 11, 4]), 2.0)

    def test_v2_equals_legacy_on_healthy_windows(self):
        # Regression guard: when every slot keeps a valid constant identity,
        # v2 must be bit-identical to legacy.
        schedule, identity = self.schedule()
        for container in range(3, 16):
            schedule[:, :, container, 3] = 1.0
            identity[:, :, container] = 200 + container
        occupancy_legacy, migrations_legacy = \
            self.encoder.graph_migration_and_occupancy(schedule)
        occupancy_v2, migrations_v2 = \
            self.encoder.graph_migration_and_occupancy(schedule, {"creation_ids": identity})
        self.assertTrue(torch.equal(occupancy_legacy, occupancy_v2))
        self.assertTrue(torch.equal(migrations_legacy, migrations_v2))
        features = self.features()
        with torch.no_grad():
            legacy_out = self.encoder(features, schedule)
            v2_out = self.encoder(features, schedule, {"creation_ids": identity})
        self.assertTrue(torch.equal(legacy_out, v2_out))

    def test_model_forward_accepts_graph_context_keyword(self):
        model = FTMoEAblation("v4", AblationConfig(
            experts=4, moe_residual_initial=0.0, eagate_residual_initial=0.5,
            graph_residual_initial=0.0, cmha_residual_initial=0.0)).eval()
        schedule, identity = self.schedule()
        host = torch.randn(self.B, self.H, self.W, self.F)
        containers = torch.randn(self.B, self.C, self.W, self.F)
        with torch.no_grad():
            legacy = model(host, schedule, containers)
            v2 = model(host, schedule, containers,
                       graph_context={"creation_ids": identity})
        for key in ("detection_logits", "class_logits"):
            self.assertTrue(torch.equal(legacy[key], v2[key]))
        # positional-only call shape (used by all offline callers) unchanged
        with torch.no_grad():
            positional = model(host, schedule, containers)
        for key in ("detection_logits", "class_logits"):
            self.assertTrue(torch.equal(positional[key], legacy[key]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
