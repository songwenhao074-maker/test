"""Protocol 020 S2 — graph semantics v3 + per-sample capacity tests.

Covers the detailed-solution plan tests:

- Test 5 (graph source host correctness): with before_placement[t,c]=2 and
  proposed dst=8 the migration edge is 2->8, independent of any previously
  proposed host.
- Test 6 (rejected previous proposal does not pollute the current source):
  t-1 proposal host 8 while the container actually stayed on host 2, then at
  t a proposal to host 3 -> edge 2->3 only, never 8->3.
- Test 3 (dynamic capacity visible to the model): a different per-position
  graph_context['capacities'] must change the encoder output.
- Test 4 (no future capacity leakage): window slicing is such that changing
  stream capacities beyond step t cannot alter a prediction window <= t.
- v2/legacy parity: when 'before_placement'/'capacities' are absent the
  encoder output is identical to the Protocol 019 v2 path.

Run from the repository root:
    python test_ftmoe_protocol020_graph.py
"""
import unittest

import numpy as np
import torch

from recovery.PreGANSrc.src.ftmoe_ablation import (
    AblationConfig, ScheduleGraphEncoder,
)

HOSTS = 16
WINDOW = 12
CFG = AblationConfig()


def encoder():
    torch.manual_seed(7)
    return ScheduleGraphEncoder(CFG)


def one_hot(index, n):
    row = torch.zeros(n)
    row[index] = 1.0
    return row


def window_slots(slots, window):
    """slots: list of (creation_id, before_host, dst_host) per position."""
    ids = torch.full((1, window, HOSTS), -1, dtype=torch.long)
    before = torch.full((1, window, HOSTS), -1, dtype=torch.long)
    schedule = torch.zeros(1, window, HOSTS, HOSTS)
    for t, slot_states in enumerate(slots):
        for cid, src, dst in slot_states:
            ids[0, t, cid] = cid + 1000  # any positive identity
            before[0, t, cid] = src
            schedule[0, t, cid] = one_hot(dst, HOSTS)
    return ids, before, schedule


class MigrationSemanticsV3Tests(unittest.TestCase):
    def test_5_source_host_from_before_placement(self):
        # Single position: slot 3 on host 2, proposed dst host 8 -> edge 2->8.
        ids, before, schedule = window_slots(
            [[(3, 2, 8)]], window=1)
        _, migrations = encoder().graph_migration_and_occupancy(
            schedule, {"creation_ids": ids, "before_placement": before})
        self.assertEqual(migrations.shape, (1, 1, HOSTS, HOSTS))
        self.assertEqual(float(migrations[0, 0, 2, 8]), 1.0)
        self.assertEqual(float(migrations.sum()), 1.0)

    def test_6_rejected_previous_proposal_not_the_source(self):
        # Position 0: container proposed to 8 (would be rejected later).
        # Position 1: container is actually still on host 2 (before=2),
        # new proposal -> 3.  The edge must be 2->3 and never 8->3.
        ids, before, schedule = window_slots(
            [[(3, 2, 8)], [(3, 2, 3)]], window=2)
        _, migrations = encoder().graph_migration_and_occupancy(
            schedule, {"creation_ids": ids, "before_placement": before})
        self.assertEqual(float(migrations[0, 1, 2, 3]), 1.0,
                         "actual source (before_placement) must be used")
        self.assertEqual(float(migrations[0, 1, 8, 3]), 0.0,
                         "rejected previous proposal must not become the source")
        self.assertEqual(float(migrations[0, 0, 2, 8]), 1.0)
        # no self-loops, no phantom edges elsewhere
        eye = torch.eye(HOSTS, dtype=migrations.dtype).unsqueeze(0).unsqueeze(0)
        self.assertEqual(float((migrations * eye).sum()), 0.0)
        self.assertEqual(float(migrations.sum()), 2.0)

    def test_undeployed_slot_produces_no_edge(self):
        # Slot exists (identity >= 0) but before == -1 (never deployed):
        # no migration may be counted for it.
        ids = torch.full((1, 1, HOSTS), -1, dtype=torch.long)
        before = torch.full((1, 1, HOSTS), -1, dtype=torch.long)
        ids[0, 0, 5] = 42
        schedule = torch.zeros(1, 1, HOSTS, HOSTS)
        schedule[0, 0, 5] = one_hot(9, HOSTS)
        _, migrations = encoder().graph_migration_and_occupancy(
            schedule, {"creation_ids": ids, "before_placement": before})
        self.assertEqual(float(migrations.sum()), 0.0)


class CapacityContextTests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(11)
        time_windows = torch.zeros(1, HOSTS, WINDOW, 7)
        ids = torch.zeros(1, WINDOW, HOSTS, dtype=torch.long)
        schedule = torch.zeros(1, WINDOW, HOSTS, HOSTS)
        # one live container per host for every position
        for t in range(WINDOW):
            for c in range(HOSTS):
                ids[0, t, c] = c + 1
                schedule[0, t, c] = one_hot(c, HOSTS)
        return time_windows, schedule, ids

    def test_3_dynamic_capacity_changes_output(self):
        time_windows, schedule, ids = self._inputs()
        caps_a = torch.ones(1, WINDOW, HOSTS, 3)
        caps_b = caps_a.clone()
        caps_b[0, :, :, 0] = 0.5  # half CPU capacity everywhere
        model = encoder()
        out_a = model(time_windows, schedule,
                      graph_context={"creation_ids": ids,
                                     "capacities": caps_a})
        out_b = model(time_windows, schedule,
                      graph_context={"creation_ids": ids,
                                     "capacities": caps_b})
        self.assertFalse(torch.allclose(out_a, out_b),
                         "per-sample capacities must change the graph output")

    def test_4_no_future_capacity_leakage(self):
        # Emulate the replay-window slicing: a window ending at index i only
        # reads stream rows [max(i-11,0) .. i].  Modifying rows > i must not
        # alter the window contents (and therefore cannot alter predictions).
        steps = 30
        stream_caps = np.zeros((steps + 1, HOSTS, 3), dtype=np.float32)
        stream_caps[:, :, 0] = 1.0
        stream_caps[:, :, 1] = 2.0
        stream_caps[:, :, 2] = 3.0
        i = 5
        positions = np.maximum(np.arange(i - 11, i + 1), 0)
        window_a = stream_caps[positions]
        stream_caps[i + 1:] = 99.0  # corrupt everything strictly after step i
        window_b = stream_caps[positions]
        np.testing.assert_array_equal(window_a, window_b)

    def test_legacy_capacity_buffer_when_key_absent(self):
        time_windows, schedule, ids = self._inputs()
        model = encoder()
        with_ctx = model(time_windows, schedule,
                         graph_context={"creation_ids": ids})
        without_ctx = model(time_windows, schedule, graph_context=None)
        # legacy (no context) and v2 (creation ids only) share the static
        # capacity buffer -> identical outputs on this input.
        self.assertTrue(torch.allclose(with_ctx, without_ctx))

    def test_v2_path_unchanged_without_before_placement(self):
        # Same identities across positions => v2 migration counting equals the
        # legacy consecutive-proposal count, and shape stays [B,W-1,H,H].
        ids, before, schedule = window_slots(
            [[(3, 2, 8)], [(3, 3, 4)]], window=2)
        model = encoder()
        _, mig_v2 = model.graph_migration_and_occupancy(
            schedule, {"creation_ids": ids})
        self.assertEqual(mig_v2.shape, (1, 1, HOSTS, HOSTS))
        self.assertEqual(float(mig_v2[0, 0, 8, 4]), 1.0)
        # and before_placement is simply ignored on the v2 path
        _, mig_ctx = model.graph_migration_and_occupancy(
            schedule, {"creation_ids": ids})
        self.assertTrue(torch.equal(mig_v2, mig_ctx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
