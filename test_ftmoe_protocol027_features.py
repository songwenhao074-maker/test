"""Regression: replay padding must match the registered causal feature array."""
import unittest
import numpy as np
import torch
from run_ftmoe_protocol020 import ReplayV3
from run_ftmoe_protocol023_s4 import window_batch
from prepare_ftmoe_protocol025_stream import _common_features
from ftmoe_protocol025_model import causal_common_features


class FeatureBoundaryTests(unittest.TestCase):
    def setUp(self):
        n = 22
        rng = np.random.RandomState(27021)
        self.host = rng.uniform(1, 10, (n, 16, 7)).astype(np.float32)
        self.caps = rng.uniform(7, 14, (n, 16, 3)).astype(np.float32)
        # Real constant histories must not be confused with repeated padding.
        self.host[8:14] = self.host[8]
        self.caps[8:14] = self.caps[8]
        self.ts = rng.uniform(2, 4, (16, 7)).astype(np.float32)
        self.gs = rng.uniform(1, 3, 7).astype(np.float32)
        self.arrays = dict(host_features=self.host, raw_labels=np.zeros((n, 16), np.int64),
            demands=np.zeros_like(self.host), schedules=np.zeros((n, 16, 16), np.float32),
            creation_ids=np.zeros((n, 16), np.int64), before_placement=np.zeros((n, 16), np.int64),
            capacities_per_interval=self.caps)

    def evaluate(self, indices):
        replay = ReplayV3(self.arrays, self.ts, self.gs, 21)
        x, _, _, context = window_batch(replay, indices)
        return causal_common_features(x, context, self.ts, self.gs).numpy(), x, context

    def test_all_times_and_out_of_order_replay_match_generator(self):
        indices = [3, 0, 20, 1, 7, 2, 4, 11, 3, 15]
        got, x, context = self.evaluate(indices)
        expected = _common_features(self.host, self.caps)[indices]
        np.testing.assert_allclose(got, expected, atol=5e-5, rtol=0)
        np.testing.assert_array_equal(got[[0, 1, 3, 5, 8], :, 6:], 0)
        # This exact input exercises the old bug, not a vacuous zero fixture.
        legacy = dict(context); legacy.pop("observed_history_length")
        old = causal_common_features(x, legacy, self.ts, self.gs).numpy()
        self.assertGreater(float(np.max(np.abs(old[0] - expected[0]))), 5e-5)

    def test_every_scored_time_and_future_independence(self):
        indices = list(range(21))
        got, _, _ = self.evaluate(indices)
        np.testing.assert_allclose(got, _common_features(self.host, self.caps)[:21], atol=5e-5, rtol=0)
        before, _, _ = self.evaluate([7])
        self.host[8:] *= 50
        self.caps[8:] *= 3
        after, _, _ = self.evaluate([7])
        np.testing.assert_array_equal(before, after)

    def test_missing_history_is_not_inferred_from_constant_values(self):
        got, _, _ = self.evaluate([14])
        np.testing.assert_allclose(got, _common_features(self.host, self.caps)[[14]], atol=5e-5, rtol=0)

if __name__ == "__main__":
    unittest.main()
