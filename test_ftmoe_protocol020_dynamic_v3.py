"""Protocol 020 S8 — dynamic expert v3 trigger tests (plan Test 12 + §23 gate).

- Test 12: even when every routing token has an eligible expert (novelty
  comes from entropy/margin, not unmatched), high novelty EMA + sustained
  matured-loss ratio >= 1.25 still triggers the birth gate.
- Trigger statistics accumulate independently of eligibility.
- Baselines must be registered before observation; gate arms are AND-ed.

Run from the repository root:
    python test_ftmoe_protocol020_dynamic_v3.py
"""
import unittest

from recovery.PreGANSrc.src.ftmoe_dynamic_expert_v3 import (
    V3TriggerState, V3_CONSECUTIVE_WINDOWS, V3_LOSS_RATIO_MIN,
    V3_CANDIDATE_SAMPLES,
)


def warm(state, windows=40, samples=1, loss=1.0, novelty=0.05):
    for _ in range(windows):
        state.observe_window(novelty, samples, matured_loss=loss)


class V3TriggerTests(unittest.TestCase):
    def test_12_birth_trigger_independent_of_eligibility(self):
        state = V3TriggerState()
        state.register_baselines(novelty_threshold=0.30, loss_baseline=1.0)
        warm(state)  # EMA reaches the stationary level
        for _ in range(12):
            record = state.observe_window(novelty_value=0.9, samples=40,
                                          matured_loss=1.5)
        self.assertTrue(record["novelty_above"])
        self.assertGreaterEqual(record["loss_ratio"], V3_LOSS_RATIO_MIN)
        ready, checks = state.ready(expert_count=4)
        self.assertTrue(ready, checks)

    def test_sustained_windows_required(self):
        state = V3TriggerState()
        state.register_baselines(novelty_threshold=0.30, loss_baseline=1.0)
        warm(state)
        # repeated novelty bursts separated by long decay runs (EMA resets
        # below the threshold): the counter never accumulates 3 consecutive
        # above-threshold windows
        for _ in range(2):
            for _ in range(5):
                state.observe_window(0.9, 40, matured_loss=1.5)
            for _ in range(12):
                state.observe_window(0.0, 40, matured_loss=1.5)
        for _ in range(2):
            state.observe_window(0.9, 40, matured_loss=1.5)
        self.assertLess(state.novelty_windows_above,
                        V3_CONSECUTIVE_WINDOWS)
        ready, checks = state.ready(expert_count=4)
        self.assertFalse(ready, "must need %d consecutive windows"
                         % V3_CONSECUTIVE_WINDOWS)
        self.assertFalse(checks["novelty_ok"])

    def test_low_loss_ratio_blocks_birth(self):
        state = V3TriggerState()
        state.register_baselines(novelty_threshold=0.30, loss_baseline=10.0)
        warm(state, loss=10.0)
        for _ in range(12):
            state.observe_window(0.9, 40, matured_loss=10.4)
        ready, checks = state.ready(expert_count=4)
        self.assertFalse(ready)
        self.assertTrue(checks["novelty_ok"])
        self.assertFalse(checks["loss_ok"])

    def test_candidate_samples_floor(self):
        state = V3TriggerState()
        state.register_baselines(novelty_threshold=0.30, loss_baseline=1.0)
        warm(state, samples=1)
        for _ in range(4):
            state.observe_window(0.9, 10, matured_loss=1.5)
        self.assertLess(state.candidate_samples, V3_CANDIDATE_SAMPLES)
        ready, checks = state.ready(expert_count=4)
        self.assertFalse(checks["samples_ok"])

    def test_expert_capacity_gate(self):
        state = V3TriggerState()
        state.register_baselines(novelty_threshold=0.30, loss_baseline=1.0)
        warm(state)
        for _ in range(12):
            state.observe_window(0.9, 40, matured_loss=1.5)
        ready, checks = state.ready(expert_count=8)
        self.assertFalse(ready)
        self.assertFalse(checks["expert_capacity_ok"])

    def test_baselines_must_be_registered(self):
        state = V3TriggerState()
        with self.assertRaises(ValueError):
            state.observe_window(0.9, 40)
        with self.assertRaises(ValueError):
            state.ready(4)

    def test_save_restore_roundtrip(self):
        state = V3TriggerState()
        state.register_baselines(0.30, 1.0)
        warm(state)
        for _ in range(12):
            state.observe_window(0.9, 40, matured_loss=1.5)
        clone = V3TriggerState()
        clone.restore(state.state())
        self.assertEqual(clone.novelty_ema, state.novelty_ema)
        self.assertEqual(clone.candidate_samples, state.candidate_samples)
        ready_a, _ = state.ready(4)
        ready_b, _ = clone.ready(4)
        self.assertEqual(ready_a, ready_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
