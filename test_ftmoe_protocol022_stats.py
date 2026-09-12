"""Protocol 022 — Stats series-append equivalence test (problem P22-17).

``Stats.saveStats`` used ``np.append`` per interval, which reallocates the whole
series each step: quadratic time and an unbounded per-step RSS rise that aborts
the registered 3.0 GiB guard on a 3400-interval stream.  The replacement grows
capacity geometrically while keeping the public attribute a view trimmed to the
used rows.

This test pins the contract that matters: for the same sequence of rows, the
values, order, dtype and the consumer-visible length must be exactly what the
original implementation produced.

Run from the repository root:
    python test_ftmoe_protocol022_stats.py
"""
import unittest

import numpy as np

from stats.Stats import Stats


def reference_append(initial, rows):
    """The original implementation: np.append on every interval."""
    series = initial
    for row in rows:
        series = np.append(series, row, axis=0)
    return series


class SeriesAppendEquivalenceTests(unittest.TestCase):
    def _stats(self, rows=7):
        stats = Stats.__new__(Stats)          # no simulator needed
        stats.time_series = np.zeros((1, rows), dtype=float)
        stats.schedule_series = np.zeros((1, 16, 16), dtype=float)
        return stats

    def _run(self, count=25, rows=7, seed=3):
        rng = np.random.default_rng(seed)
        points = [rng.normal(size=(1, rows)) for _ in range(count)]
        stats = self._stats(rows)
        for point in points:
            stats.appendSeries("time_series", point)
        expected = reference_append(np.zeros((1, rows)), points)
        return stats, expected

    def test_values_dtype_and_order_match_np_append(self):
        stats, expected = self._run()
        self.assertEqual(stats.time_series.dtype, expected.dtype)
        self.assertEqual(stats.time_series.shape, expected.shape)
        np.testing.assert_array_equal(stats.time_series, expected)

    def test_consumer_visible_length_is_exact(self):
        """len() must count appended rows, not reserved capacity."""
        stats, expected = self._run(count=25)
        self.assertEqual(len(stats.time_series), len(expected))
        self.assertEqual(stats.seriesLength("time_series"), len(expected))

    def test_tail_slicing_still_gives_the_last_k_rows(self):
        stats, expected = self._run(count=25)
        np.testing.assert_array_equal(stats.time_series[-12:], expected[-12:])
        np.testing.assert_array_equal(stats.time_series[:1], expected[:1])

    def test_growth_crosses_several_capacity_doublings(self):
        stats, expected = self._run(count=200, seed=11)
        np.testing.assert_array_equal(stats.time_series, expected)
        self.assertGreaterEqual(len(stats.time_series), 200)

    def test_three_dimensional_series_behave_the_same(self):
        rng = np.random.default_rng(5)
        points = [rng.normal(size=(1, 16, 16)) for _ in range(40)]
        stats = self._stats()
        for point in points:
            stats.appendSeries("schedule_series", point)
        expected = reference_append(np.zeros((1, 16, 16)), points)
        np.testing.assert_array_equal(stats.schedule_series, expected)
        self.assertEqual(len(stats.schedule_series), len(expected))

    def test_reserved_capacity_never_leaks_values(self):
        """Unused capacity must never be readable as data."""
        stats = self._stats()
        stats.appendSeries("time_series", np.ones((1, 7)))
        buffer = stats._buffer_time_series
        self.assertGreater(buffer.shape[0], len(stats.time_series))
        self.assertTrue(np.all(buffer[len(stats.time_series):] == 0.0))
        self.assertEqual(len(stats.time_series), 2)

    def test_single_append_after_many_matches_reference(self):
        stats, expected = self._run(count=17)
        extra = np.full((1, 7), 9.5)
        stats.appendSeries("time_series", extra)
        expected = np.append(expected, extra, axis=0)
        np.testing.assert_array_equal(stats.time_series, expected)


class HistoryLimitTests(unittest.TestCase):
    """Problem P22-18: bounded bookkeeping history must not change any value."""

    def _stats(self, rows=7, limit=None):
        stats = Stats.__new__(Stats)
        stats.time_series = np.zeros((1, rows), dtype=float)
        stats.schedule_series = np.zeros((1, 16, 16), dtype=float)
        stats.history_limit = limit
        for name in ("hostinfo", "workloadinfo", "activecontainerinfo",
                     "allcontainerinfo", "metrics", "schedulerinfo"):
            setattr(stats, name, [])
        return stats

    def test_default_keeps_full_history(self):
        stats = self._stats(limit=None)
        for i in range(200):
            stats.recordHistory("hostinfo", {"interval": i})
        self.assertEqual(len(stats.hostinfo), 200)

    def test_bound_keeps_only_the_most_recent_entries(self):
        stats = self._stats(limit=8)
        for i in range(200):
            stats.recordHistory("hostinfo", {"interval": i})
        self.assertEqual(len(stats.hostinfo), 8)
        self.assertEqual(stats.hostinfo[-1]["interval"], 199,
                         "consumers read the LAST interval")
        self.assertEqual([e["interval"] for e in stats.hostinfo],
                         list(range(192, 200)))

    def test_all_six_lists_are_bounded(self):
        stats = self._stats(limit=4)
        for i in range(20):
            for name in ("hostinfo", "workloadinfo", "activecontainerinfo",
                         "allcontainerinfo", "metrics", "schedulerinfo"):
                stats.recordHistory(name, i)
        for name in ("hostinfo", "workloadinfo", "activecontainerinfo",
                     "allcontainerinfo", "metrics", "schedulerinfo"):
            self.assertEqual(len(getattr(stats, name)), 4, name)

    def test_bounding_does_not_touch_the_series(self):
        stats = self._stats(limit=4)
        for _ in range(50):
            stats.appendSeries("time_series", np.ones((1, 7)))
            stats.recordHistory("hostinfo", {"x": 1})
        self.assertEqual(len(stats.time_series), 51,
                         "the series must still carry every interval")


class SeriesTailTests(unittest.TestCase):
    """Problem P22-20: a bounded series tail must keep the rows consumers read."""

    def _stats(self, rows=7, tail=None):
        stats = Stats.__new__(Stats)
        stats.time_series = np.zeros((1, rows), dtype=float)
        stats.schedule_series = np.zeros((1, 16, 16), dtype=float)
        stats.series_tail = tail
        return stats

    def test_unbounded_by_default(self):
        stats = self._stats(tail=None)
        for i in range(120):
            stats.appendSeries("time_series", np.full((1, 7), float(i)))
        self.assertEqual(len(stats.time_series), 121)

    def test_bound_keeps_exactly_the_tail(self):
        stats = self._stats(tail=10)
        for i in range(120):
            stats.appendSeries("time_series", np.full((1, 7), float(i)))
        self.assertEqual(len(stats.time_series), 10)
        self.assertEqual(float(stats.time_series[-1, 0]), 119.0,
                         "the last appended row must be the newest")

    def test_tail_is_a_suffix_of_the_unbounded_series(self):
        """The retained rows must equal what the full series would show."""
        unbounded = self._stats(tail=None)
        bounded = self._stats(tail=16)
        for i in range(80):
            row = np.full((1, 7), float(i))
            unbounded.appendSeries("time_series", row)
            bounded.appendSeries("time_series", row)
        np.testing.assert_array_equal(bounded.time_series,
                                      unbounded.time_series[-16:])

    def test_recovery_window_is_still_available(self):
        """The online path reads the last 12 rows; the tail must cover it."""
        stats = self._stats(tail=96)
        for i in range(500):
            stats.appendSeries("time_series", np.full((1, 7), float(i)))
        self.assertGreaterEqual(len(stats.time_series), 12)
        # rows are 0..499 (the initial zero row plus appends of 0..499 minus the
        # rows the tail dropped), so the last 12 rows are 488..499
        self.assertEqual(float(stats.time_series[-12:, 0].min()), 488.0)

    def test_schedule_series_is_bounded_too(self):
        stats = self._stats(tail=8)
        for _ in range(40):
            stats.appendSeries("schedule_series", np.ones((1, 16, 16)))
        self.assertEqual(len(stats.schedule_series), 8)
        self.assertEqual(stats.schedule_series.shape[1:], (16, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
