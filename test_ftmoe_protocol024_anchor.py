import unittest

import numpy as np

from ftmoe_protocol024_anchor import raw_next_anchor_labels


class TestProtocol024Anchor(unittest.TestCase):
    def test_anchor_uses_same_host_raw_i_plus_one(self):
        raw = np.zeros((6, 16), dtype=np.int64)
        raw[1, 0] = 1
        raw[3, 0] = 2
        raw[3, 7] = 3
        target = raw_next_anchor_labels(raw, [0, 2])
        self.assertEqual(target.shape, (2, 16))
        self.assertEqual(int(target[0, 0]), 1)
        self.assertEqual(int(target[1, 0]), 2)
        self.assertEqual(int(target[1, 7]), 3)

    def test_last_scored_row_may_use_physical_guard_row(self):
        raw = np.zeros((5, 16), dtype=np.int64)
        raw[4, 4] = 3
        target = raw_next_anchor_labels(raw, [3])
        self.assertEqual(int(target[0, 4]), 3)

    def test_missing_future_row_is_rejected(self):
        raw = np.zeros((4, 16), dtype=np.int64)
        with self.assertRaises(ValueError):
            raw_next_anchor_labels(raw, [3])

    def test_illegal_resource_class_is_rejected(self):
        raw = np.zeros((4, 16), dtype=np.int64)
        raw[2, 1] = 4
        with self.assertRaises(ValueError):
            raw_next_anchor_labels(raw, [1])


if __name__ == "__main__":
    unittest.main()
