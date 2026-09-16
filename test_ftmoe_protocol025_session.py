import unittest

from ftmoe_protocol025_session import FIXED_COMPARATORS
from ftmoe_protocol024_v2c import V2C_DEFAULT


class TestProtocol025SessionRegistration(unittest.TestCase):
    def test_registered_fixed_comparator_shapes(self):
        self.assertEqual(FIXED_COMPARATORS, {
            'C_fixed4': (4, None),
            'C_fixed5': (5, None),
            'C_fixed8_dense': (8, None),
            'C_fixed8_top5': (8, 5),
        })

    def test_dynamic_policy_inherits_frozen_v2c_budget(self):
        self.assertEqual(V2C_DEFAULT['shadow_budget_version'], 'buffer128_4x4')
        self.assertEqual(V2C_DEFAULT['proposal_start_matured'], 600)
        self.assertEqual(V2C_DEFAULT['proposal_every_matured'], 256)
        self.assertEqual(V2C_DEFAULT['candidate_train_intervals'], 128)
        self.assertEqual(V2C_DEFAULT['reuse_every_matured'], 32)
        self.assertEqual(V2C_DEFAULT['reuse_validation_intervals'], 16)
        self.assertEqual(V2C_DEFAULT['crossfade_prediction_intervals'], 8)
        self.assertTrue(V2C_DEFAULT['replacement_consistent_birth'])


if __name__ == '__main__':
    unittest.main()
