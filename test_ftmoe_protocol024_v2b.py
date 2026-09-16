import unittest
import numpy as np
from ftmoe_protocol024_v2b import V2B_DEFAULT, _unit

class TestProtocol024V2BRegistration(unittest.TestCase):
    def test_memory_policy_is_bounded_and_prediction_counted(self):
        self.assertEqual(V2B_DEFAULT['generalist_ids'], ['0','1','2','3'])
        self.assertEqual(V2B_DEFAULT['max_live_specialists'], 1)
        self.assertEqual(V2B_DEFAULT['reuse_every_matured'], 32)
        self.assertEqual(V2B_DEFAULT['reuse_validation_intervals'], 16)
        self.assertEqual(V2B_DEFAULT['crossfade_prediction_intervals'], 8)
        self.assertEqual(V2B_DEFAULT['shadow_budget_version'], 'buffer128_4x4')
    def test_unit_representation_is_scale_invariant(self):
        a=_unit(np.array([3.0,4.0])); b=_unit(np.array([30.0,40.0]))
        self.assertTrue(np.allclose(a,b)); self.assertAlmostEqual(float(np.linalg.norm(a)),1.0)
    def test_memory_threshold_is_causal_self_quantile(self):
        self.assertEqual(V2B_DEFAULT['memory_similarity_quantile'],10.0)
        self.assertTrue(V2B_DEFAULT['freeze_specialist_parameters'])

if __name__=='__main__': unittest.main()
