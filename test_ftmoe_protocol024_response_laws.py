import unittest
import numpy as np

from simulator.workload.BitbrainWorkloadProtocol024 import (
    RESPONSE_LAWS, LAW_IDS, Protocol024ResponseLawBWGD2,
    _blend_to_target)


class TestProtocol024ResponseLaws(unittest.TestCase):
    def test_every_law_is_admission_safe_at_age_zero(self):
        for law in LAW_IDS:
            spec = RESPONSE_LAWS[law]
            self.assertEqual(spec["cpu_shape"][0], 0.0)
            self.assertEqual(spec["ram_shape"][0], 0.0)
            self.assertEqual(spec["disk_shape"][0], 0.0)
            base = np.array([123.0, 456.0, 789.0])
            got = _blend_to_target(base, [0.0, 1.0, 0.5], 1000.0, 1200.0)
            self.assertEqual(float(got[0]), float(base[0]))

    def test_registered_laws_have_distinct_response_dynamics(self):
        r1, r2, r3 = (RESPONSE_LAWS[x] for x in LAW_IDS)
        self.assertLess(len(r1["cpu_shape"]), len(r2["cpu_shape"]))
        self.assertGreater(max(r1["cpu_shape"]), 0.9)
        self.assertEqual(max(r1["disk_shape"]), 0.0)
        self.assertEqual(max(r2["disk_shape"]), 0.0)
        self.assertGreater(max(r3["disk_shape"]), 0.9)
        self.assertGreater(r2["ram_shape"][8], r2["ram_shape"][2])
        self.assertGreater(r3["disk_shape"][5], r3["disk_shape"][1])
        self.assertGreater(r3["disk_shape"][5], r3["disk_shape"][10])

    def test_event_draw_is_deterministic_and_mode_specific(self):
        obj = Protocol024ResponseLawBWGD2.__new__(Protocol024ResponseLawBWGD2)
        obj.replay_seed = 700
        obj.event_probability = 1.0
        draws = {}
        for law in LAW_IDS:
            obj._active_law = law
            first = obj._draw_event(12345)
            second = obj._draw_event(12345)
            self.assertEqual(first, second)
            draws[law] = first["draw"]
        self.assertEqual(len(set(draws.values())), 3)

    def test_event_probability_zero_disables_mechanism(self):
        obj = Protocol024ResponseLawBWGD2.__new__(Protocol024ResponseLawBWGD2)
        obj.replay_seed = 700
        obj._active_law = "R1"
        obj.event_probability = 0.0
        self.assertIsNone(obj._draw_event(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
