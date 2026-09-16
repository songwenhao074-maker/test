import json
import unittest
from pathlib import Path

import numpy as np

from simulator.workload.BitbrainWorkloadProtocol025 import (
    SERVICE_IDS, SERVICE_LAWS, EVENT_PROBABILITY, DATA_REVISION_ID,
    _periodic_growth_release, s4_cpu_target_fraction, service_shapes,
    service_active_at,
)
from prepare_ftmoe_protocol025_stream import phase_table, _common_features

ROOT = Path(__file__).resolve().parent
REG = ROOT / 'artifacts/ftmoe_online/protocol_025/registration.json'
REV = ROOT / 'artifacts/ftmoe_online/protocol_025/data_revision_001.json'


class TestProtocol025Generation(unittest.TestCase):
    def setUp(self):
        self.reg = json.loads(REG.read_text(encoding='utf8'))
        self.rev = json.loads(REV.read_text(encoding='utf8'))

    def test_registration_is_development_only_and_sealed(self):
        self.assertEqual(self.reg['development_seed'], 700)
        self.assertEqual(self.reg['model_seed'], 1)
        self.assertEqual(self.reg['confirmation_seeds_forbidden'], [701,702,703])
        self.assertEqual(self.reg['test_seeds_forbidden'], [201,202,203,204,205])
        self.assertTrue(self.reg['registered_before_generation'])
        self.assertEqual(self.reg['generation']['task_response_event_probability'], 0.30)
        self.assertEqual(self.reg['generation']['chunk_intervals'], 200)
        self.assertEqual(self.reg['data_revision']['used_count'], 1)
        self.assertEqual(self.reg['data_revision']['active_revision_id'], DATA_REVISION_ID)
        self.assertFalse(self.reg['data_revision']['pre_revision_stream_eligible_for_model_results'])
        self.assertTrue(self.reg['data_revision']['no_further_data_revision_allowed'])
        self.assertEqual(self.rev['revision_number'], 1)
        self.assertTrue(self.rev['registered_before_any_protocol025_model_result'])
        self.assertTrue(self.rev['uses_only_allowed_data_revision'])

    def test_timeline_sums_to_registered_length(self):
        phases = phase_table(self.reg)
        self.assertEqual(sum(x['length'] for x in phases), 5520)
        self.assertEqual(phases[0]['name'], 'F0')
        self.assertEqual(phases[-1]['name'], 'S3_rec2')
        self.assertEqual(len([x for x in phases if '_rec' in x['name']]), 9)

    def test_six_service_shapes_are_admission_safe(self):
        self.assertEqual(tuple(SERVICE_IDS), ('S1','S2','S3','S4','S5','S6'))
        self.assertAlmostEqual(EVENT_PROBABILITY, 0.30)
        expected_min = {'S1':24,'S2':46,'S3':26,'S4':52,'S5':36,'S6':26}
        for key in SERVICE_IDS:
            spec = SERVICE_LAWS[key]
            self.assertEqual(spec['minimum_trace'], expected_min[key])
            self.assertEqual(float(spec['cpu_shape'][0]), 0.0)
            self.assertEqual(float(spec['ram_shape'][0]), 0.0)
            self.assertEqual(float(spec['disk_shape'][0]), 0.0)
            for name, value in spec['registered_parameters'].items():
                if name in self.reg['service_laws'][key] and isinstance(value,(int,float)):
                    self.assertEqual(float(value), float(self.reg['service_laws'][key][name]))

    def test_s4_registered_floor_and_half_life_are_effective(self):
        # First decay sample starts at the registered peak; each half-life moves
        # half of the remaining excess toward the registered 0.55 floor.
        self.assertAlmostEqual(s4_cpu_target_fraction(16), 1.16, places=12)
        self.assertAlmostEqual(s4_cpu_target_fraction(28), 0.55 + (1.16-0.55)*0.5, places=12)
        self.assertAlmostEqual(s4_cpu_target_fraction(40), 0.55 + (1.16-0.55)*0.25, places=12)
        self.assertGreater(s4_cpu_target_fraction(51), 0.55)
        self.assertLess(s4_cpu_target_fraction(40), s4_cpu_target_fraction(28))
        self.assertEqual(SERVICE_LAWS['S4']['semantic_revision'], DATA_REVISION_ID)

    def test_s5_repeats_and_release_fraction_changes_curve(self):
        shape = np.asarray(_periodic_growth_release(72,36,22,6,0.42), dtype=np.float64)
        self.assertEqual(shape.shape, (72,))
        self.assertEqual(float(shape[0]), 0.0)
        # period = 22 growth + 8 hold + 6 release; end of each cycle retains
        # 58% of the induced peak excess because release_fraction=0.42.
        self.assertAlmostEqual(float(shape[35]), 0.58, places=12)
        self.assertAlmostEqual(float(shape[36]), 0.58, places=12)
        self.assertGreater(float(shape[50]), float(shape[36]))
        self.assertAlmostEqual(float(shape[71]), 0.58, places=12)
        no_release = np.asarray(_periodic_growth_release(36,36,22,6,0.0), dtype=np.float64)
        self.assertAlmostEqual(float(no_release[35]), 1.0, places=12)
        self.assertNotAlmostEqual(float(shape[35]), float(no_release[35]), places=12)
        cpu, ram, disk = service_shapes('S5', 108)
        self.assertEqual(len(cpu), 108); self.assertEqual(len(ram),108); self.assertEqual(len(disk),108)
        self.assertTrue(service_active_at('S5', 72))
        self.assertEqual(SERVICE_LAWS['S5']['semantic_revision'], DATA_REVISION_ID)

    def test_common_feature_order_and_causality(self):
        host = np.zeros((6,16,7), np.float32)
        caps = np.ones((6,16,3), np.float64)
        host[:, :, 0] = np.arange(6)[:,None]
        host[:, :, 1] = 2*np.arange(6)[:,None]
        host[:, :, 4] = 3*np.arange(6)[:,None]
        host[:, :, 2] = 9999.0; host[:, :, 3] = 9999.0; host[:, :, 5] = 9999.0; host[:, :, 6] = 9999.0
        f = _common_features(host, caps)
        self.assertEqual(f.shape, (6,16,9))
        self.assertTrue(np.allclose(f[5,0,:3], [5.0,10.0,15.0]))
        self.assertTrue(np.allclose(f[5,0,3:6], [1.0,2.0,3.0]))
        self.assertTrue(np.allclose(f[5,0,6:9], [1.0,2.0,3.0]))
        self.assertEqual(self.reg['causality']['common_observable_features'], [
            'normalized_cpu_pressure','normalized_ram_pressure','normalized_disk_pressure',
            'cpu_delta_1','ram_delta_1','disk_delta_1','cpu_slope_4','ram_slope_4','disk_slope_4'])
        self.assertTrue(self.reg['causality']['generator_service_id_not_model_input'])
        self.assertTrue(self.reg['causality']['future_switch_times_not_model_input'])


if __name__ == '__main__':
    unittest.main()
