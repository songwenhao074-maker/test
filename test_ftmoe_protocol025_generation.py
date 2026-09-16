import json
import unittest
from pathlib import Path

import numpy as np

from simulator.workload.BitbrainWorkloadProtocol025 import (
    SERVICE_IDS, SERVICE_LAWS, EVENT_PROBABILITY,
)
from prepare_ftmoe_protocol025_stream import phase_table, _common_features

ROOT = Path(__file__).resolve().parent
REG = ROOT / 'artifacts/ftmoe_online/protocol_025/registration.json'


class TestProtocol025Generation(unittest.TestCase):
    def setUp(self):
        self.reg = json.loads(REG.read_text(encoding='utf8'))

    def test_registration_is_development_only_and_sealed(self):
        self.assertEqual(self.reg['development_seed'], 700)
        self.assertEqual(self.reg['model_seed'], 1)
        self.assertEqual(self.reg['confirmation_seeds_forbidden'], [701,702,703])
        self.assertEqual(self.reg['test_seeds_forbidden'], [201,202,203,204,205])
        self.assertTrue(self.reg['registered_before_generation'])
        self.assertEqual(self.reg['generation']['task_response_event_probability'], 0.30)
        self.assertEqual(self.reg['generation']['chunk_intervals'], 200)

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

    def test_common_feature_order_and_causality(self):
        host = np.zeros((6,16,7), np.float32)
        caps = np.ones((6,16,3), np.float64)
        host[:, :, 0] = np.arange(6)[:,None]
        host[:, :, 1] = 2*np.arange(6)[:,None]
        host[:, :, 4] = 3*np.arange(6)[:,None]
        # read/write columns must not enter physical pressure.
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
