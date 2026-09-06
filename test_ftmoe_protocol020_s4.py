"""Protocol 020 S3/S4 — cohort workload + capacity controller unit tests.

- Protocol020AdaptedBWGD2: arrival pool == registered cohort from
  vm_split.json (no full-pool scan), cohort VMs verified on disk, cohort
  validation errors raised for unknown names.
- RPiCapacity: applies per-phase scales to live host objects (values read
  back through controller.current()), rejects double-scaled hosts, rejects
  invalid scales.

Run from the repository root:
    python test_ftmoe_protocol020_s4.py
"""
import os
import unittest

import numpy as np

from simulator.environment.RPiCapacity import RPiCapacity, PHYSICAL_CPU, PHYSICAL_RAM, PHYSICAL_DISK
from simulator.workload.BitbrainWorkloadProtocol020 import (
    Protocol020AdaptedBWGD2, SPLIT_PATH,
)

import json


class CohortWorkloadTests(unittest.TestCase):
    def test_pool_matches_registered_cohort(self):
        split = json.loads(SPLIT_PATH.read_text(encoding="utf8"))
        for cohort in ("train", "dev", "online"):
            workload = Protocol020AdaptedBWGD2(1, 1.5, 401, cohort=cohort)
            self.assertEqual(workload.possible_indices,
                             list(split["cohorts"][cohort]))
            self.assertEqual(workload.cohort, cohort)

    def test_unknown_cohort_rejected(self):
        with self.assertRaises(ValueError):
            Protocol020AdaptedBWGD2(1, 1.5, 401, cohort="nope")

    def test_workload_state_initialised_like_bwgd2(self):
        workload = Protocol020AdaptedBWGD2(1, 1.5, 401, cohort="train")
        self.assertEqual(workload.creation_id, 0)
        self.assertEqual(workload.createdContainers, [])
        self.assertEqual(workload.meanSLA, 20)
        self.assertIsInstance(workload.disk_law, dict)
        self.assertIn("values", workload.disk_law)
        self.assertEqual(workload.replay_seed, 401)
        # arrival param wiring
        self.assertEqual(workload.mean, 1)
        self.assertEqual(workload.sigma, 1.5)


class CapacityControllerTests(unittest.TestCase):
    def _hosts(self, cpu_scale="1.0", ram_scale="1.0", disk_scale="1.0"):
        saved = {name: os.environ.pop(name, None)
                 for name in ("CPU_CAP_SCALE", "RAM_CAP_SCALE", "DISK_CAP_SCALE")}
        try:
            for name, value in (("CPU_CAP_SCALE", cpu_scale),
                                ("RAM_CAP_SCALE", ram_scale),
                                ("DISK_CAP_SCALE", disk_scale)):
                os.environ[name] = value
            from simulator.environment.RPiEdge import RPiEdge
            from simulator.Simulator import Simulator
            # Simulator needs scheduler/stats; instead test on Host objects
            # built the same way Simulator.addHostInit does.
            from simulator.host.Host import Host
            from simulator.environment.RPiEdge import RPiEdge as RPi  # noqa
            tuples = RPiEdge(16).generateHosts()
            hosts = []
            for i, (ips, ram, disk, bw, latency, power) in enumerate(tuples):
                host = Host.__new__(Host)  # avoid powermodel/env wiring
                host.id = i
                host.ipsCap = ips
                host.ramCap = ram
                host.diskCap = disk
                hosts.append(host)
            return hosts
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_apply_scales(self):
        hosts = self._hosts()
        controller = RPiCapacity(hosts)
        controller.apply(0.75, 0.45, 0.30)
        current = controller.current()
        np.testing.assert_allclose(current[:, 0], PHYSICAL_CPU * 0.75)
        np.testing.assert_allclose(current[:, 1], PHYSICAL_RAM * 0.45)
        np.testing.assert_allclose(current[:, 2], PHYSICAL_DISK * 0.30)
        # live host fields are what simulators read
        self.assertEqual(hosts[0].ipsCap, PHYSICAL_CPU * 0.75)
        self.assertEqual(hosts[0].ramCap.size, 4295.0 * 0.45)
        self.assertEqual(hosts[0].diskCap.size, PHYSICAL_DISK * 0.30)

    def test_reapply_restores_base(self):
        hosts = self._hosts()
        controller = RPiCapacity(hosts)
        controller.apply(0.5, 0.5, 0.5)
        controller.apply(1.0, 1.0, 1.0)
        current = controller.current()
        np.testing.assert_allclose(current[:, 1], PHYSICAL_RAM, rtol=0, atol=1e-9)

    def test_rejects_double_scaled_hosts(self):
        hosts = self._hosts(cpu_scale="0.8")
        with self.assertRaises(ValueError):
            RPiCapacity(hosts)

    def test_rejects_invalid_scales(self):
        hosts = self._hosts()
        controller = RPiCapacity(hosts)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                controller.apply(bad, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
