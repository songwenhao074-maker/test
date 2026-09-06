"""Protocol 020 S1 — RAM_CAP_SCALE capacity-control tests.

Covers the detailed-solution plan tests:

- Test 1 (legacy): without RAM_CAP_SCALE the generated host capacities are
  byte-identical to Protocol 019 (RPi4B RAM 4295 / RPi4B8G RAM 8192).
- Test 2 (heterogeneity preserved): RAM_CAP_SCALE=0.5 scales both groups by
  0.5 (2147.5 / 4096.0) and keeps the 8192:4295 ratio; CPU/Disk untouched.
- Env parsing: non-finite / non-positive scales are rejected loudly instead
  of silently corrupting capacities.

Run from the repository root:
    python test_ftmoe_protocol020_simulator.py
"""
import os
import unittest

import numpy as np

FIRST8 = np.full(8, 4295.0, dtype=np.float64)
LAST8 = np.full(8, 8192.0, dtype=np.float64)


def host_capacities():
    """Return [16,3] CPU/RAM/Disk capacities for the RPiEdge(16) environment."""
    from simulator.environment.RPiEdge import RPiEdge
    hosts = RPiEdge(16).generateHosts()
    caps = np.array([[h[0], h[1].size, h[2].size] for h in hosts], dtype=np.float64)
    assert caps.shape == (16, 3)
    return caps


class RAMCapScaleTests(unittest.TestCase):
    @staticmethod
    def _clean_env():
        for name in ("RAM_CAP_SCALE", "CPU_CAP_SCALE", "DISK_CAP_SCALE"):
            os.environ.pop(name, None)

    def test_legacy_byte_identical_without_variable(self):
        # Test 1 — legacy behavior when RAM_CAP_SCALE is unset.
        self._clean_env()
        caps = host_capacities()
        np.testing.assert_allclose(caps[:8, 1], FIRST8, rtol=0, atol=0)
        np.testing.assert_allclose(caps[8:, 1], LAST8, rtol=0, atol=0)
        # CPU / Disk sizes untouched by default as well (protocol-019 values).
        np.testing.assert_allclose(caps[:, 0], 4029.0, rtol=0, atol=0)
        np.testing.assert_allclose(caps[:, 2], 32212.0, rtol=0, atol=0)

    def test_legacy_explicit_1_0(self):
        # Test 1 — RAM_CAP_SCALE=1.0 must equal the no-variable case.
        self._clean_env()
        baseline = host_capacities()
        os.environ["RAM_CAP_SCALE"] = "1.0"
        try:
            scaled = host_capacities()
        finally:
            self._clean_env()
        np.testing.assert_array_equal(scaled, baseline)

    def test_half_scale_heterogeneity_preserved(self):
        # Test 2 — RAM_CAP_SCALE=0.5: 4295*0.5 and 8192*0.5, ratio preserved.
        self._clean_env()
        os.environ["RAM_CAP_SCALE"] = "0.5"
        try:
            caps = host_capacities()
        finally:
            self._clean_env()
        np.testing.assert_allclose(caps[:8, 1], FIRST8 * 0.5, rtol=0, atol=1e-12)
        np.testing.assert_allclose(caps[8:, 1], LAST8 * 0.5, rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(caps[8, 1] / caps[0, 1]), 8192.0 / 4295.0,
                               places=12)
        # CPU and Disk capacities unaffected by the RAM scale.
        np.testing.assert_allclose(caps[:, 0], 4029.0, rtol=0, atol=0)
        np.testing.assert_allclose(caps[:, 2], 32212.0, rtol=0, atol=0)

    def test_ram_only_variable(self):
        self._clean_env()
        os.environ["RAM_CAP_SCALE"] = "0.45"
        try:
            caps = host_capacities()
        finally:
            self._clean_env()
        np.testing.assert_allclose(caps[0, 1], 4295.0 * 0.45, rtol=0, atol=1e-9)
        np.testing.assert_allclose(caps[8, 1], 8192.0 * 0.45, rtol=0, atol=1e-9)

    def test_invalid_scales_rejected(self):
        self._clean_env()
        for bad in ("0", "-1", "abc", "nan", "inf"):
            os.environ["RAM_CAP_SCALE"] = bad
            with self.assertRaises(ValueError, msg="scale=%r" % bad):
                host_capacities()
        self._clean_env()


if __name__ == "__main__":
    unittest.main(verbosity=2)
