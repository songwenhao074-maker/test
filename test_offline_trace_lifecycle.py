"""Regression tests for demand continuity through container replacement and migration."""
from types import SimpleNamespace
import numpy as np
from simulator.workload.OfflineTraceWorkloadV2 import OfflineTraceWorkloadV2, _trace_columns


def main():
    workload = OfflineTraceWorkloadV2()
    trace = np.zeros((3, 112))
    for t in range(3):
        for slot in range(16):
            trace[t, slot * 7] = 10 + slot + 3 * t
            trace[t, slot * 7 + 1] = 20 + 2 * slot + t
            trace[t, slot * 7 + 4] = 1 + slot / 16 + t / 10
    workload.trace = trace
    env = SimpleNamespace(containerlist=[None] * 16, hostlist=[None] * 16, intervaltime=300, interval=1)
    workload.create_next_tasks(env, 0)
    workload.set_containers(env.containerlist)
    workload.apply_trace_row(1)
    for slot, container in enumerate(env.containerlist):
        expected = _trace_columns(trace[1], slot)
        actual = (container.getBaseIPS(), container.getRAM()[0], container.getDisk()[0])
        np.testing.assert_allclose(actual, expected)
    old = env.containerlist[5]
    old_cpu = old.getBaseIPS()
    env.containerlist[5] = None
    workload._slot_hosts[5] = 2
    workload.create_next_tasks(env, 1)
    replacement = env.containerlist[5]
    assert replacement is not old and replacement.getHostID() == 2
    np.testing.assert_allclose(replacement.getBaseIPS(), _trace_columns(trace[1], 5)[0])
    workload.apply_trace_row(2)
    np.testing.assert_allclose(replacement.getBaseIPS(), _trace_columns(trace[2], 5)[0])
    assert old.getBaseIPS() == old_cpu, 'Detached task must not continue receiving trace updates'
    # A vacant earlier slot must not shift the identity of a later slot.
    env.containerlist[0] = None
    workload.apply_trace_row(0)
    np.testing.assert_allclose(env.containerlist[2].getBaseIPS(), _trace_columns(trace[0], 2)[0])
    print('PASS: all live slots update; replacements update; migration retains workload identity; gaps do not shift trace columns')


if __name__ == '__main__':
    main()
