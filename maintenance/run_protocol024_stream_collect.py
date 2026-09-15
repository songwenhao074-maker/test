"""Portable launcher for the Protocol-024 hosted-CI collector.

Protocol-023's helper carries workstation-specific assumptions that are not
portable to GitHub-hosted Ubuntu: the Windows-only below-normal priority class
and local free-resource guards. Protocol-024 records its hosted-CI discipline
separately in ``next_round_v1/execution_environment.json``: POSIX nice +10,
RAM emergency floor 0.5 GiB and disk guard 5 GiB.

Two long, otherwise healthy hosted runs demonstrated that a fixed multi-GiB
``available`` RAM threshold is not a valid steady-state requirement on this
runner: run 34839589954 was stopped at 2.479 GiB after ~88 min and run
34850369770 was stopped at 1.484 GiB after ~163 min, with neither run reporting
an allocation failure or OOM. The guard is therefore retained only as a true
emergency floor at 0.5 GiB. This wrapper changes no workload, seed, timeline,
label rule, response-law numeric, scheduler, or model input.
"""
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    psutil.BELOW_NORMAL_PRIORITY_CLASS = 10

# Import the helper module itself so the inherited guard function resolves its
# globals to the Protocol-024 hosted-CI thresholds.
import prepare_ftmoe_protocol023_stream as _p23_guard
_p23_guard.DISK_GUARD_GIB = 5.0
_p23_guard.RAM_GUARD_GIB = 0.5

# The registered collector constant is SCORDED_STEPS. The long hosted run at
# fe718b1 reached final audit construction and exposed a single legacy typo
# (SCORED_STEPS) there. Alias the intended constant in the module namespace so
# the finalized stream can be audited without changing any scientific setting.
import prepare_ftmoe_protocol024_stream as _p24
_p24.SCORED_STEPS = _p24.SCORDED_STEPS
main = _p24.main


if __name__ == "__main__":
    main()
