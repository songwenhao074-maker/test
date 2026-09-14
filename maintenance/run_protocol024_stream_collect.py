"""Portable launcher for the Protocol-024 hosted-CI collector.

Protocol-023's helper carries workstation-specific assumptions that are not
portable to GitHub-hosted Ubuntu: the Windows-only below-normal priority class
and local free-resource guards. Protocol-024 records its hosted-CI discipline
separately in ``next_round_v1/execution_environment.json``: POSIX nice +10,
RAM guard 1.5 GiB and disk guard 5 GiB.

The RAM guard was reduced from 2.5 GiB after run 34839589954 spent ~88 minutes
successfully executing the real simulator and was then stopped solely because
available RAM stayed at 2.479 GiB for the inherited 180-second guard window.
There was no allocation/OOM failure. 1.5 GiB still leaves a material emergency
margin while avoiding a false abort on the hosted runner. This wrapper changes
no workload, seed, timeline, label rule, response-law numeric, scheduler, or
model input.
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
_p23_guard.RAM_GUARD_GIB = 1.5

from prepare_ftmoe_protocol024_stream import main


if __name__ == "__main__":
    main()
