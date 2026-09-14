"""Portable launcher for the Protocol-024 hosted-CI collector.

Protocol-023's helper carries two workstation-specific assumptions that are not
portable to GitHub-hosted Ubuntu: the Windows-only below-normal priority class
and a 20 GiB free-disk guard.  Protocol-024 registers its hosted-CI resource
discipline separately in ``next_round_v1/execution_environment.json``: POSIX
nice +10, RAM guard 2.5 GiB and disk guard 5 GiB.  This wrapper changes no
workload, seed, timeline, label rule, response-law numeric, or model input.
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
# globals to the Protocol-024 registered hosted-CI threshold.
import prepare_ftmoe_protocol023_stream as _p23_guard
_p23_guard.DISK_GUARD_GIB = 5.0
_p23_guard.RAM_GUARD_GIB = 2.5

from prepare_ftmoe_protocol024_stream import main


if __name__ == "__main__":
    main()
