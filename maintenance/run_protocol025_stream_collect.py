"""Portable hosted-CI launcher for Protocol-025 generation."""
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    psutil.BELOW_NORMAL_PRIORITY_CLASS = 10

import prepare_ftmoe_protocol023_stream as _p23_guard
_p23_guard.DISK_GUARD_GIB = 5.0
_p23_guard.RAM_GUARD_GIB = 0.5

import prepare_ftmoe_protocol025_stream as _p25
main = _p25.main

if __name__ == "__main__":
    main()
