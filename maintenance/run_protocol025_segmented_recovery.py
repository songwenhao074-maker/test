"""Portable launcher for the frozen Protocol-025 segmented recovery implementation.

Engineering-only repair: load the implementation from the pinned pre-fix commit
and apply the Linux psutil compatibility shim already used by the canonical CI
launcher. No scientific setting, generator source, seed, feature, or metric is
changed.
"""
from pathlib import Path
import urllib.request
import psutil

if not hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    psutil.BELOW_NORMAL_PRIORITY_CLASS = 10

PINNED = "09e86275c454a90a7d3526859634036ffdfda1d4"
URL = "https://raw.githubusercontent.com/songwenhao074-maker/test/%s/maintenance/run_protocol025_segmented_recovery.py" % PINNED
with urllib.request.urlopen(URL, timeout=60) as response:
    source = response.read()

# Preserve the repository-local __file__ expected by the pinned implementation.
code = compile(source, str(Path(__file__).resolve()), "exec")
exec(code, {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
