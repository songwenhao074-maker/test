"""Portable launcher for the frozen Protocol-025 segmented recovery implementation.

Engineering-only repair: the implementation is loaded from the pinned pre-fix
commit, while Linux receives the same psutil compatibility shim already used by
maintenance/run_protocol025_stream_collect.py. No scientific setting changes.
"""
from pathlib import Path
import runpy
import tempfile
import urllib.request
import psutil

if not hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    psutil.BELOW_NORMAL_PRIORITY_CLASS = 10

PINNED = "09e86275c454a90a7d3526859634036ffdfda1d4"
URL = "https://raw.githubusercontent.com/songwenhao074-maker/test/%s/maintenance/run_protocol025_segmented_recovery.py" % PINNED

with urllib.request.urlopen(URL, timeout=60) as response:
    source = response.read()
with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
    f.write(source)
    path = f.name
try:
    runpy.run_path(path, run_name="__main__")
finally:
    Path(path).unlink(missing_ok=True)
