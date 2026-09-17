"""Portable launcher for the frozen Protocol-025 segmented recovery implementation.

Engineering-only repair: load the implementation from the pinned pre-fix commit
and apply Linux execution/import compatibility shims. No scientific setting,
generator source, seed, feature, service law, comparator, or metric is changed.
"""
from pathlib import Path
import sys
import urllib.request
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# scheduler.GOBI normally appends this relative path before code imports
# ``src.constants``.  The pinned segmented recovery imports src.constants
# directly first, so reproduce the same import environment explicitly.
BAGTI_ROOT = REPO_ROOT / "scheduler" / "BaGTI"
if str(BAGTI_ROOT) not in sys.path:
    sys.path.append(str(BAGTI_ROOT))

if not hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    psutil.BELOW_NORMAL_PRIORITY_CLASS = 10

PINNED = "09e86275c454a90a7d3526859634036ffdfda1d4"
URL = "https://raw.githubusercontent.com/songwenhao074-maker/test/%s/maintenance/run_protocol025_segmented_recovery.py" % PINNED
with urllib.request.urlopen(URL, timeout=60) as response:
    source = response.read()

# Preserve the repository-local __file__ expected by the pinned implementation.
# The repository root is also explicitly on sys.path because invoking this
# launcher by filename otherwise makes Python put maintenance/ at sys.path[0].
code = compile(source, str(Path(__file__).resolve()), "exec")
exec(code, {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
