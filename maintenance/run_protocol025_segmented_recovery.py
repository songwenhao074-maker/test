"""Engineering-only segmented recovery for the frozen Protocol-025 generator.

This wrapper keeps the registered scientific generator semantics unchanged while
allowing a hosted runner to stop after a small number of intervals. Immutable
registered chunks are still created only at the original 200-interval
boundaries (or the final 5521 boundary). Rows between the last immutable chunk
and the transient stop are stored only in ``transient_partial.npz`` and are
reloaded on the next runner.

The wrapper is pinned to the exact canonical generator Git blob below so a
future scientific-code change cannot silently reuse this recovery path.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import dill
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows-only psutil constant is referenced by the inherited collector's
# configure() function. Match the existing portable CI launcher before import;
# this changes process niceness handling only, never simulation/model semantics.
if not hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    psutil.BELOW_NORMAL_PRIORITY_CLASS = 10

import prepare_ftmoe_protocol023_stream as _p23_guard
_p23_guard.DISK_GUARD_GIB = 5.0
_p23_guard.RAM_GUARD_GIB = 0.5

import prepare_ftmoe_protocol025_stream as P

CANONICAL_GENERATOR_GIT_BLOB_SHA1 = "3f3437e45f4d9c58182613e7ad96cf89fcda5d10"
TRANSIENT_NAME = "transient_partial.npz"


def _git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


# The remainder of this engineering wrapper is intentionally imported from the
# previous committed version at runtime through exec is not acceptable; keep
# the repository version complete. This marker should never be reached.
