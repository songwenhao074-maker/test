"""Protocol-024 anchor construction for the ``raw_next_fault`` task.

The Protocol-023/P20 anchor pool stores labels for the old tolerance/current-
detection objective and therefore cannot be reused as labels in Protocol-024.
This module deliberately reuses only an *independent historical stream's*
observable 12-step windows.  Every anchor target is reconstructed as the same-
host physical ``raw_label[i+1]`` resource class.  Stored legacy labels are never
read.

The default source is the already-collected calibrated Protocol-023 development
stream.  It is independent of the new R1/R2/R3 response-law stream and was
physically observed before this next-round pilot.  A deterministic stride is
used rather than target-dependent sampling so the anchor population cannot be
post-selected for a favorable class mix.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4


ANCHOR_HISTORY = 12
ANCHOR_START = ANCHOR_HISTORY - 1
ANCHOR_STRIDE = 6
ANCHOR_TARGET = "raw_label[i+1] same host/resource class"


def raw_next_anchor_labels(raw_labels, indices):
    """Return resource labels at i+1; legacy/tolerance labels are not accepted.

    ``raw_labels`` must carry the future guard/next row.  The explicit helper is
    intentionally tiny so a regression test can prove the one-step target
    without loading a model or historical artifact.
    """
    raw = np.asarray(raw_labels, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != 16:
        raise ValueError("raw_labels must have shape [time,16]")
    if idx.size == 0:
        raise ValueError("anchor index set is empty")
    if int(idx.min()) < 0 or int(idx.max()) + 1 >= raw.shape[0]:
        raise ValueError("anchor index lacks a physical i+1 raw-label row")
    target = raw[idx + 1].copy()
    if not np.isin(target, [0, 1, 2, 3]).all():
        raise ValueError("raw-next anchor target must be resource class 0..3")
    return target


def _indices_digest(indices):
    array = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_protocol024_raw_next_anchor(source_stream=None, chunk_size=32):
    """Build the shared static P24 anchor pool from independent history.

    The returned tensor keys match ``PrequentialS4.update``.  ``meta`` is audit
    metadata and is ignored by the optimizer.  No row from the new Protocol-024
    response-law stream is used here.
    """
    source = Path(s4.DEV_STREAM if source_stream is None else source_stream)
    bundle = s4.build_replay(source)
    steps = int(bundle["steps"])
    raw = np.asarray(bundle["arrays"]["raw_labels"], dtype=np.int64)
    if raw.shape[0] < steps + 1:
        raise ValueError("historical anchor source has no physical future row")

    indices = np.arange(ANCHOR_START, steps, ANCHOR_STRIDE, dtype=np.int64)
    # Keep only scored t for which t+1 physically exists.  ``steps`` is the
    # scored count, so the final allowed t is steps-1 and raw[steps] is its
    # already-observed guard/next row.
    indices = indices[indices + 1 < raw.shape[0]]
    targets = raw_next_anchor_labels(raw, indices)

    parts = {key: [] for key in ("x", "schedule", "graph_x", "ids", "before", "caps")}
    for left in range(0, len(indices), int(chunk_size)):
        chunk = indices[left:left + int(chunk_size)].tolist()
        x, schedule, graph_x, context = s4.window_batch(bundle["replay"], chunk)
        parts["x"].append(x.detach().cpu())
        parts["schedule"].append(schedule.detach().cpu())
        parts["graph_x"].append(graph_x.detach().cpu())
        parts["ids"].append(context["creation_ids"].detach().cpu())
        parts["before"].append(context["before_placement"].detach().cpu())
        parts["caps"].append(context["capacities"].detach().cpu())

    anchor = {key: torch.cat(value, dim=0) for key, value in parts.items()}
    anchor["labels"] = torch.from_numpy(targets).long()
    anchor["meta"] = {
        "protocol": "024",
        "kind": "independent_historical_raw_next_anchor",
        "target": ANCHOR_TARGET,
        "legacy_stored_labels_used": False,
        "new_response_law_stream_used": False,
        "source_stream": str(source),
        "source_stream_sha256": bundle["manifest"].get("stream_sha256"),
        "source_steps": steps,
        "history_intervals": ANCHOR_HISTORY,
        "index_start": int(indices[0]),
        "index_end_inclusive": int(indices[-1]),
        "index_stride": ANCHOR_STRIDE,
        "anchor_samples": int(len(indices)),
        "indices_sha256": _indices_digest(indices),
        "future_row_offset": 1,
        "selection_depends_on_target": False,
        "fully_observed_historical_source": True,
    }
    return anchor


def anchor_metadata(anchor):
    """Return a JSON-safe copy of the audit metadata."""
    return dict(anchor.get("meta") or {})
