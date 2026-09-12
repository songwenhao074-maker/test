"""Protocol 023 — why is the io_first disk-fault rate so low? (read-only probe)

Reads a single-regime stream and reports, for the regime's own cascade tasks:
per-task onset-resource excursion, the label-class the host carries during the
task's window, and the host-level demand that produced it.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import probe_ftmoe_protocol022_learnability as probe

STREAMS = ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
TAGS = {"A": "single_compute_first_seed700_steps1200",
        "B": "single_memory_first_seed700_steps1200",
        "C": "single_io_first_seed700_steps1200"}
RES_INDEX = {"cpu": 0, "ram": 1, "disk": 2}


def load(tag):
    d = STREAMS / tag
    data = probe.load_stream(d)
    with np.load(d / "stream.npz", allow_pickle=True) as z:
        stream = {k: np.asarray(z[k]) for k in z.files}
    with np.load(d / "task_timeline.npz", allow_pickle=True) as z:
        timeline = {k: np.asarray(z[k]) for k in z.files}
    return data, stream, timeline


def main():
    for regime, tag in TAGS.items():
        data, stream, tl = load(tag)
        steps = data["steps"]
        labels = data["labels"]
        casc = stream["host_cascade_event"][:steps]
        cflags = stream["cascade_task_flags"][:steps]
        cphase = stream["cascade_phases"][:steps]
        regimes = stream["cascade_regimes"][:steps]
        demand = tl["demand"]
        cid = tl["creation_id"]
        tcol = tl["time"]
        host = tl["host_id"]

        # Per-task max of each resource, and the mask of "the task is inside its
        # registered cascade window" from the audit columns.
        per_task = {}
        for i in range(tcol.size):
            c = int(cid[i])
            rec = per_task.setdefault(c, {"peak": np.zeros(3), "first": int(tcol[i]),
                                          "last": int(tcol[i]), "hosts": set(),
                                          "obs": 0})
            rec["peak"] = np.maximum(rec["peak"], demand[i, :3])
            rec["obs"] += 1
            rec["hosts"].add(int(host[i]))
        print("=== regime %s (%s) ===" % (regime, tag))
        print("steps=%d labels class counts=%s" % (
            steps, [int((labels[:steps] == k).sum()) for k in range(4)]))

        # cascade flags on the model-facing slot array: how many cascade-task
        # observations are there at all?
        n_flag = int(cflags[:steps].sum())
        print("cascade_task_flag cells=%d  cascade window cells=%d  cascade masked cells=%d"
              % (n_flag, int((cphase[:steps] > 0).sum()),
                 int(((cphase[:steps] > 0) & (cflags[:steps] > 0)).sum())))
        print("host_cascade_event cells=%d" % int((casc >= 0).sum()))

        # What demand does a cascade task actually reach?
        peak_arr = np.array([per_task[c]["peak"] for c in per_task])
        print("all tasks: %d (timeline rows %d)" % (peak_arr.shape[0], tcol.size))
        for j, name in enumerate(("cpu", "ram", "disk")):
            v = peak_arr[:, j]
            print("  ALL task %s peak: n=%d min=%.1f median=%.1f p99=%.1f max=%.1f"
                  % (name, v.size, v.min(), np.median(v),
                     np.percentile(v, 99), v.max()))

        # host-level aggregate demand vs capacity: what drives the label?
        caps = data["capacities"]
        totals = stream["post_totals"][:steps]
        overflow = totals - caps
        worst = np.nanmax(overflow, axis=1)          # [T, 3]
        lab_hist = {}
        for t in range(steps):
            k = int(labels[t].argmax()) if (labels[t] > 0).any() else -1
            lab_hist[k] = lab_hist.get(k, 0) + 1
        print("  per-interval argmax-overflow resource: %s"
              % {(-1 if k < 0 else k): v for k, v in sorted(lab_hist.items())})
        for j, name in enumerate(("cpu", "ram", "disk")):
            over = (overflow[:, :, j] > 0).sum()
            print("  %s: host cells above capacity=%d  max overflow=%.1f"
                  % (name, over, np.nanmax(overflow[:, :, j])))
        print()


if __name__ == "__main__":
    main()
