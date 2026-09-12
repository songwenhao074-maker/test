"""Protocol 023 — is the registered disk envelope actually reaching the task?

Every regime registers a disk phase (``disk_retained_peak``).  This script reads
one collected stream and reports, per resource, the per-task peak demand of the
cascade tasks that the audit itself stamped with a cascade event id, next to the
registered floor of that resource's phase.

A resource whose measured peak never approaches its registered floor means the
phase is not landing on the task.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import probe_ftmoe_protocol022_learnability as probe
from simulator.workload.BitbrainWorkloadProtocol023 import REGIMES_V3

STREAMS = ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/diagnostics"
TAGS = {"compute_first": "single_compute_first_seed700_steps1200",
        "memory_first": "single_memory_first_seed700_steps1200",
        "io_first": "single_io_first_seed700_steps1200"}
FLOOR_KEY = {"cpu": "cpu_burst_floor", "ram": "ram_target_floor",
             "disk": "disk_retained_peak"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for regime, tag in TAGS.items():
        d = STREAMS / tag
        data = probe.load_stream(d)
        steps = data["steps"]
        with np.load(d / "stream.npz", allow_pickle=True) as z:
            stream = {k: np.asarray(z[k]) for k in z.files}
        with np.load(d / "task_timeline.npz", allow_pickle=True) as z:
            tl = {k: np.asarray(z[k]) for k in z.files}
        params = REGIMES_V3[regime]
        cascade_ids = tl["cascade_event_id"]
        creation = tl["creation_id"]
        demand = tl["demand"]
        cascade_cids = {int(c) for c, e in zip(creation, cascade_ids) if int(e) >= 0}
        familiar_cids = {int(c) for c, e in zip(creation, cascade_ids) if int(e) < 0}

        def peaks(cids):
            out = {}
            for i in range(creation.size):
                c = int(creation[i])
                if c not in cids:
                    continue
                rec = out.setdefault(c, np.zeros(3))
                # task_timeline.demand columns are [cpu, ram, ram_read, ram_write,
                # disk, disk_read, disk_write]
                out[c] = np.maximum(rec, demand[i, [0, 1, 4]])
            return np.array(list(out.values())) if out else np.zeros((0, 3))

        casc = peaks(cascade_cids)
        fam = peaks(familiar_cids)
        entry = {"stream": tag, "n_cascade_tasks": int(casc.shape[0]),
                 "n_familiar_tasks": int(fam.shape[0]),
                 "registered_floors": {r: float(params[FLOOR_KEY[r]])
                                       for r in ("cpu", "ram", "disk")},
                 "registered_sequence": [list(p) for p in params["sequence"]],
                 "per_resource": {}}
        for j, name in enumerate(("cpu", "ram", "disk")):
            v = casc[:, j] if casc.size else np.zeros(0)
            f = fam[:, j] if fam.size else np.zeros(0)
            floor = float(params[FLOOR_KEY[name]])
            entry["per_resource"][name] = {
                "cascade_peak_min": float(v.min()) if v.size else None,
                "cascade_peak_median": float(np.median(v)) if v.size else None,
                "cascade_peak_max": float(v.max()) if v.size else None,
                "cascade_tasks_above_floor": int((v >= floor * 0.99).sum()) if v.size else 0,
                "familiar_peak_max": float(f.max()) if f.size else None,
                "registered_floor": floor,
            }
        report[regime] = entry
        print("=== %s (sequence %s) ===" % (regime, entry["registered_sequence"]))
        for name, e in entry["per_resource"].items():
            print("  %-5s floor=%9.1f  cascade peaks min/med/max = %s / %s / %s  "
                  "above_floor=%d/%d   familiar max=%s"
                  % (name, e["registered_floor"],
                     _f(e["cascade_peak_min"]), _f(e["cascade_peak_median"]),
                     _f(e["cascade_peak_max"]), e["cascade_tasks_above_floor"],
                     entry["n_cascade_tasks"], _f(e["familiar_peak_max"])))
    (OUT / "envelope_landing.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf8")


def _f(v):
    return "None" if v is None else "%.1f" % v


if __name__ == "__main__":
    main()
