"""Protocol 023 round 2A — measure the *quantitative* disk-onset deficit.

Why the io-first h=1 positive budget is small: a disk onset needs the host's
disk ratio to be the largest of the three overflows and to cross 1.  This script
measures, on the registered io-first stream, what that competition looks like:

    * how many host-intervals cross each resource's capacity
    * how many of those are won by disk (this IS the label)
    * how much disk demand is still missing at the crossing intervals
    * how the deficit distributes over cascade vs familiar tasks
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
OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/diagnostics"
TAGS = {"A": "single_compute_first_seed700_steps1200",
        "B": "single_memory_first_seed700_steps1200",
        "C": "single_io_first_seed700_steps1200"}
WINDOW = (150, 1200)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for regime, tag in TAGS.items():
        d = STREAMS / tag
        data = probe.load_stream(d)
        with np.load(d / "stream.npz", allow_pickle=True) as z:
            stream = {k: np.asarray(z[k]) for k in z.files}
        steps = data["steps"]
        labels = data["labels"][:steps]
        caps = data["capacities"]
        totals = stream["post_totals"][:steps]
        ratio = totals / np.maximum(caps, 1e-9)
        win = slice(WINDOW[0], WINDOW[1])

        r = ratio[win]
        lab = labels[win]
        argmax_res = r.argmax(-1)
        over_any = (r > 1.0).any(-1)
        entry = {"stream": tag, "window": list(WINDOW),
                 "host_cells": int(r.shape[0] * r.shape[1]),
                 "overload_cells": int((over_any).sum())}
        per = {}
        for j, name in enumerate(("cpu", "ram", "disk")):
            over = r[:, :, j] > 1.0
            argmax_and_over = over & (argmax_res == j)
            per[name] = {
                "cells_over_capacity": int(over.sum()),
                "cells_over_and_argmax": int(argmax_and_over.sum()),
                "label_cells": int((lab == j + 1).sum()),
                "ratio_max": float(r[:, :, j].max()),
                "ratio_p99": float(np.percentile(r[:, :, j], 99)),
                "ratio_p999": float(np.percentile(r[:, :, j], 99.9)),
                "margin_when_over_max": float((r[:, :, j][over] - 1.0).max())
                if over.any() else None,
            }
            # how far below 1.0 the argmax-of-this-resource cells sit
            am = (argmax_res == j)
            per[name]["argmax_cells"] = int(am.sum())
            if am.any():
                per[name]["argmax_ratio_median"] = float(np.median(r[:, :, j][am]))
                per[name]["argmax_ratio_p90"] = float(np.percentile(r[:, :, j][am], 90))
        entry["per_resource"] = per
        entry["argmax_without_overload"] = int((~over_any).sum())
        # the decisive number: when disk is the argmax but under 1.0, the host's
        # other resources are even lower, so "disk cannot win" is the whole story
        report[regime] = entry
        print("=== %s ===" % regime)
        print(json.dumps(entry, indent=2))
    (OUT / "disk_onset_deficit.json").write_text(
        json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
