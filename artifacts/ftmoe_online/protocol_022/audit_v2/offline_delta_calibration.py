"""Throwaway diagnostic: offline task-level CPU delta distribution.

Decides how the U4-v2 reference must be calibrated.  If every offline per-task
CPU demand is clipped at the familiar ceiling (1860) then a registered onset
threshold of 2600 has ZERO offline task-level onsets, and the plan's rule
"candidate p90 > offline p97.5" is undefined rather than passed.  This script
measures that instead of assuming it.

OFFLINE DATA ONLY - no candidate stream is read.
"""
import json
from pathlib import Path

import numpy as np

import ftmoe_protocol022_core as core
import analyze_ftmoe_protocol022_unseen as az

ROOT = Path(__file__).resolve().parent
corpora = az.offline_task_corpora()
print("corpora:", {k: len(v.get("timelines") or v.get("arrays") or [])
                   for k, v in corpora.items()})

summary = {}
for name, corpus in corpora.items():
    if corpus["kind"] != "task_level":
        print("%-40s HOST-AGGREGATE ONLY (%s)" % (name, corpus.get("exclusion_reason", "")[:60]))
        continue
    cpu_max = -1.0
    deltas = []
    n_tasks = 0
    onsets_registered = 0
    onsets_cal = 0
    for timeline in corpus["timelines"]:
        for entry in timeline.values():
            n_tasks += 1
            cpu_max = max(cpu_max, float(entry["cpu"].max()))
            d = core.delta_cpu(entry["cpu"])
            finite = d[np.isfinite(d)]
            if finite.size:
                deltas.append(finite)
        onsets_registered += len(core.onset_positions.__wrapped__(  # noqa
            np.zeros(0))) if False else 0
        onsets_registered += len([e for e in core.onset_events(timeline)
                                  if e["delta_cpu"] >= core.ONSET_TAU_CPU])
        onsets_cal += len([e for e in core.onset_events(
            timeline, tau=930.0) ])
    pooled = np.concatenate(deltas) if deltas else np.zeros(0)
    summary[name] = {
        "tasks": n_tasks,
        "cpu_max": cpu_max,
        "n_deltas": int(pooled.size),
        "delta_p50": float(np.percentile(pooled, 50)) if pooled.size else None,
        "delta_p75": float(np.percentile(pooled, 75)) if pooled.size else None,
        "delta_p90": float(np.percentile(pooled, 90)) if pooled.size else None,
        "delta_p99": float(np.percentile(pooled, 99)) if pooled.size else None,
        "delta_max": float(pooled.max()) if pooled.size else None,
        "onsets_tau2600": onsets_registered,
        "onsets_tau930": onsets_cal,
    }
    print("%-40s tasks=%6d cpu_max=%8.1f delta_p99=%8.1f delta_max=%8.1f "
          "onsets@2600=%d onsets@930=%d"
          % (name, n_tasks, cpu_max,
             summary[name]["delta_p99"] or -1, summary[name]["delta_max"] or -1,
             onsets_registered, onsets_cal))

print(json.dumps(summary, ensure_ascii=False, indent=2))
