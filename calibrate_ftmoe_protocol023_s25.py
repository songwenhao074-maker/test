"""Protocol 023 round 2A — S2.5 data-only marginal calibration harness.

Purpose (directive §4 and §5)
----------------------------
The io-first regime produces far too few disk onsets for the registered
specialization gate (5 h=1 within-regime test positives, directive §5 requires
>= 30 and recommends >= 50), and the three regimes' marginals are further apart
than plan §9's tolerances.  This harness runs the *registered collector loop*
with a candidate B/C parameter set injected at run time and reports the
data-only quantities the directive names:

    event count, prevalence, duration, peak ratio, rejection rates,
    h=1 onset positives, primary-window follow-up

A is never touched: it must stay byte-identical to Protocol 022's ``cascade_v2``.
Nothing here writes into a registered stream directory; every candidate lands in
its own ``_calibration`` directory and the candidate table is what the final
selection is read from.

Usage
-----
    python calibrate_ftmoe_protocol023_s25.py --list
    python calibrate_ftmoe_protocol023_s25.py --candidate c_peak24 --steps 640
    python calibrate_ftmoe_protocol023_s25.py --sweep c_probe --steps 640
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_ftmoe_protocol023_stream as collector
from simulator.workload import BitbrainWorkloadProtocol023 as gen023

OUT_ROOT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/data_calibration"
CALIB_TAG = "calibration_streams"

SCORE_RESOURCES = {"cpu": 1, "ram": 2, "disk": 3}
RES_INDEX = {"cpu": 0, "ram": 1, "disk": 2}
REGISTERED_ONSET_TAU = {"cpu": 2600.0, "ram": 2810.0, "disk": 11600.0}

# --------------------------------------------------------------------------
# Candidate parameter sets.  ``compute_first`` is never present: A is frozen.
# Every candidate is a full override dict for one regime, validated against the
# registered forbidden-change list before it is allowed to run.
# --------------------------------------------------------------------------
FORBIDDEN_KEYS = ("sequence", "onset_resource", "response_windows", "family",
                  "regime_id", "mechanism_id")

CANDIDATES = {
    # ---- io_first (C): the h=1 positive budget is the binding constraint ----
    "c_registered": {"regime": "io_first", "params": {}},
    "c_peak20": {"regime": "io_first",
                 "params": {"disk_retained_peak": 20000.0}},
    "c_peak24": {"regime": "io_first",
                 "params": {"disk_retained_peak": 24000.0}},
    "c_peak24_cpu3800": {"regime": "io_first",
                         "params": {"disk_retained_peak": 24000.0,
                                    "cpu_burst_floor": 3800.0,
                                    "cpu_burst_upper": 4600.0}},
    "c_peak24_cpu3400_ram5000": {"regime": "io_first",
                                 "params": {"disk_retained_peak": 24000.0,
                                            "cpu_burst_floor": 3400.0,
                                            "cpu_burst_upper": 4200.0,
                                            "ram_target_floor": 5000.0,
                                            "ram_burst_upper": 5400.0}},
    "c_peak24_longdisk": {"regime": "io_first",
                          "params": {"disk_retained_peak": 24000.0,
                                     "disk_duration": [6, 10]}},
    "c_peak24_prob32": {"regime": "io_first",
                        "params": {"disk_retained_peak": 24000.0,
                                   "cascade_task_probability": 0.32}},
    # The measured blocker (diagnostics/disk_onset_deficit.json): a single
    # io-first disk phase reaches ratio <= 0.83 against the registered disk
    # capacity, so a disk *cell* needs two disk phases co-located on one host.
    # These candidates therefore work on co-location density (probability) and
    # on the per-task crossing margin (peak vs cap), not on the onset law.
    "c_peak24_prob45": {"regime": "io_first",
                        "params": {"disk_retained_peak": 24000.0,
                                   "cascade_task_probability": 0.45}},
    "c_peak24_prob60": {"regime": "io_first",
                        "params": {"disk_retained_peak": 24000.0,
                                   "cascade_task_probability": 0.60}},
    "c_peak24_longdisk_prob45": {"regime": "io_first",
                                 "params": {"disk_retained_peak": 24000.0,
                                            "disk_duration": [6, 10],
                                            "cascade_task_probability": 0.45}},
    "c_peak24_shortdisk_prob45": {"regime": "io_first",
                                  "params": {"disk_retained_peak": 24000.0,
                                             "disk_duration": [2, 3],
                                             "cascade_task_probability": 0.45}},
    # Probability-response curve: the disk *cell* rate was measured to scale
    # with co-location density, so the probability is the only lever that can
    # move io-first's positive budget without touching the registered law.
    "c_prob75": {"regime": "io_first",
                 "params": {"disk_retained_peak": 24000.0,
                            "cascade_task_probability": 0.75}},
    "c_prob90": {"regime": "io_first",
                 "params": {"disk_retained_peak": 24000.0,
                            "cascade_task_probability": 0.90}},
    # memory-first: same question for the RAM onset, whose response window is
    # the disk phase (also a rare label).
    "b_prob45": {"regime": "memory_first",
                 "params": {"cascade_task_probability": 0.45}},
    "b_prob75": {"regime": "memory_first",
                 "params": {"cascade_task_probability": 0.75}},
    # The probability sweep showed the disk *cell* rate scales linearly with
    # co-location density, while shortening the disk phase destroys it (a disk
    # cell needs two disk phases on one host).  These keep the registered
    # duration bands and use the only two levers that were measured to work:
    # the cascade probability and the registered disk peak (16000 -> 24000).
    "c_prob60_peak24dur": {"regime": "io_first",
                           "params": {"disk_retained_peak": 24000.0,
                                      "cascade_task_probability": 0.60}},
    "c_prob45_peak24dur": {"regime": "io_first",
                           "params": {"disk_retained_peak": 24000.0,
                                      "cascade_task_probability": 0.45}},
    "c_prob75_peak24dur": {"regime": "io_first",
                           "params": {"disk_retained_peak": 24000.0,
                                      "cascade_task_probability": 0.75}},
    "b_prob60": {"regime": "memory_first",
                 "params": {"cascade_task_probability": 0.60}},
    "b_prob50": {"regime": "memory_first",
                 "params": {"cascade_task_probability": 0.50}},
    # Prevalence is dominated by the OTHER phases of a cascade task, so trimming
    # the downstream phase durations buys the headroom to raise the cascade
    # probability without leaving plan §9's 3%-12% band.
    "c_prob90_shortdown": {"regime": "io_first",
                           "params": {"disk_retained_peak": 24000.0,
                                      "cascade_task_probability": 0.90,
                                      "cpu_duration": [2, 3],
                                      "ram_duration": [2, 3]}},
    "c_prob100_shortdown": {"regime": "io_first",
                            "params": {"disk_retained_peak": 24000.0,
                                       "cascade_task_probability": 1.0,
                                       "cpu_duration": [2, 3],
                                       "ram_duration": [2, 3]}},
    "b_prob75_shortdown": {"regime": "memory_first",
                           "params": {"cascade_task_probability": 0.75,
                                      "disk_duration": [2, 3],
                                      "cpu_duration": [2, 3]}},
    # ---- memory_first (B): marginal tightening toward A ---------------------
    "b_registered": {"regime": "memory_first", "params": {}},
    "b_ram_floor5000": {"regime": "memory_first",
                        "params": {"ram_target_floor": 5000.0}},
    "b_ram_floor4200": {"regime": "memory_first",
                        "params": {"ram_target_floor": 4200.0,
                                   "ram_ramp_peak_mult": 2.2}},
}


def resolve_candidate(name):
    if name not in CANDIDATES:
        raise SystemExit("unknown candidate %r (have %s)"
                         % (name, sorted(CANDIDATES)))
    spec = copy.deepcopy(CANDIDATES[name])
    for key in spec["params"]:
        if key in FORBIDDEN_KEYS:
            raise SystemExit("candidate %s changes the forbidden key %s"
                             % (name, key))
    return spec


def install_candidate(spec):
    """Inject the candidate into the generator's live parameter tables.

    Two tables have to be written, and *both* facts were measured the hard way
    (two candidates once produced byte-identical streams, which is how this
    function was found to be incomplete):

    * ``gen023._REGISTERED_COMMON`` is the family dict that
      ``Protocol023MultiRegimeBWGD2.regime_params`` overlays onto every regime,
      so a shared numeric written only into ``REGIMES_V3[regime]`` is silently
      replaced by the family default;
    * ``gen023.REGIMES_V3[regime]`` holds the per-regime values (and must be
      kept in sync, because ``assert_registered_physics`` compares the two).

    Returns the previous tables so the caller can restore them.
    """
    regime = spec["regime"]
    if regime == "compute_first":
        raise SystemExit("regime A is frozen; it must stay cascade_v2")
    previous = {"common": copy.deepcopy(gen023._REGISTERED_COMMON),
                "regime": copy.deepcopy(gen023.REGIMES_V3[regime])}
    for key, value in spec["params"].items():
        if key not in gen023.REGIMES_V3[regime]:
            raise SystemExit("candidate changes an unknown key %s" % key)
        if key in gen023._REGISTERED_COMMON:
            gen023._REGISTERED_COMMON[key] = value
        gen023.REGIMES_V3[regime][key] = value
    # The on-disk registration file is not touched: only this process's tables.
    return previous


def restore_candidate(previous, regime):
    gen023._REGISTERED_COMMON = previous["common"]
    gen023.REGIMES_V3[regime] = previous["regime"]


# --------------------------------------------------------------------------
# measurement on a collected stream directory
# --------------------------------------------------------------------------
def measure(stream_dir, regime, window, mode):
    import probe_ftmoe_protocol022_learnability as probe

    data = probe.load_stream(stream_dir)
    steps = data["steps"]
    labels = data["labels"]
    with np.load(stream_dir / "stream.npz", allow_pickle=True) as z:
        stream = {k: np.asarray(z[k]) for k in z.files}
    with np.load(stream_dir / "task_timeline.npz", allow_pickle=True) as z:
        tl = {k: np.asarray(z[k]) for k in z.files}
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))

    resource = gen023.REGIMES_V3[regime]["onset_resource"]
    klass = SCORE_RESOURCES[resource]
    tau = float(REGISTERED_ONSET_TAU[resource])

    win = np.zeros(steps, dtype=bool)
    win[window[0]:window[1]] = True
    host_cells = int(win.sum() * labels.shape[1])
    positives = int(((labels[:steps] > 0) & win[:, None]).sum())
    per_class = {int(k): int(((labels[:steps] == k) & win[:, None]).sum())
                 for k in range(4)}

    # h=1 onset target, exactly the probe's definition
    in_class = (labels == klass).astype(float)
    future = np.zeros((steps, labels.shape[1]), dtype=float)
    future[:steps - 1] = in_class[1:steps]
    valid = (in_class[:steps] == 0)
    y = np.where(valid, future, np.nan)[window[0]:window[1]]
    h1_pos = int(np.nansum(y))

    # task-level onsets of the regime's own onset resource
    creation = tl["creation_id"]
    demand = tl["demand"]
    tcol = tl["time"]
    j = RES_INDEX[resource]
    per_task = {}
    for i in range(tcol.size):
        c = int(creation[i])
        rec = per_task.setdefault(c, {"first": int(tcol[i]),
                                      "peak": np.zeros(3)})
        rec["first"] = min(rec["first"], int(tcol[i]))
        rec["peak"] = np.maximum(rec["peak"], demand[i, :3])
    cohort = [c for c, r in per_task.items()
              if window[0] <= r["first"] < window[1]]
    peaks = np.array([per_task[c]["peak"][j] for c in cohort]) if cohort else np.zeros(0)
    above = int((peaks > tau).sum())

    cascade_ids = tl["cascade_event_id"] if "cascade_event_id" in tl else None
    n_cascade = 0
    if cascade_ids is not None:
        n_cascade = len({int(v) for v in cascade_ids if int(v) >= 0})

    # durations of the onset-resource label runs (fault life)
    runs = []
    mask = (labels[:steps] == klass)
    for h in range(mask.shape[1]):
        run = 0
        for t in range(steps):
            if mask[t, h]:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)

    # deployment / migration rejection inside the window
    attempts = int(stream["deploy_attempts"][:steps][win].sum())
    rejected = int(stream["deploy_rejected"][:steps][win].sum())
    m_attempts = int(stream["migrate_attempts"][:steps][win].sum())
    m_rejected = int(stream["migrate_rejected"][:steps][win].sum())

    out = {
        "stream": stream_dir.name, "mode": mode, "regime": regime,
        "window": list(window), "steps": steps,
        "onset_resource": resource, "onset_tau": tau,
        "host_cells": host_cells,
        "positive_hoststeps": positives,
        "prevalence": positives / float(host_cells) if host_cells else 0.0,
        "label_class_counts": per_class,
        "h1_onset_positives": h1_pos,
        "tasks_born_in_window": len(cohort),
        "tasks_above_tau": above,
        "cascade_events_in_stream": n_cascade,
        "onset_label_run_lengths": {
            "n": len(runs),
            "median": float(np.median(runs)) if runs else None,
            "mean": float(np.mean(runs)) if runs else None,
            "max": int(max(runs)) if runs else None},
        "deploy_attempts": attempts, "deploy_rejected": rejected,
        "deploy_rejection_rate": rejected / max(attempts, 1),
        "migrate_attempts": m_attempts, "migrate_rejected": m_rejected,
        "migrate_rejection_rate": m_rejected / max(m_attempts, 1),
    }
    return out


def run_candidate(name, steps, phase_length, out_root, keep=True):
    spec = resolve_candidate(name)
    regime = spec["regime"]
    previous = install_candidate(spec)
    root = out_root / CALIB_TAG
    root.mkdir(parents=True, exist_ok=True)
    tag = "%s_%s_%s" % (name, regime, steps)
    output = root / tag
    if output.exists():
        raise SystemExit("calibration directory already exists: %s" % output)

    # Registered phase template of the *calibration* shape: a short familiar
    # head so the mechanism starts from the frozen familiar generator, then the
    # regime-only window.
    template = (("F0_baseline", int(round(steps * 0.15)), None),
                ("%s_only" % regime, 0, regime))
    template = (template[0], (template[1][0], steps - template[0][1],
                              regime))
    original_template = collector.phase_template
    original_levels = collector.SINGLE_PHASE_LENGTHS
    original_probability = collector.REGISTERED_PROBABILITY
    collector.phase_template = lambda mode, r=None: tuple(template)
    collector.SINGLE_PHASE_LENGTHS = ((template[0][0], template[0][1], None),
                                      (template[1][0], template[1][1], regime))
    # ``registered_phases`` reads the per-phase probability from this module
    # constant, so a candidate that only touched ``REGIMES_V3`` was measured to
    # be silently reset to the registered value (two candidates once produced
    # byte-identical streams, which is how this was found).
    probability = float(spec["params"].get("cascade_task_probability",
                                           collector.REGISTERED_PROBABILITY))
    collector.REGISTERED_PROBABILITY = probability
    started = time.perf_counter()
    try:
        collector.collect("single", regime, output, smoke=False, steps=steps)
    finally:
        # Capture what the generator actually carried *before* restoring: after
        # the restore these fields describe the registered tables again, not the
        # stream that was just collected.
        effective = {
            "cascade_task_probability": probability,
            "registered_common": {k: gen023._REGISTERED_COMMON[k]
                                  for k in sorted(spec["params"])
                                  if k in gen023._REGISTERED_COMMON},
            "regimes_v3": {k: gen023.REGIMES_V3[regime][k]
                           for k in sorted(spec["params"])}}
        collector.phase_template = original_template
        collector.SINGLE_PHASE_LENGTHS = original_levels
        collector.REGISTERED_PROBABILITY = original_probability
        restore_candidate(previous, regime)
    elapsed = time.perf_counter() - started

    window = (template[0][1], steps)
    measured = measure(output, regime, window, "single-calib")
    measured.update({"candidate": name, "params": spec["params"],
                     "effective": effective,
                     "elapsed_seconds": round(elapsed, 1),
                     "phase_template": [[n, l, r] for n, l, r in template]})
    if not keep:
        import shutil
        shutil.rmtree(output, ignore_errors=True)
    return measured


def append_candidate_table(row, out_root):
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "candidate_table.json"
    table = {"kind": "round2a_s25_data_only_calibration",
             "seeds": {"replay_seed": collector.REGISTERED_SEED,
                       "note": "seed 700 only; 701/702/703 stay frozen "
                               "(directive §4)"},
             "model_metrics_seen": False,
             "candidates": []}
    if path.is_file():
        table = json.loads(path.read_text(encoding="utf8"))
        table.setdefault("candidates", [])
    table["candidates"] = [c for c in table["candidates"]
                           if c.get("candidate") != row.get("candidate")]
    table["candidates"].append(row)
    path.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--sweep", default=None,
                        help="run every candidate whose name starts with this")
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    if args.list:
        for name, spec in sorted(CANDIDATES.items()):
            print("%-28s %-14s %s" % (name, spec["regime"],
                                      json.dumps(spec["params"])))
        return

    names = []
    if args.sweep:
        names = sorted(n for n in CANDIDATES if n.startswith(args.sweep))
    elif args.candidate:
        names = [args.candidate]
    else:
        raise SystemExit("pass --candidate NAME, --sweep PREFIX or --list")

    for name in names:
        try:
            row = run_candidate(name, args.steps, None, args.out)
        except SystemExit:
            raise
        except Exception as exc:              # a candidate that cannot run is data
            row = {"candidate": name, "error": "%s: %s" % (type(exc).__name__, exc),
                   "params": CANDIDATES[name]["params"],
                   "regime": CANDIDATES[name]["regime"]}
        path = append_candidate_table(row, args.out)
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ("onset_label_run_lengths",)}, ensure_ascii=False),
              flush=True)
    print("candidate table:", path.relative_to(ROOT))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    main()
