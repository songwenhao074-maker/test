"""Protocol 023 S3 — gradient interference gate, plan §14 (gate H3).

The question this answers
-------------------------
Plan §14 is the decisive gate in front of *dynamic experts* (D).  Three
heterogeneous cascade regimes are registered (A compute-first CPU->RAM->Disk,
B memory-first RAM->Disk->CPU, C io-first Disk->CPU->RAM).  A *fixed* online
learner (Protocol 020 R1: frozen v4 base + four fixed residual experts + one
dense router) is supposed to be pulled in different directions by the three
regimes.  If it is not — if the three regimes' residual gradients are nearly
co-directional — then there is no continual-learning conflict for a dynamic
expert to resolve, and the plan's own instruction is `STOP-MR`: do not
implement D.

The registered instrument (plan §14, mirrored by
``artifacts/ftmoe_online/protocol_023/protocol.json`` -> ``gradient_gate``)::

    g_A, g_B, g_C = mean residual-parameter gradient over the mature fault
                    events of each regime
    report  cos(g_A,g_B), cos(g_A,g_C), cos(g_B,g_C)
    also    router gradient cosine, expert gradient cosine, per-layer cosine

    conflict_present  iff  at least one pair has mean cosine <= -0.05
                     or    at least one pair has >= 30% of its sampled
                           cross-regime event pairs with cosine < 0

Everything the instrument needs is reused from the frozen protocol code
(nothing is re-implemented from memory):

* ``run_ftmoe_protocol020.ReplayV3``          — v2 time/graph normalization,
  the 12-interval causal window and graph-semantics-v3 context
  (``window_v3``), exactly as the P20/P22 runners call it;
* ``run_ftmoe_protocol020.load_v2_time_scale_p20`` — the registered v2 scale;
* ``run_ftmoe_online.tolerance_label``        — the frozen label convention the
  online learner consumes (nearest anomaly wins, an equal-distance tie keeps
  the earlier interval) and therefore also the maturity rule: a host-step is
  mature once the raw labels of ``t-1, t, t+1`` have all been observed;
* ``recovery.PreGANSrc.src.ftmoe_online_s7.balanced_weights`` — the registered
  class-balanced loss weights, computed on the protocol-020 adaptation train
  bundle only (never on the stream);
* ``recovery.PreGANSrc.src.ftmoe_online_r1.FrozenResidualFTMoE`` — the fixed-C
  model.  The online-trainable residual parameters are read from the model
  itself: ``FrozenResidualFTMoE.set_trainability`` sets
  ``requires_grad_(name.startswith("learner."))`` (ftmoe_online_r1.py:183-186)
  and ``correction_parameters`` returns ``list(bank.parameters())`` of the
  ``learner`` bank (ftmoe_online_r1.py:198-200).  The instrument therefore
  takes ``[p for p in model.named_parameters() if p.requires_grad]`` — no name
  is guessed.

Honesty notes registered with the result
----------------------------------------
1. **No parameter update.**  The model is put in the R1 train-mode contract
   (``FrozenResidualFTMoE.train``, which forces every module into the
   deterministic eval path — the residual has no dropout), gradients are
   obtained with ``torch.autograd.grad`` (which never writes ``.grad``), no
   optimizer is constructed, and the SHA-256 of every parameter/buffer tensor
   is recorded before and after.  A single changed byte aborts the run.
2. **NaN discipline.**  A single non-finite gradient or loss raises
   ``NonFiniteGradientError``.  Protocol 022 lost a round to exactly this
   failure mode: a NaN gradient silently produced NaN cosines which compared
   ``False`` against every threshold and were read as "no conflict".
3. **The frozen start state is a residual no-op.**  The registered R1 start
   bank zeroes its final layer (ftmoe_online_r1.py:80-84), so at the loaded
   state the residual correction is exactly zero.  The gradients of the
   *hidden* residual layers are then structurally zero (they are multiplied by
   the zero read-out matrix), and the router only receives the registered
   balance term.  The instrument measures this instead of hiding it: every
   group whose gradient is exactly zero is reported under
   ``degenerate_groups`` with a ``null`` cosine (a zero-length vector has no
   cosine) rather than a fake 0.0, and the verdict is computed only from the
   groups where the cosine is defined.  ``--learner-state`` accepts a mature
   fixed-C learner state so the same instrument can be re-run on a residual
   that has actually been trained.
4. **Per-layer decomposition.**  It is possible without touching frozen code
   (the residual bank is ``router`` + 4 x [LayerNorm, Linear, GELU, Linear] and
   every tensor name encodes its module), and is reported under ``per_layer``.

Usage:
    python probe_ftmoe_protocol023_gradient.py
    python probe_ftmoe_protocol023_gradient.py --events 32 --seed 700
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# The instrument is registered CPU-only (protocol.json -> resource_discipline):
# CUDA is hidden from the process *before* torch is imported, so no code path in
# this probe can silently move a tensor to a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "3"

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
import torch.nn.functional as F                            # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:                 # ``python -c`` / foreign cwd
    sys.path.insert(0, str(ROOT))

PROTOCOL = "023"
STAGE = "P23-S3"
KIND = "gradient_interference_probe"
PLAN = "指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md"
PLAN_SECTION = "§14 Gradient Interference Gate"
PROTOCOL_JSON = ROOT / "artifacts/ftmoe_online/protocol_023/protocol.json"
STREAM_DIR = (ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
                     "/dev_seed700_steps2880")
OUT_DIR = ROOT / "artifacts/ftmoe_online/protocol_023/specialization"
OUT_NAME = "gradient_interference.json"
CHECKPOINT = ROOT / "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt"
CHECKPOINT_SHA256 = ("10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594"
                     "e4d8792519dfe03b")
P20_ADAPTATION = ROOT / "artifacts/ftmoe_online/protocol_020/adaptation_data/v1"

# Registered regime ids (plan §8) and the registered development timeline
# (plan §10), which is also written into the stream manifest's ``phases``.
REGIME_IDS = ("compute_first", "memory_first", "io_first")
PHASE_REGIME = {
    "A1_compute": "compute_first",
    "A2_recur": "compute_first",
    "B1_memory": "memory_first",
    "B2_recur": "memory_first",
    "C1_io": "io_first",
    "C2_recur": "io_first",
}
REGISTERED_DEV_PHASES = (("F0_baseline", 300), ("A1_compute", 420),
                         ("B1_memory", 420), ("C1_io", 420),
                         ("F1_baseline", 240), ("A2_recur", 360),
                         ("C2_recur", 360), ("B2_recur", 360))
REGISTERED_DEV_STEPS = sum(length for _, length in REGISTERED_DEV_PHASES)

# Registered gate (plan §14; protocol.json -> gradient_gate).  Both boundaries
# are inclusive exactly as registered: ``<= -0.05`` and ``>= 30%``.
PAIR_MEAN_COSINE_MAX = -0.05
NEGATIVE_PAIR_FRACTION_MIN = 0.30
PRIMARY_GROUP = "all"

DEFAULT_EVENTS = 64
#: Sampling seed: the P20 R1 RNG convention (``model_seed * 7919 + replay_seed``)
#: with the registered development replay seed 700 and the plan section number
#: (14) as the offset, so the default is reproducible without a magic constant.
DEFAULT_SEED = (700 * 7919 + 14) % (2 ** 32)
BOOTSTRAP_REPS = 1000
BOOTSTRAP_LEVEL = 0.95

#: Registered S7 loss v3 weights (run_ftmoe_protocol020_r1.py configuration:
#: ".7 detection CE + .3 positive classification CE + .5 joint ranking
#: + .01 balance").
LOSS_WEIGHTS = {"detection": 0.7, "classification": 0.3,
                "ranking": 0.5, "balance": 0.01}

RAM_GUARD_GIB = float(os.environ.get("FTMOE023_RAM_GUARD_GIB", "2.5"))
TORCH_THREADS = 3
INTEROP_THREADS = 1
EXPERT_COUNT = 4


class NonFiniteGradientError(RuntimeError):
    """A non-finite loss or gradient reached the instrument.

    Protocol 022 lost a round to this exact failure mode: one NaN gradient made
    every cosine NaN, every comparison against the gate threshold returned
    False, and the run was read as "no conflict".  A NaN here is a broken
    measurement, not a result, so the instrument refuses to report it.
    """


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hashes(model):
    """Per-tensor SHA-256 of every parameter *and* buffer of the model."""
    return {name: tensor_sha256(value)
            for name, value in model.state_dict().items()}


def digest_of(mapping):
    digest = hashlib.sha256()
    for name, value in sorted(mapping.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def assert_finite(values, what):
    """The NaN guard: raise on the first non-finite entry."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise NonFiniteGradientError(
            "%s contains %d non-finite value(s) (Protocol 022's silent-NaN "
            "failure mode); refusing to report a cosine from it"
            % (what, bad))
    return array


def _as_vector(values, what="gradient"):
    array = assert_finite(values, what).reshape(-1)
    if array.size == 0:
        raise ValueError("%s is empty" % what)
    return array


def cosine(left, right):
    """Cosine of two gradient vectors, or ``None`` when it is undefined.

    ``None`` is returned when either vector has zero norm: a zero gradient
    carries no direction, and reporting 0.0 for it would be a fabricated
    measurement (the caller records the degeneracy instead).
    """
    a = _as_vector(left, "left gradient")
    b = _as_vector(right, "right gradient")
    if a.shape != b.shape:
        raise ValueError("gradient shapes differ: %s vs %s" % (a.shape, b.shape))
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(np.dot(a, b) / (norm_a * norm_b))


def mean_gradient(vectors, what="gradient"):
    """Mean of a list/array of gradient vectors, guarded against NaN."""
    stacked = np.asarray(vectors, dtype=np.float64)
    if stacked.ndim != 2 or stacked.shape[0] == 0:
        raise ValueError("%s: expected [n_events, n_parameters], got %s"
                         % (what, (stacked.shape,)))
    assert_finite(stacked, what)
    return stacked.mean(axis=0)


def cross_pair_cosines(left, right):
    """``[n_left, n_right]`` cosine matrix; NaN where a row is zero-norm."""
    a = assert_finite(left, "left event gradients")
    b = assert_finite(right, "right event gradients")
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    if a.shape[1] != b.shape[1]:
        raise ValueError("gradient widths differ: %d vs %d"
                         % (a.shape[1], b.shape[1]))
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    defined = (norm_a > 0)[:, None] & (norm_b > 0)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        values = ((a / np.where(norm_a > 0, norm_a, 1.0)[:, None])
                  @ (b / np.where(norm_b > 0, norm_b, 1.0)[:, None]).T)
    values = np.clip(values, -1.0, 1.0)
    values[~defined] = np.nan
    return values


def bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, level=BOOTSTRAP_LEVEL,
                      rng=None):
    """Percentile bootstrap CI of the mean over resampled event pairs.

    The resampling unit is the *event pair* (``n_pairs`` cosines), which is the
    unit the registered rule counts its 30% on.
    """
    data = assert_finite(values, "bootstrap input")
    if data.size == 0:
        return None
    if rng is None:
        rng = np.random.default_rng(0)
    indices = rng.integers(0, data.size, size=(int(reps), data.size))
    means = data[indices].mean(axis=1)
    low, high = np.percentile(means, [100.0 * (1.0 - level) / 2.0,
                                      100.0 * (1.0 + level) / 2.0])
    return [float(low), float(high)]


def pair_statistics(left, right, group, pair, seed, reps=BOOTSTRAP_REPS,
                    level=BOOTSTRAP_LEVEL):
    """Event-pair statistics of one regime pair for one parameter group.

    Reported over the *event pairs* (not only the means):
    mean cosine, the count/fraction of pairs with cosine < 0, and a bootstrap
    95% CI of that mean.  Pairs whose cosine is undefined (a zero-norm event
    gradient on either side) are excluded from every statistic and counted
    separately, so a degenerate group cannot masquerade as "no conflict".
    """
    matrix = cross_pair_cosines(left, right)
    flat = matrix.reshape(-1)
    defined = flat[np.isfinite(flat)]
    undefined = int((~np.isfinite(flat)).sum())
    entry = {
        "group": group,
        "pair": pair,
        "n_events_left": int(len(left)),
        "n_events_right": int(len(right)),
        "n_pairs": int(flat.size),
        "n_pairs_defined": int(defined.size),
        "n_pairs_undefined": undefined,
        "mean_cosine": (float(defined.mean()) if defined.size else None),
        "median_cosine": (float(np.median(defined)) if defined.size else None),
        "negative_pairs": int((defined < 0).sum()),
        "negative_fraction": (float((defined < 0).mean()) if defined.size
                              else None),
        "min_cosine": (float(defined.min()) if defined.size else None),
        "max_cosine": (float(defined.max()) if defined.size else None),
        "bootstrap_ci95": (bootstrap_mean_ci(
            defined, reps=reps, level=level,
            rng=np.random.default_rng([int(seed), int(group_ordinal(group)),
                                       int(pair_ordinal(pair))]))
            if defined.size else None),
        "bootstrap_reps": int(reps),
    }
    if undefined:
        entry["undefined_reason"] = ("zero-norm event gradient on one side "
                                     "(no direction to compare)")
    return entry


def group_ordinal(group):
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16)


def pair_ordinal(pair):
    return int(hashlib.sha256(pair.encode("utf-8")).hexdigest()[:8], 16)


def regime_pairs(regimes=REGIME_IDS):
    return [(regimes[i], regimes[j])
            for i in range(len(regimes)) for j in range(i + 1, len(regimes))]


def pair_name(left, right):
    return "%s|%s" % (left, right)


# --------------------------------------------------------------------------
# the registered pass/fail rule
# --------------------------------------------------------------------------
def conflict_verdict(pair_means, pair_negative_fractions,
                     mean_max=PAIR_MEAN_COSINE_MAX,
                     fraction_min=NEGATIVE_PAIR_FRACTION_MIN):
    """The registered §14 rule, evaluated exactly as written.

    ``conflict_present`` is true when at least one regime pair has a mean
    cosine ``<= mean_max`` (-0.05) **or** at least one pair has a fraction of
    negative-cosine event pairs ``>= fraction_min`` (0.30).  Both boundaries
    are inclusive: the plan writes ``<= -0.05`` and ``>= 30%``.  A pair whose
    statistic is undefined (``None``, e.g. a zero-norm mean gradient) counts as
    neither branch — it is reported under ``undefined_pairs`` instead.
    """
    mean_conflicts, fraction_conflicts, undefined = [], [], []
    for pair in sorted(set(pair_means) | set(pair_negative_fractions)):
        mean = pair_means.get(pair)
        fraction = pair_negative_fractions.get(pair)
        if mean is None and fraction is None:
            undefined.append(pair)
            continue
        if mean is not None and float(mean) <= float(mean_max):
            mean_conflicts.append(pair)
        if fraction is not None and float(fraction) >= float(fraction_min):
            fraction_conflicts.append(pair)
    present = bool(mean_conflicts or fraction_conflicts)
    return {
        "conflict_present": present,
        "mean_cosine_conflict_pairs": mean_conflicts,
        "negative_fraction_conflict_pairs": fraction_conflicts,
        "undefined_pairs": undefined,
        "mean_cosine_max": float(mean_max),
        "negative_pair_fraction_min": float(fraction_min),
    }


def gate_verdict(pair_means, pair_negative_fractions, group=PRIMARY_GROUP):
    """The full registered verdict block for the primary parameter group."""
    verdict = conflict_verdict(pair_means, pair_negative_fractions)
    verdict["group"] = group
    verdict["registration"] = (
        "plan %s / protocol.json gradient_gate: conflict_present iff at least "
        "one regime pair has mean residual-gradient cosine <= %.2f, or at "
        "least one pair has >= %.0f%% of its sampled cross-regime event pairs "
        "with cosine < 0" % (PLAN_SECTION, PAIR_MEAN_COSINE_MAX,
                             100.0 * NEGATIVE_PAIR_FRACTION_MIN))
    verdict["mean_cosine"] = {pair: pair_means.get(pair)
                              for pair in sorted(pair_means)}
    verdict["negative_fraction"] = {pair: pair_negative_fractions.get(pair)
                                    for pair in sorted(pair_negative_fractions)}
    verdict["verdict"] = ("conflict_present" if verdict["conflict_present"]
                          else "STOP-GI")
    verdict["verdict_text"] = (
        "the three regime gradients carry a registered conflict; D may be "
        "considered (the §13 and §14 gates are separate and both must pass)"
        if verdict["conflict_present"] else
        "STOP-GI: the three regime gradients are co-directional, so there is "
        "no continual-learning conflict for a dynamic expert to resolve; do "
        "not implement D (plan §14 / protocol.json stop_table)")
    return verdict


# --------------------------------------------------------------------------
# stream segmentation, labels and sampling
# --------------------------------------------------------------------------
def normalize_phases(phases):
    """Manifest phases as ``[{name, start, end, regime}]`` (contiguity checked)."""
    return [dict(phase, regime=phase_regime(phase))
            for phase in canonical_phases(phases)]


def canonical_phases(phases):
    """Manifest phases as ``[{name, start, end}]`` with contiguity checked."""
    out = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError("phase %d is not an object" % index)
        name = phase.get("name")
        start = int(phase.get("start", 0))
        end = int(phase.get("end", 0))
        if not name or end <= start:
            raise ValueError("phase %d has no usable bounds: %r" % (index, phase))
        if start != (out[-1]["end"] if out else 0):
            raise ValueError("phases are not contiguous at %s (start %d, "
                             "previous end %s)"
                             % (name, start, out[-1]["end"] if out else 0))
        out.append({"name": str(name), "start": start, "end": end,
                    "regime_id": phase.get("regime_id")})
    return out


def phase_regime(phase):
    """The registered regime of a phase, cross-checked against the manifest."""
    name = str(phase.get("name"))
    regime = PHASE_REGIME.get(name)
    declared = phase.get("regime_id")
    if declared is not None and regime is not None and str(declared) != regime:
        raise ValueError(
            "phase %s maps to %s by the registered table but the manifest "
            "declares regime_id %r" % (name, regime, declared))
    return regime


def regime_intervals(phases, steps):
    """``({regime: bool[steps]}, {regime: [[phase, start, end], ...]})``.

    Accepts manifest phases as they appear in ``manifest.json`` or phases
    already passed through :func:`normalize_phases`.
    """
    masks = {regime: np.zeros(steps, dtype=bool) for regime in REGIME_IDS}
    seen = {regime: [] for regime in REGIME_IDS}
    for phase in phases:
        regime = (phase["regime"] if "regime" in phase
                  else phase_regime(phase))
        if regime is None:
            continue
        end = min(int(phase["end"]), steps)
        if end > phase["start"]:
            masks[regime][phase["start"]:end] = True
            seen[regime].append([phase["name"], phase["start"], end])
    for regime in REGIME_IDS:
        if not seen[regime]:
            raise ValueError("no phase activates regime %s" % regime)
    return masks, seen


def registered_timeline_check(phases, steps):
    """Is this the registered development timeline (plan §10)?"""
    observed = [(phase["name"], phase["end"] - phase["start"])
                for phase in phases]
    return {
        "steps": int(steps),
        "registered_steps": int(REGISTERED_DEV_STEPS),
        "matches_registered_timeline": bool(
            int(steps) == REGISTERED_DEV_STEPS
            and tuple(observed) == REGISTERED_DEV_PHASES),
        "observed_phases": [list(item) for item in observed],
        "registered_phases": [list(item) for item in REGISTERED_DEV_PHASES],
    }


def maturity_and_labels(raw_labels, steps):
    """Tolerance labels and the maturity mask of every mature host-step.

    The frozen convention is ``run_ftmoe_online.tolerance_label``, the function
    the S7/R1 online learner itself consumes: the label of interval ``t`` is the
    raw label unless it is 0 and a neighbour is faulted, in which case the
    nearest anomaly wins and an equal-distance tie keeps the earlier interval.
    It is called exactly as ``S7Session.step`` calls it -- once per interval
    with ``observed_until = t + 1`` -- and it returns the whole host row, so the
    tolerance rule cannot drift away from the online learner.  The frozen
    call raises while the interval's neighbourhood is not fully observed; on
    top of that guard a host whose own ``t-1, t, t+1`` rows are unobserved is
    marked immature too, because its label cannot be settled.
    """
    from run_ftmoe_online import tolerance_label

    raw = np.asarray(raw_labels, dtype=np.int64)
    if raw.ndim != 2:
        raise ValueError("raw_labels must be [steps + guard, hosts]")
    hosts = int(raw.shape[1])
    if raw.shape[0] <= steps:
        raise ValueError("stream must carry the scored horizon plus one guard "
                         "interval: %d rows for %d scored intervals"
                         % (raw.shape[0], steps))
    labels = np.full((int(steps), hosts), -1, dtype=np.int64)
    mature = np.zeros((int(steps), hosts), dtype=bool)
    #: The frozen function refuses an interval whose neighbourhood is not fully
    #: observed.  A stream in which one host is still unobserved then loses the
    #: labels of its *observed* hosts too, which the online learner never sees
    #: (it reveals a whole interval row at once).  The fallback below restores
    #: per-host semantics for exactly those hosts, and it is masked to them, so
    #: a host that depends on an unobserved neighbour stays immature.
    repaired = np.where(raw < 0, 0, raw)
    for t in range(int(steps)):
        low = max(0, t - 1)
        high = min(raw.shape[0], t + 2)
        observed = (raw[low:high] >= 0).all(axis=0)
        try:
            row = np.asarray(tolerance_label(raw, t, t + 1))
        except ValueError:
            if not observed.any():
                continue
            row = np.asarray(tolerance_label(repaired, t, t + 1))
        mature[t] = observed
        labels[t, observed] = row[observed]
    return labels, mature


def candidate_events(labels, mature, masks, label_scope="fault"):
    """``{regime: [(interval, host), ...]}`` of mature events, canonically sorted."""
    out = {}
    for regime in REGIME_IDS:
        rows, columns = np.nonzero(mature & masks[regime][:, None])
        keep = np.ones(rows.size, dtype=bool)
        if label_scope == "fault":
            keep = labels[rows, columns] > 0
        elif label_scope != "all":
            raise ValueError("unknown label scope %r" % (label_scope,))
        rows, columns = rows[keep], columns[keep]
        order = np.lexsort((columns, rows))
        out[regime] = [(int(rows[i]), int(columns[i])) for i in order]
    return out


def sample_events(candidates, n_events, seed, n_hosts=16):
    """Deterministic, host-balanced sample of host-step events.

    Round-robin over the hosts in a seeded host order, each host's candidates
    shuffled with the same seeded generator, so that no single host can dominate
    the regime gradient and the sample is reproducible byte-for-byte for a fixed
    seed.  Returns ``(events, description)``.
    """
    events = sorted((int(t), int(h)) for t, h in candidates)
    n_events = int(n_events)
    description = {
        "n_candidates": len(events),
        "n_hosts": int(n_hosts),
        "requested_events": n_events,
        "seed": int(seed),
        "rule": ("host-stratified round robin: hosts are visited in a seeded "
                 "random order and each host contributes at most one event per "
                 "round, so a regime's sample is balanced across hosts"),
        "shortfall": 0,
    }
    if n_events <= 0 or not events:
        description["n_sampled"] = 0
        description["per_host"] = [0] * int(n_hosts)
        description["max_host_share"] = None
        description["shortfall"] = max(0, n_events)
        description["exhausted_pool"] = bool(n_events > 0)
        return [], description
    rng = np.random.default_rng(int(seed))
    events = [events[int(i)] for i in rng.permutation(len(events))]
    pools = {host: [] for host in range(int(n_hosts))}
    for interval, host in events:
        pools.setdefault(host, []).append(interval)
    host_order = [int(h) for h in rng.permutation(sorted(pools))]
    target = min(n_events, len(events))
    sampled = []
    while len(sampled) < target:
        progress = False
        for host in host_order:
            if len(sampled) >= target:
                break
            if pools.get(host):
                sampled.append((pools[host].pop(0), host))
                progress = True
        if not progress:
            break
    sampled.sort()
    per_host = [0] * int(n_hosts)
    for _, host in sampled:
        if 0 <= host < len(per_host):
            per_host[host] += 1
    description["n_sampled"] = len(sampled)
    description["per_host"] = per_host
    description["max_host_share"] = (float(max(per_host)) / len(sampled)
                                    if sampled else None)
    description["shortfall"] = int(max(0, n_events - len(sampled)))
    description["exhausted_pool"] = bool(len(events) < n_events)
    return sampled, description


# --------------------------------------------------------------------------
# inputs, model and the registered loss
# --------------------------------------------------------------------------
def load_stream(stream_dir):
    """Manifest + arrays + ``ReplayV3`` for a collected P23 stream."""
    from run_ftmoe_protocol020 import ReplayV3, load_v2_time_scale_p20

    stream_dir = Path(stream_dir)
    manifest_path = stream_dir / "manifest.json"
    stream_path = stream_dir / "stream.npz"
    if not manifest_path.is_file() or not stream_path.is_file():
        raise FileNotFoundError(
            "no collected protocol-023 stream at %s (need manifest.json and "
            "stream.npz); run run_ftmoe_protocol023_s2.py first" % stream_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if str(manifest.get("protocol")) != PROTOCOL:
        raise ValueError("stream %s is not a Protocol 023 artifact"
                         % stream_dir)
    if not manifest.get("phases"):
        raise ValueError("stream manifest carries no registered phase table; "
                         "the regime segmentation cannot be derived: %s"
                         % manifest_path)
    stream_sha256 = sha256_file(stream_path)
    declared = manifest.get("stream_sha256")
    if declared is not None and declared != stream_sha256:
        raise AssertionError("stream hash mismatch: %s != %s"
                             % (declared, stream_sha256))
    with np.load(stream_path) as data:
        arrays = {key: data[key] for key in data.files}
    steps = int(manifest["steps"])
    capacities = np.asarray(arrays["capacities"], dtype=np.float64)
    if capacities.shape != (steps + 1, 16, 3):
        raise ValueError("protocol-023 stream needs capacities_per_interval "
                         "of shape (%d, 16, 3), got %s"
                         % (steps + 1, (capacities.shape,)))
    arrays["capacities_per_interval"] = capacities
    arrays["capacities"] = capacities[0]
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu",
                            weights_only=False)
    v2_scale, fallback = load_v2_time_scale_p20(checkpoint["normalization"])
    graph_scale = np.asarray(checkpoint["normalization"]["graph_scale"],
                            dtype=np.float64)
    replay = ReplayV3(arrays, v2_scale, graph_scale, steps)
    description = {
        "path": str(stream_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "stream_sha256": stream_sha256,
        "declared_stream_sha256": declared,
        "steps": steps,
        "seed": manifest.get("seed"),
        "mode": manifest.get("mode"),
        "registered": bool(manifest.get("registered", False)),
        "smoke": bool(manifest.get("smoke", False)),
        "label_tolerance_intervals": 1,
        "graph_scale": [float(value) for value in graph_scale],
        "time_scale_v2": [float(value) for value in np.asarray(v2_scale).ravel()],
    }
    return manifest, arrays, replay, v2_scale, fallback, description


def registered_class_balance():
    """The registered S7/R1 class weights from the P20 training bundle only."""
    from recovery.PreGANSrc.src.ftmoe_online_s7 import balanced_weights

    manifest_path = P20_ADAPTATION / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "protocol-020 adaptation bundle missing (%s): the registered "
            "class-balanced weights are computed on that training bundle only "
            "and may never be estimated from the evaluation stream"
            % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    labels = []
    for index in manifest["train_indices"]:
        with np.load(P20_ADAPTATION / "episodes" / ("%02d.npz" % index)) as data:
            labels.append(data["labels"].astype(np.int64))
    balance = balanced_weights(np.concatenate(labels, axis=0))
    balance["source"] = ("artifacts/ftmoe_online/protocol_020/adaptation_data"
                         "/v1 train episodes (same-domain training bundle)")
    return balance


def build_model(learner_state=None, checkpoint_path=CHECKPOINT):
    """The registered fixed-C model (frozen v4 base + fixed residual bank)."""
    from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu",
                            weights_only=False)
    if checkpoint.get("variant") != "v4":
        raise AssertionError("frozen checkpoint is not a v4 model")
    if checkpoint.get("graph_semantics_version") != 3:
        raise ValueError("fixed-C requires graph_semantics_version 3")
    model = FrozenResidualFTMoE(checkpoint, "C", int(checkpoint["seed"]))
    loaded = None
    if learner_state is not None:
        loaded = load_learner_state(model, learner_state)
    # R1's own train() contract: the callers cannot put a base module into
    # train mode, and the residual has no dropout, so the forward pass stays
    # deterministic (ftmoe_online_r1.py:175-181).
    model.train()
    model.set_trainability()
    return model, checkpoint, loaded


def load_learner_state(model, path):
    """Load ONLY the ``learner.*`` tensors of a saved fixed-C state.

    Accepts a raw state dict, a checkpoint dict with ``model``, or a session
    save (``{"session": ...}``, recursively).  Anything that is not a
    ``learner.*`` tensor of the residual bank is refused, so a mature learner
    can be examined without ever touching the frozen base.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    candidate = payload
    for _ in range(3):
        if not isinstance(candidate, dict):
            break
        if any(str(key).startswith("learner.") for key in candidate):
            break
        for key in ("model", "session", "state_dict"):
            if isinstance(candidate.get(key), dict):
                candidate = candidate[key]
                break
        else:
            break
    if not isinstance(candidate, dict):
        raise ValueError("no learner state found in %s" % path)
    target = dict(model.learner.state_dict())
    selected = {}
    for key, value in candidate.items():
        name = str(key)
        if name.startswith("learner."):
            name = name[len("learner."):]
        if name in target:
            selected[name] = value
    missing = sorted(set(target) - set(selected))
    if missing:
        raise ValueError("learner state %s is incomplete; missing %s"
                         % (path, missing[:4]))
    model.learner.load_state_dict(selected, strict=True)
    model.set_trainability()
    return {"path": str(path), "sha256": sha256_file(path),
            "is_registered_start_state": False,
            "n_tensors": len(selected),
            "learner_sha256": digest_of(
                {name: tensor_sha256(value) for name, value
                 in model.learner.state_dict().items()})}


def learner_layout(model):
    """Online-trainable residual parameters and the registered groups.

    The parameter set is read from the model (``requires_grad``), never guessed:
    ``FrozenResidualFTMoE.set_trainability`` (ftmoe_online_r1.py:183-186) marks
    exactly the ``learner.*`` tensors, i.e. the residual bank returned by
    ``correction_parameters`` (ftmoe_online_r1.py:198-200).  The base/frozen
    parameters stay in the forward pass but are excluded from every group.
    """
    names = [name for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters()
              if not parameter.requires_grad]
    unexpected = [name for name in names if not name.startswith("learner.")]
    banks = {}
    for bank in ("learner", "live", "assessment", "previous"):
        prefix = bank + "."
        banks[bank] = {
            "n_tensors": sum(1 for name, _ in model.named_parameters()
                             if name.startswith(prefix)),
            "n_trainable_tensors": sum(1 for name in names
                                       if name.startswith(prefix)),
        }
    sizes = dict(model.named_parameters())
    offsets, cursor = {}, 0
    for name in names:
        width = int(sizes[name].numel())
        offsets[name] = (cursor, cursor + width)
        cursor += width
    groups = {"all": list(names),
              "router": [n for n in names if n.startswith("learner.router.")],
              "experts": [n for n in names if n.startswith("learner.experts.")]}
    for name in names:
        module = name.rsplit(".", 1)[0]
        groups.setdefault("layer:" + module, []).append(name)
    for index in range(EXPERT_COUNT):
        prefix = "learner.experts.%d." % index
        groups["expert%d" % index] = [n for n in names if n.startswith(prefix)]
    for group, members in groups.items():
        if not members:
            raise AssertionError("registered group %s is empty" % group)
    return {"names": names, "offsets": offsets, "width": cursor,
            "groups": groups, "frozen_names": frozen,
            "unexpected_trainable": unexpected, "banks": banks,
            "n_frozen_parameters": sum(int(sizes[name].numel())
                                       for name in frozen),
            "total_parameters": sum(int(p.numel()) for p in model.parameters())}


def event_ranking_term(detection, classification, target, row):
    """The registered pairwise ranking term restricted to one host-step.

    ``s7_loss`` computes ``softplus(0.15 + negative_score - positive_score)``
    over every (negative, positive) row pair of the batch
    (ftmoe_online_s7.py:87-98).  Averaging this function over all rows of one
    window reproduces that term exactly; per host-step it is the sub-average of
    the registered pairs that involve that host-step, which is what makes a
    single-row event gradient well defined at all.
    """
    anomaly_probability = detection.softmax(-1)[..., 1]
    class_probability = classification.softmax(-1)
    positive = target > 0
    if bool(positive[row]):
        negatives = ~positive
        if not bool(negatives.any()):
            return detection.new_zeros(())
        index = target[row].clamp_min(1) - 1
        positive_score = anomaly_probability[row] * class_probability[row][index]
        negative_score = (anomaly_probability[negatives]
                          * class_probability[negatives].max(-1).values)
        return F.softplus(0.15 + negative_score - positive_score).mean()
    if not bool(positive.any()):
        return detection.new_zeros(())
    negative_score = (anomaly_probability[row]
                      * class_probability[row].max(-1).values)
    index = target[positive].clamp_min(1) - 1
    positive_score = (anomaly_probability[positive]
                      * class_probability[positive].gather(
                          -1, index.unsqueeze(-1)).squeeze(-1))
    return F.softplus(0.15 + negative_score - positive_score).mean()


def event_loss(model, output, target, row, class_balance,
               weights=LOSS_WEIGHTS):
    """The registered online supervised loss for one host-step event.

    ``loss_online`` of the R1/S7 update (``ftmoe_online_s7.py:68-102`` with the
    R1 registration ``.7 detection CE + .3 positive classification CE
    + .5 joint ranking + .01 balance`` and the registered class-balanced
    weights ``balanced_weights`` computed on the training bundle only).  The
    detection/classification terms are evaluated on the event's own host row and
    the ranking term on the registered pairs that involve that row, so a single
    host-step -- the registered maturity unit -- has a well-defined gradient.
    The anchor and distillation protection terms of the S7 update are
    deliberately NOT included: they are drawn from the same-domain anchor pool
    and are therefore regime-independent by construction, so they would only
    dilute the regime signal this gate measures.  They are also not part of
    ``loss_online``.
    """
    detection = output["detection_logits"][0]
    classification = output["class_logits"][0]
    labels = torch.as_tensor(target, dtype=torch.long)
    anomaly = (labels > 0).long()
    detection_weights = torch.as_tensor(
        class_balance.get("detection_weight", [1.0, 1.0]),
        dtype=detection.dtype)
    resource_weights = torch.as_tensor(
        class_balance.get("resource_weight", [1.0, 1.0, 1.0]),
        dtype=classification.dtype)
    detection_ce = F.cross_entropy(detection[row:row + 1],
                                   anomaly[row:row + 1],
                                   weight=detection_weights)
    if bool(labels[row] > 0):
        classification_ce = F.cross_entropy(
            classification[row:row + 1], labels[row:row + 1] - 1,
            weight=resource_weights)
    else:
        classification_ce = detection.new_zeros(())
    ranking = event_ranking_term(detection, classification, labels, row)
    balance = model.auxiliary_losses(output)[1]
    return (weights["detection"] * detection_ce
            + weights["classification"] * classification_ce
            + weights["ranking"] * ranking
            + weights["balance"] * balance)


def measure_events(model, replay, layout, labels, events, class_balance):
    """Per-event residual gradients as one flat vector per event."""
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    if len(parameters) != len(layout["names"]):
        raise AssertionError("trainable parameter set changed under the probe")
    rows, losses, unused, norms = [], [], {}, []
    for interval, host in events:
        row_labels = np.asarray(labels[int(interval)])
        if (row_labels < 0).any():
            # The registered ranking term compares a host-step against the
            # window's other hosts, so an unrevealed host in the window cannot
            # be silently treated as a normal one (the frozen session refuses
            # exactly this: "R1 maturity requires a fully revealed label").
            raise RuntimeError(
                "window %d still has unrevealed labels (%d host(s)); the "
                "instrument never treats an unobserved label as normal"
                % (interval, int((row_labels < 0).sum())))
        host_window, schedule, graph, ids, before, caps = replay.window_v3(
            int(interval))
        context = {"creation_ids": ids.unsqueeze(0),
                   "before_placement": before.unsqueeze(0),
                   "capacities": caps.unsqueeze(0)}
        output = model(host_window.unsqueeze(0), schedule.unsqueeze(0),
                       graph.unsqueeze(0), graph_context=context)
        loss = event_loss(model, output, labels[int(interval)], int(host),
                          class_balance)
        if not bool(torch.isfinite(loss)):
            raise NonFiniteGradientError(
                "non-finite loss at event (interval %d, host %d) for regime "
                "event %r" % (interval, host, (interval, host)))
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True,
                                        retain_graph=False)
        flat = np.empty(layout["width"], dtype=np.float64)
        for name, gradient in zip(layout["names"], gradients):
            start, end = layout["offsets"][name]
            if gradient is None:
                unused[name] = unused.get(name, 0) + 1
                flat[start:end] = 0.0
                continue
            piece = assert_finite(
                gradient.detach().reshape(-1).numpy(),
                "gradient of %s at event (interval %d, host %d)"
                % (name, interval, host))
            flat[start:end] = piece
        assert_finite(flat, "event gradient at (interval %d, host %d)"
                      % (interval, host))
        rows.append(flat)
        losses.append(float(loss.detach()))
        norms.append(float(np.linalg.norm(flat)))
        del output, loss, gradients
    matrix = (np.stack(rows) if rows
              else np.zeros((0, layout["width"]), dtype=np.float64))
    return {"matrix": matrix, "losses": losses, "gradient_norms": norms,
            "unused_parameters": unused}


def group_matrix(matrix, layout, group):
    columns = []
    for name in layout["groups"][group]:
        start, end = layout["offsets"][name]
        columns.append(matrix[:, start:end])
    return np.concatenate(columns, axis=1) if columns else matrix[:, :0]


def analyse_group(per_regime, layout, group, seed, reps=BOOTSTRAP_REPS):
    """Mean-gradient cosines + event-pair statistics of one parameter group."""
    matrices = {regime: group_matrix(per_regime[regime]["matrix"], layout,
                                     group)
                for regime in REGIME_IDS}
    means, pair_means, pair_fractions, pairs, degenerate = {}, {}, {}, [], []
    for regime in REGIME_IDS:
        matrix = matrices[regime]
        if matrix.shape[0] == 0:
            means[regime] = None
            degenerate.append("%s: no events sampled" % regime)
            continue
        means[regime] = mean_gradient(matrix, "%s gradient of %s"
                                      % (regime, group))
    for left, right in regime_pairs():
        name = pair_name(left, right)
        if means[left] is None or means[right] is None:
            value = None
        else:
            value = cosine(means[left], means[right])
        pair_means[name] = value
        if value is None and means[left] is not None and means[right] is not None:
            degenerate.append("%s: zero-norm mean gradient" % name)
        entry = pair_statistics(matrices[left], matrices[right], group, name,
                                seed, reps=reps)
        pair_fractions[name] = entry["negative_fraction"]
        pairs.append(entry)
    norms = {}
    for regime in REGIME_IDS:
        matrix = matrices[regime]
        norms[regime] = ([float(value) for value in
                          np.linalg.norm(matrix, axis=1)] if matrix.shape[0]
                         else [])
    return {"group": group,
            "n_parameters": int(sum(layout["offsets"][n][1]
                                    - layout["offsets"][n][0]
                                    for n in layout["groups"][group])),
            "n_tensors": len(layout["groups"][group]),
            "tensors": list(layout["groups"][group]),
            "regime_mean_norms": {regime: (float(np.linalg.norm(means[regime]))
                                           if means[regime] is not None
                                           else None)
                                  for regime in REGIME_IDS},
            "event_gradient_norms": norms,
            "pair_mean_cosine": pair_means,
            "pair_negative_fraction": pair_fractions,
            "pairs": pairs,
            "degenerate": degenerate,
            "verdict": gate_verdict(pair_means, pair_fractions, group)}


def check_ram(guard_gib, stage, samples):
    """The registered RAM guard: ABORT, never thrash."""
    import psutil
    available = psutil.virtual_memory().available / 2 ** 30
    rss = psutil.Process().memory_info().rss / 2 ** 30
    samples.append({"stage": stage, "available_gib": round(available, 3),
                    "process_rss_gib": round(rss, 3)})
    if available < float(guard_gib):
        raise RuntimeError(
            "RAM guard ABORT at %s: only %.3f GiB available, below the "
            "registered %.2f GiB guard (FTMOE023_RAM_GUARD_GIB).  The probe "
            "aborts instead of thrashing; free memory and re-run."
            % (stage, available, float(guard_gib)))
    return available


def configure_runtime():
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(INTEROP_THREADS)
    except RuntimeError:
        pass
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:                                   # pragma: no cover
        pass


def jsonable(value):
    """Strict-JSON conversion: a non-finite float is a broken measurement."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise NonFiniteGradientError(
                "refusing to write a non-finite value (%r) into the report"
                % value)
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n",
                         encoding="utf8")
    temporary.replace(path)
    return path


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------
def run_probe(stream_dir=STREAM_DIR, checkpoint_path=CHECKPOINT, out_path=None,
              n_events=DEFAULT_EVENTS, seed=DEFAULT_SEED, label_scope="fault",
              learner_state=None, class_balance=None, registered_timeline=True,
              bootstrap_reps=BOOTSTRAP_REPS, ram_guard_gib=RAM_GUARD_GIB,
              write=True, command_line=None, print_summary=False):
    """Run the registered §14 instrument and return its report."""
    started = time.perf_counter()
    ram_samples = []
    configure_runtime()
    check_ram(ram_guard_gib, "start", ram_samples)

    checkpoint_sha = sha256_file(checkpoint_path)
    if Path(checkpoint_path) == CHECKPOINT and checkpoint_sha != CHECKPOINT_SHA256:
        raise AssertionError(
            "the registered frozen checkpoint changed: %s != %s (it is "
            "read-only and must never be retrained or overwritten)"
            % (checkpoint_sha, CHECKPOINT_SHA256))

    manifest, arrays, replay, v2_scale, fallback, stream_info = load_stream(
        stream_dir)
    steps = int(stream_info["steps"])
    n_hosts = int(np.asarray(arrays["raw_labels"]).shape[1])
    phases = normalize_phases(manifest["phases"])
    timeline = registered_timeline_check(phases, steps)
    if registered_timeline and not timeline["matches_registered_timeline"]:
        raise ValueError(
            "stream is not the registered development timeline (plan §10): %s"
            % json.dumps(timeline["observed_phases"]))
    masks, phase_windows = regime_intervals(phases, steps)

    labels, mature = maturity_and_labels(arrays["raw_labels"], steps)
    candidates = candidate_events(labels, mature, masks, label_scope)
    if class_balance is None:
        class_balance = registered_class_balance()

    model, checkpoint, learner_info = build_model(learner_state,
                                                  checkpoint_path)
    layout = learner_layout(model)
    if layout["unexpected_trainable"]:
        raise AssertionError("non-residual parameters are trainable: %s"
                             % layout["unexpected_trainable"][:4])
    if layout["banks"]["learner"]["n_trainable_tensors"] != len(layout["names"]):
        raise AssertionError("the trainable set is not exactly the learner bank")
    hashes_before = state_hashes(model)
    digest_before = digest_of(hashes_before)
    frozen_before = model.frozen_hash()
    learner_hash_before = model.learner_state_hash()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("a parameter already carried .grad before the run")

    sampling, per_regime = {}, {}
    for regime in REGIME_IDS:
        check_ram(ram_guard_gib, "regime:%s" % regime, ram_samples)
        events, description = sample_events(candidates[regime], n_events,
                                            seed, n_hosts=n_hosts)
        description["label_scope"] = label_scope
        description["phase_windows"] = phase_windows[regime]
        description["events"] = [[int(t), int(h), int(labels[t, h])]
                                 for t, h in events]
        sampling[regime] = description
        per_regime[regime] = measure_events(model, replay, layout, labels,
                                            events, class_balance)

    hashes_after = state_hashes(model)
    digest_after = digest_of(hashes_after)
    frozen_after = model.frozen_hash()
    learner_hash_after = model.learner_state_hash()
    grad_written = [name for name, parameter in model.named_parameters()
                    if parameter.grad is not None]
    integrity = {
        "policy": ("no optimizer is constructed, no backward pass writes .grad "
                   "and every parameter/buffer tensor must be bit-identical "
                   "after the probe"),
        "n_state_tensors": len(hashes_before),
        "state_sha256_before": digest_before,
        "state_sha256_after": digest_after,
        "unchanged": bool(digest_before == digest_after),
        "changed_tensors": sorted(
            name for name in set(hashes_before) | set(hashes_after)
            if hashes_before.get(name) != hashes_after.get(name)),
        "learner_sha256_before": learner_hash_before,
        "learner_sha256_after": learner_hash_after,
        "frozen_base_sha256_before": frozen_before,
        "frozen_base_sha256_after": frozen_after,
        "parameter_sha256_before": hashes_before,
        "parameter_sha256_after": hashes_after,
        "grad_attribute_written": grad_written,
        "optimizer_constructed": False,
        "backward_calls": 0,
        "gradient_calls": int(sum(sampling[r]["n_sampled"]
                                  for r in REGIME_IDS)),
    }
    if not integrity["unchanged"]:
        raise AssertionError(
            "the probe modified model state: %s" % integrity["changed_tensors"])
    if grad_written:
        raise AssertionError("the probe wrote .grad on %s" % grad_written[:4])

    groups = {}
    for group in layout["groups"]:
        groups[group] = analyse_group(per_regime, layout, group, seed,
                                      reps=bootstrap_reps)
    layer_groups = {name: block for name, block in groups.items()
                    if name.startswith("layer:")}
    primary = groups[PRIMARY_GROUP]
    verdict = dict(primary["verdict"])
    verdict["conflict_present_any_group"] = bool(
        any(block["verdict"]["conflict_present"] for block in groups.values()))
    verdict["groups_with_conflict"] = sorted(
        name for name, block in groups.items()
        if block["verdict"]["conflict_present"])
    verdict["gate_scope"] = (
        "the registered gate is evaluated on the '%s' group (all residual "
        "online-trainable parameters); the router/expert/per-layer verdicts are "
        "reported next to it" % PRIMARY_GROUP)
    n_events_per_regime = {regime: int(sampling[regime]["n_sampled"])
                           for regime in REGIME_IDS}
    pair_counts = {entry["pair"]: int(entry["n_pairs"])
                   for entry in primary["pairs"]}
    pair_counts_defined = {entry["pair"]: int(entry["n_pairs_defined"])
                           for entry in primary["pairs"]}
    verdict["n_events_per_regime"] = n_events_per_regime
    verdict["n_pairs"] = pair_counts
    verdict["n_pairs_defined"] = pair_counts_defined
    degenerate_groups = sorted(name for name, block in groups.items()
                               if block["degenerate"])
    degenerate_details = sorted(set(
        item for block in groups.values() for item in block["degenerate"]))
    if degenerate_groups:
        verdict["note"] = (
            "a group whose gradient is (partly) structurally zero at the loaded "
            "residual state has no cosine for that block: those pairs are "
            "reported with null cosines, counted under n_pairs_undefined and "
            "excluded from the verdict instead of being counted as 'no "
            "conflict'.  Degenerate groups: %s" % degenerate_groups)

    # ---- admissibility of the H3 verdict itself ---------------------------
    # The registered start state carries a freshly initialised residual bank
    # whose final read-out layer is zeroed (ftmoe_online_r1.py:80-84), so its
    # hidden-layer gradients are exactly zero and its read-out block is the only
    # place a cosine exists.  A verdict computed there describes the geometry of
    # an UNTRAINED bank, not the conflict the fixed learner would experience
    # once it has absorbed a regime.  The registered gate is therefore reported
    # as measured but marked inadmissible, so it can be quoted as a limitation
    # and never as the reason D was started or stopped.
    n_hidden_degenerate = len([name for name in degenerate_groups
                               if name.startswith("layer:")])
    learner_is_start_state = bool(
        (learner_info or {}).get("is_registered_start_state", True))
    admissible = not degenerate_groups and not learner_is_start_state
    verdict["admissibility"] = {
        "scientifically_admissible": bool(admissible),
        "learner_state_kind": ("registered_start_state"
                               if learner_is_start_state else "mature_learner"),
        "degenerate_groups": degenerate_groups,
        "n_degenerate_hidden_layers": int(n_hidden_degenerate),
        "rule": ("H3 may only be read as evidence about the fixed online "
                 "learner when the residual bank is mature: every parameter "
                 "group the registration names must carry a defined gradient. "
                 "At the registered start state the read-out block is zeroed by "
                 "construction, so the hidden-layer and router numbers describe "
                 "an untrained bank."),
        "what_would_make_it_admissible": (
            "run the registered gate on a mature fixed-C learner state via "
            "--learner-state <checkpoint holding learner.* tensors>; no such "
            "checkpoint exists in this round because D was out of scope and no "
            "fixed-C online run was performed"),
    }
    if not admissible:
        verdict["registered_gate_at_this_state"] = verdict["verdict"]
        verdict["verdict"] = "INADMISSIBLE"
        verdict["verdict_text"] = (
            "the registered §14 rule was evaluated on an untrained residual "
            "bank, so the result is reported as a measurement of the start "
            "state only: %s" % verdict["registered_gate_at_this_state"])

    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": STAGE,
        "kind": KIND,
        "gate": "H3",
        "registration": {
            "plan": PLAN,
            "plan_sha256": (sha256_file(ROOT / PLAN) if (ROOT / PLAN).is_file()
                            else None),
            "section": PLAN_SECTION,
            "protocol_json": str(PROTOCOL_JSON),
            "protocol_json_sha256": (sha256_file(PROTOCOL_JSON)
                                     if PROTOCOL_JSON.is_file() else None),
            "rule": ("conflict_present iff at least one regime pair has a mean "
                     "residual-gradient cosine <= -0.05, or at least one pair "
                     "has >= 30%% of its sampled cross-regime event pairs with "
                     "cosine < 0"),
            "pair_mean_cosine_max": PAIR_MEAN_COSINE_MAX,
            "negative_pair_fraction_min": NEGATIVE_PAIR_FRACTION_MIN,
            "primary_group": PRIMARY_GROUP,
            "regimes": list(REGIME_IDS),
            "phase_regime_map": dict(PHASE_REGIME),
            "group_definition": ("FrozenResidualFTMoE.set_trainability marks "
                                 "exactly the learner.* residual tensors "
                                 "(ftmoe_online_r1.py:183-186); "
                                 "correction_parameters returns the learner "
                                 "bank (ftmoe_online_r1.py:198-200)"),
            "base_parameters_in_forward_but_not_in_any_cosine": True,
        },
        "stream": stream_info,
        "normalization": {
            "artifact": ("artifacts/ftmoe_online/protocol_020/"
                         "normalization_v2_time_scale.json"),
            "time_scale_v2_shape": list(np.asarray(v2_scale).shape),
            "fallback_columns": list(fallback),
            "graph_scale": stream_info["graph_scale"],
            "note": ("the registered v2 time scale and the checkpoint's graph "
                     "scale, applied through run_ftmoe_protocol020.ReplayV3."
                     "window_v3 -- the same call the P20/P22 runners make"),
        },
        "timeline": timeline,
        "regime_phase_windows": phase_windows,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "registered_sha256": CHECKPOINT_SHA256,
            "matches_registration": bool(checkpoint_sha == CHECKPOINT_SHA256),
            "variant": checkpoint.get("variant"),
            "seed": int(checkpoint.get("seed", -1)),
            "epoch": checkpoint.get("epoch"),
            "access": "read-only; the file is never written and the frozen "
                      "base parameters are never updated",
        },
        "learner": {
            "initialization": ("loaded from --learner-state"
                               if learner_info else
                               "fresh R1 residual bank at construction "
                               "(the registered fixed-C start state)"),
            "loaded_state": learner_info,
            "n_online_trainable_tensors": len(layout["names"]),
            "n_online_trainable_parameters": layout["width"],
            "n_frozen_tensors": len(layout["frozen_names"]),
            "n_frozen_parameters": layout["n_frozen_parameters"],
            "n_model_parameters": layout["total_parameters"],
            "banks": layout["banks"],
            "unexpected_trainable": layout["unexpected_trainable"],
            "trainable_names": list(layout["names"]),
            "trainable_rule": "requires_grad (name.startswith('learner.'))",
            "model_training_flag": bool(model.training),
            "train_mode_call": ("the probe calls FrozenResidualFTMoE.train() "
                                "before measuring; R1's override forwards to "
                                "nn.Module.train(False) and re-forces every "
                                "base module into eval (ftmoe_online_r1.py:"
                                "175-181), so model.training stays False by "
                                "the model's own contract.  The residual has "
                                "no dropout or batch norm, so the forward pass "
                                "is identical either way and the override is "
                                "the registered online configuration"),
            "grad_enabled": bool(torch.is_grad_enabled()),
            "forward_determinism": ("every module runs the deterministic eval "
                                    "path; gradient computation cannot change "
                                    "the forward pass"),
        },
        "label_convention": {
            "maturity": ("a host-step (t, h) is mature once the physical raw "
                         "labels of t-1, t and t+1 have all been observed for "
                         "that host -- exactly the condition "
                         "run_ftmoe_online.tolerance_label enforces when "
                         "S7Session.step matures index t-1 at cursor t"),
            "tolerance": ("run_ftmoe_online.tolerance_label: the raw label "
                          "unless it is 0 and a neighbour is faulted; the "
                          "nearest anomaly wins and an equal-distance tie "
                          "keeps the earlier interval"),
            "function": "run_ftmoe_online.tolerance_label",
            "label_scope": label_scope,
            "n_mature_host_steps": int(mature.sum()),
            "n_fault_host_steps": int((labels > 0).sum()),
            "n_candidates_per_regime": {regime: len(candidates[regime])
                                        for regime in REGIME_IDS},
        },
        "loss_definition": {
            "function": "probe_ftmoe_protocol023_gradient.event_loss",
            "weights": LOSS_WEIGHTS,
            "class_weights": class_balance,
            "scope": ("registered loss_online of the R1/S7 update, evaluated "
                      "on the event's own host-step (detection CE + positive "
                      "classification CE) with the registered pairwise ranking "
                      "term over the pairs that involve that host-step and the "
                      "registered router balance term of the window"),
            "excluded_terms": ("the S7 anchor and distillation protection "
                               "terms (0.25 * loss_anchor + 0.10 * distill): "
                               "they are drawn from the same-domain anchor "
                               "pool, are regime-independent by construction "
                               "and are not part of loss_online"),
            "per_event_gradient": ("torch.autograd.grad of that scalar loss "
                                   "w.r.t. the online-trainable residual "
                                   "parameters; allow_unused=True so a "
                                   "structurally unused tensor is reported as "
                                   "zeros and counted, never silently dropped"),
        },
        "sampling": {
            "seed": int(seed),
            "events_per_regime_requested": int(n_events),
            "n_events_total_sampled": int(sum(sampling[r]["n_sampled"]
                                              for r in REGIME_IDS)),
            "per_regime": sampling,
            "host_balanced": True,
            "losses_per_regime": {regime: per_regime[regime]["losses"]
                                  for regime in REGIME_IDS},
            "gradient_norms_per_regime": {
                regime: per_regime[regime]["gradient_norms"]
                for regime in REGIME_IDS},
            "unused_parameters": {regime: per_regime[regime]
                                  ["unused_parameters"]
                                  for regime in REGIME_IDS},
        },
        "groups": groups,
        "per_layer": {
            "available": True,
            "granularity": "residual module (parameter name minus its last "
                           "component), i.e. the router and each expert's "
                           "LayerNorm / Linear(64->32) / Linear(32->5)",
            "n_groups": len(layer_groups),
            "groups": sorted(layer_groups),
            "note": ("the decomposition needs no change to the frozen model: "
                     "the residual bank is nn.Sequential + nn.ModuleList and "
                     "every tensor name encodes its module"),
        },
        "verdict": verdict,
        "n_events_per_regime": n_events_per_regime,
        "n_pairs": pair_counts,
        "seed": int(seed),
        "degenerate_groups": degenerate_groups,
        "degenerate_details": degenerate_details,
        "degeneracy_note": (
            "the registered R1 start bank zeroes its final layer "
            "(ftmoe_online_r1.py:80-84), so at the frozen start state the "
            "residual correction is exactly zero: the expert hidden layers "
            "(LayerNorm and Linear 64->32) receive a structurally zero "
            "gradient -- they are multiplied by the zero read-out matrix -- "
            "while the read-out layer carries the signal and the router "
            "receives only the registered balance term (weight 0.01).  A group "
            "with a zero-norm gradient has no cosine; it is reported as null "
            "and excluded from the verdict instead of being counted as 'no "
            "conflict'.  Re-run with --learner-state on a mature fixed-C "
            "learner to measure the hidden and router blocks as well"),
        "parameter_integrity": integrity,
        "resource_discipline": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
            "single_process": True,
            "ram_guard_gib": float(ram_guard_gib),
            "ram_samples": ram_samples,
        },
        "command_line": command_line if command_line is not None
        else " ".join(sys.argv),
        "elapsed_seconds": None,
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    if write:
        path = Path(out_path) if out_path is not None else (OUT_DIR / OUT_NAME)
        report["written"] = str(write_json(path, report))
    if print_summary:
        print(json.dumps(jsonable(summarize(report)), ensure_ascii=False,
                         allow_nan=False), flush=True)
    return report


def summarize(report):
    """The compact JSON summary printed by the CLI."""
    primary = report["groups"][PRIMARY_GROUP]
    return {
        "protocol": report["protocol"],
        "stage": report["stage"],
        "kind": report["kind"],
        "gate": report["gate"],
        "stream": Path(report["stream"]["path"]).name,
        "steps": report["stream"]["steps"],
        "events_per_regime": report["sampling"]["events_per_regime_requested"],
        "n_events_total": report["sampling"]["n_events_total_sampled"],
        "seed": report["sampling"]["seed"],
        "learner": report["learner"]["initialization"],
        "mean_cosine": primary["pair_mean_cosine"],
        "negative_fraction": primary["pair_negative_fraction"],
        "conflict_present": report["verdict"]["conflict_present"],
        "conflict_present_any_group": report["verdict"]
        ["conflict_present_any_group"],
        "verdict": report["verdict"]["verdict"],
        "params_unchanged": report["parameter_integrity"]["unchanged"],
        "degenerate_groups": report["degenerate_groups"],
        "written": report.get("written"),
        "elapsed_seconds": round(float(report["elapsed_seconds"]), 1),
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream", type=Path, default=STREAM_DIR)
    p.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT)
    p.add_argument("--out", type=Path, default=OUT_DIR / OUT_NAME)
    p.add_argument("--events", type=int, default=DEFAULT_EVENTS,
                   help="mature events sampled per regime (default %d)"
                        % DEFAULT_EVENTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--label-scope", choices=("fault", "all"), default="fault")
    p.add_argument("--learner-state", type=Path, default=None,
                   help="optional mature fixed-C learner state (learner.* only)")
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--ram-guard-gib", type=float, default=RAM_GUARD_GIB)
    p.add_argument("--allow-unregistered-timeline", action="store_true",
                   help="test-only: accept a stream whose phases are not the "
                        "registered development timeline (plan §10)")
    p.add_argument("--no-write", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    report = run_probe(
        stream_dir=args.stream, checkpoint_path=args.checkpoint_path,
        out_path=args.out, n_events=args.events, seed=args.seed,
        label_scope=args.label_scope, learner_state=args.learner_state,
        registered_timeline=not args.allow_unregistered_timeline,
        bootstrap_reps=args.bootstrap_reps, ram_guard_gib=args.ram_guard_gib,
        write=not args.no_write, print_summary=True)
    return report


if __name__ == "__main__":
    main()
