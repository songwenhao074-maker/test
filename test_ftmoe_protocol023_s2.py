"""Protocol 023 S2 — analyzer tests (T023-S2-01..08).

These tests pin ``analyze_ftmoe_protocol023_s2`` -- the plan §9 Data Gate and
marginal-matching analyzer -- against hand-built streams written to a temporary
directory under ``artifacts/ftmoe_online/protocol_023/_selftest/`` and removed
when the module finishes.  Nothing is collected (no simulator, no torch, single
process, numpy only) and nothing is left behind.

    T023-S2-01  the gate thresholds are exactly the plan §9 numbers, they agree
                with the instrument's registered table, and a DRIFTED threshold
                cannot pass: the analyzer's own re-derivation disagrees with the
                instrument and the verdict fails
    T023-S2-02  the operator semantics of each registered comparison (a value
                exactly on the boundary)
    T023-S2-03  the onset scan uses the regime's OWN task cohort: on a mixed
                timeline where one regime-A task crosses the A, B and C
                thresholds, the analyzer attributes the event to A only and
                reports the unmapped hits as the cross-regime confound
    T023-S2-04  recomputed prevalence / fault runs / rejections / worst-event
                share / follow-up validity match hand-computed values
    T023-S2-05  an inadmissible onset threshold is REPORTED (with its negative
                margin) and is never silently retuned or accepted
    T023-S2-06  a full synthetic gate root passes the registered gate and writes
                every artifact (regime_A/B/C.json, marginal_match.json,
                DATA_GATE.md) before the exit code is decided
    T023-S2-07  a missing stream directory is a clear error (exit 2, no
                zero-filled PASS); ``--only`` still analyses one regime
    T023-S2-08  the collector's declared audit numbers are never inputs: a lying
                ``unseen_data_audit.json`` is reported as a disagreement while
                the recomputed numbers stand; and the analysis is deterministic

Run from the repository root:
    python -m unittest test_ftmoe_protocol023_s2 -v
"""
import json
import shutil
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

import analyze_ftmoe_protocol023_s2 as analyzer
import ftmoe_protocol023_core as core23

ROOT = Path(__file__).resolve().parent
SELFTEST_ROOT = ROOT / "artifacts/ftmoe_online/protocol_023/_selftest"

#: plan §9 "建议 Data Gate", transcribed independently of the analyzer so a
#: drifted threshold in either place fails a test.
PLAN_SECTION_9_THRESHOLDS = {
    "prevalence_min": 0.03,
    "prevalence_max": 0.12,
    "deployment_rejection_max": 0.25,
    "migration_rejection_max": 0.40,
    "independent_fault_events_min": 80,
    "valid_onset_followup_min": 50,
    "worst_event_share_max": 0.10,
    "prevalence_spread_max": 0.04,
}
#: The generator's registered thresholds and burst floors (plan §5-§7).
REGISTERED_TAU = {"cpu": 2600.0, "ram": 2810.0, "disk": 11600.0}
REGISTERED_FLOOR = {"cpu": 4400.0, "ram": 4500.0, "disk": 16000.0}
PROVISIONAL_RESOURCES = ("ram", "disk")

FAMILIAR = dict(core23.FAMILIAR_CLIP)          # cpu 1860, ram 1400, disk 9000
WINDOW_BIT = {"cpu": 1, "ram": 2, "disk": 4}
#: Response levels used by the matched synthetic regimes: the same three levels
#: in every regime, only their order and lag differ (plan §9's matched marginals
#: with a different joint structure).
LEVEL = {"cpu": 6250.0, "ram": 4700.0, "disk": 18000.0}
#: Registered onset levels (each strictly above its own tau and below the floor).
TRIGGER = {"cpu": 4700.0, "ram": 4800.0, "disk": 18000.0}


def setUpModule():
    SELFTEST_ROOT.mkdir(parents=True, exist_ok=True)


def tearDownModule():
    if SELFTEST_ROOT.exists():
        shutil.rmtree(SELFTEST_ROOT, ignore_errors=True)


# --------------------------------------------------------------------------
# synthetic stream builder (minimal but coherent P23 stream layout)
# --------------------------------------------------------------------------
def flat(resource, length, level=None):
    return np.full(int(length),
                   FAMILIAR[resource] if level is None else float(level),
                   dtype=np.float64)


def task(creation, slot, host, start, cpu, ram, disk, event_id=-1,
         mechanism_id=-1, windows=None):
    """One synthetic task trajectory (with its registered envelope, if any)."""
    cpu = np.asarray(cpu, dtype=np.float64)
    ram = np.asarray(ram, dtype=np.float64)
    disk = np.asarray(disk, dtype=np.float64)
    if not (cpu.size == ram.size == disk.size):
        raise AssertionError("a task's three resource series must be parallel")
    mask = np.zeros(cpu.size, dtype=np.int64)
    if event_id >= 0 and windows:
        for name, (first, last) in windows.items():
            mask[int(first):min(int(last) + 1, cpu.size)] |= WINDOW_BIT[name]
    return {"creation": int(creation), "slot": int(slot), "host": int(host),
            "start": int(start), "cpu": cpu, "ram": ram, "disk": disk,
            "event_id": int(event_id), "mechanism_id": int(mechanism_id),
            "mask": mask}


def build_stream(directory, steps, phases, tasks, n_hosts=4, n_slots=16,
                 labels=None, deploy_attempts=None, deploy_rejected=None,
                 migrate_attempts=None, migrate_rejected=None, mode="single",
                 regime=None, audit=None):
    """Write stream.npz / task_timeline.npz / manifest.json (+ a fake audit)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    steps = int(steps)
    rows = steps + 1
    phase_ids = np.zeros(rows, dtype=np.int64)
    phase_kind = np.zeros(rows, dtype=np.int64)
    phase_probability = np.zeros(rows, dtype=np.float64)
    phase_regime = np.full(rows, -1, dtype=np.int64)
    cursor = 0
    for index, (name, length, gen_id) in enumerate(phases):
        for t in range(cursor, min(cursor + int(length), rows)):
            phase_ids[t] = index
            phase_kind[t] = 0 if gen_id is None else 1
            phase_probability[t] = 0.0 if gen_id is None else 0.25
            phase_regime[t] = (-1 if gen_id is None
                               else analyzer.REGISTERED_MECHANISM_IDS[gen_id])
        cursor += int(length)
    if cursor != steps:
        raise AssertionError("phase lengths sum to %d, expected %d"
                             % (cursor, steps))

    creation = np.full((rows, n_slots), -1, dtype=np.int64)
    demands = np.zeros((rows, n_slots, 7), dtype=np.float64)
    slot_host = np.full((rows, n_slots), -1, dtype=np.int64)
    cascade_ids = np.full((rows, n_slots), -1, dtype=np.int64)
    cascade_phases = np.zeros((rows, n_slots), dtype=np.int64)
    cascade_regimes = np.full((rows, n_slots), -1, dtype=np.int64)
    host_any = np.zeros((rows, n_hosts), dtype=np.int64)
    host_event = np.full((rows, n_hosts), -1, dtype=np.int64)
    host_regime = np.full((rows, n_hosts), -1, dtype=np.int64)
    host_mask = np.zeros((rows, n_hosts), dtype=np.int64)

    for spec in tasks:
        for age in range(int(spec["cpu"].size)):
            t = spec["start"] + age
            if t >= rows:
                break
            slot = spec["slot"]
            if creation[t, slot] != -1:
                raise AssertionError("slot %d is occupied twice at t=%d"
                                     % (slot, t))
            creation[t, slot] = spec["creation"]
            demands[t, slot] = [spec["cpu"][age], spec["ram"][age], 0.0, 0.0,
                                spec["disk"][age], 0.0, 0.0]
            slot_host[t, slot] = spec["host"]
            if spec["event_id"] >= 0:
                cascade_ids[t, slot] = spec["event_id"]
                cascade_regimes[t, slot] = spec["mechanism_id"]
                cascade_phases[t, slot] = int(spec["mask"][age])
    for t in range(rows):
        for slot in range(n_slots):
            event = int(cascade_ids[t, slot])
            if event < 0 or int(cascade_phases[t, slot]) == 0:
                continue
            host = int(slot_host[t, slot])
            if host < 0:
                continue
            host_any[t, host] = 1
            host_mask[t, host] |= int(cascade_phases[t, slot])
            if host_event[t, host] < 0 or event < host_event[t, host]:
                host_event[t, host] = event
                host_regime[t, host] = int(cascade_regimes[t, slot])

    if labels is None:
        labels = np.zeros((rows, n_hosts), dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (rows, n_hosts):
        raise AssertionError("labels must be [%d, %d], got %s"
                             % (rows, n_hosts, labels.shape))

    def counter(values):
        if values is None:
            return np.zeros(rows, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        if values.shape != (rows,):
            raise AssertionError("counter must be [%d], got %s"
                                 % (rows, values.shape))
        return values

    deploy_attempts = counter(deploy_attempts)
    deploy_rejected = counter(deploy_rejected)
    migrate_attempts = counter(migrate_attempts)
    migrate_rejected = counter(migrate_rejected)

    np.savez_compressed(
        directory / "stream.npz",
        raw_labels=labels, phase_ids=phase_ids, phase_kind=phase_kind,
        phase_cascade_probability=phase_probability,
        phase_regime_ids=phase_regime, creation_ids=creation,
        after_creation_ids=creation.copy(), cascade_event_ids=cascade_ids,
        cascade_regimes=cascade_regimes, cascade_phases=cascade_phases,
        cascade_task_flags=(cascade_ids >= 0).astype(np.int64),
        host_cascade_any=host_any, host_cascade_event=host_event,
        host_cascade_regime=host_regime, host_cascade_mask=host_mask,
        deploy_attempts=deploy_attempts, deploy_rejected=deploy_rejected,
        migrate_attempts=migrate_attempts, migrate_rejected=migrate_rejected)
    live = creation[:steps] >= 0
    t_index, s_index = np.nonzero(live)
    np.savez_compressed(
        directory / "task_timeline.npz",
        time=t_index, slot_index=s_index,
        creation_id=creation[t_index, s_index],
        demand=demands[t_index, s_index],
        host_id=slot_host[t_index, s_index],
        cascade_event_id=cascade_ids[t_index, s_index],
        cascade_phase=cascade_phases[t_index, s_index],
        cascade_regime=cascade_regimes[t_index, s_index])
    manifest = {
        "protocol": "023", "mode": mode, "regime": regime, "steps": steps,
        "synthetic": True,
        "phases": [{"name": name, "length": int(length),
                    "regime_id": gen_id} for name, length, gen_id in phases],
        "purpose": "analyzer selftest stream (never a formal stream)",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8")
    if audit is not None:
        (directory / "unseen_data_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf8")
    return directory


def envelope_task(creation, slot, host, start, gen_id, life=24, event_id=None,
                  windows=None):
    """A registered cascade task of ``gen_id`` at the registered lags."""
    spec = core23.regime(gen_id)
    onset = spec["onset_resource"]
    values = {name: flat(name, life) for name in core23.RESOURCES}
    values[onset][1:4] = TRIGGER[onset]
    masks = {onset: (1, 3)}
    for name, (first, _last) in spec["response_windows"].items():
        values[name][first:first + 10] = LEVEL[name]
        masks[name] = (first, first + 9)
    return task(creation, slot, host, start, values["cpu"], values["ram"],
                values["disk"], event_id=event_id,
                mechanism_id=analyzer.REGISTERED_MECHANISM_IDS[gen_id],
                windows=masks)


# --------------------------------------------------------------------------
# the hand-computed mixed stream (a single A task crossing A, B and C thresholds)
# --------------------------------------------------------------------------
TINY_STEPS = 30
TINY_PHASES = (("F0_baseline", 6, None), ("A1_compute", 6, "compute_first"),
               ("B1_memory", 6, "memory_first"), ("C1_io", 6, "io_first"),
               ("F1_baseline", 6, None))


def tiny_stream(directory):
    """30 intervals, 4 hosts, three A cascade tasks (the first one crossing all
    three registered thresholds) plus one B and one C task.

    Hand-computed expectations (see the tests):

        A window 6..11   labels 1 at (6,0) (7,0) (8,0) (10,2): 4 positives over
                         6*4 = 24 host-steps -> prevalence 4/24; two runs
                         (duration 3 and 1); the run at t=10 sits on host 2 where
                         the A task a2 (event 11) is live and masked, so the
                         worst cascade event share is 1/4 while the label-free
                         worst run share is 3/4
        B window 12..17  labels 1 at (12,3) (13,3): 2 positives, 1 run
        C window 18..23  no positive
        onsets           A: a1 (t=7, full 18-interval follow-up), a2 (t=9, the
                         response window is only partly observed) and a3 (t=11,
                         observed life 5: no usable follow-up at all);
                         B: b1 only (the A task a1 crosses the B RAM threshold
                         but is not in the B cohort); C: c1 only (a1 crosses the
                         C disk threshold but is not in the C cohort)
        follow-up        A has 3 events, 1 with a FULL window and 2 with a
                         usable (P22) one, so the two definitions are pinned to
                         different numbers
        familiar rows    c1's disk burst (18000) spills into the trailing
                         familiar phase, so the conservative familiar disk
                         maximum is 18000 while the envelope-free maximum is
                         9000
    """
    tasks = [
        task(1, 0, 0, 0, flat("cpu", TINY_STEPS), flat("ram", TINY_STEPS),
             flat("disk", TINY_STEPS)),
    ]
    envelope = {"cpu": (1, 3), "ram": (4, 13), "disk": (8, 17)}
    cpu = flat("cpu", 24)
    cpu[1:4] = TRIGGER["cpu"]
    ram = flat("ram", 24)
    ram[4:14] = LEVEL["ram"]
    disk = flat("disk", 24)
    disk[8:18] = LEVEL["disk"]
    tasks.append(task(2, 1, 1, 6, cpu, ram, disk, event_id=10, mechanism_id=0,
                      windows=envelope))
    cpu2 = flat("cpu", 12)
    cpu2[1:4] = TRIGGER["cpu"]
    ram2 = flat("ram", 12)
    ram2[4:12] = LEVEL["ram"]
    tasks.append(task(3, 2, 2, 8, cpu2, ram2, flat("disk", 12), event_id=11,
                      mechanism_id=0, windows=envelope))
    cpu3 = flat("cpu", 5)
    cpu3[1:4] = TRIGGER["cpu"]
    tasks.append(task(6, 5, 0, 10, cpu3, flat("ram", 5), flat("disk", 5),
                      event_id=14, mechanism_id=0, windows=envelope))
    ram_b = flat("ram", 12)
    ram_b[1:4] = TRIGGER["ram"]
    tasks.append(task(5, 4, 0, 12, flat("cpu", 12), ram_b, flat("disk", 12),
                      event_id=13, mechanism_id=1, windows={"ram": (1, 3)}))
    disk_c = flat("disk", 12)
    disk_c[1:9] = TRIGGER["disk"]
    tasks.append(task(4, 3, 3, 18, flat("cpu", 12), flat("ram", 12), disk_c,
                      event_id=12, mechanism_id=2, windows={"disk": (1, 8)}))
    labels = np.zeros((TINY_STEPS + 1, 4), dtype=np.int64)
    labels[6, 0] = labels[7, 0] = labels[8, 0] = 1
    labels[10, 2] = 1
    labels[12, 3] = labels[13, 3] = 1
    deploy_attempts = np.ones(TINY_STEPS + 1, dtype=np.int64)
    deploy_rejected = np.zeros(TINY_STEPS + 1, dtype=np.int64)
    deploy_rejected[6] = 1
    migrate_attempts = np.ones(TINY_STEPS + 1, dtype=np.int64)
    migrate_rejected = np.zeros(TINY_STEPS + 1, dtype=np.int64)
    migrate_rejected[12] = migrate_rejected[13] = migrate_rejected[14] = 1
    return build_stream(directory, TINY_STEPS, TINY_PHASES, tasks, n_hosts=4,
                        n_slots=6, labels=labels,
                        deploy_attempts=deploy_attempts,
                        deploy_rejected=deploy_rejected,
                        migrate_attempts=migrate_attempts,
                        migrate_rejected=migrate_rejected)


# --------------------------------------------------------------------------
# the full synthetic gate root (three single streams + one dev stream)
# --------------------------------------------------------------------------
SINGLE_FAMILIAR_STEPS = 60
SINGLE_REGIME_STEPS = 300
GATE_GROUPS = 7
GATE_GROUP_SIZE = 16
GATE_GROUP_GAP = 30
GATE_LIFE = 24
GATE_LABEL_OFFSETS = (2, 4, 6, 8)
#: full groups of 16 tasks, one isolated positive label per task on its own host
GATE_TASKS = GATE_GROUPS * GATE_GROUP_SIZE
GATE_POSITIVES = GATE_TASKS
GATE_HAND = {
    "prevalence": float(GATE_POSITIVES) / float(SINGLE_REGIME_STEPS * 4),
    "positives": GATE_POSITIVES,
    "host_steps": SINGLE_REGIME_STEPS * 4,
    "independent_fault_events": GATE_POSITIVES,
    "valid_onset_followup": GATE_TASKS,
    "event_count": GATE_TASKS,
    # each positive is attributed to the four events co-live on its host
    "worst_event_share": 4.0 / float(GATE_POSITIVES),
}


def single_stream(directory, gen_id, familiar_ram=1400.0, life=GATE_LIFE):
    """One registered single-regime stream with hand-computable gate numbers.

    ``SINGLE_FAMILIAR_STEPS`` familiar intervals (four flat familiar tasks at
    slots 16..19) followed by ``SINGLE_REGIME_STEPS`` intervals of the regime,
    holding ``GATE_TASKS`` registered cascade tasks in groups of 16 (one group
    every ``GATE_GROUP_GAP`` intervals so no two groups overlap on a host) and
    one isolated positive label per task on its own host.
    """
    steps = SINGLE_FAMILIAR_STEPS + SINGLE_REGIME_STEPS
    n_slots = 20
    tasks = []
    for index in range(4):
        # The first familiar task carries ``familiar_ram`` so a test can plant a
        # familiar RAM level above the provisional B threshold (2810.0).
        ram_level = familiar_ram if index == 0 else FAMILIAR["ram"]
        tasks.append(task(1000 + index, 16 + index, index, 0,
                          flat("cpu", SINGLE_FAMILIAR_STEPS),
                          flat("ram", SINGLE_FAMILIAR_STEPS, ram_level),
                          flat("disk", SINGLE_FAMILIAR_STEPS)))
    labels = np.zeros((steps + 1, 4), dtype=np.int64)
    for index in range(GATE_TASKS):
        group, slot = divmod(index, GATE_GROUP_SIZE)
        host = slot % 4
        start = SINGLE_FAMILIAR_STEPS + group * GATE_GROUP_GAP
        spec = envelope_task(2000 + index, slot, host, start, gen_id,
                             life=life, event_id=3000 + index)
        tasks.append(spec)
        labels[start + GATE_LABEL_OFFSETS[slot // 4], host] = 1
    deploy_attempts = np.ones(steps + 1, dtype=np.int64)
    deploy_rejected = np.zeros(steps + 1, dtype=np.int64)
    deploy_rejected[SINGLE_FAMILIAR_STEPS] = 1
    migrate_attempts = np.ones(steps + 1, dtype=np.int64)
    migrate_rejected = np.zeros(steps + 1, dtype=np.int64)
    migrate_rejected[SINGLE_FAMILIAR_STEPS + 140] = 1
    return build_stream(directory, steps,
                        (("F0_baseline", SINGLE_FAMILIAR_STEPS, None),
                         ("%s_only" % gen_id, SINGLE_REGIME_STEPS, gen_id)),
                        tasks, n_hosts=4, n_slots=n_slots, labels=labels,
                        deploy_attempts=deploy_attempts,
                        deploy_rejected=deploy_rejected,
                        migrate_attempts=migrate_attempts,
                        migrate_rejected=migrate_rejected,
                        mode="single", regime=gen_id)


def dev_stream(directory):
    """The multi-regime development context (familiar + A + B + C phases)."""
    steps = 30 + 60 + 60 + 60
    tasks = [task(1, 0, 0, 0, flat("cpu", steps), flat("ram", steps),
                  flat("disk", steps)),
             task(2, 1, 1, 0, flat("cpu", steps), flat("ram", steps),
                  flat("disk", steps))]
    labels = np.zeros((steps + 1, 4), dtype=np.int64)
    for index, (gen_id, phase_start) in enumerate(
            (("compute_first", 30), ("memory_first", 90), ("io_first", 150))):
        for k in range(4):
            start = phase_start + 2 + k * 12
            tasks.append(envelope_task(100 + index * 10 + k, 2 + k, 2 + (k % 2),
                                       start, gen_id, life=16,
                                       event_id=500 + index * 10 + k))
        labels[phase_start + 5, 0] = 1
        labels[phase_start + 9, 1] = 1
    return build_stream(directory, steps,
                        (("F0_baseline", 30, None), ("A1_compute", 60,
                                                     "compute_first"),
                         ("B1_memory", 60, "memory_first"),
                         ("C1_io", 60, "io_first")),
                        tasks, n_hosts=4, n_slots=8, labels=labels,
                        deploy_attempts=np.ones(steps + 1, dtype=np.int64),
                        migrate_attempts=np.ones(steps + 1, dtype=np.int64),
                        mode="dev", regime=None)


def build_gate_root(root, familiar_ram=1400.0, life=GATE_LIFE):
    """The three single streams + the development stream under one root."""
    root = Path(root)
    for gen_id in analyzer.REGISTERED_REGIME_IDS:
        single_stream(root / (analyzer.REGISTERED_SINGLE_TAG % gen_id), gen_id,
                      familiar_ram=(familiar_ram if gen_id == "memory_first"
                                    else 1400.0), life=life)
    dev_stream(root / analyzer.REGISTERED_DEV_TAG)
    return root


def case(name):
    return SELFTEST_ROOT / name


# --------------------------------------------------------------------------
# T023-S2-01/02 — the registered thresholds
# --------------------------------------------------------------------------
class ThresholdTests(unittest.TestCase):
    def test_gate_thresholds_are_exactly_the_plan_section_9_numbers(self):
        self.assertEqual(dict(analyzer.GATE_THRESHOLDS),
                         PLAN_SECTION_9_THRESHOLDS,
                         "the analyzer's gate table drifted from plan §9")
        self.assertIn("§9", analyzer.PLAN_SECTION)

    def test_thresholds_match_the_instruments_registered_table(self):
        drift = analyzer.threshold_drift_report()
        self.assertTrue(drift["passed"], drift["drift"])
        self.assertEqual(drift["drift"], [])
        self.assertEqual(dict(core23.MARGINAL_MATCH_THRESHOLDS),
                         PLAN_SECTION_9_THRESHOLDS,
                         "the instrument's registered gate table drifted from "
                         "plan §9")

    def test_a_drifted_threshold_is_reported(self):
        with unittest.mock.patch.dict(analyzer.GATE_THRESHOLDS,
                                      {"prevalence_min": 0.09}):
            drift = analyzer.threshold_drift_report()
        self.assertFalse(drift["passed"])
        self.assertEqual([item["key"] for item in drift["drift"]],
                         ["prevalence_min"])
        self.assertEqual(drift["drift"][0]["analyzer"], 0.09)
        self.assertEqual(drift["drift"][0]["instrument"], 0.03)

    def test_registered_taus_and_floors_are_the_generator_registration(self):
        generator = analyzer.generator_registration()
        self.assertTrue(generator["available"],
                        "the generator registration module must be importable "
                        "torch-free for the floors to be reported: %s"
                        % generator["error"])
        registration = analyzer.registered_table()
        self.assertTrue(registration["tau_table_matches_instrument"],
                        registration["tau_drift"])
        for gen_id in analyzer.REGISTERED_REGIME_IDS:
            canonical = core23.canonical_regime_id(gen_id)
            entry = registration["table"][canonical]
            resource = entry["onset_resource"]
            self.assertAlmostEqual(entry["onset_tau"],
                                   REGISTERED_TAU[resource], places=9)
            self.assertAlmostEqual(entry["registered_floor"],
                                   REGISTERED_FLOOR[resource], places=9)
            self.assertEqual(entry["provisional"],
                             resource in PROVISIONAL_RESOURCES)

    def test_boundary_values_use_the_registered_operators(self):
        stats = {"prevalence": 0.03, "independent_fault_events": 80,
                 "valid_onset_followup": 50, "deployment_rejection_rate": 0.25,
                 "migration_rejection_rate": 0.40, "worst_event_share": 0.10}
        checks, _spread = analyzer.plan_section_9_checks({"A": dict(stats)})
        self.assertTrue(checks["A_prevalence_in_range"])
        self.assertTrue(checks["A_independent_fault_events_enough"])
        self.assertTrue(checks["A_valid_onset_followup_enough"])
        self.assertTrue(checks["A_deployment_rejection_ok"])
        self.assertTrue(checks["A_migration_rejection_ok"])
        self.assertFalse(checks["A_worst_event_share_ok"],
                         "the worst event share threshold is strict (< 10%)")
        low = dict(stats, prevalence=0.029999)
        self.assertFalse(analyzer.plan_section_9_checks({"A": low})[0]
                         ["A_prevalence_in_range"])
        high = dict(stats, prevalence=0.120001)
        self.assertFalse(analyzer.plan_section_9_checks({"A": high})[0]
                         ["A_prevalence_in_range"])
        few = dict(stats, independent_fault_events=79)
        self.assertFalse(analyzer.plan_section_9_checks({"A": few})[0]
                         ["A_independent_fault_events_enough"])
        unmeasurable = dict(stats, migration_rejection_rate=None)
        self.assertFalse(analyzer.plan_section_9_checks({"A": unmeasurable})[0]
                         ["A_migration_rejection_ok"],
                         "an unmeasurable rate is not a passing rate")

    def test_prevalence_spread_threshold_is_four_points(self):
        base = {"independent_fault_events": 200, "valid_onset_followup": 140,
                "deployment_rejection_rate": 0.0, "migration_rejection_rate": 0.0,
                "worst_event_share": 0.0}
        at_limit = {"A": dict(base, prevalence=0.05),
                    "B": dict(base, prevalence=0.09),
                    "C": dict(base, prevalence=0.07)}
        checks, spread = analyzer.plan_section_9_checks(at_limit)
        self.assertAlmostEqual(spread, 0.04, places=12)
        self.assertTrue(checks["prevalence_spread_ok"])
        over = {"A": dict(base, prevalence=0.05),
                "B": dict(base, prevalence=0.090001),
                "C": dict(base, prevalence=0.07)}
        self.assertFalse(analyzer.plan_section_9_checks(over)[0]
                         ["prevalence_spread_ok"])


# --------------------------------------------------------------------------
# T023-S2-03 — the cohort discipline on a mixed timeline
# --------------------------------------------------------------------------
class CohortDisciplineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = case("cohort")
        cls.directory = tiny_stream(
            cls.root / (analyzer.REGISTERED_SINGLE_TAG % "compute_first"))
        cls.measured = analyzer.measure_stream(cls.directory)
        cls.verdict = analyzer.analyse(cls.root, only="compute_first")

    def test_the_a_task_crosses_all_three_registered_thresholds(self):
        """Pinned hazard: the synthetic A cascade is a B and C crossing too."""
        confound = self.measured["cross_regime_confound"]
        self.assertEqual(confound["A"]["onsets_without_task_map"], 3)
        self.assertEqual(confound["B"]["onsets_without_task_map"], 3,
                         "the A tasks' RAM responses cross the B threshold "
                         "(2810)")
        self.assertEqual(confound["C"]["onsets_without_task_map"], 2,
                         "a1's Disk response crosses the C threshold (11600)")

    def test_each_regime_is_attributed_only_its_own_task_cohort(self):
        per_regime = self.measured["per_regime"]
        self.assertEqual(per_regime["A"]["onsets"]["n_events"], 3)
        self.assertEqual(per_regime["B"]["onsets"]["n_events"], 1,
                         "only the B-registered task b1 may be a B event; the A "
                         "task a1 crossing the B RAM threshold must not be")
        self.assertEqual(per_regime["C"]["onsets"]["n_events"], 1)
        self.assertEqual(per_regime["A"]["cohort"]["n_tasks"], 3)
        self.assertEqual(per_regime["B"]["cohort"]["n_tasks"], 1)
        self.assertEqual(per_regime["C"]["cohort"]["n_tasks"], 1)

    def test_the_event_keys_name_the_right_tasks(self):
        per_regime = self.measured["per_regime"]
        self.assertEqual(sorted(per_regime["A"]["onsets"]["event_keys"]),
                         ["2:7", "3:9", "6:11"])
        self.assertEqual(per_regime["B"]["onsets"]["event_keys"], ["5:13"])
        self.assertEqual(per_regime["C"]["onsets"]["event_keys"], ["4:19"])

    def test_cross_hits_are_reported_and_never_counted(self):
        confound = self.measured["cross_regime_confound"]
        self.assertEqual(confound["A"]["cross_hits"], 0)
        self.assertEqual(confound["B"]["cross_hits"], 2)
        self.assertEqual(confound["C"]["cross_hits"], 1)
        self.assertEqual(confound["B"]["onsets_with_task_map"], 1)
        self.assertEqual(confound["B"]["onsets_without_task_map"], 3)
        self.assertIn("NEVER a gate input",
                      self.measured["per_regime"]["B"]
                      ["cross_regime_confound"]["note"])

    def test_the_end_to_end_path_keeps_the_cohort(self):
        verdict = self.verdict
        self.assertEqual(verdict["partial"], True)
        self.assertEqual(verdict["only"], "A")
        self.assertEqual(
            verdict["per_regime"]["A"]["onsets"]["n_events"], 3)
        self.assertEqual(verdict["per_regime"]["A"]["cohort"]["n_tasks"], 3)
        self.assertEqual(verdict["measurements"]["A"]["cross_regime_confound"]
                         ["B"]["cross_hits"], 2)

    def test_the_followup_summary_is_machine_readable(self):
        summary = self.verdict["followup"]["A"]
        self.assertEqual(summary["n_events"], 3)
        self.assertEqual(summary["n_valid_followup_full_window_gated"], 1)
        self.assertEqual(summary["n_usable_followup_p22_definition"], 2)
        self.assertEqual(summary["followup_horizon"], 18)
        self.assertEqual(summary["cohort_observed_life"]["max"], 24.0)
        self.assertEqual(
            summary["n_valid_followup_full_window_gated"],
            self.measured["per_regime"]["A"]["onsets"]["n_valid_followup"])
        self.assertEqual(
            summary["n_usable_followup_p22_definition"],
            self.measured["per_regime"]["A"]["onsets"]
            ["n_events_with_a_usable_followup"])


# --------------------------------------------------------------------------
# T023-S2-04 — hand-computed recomputation
# --------------------------------------------------------------------------
class HandComputedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = case("hand")
        cls.directory = tiny_stream(
            cls.root / (analyzer.REGISTERED_SINGLE_TAG % "compute_first"))
        cls.measured = analyzer.measure_stream(cls.directory)

    def stats(self, canonical):
        return self.measured["per_regime"][canonical]["gate_stats"]

    def test_a_window_prevalence_runs_and_rejections(self):
        stats = self.stats("A")
        self.assertEqual(stats["host_steps"], 6 * 4)
        self.assertEqual(stats["positive_hoststeps"], 4)
        self.assertAlmostEqual(stats["prevalence"], 4.0 / 24.0, places=12)
        self.assertEqual(stats["independent_fault_events"], 2)
        self.assertEqual(stats["deployment_attempts"], 6)
        self.assertEqual(stats["deployment_rejected"], 1)
        self.assertAlmostEqual(stats["deployment_rejection_rate"], 1.0 / 6.0,
                               places=12)
        self.assertEqual(stats["migration_attempts"], 6)
        self.assertEqual(stats["migration_rejected"], 0)
        self.assertAlmostEqual(stats["migration_rejection_rate"], 0.0,
                               places=12)

    def test_b_window_prevalence_and_rejections(self):
        stats = self.stats("B")
        self.assertEqual(stats["positive_hoststeps"], 2)
        self.assertAlmostEqual(stats["prevalence"], 2.0 / 24.0, places=12)
        self.assertEqual(stats["independent_fault_events"], 1)
        self.assertEqual(stats["migration_rejected"], 3)
        self.assertAlmostEqual(stats["migration_rejection_rate"], 0.5,
                               places=12)

    def test_c_window_is_empty_and_reports_null_free_numbers(self):
        stats = self.stats("C")
        self.assertEqual(stats["positive_hoststeps"], 0)
        self.assertAlmostEqual(stats["prevalence"], 0.0, places=12)
        self.assertEqual(stats["independent_fault_events"], 0)
        self.assertEqual(stats["worst_event_share"], 0.0)
        self.assertAlmostEqual(stats["deployment_rejection_rate"], 0.0,
                               places=12)

    def test_worst_event_share_is_the_cascade_attribution(self):
        block = self.measured["per_regime"]["A"]
        self.assertAlmostEqual(block["gate_stats"]["worst_event_share"], 0.25,
                               places=12)
        self.assertAlmostEqual(
            block["fault_runs"]["worst_run_share_of_positives"], 0.75,
            places=12)
        self.assertEqual(
            block["fault_runs"]["n_distinct_cascade_events_attributed"], 1,
            "only the run on the host carrying event 11 is attributed")

    def test_valid_followup_needs_the_whole_registered_horizon(self):
        block = self.measured["per_regime"]["A"]
        self.assertEqual(block["onsets"]["n_events"], 3)
        self.assertEqual(block["onsets"]["n_valid_followup"], 1)
        self.assertEqual(block["onsets"]["n_invalid_followup"], 2)
        self.assertEqual(block["onsets"]["followup_horizon"], 18)
        self.assertEqual(block["durations"]["n"], 3)
        self.assertEqual(block["peak_ratio"]["n"], 2,
                         "a3 (observed life 5) dies before any response window "
                         "interval, so it has no measurable peak ratio; a NaN "
                         "must not be read as a measurement")

    def test_the_usable_followup_is_reported_next_to_the_gated_one(self):
        onsets = self.measured["per_regime"]["A"]["onsets"]
        self.assertEqual(onsets["n_valid_followup"], 1)
        self.assertEqual(onsets["n_events_with_a_usable_followup"], 2,
                         "a2 observes part of its RAM response window (a usable "
                         "P22 follow-up) but not the whole registered horizon")
        self.assertIn("full window", onsets["gated_definition"])
        self.assertIn("NOT the gated number", onsets["usable_followup_rule"])

    def test_familiar_maxima_are_measured_twice(self):
        familiar = self.measured["familiar"]
        self.assertEqual(familiar["familiar_intervals"], 12)
        self.assertAlmostEqual(
            familiar["measured_familiar_task_maximum"]["disk"], 18000.0,
            places=9)
        self.assertAlmostEqual(
            familiar["envelope_free_familiar_task_maximum"]["disk"], 9000.0,
            places=9)
        self.assertAlmostEqual(
            familiar["measured_familiar_task_maximum"]["cpu"], FAMILIAR["cpu"],
            places=9)
        self.assertEqual(familiar["n_tasks_with_familiar_rows"], 3)
        self.assertEqual(familiar["n_tasks_envelope_free"], 1)

    def test_recomputation_ignores_the_collectors_declared_numbers(self):
        lying = {"per_regime": {"A": {"gate": {"metrics": {
            "prevalence": 0.99, "positive_hoststeps": 10 ** 6,
            "host_steps": 10 ** 6, "independent_fault_events": 9999,
            "deployment_rejection_rate": 0.99,
            "migration_rejection_rate": 0.99,
            "worst_event_share_of_positives": 0.99}},
            "task_level": {"n_onsets": 4242}}}}
        path = self.directory / "unseen_data_audit.json"
        path.write_text(json.dumps(lying), encoding="utf8")
        try:
            measured = analyzer.measure_stream(self.directory)
        finally:
            path.unlink()
        stats = measured["per_regime"]["A"]["gate_stats"]
        self.assertAlmostEqual(stats["prevalence"], 4.0 / 24.0, places=12)
        self.assertEqual(stats["independent_fault_events"], 2)
        self.assertEqual(measured["per_regime"]["A"]["onsets"]["n_events"], 3)


# --------------------------------------------------------------------------
# T023-S2-05 — threshold admissibility is reported, never retuned
# --------------------------------------------------------------------------
class AdmissibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = case("admissibility")
        cls.directory = tiny_stream(
            cls.root / (analyzer.REGISTERED_SINGLE_TAG % "io_first"))
        cls.verdict = analyzer.analyse(cls.root, only="io_first")

    def test_the_disk_threshold_is_inadmissible_on_this_stream(self):
        verdict = self.verdict
        self.assertEqual(verdict["inadmissible_thresholds"], ["C"])
        self.assertFalse(verdict["thresholds_admissible"])
        admission = verdict["admissibility"]["C"]
        self.assertIs(admission["threshold_admissible"], False)
        self.assertAlmostEqual(admission["measured_familiar_task_maximum"],
                               18000.0, places=9)
        self.assertAlmostEqual(admission["margin_above_measured_maximum"],
                               11600.0 - 18000.0, places=9)
        self.assertTrue(admission["admissible_against_envelope_free_tasks"],
                        "the envelope-free measurement shows the burst is a "
                        "cascade spill, not a familiar maximum")
        self.assertTrue(admission["provisional"],
                        "the disk threshold is provisional and must say so")

    def test_the_threshold_itself_is_not_touched(self):
        admission = self.verdict["admissibility"]["C"]
        self.assertAlmostEqual(admission["registered_onset_threshold"],
                               REGISTERED_TAU["disk"], places=9)
        self.assertAlmostEqual(admission["registered_floor"],
                               REGISTERED_FLOOR["disk"], places=9)
        self.assertAlmostEqual(admission["margin_below_floor"],
                               16000.0 - 11600.0, places=9)
        self.assertEqual(self.verdict["thresholds"]["prevalence_min"], 0.03,
                         "no gate threshold may be retuned by the analyzer")

    def test_inadmissibility_fails_the_overall_verdict(self):
        self.assertIn("threshold_admissibility", self.verdict["failed_on"])
        self.assertFalse(self.verdict["passed"])
        self.assertTrue(any("STOP-THRESHOLD" in item for item in
                            self.verdict["stop_conditions"]))

    def test_a_pass_through_the_gate_still_stops_on_an_inadmissible_tau(self):
        root = case("admissibility_pass")
        build_gate_root(root, familiar_ram=3000.0)
        verdict = analyzer.analyse(root)
        self.assertTrue(verdict["gate_passed"],
                        "the eight registered checks still pass")
        self.assertEqual(verdict["inadmissible_thresholds"], ["B"])
        self.assertEqual(verdict["failed_on"], ["threshold_admissibility"])
        self.assertFalse(verdict["passed"])
        self.assertAlmostEqual(
            verdict["admissibility"]["B"]["measured_familiar_task_maximum"],
            3000.0, places=9)


# --------------------------------------------------------------------------
# T023-S2-06 — the full synthetic root passes and writes every artifact
# --------------------------------------------------------------------------
class GatePassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = case("gate_pass")
        build_gate_root(cls.root)
        cls.audit_out = cls.root / "_data_audit"
        cls.verdict = analyzer.analyse(cls.root)
        cls.artifacts = analyzer.write_artifacts(cls.verdict, cls.audit_out)

    def test_the_registered_gate_passes_on_the_matched_synthetic_regimes(self):
        verdict = self.verdict
        self.assertTrue(verdict["gate_passed"],
                        json.dumps(verdict["gate_checks"], indent=2))
        self.assertTrue(all(verdict["gate_checks"].values()),
                        verdict["gate_checks"])
        self.assertTrue(verdict["analyzer_matches_the_instrument"]["passed"],
                        verdict["analyzer_matches_the_instrument"]["mismatched"])
        self.assertTrue(verdict["thresholds_admissible"])
        self.assertTrue(verdict["integrity"]["passed"],
                        [key for key, value in verdict["integrity"]["checks"]
                         .items() if not value])
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["failed_on"], [])
        self.assertEqual(verdict["stop_conditions"], [])
        self.assertEqual(verdict["warnings"], [])

    def test_recomputed_numbers_are_the_hand_computed_ones(self):
        for canonical in ("A", "B", "C"):
            stats = self.verdict["gate_stats"][canonical]
            self.assertEqual(stats["host_steps"], GATE_HAND["host_steps"])
            self.assertEqual(stats["positive_hoststeps"],
                             GATE_HAND["positives"])
            self.assertAlmostEqual(stats["prevalence"],
                                   GATE_HAND["prevalence"], places=12)
            self.assertEqual(stats["independent_fault_events"],
                             GATE_HAND["independent_fault_events"])
            self.assertEqual(stats["valid_onset_followup"],
                             GATE_HAND["valid_onset_followup"])
            self.assertEqual(stats["event_count"], GATE_HAND["event_count"])
            self.assertAlmostEqual(stats["worst_event_share"],
                                   GATE_HAND["worst_event_share"], places=12)

    def test_the_marginals_are_matched_while_the_joint_structure_differs(self):
        contrast = self.verdict["regime_contrast"]
        self.assertTrue(contrast["passed"],
                        {key: value for key, value in contrast["checks"].items()
                         if not value})
        for pair, block in contrast["pairs"].items():
            self.assertTrue(block["marginals_matched"], pair)
            self.assertTrue(block["allowed_joint_differences"]
                            ["joint_structure_differs"], pair)

    def test_every_artifact_is_written(self):
        for name in ("regime_A.json", "regime_B.json", "regime_C.json",
                     "marginal_match.json", "DATA_GATE.md"):
            self.assertTrue((self.audit_out / name).is_file(), name)
        document = json.loads((self.audit_out / "regime_A.json").read_text(
            encoding="utf8"))
        self.assertEqual(document["canonical_regime_id"], "A")
        self.assertEqual(document["regime_id"], "compute_first")
        self.assertTrue(document["passed"])
        marginal = json.loads((self.audit_out / "marginal_match.json").read_text(
            encoding="utf8"))
        self.assertTrue(marginal["marginal_match_report"]["passed"])
        self.assertEqual(sorted(marginal["marginal_match_report"]["metrics"]
                                ["regimes_reported"]), ["A", "B", "C"])
        markdown = (self.audit_out / "DATA_GATE.md").read_text(encoding="utf8")
        self.assertIn("overall: PASS", markdown)
        self.assertIn("PROVISIONAL", markdown)
        self.assertIn("admissible", markdown.lower())

    def test_a_drifted_threshold_cannot_pass_the_verdict(self):
        # 0.095 sits just above the measured 0.0933, so the drifted analyzer
        # table flips a check that the instrument (3%) still passes.
        with unittest.mock.patch.dict(analyzer.GATE_THRESHOLDS,
                                      {"prevalence_min": 0.095}):
            verdict = analyzer.analyse(self.root)
        self.assertFalse(verdict["gate_passed"])
        self.assertFalse(verdict["analyzer_matches_the_instrument"]["passed"])
        self.assertEqual(
            sorted(item["check"] for item in
                   verdict["analyzer_matches_the_instrument"]["mismatched"]),
            ["A_prevalence_in_range", "B_prevalence_in_range",
             "C_prevalence_in_range"],
            "the three regimes carry the same measured prevalence, so the "
            "drifted floor flips all three checks")
        self.assertFalse(
            verdict["analyzer_checks"]["A_prevalence_in_range"])
        self.assertTrue(verdict["gate_checks"]["A_prevalence_in_range"],
                        "the instrument still uses the registered 3% floor")
        self.assertFalse(verdict["threshold_drift"]["passed"])
        self.assertIn("data_gate", verdict["failed_on"])
        self.assertFalse(verdict["passed"])
        self.assertTrue(any("WARN-THRESHOLD-DRIFT" in item
                            for item in verdict["warnings"]))

    def test_a_short_lived_cohort_warns_about_the_followup_definition(self):
        """The two follow-up definitions can flip the >= 50 check: say so."""
        root = case("followup_gap")
        build_gate_root(root, life=14)
        verdict = analyzer.analyse(root)
        for canonical in ("A", "B", "C"):
            summary = verdict["followup"][canonical]
            self.assertEqual(
                summary["n_valid_followup_full_window_gated"], 0,
                "a 14-interval observed life cannot cover a 16/18-interval "
                "response horizon")
            self.assertEqual(summary["n_usable_followup_p22_definition"],
                             GATE_TASKS)
            self.assertFalse(verdict["gate_checks"]
                             ["%s_valid_onset_followup_enough" % canonical])
        self.assertTrue(any("WARN-FOLLOWUP-DEFINITION" in item
                            for item in verdict["warnings"]),
                        "the definitional gap must be impossible to miss")
        self.assertIn("data_gate", verdict["failed_on"])

    def test_compact_verdict_carries_the_gate_booleans(self):
        compact = analyzer.compact_verdict(self.verdict)
        self.assertTrue(compact["gate_passed"])
        self.assertTrue(compact["passed"])
        self.assertEqual(sorted(compact["per_regime"]), ["A", "B", "C"])
        self.assertEqual(compact["per_regime"]["B"]["onset_tau"],
                         REGISTERED_TAU["ram"])
        self.assertTrue(compact["per_regime"]["C"]["onset_provisional"])
        self.assertFalse(compact["per_regime"]["A"]["onset_provisional"])
        json.dumps(compact)                       # strict JSON, no numpy types


# --------------------------------------------------------------------------
# T023-S2-07 — a missing stream is an error, never a zero-filled PASS
# --------------------------------------------------------------------------
class MissingStreamTests(unittest.TestCase):
    def test_an_absent_root_is_reported_with_its_path(self):
        root = case("missing_root/nothing")
        with self.assertRaises(analyzer.StreamMissingError) as caught:
            analyzer.analyse(root)
        self.assertIn(str(root), str(caught.exception))

    def test_a_partial_root_names_every_missing_single_stream(self):
        root = case("partial")
        single_stream(root / (analyzer.REGISTERED_SINGLE_TAG
                              % "compute_first"), "compute_first")
        with self.assertRaises(analyzer.StreamMissingError) as caught:
            analyzer.analyse(root)
        message = str(caught.exception)
        for missing in ("single_memory_first", "single_io_first"):
            self.assertIn(missing, message)
        self.assertIn("no zero-filled PASS", message)

    def test_a_missing_development_stream_warns_instead_of_blocking(self):
        """The three cohorts fully determine the plan §9 gate; say so loudly."""
        root = case("no_dev")
        for gen_id in analyzer.REGISTERED_REGIME_IDS:
            single_stream(root / (analyzer.REGISTERED_SINGLE_TAG % gen_id),
                          gen_id)
        verdict = analyzer.analyse(root)
        self.assertIsNone(verdict["dev_stream"])
        self.assertEqual(verdict["dev_stream_missing"]["role"], "dev")
        self.assertEqual(verdict["dev_stream_missing"]["reason"], "not found")
        self.assertIn(analyzer.REGISTERED_DEV_TAG,
                      verdict["dev_stream_missing"]["expected_path"])
        self.assertTrue(verdict["gate_passed"],
                        "the gate is per regime and the three cohorts are here")
        self.assertTrue(any("WARN-NO-DEV-STREAM" in item
                            for item in verdict["warnings"]))
        self.assertEqual(verdict["warnings"],
                         [item for item in verdict["warnings"]
                          if "WARN-NO-DEV-STREAM" in item],
                         "no other warning on a clean synthetic root")

    def test_a_half_written_development_stream_is_reported_not_read(self):
        """The collector creates the tag before simulating: do not read it."""
        root = case("dev_in_progress")
        for gen_id in analyzer.REGISTERED_REGIME_IDS:
            single_stream(root / (analyzer.REGISTERED_SINGLE_TAG % gen_id),
                          gen_id)
        in_progress = root / analyzer.REGISTERED_DEV_TAG
        in_progress.mkdir(parents=True)
        (in_progress / "generation.log").write_text("", encoding="utf8")
        verdict = analyzer.analyse(root)
        self.assertIsNone(verdict["dev_stream"])
        self.assertIn("incomplete",
                      verdict["dev_stream_missing"]["reason"])
        self.assertIn("still in progress",
                      verdict["dev_stream_missing"]["reason"])
        self.assertTrue(verdict["gate_passed"])

    def test_main_exits_two_and_writes_an_error_artifact(self):
        root = case("missing_root/nothing")
        audit_out = case("missing_audit")
        code = analyzer.main(["--streams-root", str(root),
                              "--audit-out", str(audit_out)])
        self.assertEqual(code, 2)
        error = json.loads((audit_out / "analysis_error.json").read_text(
            encoding="utf8"))
        self.assertFalse(error["passed"])
        self.assertFalse(error["gate_passed"])
        self.assertEqual(error["exit_code"], 2)
        self.assertIn("StreamMissingError", error["error"])
        self.assertFalse((audit_out / "DATA_GATE.md").exists(),
                         "no verdict artifact may be written without data")

    def test_only_analyses_one_regime_without_the_other_streams(self):
        root = case("only_root")
        tiny_stream(root / (analyzer.REGISTERED_SINGLE_TAG % "io_first"))
        audit_out = case("only_audit")
        code = analyzer.main(["--streams-root", str(root), "--only",
                              "io_first", "--audit-out", str(audit_out)])
        self.assertIn(code, (0, 1))
        self.assertTrue((audit_out / "regime_C.json").is_file())
        self.assertFalse((audit_out / "regime_A.json").exists())
        marginal = json.loads((audit_out / "marginal_match.json").read_text(
            encoding="utf8"))
        self.assertTrue(marginal["partial_run"])
        self.assertIn("prevalence_spread_ok", marginal["checks_not_evaluated"])
        markdown = (audit_out / "DATA_GATE.md").read_text(encoding="utf8")
        self.assertIn("PARTIAL RUN", markdown)

    def test_an_unregistered_only_value_is_rejected(self):
        root = case("only_root")
        with self.assertRaises(analyzer.AnalyzerError):
            analyzer.analyse(root, only="regime_D")


# --------------------------------------------------------------------------
# T023-S2-08 — declared numbers are claims, and the analysis is deterministic
# --------------------------------------------------------------------------
class DeclaredAndDeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = case("declared")
        directory = single_stream(
            cls.root / (analyzer.REGISTERED_SINGLE_TAG % "memory_first"),
            "memory_first")
        lying = {"protocol": "023", "passed": True,
                 "per_regime": {"B": {
                     "gate": {"passed": True, "metrics": {
                         "prevalence": 0.5, "independent_fault_events": 4242,
                         "independent_cascade_fault_events": 4242,
                         "deployment_attempts": 1, "deployment_rejected": 0,
                         "deployment_rejection_rate": 0.0,
                         "migration_attempts": 1, "migration_rejected": 0,
                         "migration_rejection_rate": 0.0,
                         "worst_event_share_of_positives": 0.9,
                         "host_steps": 1, "positive_hoststeps": 1}},
                     "task_level": {"n_onsets": 4242}}}}
        (Path(directory) / "unseen_data_audit.json").write_text(
            json.dumps(lying), encoding="utf8")
        cls.verdict = analyzer.analyse(cls.root, only="memory_first")

    def test_the_recomputed_numbers_stand_and_the_claim_is_flagged(self):
        verdict = self.verdict
        stats = verdict["gate_stats"]["B"]
        self.assertAlmostEqual(stats["prevalence"], GATE_HAND["prevalence"],
                               places=12)
        self.assertEqual(stats["independent_fault_events"],
                         GATE_HAND["independent_fault_events"])
        declared = verdict["declared_vs_recomputed"]
        self.assertFalse(declared["agree"])
        compared = declared["per_stream"]["B"]["comparison"]
        self.assertFalse(compared["prevalence"]["match"])
        self.assertEqual(compared["prevalence"]["declared"], 0.5)
        self.assertFalse(compared["n_onsets"]["match"])
        self.assertTrue(any("WARN-DECLARED-AUDIT" in item
                            for item in verdict["warnings"]))
        self.assertFalse(verdict["declared_vs_recomputed"]["per_stream"]["B"]
                         ["all_match"])

    def test_a_lying_declaration_does_not_change_the_gate(self):
        self.assertTrue(self.verdict["gate_passed"],
                        self.verdict["gate_checks"])
        self.assertTrue(self.verdict["gate_checks"]["B_prevalence_in_range"])

    def test_the_analysis_is_deterministic(self):
        first = analyzer.analyse(self.root, only="memory_first")
        second = analyzer.analyse(self.root, only="memory_first")

        def payload(verdict):
            value = dict(verdict)
            value.pop("generated_at")
            return json.dumps(analyzer.json_safe(value), sort_keys=True)

        self.assertEqual(payload(first), payload(second))


if __name__ == "__main__":
    unittest.main()
