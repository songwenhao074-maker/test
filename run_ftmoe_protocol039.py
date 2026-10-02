"""Protocol-039: one causal birth -> sleep -> wake lifecycle on cached seed3601.
Only D_sleepwake is newly trained. C/B/D_keep are exact cached controls.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, shutil, subprocess, time, traceback
from pathlib import Path

import numpy as np
import torch

import run_ftmoe_protocol037 as p37
from protocol035_common import dump_json, load_npz, sha256_file, sha256_state_dict

PLAN = Path("artifacts/ftmoe_online/protocol_039/plan.json")
PLAN_SHA = "2182aa9c08a2067ea8abb89b70555a8174fc9a2809d0655c59effa4174cd9811"
N, HOSTS, DIM = 5968, 16, 73
ABSENT = "ABSENT"
ACTIVE_BEFORE_SLEEP = "ACTIVE_BEFORE_SLEEP"
SLEEPING = "SLEEPING"
ACTIVE_FINAL = "ACTIVE_FINAL"
ACTIVE_STATES = (ACTIVE_BEFORE_SLEEP, ACTIVE_FINAL)

SOURCE036 = {
    "stage_B/seed3601/C_ref/feature_tape.npz": "c71a2ea6d59045fe4cc8a953a776d37f48dbe1feec4aba859cd00f11c984529c",
    "stage_B/seed3601/C_ref/update_batches.json": "4d32d2ece194cb41b8bc16de1c89ddfedcd42da668b96446937fb430d6246b6f",
    "stage_B/seed3601/D_lin/predictions.npz": "1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
    "inputs/seed3601/manifest.json": "73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b",
    "stage_B/seed3601/C_ref/predictions.npz": "da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
}
SOURCE037 = {
    "D_birth/predictions.npz": "badf9005b2be9fc6436d1ff2b9377b27715e846d7d16f8e4bdfcbfa76f732e8c",
    "D_birth/birth_checks.json": "4b8afb4f98abdc5405a69310206d0f145e981b28069883138fad9b7468a70d98",
    "D_birth/lifecycle_events.json": "b2eb7dc30f4855e151d39e1ef8c633eef8505547c944f21bc5d3ed33b59a2711",
    "D_birth/update_log.json": "b1b64946dc8029938e6ead7ca8e2ddb0a2decb5b0928efdcebe390a4e82f6838",
}


def J(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def W(p, x):
    dump_json(p, x)


def runtime():
    p37.runtime()


def validate_plan():
    if sha256_file(PLAN) != PLAN_SHA:
        raise AssertionError("Protocol039 plan hash changed")
    p = J(PLAN)
    if p.get("protocol") != "039" or p["budget"]["sequence_names"] != ["D_sleepwake"]:
        raise AssertionError("Protocol039 plan identity")
    if p["F_research"]["status"] != "DEFERRED":
        raise AssertionError("F research must remain deferred")
    return p


def recursive_digest(x):
    h = hashlib.sha256()

    def add(v):
        if torch.is_tensor(v):
            a = v.detach().cpu().contiguous().numpy()
            h.update(b"T"); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(b"N"); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(v, dict):
            h.update(b"D")
            for k in sorted(v, key=lambda q: str(q)):
                h.update(str(k).encode()); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(b"L")
            for z in v: add(z)
        else:
            h.update(b"S"); h.update(repr(v).encode())
    add(x)
    return h.hexdigest()


def optimizer_digest(opt):
    return recursive_digest(opt.state_dict())


def optimizer_step_value(opt):
    vals = []
    for s in opt.state.values():
        v = s.get("step")
        if torch.is_tensor(v):
            vals.append(int(v.item()))
        elif v is not None:
            vals.append(int(v))
    return max(vals) if vals else 0


def stable_interval_bce_from_logits(logits, labels):
    m = np.asarray(logits, dtype=np.float64)
    margin = m[..., 1] - m[..., 0]
    return p37.stable_bce_interval(margin, labels)


def verify_file_set(root, expected, label):
    root = Path(root)
    actual = {}
    for rel, digest in expected.items():
        q = root / rel
        if not q.exists():
            raise AssertionError(f"{label} missing locked file: {rel}")
        got = sha256_file(q)
        actual[rel] = got
        if got != digest:
            raise AssertionError(f"{label} hash mismatch: {rel}")
    return actual


def load_sources(source036, source037):
    source036, source037 = Path(source036), Path(source037)
    h36 = verify_file_set(source036, SOURCE036, "source036")
    h37 = verify_file_set(source037, SOURCE037, "source037")
    core = p37.load_source(source036, verify=True)
    C = load_npz(source036 / "stage_B/seed3601/C_ref/predictions.npz")
    B = core["B"]
    K = load_npz(source037 / "D_birth/predictions.npz")
    keep_checks = J(source037 / "D_birth/birth_checks.json")
    keep_lifecycle = J(source037 / "D_birth/lifecycle_events.json")
    keep_updates = J(source037 / "D_birth/update_log.json")

    for name, arr in (("C", C), ("B", B), ("D_keep", K)):
        if arr["probability"].shape != (N, HOSTS) or arr["detection_logits"].shape != (N, HOSTS, 2):
            raise AssertionError(name + " cached prediction geometry")
    for arr in (B, K):
        for key in ("labels", "raw_labels", "class_probability"):
            if key not in C or key not in arr or not np.array_equal(C[key], arr[key]):
                raise AssertionError("cached control alignment mismatch: " + key)
    if not np.array_equal(core["T"]["labels"], B["labels"]):
        raise AssertionError("issued tape labels do not match B")

    events = keep_lifecycle.get("events", keep_lifecycle if isinstance(keep_lifecycle, list) else [])
    births = [x for x in events if x.get("event") == "birth"]
    if len(births) != 1 or int(births[0].get("at_interval", -1)) != 351:
        raise AssertionError("cached D_keep birth is not exactly t351")
    if int(births[0].get("first_affected_prediction", -1)) != 352:
        raise AssertionError("cached D_keep first affected prediction is not 352")
    updates = keep_updates.get("updates", keep_updates if isinstance(keep_updates, list) else [])
    if len(updates) != 352:
        raise AssertionError("cached D_keep expected exactly 352 updates")
    keep_update_by_t = {int(x["at_interval"]): x for x in updates}
    if len(keep_update_by_t) != 352:
        raise AssertionError("cached D_keep update times are not unique")

    historical_cost = {
        "C_ref": J(source036 / "stage_B/seed3601/C_ref/summary.json"),
        "D_lin": J(source036 / "stage_B/seed3601/D_lin/summary.json"),
        "D_keep": J(source037 / "D_birth/summary.json"),
    }
    return {
        "source036_root": source036,
        "source037_root": source037,
        "core": core,
        "C": C, "B": B, "D_keep": K,
        "keep_birth_checks": keep_checks,
        "keep_lifecycle": keep_lifecycle,
        "keep_updates": updates,
        "keep_update_by_t": keep_update_by_t,
        "historical_cost": historical_cost,
        "hashes036": h36, "hashes037": h37,
        "source_id": "p036_run36831958978+p037_run36871854703_seed3601",
        "n": N,
    }


def default_ledger(fixture):
    return {
        "protocol": "039",
        "training_sequence_budget": 1,
        "sequence": {
            "name": "D_sleepwake", "started": False, "completed": False,
            "restart_from_zero": False, "resume_events": [], "optimizer_steps": 0,
            "prediction_forward_calls": 0,
        },
        "scientific_optimizer_steps_max": 352,
        "new_streams": 0, "baseline_retraining": 0, "other_real_streams": 0,
        "hyperparameter_sweeps": 0, "donor_updates": 0, "F_training": 0,
        "permanent_deletions": 0,
        "max_births": 1, "max_sleeps": 1, "max_wakes": 1,
        "real_engineering_prefixes": int(fixture.get("real_engineering_prefixes", 0)),
        "real_engineering_prefix_steps_max": int(fixture.get("real_engineering_prefix_steps_max", 0)),
        "real_engineering_gradient_steps": int(fixture.get("real_engineering_gradient_steps", 0)),
        "births": 0, "sleeps": 0, "wakes": 0,
        "automatic_next_protocol": False,
    }


def ledger_event_path(path):
    return Path(path).with_name("budget_ledger_events.jsonl")


def append_ledger(path, event):
    p = ledger_event_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf8") as f:
        f.write(json.dumps({"protocol": "039", "time_unix": time.time(), **event}, ensure_ascii=False, allow_nan=False) + "\n")


def init_or_load_ledger(path, fixture):
    p = Path(path)
    if not p.exists():
        W(p, default_ledger(fixture))
    d = J(p)
    if d.get("protocol") != "039" or d.get("training_sequence_budget") != 1:
        raise AssertionError("bad Protocol039 ledger")
    return d


def start_sequence(path, run_id, fixture, resume_checkpoint=None):
    d = init_or_load_ledger(path, fixture)
    s = d["sequence"]
    if s["completed"]:
        raise RuntimeError("Protocol039 scientific sequence already completed")
    if s["started"]:
        if not resume_checkpoint:
            raise RuntimeError("D_sleepwake already started; exact resume required")
        ev = {"event": "resume", "run_id": str(run_id), "checkpoint": str(resume_checkpoint)}
        s["resume_events"].append({**ev, "time_unix": time.time()})
        append_ledger(path, ev)
    else:
        if resume_checkpoint:
            raise RuntimeError("resume checkpoint provided before sequence start")
        s["started"] = True
        s["started_run_id"] = str(run_id)
        s["started_time_unix"] = time.time()
        append_ledger(path, {"event": "start", "sequence": "D_sleepwake", "run_id": str(run_id)})
    W(path, d)


def sync_ledger_counts(path, machine):
    d = J(path)
    s = d["sequence"]
    s["optimizer_steps"] = int(machine.optimizer_step_calls)
    s["prediction_forward_calls"] = int(machine.prediction_forward_calls)
    d["births"] = int(machine.birth_t is not None)
    d["sleeps"] = int(machine.sleep_t is not None)
    d["wakes"] = int(machine.wake_t is not None)
    W(path, d)


def complete_ledger(path, machine, pred_sha):
    sync_ledger_counts(path, machine)
    d = J(path)
    s = d["sequence"]
    s["completed"] = True
    s["completed_time_unix"] = time.time()
    s["prediction_sha256"] = pred_sha
    s["birth_t"] = machine.birth_t
    s["sleep_t"] = machine.sleep_t
    s["wake_t"] = machine.wake_t
    W(path, d)
    append_ledger(path, {"event": "complete", "sequence": "D_sleepwake"})


def alloc_outputs(n):
    return {
        "probability": np.full((n, HOSTS), np.nan, np.float32),
        "detection_logits": np.full((n, HOSTS, 2), np.nan, np.float32),
        "delta": np.full((n, HOSTS), np.nan, np.float32),
        "issued_state": np.full(n, "", dtype="<U24"),
        "expert_active": np.zeros(n, np.int8),
        "expert_version": np.full(n, -1, np.int64),
        "expert_hash": np.full(n, "", dtype="<U64"),
    }


class Machine:
    """Production state machine. Synthetic fixtures use this exact class and step interface."""
    def __init__(self, src, work_dir, identity=None, forced_events=None, allow_gradient=True):
        self.src = src
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity or {})
        self.forced_events = dict(forced_events or {})
        self.allow_gradient = bool(allow_gradient)

        self.cursor = 0
        self.substep = "before_predict"
        self.state = ABSENT
        self.expert = None
        self.optimizer = None
        self.version = -1

        self.birth_streak = 0
        self.sleep_streak = 0
        self.wake_streak = 0
        self.birth_t = None
        self.sleep_t = None
        self.wake_t = None
        self.first_birth_prediction = None
        self.first_sleep_prediction = None
        self.first_wake_prediction = None

        n = int(src["n"])
        self.out = alloc_outputs(n)
        self.b_loss = np.full(n, np.nan, np.float64)
        self.d_loss = np.full(n, np.nan, np.float64)
        self.settled = np.zeros(n, np.int8)

        self.birth_checks = []
        self.sleep_checks = []
        self.wake_checks = []
        self.lifecycle_events = []
        self.update_events = []
        self.forward_events = []
        self.settlement_events = []

        self.prediction_forward_calls = 0
        self.training_forward_calls = 0
        self.optimizer_step_calls = 0
        self.lifecycle_check_calls = 0
        self.state_check_cpu_seconds = 0.0
        self.prediction_forward_seconds = 0.0
        self.update_seconds = 0.0
        self.sleep_prediction_count = 0

        self.action_seq = 0
        self.action_journal = self.work_dir / "action_journal.json"
        self.sleep_snapshot = None
        self.sleep_expert_hash = None
        self.sleep_optimizer_hash = None
        self.sleep_optimizer_step = None
        self.wake_pre_expert_hash = None
        self.wake_pre_optimizer_hash = None
        self.wake_pre_optimizer_step = None
        self.terminal_actual = None

        self.checkpoint_hook = None

    def model_hash(self):
        return None if self.expert is None else sha256_state_dict(self.expert.state_dict())

    def opt_hash(self):
        return None if self.optimizer is None else optimizer_digest(self.optimizer)

    def _journal_begin(self, kind, t, meta=None):
        self.action_seq += 1
        row = {
            "protocol": "039", "status": "pending", "action_seq": int(self.action_seq),
            "kind": kind, "cursor": int(t), "state": self.state, "meta": meta or {},
        }
        W(self.action_journal, row)
        return row

    def _journal_commit(self, row, extra=None):
        row = dict(row)
        row["status"] = "committed"
        if extra: row.update(extra)
        W(self.action_journal, row)

    def _checkpoint(self, reason, named=False):
        if self.checkpoint_hook:
            self.checkpoint_hook(self, reason, named=named)

    def _predict(self, t):
        B = self.src["B"]
        brow = B["detection_logits"][t]
        self.out["issued_state"][t] = self.state
        if self.state in ACTIVE_STATES:
            if self.expert is None:
                raise AssertionError("active state without expert")
            row = self._journal_begin("prediction_forward", t)
            t0 = time.perf_counter()
            dd, logits, prob, _ = p37.expert_predict(self.expert, self.src["core"]["T"]["z"][t], brow, exact_b_if_zero=True)
            elapsed = time.perf_counter() - t0
            self.prediction_forward_calls += 1
            self.prediction_forward_seconds += elapsed
            self._journal_commit(row, {"elapsed_seconds": elapsed})
            if logits is None:
                self.out["detection_logits"][t] = brow
                self.out["probability"][t] = B["probability"][t]
            else:
                self.out["detection_logits"][t] = logits
                self.out["probability"][t] = prob
            self.out["delta"][t] = dd
            self.out["expert_active"][t] = 1
            self.out["expert_version"][t] = int(self.version)
            self.out["expert_hash"][t] = self.model_hash()
            self.forward_events.append({
                "at_interval": int(t), "issued_state": self.state,
                "expert_hash": self.out["expert_hash"][t], "expert_version": int(self.version),
            })
        else:
            self.out["detection_logits"][t] = brow
            self.out["probability"][t] = B["probability"][t]
            self.out["delta"][t] = 0.0
            self.out["expert_active"][t] = 0
            self.out["expert_version"][t] = int(self.version)
            self.out["expert_hash"][t] = "absent" if self.expert is None else self.model_hash()
            if self.state == SLEEPING:
                self.sleep_prediction_count += 1

    def _settle(self, t):
        i = int(t) - 2
        if i < 0:
            return
        if self.settled[i]:
            raise AssertionError("duplicate label settlement")
        labels = self.src["B"]["labels"][i]
        self.b_loss[i] = stable_interval_bce_from_logits(self.src["B"]["detection_logits"][i], labels)
        self.d_loss[i] = stable_interval_bce_from_logits(self.out["detection_logits"][i], labels)
        self.settled[i] = 1
        self.settlement_events.append({
            "at_interval": int(t), "settled_interval": i,
            "issued_state": str(self.out["issued_state"][i]),
            "B_mean_host_bce": float(self.b_loss[i]),
            "D_mean_host_bce": float(self.d_loss[i]),
        })

    def _due(self, t):
        return ((int(t) + 1) % 16) == 0

    def _birth_check(self, t):
        m = int(t) - 2
        if m < 319:
            return None
        recent = np.arange(m - 63, m + 1)
        previous = np.arange(m - 319, m - 63)
        if np.isnan(self.b_loss[recent]).any() or np.isnan(self.b_loss[previous]).any():
            raise AssertionError("birth check missing matured B losses")
        R = float(self.b_loss[recent].mean())
        P = float(self.b_loss[previous].mean())
        labels = self.src["B"]["labels"][recent]
        pos, neg = int((labels > 0).sum()), int((labels <= 0).sum())
        ratio = bool(R >= 1.25 * max(P, 1e-6))
        increase = bool(R - P >= 0.02)
        support = bool(pos >= 16 and neg >= 16)
        cand = bool(ratio and increase and support)
        before = int(self.birth_streak)
        self.birth_streak = before + 1 if cand else 0
        forced = self.forced_events.get(int(t)) == "birth"
        if forced:
            self.birth_streak = 2
            cand = True
        row = {
            "at_interval": int(t), "m": m, "recent": [int(m-63), int(m)],
            "previous": [int(m-319), int(m-64)], "R": R, "P": P,
            "positive_support": pos, "negative_support": neg,
            "ratio_condition": ratio, "difference_condition": increase, "support_condition": support,
            "candidate": cand, "streak_before": before, "streak_after": int(self.birth_streak),
            "forced_fixture": bool(forced),
        }
        self.birth_checks.append(row)
        return row

    def _sleep_check(self, t):
        m = int(t) - 2
        if m < 0:
            return None
        recent = np.arange(max(0, m - 127), m + 1)
        active_matured = int(np.sum(self.out["issued_state"][:m+1] == ACTIVE_BEFORE_SLEEP))
        all_recent_active = bool(len(recent) == 128 and np.all(self.out["issued_state"][recent] == ACTIVE_BEFORE_SLEEP))
        eligible = bool(active_matured >= 256 and all_recent_active)
        U = None
        pos = neg = 0
        cand = False
        if eligible:
            if np.isnan(self.b_loss[recent]).any() or np.isnan(self.d_loss[recent]).any():
                raise AssertionError("sleep check missing issued losses")
            U = float(np.mean(self.b_loss[recent] - self.d_loss[recent]))
            labels = self.src["B"]["labels"][recent]
            pos, neg = int((labels > 0).sum()), int((labels <= 0).sum())
            cand = bool(U <= 0.0)
        before = int(self.sleep_streak)
        self.sleep_streak = before + 1 if cand else 0
        forced = self.forced_events.get(int(t)) == "sleep"
        if forced:
            self.sleep_streak = 3
            cand = True
            eligible = True
        row = {
            "at_interval": int(t), "m": m, "recent": [int(max(0,m-127)), int(m)],
            "active_matured_intervals": active_matured, "all_recent_issued_active": all_recent_active,
            "eligible": eligible, "U": U, "positive_support": pos, "negative_support": neg,
            "candidate": cand, "streak_before": before, "streak_after": int(self.sleep_streak),
            "forced_fixture": bool(forced),
        }
        self.sleep_checks.append(row)
        return row

    def _wake_check(self, t):
        m = int(t) - 2
        if m < 319:
            return None
        recent = np.arange(m - 63, m + 1)
        previous = np.arange(m - 319, m - 63)
        sleep_matured = int(np.sum(self.out["issued_state"][:m+1] == SLEEPING))
        all_recent_sleep = bool(np.all(self.out["issued_state"][recent] == SLEEPING))
        eligible = bool(sleep_matured >= 64 and all_recent_sleep)
        R = P = None
        pos = neg = 0
        ratio = increase = support = cand = False
        if eligible:
            if np.isnan(self.b_loss[recent]).any() or np.isnan(self.b_loss[previous]).any():
                raise AssertionError("wake check missing B losses")
            R = float(self.b_loss[recent].mean())
            P = float(self.b_loss[previous].mean())
            labels = self.src["B"]["labels"][recent]
            pos, neg = int((labels > 0).sum()), int((labels <= 0).sum())
            ratio = bool(R >= 1.25 * max(P, 1e-6))
            increase = bool(R - P >= 0.02)
            support = bool(pos >= 16 and neg >= 16)
            cand = bool(ratio and increase and support)
        before = int(self.wake_streak)
        self.wake_streak = before + 1 if cand else 0
        forced = self.forced_events.get(int(t)) == "wake"
        if forced:
            self.wake_streak = 2
            cand = True
            eligible = True
        row = {
            "at_interval": int(t), "m": m, "recent": [int(m-63), int(m)],
            "previous": [int(m-319), int(m-64)], "sleep_matured_intervals": sleep_matured,
            "all_recent_issued_sleeping": all_recent_sleep, "eligible": eligible,
            "R": R, "P": P, "positive_support": pos, "negative_support": neg,
            "ratio_condition": ratio, "difference_condition": increase, "support_condition": support,
            "candidate": cand, "streak_before": before, "streak_after": int(self.wake_streak),
            "forced_fixture": bool(forced),
        }
        self.wake_checks.append(row)
        return row

    def _apply_birth(self, t):
        if not self.forced_events and int(t) != 351:
            raise AssertionError("Protocol039 real birth occurred at unexpected time")
        self._checkpoint("pre_birth_t%d" % t, named=True)
        self.expert = p37.make_expert()
        self.optimizer = p37.make_optimizer(self.expert)
        self.version = 0
        if any(torch.count_nonzero(q).item() for q in self.expert.parameters()):
            raise AssertionError("birth expert not zero")
        self.state = ACTIVE_BEFORE_SLEEP
        self.birth_t = int(t)
        self.first_birth_prediction = int(t) + 1
        self.lifecycle_events.append({
            "event": "birth", "at_interval": int(t), "first_affected_prediction": int(t)+1,
            "expert_hash_before_update": self.model_hash(), "optimizer_hash_before_update": self.opt_hash(),
            "optimizer_step_before_update": optimizer_step_value(self.optimizer),
        })
        self._checkpoint("post_birth_t%d" % t, named=True)

    def _apply_sleep(self, t):
        self._checkpoint("pre_sleep_t%d" % t, named=True)
        if self.expert is None or self.optimizer is None:
            raise AssertionError("sleep without expert")
        self.sleep_expert_hash = self.model_hash()
        self.sleep_optimizer_hash = self.opt_hash()
        self.sleep_optimizer_step = optimizer_step_value(self.optimizer)
        self.sleep_snapshot = {
            "expert": copy.deepcopy(self.expert.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "expert_hash": self.sleep_expert_hash,
            "optimizer_hash": self.sleep_optimizer_hash,
            "optimizer_step": self.sleep_optimizer_step,
        }
        self.state = SLEEPING
        self.sleep_t = int(t)
        self.first_sleep_prediction = int(t) + 1
        self.lifecycle_events.append({
            "event": "sleep", "at_interval": int(t), "first_affected_prediction": int(t)+1,
            "expert_hash": self.sleep_expert_hash, "optimizer_hash": self.sleep_optimizer_hash,
            "optimizer_step": self.sleep_optimizer_step,
        })
        self._checkpoint("post_sleep_t%d" % t, named=True)

    def _apply_wake(self, t):
        self._checkpoint("pre_wake_t%d" % t, named=True)
        if self.sleep_snapshot is None or self.expert is None or self.optimizer is None:
            raise AssertionError("wake without preserved sleeping state")
        self.wake_pre_expert_hash = self.model_hash()
        self.wake_pre_optimizer_hash = self.opt_hash()
        self.wake_pre_optimizer_step = optimizer_step_value(self.optimizer)
        if self.wake_pre_expert_hash != self.sleep_snapshot["expert_hash"]:
            raise AssertionError("sleeping expert weights changed")
        if self.wake_pre_optimizer_hash != self.sleep_snapshot["optimizer_hash"]:
            raise AssertionError("sleeping optimizer changed")
        if self.wake_pre_optimizer_step != self.sleep_snapshot["optimizer_step"]:
            raise AssertionError("sleeping Adam step changed")
        self.expert.load_state_dict(copy.deepcopy(self.sleep_snapshot["expert"]))
        self.optimizer.load_state_dict(copy.deepcopy(self.sleep_snapshot["optimizer"]))
        if self.model_hash() != self.sleep_snapshot["expert_hash"] or self.opt_hash() != self.sleep_snapshot["optimizer_hash"]:
            raise AssertionError("wake restore not exact")
        self.state = ACTIVE_FINAL
        self.wake_t = int(t)
        self.first_wake_prediction = int(t) + 1
        self.lifecycle_events.append({
            "event": "wake", "at_interval": int(t), "first_affected_prediction": int(t)+1,
            "expert_hash_before_update": self.model_hash(), "optimizer_hash_before_update": self.opt_hash(),
            "optimizer_step_before_update": optimizer_step_value(self.optimizer),
        })
        self._checkpoint("post_wake_t%d" % t, named=True)

    def _transition(self, t):
        if not self._due(t) or self.state == ACTIVE_FINAL:
            return
        t0 = time.process_time()
        self.lifecycle_check_calls += 1
        before_events = len(self.lifecycle_events)
        if self.state == ABSENT:
            row = self._birth_check(t)
            if row and row["streak_after"] >= 2:
                self._apply_birth(t)
                row["state_change"] = "birth"
            elif row:
                row["state_change"] = "none"
            # real source must reproduce registered t351 birth exactly.
            if not self.forced_events and int(t) == 351 and self.birth_t != 351:
                raise AssertionError("Protocol039 birth did not reproduce expected t351")
            if not self.forced_events and self.birth_t is not None and self.birth_t != 351:
                raise AssertionError("Protocol039 birth occurred at unexpected time")
        elif self.state == ACTIVE_BEFORE_SLEEP:
            row = self._sleep_check(t)
            if row and row["streak_after"] >= 3:
                self._apply_sleep(t)
                row["state_change"] = "sleep"
            elif row:
                row["state_change"] = "none"
        elif self.state == SLEEPING:
            row = self._wake_check(t)
            if row and row["streak_after"] >= 2:
                self._apply_wake(t)
                row["state_change"] = "wake"
            elif row:
                row["state_change"] = "none"
        if len(self.lifecycle_events) - before_events > 1:
            raise AssertionError("more than one state transition at a cursor")
        self.state_check_cpu_seconds += time.process_time() - t0
        # persist state after every real lifecycle check, even when no transition.
        self._checkpoint("after_check_t%d" % t, named=False)

    def _update(self, t):
        if t not in self.src["core"]["by_t"] or self.state not in ACTIVE_STATES:
            return
        if not self.allow_gradient:
            raise AssertionError("engineering prefix attempted gradient update")
        if self.optimizer_step_calls >= 352:
            raise AssertionError("Protocol039 scientific optimizer step budget exhausted")
        batch = [int(x) for x in self.src["core"]["by_t"][t]["batch_indices"]]
        if batch and max(batch) + 2 > int(t):
            raise AssertionError("immature update batch")
        before_h, before_o = self.model_hash(), self.opt_hash()
        rowj = self._journal_begin("optimizer_step", t, {"batch_indices": batch})
        t0 = time.perf_counter()
        loss, gn, _ = p37.do_update(self.expert, self.optimizer, self.src["core"], batch)
        elapsed = time.perf_counter() - t0
        self.training_forward_calls += 1
        self.optimizer_step_calls += 1
        self.version += 1
        self.update_seconds += elapsed
        after_h, after_o = self.model_hash(), self.opt_hash()
        ev = {
            "at_interval": int(t), "issued_state_at_t": str(self.out["issued_state"][t]),
            "post_transition_state": self.state, "batch_indices": batch,
            "loss": float(loss), "grad_norm": float(gn), "hash_before": before_h, "hash_after": after_h,
            "optimizer_hash_before": before_o, "optimizer_hash_after": after_o,
            "optimizer_step_after": optimizer_step_value(self.optimizer), "version_after": int(self.version),
        }
        self.update_events.append(ev)
        self._journal_commit(rowj, {"hash_after": after_h, "optimizer_step_after": ev["optimizer_step_after"]})
        self._checkpoint("after_update_t%d" % t, named=False)

    def advance(self, stop_cursor=None, pause=None):
        """Run production phases. pause=(t, substep_name_after_phase) supports fixture disk-resume tests."""
        stop_cursor = self.src["n"] if stop_cursor is None else int(stop_cursor)
        while self.cursor < stop_cursor:
            t = int(self.cursor)
            if self.substep == "before_predict":
                self._predict(t); self.substep = "after_predict"
                if pause == (t, self.substep): return
            if self.substep == "after_predict":
                self._settle(t); self.substep = "after_settle"
                if pause == (t, self.substep): return
            if self.substep == "after_settle":
                self._transition(t); self.substep = "after_transition"
                if pause == (t, self.substep): return
            if self.substep == "after_transition":
                self._update(t); self.substep = "after_update"
                if pause == (t, self.substep): return
            if self.substep == "after_update":
                self.cursor += 1
                self.substep = "before_predict"
                if self.cursor % 512 == 0:
                    self._checkpoint("cursor_%04d" % self.cursor, named=True)

    def terminal_settle(self):
        before = {
            "forward": int(self.prediction_forward_calls), "steps": int(self.optimizer_step_calls),
            "events": len(self.lifecycle_events), "checks": int(self.lifecycle_check_calls),
        }
        for i in range(max(0, self.src["n"] - 2), self.src["n"]):
            if not self.settled[i]:
                labels = self.src["B"]["labels"][i]
                self.b_loss[i] = stable_interval_bce_from_logits(self.src["B"]["detection_logits"][i], labels)
                self.d_loss[i] = stable_interval_bce_from_logits(self.out["detection_logits"][i], labels)
                self.settled[i] = 1
                self.settlement_events.append({
                    "at_interval": "terminal", "settled_interval": int(i),
                    "issued_state": str(self.out["issued_state"][i]),
                    "B_mean_host_bce": float(self.b_loss[i]), "D_mean_host_bce": float(self.d_loss[i]),
                })
        after = {
            "forward": int(self.prediction_forward_calls), "steps": int(self.optimizer_step_calls),
            "events": len(self.lifecycle_events), "checks": int(self.lifecycle_check_calls),
        }
        self.terminal_actual = {k + "_delta": int(after[k] - before[k]) for k in before}
        return self.terminal_actual


def machine_payload(m, ledger_snapshot=None):
    return {
        "protocol": "039", "plan_sha256": PLAN_SHA, "source_id": m.src["source_id"],
        "identity": m.identity, "cursor": int(m.cursor), "substep": m.substep, "state": m.state,
        "expert_exists": m.expert is not None,
        "expert_state": None if m.expert is None else m.expert.state_dict(),
        "optimizer_state": None if m.optimizer is None else m.optimizer.state_dict(),
        "version": int(m.version),
        "birth_streak": int(m.birth_streak), "sleep_streak": int(m.sleep_streak), "wake_streak": int(m.wake_streak),
        "birth_t": m.birth_t, "sleep_t": m.sleep_t, "wake_t": m.wake_t,
        "first_birth_prediction": m.first_birth_prediction, "first_sleep_prediction": m.first_sleep_prediction,
        "first_wake_prediction": m.first_wake_prediction,
        "out": {k: v.copy() for k, v in m.out.items()},
        "b_loss": m.b_loss.copy(), "d_loss": m.d_loss.copy(), "settled": m.settled.copy(),
        "birth_checks": copy.deepcopy(m.birth_checks), "sleep_checks": copy.deepcopy(m.sleep_checks),
        "wake_checks": copy.deepcopy(m.wake_checks), "lifecycle_events": copy.deepcopy(m.lifecycle_events),
        "update_events": copy.deepcopy(m.update_events), "forward_events": copy.deepcopy(m.forward_events),
        "settlement_events": copy.deepcopy(m.settlement_events),
        "prediction_forward_calls": int(m.prediction_forward_calls), "training_forward_calls": int(m.training_forward_calls),
        "optimizer_step_calls": int(m.optimizer_step_calls), "lifecycle_check_calls": int(m.lifecycle_check_calls),
        "state_check_cpu_seconds": float(m.state_check_cpu_seconds),
        "prediction_forward_seconds": float(m.prediction_forward_seconds), "update_seconds": float(m.update_seconds),
        "sleep_prediction_count": int(m.sleep_prediction_count),
        "action_seq": int(m.action_seq),
        "sleep_snapshot": copy.deepcopy(m.sleep_snapshot),
        "sleep_expert_hash": m.sleep_expert_hash, "sleep_optimizer_hash": m.sleep_optimizer_hash,
        "sleep_optimizer_step": m.sleep_optimizer_step,
        "wake_pre_expert_hash": m.wake_pre_expert_hash, "wake_pre_optimizer_hash": m.wake_pre_optimizer_hash,
        "wake_pre_optimizer_step": m.wake_pre_optimizer_step, "terminal_actual": copy.deepcopy(m.terminal_actual),
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "ledger_snapshot": copy.deepcopy(ledger_snapshot),
    }


def atomic_torch_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(m, checkpoint_dir, reason, ledger_path=None, named=False):
    checkpoint_dir = Path(checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ledger = None if ledger_path is None or not Path(ledger_path).exists() else J(ledger_path)
    payload = machine_payload(m, ledger)
    latest = checkpoint_dir / "latest.pt"
    atomic_torch_save(payload, latest)
    meta = {
        "protocol": "039", "reason": reason, "cursor": int(m.cursor), "substep": m.substep,
        "state": m.state, "action_seq": int(m.action_seq), "sha256": sha256_file(latest),
        "plan_sha256": PLAN_SHA, "source_id": m.src["source_id"],
    }
    W(checkpoint_dir / "latest.json", meta)
    if named:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in reason)
        dst = checkpoint_dir / (safe + ".pt")
        shutil.copy2(latest, dst)
        W(str(dst) + ".json", {**meta, "sha256": sha256_file(dst), "file": dst.name})
    return latest


def restore_checkpoint(path, src, work_dir, identity=None, forced_events=None, allow_gradient=True):
    path = Path(path)
    meta_path = path.with_suffix(".json") if path.name != "latest.pt" else path.with_name("latest.json")
    meta = J(meta_path)
    if sha256_file(path) != meta["sha256"]:
        raise AssertionError("checkpoint hash mismatch")
    x = torch.load(path, map_location="cpu", weights_only=False)
    if x.get("protocol") != "039" or x.get("plan_sha256") != PLAN_SHA or x.get("source_id") != src["source_id"]:
        raise AssertionError("checkpoint identity mismatch")
    journal = Path(work_dir) / "action_journal.json"
    if journal.exists():
        j = J(journal)
        if j.get("status") == "pending":
            raise RuntimeError("ambiguous_step: pending instrumented action exists")
        if int(j.get("action_seq", 0)) > int(x.get("action_seq", 0)):
            raise RuntimeError("ambiguous_step: committed action newer than checkpoint")
    m = Machine(src, work_dir, identity=identity or x.get("identity"), forced_events=forced_events, allow_gradient=allow_gradient)
    m.cursor = int(x["cursor"]); m.substep = x["substep"]; m.state = x["state"]
    if x["expert_exists"]:
        m.expert = p37.make_expert(); m.optimizer = p37.make_optimizer(m.expert)
        m.expert.load_state_dict(x["expert_state"]); m.optimizer.load_state_dict(x["optimizer_state"])
    m.version = int(x["version"])
    for k in ("birth_streak","sleep_streak","wake_streak"):
        setattr(m, k, int(x[k]))
    for k in ("birth_t","sleep_t","wake_t","first_birth_prediction","first_sleep_prediction","first_wake_prediction"):
        setattr(m, k, x[k])
    m.out = {k: v.copy() for k, v in x["out"].items()}
    m.b_loss = x["b_loss"].copy(); m.d_loss = x["d_loss"].copy(); m.settled = x["settled"].copy()
    for k in ("birth_checks","sleep_checks","wake_checks","lifecycle_events","update_events","forward_events","settlement_events"):
        setattr(m, k, copy.deepcopy(x[k]))
    for k in ("prediction_forward_calls","training_forward_calls","optimizer_step_calls","lifecycle_check_calls","action_seq","sleep_prediction_count"):
        setattr(m, k, int(x[k]))
    for k in ("state_check_cpu_seconds","prediction_forward_seconds","update_seconds"):
        setattr(m, k, float(x[k]))
    for k in ("sleep_snapshot","sleep_expert_hash","sleep_optimizer_hash","sleep_optimizer_step","wake_pre_expert_hash","wake_pre_optimizer_hash","wake_pre_optimizer_step","terminal_actual"):
        setattr(m, k, copy.deepcopy(x[k]))
    torch.set_rng_state(x["torch_rng"]); np.random.set_state(x["numpy_rng"]); random.setstate(x["python_rng"])
    return m, x.get("ledger_snapshot")


def _array_exact(a,b):
    aa=np.asarray(a); bb=np.asarray(b)
    if aa.dtype.kind in ("f","c") or bb.dtype.kind in ("f","c"):
        return np.array_equal(aa,bb,equal_nan=True)
    return np.array_equal(aa,bb)

def machine_equivalent(a, b, n=None):
    n = int(a.src["n"] if n is None else n)
    if a.cursor != b.cursor or a.substep != b.substep or a.state != b.state:
        return False
    scalar = ("version","birth_streak","sleep_streak","wake_streak","birth_t","sleep_t","wake_t",
              "prediction_forward_calls","training_forward_calls","optimizer_step_calls","lifecycle_check_calls","sleep_prediction_count","action_seq")
    if any(getattr(a,k) != getattr(b,k) for k in scalar):
        return False
    for k in a.out:
        if not _array_exact(a.out[k][:n], b.out[k][:n]):
            return False
    if not _array_exact(a.b_loss[:n], b.b_loss[:n]) or not _array_exact(a.d_loss[:n], b.d_loss[:n]):
        return False
    for k in ("birth_checks","sleep_checks","wake_checks","lifecycle_events","update_events","forward_events","settlement_events"):
        if getattr(a,k) != getattr(b,k):
            return False
    if (a.expert is None) != (b.expert is None):
        return False
    if a.expert is not None and a.model_hash() != b.model_hash():
        return False
    if (a.optimizer is None) != (b.optimizer is None):
        return False
    if a.optimizer is not None and a.opt_hash() != b.opt_hash():
        return False
    if recursive_digest(torch.get_rng_state()) != recursive_digest(torch.get_rng_state()):
        return False
    return True


def synthetic_source():
    n = 1050
    rng = np.random.default_rng(39039)
    z = rng.normal(0, 0.25, size=(n, HOSTS, DIM)).astype(np.float32)
    y = (((np.arange(n)[:,None] * 3 + np.arange(HOSTS)[None,:]) % 11) < 3).astype(np.int64)
    margin = (0.25*z[...,0] - 0.12*z[...,1] + 0.08*z[...,2] + rng.normal(0,0.04,size=(n,HOSTS))).astype(np.float32)
    logits = np.stack((-margin/2, margin/2), axis=-1).astype(np.float32)
    prob = (1/(1+np.exp(-margin))).astype(np.float32)
    cls = np.zeros((n,HOSTS,4), np.float32); cls[...,0] = 1.0
    T = {"z":z, "labels":y.copy(), "raw_labels":y.copy(), "c_class_probability":cls.copy()}
    B = {"detection_logits":logits, "probability":prob, "labels":y.copy(), "raw_labels":y.copy(),
         "class_probability":cls.copy(), "model_version":np.zeros(n,np.int64)}
    updates=[]
    for t in range(15,n,16):
        hi=max(0,t-1); lo=max(0,hi-32); batch=list(range(lo,hi))
        if not batch: batch=[0]
        updates.append({"at_interval":t,"batch_indices":batch})
    core={"T":T,"B":B,"updates":updates,"by_t":{r["at_interval"]:r for r in updates}}
    return {"core":core,"B":B,"C":B,"D_keep":B,"keep_update_by_t":{},"source_id":"p039_synthetic","n":n}


def run_with_checkpoints(machine, checkpoint_dir, ledger_path=None, stop_cursor=None, pause=None):
    machine.checkpoint_hook = lambda m, reason, named=False: save_checkpoint(m, checkpoint_dir, reason, ledger_path=ledger_path, named=named)
    machine.advance(stop_cursor=stop_cursor, pause=pause)
    return machine


def fixture_recovery_case(src, base_work, pause, forced):
    case=Path(base_work)/("case_%s_%s"%pause)
    a = Machine(src, case, forced_events=forced, allow_gradient=True)
    run_with_checkpoints(a, case/"checkpoints", stop_cursor=src["n"], pause=pause)
    cp = save_checkpoint(a, case/"checkpoints", "fixture_pause", named=True)
    saved=torch.load(cp,map_location="cpu",weights_only=False)
    expected_rng=recursive_digest({"torch":saved["torch_rng"],"numpy":saved["numpy_rng"],"python":saved["python_rng"]})
    # Deliberately perturb all global RNGs before disk restore.
    torch.rand(17); np.random.rand(17); random.random()
    b, _ = restore_checkpoint(cp, src, case, forced_events=forced, allow_gradient=True)
    restored_rng=recursive_digest({"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate()})
    if restored_rng!=expected_rng:
        raise AssertionError("checkpoint failed to restore RNG exactly")
    b._fixture_rng_restore_pass=True
    run_with_checkpoints(b, case/"checkpoints", stop_cursor=src["n"])
    b.terminal_settle()
    return b


def cmd_preflight(a):
    runtime(); validate_plan()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    src=load_sources(a.source036,a.source037)

    # Fixed read-only baseline/source validation. No F paths are opened.
    baseline = {
        "protocol":"039","source036_hashes":src["hashes036"],"source037_hashes":src["hashes037"],
        "C_B_Dkeep_labels_equal":bool(np.array_equal(src["C"]["labels"],src["B"]["labels"]) and np.array_equal(src["C"]["labels"],src["D_keep"]["labels"])),
        "C_B_Dkeep_raw_labels_equal":bool(np.array_equal(src["C"]["raw_labels"],src["B"]["raw_labels"]) and np.array_equal(src["C"]["raw_labels"],src["D_keep"]["raw_labels"])),
        "C_B_Dkeep_class_probability_equal":bool(np.array_equal(src["C"]["class_probability"],src["B"]["class_probability"]) and np.array_equal(src["C"]["class_probability"],src["D_keep"]["class_probability"])),
        "cached_D_keep_birth_t":351,"cached_D_keep_update_count":len(src["keep_updates"]),
        "F_files_loaded":False,
    }
    baseline["pass"]=bool(all([baseline["C_B_Dkeep_labels_equal"],baseline["C_B_Dkeep_raw_labels_equal"],baseline["C_B_Dkeep_class_probability_equal"],baseline["cached_D_keep_update_count"]==352]))
    W(out/"baseline_validation.json",baseline)
    if not baseline["pass"]: raise AssertionError("cached baseline validation failed")

    # Synthetic production-entrypoint fixture with forced engineering-only transitions.
    forced={351:"birth",671:"sleep",751:"wake"}
    syn=synthetic_source()
    cont=Machine(syn,out/"synthetic_continuous",forced_events=forced,allow_gradient=True)
    run_with_checkpoints(cont,out/"synthetic_continuous/checkpoints")
    terminal=cont.terminal_settle()

    pauses=[(351,"after_settle"),(351,"after_transition"),(671,"after_settle"),(671,"after_transition"),(751,"after_transition"),(783,"after_update")]
    recoveries=[]
    for pause in pauses:
        rr=fixture_recovery_case(syn,out/"synthetic_resume",pause,forced)
        ok=machine_equivalent(cont,rr,syn["n"]) and rr.terminal_actual==cont.terminal_actual
        rng_ok=bool(getattr(rr,"_fixture_rng_restore_pass",False))
        recoveries.append({"pause":[pause[0],pause[1]],"pass":bool(ok and rng_ok),"rng_restore_pass":rng_ok})
        if not (ok and rng_ok): raise AssertionError("synthetic disk recovery mismatch: "+repr(pause))

    # True future feature/label/B-logit perturbation, rerun through same production entrypoint.
    cutoff=500
    base500=Machine(syn,out/"future_base",forced_events=forced,allow_gradient=True)
    run_with_checkpoints(base500,out/"future_base/checkpoints",stop_cursor=cutoff)
    pert=copy.deepcopy(syn)
    pert["core"]=dict(syn["core"]); pert["core"]["T"]={k:v.copy() for k,v in syn["core"]["T"].items()}
    pert["B"]={k:v.copy() for k,v in syn["B"].items()}; pert["core"]["B"]=pert["B"]
    z0=pert["core"]["T"]["z"][cutoff:].copy(); y0=pert["B"]["labels"][cutoff:].copy(); l0=pert["B"]["detection_logits"][cutoff:].copy()
    pert["core"]["T"]["z"][cutoff:]+=np.float32(.333)
    pert["B"]["labels"][cutoff:]=1-pert["B"]["labels"][cutoff:]
    pert["core"]["T"]["labels"]=pert["B"]["labels"]
    pert["B"]["detection_logits"][cutoff:,:,1]+=np.float32(.271)
    pert["B"]["probability"][cutoff:]=(1/(1+np.exp(-(pert["B"]["detection_logits"][cutoff:,:,1]-pert["B"]["detection_logits"][cutoff:,:,0])))).astype(np.float32)
    changed={"feature_values":int(np.count_nonzero(pert["core"]["T"]["z"][cutoff:]!=z0)),
             "label_values":int(np.count_nonzero(pert["B"]["labels"][cutoff:]!=y0)),
             "logit_values":int(np.count_nonzero(pert["B"]["detection_logits"][cutoff:]!=l0))}
    if not all(v>0 for v in changed.values()): raise AssertionError("future perturbation was empty")
    pert500=Machine(pert,out/"future_perturbed",forced_events=forced,allow_gradient=True)
    run_with_checkpoints(pert500,out/"future_perturbed/checkpoints",stop_cursor=cutoff)
    future_pass=machine_equivalent(base500,pert500,cutoff)
    if not future_pass: raise AssertionError("future perturbation changed truncated production state")

    # Actual sleep inactivity and preserved parameter/optimizer state.
    sleeping_forward=[x for x in cont.forward_events if x["issued_state"]==SLEEPING]
    sleeping_updates=[x for x in cont.update_events if x["issued_state_at_t"]==SLEEPING and x["post_transition_state"]==SLEEPING]
    sleep_hash_pass=bool(cont.sleep_expert_hash==cont.wake_pre_expert_hash and cont.sleep_optimizer_hash==cont.wake_pre_optimizer_hash and cont.sleep_optimizer_step==cont.wake_pre_optimizer_step)

    # One real engineering prefix, <=256, zero gradients and no lifecycle scans past bound.
    real=Machine(src,out/"real_prefix",identity={"kind":"real_engineering_prefix"},allow_gradient=False)
    run_with_checkpoints(real,out/"real_prefix/checkpoints",stop_cursor=256)
    real_pass=bool(real.cursor==256 and real.optimizer_step_calls==0 and real.prediction_forward_calls==0 and real.birth_t is None and np.array_equal(real.out["probability"][:256],src["B"]["probability"][:256]))
    if not real_pass: raise AssertionError("real engineering prefix failed")

    report={
        "protocol":"039",
        "synthetic_forced_events_only":forced,
        "synthetic_disk_recovery":recoveries,
        "future_perturbation":{"cutoff":cutoff,"changed":changed,"truncated_state_unchanged":future_pass},
        "sleeping_actual_prediction_forward_calls":len(sleeping_forward),
        "sleeping_actual_optimizer_steps":len(sleeping_updates),
        "sleep_preserved_expert_optimizer_hash":sleep_hash_pass,
        "terminal_actual_counter_deltas":terminal,
        "real_engineering_prefixes":1,"real_engineering_prefix_steps_max":256,"real_engineering_gradient_steps":0,
        "real_prefix_pass":real_pass,
        "production_entrypoint_shared":True,
    }
    report["all_pass"]=bool(all(x["pass"] for x in recoveries) and future_pass and len(sleeping_forward)==0 and len(sleeping_updates)==0 and sleep_hash_pass and all(v==0 for v in terminal.values()) and real_pass)
    W(out/"fixture_report.json",report)
    if not report["all_pass"]: raise AssertionError("Protocol039 fixture aggregate failed")
    print(json.dumps(report,indent=2))


def save_predictions(root, src, m):
    root=Path(root); d=root/"D_sleepwake"; d.mkdir(parents=True,exist_ok=True)
    B=src["B"]
    payload={
        "probability":m.out["probability"],"detection_logits":m.out["detection_logits"],
        "class_probability":B["class_probability"],"labels":B["labels"],"raw_labels":B["raw_labels"],
        "model_version":B.get("model_version",np.zeros(src["n"],np.int64)),
        "delta":m.out["delta"],"issued_state":m.out["issued_state"],"expert_active":m.out["expert_active"],
        "expert_version":m.out["expert_version"],"expert_hash":m.out["expert_hash"],
    }
    if "class_logits" in B: payload["class_logits"]=B["class_logits"]
    np.savez_compressed(d/"predictions.npz",**payload)
    W(d/"update_log.json",{"protocol":"039","updates":m.update_events})
    W(d/"forward_log.json",{"protocol":"039","prediction_forwards":m.forward_events})
    W(d/"birth_checks.json",{"protocol":"039","checks":m.birth_checks})
    W(d/"sleep_checks.json",{"protocol":"039","checks":m.sleep_checks})
    W(d/"wake_checks.json",{"protocol":"039","checks":m.wake_checks})
    W(d/"lifecycle_events.json",{"protocol":"039","events":m.lifecycle_events})
    W(d/"settlement_events.json",{"protocol":"039","events":m.settlement_events})


def science_audit(src,m,fixture,ledger):
    sleep_t=m.sleep_t
    compare_end=src["n"] if sleep_t is None else int(sleep_t)+1
    pre_pred=bool(np.array_equal(m.out["probability"][:compare_end],src["D_keep"]["probability"][:compare_end]) and np.array_equal(m.out["detection_logits"][:compare_end],src["D_keep"]["detection_logits"][:compare_end]))
    pre_hash=True
    for ev in m.update_events:
        t=int(ev["at_interval"])
        if sleep_t is not None and t>=int(sleep_t): break
        k=src["keep_update_by_t"].get(t)
        if k is None or k.get("hash_after")!=ev.get("hash_after"):
            pre_hash=False; break
    no_sleep_exact=True
    if sleep_t is None:
        no_sleep_exact=bool(np.array_equal(m.out["probability"],src["D_keep"]["probability"]) and np.array_equal(m.out["detection_logits"],src["D_keep"]["detection_logits"]))
    sleep_forward_zero=not any(x["issued_state"]==SLEEPING for x in m.forward_events)
    sleep_step_zero=not any(x["issued_state_at_t"]==SLEEPING and x["post_transition_state"]==SLEEPING for x in m.update_events)
    limits=bool(sum(x["event"]=="birth" for x in m.lifecycle_events)<=1 and sum(x["event"]=="sleep" for x in m.lifecycle_events)<=1 and sum(x["event"]=="wake" for x in m.lifecycle_events)<=1)
    ledger_pass=bool(ledger["new_streams"]==ledger["baseline_retraining"]==ledger["other_real_streams"]==ledger["hyperparameter_sweeps"]==ledger["donor_updates"]==ledger["F_training"]==ledger["permanent_deletions"]==0 and ledger["real_engineering_prefixes"]==1 and ledger["real_engineering_prefix_steps_max"]==256 and ledger["real_engineering_gradient_steps"]==0)
    terminal=bool(m.terminal_actual is not None and all(int(v)==0 for v in m.terminal_actual.values()))
    birth_ok=bool(m.birth_t==351)
    step_budget=bool(m.optimizer_step_calls<=352)
    source_still=bool(verify_file_set(src["source036_root"],SOURCE036,"source036_after")==SOURCE036 and verify_file_set(src["source037_root"],SOURCE037,"source037_after")==SOURCE037)
    out={
        "source_hashes_unchanged":source_still,"cached_alignment_pass":True,"F_files_loaded":False,
        "birth_exact_t351":birth_ok,"pre_sleep_predictions_exact_D_keep":pre_pred,
        "pre_sleep_update_hash_chain_exact_D_keep":pre_hash,"no_sleep_implies_full_exact_D_keep":no_sleep_exact,
        "sleep_prediction_forward_calls_zero":sleep_forward_zero,"sleep_optimizer_steps_zero":sleep_step_zero,
        "sleep_expert_optimizer_preserved":bool(
            m.sleep_t is None or
            (m.wake_t is not None and m.sleep_expert_hash==m.wake_pre_expert_hash and m.sleep_optimizer_hash==m.wake_pre_optimizer_hash and m.sleep_optimizer_step==m.wake_pre_optimizer_step) or
            (m.wake_t is None and m.model_hash()==m.sleep_expert_hash and m.opt_hash()==m.sleep_optimizer_hash and optimizer_step_value(m.optimizer)==m.sleep_optimizer_step)
        ),
        "lifecycle_event_limits_pass":limits,"optimizer_step_budget_pass":step_budget,
        "actual_optimizer_steps":int(m.optimizer_step_calls),"actual_prediction_forward_calls":int(m.prediction_forward_calls),
        "terminal_actual_zero_calls":terminal,"fixture_all_pass":bool(fixture.get("all_pass")),
        "ledger_zero_budget_fields_pass":ledger_pass,
    }
    out["all_pass"]=bool(all([
        source_still,birth_ok,pre_pred,pre_hash,no_sleep_exact,sleep_forward_zero,sleep_step_zero,
        out["sleep_expert_optimizer_preserved"],limits,step_budget,terminal,fixture.get("all_pass"),ledger_pass
    ]))
    return out


def cmd_science(a):
    runtime(); validate_plan()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    src=load_sources(a.source036,a.source037)
    fixture=J(a.fixture_report)
    if not fixture.get("all_pass"): raise AssertionError("preflight fixture failed")
    shutil.copy2(a.fixture_report,out/"fixture_report.json")
    shutil.copy2(a.implementation_manifest,out/"implementation_manifest.json")
    source_lock={"protocol":"039","source036":src["hashes036"],"source037":src["hashes037"],"F_files_loaded":False,"verified":True}
    W(out/"source_lock.json",source_lock)
    ledger=out/"budget_ledger.json"
    start_sequence(ledger,a.run_id,fixture,a.resume_checkpoint)
    m=None
    try:
        identity=J(a.implementation_manifest)
        if a.resume_checkpoint:
            m,snap=restore_checkpoint(a.resume_checkpoint,src,out/"runtime_state",identity=identity,allow_gradient=True)
            if snap is None or not snap.get("sequence",{}).get("started"):
                raise AssertionError("resume checkpoint lacks started ledger snapshot")
            if m.cursor==0 and m.substep=="before_predict":
                raise RuntimeError("zero restart forbidden after science start")
        else:
            m=Machine(src,out/"runtime_state",identity=identity,allow_gradient=True)
        m.checkpoint_hook=lambda mm,reason,named=False: (sync_ledger_counts(ledger,mm),save_checkpoint(mm,out/"checkpoints",reason,ledger_path=ledger,named=named))[1]
        t0=time.perf_counter(); c0=time.process_time()
        m.advance()
        terminal=m.terminal_settle()
        sync_ledger_counts(ledger,m)
        save_checkpoint(m,out/"checkpoints","final_settled_5968",ledger_path=ledger,named=True)
        save_predictions(out,src,m)
        pred_sha=sha256_file(out/"D_sleepwake/predictions.npz")
        complete_ledger(ledger,m,pred_sha)
        led=J(ledger)
        audit=science_audit(src,m,fixture,led)
        W(out/"run_audit.json",audit)
        W(out/"D_sleepwake/summary.json",{
            "protocol":"039","arm":"D_sleepwake","completed":True,"birth_t":m.birth_t,"sleep_t":m.sleep_t,"wake_t":m.wake_t,
            "cycle_status":"cycle_observed" if m.sleep_t is not None and m.wake_t is not None else ("sleep_without_wake" if m.sleep_t is not None else "no_sleep_observed"),
            "prediction_forward_calls":int(m.prediction_forward_calls),"training_forward_calls":int(m.training_forward_calls),
            "optimizer_steps":int(m.optimizer_step_calls),"sleep_predictions":int(m.sleep_prediction_count),
            "lifecycle_check_calls":int(m.lifecycle_check_calls),"state_check_cpu_seconds":m.state_check_cpu_seconds,
            "prediction_forward_seconds":m.prediction_forward_seconds,"update_seconds":m.update_seconds,
            "resident_parameter_interval_integral":int(74*max(0,src["n"]-(m.birth_t+1 if m.birth_t is not None else src["n"]))),
            "final_expert_hash":m.model_hash(),"final_optimizer_hash":m.opt_hash(),"terminal_actual":terminal,
            "prediction_sha256":pred_sha,
        })
        # exact cached controls copied without retraining/recomputation.
        for arm,srcfile in (("C_ref",src["source036_root"]/"stage_B/seed3601/C_ref/predictions.npz"),
                            ("B_ref",src["source036_root"]/"stage_B/seed3601/D_lin/predictions.npz"),
                            ("D_keep",src["source037_root"]/"D_birth/predictions.npz")):
            d=out/arm; d.mkdir(exist_ok=True); shutil.copy2(srcfile,d/"predictions.npz")
            W(d/"summary.json",{"protocol":"039","arm":arm,"cached_control":True,"training_this_protocol":0,"prediction_sha256":sha256_file(d/"predictions.npz")})
        W(out/"science_cost_raw.json",{
            "protocol":"039","science_invocation_wall_seconds":time.perf_counter()-t0,"science_invocation_cpu_seconds":time.process_time()-c0,
            "incremental_expert":{"prediction_forward_calls":m.prediction_forward_calls,"training_forward_calls":m.training_forward_calls,"optimizer_steps":m.optimizer_step_calls,
                                  "prediction_forward_seconds":m.prediction_forward_seconds,"update_seconds":m.update_seconds,"sleep_predictions":m.sleep_prediction_count},
            "controller":{"lifecycle_check_calls":m.lifecycle_check_calls,"cpu_seconds":m.state_check_cpu_seconds},
            "historical_B_components":{"C_ref":src["historical_cost"]["C_ref"],"D_lin":src["historical_cost"]["D_lin"]},
            "historical_D_keep":src["historical_cost"]["D_keep"],
            "sleep_memory_note":"sleep preserves expert weights and Adam state; no expert-memory deletion is claimed",
            "deployment_note":"Complete D includes historical C + D_lin costs plus controller + incremental expert. Historical source-run timing is reported separately and is not summed into a same-run speed benchmark.",
        })
        W(out/"scientific_execution_complete.json",{"protocol":"039","run_id":str(a.run_id),"completed":True,"training_sequences_completed":1,"optimizer_steps":m.optimizer_step_calls,"audit_all_pass":audit["all_pass"]})
        W(out/"status.json",{
            "protocol":"039","run_id":str(a.run_id),"scientific_status":"completed","publication_status":"pending_until_main_sync",
            "training_sequences_registered":1,"training_sequences_completed":1,"new_streams":0,
            "birth_t":m.birth_t,"sleep_t":m.sleep_t,"wake_t":m.wake_t,"validity_all_pass":audit["all_pass"],
            "automatic_followup_training":False,"stop_after_registered_budget":True,
        })
        print(json.dumps(J(out/"status.json"),indent=2))
    except Exception as e:
        if m is not None:
            try:
                sync_ledger_counts(ledger,m)
                save_checkpoint(m,out/"checkpoints","interrupted",ledger_path=ledger,named=True)
            except Exception:
                pass
        if ledger.exists():
            append_ledger(ledger,{"event":"interrupted","sequence":"D_sleepwake","error":str(e)})
        W(out/"status.json",{"protocol":"039","run_id":str(a.run_id),"scientific_status":"interrupted","publication_status":"not_attempted","error":str(e),"automatic_followup_training":False,"stop_after_registered_budget":True})
        (out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf8")
        raise


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("preflight"); p.add_argument("--source036",required=True); p.add_argument("--source037",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_preflight)
    p=sub.add_parser("science"); p.add_argument("--source036",required=True); p.add_argument("--source037",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--run-id",required=True); p.add_argument("--fixture-report",required=True); p.add_argument("--implementation-manifest",required=True); p.add_argument("--resume-checkpoint"); p.set_defaults(fn=cmd_science)
    a=ap.parse_args(); a.fn(a)


if __name__=="__main__":
    main()
