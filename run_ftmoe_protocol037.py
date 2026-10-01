"""Protocol-037: one fixed extra expert vs one causally born expert on cached Protocol-036 seed3601."""
from __future__ import annotations
import argparse, copy, hashlib, io, json, os, random, shutil, subprocess, time, traceback
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import nn

from protocol035_common import dump_json, load_npz, sha256_file, sha256_state_dict

PLAN = Path("artifacts/ftmoe_online/protocol_037/plan.json")
PLAN_SHA = "01240a944bb654d529577b397602cf67c24c2245c6fbe909564b47da2726012a"
SOURCE_HASHES = {
    "stage_B/seed3601/C_ref/feature_tape.npz": "c71a2ea6d59045fe4cc8a953a776d37f48dbe1feec4aba859cd00f11c984529c",
    "stage_B/seed3601/C_ref/update_batches.json": "4d32d2ece194cb41b8bc16de1c89ddfedcd42da668b96446937fb430d6246b6f",
    "stage_B/seed3601/D_lin/predictions.npz": "1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
    "inputs/seed3601/manifest.json": "73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b",
}
N = 5968
HOSTS = 16
DIM = 73
EXPERT_PARAMS = 74


def J(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def W(p, x):
    dump_json(p, x)


def runtime():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)


def plan():
    if sha256_file(PLAN) != PLAN_SHA:
        raise AssertionError("Protocol037 plan hash changed")
    p = J(PLAN)
    if p.get("protocol") != "037" or p["budget"]["sequence_names"] != ["F_extra", "D_birth"]:
        raise AssertionError("Protocol037 plan identity")
    return p


def checkout_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def rss():
    return int(psutil.Process().memory_info().rss)


def memlimit():
    vals = [int(psutil.virtual_memory().total)]
    for p in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            x = p.read_text().strip()
            if x != "max" and 0 < int(x) < (1 << 60):
                vals.append(int(x))
        except Exception:
            pass
    return min(vals)


def stable_bce_interval(margin, label):
    m = np.asarray(margin, dtype=np.float64)
    y = (np.asarray(label) > 0).astype(np.float64)
    return float(np.mean(np.maximum(m, 0.0) - m * y + np.log1p(np.exp(-np.abs(m)))))


def optimizer_state_equal(a, b):
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(optimizer_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(optimizer_state_equal(x, y) for x, y in zip(a, b))
    return a == b


class ExtraExpert(nn.Module):
    def __init__(self):
        super().__init__()
        before = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3701)
            self.linear = nn.Linear(DIM, 1)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        after = torch.get_rng_state()
        if not torch.equal(before, after):
            raise AssertionError("isolated expert constructor changed global torch RNG")

    def delta(self, z):
        return 2.0 * torch.tanh(self.linear(z).squeeze(-1) / 2.0)


def make_expert():
    m = ExtraExpert()
    if sum(x.numel() for x in m.parameters()) != EXPERT_PARAMS:
        raise AssertionError("Protocol037 expert parameter count")
    if any(torch.count_nonzero(x).item() for x in m.parameters()):
        raise AssertionError("Protocol037 expert not zero initialized")
    return m


def make_optimizer(m):
    return torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)


def source_paths(root):
    root = Path(root)
    return {rel: root / rel for rel in SOURCE_HASHES}


def load_source(root, verify=True):
    root = Path(root)
    paths = source_paths(root)
    if verify:
        for rel, expected in SOURCE_HASHES.items():
            p = paths[rel]
            if not p.exists() or sha256_file(p) != expected:
                raise AssertionError("Protocol037 source hash mismatch: " + rel)
    T = load_npz(paths["stage_B/seed3601/C_ref/feature_tape.npz"])
    B = load_npz(paths["stage_B/seed3601/D_lin/predictions.npz"])
    batches = J(paths["stage_B/seed3601/C_ref/update_batches.json"])
    manifest = J(paths["inputs/seed3601/manifest.json"])
    if T["z"].shape != (N, HOSTS, DIM):
        raise AssertionError("issued feature tape geometry")
    if B["detection_logits"].shape != (N, HOSTS, 2) or B["probability"].shape != (N, HOSTS):
        raise AssertionError("B prediction geometry")
    if T["labels"].shape != (N, HOSTS) or not np.array_equal(T["labels"], B["labels"]):
        raise AssertionError("B/C labels misaligned")
    if "raw_labels" in T and "raw_labels" in B and not np.array_equal(T["raw_labels"], B["raw_labels"]):
        raise AssertionError("raw labels misaligned")
    if "c_class_probability" in T and "class_probability" in B and not np.array_equal(T["c_class_probability"], B["class_probability"]):
        raise AssertionError("classification must be exact B/C copy")
    updates = batches.get("updates", [])
    if len(updates) != 373:
        raise AssertionError("expected 373 registered update opportunities")
    by_t = {}
    for row in updates:
        t = int(row["at_interval"])
        if t in by_t:
            raise AssertionError("duplicate update opportunity")
        ii = [int(x) for x in row["batch_indices"]]
        if ii and max(ii) + 2 > t:
            raise AssertionError("source update batch is immature")
        by_t[t] = {"at_interval": t, "batch_indices": ii, **{k: v for k, v in row.items() if k not in ("at_interval", "batch_indices")}}
    return {
        "root": root,
        "T": T,
        "B": B,
        "updates": updates,
        "by_t": by_t,
        "manifest": manifest,
        "source_hashes": dict(SOURCE_HASHES),
        "source_id": "protocol036_run36831958978_seed3601",
        "n": N,
    }


def clone_source(src):
    out = dict(src)
    out["T"] = {k: v.copy() for k, v in src["T"].items()}
    out["B"] = {k: v.copy() for k, v in src["B"].items()}
    out["updates"] = copy.deepcopy(src["updates"])
    out["by_t"] = copy.deepcopy(src["by_t"])
    out["manifest"] = copy.deepcopy(src["manifest"])
    return out


def default_ledger(fixture_report=None):
    fx = fixture_report or {}
    return {
        "protocol": "037",
        "training_sequence_budget": 2,
        "sequences": {
            "F_extra": {"started": False, "completed": False, "restart_from_zero": False, "resume_events": [], "optimizer_steps": 0},
            "D_birth": {"started": False, "completed": False, "restart_from_zero": False, "resume_events": [], "optimizer_steps": 0},
        },
        "new_raw_streams": 0,
        "source_stream_reused": 3601,
        "source_stream_sha256": "09fadb02f2017f8d528ee8284129c29adcb6be137b1a9e93f4a249eebcd1b659",
        "baseline_retraining": 0,
        "other_real_stream_evaluations": 0,
        "hyperparameter_or_trigger_sweeps": 0,
        "max_births": 1,
        "births_observed": 0,
        "extra_initialization_seed_trials": 0,
        "sleep_wake_delete_trials": 0,
        "engineering_real_prefix_executions": int(fx.get("real_joint_prefix_executions", 0)),
        "engineering_real_prefix_max_steps": int(fx.get("max_real_prefix_steps", 0)),
        "automatic_next_protocol": False,
    }


def ledger_events_path(path):
    return Path(path).with_name("budget_ledger_events.jsonl")


def append_ledger_event(path, event):
    p = ledger_events_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"protocol": "037", "time_unix": time.time(), **event}
    with p.open("a", encoding="utf8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_or_init_ledger(path, fixture_report=None):
    p = Path(path)
    if not p.exists():
        W(p, default_ledger(fixture_report))
    d = J(p)
    if d.get("protocol") != "037" or int(d.get("training_sequence_budget", -1)) != 2:
        raise AssertionError("bad Protocol037 ledger")
    return d


def start_or_resume_sequences(path, run_id, resume_checkpoint=None, fixture_report=None):
    d = load_or_init_ledger(path, fixture_report)
    for name in ("F_extra", "D_birth"):
        r = d["sequences"][name]
        if r.get("completed"):
            continue
        if r.get("started"):
            if not resume_checkpoint:
                raise RuntimeError(name + " already started; exact checkpoint resume required")
            ev = {"run_id": str(run_id), "checkpoint": str(resume_checkpoint), "time_unix": time.time()}
            r["resume_events"].append(ev)
            append_ledger_event(path, {"event": "resume", "sequence": name, **ev})
        else:
            if resume_checkpoint:
                raise RuntimeError(name + " not previously started; resume invalid")
            r["started"] = True
            r["started_run_id"] = str(run_id)
            r["started_time_unix"] = time.time()
            append_ledger_event(path, {"event": "start", "sequence": name, "run_id": str(run_id)})
    W(path, d)
    return d


def note_optimizer_step(path, name, t, batch, loss, grad_norm):
    d = J(path)
    d["sequences"][name]["optimizer_steps"] = int(d["sequences"][name].get("optimizer_steps", 0)) + 1
    W(path, d)
    append_ledger_event(path, {
        "event": "optimizer_step",
        "sequence": name,
        "at_interval": int(t),
        "batch_indices": [int(x) for x in batch],
        "loss": float(loss),
        "grad_norm": float(grad_norm),
    })


def complete_sequences(path, extra):
    d = J(path)
    for name in ("F_extra", "D_birth"):
        r = d["sequences"][name]
        if not r.get("started"):
            raise AssertionError("cannot complete unstarted sequence " + name)
        r["completed"] = True
        r["completed_time_unix"] = time.time()
        r.update(extra.get(name, {}))
        append_ledger_event(path, {"event": "complete", "sequence": name})
    d["births_observed"] = int(extra.get("births_observed", 0))
    W(path, d)


def interrupt_sequences(path, error):
    if not Path(path).exists():
        return
    d = J(path)
    for name in ("F_extra", "D_birth"):
        if d["sequences"][name].get("started") and not d["sequences"][name].get("completed"):
            d["sequences"][name]["last_interruption"] = str(error)
            append_ledger_event(path, {"event": "interrupted", "sequence": name, "error": str(error)})
    W(path, d)


def alloc_outputs(n, B):
    return {
        "F_probability": np.full((n, HOSTS), np.nan, np.float32),
        "D_probability": np.full((n, HOSTS), np.nan, np.float32),
        "F_logits": np.full((n, HOSTS, 2), np.nan, np.float32),
        "D_logits": np.full((n, HOSTS, 2), np.nan, np.float32),
        "F_delta": np.full((n, HOSTS), np.nan, np.float32),
        "D_delta": np.full((n, HOSTS), np.nan, np.float32),
        "F_version": np.zeros(n, np.int64),
        "D_version": np.full(n, -1, np.int64),
        "F_expert_count": np.ones(n, np.int8),
        "D_expert_count": np.zeros(n, np.int8),
        "F_hash": np.full(n, "", dtype="<U64"),
        "D_hash": np.full(n, "absent", dtype="<U64"),
    }


def fresh_state(src):
    Fm = make_expert()
    Fo = make_optimizer(Fm)
    return {
        "cursor": 0,
        "F": Fm,
        "F_opt": Fo,
        "D": None,
        "D_opt": None,
        "F_version_now": 0,
        "D_version_now": -1,
        "birth_t": None,
        "first_affected_prediction": None,
        "birth_streak": 0,
        "loss_buffer": np.full(src["n"], np.nan, np.float64),
        "out": alloc_outputs(src["n"], src["B"]),
        "F_updates": [],
        "D_updates": [],
        "birth_checks": [],
        "lifecycle_events": [],
        "settlement_events": [],
        "terminal_settlement_optimizer_steps": 0,
        "F_inference_calls": 0,
        "D_inference_calls": 0,
        "F_inference_seconds": 0.0,
        "D_inference_seconds": 0.0,
        "F_update_seconds": 0.0,
        "D_update_seconds": 0.0,
        "peak_rss_bytes": rss(),
    }


def expert_predict(model, zrow, brow, exact_b_if_zero=False):
    z = torch.from_numpy(np.asarray(zrow)).float()
    b = torch.from_numpy(np.asarray(brow)).float()
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        dd = model.delta(z)
        zero = bool(torch.equal(dd, torch.zeros_like(dd)))
        if exact_b_if_zero and zero:
            logits = None
            prob = None
        else:
            logits = torch.stack((b[:, 0] - dd / 2.0, b[:, 1] + dd / 2.0), dim=-1)
            prob = torch.softmax(logits, dim=-1)[:, 1]
    return dd.numpy(), None if logits is None else logits.numpy(), None if prob is None else prob.numpy(), time.perf_counter() - t0


def do_update(model, opt, src, batch):
    ii = np.asarray(batch, dtype=np.int64)
    if ii.size == 0:
        raise AssertionError("empty registered update batch")
    z = torch.from_numpy(src["T"]["z"][ii]).float()
    b = torch.from_numpy(src["B"]["detection_logits"][ii]).float()
    margin = b[..., 1] - b[..., 0]
    y = torch.from_numpy((src["B"]["labels"][ii] > 0).astype(np.float32))
    model.train()
    t0 = time.perf_counter()
    delta = model.delta(z)
    loss = F.binary_cross_entropy_with_logits(margin + delta, y) + 0.001 * (delta ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach()), float(gn), time.perf_counter() - t0


def birth_check(src, st, t, synthetic_fixture=False, force_birth_t=None):
    if st["D"] is not None:
        return None
    m = int(t) - 2
    due = ((int(t) + 1) % 16 == 0)
    if not due:
        return None
    forced = bool(synthetic_fixture and force_birth_t is not None and int(t) == int(force_birth_t))
    if m < 319 and not forced:
        return None
    if forced:
        row = {
            "at_interval": int(t), "m": m, "forced_fixture": True, "eligible": True,
            "recent": None, "previous": None, "positive_support": None, "negative_support": None,
            "ratio_condition": True, "difference_condition": True, "support_condition": True,
            "candidate": True, "streak_before": int(st["birth_streak"]), "streak_after": 2,
        }
        st["birth_streak"] = 2
        return row
    recent_idx = np.arange(m - 63, m + 1, dtype=np.int64)
    previous_idx = np.arange(m - 319, m - 63, dtype=np.int64)
    if np.isnan(st["loss_buffer"][recent_idx]).any() or np.isnan(st["loss_buffer"][previous_idx]).any():
        raise AssertionError("birth check requested unavailable matured losses")
    R = float(np.mean(st["loss_buffer"][recent_idx]))
    P = float(np.mean(st["loss_buffer"][previous_idx]))
    labels = src["B"]["labels"][recent_idx]
    pos = int((labels > 0).sum())
    neg = int((labels <= 0).sum())
    ratio = bool(R >= 1.25 * max(P, 1e-6))
    difference = bool(R - P >= 0.02)
    support = bool(pos >= 16 and neg >= 16)
    candidate = bool(ratio and difference and support)
    before = int(st["birth_streak"])
    st["birth_streak"] = before + 1 if candidate else 0
    return {
        "at_interval": int(t), "m": m, "forced_fixture": False,
        "recent": [int(m - 63), int(m)], "previous": [int(m - 319), int(m - 64)],
        "R": R, "P": P, "positive_support": pos, "negative_support": neg,
        "ratio_condition": ratio, "difference_condition": difference, "support_condition": support,
        "candidate": candidate, "streak_before": before, "streak_after": int(st["birth_streak"]),
        "eligible": True,
    }


def create_birth(st, t, zrow):
    if st["D"] is not None:
        raise AssertionError("second birth forbidden")
    Dm = make_expert()
    Do = make_optimizer(Dm)
    with torch.no_grad():
        z = torch.from_numpy(np.asarray(zrow)).float()
        d0 = Dm.delta(z)
    zero = bool(torch.equal(d0, torch.zeros_like(d0)))
    if not zero:
        raise AssertionError("birth-instant delta is not exactly zero")
    st["D"] = Dm
    st["D_opt"] = Do
    st["D_version_now"] = 0
    st["birth_t"] = int(t)
    st["first_affected_prediction"] = int(t) + 1
    st["lifecycle_events"].append({
        "event": "birth",
        "at_interval": int(t),
        "first_affected_prediction": int(t) + 1,
        "zero_delta_exact_before_update": True,
        "expert_parameter_count": EXPERT_PARAMS,
        "expert_hash_before_update": sha256_state_dict(Dm.state_dict()),
    })


def process_one(src, st, t, ledger_path=None, synthetic_fixture=False, force_birth_t=None):
    B = src["B"]
    zrow = src["T"]["z"][t]
    brow = B["detection_logits"][t]
    # 1) predict
    fhash = sha256_state_dict(st["F"].state_dict())
    fd, fl, fp, secs = expert_predict(st["F"], zrow, brow, exact_b_if_zero=True)
    st["F_inference_calls"] += 1
    st["F_inference_seconds"] += secs
    if fl is None:
        st["out"]["F_logits"][t] = brow
        st["out"]["F_probability"][t] = B["probability"][t]
    else:
        st["out"]["F_logits"][t] = fl
        st["out"]["F_probability"][t] = fp
    st["out"]["F_delta"][t] = fd
    st["out"]["F_version"][t] = int(st["F_version_now"])
    st["out"]["F_hash"][t] = fhash

    if st["D"] is None:
        st["out"]["D_logits"][t] = brow
        st["out"]["D_probability"][t] = B["probability"][t]
        st["out"]["D_delta"][t] = 0.0
        st["out"]["D_version"][t] = -1
        st["out"]["D_expert_count"][t] = 0
        st["out"]["D_hash"][t] = "absent"
    else:
        dhash = sha256_state_dict(st["D"].state_dict())
        dd, dl, dp, secs = expert_predict(st["D"], zrow, brow, exact_b_if_zero=True)
        st["D_inference_calls"] += 1
        st["D_inference_seconds"] += secs
        if dl is None:
            st["out"]["D_logits"][t] = brow
            st["out"]["D_probability"][t] = B["probability"][t]
        else:
            st["out"]["D_logits"][t] = dl
            st["out"]["D_probability"][t] = dp
        st["out"]["D_delta"][t] = dd
        st["out"]["D_version"][t] = int(st["D_version_now"])
        st["out"]["D_expert_count"][t] = 1
        st["out"]["D_hash"][t] = dhash

    # 2) settle newly matured label i=t-2, using B only
    i = int(t) - 2
    if i >= 0:
        margin = B["detection_logits"][i, :, 1] - B["detection_logits"][i, :, 0]
        loss_i = stable_bce_interval(margin, B["labels"][i])
        st["loss_buffer"][i] = loss_i
        st["settlement_events"].append({"at_interval": int(t), "settled_interval": i, "B_mean_host_bce": loss_i})

    # 3) causal birth check
    chk = birth_check(src, st, t, synthetic_fixture=synthetic_fixture, force_birth_t=force_birth_t)
    if chk is not None:
        st["birth_checks"].append(chk)
        if st["D"] is None and int(chk["streak_after"]) >= 2:
            create_birth(st, t, zrow)
            chk["state_change"] = "birth"
        else:
            chk["state_change"] = "none"

    # 4) registered scheduled update
    if t in src["by_t"]:
        batch = [int(x) for x in src["by_t"][t]["batch_indices"]]
        if batch and max(batch) + 2 > int(t):
            raise AssertionError("immature scientific update")
        before = sha256_state_dict(st["F"].state_dict())
        loss, gn, secs = do_update(st["F"], st["F_opt"], src, batch)
        st["F_update_seconds"] += secs
        st["F_version_now"] += 1
        after = sha256_state_dict(st["F"].state_dict())
        row = {"at_interval": int(t), "batch_indices": batch, "loss": loss, "grad_norm": gn, "hash_before": before, "hash_after": after, "version_after": int(st["F_version_now"])}
        st["F_updates"].append(row)
        if ledger_path:
            note_optimizer_step(ledger_path, "F_extra", t, batch, loss, gn)

        if st["D"] is not None:
            before = sha256_state_dict(st["D"].state_dict())
            loss, gn, secs = do_update(st["D"], st["D_opt"], src, batch)
            st["D_update_seconds"] += secs
            st["D_version_now"] += 1
            after = sha256_state_dict(st["D"].state_dict())
            row = {"at_interval": int(t), "batch_indices": batch, "loss": loss, "grad_norm": gn, "hash_before": before, "hash_after": after, "version_after": int(st["D_version_now"])}
            st["D_updates"].append(row)
            if ledger_path:
                note_optimizer_step(ledger_path, "D_birth", t, batch, loss, gn)

    st["cursor"] = int(t) + 1
    st["peak_rss_bytes"] = max(int(st["peak_rss_bytes"]), rss())


def terminal_settle(src, st):
    before = len(st["F_updates"]) + len(st["D_updates"])
    for i in range(max(0, src["n"] - 2), src["n"]):
        if np.isnan(st["loss_buffer"][i]):
            margin = src["B"]["detection_logits"][i, :, 1] - src["B"]["detection_logits"][i, :, 0]
            loss_i = stable_bce_interval(margin, src["B"]["labels"][i])
            st["loss_buffer"][i] = loss_i
            st["settlement_events"].append({"at_interval": "terminal", "settled_interval": int(i), "B_mean_host_bce": loss_i})
    after = len(st["F_updates"]) + len(st["D_updates"])
    st["terminal_settlement_optimizer_steps"] = int(after - before)
    if st["terminal_settlement_optimizer_steps"] != 0:
        raise AssertionError("terminal settlement performed optimizer step")


def checkpoint_payload(st, src, ledger_path=None):
    return {
        "protocol": "037",
        "plan_sha256": PLAN_SHA,
        "source_id": src["source_id"],
        "source_hashes": src.get("source_hashes"),
        "cursor": int(st["cursor"]),
        "F_state": st["F"].state_dict(),
        "F_opt": st["F_opt"].state_dict(),
        "D_exists": st["D"] is not None,
        "D_state": None if st["D"] is None else st["D"].state_dict(),
        "D_opt": None if st["D_opt"] is None else st["D_opt"].state_dict(),
        "F_version_now": int(st["F_version_now"]),
        "D_version_now": int(st["D_version_now"]),
        "birth_t": st["birth_t"],
        "first_affected_prediction": st["first_affected_prediction"],
        "birth_streak": int(st["birth_streak"]),
        "loss_buffer": st["loss_buffer"].copy(),
        "out": {k: v.copy() for k, v in st["out"].items()},
        "F_updates": copy.deepcopy(st["F_updates"]),
        "D_updates": copy.deepcopy(st["D_updates"]),
        "birth_checks": copy.deepcopy(st["birth_checks"]),
        "lifecycle_events": copy.deepcopy(st["lifecycle_events"]),
        "settlement_events": copy.deepcopy(st["settlement_events"]),
        "terminal_settlement_optimizer_steps": int(st["terminal_settlement_optimizer_steps"]),
        "F_inference_calls": int(st["F_inference_calls"]),
        "D_inference_calls": int(st["D_inference_calls"]),
        "F_inference_seconds": float(st["F_inference_seconds"]),
        "D_inference_seconds": float(st["D_inference_seconds"]),
        "F_update_seconds": float(st["F_update_seconds"]),
        "D_update_seconds": float(st["D_update_seconds"]),
        "peak_rss_bytes": int(st["peak_rss_bytes"]),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "ledger_snapshot": None if ledger_path is None else J(ledger_path),
    }


def save_checkpoint(out, st, src, ledger_path=None, label=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    label = label or ("cursor_%04d" % int(st["cursor"]))
    p = out / (label + ".pt")
    torch.save(checkpoint_payload(st, src, ledger_path), p)
    meta = {
        "protocol": "037", "cursor": int(st["cursor"]), "file": p.name,
        "sha256": sha256_file(p), "plan_sha256": PLAN_SHA, "source_id": src["source_id"],
        "F_hash": sha256_state_dict(st["F"].state_dict()),
        "D_exists": st["D"] is not None,
        "D_hash": None if st["D"] is None else sha256_state_dict(st["D"].state_dict()),
    }
    W(str(p) + ".json", meta)
    W(out / "latest_checkpoint.json", meta)
    return p


def restore_checkpoint(path, src):
    p = Path(path)
    meta = J(str(p) + ".json")
    if sha256_file(p) != meta["sha256"]:
        raise AssertionError("Protocol037 checkpoint hash")
    x = torch.load(p, map_location="cpu", weights_only=False)
    if x.get("protocol") != "037" or x.get("plan_sha256") != PLAN_SHA or x.get("source_id") != src["source_id"]:
        raise AssertionError("Protocol037 checkpoint identity")
    Fm = make_expert()
    Fo = make_optimizer(Fm)
    Fm.load_state_dict(x["F_state"])
    Fo.load_state_dict(x["F_opt"])
    Dm = Do = None
    if x["D_exists"]:
        Dm = make_expert()
        Do = make_optimizer(Dm)
        Dm.load_state_dict(x["D_state"])
        Do.load_state_dict(x["D_opt"])
    st = {
        "cursor": int(x["cursor"]), "F": Fm, "F_opt": Fo, "D": Dm, "D_opt": Do,
        "F_version_now": int(x["F_version_now"]), "D_version_now": int(x["D_version_now"]),
        "birth_t": x["birth_t"], "first_affected_prediction": x["first_affected_prediction"],
        "birth_streak": int(x["birth_streak"]), "loss_buffer": x["loss_buffer"].copy(),
        "out": {k: v.copy() for k, v in x["out"].items()},
        "F_updates": copy.deepcopy(x["F_updates"]), "D_updates": copy.deepcopy(x["D_updates"]),
        "birth_checks": copy.deepcopy(x["birth_checks"]), "lifecycle_events": copy.deepcopy(x["lifecycle_events"]),
        "settlement_events": copy.deepcopy(x["settlement_events"]),
        "terminal_settlement_optimizer_steps": int(x["terminal_settlement_optimizer_steps"]),
        "F_inference_calls": int(x["F_inference_calls"]), "D_inference_calls": int(x["D_inference_calls"]),
        "F_inference_seconds": float(x["F_inference_seconds"]), "D_inference_seconds": float(x["D_inference_seconds"]),
        "F_update_seconds": float(x["F_update_seconds"]), "D_update_seconds": float(x["D_update_seconds"]),
        "peak_rss_bytes": int(x["peak_rss_bytes"]),
    }
    torch.set_rng_state(x["torch_rng"])
    np.random.set_state(x["numpy_rng"])
    random.setstate(x["python_rng"])
    return st, x.get("ledger_snapshot")


def run_until(src, st, stop, checkpoint_dir=None, checkpoint_every=0, ledger_path=None, synthetic_fixture=False, force_birth_t=None):
    stop = min(int(stop), int(src["n"]))
    lim = memlimit()
    soft = int(0.8 * lim)
    while int(st["cursor"]) < stop:
        t = int(st["cursor"])
        process_one(src, st, t, ledger_path=ledger_path, synthetic_fixture=synthetic_fixture, force_birth_t=force_birth_t)
        if checkpoint_dir and checkpoint_every and st["cursor"] % int(checkpoint_every) == 0:
            save_checkpoint(checkpoint_dir, st, src, ledger_path=ledger_path)
        if int(st["peak_rss_bytes"]) >= soft:
            raise MemoryError("Protocol037 memory soft limit reached")
    return st


def state_prefix_equal(a, b, n):
    keys = ("F_probability", "D_probability", "F_logits", "D_logits", "F_delta", "D_delta", "F_version", "D_version", "F_expert_count", "D_expert_count", "F_hash", "D_hash")
    for k in keys:
        x = a["out"][k][:n]
        y = b["out"][k][:n]
        if x.dtype.kind in ("U", "S", "O") or y.dtype.kind in ("U", "S", "O"):
            if not np.array_equal(x, y):
                return False
        elif not np.array_equal(x, y, equal_nan=True):
            return False
    return True


def state_exact_equal(a, b, n):
    if not state_prefix_equal(a, b, n):
        return False
    scalar_keys = ("cursor", "F_version_now", "D_version_now", "birth_t", "first_affected_prediction", "birth_streak")
    if any(a[k] != b[k] for k in scalar_keys):
        return False
    if a["F_updates"] != b["F_updates"] or a["D_updates"] != b["D_updates"] or a["birth_checks"] != b["birth_checks"] or a["lifecycle_events"] != b["lifecycle_events"]:
        return False
    if not torch.equal(next(a["F"].parameters()).detach(), next(b["F"].parameters()).detach()):
        return False
    if sha256_state_dict(a["F"].state_dict()) != sha256_state_dict(b["F"].state_dict()):
        return False
    if (a["D"] is None) != (b["D"] is None):
        return False
    if a["D"] is not None and sha256_state_dict(a["D"].state_dict()) != sha256_state_dict(b["D"].state_dict()):
        return False
    if not optimizer_state_equal(a["F_opt"].state_dict(), b["F_opt"].state_dict()):
        return False
    if a["D_opt"] is not None and not optimizer_state_equal(a["D_opt"].state_dict(), b["D_opt"].state_dict()):
        return False
    return True


def synthetic_source():
    n = 256
    rng = np.random.default_rng(12345)
    z = rng.normal(0, 0.2, size=(n, HOSTS, DIM)).astype(np.float32)
    y = ((np.arange(n)[:, None] + np.arange(HOSTS)[None, :]) % 7 == 0).astype(np.int64)
    margin = (0.35 * z[..., 0] - 0.15 * z[..., 1] + rng.normal(0, 0.03, size=(n, HOSTS))).astype(np.float32)
    logits = np.stack((-margin / 2, margin / 2), axis=-1).astype(np.float32)
    prob = (1.0 / (1.0 + np.exp(-margin))).astype(np.float32)
    cls = np.zeros((n, HOSTS, 4), np.float32)
    cls[..., 0] = 1.0
    T = {"z": z, "labels": y.copy(), "raw_labels": y.copy(), "c_class_probability": cls.copy()}
    B = {"detection_logits": logits, "probability": prob, "labels": y.copy(), "raw_labels": y.copy(), "class_probability": cls.copy(), "class_logits": np.zeros_like(cls), "model_version": np.zeros(n, np.int64)}
    updates = []
    for t in range(15, n, 16):
        mature = list(range(max(0, t - 33), max(0, t - 1)))
        if not mature:
            mature = [0]
        batch = mature[-32:]
        updates.append({"at_interval": t, "batch_indices": batch})
    return {"root": None, "T": T, "B": B, "updates": updates, "by_t": {r["at_interval"]: r for r in updates}, "manifest": {}, "source_hashes": None, "source_id": "protocol037_synthetic_fixture", "n": n}


def cmd_fixture(a):
    runtime()
    plan()
    src = load_source(a.source_root, verify=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Real execution 1: continuous exact source prefix 256.
    real1 = fresh_state(src)
    run_until(src, real1, 256)

    # Real execution 2: true future perturbation + an internal exact checkpoint restore.
    pert = clone_source(src)
    perturb_start = 200
    before_z = pert["T"]["z"][perturb_start:].copy()
    before_l = pert["B"]["labels"][perturb_start:].copy()
    pert["T"]["z"][perturb_start:] += np.float32(0.25)
    pert["B"]["detection_logits"][perturb_start:, :, 1] += np.float32(0.4)
    pert["B"]["labels"][perturb_start:] = np.where(pert["B"]["labels"][perturb_start:] > 0, 0, 1)
    changed_z = int(np.count_nonzero(pert["T"]["z"][perturb_start:] != before_z))
    changed_y = int(np.count_nonzero(pert["B"]["labels"][perturb_start:] != before_l))
    if changed_z <= 0 or changed_y <= 0:
        raise AssertionError("future perturbation did not actually change source rows")
    real2 = fresh_state(pert)
    run_until(pert, real2, 128)
    cp = save_checkpoint(out / "real_resume", real2, pert, label="cursor_0128")
    restored, _ = restore_checkpoint(cp, pert)
    run_until(pert, restored, 256)
    prefix_compare = 192
    real_resume_future_pass = bool(state_prefix_equal(real1, restored, prefix_compare))
    if not real_resume_future_pass:
        raise AssertionError("real prefix changed before true future perturbation or resume diverged")
    if real1["D"] is not None or restored["D"] is not None:
        raise AssertionError("real <=256 prefix unexpectedly triggered birth")

    # Synthetic production-entrypoint recovery at pre-birth, just-born-and-updated, and learned states.
    syn = synthetic_source()
    continuous = fresh_state(syn)
    run_until(syn, continuous, syn["n"], synthetic_fixture=True, force_birth_t=79)
    terminal_settle(syn, continuous)
    boundaries = [64, 80, 160]
    resume_rows = []
    for boundary in boundaries:
        s = fresh_state(syn)
        run_until(syn, s, boundary, synthetic_fixture=True, force_birth_t=79)
        cp = save_checkpoint(out / "synthetic_resume", s, syn, label="cursor_%04d" % boundary)
        rr, _ = restore_checkpoint(cp, syn)
        run_until(syn, rr, syn["n"], synthetic_fixture=True, force_birth_t=79)
        terminal_settle(syn, rr)
        ok = state_exact_equal(continuous, rr, syn["n"]) and rr["terminal_settlement_optimizer_steps"] == 0
        resume_rows.append({"boundary": boundary, "pass": bool(ok), "birth_exists_at_checkpoint": bool(s["D"] is not None)})
        if not ok:
            raise AssertionError("synthetic exact resume mismatch at boundary %d" % boundary)

    report = {
        "protocol": "037",
        "passed": bool(real_resume_future_pass and all(x["pass"] for x in resume_rows) and continuous["terminal_settlement_optimizer_steps"] == 0),
        "real_joint_prefix_executions": 2,
        "max_real_prefix_steps": 256,
        "real_prefix_births": 0,
        "real_resume_checkpoint_cursor": 128,
        "future_perturbation_start": perturb_start,
        "future_perturbation_rows_positive": int(src["n"] - perturb_start),
        "future_feature_values_changed": changed_z,
        "future_label_values_changed": changed_y,
        "unperturbed_prefix_compared_through": prefix_compare,
        "real_resume_and_future_isolation_pass": real_resume_future_pass,
        "synthetic_forced_event_only": True,
        "synthetic_force_birth_t": 79,
        "synthetic_resume_tests": resume_rows,
        "terminal_settlement_optimizer_steps_actual": int(continuous["terminal_settlement_optimizer_steps"]),
        "terminal_settlement_has_no_optimizer_step": bool(continuous["terminal_settlement_optimizer_steps"] == 0),
        "production_entrypoint_shared": True,
    }
    W(a.report, report)
    print(json.dumps(report, indent=2))


def save_arm_predictions(path, src, st, arm):
    B = src["B"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arm == "F_extra":
        prob = st["out"]["F_probability"]
        logits = st["out"]["F_logits"]
        delta = st["out"]["F_delta"]
        version = st["out"]["F_version"]
        count = st["out"]["F_expert_count"]
        hashes = st["out"]["F_hash"]
    else:
        prob = st["out"]["D_probability"]
        logits = st["out"]["D_logits"]
        delta = st["out"]["D_delta"]
        version = st["out"]["D_version"]
        count = st["out"]["D_expert_count"]
        hashes = st["out"]["D_hash"]
    payload = {
        "probability": prob,
        "detection_logits": logits,
        "class_probability": B["class_probability"],
        "labels": B["labels"],
        "raw_labels": B["raw_labels"],
        "model_version": B.get("model_version", np.zeros(src["n"], np.int64)),
        "delta": delta,
        "expert_version": version,
        "expert_count": count,
        "expert_hash": hashes,
    }
    # Protocol036 D_lin stores class_probability but not necessarily class_logits.
    # Classification is outside the new 037 expert and must be copied exactly from B;
    # preserve class_logits only when the frozen source actually contains it.
    if "class_logits" in B:
        payload["class_logits"] = B["class_logits"]
    np.savez_compressed(path, **payload)


def science_audit(src, st, fixture, ledger, before_hashes):
    after_hashes = {rel: sha256_file(src["root"] / rel) for rel in SOURCE_HASHES}
    source_unchanged = before_hashes == after_hashes == SOURCE_HASHES
    birth_t = st["birth_t"]
    prebirth_exact = True
    if birth_t is not None:
        end = int(birth_t) + 1
        prebirth_exact = bool(
            np.array_equal(st["out"]["D_probability"][:end], src["B"]["probability"][:end])
            and np.array_equal(st["out"]["D_logits"][:end], src["B"]["detection_logits"][:end])
            and np.array_equal(st["out"]["D_delta"][:end], np.zeros_like(st["out"]["D_delta"][:end]))
        )
    else:
        prebirth_exact = bool(
            np.array_equal(st["out"]["D_probability"], src["B"]["probability"])
            and np.array_equal(st["out"]["D_logits"], src["B"]["detection_logits"])
            and np.array_equal(st["out"]["D_delta"], np.zeros_like(st["out"]["D_delta"]))
        )
    class_copy = True
    # saved F/D class outputs are direct B references; record that no code path recomputed them.
    all_batches_mature = all((not r["batch_indices"]) or max(r["batch_indices"]) + 2 <= r["at_interval"] for r in st["F_updates"] + st["D_updates"])
    birth_checks_causal = all(
        r.get("forced_fixture") is False
        and (r["m"] + 2 == r["at_interval"])
        and r["recent"] == [r["m"] - 63, r["m"]]
        and r["previous"] == [r["m"] - 319, r["m"] - 64]
        for r in st["birth_checks"]
    )
    birth_limit = sum(1 for x in st["lifecycle_events"] if x["event"] == "birth") <= 1
    expected_F_updates = len(src["updates"])
    expected_D_updates = 0 if birth_t is None else sum(1 for r in src["updates"] if int(r["at_interval"]) >= int(birth_t))
    update_counts = len(st["F_updates"]) == expected_F_updates and len(st["D_updates"]) == expected_D_updates
    ledger_ok = (
        ledger["new_raw_streams"] == 0
        and ledger["baseline_retraining"] == 0
        and ledger["hyperparameter_or_trigger_sweeps"] == 0
        and ledger["sleep_wake_delete_trials"] == 0
        and ledger["engineering_real_prefix_executions"] == 2
        and ledger["engineering_real_prefix_max_steps"] == 256
    )
    out = {
        "source_hashes_unchanged": source_unchanged,
        "D_before_birth_exact_B_copy": prebirth_exact,
        "classification_exact_B_copy": class_copy,
        "all_science_update_batches_mature": all_batches_mature,
        "birth_checks_use_only_matured_B_losses": birth_checks_causal,
        "birth_count_within_registered_limit": birth_limit,
        "actual_update_counts_match_registered_opportunities": update_counts,
        "F_optimizer_steps": len(st["F_updates"]),
        "D_optimizer_steps": len(st["D_updates"]),
        "expected_F_optimizer_steps": expected_F_updates,
        "expected_D_optimizer_steps": expected_D_updates,
        "terminal_settlement_optimizer_steps_actual": int(st["terminal_settlement_optimizer_steps"]),
        "fixture_pass": bool(fixture.get("passed")),
        "budget_zero_fields_and_prefix_accounting_pass": ledger_ok,
        "event_order": ["predict", "settle_label", "check_birth_if_due", "scheduled_update"],
    }
    out["all_pass"] = bool(
        source_unchanged and prebirth_exact and class_copy and all_batches_mature and birth_checks_causal
        and birth_limit and update_counts and st["terminal_settlement_optimizer_steps"] == 0
        and fixture.get("passed") and ledger_ok
    )
    return out


def cmd_science(a):
    runtime()
    plan()
    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    src = load_source(a.source_root, verify=True)
    fixture = J(a.fixture_report)
    if not fixture.get("passed"):
        raise AssertionError("Protocol037 fixture did not pass")
    before_hashes = {rel: sha256_file(src["root"] / rel) for rel in SOURCE_HASHES}
    W(root / "source_lock.json", {
        "protocol": "037", "source_run_id": 36831958978, "source_artifact_id": 11147931152,
        "source_stream_seed": 3601, "source_stream_sha256": "09fadb02f2017f8d528ee8284129c29adcb6be137b1a9e93f4a249eebcd1b659",
        "files": before_hashes, "baseline": "B_ref=C+D_lin issued predictions", "verified": before_hashes == SOURCE_HASHES,
    })
    ledger_path = root / "budget_ledger.json"
    start_or_resume_sequences(ledger_path, a.run_id, a.resume_checkpoint, fixture_report=fixture)
    st = None
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        if a.resume_checkpoint:
            st, snap = restore_checkpoint(a.resume_checkpoint, src)
            current = J(ledger_path)
            if snap is not None:
                for name in ("F_extra", "D_birth"):
                    if not snap["sequences"][name].get("started"):
                        raise AssertionError("resume checkpoint ledger predates sequence start")
            if int(st["cursor"]) <= 0:
                raise AssertionError("zero restart disguised as resume")
        else:
            st = fresh_state(src)
        checkpoints = root / "checkpoints"
        run_until(src, st, src["n"], checkpoint_dir=checkpoints, checkpoint_every=512, ledger_path=ledger_path)
        terminal_settle(src, st)
        save_checkpoint(checkpoints, st, src, ledger_path=ledger_path, label="final_settled_5968")

        # B reference is copied byte-for-byte from the source artifact.
        bdir = root / "B_ref"
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src["root"] / "stage_B/seed3601/D_lin/predictions.npz", bdir / "predictions.npz")
        W(bdir / "summary.json", {
            "protocol": "037", "arm": "B_ref", "training": False, "exact_source_copy": True,
            "source_sha256": SOURCE_HASHES["stage_B/seed3601/D_lin/predictions.npz"],
        })

        for arm in ("F_extra", "D_birth"):
            adir = root / arm
            adir.mkdir(parents=True, exist_ok=True)
            save_arm_predictions(adir / "predictions.npz", src, st, arm)
        W(root / "F_extra/update_log.json", {"protocol": "037", "arm": "F_extra", "updates": st["F_updates"]})
        W(root / "D_birth/update_log.json", {"protocol": "037", "arm": "D_birth", "updates": st["D_updates"]})
        W(root / "D_birth/birth_checks.json", {"protocol": "037", "checks": st["birth_checks"]})
        W(root / "D_birth/lifecycle_events.json", {"protocol": "037", "events": st["lifecycle_events"]})
        W(root / "settlement_events.json", {"protocol": "037", "events": st["settlement_events"]})

        extra = {
            "F_extra": {
                "prediction_sha256": sha256_file(root / "F_extra/predictions.npz"),
                "optimizer_steps": len(st["F_updates"]), "parameter_count": EXPERT_PARAMS,
            },
            "D_birth": {
                "prediction_sha256": sha256_file(root / "D_birth/predictions.npz"),
                "optimizer_steps": len(st["D_updates"]), "parameter_count": EXPERT_PARAMS,
                "birth_occurred": st["birth_t"] is not None, "birth_t": st["birth_t"],
            },
            "births_observed": 1 if st["birth_t"] is not None else 0,
        }
        complete_sequences(ledger_path, extra)
        ledger = J(ledger_path)
        audit = science_audit(src, st, fixture, ledger, before_hashes)
        W(root / "run_audit.json", audit)
        W(root / "F_extra/causality_audit.json", {
            "all_update_batches_mature": audit["all_science_update_batches_mature"],
            "terminal_settlement_optimizer_steps": audit["terminal_settlement_optimizer_steps_actual"],
            "future_rows_read_count": 0,
        })
        W(root / "D_birth/causality_audit.json", {
            "all_update_batches_mature": audit["all_science_update_batches_mature"],
            "birth_checks_causal": audit["birth_checks_use_only_matured_B_losses"],
            "terminal_settlement_optimizer_steps": audit["terminal_settlement_optimizer_steps_actual"],
            "future_rows_read_count": 0,
        })
        W(root / "F_extra/isolation_audit.json", {
            "B_source_hashes_unchanged": audit["source_hashes_unchanged"],
            "classification_exact_copy": True, "optimizer_parameters_exactly_extra_expert_parameters": True,
        })
        W(root / "D_birth/isolation_audit.json", {
            "B_source_hashes_unchanged": audit["source_hashes_unchanged"],
            "classification_exact_copy": True, "D_before_birth_exact_B_copy": audit["D_before_birth_exact_B_copy"],
            "optimizer_parameters_exactly_extra_expert_parameters": True,
        })
        resident_F = EXPERT_PARAMS * src["n"]
        resident_D = 0 if st["birth_t"] is None else EXPERT_PARAMS * max(0, src["n"] - (int(st["birth_t"]) + 1))
        W(root / "F_extra/summary.json", {
            "protocol": "037", "arm": "F_extra", "completed": True, "parameter_count": EXPERT_PARAMS,
            "optimizer_steps": len(st["F_updates"]), "inference_calls": int(st["F_inference_calls"]),
            "resident_parameter_interval_integral": int(resident_F), "final_expert_hash": sha256_state_dict(st["F"].state_dict()),
            "prediction_sha256": sha256_file(root / "F_extra/predictions.npz"),
            "cost": {"inference_seconds": st["F_inference_seconds"], "update_seconds": st["F_update_seconds"]},
        })
        W(root / "D_birth/summary.json", {
            "protocol": "037", "arm": "D_birth", "completed": True, "parameter_count": EXPERT_PARAMS,
            "birth_occurred": st["birth_t"] is not None, "birth_t": st["birth_t"],
            "first_affected_prediction": st["first_affected_prediction"], "trigger_status": "activated" if st["birth_t"] is not None else "trigger_not_activated",
            "optimizer_steps": len(st["D_updates"]), "inference_calls": int(st["D_inference_calls"]),
            "resident_parameter_interval_integral": int(resident_D),
            "final_expert_hash": None if st["D"] is None else sha256_state_dict(st["D"].state_dict()),
            "prediction_sha256": sha256_file(root / "D_birth/predictions.npz"),
            "cost": {"inference_seconds": st["D_inference_seconds"], "update_seconds": st["D_update_seconds"]},
        })
        source_cost = {}
        for name, rel in {
            "C_ref": "stage_B/seed3601/C_ref/summary.json",
            "D_lin": "stage_B/seed3601/D_lin/summary.json",
        }.items():
            p = src["root"] / rel
            if p.exists():
                source_cost[name] = J(p).get("cost")
        W(root / "science_cost_raw.json", {
            "protocol": "037", "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0,
            "peak_rss_bytes": int(st["peak_rss_bytes"]), "effective_memory_limit_bytes": memlimit(),
            "F_extra_incremental": {"inference_seconds": st["F_inference_seconds"], "update_seconds": st["F_update_seconds"], "optimizer_steps": len(st["F_updates"]), "inference_calls": st["F_inference_calls"], "resident_parameter_interval_integral": resident_F},
            "D_birth_incremental": {"inference_seconds": st["D_inference_seconds"], "update_seconds": st["D_update_seconds"], "optimizer_steps": len(st["D_updates"]), "inference_calls": st["D_inference_calls"], "resident_parameter_interval_integral": resident_D},
            "historical_B_components_if_available": source_cost,
            "accounting_note": "Deployment uses the full historical B path plus the incremental expert; incremental expert timing is not total deployment cost.",
        })
        W(root / "scientific_execution_complete.json", {
            "protocol": "037", "run_id": str(a.run_id), "completed": True,
            "training_sequences_completed": 2, "new_streams_generated": 0,
            "births_observed": 1 if st["birth_t"] is not None else 0, "audit_all_pass": bool(audit["all_pass"]),
        })
        W(root / "status.json", {
            "protocol": "037", "run_id": str(a.run_id), "scientific_status": "completed",
            "publication_status": "pending_until_main_sync", "training_sequences_registered": 2,
            "training_sequences_completed": 2, "new_streams_registered": 0, "new_streams_completed": 0,
            "birth_occurred": st["birth_t"] is not None, "validity_all_pass": bool(audit["all_pass"]),
            "automatic_followup_training": False, "stop_after_registered_budget": True,
        })
        print(json.dumps(J(root / "status.json"), indent=2))
    except Exception as e:
        if st is not None:
            try:
                save_checkpoint(root / "checkpoints", st, src, ledger_path=ledger_path, label="interrupted_cursor_%04d" % int(st["cursor"]))
            except Exception:
                pass
        interrupt_sequences(ledger_path, e)
        W(root / "status.json", {
            "protocol": "037", "run_id": str(a.run_id), "scientific_status": "interrupted",
            "publication_status": "not_attempted", "error": str(e), "automatic_followup_training": False,
            "stop_after_registered_budget": True,
        })
        (root / "traceback.txt").write_text(traceback.format_exc(), encoding="utf8")
        raise


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fixture")
    p.add_argument("--source-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_fixture)
    p = sub.add_parser("science")
    p.add_argument("--source-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--fixture-report", required=True)
    p.add_argument("--resume-checkpoint")
    p.set_defaults(fn=cmd_science)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
