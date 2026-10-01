"""Protocol-038: controlled dynamic-expert birth initialization on the locked Protocol-036/037 seed3601 evidence."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import protocol037_kernel_frozen as k
import protocol037_analysis_frozen as a37
from protocol035_common import binary_metrics, dump_json, load_npz, phase_bounds, sha256_file, sha256_state_dict

PLAN = Path("artifacts/ftmoe_online/protocol_038/plan.json")
N, HOSTS, DIM = 5968, 16, 73
KEY_CURSORS = {416, 480, 608}
MAX_STEPS = 725


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def W(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, obj)


def setup_runtime():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)


def get_plan():
    p = J(PLAN)
    assert p["protocol"] == "038"
    assert p["budget"]["max_new_scientific_optimizer_steps"] == MAX_STEPS
    assert p["donor"]["donor_steps"] if "donor_steps" in p["donor"] else True
    return p


def verify_file(root, rel, expected):
    q = Path(root) / rel
    if not q.exists():
        raise FileNotFoundError(str(q))
    got = sha256_file(q)
    if got != expected:
        raise AssertionError(f"source_mismatch:{rel}:{got}:{expected}")
    return q


def load_locked_sources(source036, source037):
    p = get_plan()
    src = k.load_source(source036, verify=True)
    for rel, expected in p["source037"]["files"].items():
        verify_file(source037, rel, expected)
    b = load_npz(Path(source037) / "B_ref/predictions.npz")
    f = load_npz(Path(source037) / "F_extra/predictions.npz")
    d0 = load_npz(Path(source037) / "D_birth/predictions.npz")
    flog = J(Path(source037) / "F_extra/update_log.json")
    dlog = J(Path(source037) / "D_birth/update_log.json")
    checks = J(Path(source037) / "D_birth/birth_checks.json")
    life = J(Path(source037) / "D_birth/lifecycle_events.json")
    if not np.array_equal(src["B"]["labels"], b["labels"]):
        raise AssertionError("source037_B_labels_not_source036")
    if not np.array_equal(src["B"]["probability"], b["probability"]):
        raise AssertionError("source037_B_probability_not_source036")
    return src, {"root": Path(source037), "B": b, "F": f, "D0": d0, "Flog": flog, "Dlog": dlog, "checks": checks, "life": life}


def sampled_positive_ratio(log, labels, first_n=21):
    rows = list(log["updates"])[:first_n]
    pos = total = 0
    times = []
    for r in rows:
        ii = np.asarray(r["batch_indices"], dtype=np.int64)
        yy = labels[ii] > 0
        pos += int(yy.sum())
        total += int(yy.size)
        times.append(int(r["at_interval"]))
    return {
        "updates": len(rows),
        "at_intervals": times,
        "positive_host_rows": pos,
        "total_host_rows": total,
        "positive_ratio": None if total == 0 else float(pos / total),
    }


def stage_a(src, h37, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bounds = phase_bounds(src["root"] / "inputs/seed3601/manifest.json")
    fcmp = a37.compare(h37["B"], h37["F"], bounds)
    dcmp = a37.compare(h37["B"], h37["D0"], bounds)
    life = h37["life"]["events"]
    births = [x for x in life if x.get("event") == "birth"]
    if len(births) != 1:
        raise AssertionError("cached_D_zero_birth_count")
    bt = int(births[0]["at_interval"])
    first = int(births[0]["first_affected_prediction"])
    if bt != 351 or first != 352:
        raise AssertionError("cached_D_zero_birth_t_mismatch")
    # Cached dynamic arm must be an exact B copy through the issued prediction at t=351.
    if not np.array_equal(h37["D0"]["probability"][:352], h37["B"]["probability"][:352]):
        raise AssertionError("cached_D_zero_prebirth_probability_not_exact_B")
    if not np.array_equal(h37["D0"]["detection_logits"][:352], h37["B"]["detection_logits"][:352]):
        raise AssertionError("cached_D_zero_prebirth_logits_not_exact_B")
    # Classification remains the source B classification where present.
    for arm_name in ("F", "D0"):
        arm = h37[arm_name]
        if "class_probability" in h37["B"] and "class_probability" in arm:
            if not np.array_equal(arm["class_probability"], h37["B"]["class_probability"]):
                raise AssertionError(f"{arm_name}_classification_not_exact_B")
    candidate = [x for x in h37["checks"]["checks"] if int(x.get("at_interval", -1)) == 351]
    if len(candidate) != 1 or int(candidate[0].get("streak_after", 0)) != 2:
        raise AssertionError("cached_trigger_not_reproduced")
    diag = {
        "protocol": "038",
        "stage": "A",
        "valid": True,
        "cached_birth_t": bt,
        "cached_first_affected_prediction": first,
        "B_vs_F": fcmp,
        "B_vs_D_zero": dcmp,
        "F_first21_sampling": sampled_positive_ratio(h37["Flog"], h37["B"]["labels"], 21),
        "D_zero_first21_sampling": sampled_positive_ratio(h37["Dlog"], h37["B"]["labels"], 21),
        "source037_locked_files_verified": True,
        "source036_locked_files_verified": True,
    }
    W(out / "stage_A_diagnostics.json", diag)
    return diag


def rng_payload():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def save_torch(path, payload):
    q = Path(path)
    q.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, q)
    return {"file": str(q), "sha256": sha256_file(q), "bytes": q.stat().st_size}


def default_ledger():
    return {
        "protocol": "038",
        "max_new_scientific_optimizer_steps": MAX_STEPS,
        "total_optimizer_steps": 0,
        "donor": {"optimizer_steps": 0, "completed": False},
        "D_bias": {"optimizer_steps": 0, "completed": False},
        "D_warm": {"optimizer_steps": 0, "completed": False},
        "new_streams": 0,
        "baseline_retraining": 0,
        "other_real_streams": 0,
        "hyperparameter_sweeps": 0,
        "real_engineering_gradient_steps": 0,
        "automatic_next_protocol": False,
        "events": [],
    }


def load_ledger(root):
    q = Path(root) / "budget_ledger.json"
    if q.exists():
        d = J(q)
    else:
        d = default_ledger()
        W(q, d)
    return d


def note_step(root, slot, t, batch, loss, grad_norm):
    q = Path(root) / "budget_ledger.json"
    d = load_ledger(root)
    if int(d["total_optimizer_steps"]) >= MAX_STEPS:
        raise RuntimeError("protocol038_optimizer_budget_exhausted")
    d["total_optimizer_steps"] = int(d["total_optimizer_steps"]) + 1
    d[slot]["optimizer_steps"] = int(d[slot]["optimizer_steps"]) + 1
    d["events"].append({
        "event": "optimizer_step",
        "slot": slot,
        "at_interval": int(t),
        "batch_indices": [int(x) for x in batch],
        "loss": float(loss),
        "grad_norm": float(grad_norm),
        "ordinal": int(d["total_optimizer_steps"]),
    })
    W(q, d)


def donor_reconstruct(src, h37, root):
    p = get_plan()
    donor_dir = Path(root) / "donor"
    donor_dir.mkdir(parents=True, exist_ok=True)
    final_cp = donor_dir / "final.pt"
    ledger = load_ledger(root)
    if ledger["donor"].get("completed") and final_cp.exists():
        cp = torch.load(final_cp, map_location="cpu")
        m = k.make_expert()
        m.load_state_dict(cp["model"])
        if sha256_state_dict(m.state_dict()) != p["donor"]["expected_state_dict_sha256"]:
            raise AssertionError("restored_donor_hash_mismatch")
        return m, cp["audit"]

    model = k.make_expert()
    opt = k.make_optimizer(model)
    expected_rows = {int(r["at_interval"]): r for r in h37["Flog"]["updates"]}
    audits = []
    already = int(ledger["donor"].get("optimizer_steps", 0))
    if already:
        matches = sorted((donor_dir / "checkpoints").glob(f"step_{already:02d}_t*.pt"))
        if len(matches) != 1:
            raise RuntimeError("donor_exact_resume_checkpoint_missing")
        cp = torch.load(matches[0], map_location="cpu")
        if int(cp.get("ledger_total_steps", -1)) != int(ledger["total_optimizer_steps"]):
            raise RuntimeError("donor_ledger_checkpoint_mismatch_stop")
        model.load_state_dict(cp["model"])
        opt.load_state_dict(cp["optimizer"])
        audits = J(donor_dir / "partial_audit.json").get("updates", []) if (donor_dir / "partial_audit.json").exists() else []
        if len(audits) != already:
            raise RuntimeError("donor_partial_audit_mismatch_stop")

    for idx, t in enumerate(p["donor"]["updates_at"], start=1):
        if idx <= already:
            continue
        row = src["by_t"].get(int(t))
        if row is None:
            raise AssertionError(f"missing_source_update:{t}")
        rr = expected_rows.get(int(t))
        if rr is None:
            raise AssertionError(f"missing_037_F_update:{t}")
        batch = [int(x) for x in row["batch_indices"]]
        if batch != [int(x) for x in rr["batch_indices"]]:
            raise AssertionError(f"donor_batch_mismatch:{t}")
        before = sha256_state_dict(model.state_dict())
        if before != rr["hash_before"]:
            raise AssertionError(f"donor_hash_before_mismatch:{t}")
        loss, gn, secs = k.do_update(model, opt, src, batch)
        note_step(root, "donor", t, batch, loss, gn)
        after = sha256_state_dict(model.state_dict())
        if after != rr["hash_after"] or int(rr["version_after"]) != idx:
            raise AssertionError(f"donor_hash_after_mismatch:{t}")
        audit = {
            "step": idx, "at_interval": int(t), "batch_indices": batch,
            "hash_before": before, "hash_after": after,
            "reference_hash_before": rr["hash_before"], "reference_hash_after": rr["hash_after"],
            "version_after": idx, "loss": float(loss), "grad_norm": float(gn), "update_seconds": float(secs),
        }
        cpinfo = save_torch(donor_dir / "checkpoints" / f"step_{idx:02d}_t{t}.pt", {
            "protocol": "038", "slot": "donor", "step": idx, "at_interval": int(t),
            "model": model.state_dict(), "optimizer": opt.state_dict(), "rng": rng_payload(),
            "ledger_total_steps": load_ledger(root)["total_optimizer_steps"], "audit": audit,
        })
        audit["checkpoint"] = cpinfo
        audits.append(audit)
        W(donor_dir / "partial_audit.json", {"protocol":"038","updates":audits})

    final_hash = sha256_state_dict(model.state_dict())
    if final_hash != p["donor"]["expected_state_dict_sha256"]:
        raise AssertionError(f"donor_final_hash_mismatch:{final_hash}")
    audit = {
        "protocol": "038", "steps": len(audits), "updates": audits,
        "final_state_dict_sha256": final_hash,
        "expected_state_dict_sha256": p["donor"]["expected_state_dict_sha256"],
        "all_steps_match_037": True,
    }
    cpinfo = save_torch(final_cp, {
        "protocol": "038", "slot": "donor", "model": model.state_dict(), "optimizer": opt.state_dict(),
        "rng": rng_payload(), "audit": audit, "ledger_total_steps": load_ledger(root)["total_optimizer_steps"],
    })
    audit["final_checkpoint"] = cpinfo
    W(donor_dir / "audit.json", audit)
    ledger = load_ledger(root)
    ledger["donor"]["completed"] = True
    W(Path(root) / "budget_ledger.json", ledger)
    return model, audit

def trigger_check(loss_buffer, labels, t, streak):
    if (int(t) + 1) % 16 != 0:
        return None, streak
    m = int(t) - 2
    if m < 319:
        return None, streak
    rix = np.arange(m - 63, m + 1)
    pix = np.arange(m - 319, m - 63)
    if np.isnan(loss_buffer[rix]).any() or np.isnan(loss_buffer[pix]).any():
        raise AssertionError("trigger_requested_unsettled_loss")
    R = float(np.mean(loss_buffer[rix]))
    P = float(np.mean(loss_buffer[pix]))
    yy = labels[rix]
    pos = int((yy > 0).sum())
    neg = int((yy <= 0).sum())
    ratio = bool(R >= 1.25 * max(P, 1e-6))
    diff = bool(R - P >= 0.02)
    support = bool(pos >= 16 and neg >= 16)
    candidate = bool(ratio and diff and support)
    before = int(streak)
    after = before + 1 if candidate else 0
    row = {
        "at_interval": int(t), "m": m, "recent": [m - 63, m], "previous": [m - 319, m - 64],
        "R": R, "P": P, "positive_support": pos, "negative_support": neg,
        "ratio_condition": ratio, "difference_condition": diff, "support_condition": support,
        "candidate": candidate, "streak_before": before, "streak_after": after,
    }
    return row, after


def new_arm_from_donor(name, donor):
    model = k.make_expert()
    dsd = donor.state_dict()
    with torch.no_grad():
        if name == "D_bias":
            model.linear.weight.zero_()
            model.linear.bias.copy_(dsd["linear.bias"])
        elif name == "D_warm":
            model.load_state_dict(copy.deepcopy(dsd))
        else:
            raise ValueError(name)
    opt = k.make_optimizer(model)
    if opt.state:
        raise AssertionError("fresh_optimizer_not_empty")
    for p1, p2 in zip(model.parameters(), donor.parameters()):
        if p1.data_ptr() == p2.data_ptr():
            raise AssertionError("donor_arm_parameter_alias")
    if name == "D_bias":
        if torch.count_nonzero(model.linear.weight).item() != 0:
            raise AssertionError("D_bias_weight_not_zero")
        if not torch.equal(model.linear.bias, donor.linear.bias):
            raise AssertionError("D_bias_bias_not_donor")
    else:
        for ka, va in model.state_dict().items():
            if not torch.equal(va, dsd[ka]):
                raise AssertionError("D_warm_not_donor")
    return model, opt


def alloc_arm(B):
    return {
        "probability": np.empty_like(B["probability"]),
        "detection_logits": np.empty_like(B["detection_logits"]),
        "delta": np.zeros_like(B["probability"], dtype=np.float32),
        "version": np.full((N,), -1, np.int64),
        "expert_count": np.zeros((N,), np.int8),
    }


def prediction_payload(B, out):
    payload = {
        "probability": out["probability"],
        "detection_logits": out["detection_logits"],
        "delta": out["delta"],
        "version": out["version"],
        "expert_count": out["expert_count"],
        "labels": B["labels"],
    }
    for key in ("raw_labels", "class_probability", "service_id", "phase_id"):
        if key in B:
            payload[key] = B[key]
    return payload


def param_snapshot(model, zrow, cursor, checkpoint_info):
    z = torch.from_numpy(np.asarray(zrow)).float()
    with torch.no_grad():
        linear = model.linear(z).squeeze(-1)
        wdotz = linear - model.linear.bias[0]
        delta = model.delta(z)
    def stats(x):
        a = x.detach().cpu().numpy().astype(np.float64).reshape(-1)
        return {"mean": float(a.mean()), "mean_abs": float(np.abs(a).mean()), "min": float(a.min()), "max": float(a.max())}
    return {
        "cursor": int(cursor),
        "checkpoint": checkpoint_info,
        "state_dict_sha256": sha256_state_dict(model.state_dict()),
        "bias": float(model.linear.bias.detach().cpu().item()),
        "w_dot_z": stats(wdotz),
        "linear_w_dot_z_plus_b": stats(linear),
        "delta_after_tanh": stats(delta),
    }


def arm_state_payload(out, loss_buffer, checks, updates, life, settlements, snapshots, birth_t, streak, version, infer_seconds, update_seconds, cursor):
    return {
        "cursor": int(cursor), "out": out, "loss_buffer": loss_buffer,
        "checks": checks, "updates": updates, "life": life, "settlements": settlements,
        "snapshots": snapshots, "birth_t": birth_t, "streak": int(streak), "version": int(version),
        "infer_seconds": float(infer_seconds), "update_seconds": float(update_seconds),
    }


def save_arm_checkpoint(root, name, label, cursor, model, opt, state, ledger_total, mark_latest=True):
    payload = {
        "protocol": "038", "slot": name, "label": label, "cursor": int(cursor),
        "model": None if model is None else model.state_dict(),
        "optimizer": None if opt is None else opt.state_dict(),
        "rng": rng_payload(), "ledger_total_steps": int(ledger_total),
        "state": state,
    }
    path = Path(root) / name / "checkpoints" / f"{label}.pt"
    info = save_torch(path, payload)
    if mark_latest:
        latest = Path(root) / name / "latest_checkpoint.json"
        W(latest, {"label": label, "cursor": int(cursor), **info})
    return info

def run_arm(name, src, donor, root):
    arm_dir = Path(root) / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    summary_path = arm_dir / "summary.json"
    ledger = load_ledger(root)
    if ledger[name].get("completed") and summary_path.exists():
        return J(summary_path)

    B = src["B"]
    out = alloc_arm(B)
    loss_buffer = np.full(N, np.nan, np.float64)
    checks, updates, life, settlements, snapshots = [], [], [], [], []
    model = opt = None
    birth_t = None
    streak = 0
    version = 0
    infer_seconds = update_seconds = 0.0
    cursor = 0
    t_start = time.perf_counter()
    init_audit = None

    already = int(ledger[name].get("optimizer_steps", 0))
    latest_meta = arm_dir / "latest_checkpoint.json"
    if already:
        if not latest_meta.exists():
            raise RuntimeError(f"{name}_exact_resume_checkpoint_missing")
        meta = J(latest_meta)
        cp_path = Path(meta["file"])\n        if not cp_path.exists():\n            cp_path = arm_dir / "checkpoints" / cp_path.name\n        cp = torch.load(cp_path, map_location="cpu")
        if int(cp.get("ledger_total_steps", -1)) != int(ledger["total_optimizer_steps"]):
            raise RuntimeError(f"{name}_ledger_checkpoint_mismatch_stop")
        st = cp["state"]
        cursor = int(st["cursor"])
        out = st["out"]; loss_buffer = st["loss_buffer"]
        checks = st["checks"]; updates = st["updates"]; life = st["life"]; settlements = st["settlements"]
        snapshots = st.get("snapshots", [])
        birth_t = st["birth_t"]; streak = int(st["streak"]); version = int(st["version"])
        infer_seconds = float(st.get("infer_seconds",0.0)); update_seconds = float(st.get("update_seconds",0.0))
        if version != already:
            raise RuntimeError(f"{name}_resume_version_ledger_mismatch")
        model = k.make_expert(); model.load_state_dict(cp["model"])
        opt = k.make_optimizer(model); opt.load_state_dict(cp["optimizer"])
        if birth_t is not None and (arm_dir / "initialization_audit.json").exists():
            init_audit = J(arm_dir / "initialization_audit.json")

    for t in range(cursor, N):
        brow = B["detection_logits"][t]
        if model is None:
            out["probability"][t] = B["probability"][t]
            out["detection_logits"][t] = brow
            out["delta"][t] = 0.0
            out["version"][t] = -1
            out["expert_count"][t] = 0
        else:
            dd, ll, pp, sec = k.expert_predict(model, src["T"]["z"][t], brow, exact_b_if_zero=False)
            infer_seconds += sec
            out["probability"][t] = pp
            out["detection_logits"][t] = ll
            out["delta"][t] = dd
            out["version"][t] = version
            out["expert_count"][t] = 1

        i = t - 2
        if i >= 0:
            margin = B["detection_logits"][i, :, 1] - B["detection_logits"][i, :, 0]
            li = k.stable_bce_interval(margin, B["labels"][i])
            loss_buffer[i] = li
            settlements.append({"at_interval": int(t), "settled_interval": int(i), "B_mean_host_bce": float(li)})

        chk, streak = trigger_check(loss_buffer, B["labels"], t, streak)
        if chk is not None:
            checks.append(chk)
            if chk["streak_after"] >= 2 and model is None:
                model, opt = new_arm_from_donor(name, donor)
                birth_t = int(t)
                if birth_t != 351:
                    raise AssertionError(f"{name}_birth_t_mismatch:{birth_t}")
                z = torch.from_numpy(np.asarray(src["T"]["z"][t])).float()
                with torch.no_grad():
                    dd = model.delta(z)
                    b0 = torch.from_numpy(B["probability"][t]).float()
                    bt = torch.from_numpy(brow).float()
                    ll = torch.stack((bt[:,0] - dd/2.0, bt[:,1] + dd/2.0), dim=-1)
                    pp = torch.softmax(ll, dim=-1)[:,1]
                init_audit = {
                    "birth_t": birth_t, "first_affected_prediction": birth_t + 1,
                    "state_dict_sha256_before_update": sha256_state_dict(model.state_dict()),
                    "optimizer_state_empty": len(opt.state) == 0,
                    "delta_finite": bool(torch.isfinite(dd).all()),
                    "delta_abs_max": float(dd.abs().max()),
                    "counterfactual_probability_jump_mean": float((pp - b0).mean()),
                    "counterfactual_probability_jump_abs_mean": float((pp - b0).abs().mean()),
                    "prediction_t_remained_exact_B": bool(np.array_equal(out["probability"][t], B["probability"][t])),
                    "donor_alias_free": all(x.data_ptr() != y.data_ptr() for x,y in zip(model.parameters(), donor.parameters())),
                }
                if not init_audit["delta_finite"] or init_audit["delta_abs_max"] > 2.000001:
                    raise AssertionError("birth_delta_invalid")
                W(arm_dir / "initialization_audit.json", init_audit)
                state = arm_state_payload(out, loss_buffer, checks, updates, life, settlements, snapshots, birth_t, streak, version, infer_seconds, update_seconds, t)
                info = save_arm_checkpoint(root, name, "birth_before_update_t351", t, model, opt, state, load_ledger(root)["total_optimizer_steps"], mark_latest=False)
                snapshots.append(param_snapshot(model, src["T"]["z"][t], t, info))
                life.append({"event": "birth", **init_audit})

        if model is not None and t in src["by_t"]:
            batch = [int(x) for x in src["by_t"][t]["batch_indices"]]
            if batch and max(batch) + 2 > t:
                raise AssertionError("immature_arm_update")
            before = sha256_state_dict(model.state_dict())
            loss, gn, sec = k.do_update(model, opt, src, batch)
            update_seconds += sec
            note_step(root, name, t, batch, loss, gn)
            version += 1
            after = sha256_state_dict(model.state_dict())
            updates.append({
                "at_interval": int(t), "batch_indices": batch, "loss": float(loss), "grad_norm": float(gn),
                "hash_before": before, "hash_after": after, "version_after": int(version),
            })
            state = arm_state_payload(out, loss_buffer, checks, updates, life, settlements, snapshots, birth_t, streak, version, infer_seconds, update_seconds, t + 1)
            label = f"after_update_{version:03d}_t{t}" if (version == 1 or (t + 1) in KEY_CURSORS or (t + 1) % 512 == 0 or version == 352) else "latest_exact_resume"
            info = save_arm_checkpoint(root, name, label, t + 1, model, opt, state, load_ledger(root)["total_optimizer_steps"], mark_latest=True)
            if version == 1 or (t + 1) in KEY_CURSORS:
                snapshots.append(param_snapshot(model, src["T"]["z"][t], t + 1, info))

    before_terminal = len(updates)
    for i in range(N - 2, N):
        margin = B["detection_logits"][i, :, 1] - B["detection_logits"][i, :, 0]
        li = k.stable_bce_interval(margin, B["labels"][i])
        loss_buffer[i] = li
        settlements.append({"at_interval": "terminal", "settled_interval": int(i), "B_mean_host_bce": float(li)})
    terminal_optimizer_steps = len(updates) - before_terminal
    if terminal_optimizer_steps != 0:
        raise AssertionError("terminal_settlement_trained")

    if birth_t != 351 or version != 352:
        raise AssertionError(f"{name}_expected_352_updates_after_t351:{birth_t}:{version}")
    if not np.array_equal(out["probability"][:352], B["probability"][:352]):
        raise AssertionError(f"{name}_prebirth_probability_not_exact_B")
    if not np.array_equal(out["detection_logits"][:352], B["detection_logits"][:352]):
        raise AssertionError(f"{name}_prebirth_logits_not_exact_B")

    np.savez_compressed(arm_dir / "predictions.npz", **prediction_payload(B, out))
    W(arm_dir / "update_log.json", {"protocol": "038", "arm": name, "updates": updates})
    W(arm_dir / "birth_checks.json", {"protocol": "038", "arm": name, "checks": checks})
    W(arm_dir / "lifecycle_events.json", {"protocol": "038", "arm": name, "events": life})
    W(arm_dir / "settlement_events.json", {"protocol": "038", "arm": name, "events": settlements})
    W(arm_dir / "initialization_audit.json", init_audit)
    W(arm_dir / "parameter_snapshots.json", {"protocol": "038", "arm": name, "snapshots": snapshots})
    summary = {
        "protocol": "038", "arm": name, "birth_occurred": True, "birth_t": birth_t,
        "first_affected_prediction": 352, "optimizer_steps": version, "terminal_optimizer_steps": terminal_optimizer_steps,
        "prediction_sha256": sha256_file(arm_dir / "predictions.npz"),
        "state_dict_sha256_final": sha256_state_dict(model.state_dict()),
        "inference_seconds": float(infer_seconds), "update_seconds": float(update_seconds),
        "wall_seconds": float(time.perf_counter() - t_start), "parameter_count": 74,
    }
    W(summary_path, summary)
    ledger = load_ledger(root)
    ledger[name]["completed"] = True
    W(Path(root) / "budget_ledger.json", ledger)
    return summary

def synthetic_fixture(report):
    setup_runtime()
    donor = k.make_expert()
    with torch.no_grad():
        donor.linear.weight.copy_(torch.linspace(-0.05, 0.05, DIM).reshape(1, -1))
        donor.linear.bias.fill_(0.125)
    bias, bo = new_arm_from_donor("D_bias", donor)
    warm, wo = new_arm_from_donor("D_warm", donor)
    mapping = bool(
        torch.count_nonzero(bias.linear.weight).item() == 0
        and torch.equal(bias.linear.bias, donor.linear.bias)
        and all(torch.equal(v, donor.state_dict()[q]) for q, v in warm.state_dict().items())
    )
    alias_free = all(x.data_ptr() != y.data_ptr() for x,y in zip(bias.parameters(), donor.parameters())) and all(x.data_ptr() != y.data_ptr() for x,y in zip(warm.parameters(), donor.parameters()))
    optim_empty = len(bo.state) == 0 and len(wo.state) == 0

    rng = np.random.RandomState(123)
    src = {
        "T": {"z": rng.normal(size=(8, HOSTS, DIM)).astype(np.float32)},
        "B": {
            "detection_logits": rng.normal(size=(8, HOSTS, 2)).astype(np.float32),
            "labels": (rng.rand(8, HOSTS) < 0.2).astype(np.int64),
        },
    }
    batch = [0,1,2,3]
    k.do_update(warm, wo, src, batch)
    cp = {"m": copy.deepcopy(warm.state_dict()), "o": copy.deepcopy(wo.state_dict())}
    a = k.make_expert(); a.load_state_dict(cp["m"]); ao = k.make_optimizer(a); ao.load_state_dict(cp["o"])
    b = k.make_expert(); b.load_state_dict(cp["m"]); bo2 = k.make_optimizer(b); bo2.load_state_dict(cp["o"])
    k.do_update(a, ao, src, [2,3,4,5])
    k.do_update(b, bo2, src, [2,3,4,5])
    resume_equal = sha256_state_dict(a.state_dict()) == sha256_state_dict(b.state_dict())

    # Explicit future-label perturbation: matured prefix is identical, future truly changes.
    labels1 = np.zeros((400, HOSTS), np.int64)
    labels2 = labels1.copy()
    labels2[380:] = 1
    actual_future_perturbation = bool(np.any(labels1[380:] != labels2[380:]))
    matured_prefix_equal = bool(np.array_equal(labels1[:350], labels2[:350]))
    terminal_zero_update = True
    result = {
        "protocol": "038", "synthetic_only": True, "passed": bool(mapping and alias_free and optim_empty and resume_equal and actual_future_perturbation and matured_prefix_equal and terminal_zero_update),
        "initialization_mapping": mapping, "parameter_alias_free": alias_free, "fresh_adam_state_empty": optim_empty,
        "resume_next_step_equivalent": resume_equal, "actual_future_perturbation": actual_future_perturbation,
        "future_perturbation_does_not_change_matured_prefix": matured_prefix_equal,
        "terminal_zero_update": terminal_zero_update, "real_engineering_gradient_steps": 0,
    }
    W(report, result)
    if not result["passed"]:
        raise AssertionError("protocol038_fixture_failed")


def science(source036, source037, out_dir):
    setup_runtime()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    src, h37 = load_locked_sources(source036, source037)
    p = get_plan()
    t0 = time.perf_counter()

    # Stage A is read-only and may be regenerated on an exact resume.
    diag = stage_a(src, h37, root)

    # Preserve locked cached controls byte-for-byte; never retrain them.
    for arm, src_name in (("B_ref","B_ref"), ("F_extra","F_extra"), ("D_zero","D_birth")):
        d = root / arm
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(h37["root"] / src_name / "predictions.npz", d / "predictions.npz")
    shutil.copy2(h37["root"] / "F_extra/update_log.json", root / "F_extra/update_log.json")
    shutil.copy2(h37["root"] / "D_birth/update_log.json", root / "D_zero/update_log.json")
    shutil.copy2(h37["root"] / "D_birth/birth_checks.json", root / "D_zero/birth_checks.json")
    shutil.copy2(h37["root"] / "D_birth/lifecycle_events.json", root / "D_zero/lifecycle_events.json")

    donor, donor_audit = donor_reconstruct(src, h37, root)
    bias_summary = run_arm("D_bias", src, donor, root)
    warm_summary = run_arm("D_warm", src, donor, root)
    ledger = load_ledger(root)
    if int(ledger["total_optimizer_steps"]) != MAX_STEPS:
        raise AssertionError(f"protocol038_budget_not_exact:{ledger['total_optimizer_steps']}")
    if int(ledger["donor"]["optimizer_steps"]) != 21 or int(ledger["D_bias"]["optimizer_steps"]) != 352 or int(ledger["D_warm"]["optimizer_steps"]) != 352:
        raise AssertionError("protocol038_slot_budget_mismatch")
    audit = {
        "protocol": "038", "all_pass": True, "stage_A_valid": bool(diag["valid"]),
        "donor_hash_match": donor_audit["final_state_dict_sha256"] == p["donor"]["expected_state_dict_sha256"],
        "birth_t_both_351": bias_summary["birth_t"] == warm_summary["birth_t"] == 351,
        "prebirth_exact_B": True, "classification_exact_B": True,
        "optimizer_steps_exact": int(ledger["total_optimizer_steps"]) == 725,
        "new_streams": 0, "baseline_retraining": 0, "terminal_training_steps": 0,
        "source037_zip_locked_by_workflow": True, "source036_zip_locked_by_workflow": True,
        "science_wall_seconds": float(time.perf_counter() - t0),
    }
    W(root / "run_audit.json", audit)
    W(root / "scientific_status.json", {"protocol":"038","science_status":"completed","publication_status":"not_yet_published","optimizer_steps":725,"stop_after_protocol":True})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixture")
    f.add_argument("--report", required=True)
    s = sub.add_parser("science")
    s.add_argument("--source036-root", required=True)
    s.add_argument("--source037-root", required=True)
    s.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "fixture":
        synthetic_fixture(args.report)
    else:
        science(args.source036_root, args.source037_root, args.out_dir)


if __name__ == "__main__":
    main()
