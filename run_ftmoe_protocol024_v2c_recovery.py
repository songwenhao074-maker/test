"""Engineering-only v2c recovery wrapper.

The preregistration allows a real birth/retirement/reactivation checkpoint.  The
first v2c run happened to contain no accepted reactivation, while the original
runner only persisted reactivation checkpoints.  This wrapper changes no model,
threshold, budget, reuse, or evaluation rule: it persists the first real
accepted-birth/retirement/reactivation state and runs the already registered
strict recovery comparisons from that checkpoint.
"""
from pathlib import Path
import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol024_v2c as r
from ftmoe_protocol024_session import NEXT_TARGET_MODE
from ftmoe_protocol024_v2c import V2CProtocol024Session, V2C_DEFAULT, model_tensor_hash


_ORIGINAL_STEP = V2CProtocol024Session.step


def _checkpointing_step(self, *args, **kwargs):
    before = len(self.lifecycle_controller.events)
    result = _ORIGINAL_STEP(self, *args, **kwargs)
    if not hasattr(self, "v2c_recovery_checkpoint_records"):
        self.v2c_recovery_checkpoint_records = []
    if not self.v2c_recovery_checkpoint_records:
        new_events = self.lifecycle_controller.events[before:]
        accepted = [e for e in new_events if e.get("kind") in (
            "candidate_accepted", "specialist_memory_frozen",
            "specialist_retired", "specialist_reactivated")]
        if accepted:
            event = accepted[0]
            label = "v2c_recovery_%s_%d" % (event.get("kind", "lifecycle"), int(self.cursor))
            record = self.save_checkpoint(label, int(self.cursor))
            record = dict(record)
            record["source_event"] = event
            record["registered_source_class"] = "real birth/retirement/reactivation checkpoint"
            self.v2c_recovery_checkpoint_records.append(record)
    return result


V2CProtocol024Session.step = _checkpointing_step


def _strict_recovery(main_session, bundle, stream_dir, out_root, train_anchor, guard_anchor, run_id):
    records = getattr(main_session, "v2c_recovery_checkpoint_records", [])
    if not records:
        return {"available": False, "reason": "no real birth/retirement/reactivation checkpoint", "verified": False}
    ck = records[0]
    path = Path(ck["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ck_cursor = int(payload["session"]["cursor"])
    ck_updates = int(payload["session"]["updates"])
    event_count = len(payload["session"]["lifecycle_events"])
    cfg = dict(V2C_DEFAULT); cfg["reuse_enabled"] = True

    ref = V2CProtocol024Session(
        "D", 1, s4.build_replay(stream_dir), r.budget(), Path(out_root)/"recovery_reference_uninterrupted",
        anchor=train_anchor, guard_anchor=guard_anchor, v2c_config=cfg, learning_rate=1e-4, max_experts=8,
        run_id=run_id+"_recovery_ref", stream_dir=stream_dir, phase_defs=r.phase_defs(bundle["manifest"]),
        target_mode=NEXT_TARGET_MODE, stream_sha=r.EXPECTED_STREAM_SHA,
        registration=r.session_registration(run_id+"_recovery_ref", train_anchor, guard_anchor, True))
    while ref.cursor < ck_cursor:
        ref.step()
    pred_prefix_match = np.allclose(
        ref.predictions["probability"][:ck_cursor], main_session.predictions["probability"][:ck_cursor],
        atol=1e-7, rtol=1e-6)
    ref_update, ref_after, ref_event = r.capture_after_next_update(ref, ck_updates, event_count)

    restored = V2CProtocol024Session(
        "D", 1, s4.build_replay(stream_dir), r.budget(), Path(out_root)/"recovery_restored",
        anchor=train_anchor, guard_anchor=guard_anchor, v2c_config=cfg, learning_rate=1e-4, max_experts=8,
        run_id=payload["run_id"], stream_dir=stream_dir, phase_defs=r.phase_defs(bundle["manifest"]),
        target_mode=NEXT_TARGET_MODE, stream_sha=r.EXPECTED_STREAM_SHA, registration=payload["registration"])
    restored.restore_checkpoint(path)
    first_idx = restored.cursor
    p0, c0 = restored.step()
    next_prediction_match = (
        np.allclose(p0, main_session.predictions["probability"][first_idx], atol=1e-7, rtol=1e-6)
        and np.allclose(c0, main_session.predictions["class_probability"][first_idx], atol=1e-7, rtol=1e-6))

    if restored.updates > ck_updates:
        rst_update = {
            "after_step_index": first_idx, "update_count": restored.updates,
            "model_hash": model_tensor_hash(restored.model), "adam_hash": r.adam_archive_hash(restored),
            "rng_hash": r.full_rng_hash(), "state": r.state_manifest(restored)}
        idx = restored.cursor; pa, ca = restored.step()
        rst_after = {"index": idx, "probability": pa.copy(), "class_probability": ca.copy()}
        rst_event = (r.semantic_event(restored.lifecycle_controller.events[event_count])
                     if len(restored.lifecycle_controller.events) > event_count else None)
    else:
        rst_update, rst_after, rst_event = r.capture_after_next_update(restored, ck_updates, event_count)

    update_match = bool(ref_update and rst_update
        and ref_update["model_hash"] == rst_update["model_hash"]
        and ref_update["adam_hash"] == rst_update["adam_hash"]
        and ref_update["rng_hash"] == rst_update["rng_hash"]
        and ref_update["state"] == rst_update["state"])
    after_match = bool(ref_after and rst_after and ref_after["index"] == rst_after["index"]
        and np.allclose(ref_after["probability"], rst_after["probability"], atol=1e-7, rtol=1e-6)
        and np.allclose(ref_after["class_probability"], rst_after["class_probability"], atol=1e-7, rtol=1e-6))
    event_match = (ref_event == rst_event) if ref_event is not None else None
    return {
        "available": True, "checkpoint": ck, "checkpoint_cursor": ck_cursor,
        "checkpoint_source_event": ck.get("source_event"),
        "registered_source_class": ck.get("registered_source_class"),
        "uninterrupted_prefix_matches_main": bool(pred_prefix_match),
        "next_prediction_match": bool(next_prediction_match),
        "next_live_update_match": update_match,
        "prediction_after_update_match": after_match,
        "next_nonempty_event_covered": ref_event is not None,
        "next_event_match": event_match,
        "reference_update": ref_update, "restored_update": rst_update,
        "reference_next_event": ref_event, "restored_next_event": rst_event,
        "verified": bool(pred_prefix_match and next_prediction_match and update_match and after_match and event_match is True)}


r.strict_recovery = _strict_recovery

if __name__ == "__main__":
    r.main()
