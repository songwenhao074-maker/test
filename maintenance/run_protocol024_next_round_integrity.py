"""Protocol-024 next-round engineering gate on the tracked seed-700 replay.

This is not a performance experiment.  It executes real A/C/D sessions only to
prove namespace isolation, dynamic gap restore, optimizer preservation and an
uninterrupted-vs-resume next-update trajectory before lifecycle-on experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_session import Protocol024Session


def file_tree_sha(root):
    root = Path(root)
    rows = []
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        size = path.stat().st_size
        total += size
        rows.append((str(path.relative_to(root)).replace("\\", "/"), size,
                     h.hexdigest()))
    manifest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return {"file_count": len(rows), "bytes": total, "manifest_sha256": manifest}


def bank_tensor_digest(session):
    bank = session.model.learner
    digest = hashlib.sha256()
    for key in bank.ids:
        for name, tensor in sorted(bank.experts[key].state_dict().items()):
            digest.update((key + ":" + name).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(bank.router_weights[key].detach().cpu().contiguous().numpy().tobytes())
        digest.update(bank.router_biases[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_session(arm, bundle, frozen, anchor, out_dir, run_id):
    return Protocol024Session(
        arm, 1, bundle, frozen, out_dir,
        anchor=anchor, learning_rate=1e-4, run_id=run_id,
        stream_dir=s4.DEV_STREAM,
        phase_defs=[{"name": "engineering", "start": 0, "end": 12,
                     "regime": "tracked_seed700_regression"}],
        target_mode="tol1_regression",
        stream_sha=bundle["manifest"]["stream_sha256"],
        registration={"kind": "engineering_gate_only",
                      "source": "tracked Protocol-023 seed700 replay",
                      "formal_performance_result": False})


def first_adam_step_by_name(session):
    session._capture_live_optimizer_state()
    for name in sorted(session.optimizer_archive):
        state = session.optimizer_archive[name]
        if "step" in state:
            value = state["step"]
            if torch.is_tensor(value):
                value = float(value.item())
            return name, float(value)
    return None, None


def run(out_path):
    s4.configure()
    torch.use_deterministic_algorithms(True)
    p23 = ROOT / "artifacts/ftmoe_online/protocol_023"
    protected_before = file_tree_sha(p23)

    budget = json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))
    frozen = budget["frozen_configuration"]
    bundle = s4.build_replay(s4.DEV_STREAM)
    anchor = s4.load_anchor_pool(bundle["replay"].time_scale)
    result = {"protocol": "024", "kind": "next_round_engineering_integrity",
              "formal_performance_result": False,
              "source_stream_sha256": bundle["manifest"]["stream_sha256"],
              "protected_protocol023_before": protected_before,
              "checks": {}}

    with tempfile.TemporaryDirectory(prefix="p24_next_integrity_") as temp:
        temp = Path(temp)
        saved = {}
        # Actually save one checkpoint for every arm in an isolated P24 run dir.
        for arm in ("A", "C", "D"):
            session = make_session(arm, bundle, frozen, anchor,
                                   temp / ("arm_" + arm), "integrity_namespace")
            for _ in range(5):
                session.step()
            record = session.save_checkpoint("after_five", session.cursor)
            payload = torch.load(Path(record["path"]), map_location="cpu",
                                 weights_only=False)
            saved[arm] = {"record": record,
                          "payload_protocol": payload["protocol"],
                          "payload_arm": payload["arm"],
                          "payload_run_id": payload["run_id"],
                          "payload_stream_sha256": payload["stream_sha256"],
                          "inside_temp": str(Path(record["path"]).resolve()).startswith(
                              str(temp.resolve()))}
            if payload["protocol"] != "024" or payload["arm"] != arm:
                raise AssertionError("P24 checkpoint metadata mismatch for " + arm)
            if payload["run_id"] != "integrity_namespace":
                raise AssertionError("P24 run_id missing from checkpoint")
            if not saved[arm]["inside_temp"]:
                raise AssertionError("checkpoint escaped P24 temporary namespace")
        result["checkpoint_namespace"] = saved

        # Real online Adam state exists before topology mutation.
        uninterrupted = make_session("D", bundle, frozen, anchor,
                                     temp / "resume_source", "resume_gate")
        for _ in range(7):
            uninterrupted.step()
        old_name, old_step = first_adam_step_by_name(uninterrupted)
        if old_name is None:
            raise AssertionError("no pre-birth Adam moment was created")

        # Force only topology events (test-only): reject id4, activate id5.
        bank = uninterrupted.model.learner
        rejected = bank.create_shadow("0")
        uninterrupted.create_shadow_optimizer()
        uninterrupted.discard_shadow()
        child = bank.create_shadow("0")
        uninterrupted.create_shadow_optimizer()
        uninterrupted.activate_shadow()
        bank.ramp_step()
        uninterrupted.learner_hash = uninterrupted.learner_state_hash()
        same_name, same_step = first_adam_step_by_name(uninterrupted)
        if old_name != same_name or old_step != same_step:
            raise AssertionError("existing Adam moment changed during birth")
        if rejected != "4" or child != "5":
            raise AssertionError("gap-ID fixture did not create 4->reject,5->activate")

        checkpoint = uninterrupted.save_checkpoint("resume_point", uninterrupted.cursor)
        resumed = make_session("D", bundle, frozen, anchor,
                               temp / "resume_clone", "resume_gate")
        resumed.restore_checkpoint(checkpoint["path"])
        if resumed.learner_state_hash() != uninterrupted.learner_state_hash():
            raise AssertionError("behavior hash differs immediately after restore")

        # The next step is t=7, therefore it includes the next registered C/D
        # online update opportunity (update_every=4). Compare prediction AND
        # post-update tensors, not one forward only.
        p_live, c_live = uninterrupted.step()
        hash_live = uninterrupted.learner_state_hash()
        tensor_live = bank_tensor_digest(uninterrupted)
        updates_live = uninterrupted.updates
        p_resume, c_resume = resumed.step()
        hash_resume = resumed.learner_state_hash()
        tensor_resume = bank_tensor_digest(resumed)
        updates_resume = resumed.updates

        pred_detection_diff = float(np.max(np.abs(p_live - p_resume)))
        pred_class_diff = float(np.max(np.abs(c_live - c_resume)))
        if pred_detection_diff > 1e-8 or pred_class_diff > 1e-8:
            raise AssertionError("resume next prediction diverged")
        if hash_live != hash_resume or tensor_live != tensor_resume:
            raise AssertionError("resume next update tensor state diverged")
        if updates_live != updates_resume:
            raise AssertionError("resume update count diverged")

        # Retire/reactivate must preserve archived old optimizer moments.
        old_name, old_step = first_adam_step_by_name(uninterrupted)
        uninterrupted.model.learner.set_ramp("5", 1.0)
        uninterrupted.retire_expert("5")
        uninterrupted.reactivate_expert("5")
        uninterrupted._capture_live_optimizer_state()
        archive_has_child = any(name.startswith("5|")
                                for name in uninterrupted.optimizer_archive)

        result["resume_gate"] = {
            "rejected_id": rejected, "activated_id": child,
            "topology_after_restore": resumed.model.learner.topology_manifest(),
            "preexisting_adam_name": old_name,
            "preexisting_adam_step": old_step,
            "prediction_detection_max_abs": pred_detection_diff,
            "prediction_class_max_abs": pred_class_diff,
            "post_update_behavior_hash_equal": hash_live == hash_resume,
            "post_update_tensor_digest_equal": tensor_live == tensor_resume,
            "updates_uninterrupted": updates_live,
            "updates_resumed": updates_resume,
            "reactivation_archive_contains_child": archive_has_child,
        }

    protected_after = file_tree_sha(p23)
    result["protected_protocol023_after"] = protected_after
    result["checks"] = {
        "protocol023_unchanged": protected_before == protected_after,
        "A_C_D_checkpoint_namespace_isolated": all(
            x["inside_temp"] and x["payload_protocol"] == "024"
            for x in saved.values()),
        "gap_id_restore": result["resume_gate"]["topology_after_restore"]["next_id"] >= 6,
        "resume_next_prediction_equal": (
            result["resume_gate"]["prediction_detection_max_abs"] <= 1e-8
            and result["resume_gate"]["prediction_class_max_abs"] <= 1e-8),
        "resume_next_update_equal": (
            result["resume_gate"]["post_update_behavior_hash_equal"]
            and result["resume_gate"]["post_update_tensor_digest_equal"]
            and updates_live == updates_resume),
        "preexisting_optimizer_moments_preserved": same_step == old_step,
    }
    result["passed"] = all(result["checks"].values())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
