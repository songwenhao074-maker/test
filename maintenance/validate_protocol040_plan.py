"""Validate frozen Protocol-040 registration. This validates registration identity, not scientific execution."""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/ftmoe_online/protocol_040/plan.json"
PLAN_SHA = "97839cb55ea9d00521e15529e5b5505004c69fd9587d2d22503b7b1ff99dcd35"
DOC_SHA = "b4be1863ca18e37650fcf9c275179aebcfae57fd9998158c6564a9450a1cfcc8"

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def git_blob_sha(p):
    b = Path(p).read_bytes()
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()

def main():
    p = json.loads(PLAN.read_text(encoding="utf8"))
    assert sha256(PLAN) == PLAN_SHA == (PLAN.parent / "plan.sha256").read_text().split()[0]
    assert sha256(ROOT / p["instructions"]) == DOC_SHA == p["instructions_sha256"]
    assert p["protocol"] == "040" and p["revision"] == 1
    assert p["execution_branch"] == "codex/protocol-040-two-expert-pool-20261003"
    assert p["kernel_reference_commit"] == "600ebb0a1691c72fe7f451bae2c8e8357ad4add4"
    for rel, expected in p["kernel_git_blobs"].items():
        q = ROOT / rel
        assert q.exists(), rel
        assert git_blob_sha(q) == expected, (rel, git_blob_sha(q), expected)
    b = p["budget"]
    assert b["sequence_names"] == ["D_pool2"] and b["new_science_sequences"] == 1
    assert b["live_optimizer_steps_max"] == 352
    assert b["shadow_optimizer_steps_max"] == 16
    assert b["total_optimizer_steps_max"] == 368
    assert b["deployed_dynamic_prediction_forwards_max"] == 5616
    assert b["reuse_preview_forwards_max"] == 2048
    assert b["shadow_qualification_forwards_max"] == 32
    assert b["total_preview_forwards_max"] == 2080
    for k in ("new_streams","extra_seeds","control_retraining","F_training","F_comparisons","donor_updates","hyperparameter_sweeps","real_engineering_gradient_steps","accepted_expert_permanent_deletions"):
        assert b[k] == 0, (k,b[k])
    assert p["pool"]["capacity_including_shadow"] == 2
    assert p["pool"]["active_max"] == 1 and p["pool"]["shadow_max"] == 1
    assert p["pool"]["ids_created_max"] == 2 and p["pool"]["second_candidate_attempts_max"] == 1
    assert p["analysis"]["primary_baseline"] == "C_ref"
    assert p["analysis"]["cumulative_preservation_baseline"] == "D_keep"
    assert p["runtime"]["formal_trigger"] == "workflow_dispatch_only"
    assert p["delivery"]["stop_after_one_sequence"] is True
    assert p["authorization"]["automatic_next_protocol"] is False
    print(json.dumps({
        "protocol":"040",
        "registration_files_valid":True,
        "plan_sha256":PLAN_SHA,
        "instructions_sha256":DOC_SHA,
        "scientific_execution_validated":False
    }, indent=2))

if __name__ == "__main__":
    main()
