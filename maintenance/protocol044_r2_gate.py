"""Protocol-044 revision2 recovery gate builder. Old E is inherited evidence, never rewritten."""
from __future__ import annotations
import argparse, hashlib, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
R1_PLAN="a27a6cca6abde40b6395876eccd3db5f550a044e952a701fe63aa2e4306a5f51"
R2_PLAN="823365226f0e418bcc5dbdc79e0e1b77f5adf72c3e404b67fb28fdffd9cc20b3"
SCENARIO="6e6bd03efc0d84f5e885eacf36f1ac80e770198703463cc051216ad1a741683f"
OLD_E_ID=11296492568
OLD_E_DIGEST="sha256:90b6bab6a149c46da0aa19e551c02f76b5665a27a83e97d7de192e21e08b6591"
OLD_E_EXEC="0fb751738718d3d06aeeaa32cfcb593436fd7c87"
GEN_KEY="protocol044_revision2_seed4401_reconstruction1"
SCI_KEY="protocol044_revision1_stageS_seed4401_sequence"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def J(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def T(x,k):return type(x.get(k)) is bool and x[k] is True
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--old-e-gate",required=True);ap.add_argument("--old-e-lock",required=True)
    ap.add_argument("--compatibility",required=True);ap.add_argument("--entry",required=True);ap.add_argument("--durable",required=True)
    ap.add_argument("--publication",required=True);ap.add_argument("--inventory",required=True)
    ap.add_argument("--entry-artifact-id",type=int,required=True);ap.add_argument("--entry-artifact-digest",required=True)
    ap.add_argument("--durable-artifact-id",type=int,required=True);ap.add_argument("--durable-artifact-digest",required=True)
    ap.add_argument("--execution-sha",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    if sha(ROOT/"artifacts/ftmoe_online/protocol_044/plan.json")!=R1_PLAN:raise RuntimeError("r1 plan")
    if sha(ROOT/"artifacts/ftmoe_online/protocol_044/revision_002/plan.json")!=R2_PLAN:raise RuntimeError("r2 plan")
    if sha(ROOT/"artifacts/ftmoe_online/protocol_044/scenario_registration.json")!=SCENARIO:raise RuntimeError("scenario")
    old=J(a.old_e_gate);lock=J(a.old_e_lock);comp=J(a.compatibility);entry=J(a.entry);dur=J(a.durable);pub=J(a.publication);inv=J(a.inventory)
    inherited=bool(T(old,"E_gate") and old.get("execution_sha")==OLD_E_EXEC and T(lock,"E_gate") and lock.get("branch_head")==OLD_E_EXEC)
    compat=bool(comp.get("inherited_E_artifact_id")==OLD_E_ID and comp.get("inherited_E_digest")==OLD_E_DIGEST and T(comp,"inherited_core_compatible"))
    by={r["file"]:r for r in comp.get("files",[])}
    for p in ("protocol044_common.py","protocol044_engine.py","run_ftmoe_protocol044.py"):
        if p not in by or not by[p].get("unchanged"): compat=False
        elif by[p].get("current_sha256")!=sha(ROOT/p): compat=False
    for p in ("protocol044_stream.py","run_ftmoe_protocol044_science.py"):
        if p not in by or by[p].get("current_sha256")!=sha(ROOT/p): compat=False
    entry_ok=bool((entry.get("protocol"),entry.get("execution_revision"))==("044",2) and T(entry,"all_pass") and entry.get("synthetic_only") is True and
                  entry.get("real_model_forwards")==0 and entry.get("real_gradients")==0 and len(entry.get("fixtures",[]))>=15)
    ids={r.get("id"):r for r in entry.get("fixtures",[])}
    required=("C_ref_outputs_exact","D_lin_outputs_exact","D_no_gc_outputs_exact","D_bounded_outputs_exact","C_learner_exact",
              "Dlin_full_checkpoint_state_exact","D_no_gc_semantic_state_exact","D_bounded_semantic_state_exact","budget_exact",
              "fixture_gate_rejected_in_production","started_without_resume","resume_without_state","completed_rejected",
              "previous_sequence_incomplete","wrong_state_hash_rejected","old_cursor_rejected","global_budget_exhausted_rejected","wrong_arm_checkpoint_rejected")
    entry_ok=entry_ok and all(k in ids and T(ids[k],"pass") for k in required)
    durable_ok=bool((dur.get("protocol"),dur.get("execution_revision"))==("044",2) and T(dur,"all_pass") and
                    all(T(dur,k) for k in ("state0_remote_verified","fault_after_remote_upload","orphan_artifact_discovered",
                      "orphan_receipt_recovered","different_runner_continued","corruption_rejected","stale_parent_rejected")))
    publication_ok=bool((pub.get("protocol"),pub.get("execution_revision"))==("044",2) and T(pub,"all_pass") and
                        pub.get("model_forwards")==0 and pub.get("gradient_steps")==0 and pub.get("source_code_lock_reused_as_new_execution_proof") is False)
    recovery_ok=bool((inv.get("protocol"),inv.get("execution_revision"))==("044",2) and
                     inv.get("branch_selected")=="one_authorized_same_seed4401_from_zero_reconstruction" and
                     inv.get("generation_key")==GEN_KEY and inv.get("science_key")==SCI_KEY and
                     inv.get("reconstruction_allowance_consumed") is False and
                     inv["prior_r1_generation"]["remote_artifact_count"]==0 and inv["prior_r1_generation"]["complete_verified_checkpoint_found"] is False)
    execution=str(a.execution_sha);execution_ok=len(execution)==40 and all(c in "0123456789abcdef" for c in execution)
    bundle_files=("protocol044_common.py","protocol044_engine.py","run_ftmoe_protocol044.py","protocol044_stream.py",
      "run_ftmoe_protocol044_science.py","maintenance/protocol044_r2_durable.py","analyze_ftmoe_protocol044_r2.py",
      "maintenance/protocol044_r2_publish.py","maintenance/protocol044_r2_gate.py",
      "artifacts/ftmoe_online/protocol_044/plan.json","artifacts/ftmoe_online/protocol_044/scenario_registration.json",
      "artifacts/ftmoe_online/protocol_044/revision_002/plan.json","docs/PROTOCOL044_REVISION002_DURABLE_RECOVERY_20261005.md",
      "docs/PROTOCOL044_RESULTS.md")
    bundle={p:sha(ROOT/p) for p in bundle_files}
    gate={"protocol":"044","revision":1,"execution_revision":2,"science_config_revision":1,"execution_sha":execution,
      "r1_plan_sha256":R1_PLAN,"r2_plan_sha256":R2_PLAN,"scenario_sha256":SCENARIO,"generation_key":GEN_KEY,"science_key":SCI_KEY,
      "inherited_E":{"artifact_id":OLD_E_ID,"artifact_digest":OLD_E_DIGEST,"execution_sha":OLD_E_EXEC},
      "entrypoint_evidence":{"artifact_id":a.entry_artifact_id,"artifact_digest":a.entry_artifact_digest},
      "durable_evidence":{"artifact_id":a.durable_artifact_id,"artifact_digest":a.durable_artifact_digest},
      "inherited_E_pass":bool(inherited),"compatibility_pass":bool(compat),"production_entrypoint_pass":bool(entry_ok),
      "generation_recovery_pass":bool(durable_ok),"remote_transaction_pass":bool(durable_ok),
      "budget_idempotency_pass":bool(entry_ok and durable_ok),"publication_pass":bool(publication_ok),"recovery_inventory_pass":bool(recovery_ok),
      "source_bundle_hashes":bundle,"fixture_mode":False}
    gate["E_recovery_gate"]=bool(inherited and compat and entry_ok and durable_ok and publication_ok and recovery_ok and execution_ok)
    W(a.out,gate);print(json.dumps(gate,indent=2))
    if not gate["E_recovery_gate"]:raise SystemExit(2)
if __name__=="__main__":main()
