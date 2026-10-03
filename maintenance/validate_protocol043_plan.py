#!/usr/bin/env python3
"""Fail-closed validation for Protocol-043 revision 1 registration."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"artifacts/ftmoe_online/protocol_043/plan.json"
PLAN_SHA="9cd9b6f63aed0e35361f763e532d7c785a7838c19d172ea0c97bbaae5b2b0d6b"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,label):
    if not x: raise ValueError(label)
def main():
    req(sha(PLAN)==PLAN_SHA,"plan sha256")
    p=json.loads(PLAN.read_text(encoding="utf8"))
    req((p.get("protocol"),p.get("revision"))==("043",1),"registration identity")
    req(p.get("status")=="instructions_registered_not_implemented_not_started","registration status")
    req(sha(ROOT/p["instructions"])==p["instructions_sha256"],"directive sha256")
    req(p["authorization"]["execution_after_user_handoff"] is True,"execution authorization")
    req(p["authorization"]["historical033_042_budgets_closed"] is True,"historical budgets")
    req(p["execution_order"]==["audit042_read_only","synthetic_production_preflight","freeze_both_arms","D_no_gc","D_bounded","seal_and_analyze","stop"],"execution order")
    req(p["arms"]["order"]==["D_no_gc","D_bounded"] and p["arms"]["primary"]=="D_bounded","arms")
    req(p["arms"]["only_policy_difference"]=="permanent_reclamation","arm contrast")
    s=p["structure"]; req(s["active_max"]==2 and s["shadow_max"]==1 and s["resident_including_shadow_max"]==3,"capacity")
    req(s["candidate_attempts_after_E0_max"]==4 and s["ids_created_max"]==5 and s["id_reuse"] is False,"id budget")
    req(s["admission"]=="only_if_active_count_less_than_2_no_active_replacement","no replacement")
    q=p["score"]; req(q["fixed_window"]==128 and q["dtype"]=="float64_from_issued_float32","score")
    req(q["implementation"]=="per_interval_sufficient_statistics_ring128_incremental_sums","incremental tracker")
    req(q["unbounded_online_history"] is False and q["online_AP"] is False,"bounded score")
    req(q["eligible"]=={"overall_max":0,"positive_max":0,"negative_max":0,"removal_fpr_increase_max":0.01,"removal_recall_change_min":-0.02,"consecutive_due16":3},"sleep thresholds")
    r=p["reclamation"]; req(r["only_state"]=="dormant" and r["minimum_dormant_predictions"]==256,"reclamation state")
    req(r["one_delete_per_birth_max"]==1 and r["deletions_per_arm_max"]==4 and r["delete_and_create_atomic"] is True,"reclamation budget")
    b=p["budget"]; req(b["science_sequences"]==2 and b["total"]["optimizer_steps"]==1536,"science budget")
    req(b["each"]["live_optimizer_steps"]==704 and b["each"]["shadow_optimizer_steps"]==64 and b["each"]["total_optimizer_steps"]==768,"per arm steps")
    req(b["new_streams"]==b["extra_seeds"]==b["F_loads"]==b["extra_arms"]==b["U_reproduction_runs"]==b["real_engineering_gradients"]==0,"forbidden budget")
    req(b["restart_from_zero"] is False,"no restart")
    e=p["engineering"]; req(len(e["required_fixture_ids"])==14,"fixture count")
    req(e["production_tick_required"] and e["synthetic_gradients_only"] and e["resume_must_continue_to_same_endpoint"],"engineering semantics")
    req(e["missing_or_nonboolean_pass"]=="fail_closed","fail closed")
    a=p["analysis"]; req(a["primary_arm"]=="D_bounded" and a["primary_baseline"]=="C_ref","primary comparison")
    req(a["required_superiority_baselines"]==["C_ref"] and not a["require_win_over_no_gc"] and not a["require_preserve_D_keep"],"success scope")
    rt=p["runtime"]; req(rt["workflow"]=="protocol043-bounded-lifecycle.yml" and rt["formal_trigger"]=="workflow_dispatch_only_after_user_handoff","workflow")
    req(rt["execution_branch"]=="codex/protocol-043-bounded-lifecycle-20261003" and rt["exact_resume_only"] and rt["no_force_push"],"runtime")
    d=p["delivery"]; req(d["raw_artifact_first"] and d["score_metrics_independent_full_recompute"] and d["stop_after_two_or_blocked"],"delivery")
    # Historical 041 manifest is a registration lock, not a mutable runtime hint.
    mp=ROOT/p["source041"]["manifest_path"]; req(sha(mp)==p["source041"]["manifest_sha256"],"041 manifest sha")
    entries={x["path"]:x["sha256"] for x in json.loads(mp.read_text(encoding="utf8"))["files"]}
    for rel,dig in p["source041"]["files"].items(): req(entries.get(rel)==dig,"041 lock "+rel)
    print(json.dumps({"protocol":"043","revision":1,"registration_valid":True,"plan_sha256":PLAN_SHA,"science_started":False},indent=2))
if __name__=="__main__": main()
