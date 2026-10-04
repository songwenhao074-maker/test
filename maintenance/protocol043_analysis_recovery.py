"""Read-only Protocol043 analysis recovery for NaN-aware no-GC trajectory equality."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, shutil, subprocess, sys

FROZEN_SHA="9d9320cf84ebaf034dca9e8257f0dc82171cc916"
SCIENCE_RUN_ID="37168746303"
RAW_ARTIFACT_ID=11290523632
RAW_DIGEST="sha256:6387aa37df285bf8b1842b5f1e57250a55cb3bff6004e9d4cd3b57ccde5877b3"
MANIFEST_SHA="73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b"

def J(p): return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=pathlib.Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def req(x,label):
    if not x: raise RuntimeError(label)
def run(cmd,cwd):
    print("+"," ".join(str(x) for x in cmd),flush=True)
    subprocess.run([str(x) for x in cmd],cwd=str(cwd),check=True)

def verify_raw(raw):
    raw=pathlib.Path(raw); sci=raw/"science"
    led=J(sci/"budget_ledger.json")
    req((led.get("protocol"),led.get("revision"))==("043",1),"budget identity")
    req(led.get("order")==["D_no_gc","D_bounded"],"arm order")
    for arm in ("D_no_gc","D_bounded"):
        r=led["sequences"][arm]
        req(type(r.get("started")) is bool and r["started"],arm+" started")
        req(type(r.get("completed")) is bool and r["completed"],arm+" completed")
        req(r.get("restart_from_zero") is False and r.get("resume_events")==[],arm+" restart/resume")
        au=J(sci/arm/"run_audit.json"); req(type(au.get("all_pass")) is bool and au["all_pass"],arm+" audit")
        st=J(sci/arm/"status.json"); req(st.get("science_status")=="completed",arm+" status")
    req(led["total_optimizer_steps"]==1224 and led["total_optimizer_steps"]<=1536,"optimizer budget")
    req(led["new_streams"]==led["extra_seeds"]==led["F_loads"]==led["extra_arms"]==led["U_reproduction_runs"]==0,"forbidden budgets")
    req(led["restart_from_zero"] is False,"no restart")
    gate=J(sci/"gate_report.json"); req(type(gate.get("all_pass")) is bool and gate["all_pass"],"gate")
    fix=J(sci/"fixture_report.json"); req(type(fix.get("all_pass")) is bool and fix["all_pass"],"fixture")
    a42=J(raw/"audit042/gate.json"); req(type(a42.get("all_pass")) is bool and a42["all_pass"],"audit042")
    ps=J(sci/"protocol_status.json"); req(ps.get("science_status")=="completed" and ps.get("science_rc")==0 and ps.get("restart_from_zero") is False,"protocol status")
    man=J(sci/"scientific_raw_manifest.json"); req((man.get("protocol"),man.get("revision"))==("043",1),"raw manifest identity")
    bad=[]
    for row in man.get("files",[]):
        p=raw/row["path"]
        if not p.exists() or p.stat().st_size!=row["size"] or sha(p)!=row["sha256"]: bad.append(row["path"])
    req(not bad,"raw manifest mismatch "+",".join(bad[:10]))
    return {"raw_manifest_file_count":man.get("file_count"),"optimizer_steps":led["total_optimizer_steps"],"science_complete":True}

def patch_analyzer(repo,analysis,recovery_run):
    repo=pathlib.Path(repo); analysis=pathlib.Path(analysis); analysis.mkdir(parents=True,exist_ok=True)
    p=repo/"analyze_ftmoe_protocol043.py"; original=p.read_bytes(); s=original.decode("utf8")
    old='''    exact=bool(np.array_equal(A["probability"],N["probability"]) and np.array_equal(A["active_ids"],N["active_ids"]) and np.array_equal(A["contribution"],N["contribution"]))
    logs={}
    for fn,key in names:
        x=J(root/"D_bounded"/(fn+".json"))[key]; y=J(root/"D_no_gc"/(fn+".json"))[key]; logs[fn]=norm(x)==norm(y)
    return {"pass":bool(exact and all(logs.values())),"gc_exercised":False,"whole_trajectory_exact":exact,"logs_exact":logs}'''
    new='''    array_keys=("probability","detection_logits","total_delta","live_margin","B_margin","deployment_epoch","active_ids","active_count","shadow_present","accepted_count","expert_versions","expert_hashes","contribution","labels","raw_labels","class_probability","model_version")
    array_exact={}
    for k in array_keys:
        x=A[k]; y=N[k]
        if np.issubdtype(x.dtype,np.floating):
            array_exact[k]=bool(np.array_equal(x,y,equal_nan=True))
        else:
            array_exact[k]=bool(np.array_equal(x,y))
    exact=bool(all(array_exact.values()))
    log_specs=(("birth_checks","checks"),("pressure_checks","checks"),("utility_checks","checks"),("sleep_tables","tables"),
      ("lifecycle_events","events"),("candidate_decisions","decisions"),("reuse_decisions","decisions"),
      ("opportunity_log","opportunities"),("update_log","updates"),("settlement_log","settlements"),
      ("qualification_records","records"),("reclamation_events","events"),("tombstones","tombstones"),("reuse_history","slots"))
    logs={}
    for fn,key in log_specs:
        x=J(root/"D_bounded"/(fn+".json"))[key]; y=J(root/"D_no_gc"/(fn+".json"))[key]; logs[fn]=norm(x)==norm(y)
    summaries_equal=norm(J(root/"D_bounded/summary.json"))==norm(J(root/"D_no_gc/summary.json"))
    return {"pass":bool(exact and all(logs.values()) and summaries_equal),"gc_exercised":False,
      "whole_trajectory_exact":exact,"array_exact":array_exact,"logs_exact":logs,"summaries_exact":bool(summaries_equal),
      "nan_equality":"equal_nan_true_for_float_arrays"}'''
    req(old in s,"frozen analyzer patch target not found")
    patched=s.replace(old,new).encode("utf8"); p.write_bytes(patched)
    rec={"protocol":"043","revision":1,"science_run_id":SCIENCE_RUN_ID,"analysis_recovery_run_id":str(recovery_run),
      "frozen_execution_sha":FROZEN_SHA,"raw_artifact_id":RAW_ARTIFACT_ID,"raw_artifact_digest":RAW_DIGEST,
      "original_analyzer_sha256":hashlib.sha256(original).hexdigest(),"recovered_analyzer_sha256":hashlib.sha256(patched).hexdigest(),
      "patch_scope":"no_gc_trajectory_equality_only_nan_aware_and_expanded_to_all_prediction_arrays_14_logs_and_summary",
      "scientific_metrics_or_thresholds_changed":False,"lifecycle_policy_changed":False,"science_reexecuted":False,
      "original_failure":"np.array_equal treats matching NaN contribution cells as unequal by default"}
    W(analysis/"analysis_recovery.json",rec); return rec

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True); ap.add_argument("--raw",required=True); ap.add_argument("--analysis",required=True)
    ap.add_argument("--science-run-id",required=True); ap.add_argument("--recovery-run-id",required=True)
    a=ap.parse_args()
    req(str(a.science_run_id)==SCIENCE_RUN_ID,"science run id")
    raw=pathlib.Path(a.raw).resolve(); repo=pathlib.Path(a.repo).resolve(); analysis=pathlib.Path(a.analysis).resolve()
    audit=verify_raw(raw); rec=patch_analyzer(repo,analysis,a.recovery_run_id)
    manifest=repo/"artifacts/ftmoe_online/protocol_036/runs/run_36831958978/seed3601/stream_manifest.json"
    req(sha(manifest)==MANIFEST_SHA,"phase manifest hash")
    docs=repo/"docs/PROTOCOL043_RESULTS.md"
    run([sys.executable,"analyze_ftmoe_protocol043.py","--science-root",raw/"science","--manifest",manifest,
      "--fixture-report",raw/"science/fixture_report.json","--audit042-gate",raw/"audit042/gate.json",
      "--gate-report",raw/"science/gate_report.json","--output-dir",analysis,"--docs-output",docs,
      "--run-id",SCIENCE_RUN_ID],repo)
    shutil.copy2(docs,analysis/"PROTOCOL043_RESULTS.md")
    st=J(analysis/"analysis_status.json"); div=J(analysis/"first_control_divergence.json")
    req(type(st.get("validity")) is bool and st["validity"],"recovered validity not true")
    req(st.get("D_over_C") is True and st.get("core_goal_supported") is True,"registered core goal not recovered")
    req(st.get("gc_exercised_on_stream") is False and st.get("full_reclamation_demonstrated") is False,"GC claim changed")
    req(div.get("pass") is True and div.get("whole_trajectory_exact") is True,"recovered no-GC equality failed")
    W(analysis/"analysis_recovery.json",{**rec,"raw_validation":audit,"recovered_system_result_label":st["system_result_label"],
      "recovered_validity":st["validity"],"recovered_core_goal_supported":st["core_goal_supported"]})
    print(json.dumps({"analysis_status":st,"divergence":div,"recovery":rec},indent=2,ensure_ascii=False))

if __name__=="__main__": main()
