"""Protocol-030: exactly two same-job D replays, audit_off then audit_on."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol027_pilot as p27
import run_ftmoe_protocol028_pilot as p28
import run_ftmoe_protocol029_diagnostic as p29
from ftmoe_protocol024_eval import binary_detection_metrics
from ftmoe_protocol027_data import EXPECTED_STREAM_SHA, REVISION_ID, verify_frozen
from ftmoe_protocol028_memory_protected import Protocol028DynamicSession
from ftmoe_protocol030_paired import (
    Protocol030DiagnosticSession, algorithm_state_summary, set_deterministic_runtime
)

RECURRENCE=tuple(p27.RECURRENCE)
MODES=("audit_off","audit_on")

def write_json(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")

def write_jsonl(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf8") as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+"\n")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]

def load_npz(path):
    with np.load(path) as d:return {k:d[k].copy() for k in d.files}

def _lifecycle_summary(session):
    ctrl=session.lifecycle_controller
    events=list(ctrl.events); ec=Counter(e["kind"] for e in events)
    records=list(ctrl.candidate_records.values())
    return {
        "births":int(ec.get("candidate_accepted",0)),
        "retirements":int(ctrl.retirements),
        "reactivations":int(ctrl.reactivations),
        "purges":int(ctrl.purges),
        "capacity_preserve_skips":int(ctrl.capacity_preserve_skips),
        "accepted_birth_ids":list(ctrl.accepted_ids),
        "resident_memory_ids":sorted(ctrl.specialist_memory.keys(),key=int),
        "candidate_created":len(records),
        "candidate_accepted":sum(r.get("accepted") is True for r in records),
        "candidate_rejected":sum(r.get("accepted") is False for r in records),
        "candidate_pending":sum(r.get("accepted") is None for r in records),
        "opportunity_accounting":ctrl.due_conservation(),
        "final_topology":session.model.learner.topology_manifest(),
        "event_counts":dict(ec),
    }

def _checkpoint(session,index,label):
    row=algorithm_state_summary(session)
    row.update({"prediction_index":int(index),"label":str(label)})
    return row

def _runtime_registration(reg,mode):
    return {
        "protocol":"030","mode":mode,"diagnostic_only":True,
        "online_method":"Protocol028 D_memory_protected unchanged",
        "protocol030_registration":reg,
        "stream_sha256":EXPECTED_STREAM_SHA,
        "replay_seed":700,"model_seed":1,"C_rerun":False,
        "post_hoc_oracle_is_online_method":False,
    }

def run_child(args):
    runtime=set_deterministic_runtime()
    out=Path(args.arm_dir); out.mkdir(parents=True,exist_ok=False)
    stream=Path(args.stream); verify_frozen(stream)
    reg=load_json(args.registration)
    if reg.get("protocol")!="030" or reg.get("replay_seed")!=700 or reg.get("model_seed")!=1:
        raise AssertionError("Protocol030 registration mismatch")
    if reg.get("full_replay_budget")!=2 or reg.get("rerun_C") is not False:
        raise AssertionError("Protocol030 execution budget mismatch")
    source028=load_json(args.protocol028_registration)
    if source028.get("protocol")!="028" or source028.get("seeds")!={"replay":700,"model":1}:
        raise AssertionError("source Protocol028 registration mismatch")
    bundle=s4.build_replay(stream); bundle["stream_dir"]=str(stream)
    stream_sha=p27.sha(stream/"stream.npz")
    if stream_sha!=EXPECTED_STREAM_SHA or bundle["manifest"].get("stream_sha256")!=EXPECTED_STREAM_SHA:
        raise AssertionError("Protocol030 stream mismatch")
    phases=p27.phases(bundle["manifest"]); guard=p28.build_guard(bundle)
    common=dict(
        seed=1,replay_bundle=bundle,budget=p27.budget(),out_dir=out,
        run_id=f"protocol030_{args.run_id}_{args.mode}",stream_dir=stream,
        phase_defs=phases,stream_sha=stream_sha,
        registration=_runtime_registration(reg,args.mode),learning_rate=1e-4,
    )
    session=(Protocol028DynamicSession(guard_anchor=guard,**common)
             if args.mode=="audit_off"
             else Protocol030DiagnosticSession(guard_anchor=guard,**common))
    trace=[_checkpoint(session,-1,"initialized")]
    wall0=time.perf_counter(); cpu0=time.process_time()
    for i in range(session.steps):
        session.step()
        if (i+1)%256==0 or i+1==session.steps:
            trace.append(_checkpoint(session,i,"post_step"))
    session.lifecycle_controller.mark_stream_end()
    if args.mode=="audit_on":
        session.lifecycle_controller.finalize_passive_audits(session.steps)
    session.finish()
    wall=time.perf_counter()-wall0; cpu=time.process_time()-cpu0
    trace.append(_checkpoint(session,session.steps,"post_finish"))
    session.save()
    ctrl=session.lifecycle_controller
    life=_lifecycle_summary(session)
    write_jsonl(out/"lifecycle.jsonl",ctrl.events)
    write_json(out/"lifecycle_summary.json",life)
    write_json(out/"state_trace.json",trace)
    prob=session.predictions["probability"]; cls=session.predictions["class_probability"]
    labels=session.predictions["labels"]
    summary={
        "protocol":"030","mode":args.mode,"completed":True,
        "stream_sha256":stream_sha,"data_revision":REVISION_ID,
        "runtime_profile":runtime,
        "full":p27.metrics(prob,cls,labels),
        "phases":p27.phase_metrics(prob,cls,labels,phases),
        "lifecycle":life,
        "cost":{
            "wall_seconds":float(wall),"cpu_seconds":float(cpu),
            "p95_inference_seconds":float(np.percentile(session.predictions["prediction_seconds"],95)),
            "max_rss_bytes":p27.rss_bytes(),
        },
    }
    if args.mode=="audit_on":
        p29.save_passive_predictions(ctrl,out/"passive_predictions.npz")
        write_json(out/"opportunity_audit_full.json",ctrl.opportunity_audit)
        compact=p29.compact_opportunity_audit(ctrl.opportunity_audit)
        write_json(out/"opportunity_audit.json",compact)
        source_cmp=load_json(Path(args.source_028_root)/"comparison.json")
        windows=p29.window_memory_utility(session,ctrl,stream,source_cmp)
        write_json(out/"window_memory_utility.json",windows)
        opportunities=p29.summarize_opportunities(ctrl.opportunity_audit)
        passing=[]
        for op in ctrl.opportunity_audit:
            for key,row in op["candidate_validation"].items():
                decision=row.get("decision")
                if decision and decision.get("all_gates_pass"):
                    passing.append({
                        "opportunity_id":op["opportunity_id"],"due_cursor":op["due_cursor"],
                        "expert_id":key,"controller":op["controller"],
                        "similarity":op["similarity"].get("candidates",{}).get(key),
                        "decision":decision,"post_hoc_oracle_diagnostic":True,
                    })
        summary["passive"]={
            "state_checks":int(ctrl.passive_state_checks),
            "state_check_failures":int(ctrl.passive_state_check_failures),
            "prediction_overhead_seconds":float(ctrl.passive_overhead_seconds),
            "guard_overhead_seconds":float(ctrl.passive_guard_seconds),
            "prediction_forwards":int(ctrl.passive_prediction_forwards),
            "guard_evaluations":int(ctrl.passive_guard_forwards),
            "opportunities":opportunities,
            "all_gates_passing_candidates":passing,
            "all_gates_passing_candidate_count":len(passing),
            "nine_recurrence_windows":windows,
        }
    write_json(out/"summary.json",summary)
    return summary

def _max_error(a,b):
    delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
    flat=int(np.argmax(delta)); loc=list(np.unravel_index(flat,delta.shape))
    return {
        "max_abs_error":float(delta.flat[flat]),
        "max_error_index":loc,
        "exact_equal_fraction":float(np.mean(a==b)),
        "a_value_at_max":float(a[tuple(loc)]),
        "b_value_at_max":float(b[tuple(loc)]),
    }

def _window_aps(pred,labels,stream):
    phases={p["name"]:p for p in p27.phases(load_json(Path(stream)/"manifest.json"))}
    rows=[]
    for name in RECURRENCE:
        p=phases[name]; a=int(p["start"]); b=min(int(p["end"]),a+100)
        m=binary_detection_metrics(pred[a:b],labels[a:b],.5)
        rows.append({"phase":name,"intervals":[a,b],"ap":m["ap"],"fpr":m["fpr"]})
    vals=[r["ap"] for r in rows]
    return rows, (None if any(v is None for v in vals) else float(np.mean(vals)))

def _events_equal(a_path,b_path):
    a=[p29._discrete_event(x) for x in load_jsonl(a_path)]
    b=[p29._discrete_event(x) for x in load_jsonl(b_path)]
    first=None
    for i in range(min(len(a),len(b))):
        if a[i]!=b[i]:
            first={"event_index":i,"audit_off":a[i],"audit_on":b[i]}; break
    if first is None and len(a)!=len(b):
        first={"event_index":min(len(a),len(b)),"audit_off_count":len(a),"audit_on_count":len(b)}
    return a==b, first, len(a), len(b)

def _state_trace_compare(a_path,b_path):
    a=load_json(a_path); b=load_json(b_path)
    keys=("model_state_sha256","optimizer_state_sha256","controller_state_sha256",
          "rng_state_sha256","topology","module_modes","requires_grad","last_z_sha256",
          "cursor","updates")
    first=None
    for i in range(min(len(a),len(b))):
        for k in keys:
            if a[i].get(k)!=b[i].get(k):
                first={"checkpoint_index":i,"prediction_index":a[i].get("prediction_index"),
                       "field":k,"audit_off":a[i].get(k),"audit_on":b[i].get(k)}
                return False,first
    if len(a)!=len(b):
        return False,{"field":"state_trace_length","audit_off":len(a),"audit_on":len(b)}
    return True,None

def compare_pair(root,stream):
    root=Path(root); off=load_npz(root/"audit_off"/"predictions.npz")
    on=load_npz(root/"audit_on"/"predictions.npz")
    det=_max_error(off["probability"],on["probability"])
    cls=_max_error(off["class_probability"],on["class_probability"])
    labels_exact=bool(np.array_equal(off["labels"],on["labels"]))
    raw_exact=bool(np.array_equal(off["raw_labels"],on["raw_labels"]))
    off_rows,off_mean=_window_aps(off["probability"],off["labels"],stream)
    on_rows,on_mean=_window_aps(on["probability"],on["labels"],stream)
    win=[]
    for a,b in zip(off_rows,on_rows):
        diff=None if a["ap"] is None or b["ap"] is None else float(abs(a["ap"]-b["ap"]))
        win.append({"phase":a["phase"],"audit_off":a,"audit_on":b,"ap_abs_difference":diff})
    each_ok=all(r["ap_abs_difference"] is not None and r["ap_abs_difference"]<=1e-6 for r in win)
    mean_diff=None if off_mean is None or on_mean is None else float(abs(off_mean-on_mean))
    off_full=binary_detection_metrics(off["probability"],off["labels"],.5)["ap"]
    on_full=binary_detection_metrics(on["probability"],on["labels"],.5)["ap"]
    full_diff=None if off_full is None or on_full is None else float(abs(off_full-on_full))
    lifecycle_equal,life_first,na,nb=_events_equal(
        root/"audit_off"/"lifecycle.jsonl",root/"audit_on"/"lifecycle.jsonl")
    off_life=load_json(root/"audit_off"/"lifecycle_summary.json")
    on_life=load_json(root/"audit_on"/"lifecycle_summary.json")
    topology_equal=off_life["final_topology"]==on_life["final_topology"]
    state_exact,state_first=_state_trace_compare(
        root/"audit_off"/"state_trace.json",root/"audit_on"/"state_trace.json")
    on_summary=load_json(root/"audit_on"/"summary.json")
    violations=int(on_summary.get("passive",{}).get("state_check_failures",-1))
    valid=bool(
        det["max_abs_error"]<=1e-6 and cls["max_abs_error"]<=1e-6
        and labels_exact and raw_exact and each_ok
        and mean_diff is not None and mean_diff<=1e-6
        and full_diff is not None and full_diff<=1e-6
        and lifecycle_equal and topology_equal and violations==0
        and state_exact
    )
    first=None
    checks=[
        ("detection_probability",det["max_abs_error"]<=1e-6,det),
        ("class_probability",cls["max_abs_error"]<=1e-6,cls),
        ("labels",labels_exact,{"labels_exact":labels_exact}),
        ("raw_labels",raw_exact,{"raw_labels_exact":raw_exact}),
        ("recurrence_window_ap",each_ok,win),
        ("mean_recurrence_ap",mean_diff is not None and mean_diff<=1e-6,{"difference":mean_diff}),
        ("full_stream_ap",full_diff is not None and full_diff<=1e-6,{"difference":full_diff}),
        ("lifecycle",lifecycle_equal,life_first),
        ("final_topology",topology_equal,{"audit_off":off_life["final_topology"],"audit_on":on_life["final_topology"]}),
        ("passive_state_violations",violations==0,{"violations":violations}),
        ("state_trace",state_exact,state_first),
    ]
    for name,ok,detail in checks:
        if not ok:first={"field":name,"detail":detail}; break
    return {
        "paired_audit_valid":valid,
        "thresholds":{"probability":1e-6,"each_recurrence_ap":1e-6,"mean_recurrence_ap":1e-6,"full_ap":1e-6},
        "detection_probability":det,"class_probability":cls,
        "labels_exact":labels_exact,"raw_labels_exact":raw_exact,
        "recurrence_first100":win,
        "audit_off_mean_recurrence_ap":off_mean,"audit_on_mean_recurrence_ap":on_mean,
        "mean_recurrence_ap_abs_difference":mean_diff,
        "audit_off_full_ap":off_full,"audit_on_full_ap":on_full,
        "full_ap_abs_difference":full_diff,
        "discrete_lifecycle_events_exact":lifecycle_equal,
        "audit_off_event_count":na,"audit_on_event_count":nb,
        "final_topology_exact":topology_equal,
        "state_trace_exact":state_exact,
        "passive_state_violations":violations,
        "first_divergence":first,
    }

def compare_historical(arm_root,source_root,stream):
    cur=load_npz(Path(arm_root)/"predictions.npz")
    src=load_npz(Path(source_root)/"D_memory_protected"/"predictions.npz")
    det=_max_error(src["probability"],cur["probability"])
    cls=_max_error(src["class_probability"],cur["class_probability"])
    src_rows,src_mean=_window_aps(src["probability"],src["labels"],stream)
    cur_rows,cur_mean=_window_aps(cur["probability"],cur["labels"],stream)
    wd=[None if a["ap"] is None or b["ap"] is None else float(abs(a["ap"]-b["ap"]))
        for a,b in zip(src_rows,cur_rows)]
    src_full=binary_detection_metrics(src["probability"],src["labels"],.5)["ap"]
    cur_full=binary_detection_metrics(cur["probability"],cur["labels"],.5)["ap"]
    full_diff=None if src_full is None or cur_full is None else float(abs(src_full-cur_full))
    life_equal,first,_,_=_events_equal(
        Path(source_root)/"D_memory_protected"/"lifecycle.jsonl",
        Path(arm_root)/"lifecycle.jsonl")
    match=bool(det["max_abs_error"]<=1e-6 and cls["max_abs_error"]<=1e-6
               and np.array_equal(src["labels"],cur["labels"])
               and np.array_equal(src["raw_labels"],cur["raw_labels"])
               and all(x is not None and x<=1e-6 for x in wd)
               and abs(src_mean-cur_mean)<=1e-6
               and full_diff is not None and full_diff<=1e-6 and life_equal)
    return {
        "historical_028_match":match,
        "detection_probability":det,"class_probability":cls,
        "labels_exact":bool(np.array_equal(src["labels"],cur["labels"])),
        "raw_labels_exact":bool(np.array_equal(src["raw_labels"],cur["raw_labels"])),
        "recurrence_ap_abs_differences":dict(zip(RECURRENCE,wd)),
        "mean_recurrence_ap_abs_difference":float(abs(src_mean-cur_mean)),
        "full_ap_abs_difference":full_diff,
        "discrete_lifecycle_events_exact":life_equal,
        "first_lifecycle_divergence":first,
        "note":"Historical Protocol028 used the registered three-thread profile; Protocol030 uses a preregistered single-thread deterministic profile.",
    }

def parent(args):
    root=Path(args.out_root); root.mkdir(parents=True,exist_ok=True)
    reg=load_json(args.registration)
    if reg.get("protocol")!="030" or reg.get("full_replay_budget")!=2:
        raise RuntimeError("wrong Protocol030 registration")
    verify_frozen(args.stream)
    lock=load_json(args.data_lock); elig=load_json(args.eligibility)
    if not lock.get("locked") or not elig.get("protocol027_data_eligible"):
        raise RuntimeError("inherited data eligibility failed")
    source=load_json(args.source_artifacts)
    for prefix,expected in (
        ("frozen",reg["source_data"]),("protocol028",reg),("protocol029",reg)):
        pass
    completed=[]; failures={}
    for mode in MODES:
        cmd=[sys.executable,str(Path(__file__).resolve()),"--mode",mode,
             "--stream",args.stream,"--arm-dir",str(root/mode),
             "--registration",args.registration,
             "--protocol028-registration",args.protocol028_registration,
             "--source-028-root",args.source_028_root,
             "--run-id",str(args.run_id)]
        so=root/f"{mode}.stdout.log"; se=root/f"{mode}.stderr.log"
        with so.open("w") as o,se.open("w") as e:
            proc=subprocess.run(cmd,stdout=o,stderr=e,text=True,env=dict(os.environ))
        if proc.returncode==0 and (root/mode/"summary.json").is_file() and (root/mode/"predictions.npz").is_file():
            completed.append(mode)
        else:
            failures[mode]={"returncode":proc.returncode,"stderr_tail":se.read_text(errors="replace")[-12000:]}
            break
    pair=None; hist=None; diagnostic=None
    if completed==list(MODES):
        pair=compare_pair(root,args.stream); write_json(root/"paired_consistency.json",pair)
        write_json(root/"first_divergence.json",pair.get("first_divergence"))
        hist={
            "audit_off":compare_historical(root/"audit_off",args.source_028_root,args.stream),
            "audit_on":compare_historical(root/"audit_on",args.source_028_root,args.stream),
        }
        write_json(root/"historical_comparison.json",hist)
        on=load_json(root/"audit_on"/"summary.json")
        passive=on["passive"]
        diagnostic={
            "protocol":"030","run_id":str(args.run_id),
            "paired_audit_valid":bool(pair["paired_audit_valid"]),
            "historical_028_match":hist,
            "online_method_changed":False,"new_scientific_methods":0,"C_rerun":False,
            "opportunities":passive["opportunities"],
            "all_gates_passing_candidates":passive["all_gates_passing_candidates"],
            "all_gates_passing_candidate_count":passive["all_gates_passing_candidate_count"],
            "nine_recurrence_windows":passive["nine_recurrence_windows"],
            "post_hoc_oracle_diagnostic":True,
            "interpretation_limit":"Paired validity establishes noninterference only under the registered Protocol030 runtime; oracle candidates are not an online policy or D>C evidence.",
            "cost":{
                "audit_off":load_json(root/"audit_off"/"summary.json")["cost"],
                "audit_on":on["cost"],
                "passive_prediction_overhead_seconds":passive["prediction_overhead_seconds"],
                "passive_guard_overhead_seconds":passive["guard_overhead_seconds"],
            },
        }
        write_json(root/"diagnostic_summary.json",diagnostic)
        write_json(root/"opportunity_audit.json",load_json(root/"audit_on"/"opportunity_audit.json"))
    status={
        "protocol":"030","run_id":str(args.run_id),"completed":completed==list(MODES),
        "completed_executions":completed,"full_replays_started":len(completed)+int(bool(failures)),
        "full_replay_budget":2,"paired_audit_valid":None if pair is None else bool(pair["paired_audit_valid"]),
        "new_scientific_methods":0,"C_rerun":False,"automatic_followups_started":[],
        "failure_details":failures,
        "blocker":("paired_execution_failed" if completed!=list(MODES)
                   else (None if pair["paired_audit_valid"] else "paired_noninterference_gate_failed")),
    }
    write_json(root/"status.json",status)
    print(json.dumps({"status":status,"paired":pair},indent=2,allow_nan=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stream",required=True)
    ap.add_argument("--out-root")
    ap.add_argument("--arm-dir")
    ap.add_argument("--mode",choices=MODES)
    ap.add_argument("--registration",required=True)
    ap.add_argument("--protocol028-registration",required=True)
    ap.add_argument("--source-028-root",required=True)
    ap.add_argument("--source-artifacts")
    ap.add_argument("--data-lock")
    ap.add_argument("--eligibility")
    ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    try:
        if a.mode:
            if not a.arm_dir:ap.error("--arm-dir required")
            run_child(a)
        else:
            if not all((a.out_root,a.source_artifacts,a.data_lock,a.eligibility)):
                ap.error("--out-root/source-artifacts/data-lock/eligibility required")
            parent(a)
    except Exception:
        root=Path(a.out_root or a.arm_dir or "."); root.mkdir(parents=True,exist_ok=True)
        write_json(root/"runner_exception.json",{"protocol":"030","traceback":traceback.format_exc()})
        raise

if __name__=="__main__":main()
