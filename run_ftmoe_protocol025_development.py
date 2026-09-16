"""Run the preregistered Protocol-025 seed700/model1 development comparison.

This runner MUST only be invoked after the model-free data audit has passed and
the immutable stream SHA has been locked. It runs the complete registered
comparator set once; no threshold/LR/service tuning is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_eval import (
    binary_detection_metrics, positive_resource_macro_f1,
    temporal_onset_metrics,
)
from ftmoe_protocol025_session import Protocol025FixedSession, Protocol025DynamicSession

COMPARATORS = ("C_fixed4","C_fixed5","C_fixed8_dense","C_fixed8_top5","D_dynamic")
RECURRENCE = ("S1_rec1","S3_rec1","S2_rec1","S4_rec1","S2_rec2","S6_rec1","S1_rec2","S5_rec1","S3_rec2")


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")


def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf8") as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+"\n")


def budget():
    out=dict(json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))["frozen_configuration"])
    out.update({"update_every_scored_intervals":4,"batch_size":32,"gradient_steps_per_opportunity":1,"replay_buffer_intervals":64,"learning_rate":1e-4})
    return out


def phase_defs(manifest):
    return [{"name":p["name"],"start":int(p["start"]),"end":int(p["end"]),"regime":p.get("service")} for p in manifest["timeline"]]


def build_f0_guard(bundle):
    """Causal old-knowledge guard: F0-internal raw-next rows, position mod5==0."""
    phases={p["name"]:p for p in phase_defs(bundle["manifest"])}
    f0=phases["F0"]
    # Last F0 prediction is excluded because its next target is the first S1 row.
    candidates=np.arange(int(f0["start"]),int(f0["end"])-1,dtype=np.int64)
    indices=candidates[(candidates-int(f0["start"]))%5==0]
    parts={k:[] for k in ("x","schedule","graph_x","labels","ids","before","caps")}
    raw=np.asarray(bundle["arrays"]["raw_labels"],dtype=np.int64)
    for left in range(0,len(indices),32):
        batch=indices[left:left+32].tolist(); x,s,g,context=s4.window_batch(bundle["replay"],batch)
        parts["x"].append(x); parts["schedule"].append(s); parts["graph_x"].append(g)
        parts["ids"].append(context["creation_ids"]); parts["before"].append(context["before_placement"]); parts["caps"].append(context["capacities"])
        parts["labels"].append(torch.as_tensor(raw[np.asarray(batch)+1],dtype=torch.long))
    out={k:torch.cat(v,dim=0) for k,v in parts.items()}
    y=out["labels"].numpy().reshape(-1)
    normal=int((y==0).sum()); positive=int((y>0).sum())
    if normal==0 or positive==0:
        raise RuntimeError("Protocol025 F0 guard lacks both normal and positive rows; stop before model results")
    out["meta"]={"role":"candidate_guard","source_phase":"F0","indices":indices.tolist(),"target":"raw_next_fault","last_index_excluded_to_keep_target_inside_F0":True,"split":"F0 position modulo5==0","normal_rows":normal,"positive_rows":positive,"causally_available_before_first_candidate_qualification":True}
    return out


def metric_block(prob, cls, labels, raw):
    return {"detection":binary_detection_metrics(prob,labels),"resource":positive_resource_macro_f1(cls,labels),"onset":temporal_onset_metrics(prob,raw,1)}


def learner_parameter_bytes(session):
    bank=session.model.learner
    return int(sum(p.numel()*p.element_size() for p in bank.parameters()))


def resident_parameter_bytes(session):
    bank=session.model.learner
    return int(sum(p.numel()*p.element_size() for p in bank.parameters()))


def max_rss_bytes():
    # Linux ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def sampled_training_examples(session):
    return int(sum(int(row.get("batch_size",0))*int(row.get("gradient_steps",1)) for row in session.update_log))


def run_one(name,bundle,out_root,guard,registration,stream_dir,stream_sha):
    out=Path(out_root)/name
    reg={"protocol":"025","comparator":name,"method_registration":registration,"stream_sha256":stream_sha,"development_seed":700,"model_seed":1,"confirmation_run":False}
    common=dict(seed=1,replay_bundle=bundle,budget=budget(),out_dir=out,run_id="protocol025_seed700_model1_"+name,stream_dir=stream_dir,phase_defs=phase_defs(bundle["manifest"]),stream_sha=stream_sha,registration=reg,learning_rate=1e-4)
    if name=="D_dynamic":
        session=Protocol025DynamicSession(guard_anchor=guard,**common)
    else:
        session=Protocol025FixedSession(name,**common)
    rss0=max_rss_bytes(); wall0=time.perf_counter(); cpu0=time.process_time(); peak_resident=resident_parameter_bytes(session); peak_active=0
    for _ in range(session.steps):
        session.step()
        if name=="D_dynamic":
            peak_resident=max(peak_resident,resident_parameter_bytes(session))
            peak_active=max(peak_active,len(session.model.learner.ids))
    session.finish(); wall=time.perf_counter()-wall0; cpu=time.process_time()-cpu0; session.save()
    p=session.predictions["probability"]; c=session.predictions["class_probability"]; labels=session.predictions["labels"]; raw=session.predictions["raw_labels"]
    if (labels<0).any() or (raw<0).any(): raise AssertionError("unsettled Protocol025 outputs")
    summary={
        "protocol":"025","comparator":name,"development_only":True,"confirmation_run":False,"stream_sha256":stream_sha,
        "manifest":session.comparator_manifest(),"full":metric_block(p,c,labels,raw),
        "cost":{
            "wall_seconds":float(wall),"cpu_seconds":float(cpu),"max_rss_bytes":max(max_rss_bytes(),rss0),
            "p95_inference_seconds":float(np.percentile(session.predictions["prediction_seconds"],95)),
            "online_update_opportunities_completed":int(session.updates),"online_sample_draws":sampled_training_examples(session),
            "final_resident_parameter_bytes":resident_parameter_bytes(session),"peak_resident_parameter_bytes":int(peak_resident),"peak_live_expert_count":int(peak_active if name=="D_dynamic" else session.comparator_manifest()["active_experts"]),
        },
    }
    if name=="D_dynamic":
        ctrl=session.lifecycle_controller; events=list(ctrl.events); counts=Counter(x["kind"] for x in events); ctrl.mark_stream_end()
        records=list(ctrl.candidate_records.values()); created=len(records); accepted=sum(x.get("accepted") is True for x in records); rejected=sum(x.get("accepted") is False for x in records); pending=sum(x.get("accepted") is None for x in records)
        lifecycle={
            "candidate_created":int(created),"candidate_accepted":int(accepted),"candidate_rejected":int(rejected),"candidate_pending":int(pending),
            "candidate_conservation":bool(created==accepted+rejected+pending),"stream_end_censored_ids":[x["candidate_id"] for x in records if x.get("stream_end_censored")],
            "births":int(counts.get("candidate_accepted",0)),"retirements":int(ctrl.retirements),"reactivations":int(ctrl.reactivations),"purges":int(ctrl.purges),"purged_bytes":int(ctrl.purged_bytes),
            "opportunity_accounting":ctrl.due_conservation(),"event_counts":dict(counts),"accepted_birth_ids":list(ctrl.accepted_ids),"resident_memory_ids":sorted(ctrl.specialist_memory.keys(),key=int),
            "final_topology":session.model.learner.topology_manifest(),"extra_compute":ctrl.extra_compute,
        }
        summary["lifecycle"]=lifecycle; write_jsonl(out/"lifecycle.jsonl",events); write_json(out/"candidate_records.json",records); write_json(out/"opportunity_accounting.json",lifecycle["opportunity_accounting"])
    write_json(out/"summary.json",summary)
    return session,summary


def comparison_from_sessions(sessions,summaries,bundle,registration,stream_sha):
    labels=next(iter(sessions.values())).predictions["labels"]; raw=next(iter(sessions.values())).predictions["raw_labels"]
    for s in sessions.values():
        if not np.array_equal(labels,s.predictions["labels"]): raise AssertionError("comparator targets differ")
    phases={p["name"]:p for p in phase_defs(bundle["manifest"])}; windows=[]
    for name in RECURRENCE:
        p=phases[name]; start=int(p["start"]); end=min(int(p["end"]),start+100); row={"phase":name,"intervals":[start,end],"service":p["regime"],"comparators":{}}
        for comp,s in sessions.items():
            row["comparators"][comp]=binary_detection_metrics(s.predictions["probability"][start:end],labels[start:end])
        d=row["comparators"]["D_dynamic"]["ap"]; c5=row["comparators"]["C_fixed5"]["ap"]; c4=row["comparators"]["C_fixed4"]["ap"]
        row["D_minus_C_fixed5_ap"]=None if d is None or c5 is None else float(d-c5); row["D_minus_C_fixed4_ap"]=None if d is None or c4 is None else float(d-c4); windows.append(row)
    d5=[x["D_minus_C_fixed5_ap"] for x in windows if x["D_minus_C_fixed5_ap"] is not None]; d4=[x["D_minus_C_fixed4_ap"] for x in windows if x["D_minus_C_fixed4_ap"] is not None]
    idx=np.concatenate([np.arange(x["intervals"][0],x["intervals"][1]) for x in windows])
    aggregate={comp:binary_detection_metrics(s.predictions["probability"][idx],labels[idx]) for comp,s in sessions.items()}
    common=np.load(Path(bundle["stream_dir"])/"common_observable_features.npz")["features"][:bundle["steps"]]
    pressure=np.max(common[...,:3],axis=-1); pressure_metrics=binary_detection_metrics(pressure,labels)
    return {
        "protocol":"025","development_only":True,"confirmation_run":False,"stream_sha256":stream_sha,
        "method_registration":registration,"full":{k:v["full"] for k,v in summaries.items()},
        "recurrence_first100":windows,
        "primary":{
            "comparator":"C_fixed5","valid_windows":len(d5),"equal_weight_mean_D_minus_C_fixed5_ap":float(np.mean(d5)) if d5 else None,
            "positive_windows":int(sum(x>0 for x in d5)),"legacy_equal_weight_mean_D_minus_C_fixed4_ap":float(np.mean(d4)) if d4 else None,
            "aggregate_threshold_metrics":aggregate,
        },
        "same_information_pressure_reference":pressure_metrics,
        "cost_profile":{k:v["cost"] for k,v in summaries.items()},
        "hard_delete_evidence_available":bool(summaries["D_dynamic"]["lifecycle"]["purges"]>0),
        "confirmation_seeds_used":[],"test_seeds_used":[],
    }


def main_run(stream_dir,out_root,method_registration_path,data_lock_path):
    stream_dir=Path(stream_dir); out_root=Path(out_root); out_root.mkdir(parents=True,exist_ok=False)
    method=json.loads(Path(method_registration_path).read_text(encoding="utf8")); lock=json.loads(Path(data_lock_path).read_text(encoding="utf8"))
    if not method.get("registered_before_any_model_result"): raise AssertionError("method was not preregistered")
    if method["confirmation_seeds_forbidden"]!=[701,702,703] or method["test_seeds_forbidden"]!=[201,202,203,204,205]: raise AssertionError("seed seal changed")
    audit=json.loads((stream_dir/"data_audit.json").read_text()); manifest=json.loads((stream_dir/"manifest.json").read_text())
    if not audit.get("audit_pass"): raise RuntimeError("model-free data audit did not pass")
    if manifest["stream_sha256"]!=lock["stream_sha256"]: raise AssertionError("data lock SHA mismatch")
    bundle=s4.build_replay(stream_dir); bundle["stream_dir"]=str(stream_dir); guard=build_f0_guard(bundle)
    write_json(out_root/"guard_manifest.json",guard["meta"])
    sessions={}; summaries={}
    for name in COMPARATORS:
        session,summary=run_one(name,bundle,out_root,guard,method,stream_dir,manifest["stream_sha256"]); sessions[name]=session; summaries[name]=summary
    cmp=comparison_from_sessions(sessions,summaries,bundle,method,manifest["stream_sha256"]); write_json(out_root/"comparison.json",cmp); write_json(out_root/"cost_profile.json",cmp["cost_profile"])
    status={"protocol":"025","completed":True,"development_only":True,"confirmation_run":False,"stream_sha256":manifest["stream_sha256"],"primary_gain_D_minus_C_fixed5_ap":cmp["primary"]["equal_weight_mean_D_minus_C_fixed5_ap"],"legacy_gain_D_minus_C_fixed4_ap":cmp["primary"]["legacy_equal_weight_mean_D_minus_C_fixed4_ap"],"purge_gt_0":cmp["hard_delete_evidence_available"],"confirmation_seeds_used":[],"test_seeds_used":[]}
    write_json(out_root/"status.json",status); print(json.dumps({"status":status,"primary":cmp["primary"],"D_lifecycle":summaries["D_dynamic"]["lifecycle"]},indent=2,allow_nan=False)); return status


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stream",required=True); p.add_argument("--out-root",required=True); p.add_argument("--method-registration",required=True); p.add_argument("--data-lock",required=True); a=p.parse_args(); main_run(a.stream,a.out_root,a.method_registration,a.data_lock)

if __name__=="__main__": main()
