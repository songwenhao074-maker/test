"""Protocol-027: exactly C_fixed5 vs D_dynamic on seed700/model1."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, traceback
try:
    import resource
except ImportError:  # Windows reference environment
    resource = None
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_eval import binary_detection_metrics, positive_resource_macro_f1
from ftmoe_protocol025_session import Protocol025FixedSession
from ftmoe_protocol027_normal_guard import Protocol027DynamicSession

COMPARATORS=("C_fixed5","D_dynamic")
RECURRENCE=("S1_rec1","S3_rec1","S2_rec1","S4_rec1","S2_rec2","S6_rec1","S1_rec2","S5_rec1","S3_rec2")
from ftmoe_protocol027_data import EXPECTED_STREAM_SHA, REVISION_ID, verify_frozen

def write_json(path,v):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf8")
def write_jsonl(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf8") as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+"\n")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def phases(manifest):
    return [{"name":p["name"],"start":int(p["start"]),"end":int(p["end"]),"regime":p.get("service")} for p in manifest["timeline"]]
def budget():
    x=dict(json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))["frozen_configuration"])
    x.update({"update_every_scored_intervals":4,"batch_size":32,"gradient_steps_per_opportunity":1,"replay_buffer_intervals":64,"learning_rate":1e-4}); return x

def build_guard(bundle):
    f0={p["name"]:p for p in phases(bundle["manifest"])}["F0"]
    idx=np.arange(int(f0["start"]),int(f0["end"])-1,dtype=np.int64); idx=idx[(idx-int(f0["start"]))%5==0]
    parts={k:[] for k in ("x","schedule","graph_x","labels","ids","before","caps")}; raw=np.asarray(bundle["arrays"]["raw_labels"],np.int64)
    for left in range(0,len(idx),32):
        batch=idx[left:left+32].tolist(); x,sc,g,c=s4.window_batch(bundle["replay"],batch)
        for k,v in (("x",x),("schedule",sc),("graph_x",g),("ids",c["creation_ids"]),("before",c["before_placement"]),("caps",c["capacities"])):parts[k].append(v)
        parts["labels"].append(torch.as_tensor(raw[np.asarray(batch)+1],dtype=torch.long))
    out={k:torch.cat(v,0) for k,v in parts.items()}; y=out["labels"].numpy().reshape(-1)
    n=int((y==0).sum()); pos=int((y>0).sum())
    if n==0: raise RuntimeError("Protocol027 normal guard has no normal rows")
    if not all(torch.isfinite(out[k]).all() for k in ("x","schedule","graph_x","caps")):raise RuntimeError("nonfinite guard input")
    out["observed_history_length"] = torch.as_tensor(np.minimum(idx + 1, 12), dtype=torch.long)
    out["meta"]={"protocol":"027","role":"historical_normal_regression_guard","source_phase":"F0","prediction_indices":idx.tolist(),"target":"same-host raw[t+1] inside F0","normal_rows":n,"positive_rows":pos,"positive_rows_required":False,"normal_nll_limit":"candidate <= live*1.02 + 1e-6","fpr_threshold":0.5,"candidate_minus_live_fpr_max":0.01,"candidate_training_on_guard_rows":False}
    return out

def tensor_bytes(v):
    if torch.is_tensor(v):return int(v.numel()*v.element_size())
    if isinstance(v,dict):return sum(tensor_bytes(x) for x in v.values())
    if isinstance(v,(list,tuple)):return sum(tensor_bytes(x) for x in v)
    return 0
def param_bytes(s):return int(sum(p.numel()*p.element_size() for p in s.model.learner.parameters()))
def opt_bytes(s):
    n=tensor_bytes(s.optimizer.state_dict()) if s.optimizer is not None else 0
    n+=tensor_bytes(getattr(s,"optimizer_archive",{})); sh=getattr(s,"shadow_optimizer",None)
    if sh is not None:n+=tensor_bytes(sh.state_dict())
    return int(n+tensor_bytes(getattr(s,"shadow_optimizer_state",None)))
def rss_bytes():
    if resource is not None:
        scale = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)
    import psutil
    return int(psutil.Process().memory_info().peak_wset)
def samples(s):return int(sum(int(r.get("batch_size",0))*int(r.get("gradient_steps",1)) for r in s.update_log))
def metrics(prob,cls,y):return {"detection":binary_detection_metrics(prob,y,threshold=.5),"resource":positive_resource_macro_f1(cls,y)}
def phase_metrics(prob,cls,y,pdefs):
    out=[]
    for p in pdefs:
        a,b=p["start"],p["end"]; yy=y[a:b]
        out.append({"phase":p["name"],"service":p["regime"],"intervals":[a,b],"positive_rows":int((yy>0).sum()),"negative_rows":int((yy==0).sum()),**metrics(prob[a:b],cls[a:b],yy)})
    return out
def event_phase(cursor,pdefs):
    for p in pdefs:
        if p["start"]<=int(cursor)<p["end"]:return p["name"]
    return "stream_end"

def run_arm(name,stream,arm_dir,method_registration,registration,run_id):
    if name not in COMPARATORS:raise ValueError(name)
    out=Path(arm_dir); out.mkdir(parents=True,exist_ok=False); stream=Path(stream)
    verify_frozen(stream)
    method=json.loads(Path(method_registration).read_text()); reg=json.loads(Path(registration).read_text())
    if reg.get("protocol")!="027" or reg["seeds"].get("replay")!=700 or reg["seeds"].get("model")!=1:raise AssertionError("Protocol027 registration/seed mismatch")
    bundle=s4.build_replay(stream); bundle["stream_dir"]=str(stream); stream_sha=sha(stream/"stream.npz")
    if stream_sha!=EXPECTED_STREAM_SHA or bundle["manifest"].get("stream_sha256")!=EXPECTED_STREAM_SHA:raise AssertionError("stream hash mismatch")
    pdefs=phases(bundle["manifest"]); guard=build_guard(bundle); write_json(out/"guard_manifest.json",guard["meta"])
    rr={"protocol":"027","comparator":name,"source_method_registration":method,"protocol027_registration":reg,"stream_sha256":stream_sha,"data_revision":REVISION_ID,"replay_seed":700,"model_seed":1,"confirmation_run":False,"test_run":False}
    kw=dict(seed=1,replay_bundle=bundle,budget=budget(),out_dir=out,run_id=f"protocol027_{run_id}_{name}",stream_dir=stream,phase_defs=pdefs,stream_sha=stream_sha,registration=rr,learning_rate=1e-4)
    s=Protocol027DynamicSession(guard_anchor=guard,**kw) if name=="D_dynamic" else Protocol025FixedSession("C_fixed5",**kw)
    w0=time.perf_counter(); c0=time.process_time(); peak_p=param_bytes(s); peak_o=opt_bytes(s); peak_e=5
    for i in range(s.steps):
        s.step()
        if i%64==0 or i+1==s.steps:peak_p=max(peak_p,param_bytes(s)); peak_o=max(peak_o,opt_bytes(s))
        if name=="D_dynamic":peak_e=max(peak_e,len(s.model.learner.ids))
    if name=="D_dynamic":s.lifecycle_controller.mark_stream_end()
    s.finish(); wall=time.perf_counter()-w0; cpu=time.process_time()-c0; s.save()
    prob=s.predictions["probability"]; cls=s.predictions["class_probability"]; y=s.predictions["labels"]
    if (y<0).any() or (s.predictions["raw_labels"]<0).any():raise AssertionError("unsettled outputs")
    cost={"wall_seconds":float(wall),"cpu_seconds":float(cpu),"max_rss_bytes":rss_bytes(),"p95_inference_seconds":float(np.percentile(s.predictions["prediction_seconds"],95)),"online_update_opportunities_completed":int(s.updates),"online_sample_draws":samples(s),"final_resident_parameter_bytes":param_bytes(s),"peak_resident_parameter_bytes":int(peak_p),"final_optimizer_state_bytes":opt_bytes(s),"peak_optimizer_state_bytes":int(peak_o),"peak_live_expert_count":int(peak_e),"diagnostic_and_file_io_included_in_wall_and_cpu":False}
    summary={"protocol":"027","comparator":name,"run_id":str(run_id),"completed":True,"development_only":True,"confirmation_run":False,"test_run":False,"stream_sha256":stream_sha,"data_revision":REVISION_ID,"manifest":s.comparator_manifest(),"full":metrics(prob,cls,y),"phases":phase_metrics(prob,cls,y,pdefs),"cost":cost}
    if name=="D_dynamic":
        ctrl=s.lifecycle_controller; events=list(ctrl.events); records=list(ctrl.candidate_records.values()); ec=Counter(x["kind"] for x in events); rejects=Counter(); byphase=defaultdict(Counter)
        for r in records:
            for reason in (r.get("decision") or {}).get("reject_reasons",[]):rejects[str(reason)]+=1
        for e in events:byphase[event_phase(e.get("cursor",0),pdefs)][e["kind"]]+=1
        acc=sum(r.get("accepted") is True for r in records); rej=sum(r.get("accepted") is False for r in records); pending=sum(r.get("accepted") is None for r in records)
        life={"candidate_created":len(records),"candidate_accepted":acc,"candidate_rejected":rej,"candidate_pending":pending,"candidate_conservation":len(records)==acc+rej+pending,"stream_end_censored_ids":[r["candidate_id"] for r in records if r.get("stream_end_censored")],"births":int(ec.get("candidate_accepted",0)),"retirements":int(ctrl.retirements),"reactivations":int(ctrl.reactivations),"purges":int(ctrl.purges),"purged_bytes":int(ctrl.purged_bytes),"event_counts":dict(ec),"event_counts_by_phase":{k:dict(v) for k,v in byphase.items()},"candidate_reject_reasons":dict(rejects),"accepted_birth_ids":list(ctrl.accepted_ids),"resident_memory_ids":sorted(ctrl.specialist_memory.keys(),key=int),"opportunity_accounting":ctrl.due_conservation(),"final_topology":s.model.learner.topology_manifest(),"extra_compute":dict(ctrl.extra_compute),"reuse_observed":bool(ctrl.reactivations>0),"reuse_benefit_claim_allowed":False}
        summary["lifecycle"]=life; cost.update({"shadow_training_steps":int(ctrl.extra_compute.get("shadow_train_steps",0)),"reuse_validation_forwards":int(ctrl.extra_compute.get("reuse_validation_forwards",0)),"guard_forwards":int(ctrl.extra_compute.get("guard_forwards",0)),"reuse_guard_forwards":int(ctrl.extra_compute.get("reuse_guard_forwards",0))})
        write_jsonl(out/"lifecycle.jsonl",events); write_json(out/"candidate_records.json",records); write_json(out/"opportunity_accounting.json",life["opportunity_accounting"]); write_json(out/"lifecycle_summary.json",life)
    write_json(out/"summary.json",summary); return summary

def load_npz(p):
    with np.load(p) as d:return {k:d[k].copy() for k in d.files}
def compare(root,stream,run_id):
    root=Path(root); sums={n:json.loads((root/n/"summary.json").read_text()) for n in COMPARATORS}; pred={n:load_npz(root/n/"predictions.npz") for n in COMPARATORS}; c,d=pred["C_fixed5"],pred["D_dynamic"]
    if not np.array_equal(c["labels"],d["labels"]):raise AssertionError("labels differ")
    y=c["labels"]; pp={p["name"]:p for p in phases(json.loads((Path(stream)/"manifest.json").read_text()))}; rows=[]; deltas=[]; pos=0; pooled=[]; valid_all=True
    for name in RECURRENCE:
        p=pp[name]; a=p["start"]; b=min(p["end"],a+100); pooled.extend(range(a,b)); yy=y[a:b]
        cm=binary_detection_metrics(c["probability"][a:b],yy,.5); dm=binary_detection_metrics(d["probability"][a:b],yy,.5); valid=cm["ap"] is not None and dm["ap"] is not None; delta=float(dm["ap"]-cm["ap"]) if valid else None
        valid_all&=valid
        if valid:deltas.append(delta); pos+=int(delta>0)
        rows.append({"phase":name,"service":p["regime"],"intervals":[a,b],"positive_rows":int((yy>0).sum()),"negative_rows":int((yy==0).sum()),"C_fixed5":cm,"D_dynamic":dm,"D_minus_C_fixed5_ap":delta,"valid":bool(valid),"invalid_reason":None if valid else "AP undefined for at least one comparator"})
    idx=np.asarray(pooled,np.int64); cm=binary_detection_metrics(c["probability"][idx],y[idx],.5); dm=binary_detection_metrics(d["probability"][idx],y[idx],.5); fd=None if cm["fpr"] is None or dm["fpr"] is None else float(dm["fpr"]-cm["fpr"]); mean=float(np.mean(deltas)) if valid_all and len(deltas)==9 else None
    signal=bool(mean is not None and mean>=.03 and pos>=6 and fd is not None and fd<=.01); ca=sums["C_fixed5"]["full"]["detection"]["ap"]; da=sums["D_dynamic"]["full"]["detection"]["ap"]; full_delta=None if ca is None or da is None else float(da-ca)
    return {"protocol":"027","run_id":str(run_id),"development_only":True,"confirmation_run":False,"test_run":False,"stream_sha256":EXPECTED_STREAM_SHA,"data_revision":REVISION_ID,"recurrence_first100":rows,"primary":{"metric":"equal-weight mean D_dynamic-C_fixed5 AP across all nine recurrence first100 windows","valid_windows":sum(r["valid"] for r in rows),"required_valid_windows":9,"equal_weight_mean_D_minus_C_fixed5_ap":mean,"positive_windows":pos,"pooled_normal_fpr_C_fixed5":cm["fpr"],"pooled_normal_fpr_D_dynamic":dm["fpr"],"pooled_normal_fpr_delta_D_minus_C":fd,"numeric_lead_on_this_trajectory":None if mean is None else bool(mean>0),"development_signal":signal,"development_reference":{"mean_delta_min":.03,"positive_windows_min":6,"pooled_normal_fpr_delta_max":.01}},"full_stream":{"C_fixed5":sums["C_fixed5"]["full"],"D_dynamic":sums["D_dynamic"]["full"],"D_minus_C_fixed5_ap":full_delta,"D_ap_leads":None if full_delta is None else bool(full_delta>0)},"phase_metrics":{n:sums[n]["phases"] for n in COMPARATORS},"lifecycle":sums["D_dynamic"].get("lifecycle"),"cost_profile":{n:sums[n]["cost"] for n in COMPARATORS},"interpretation_limits":{"statistical_confirmation":False,"equal_total_cost":False,"D_extra_background_compute_allowed":True,"single_replay_seed":700,"single_model_seed":1}}

def parent(args):
    root=Path(args.out_root); root.mkdir(parents=True,exist_ok=True)
    if (root/"status.json").exists() or any((root/n).exists() for n in COMPARATORS):raise FileExistsError("existing Protocol027 model run")
    frozen = verify_frozen(args.stream)
    receipt=json.loads((root/"frozen_data_archive.json").read_text())
    if not receipt.get("complete_snapshot_uploaded") or not receipt.get("artifact_id") or not receipt.get("artifact_digest"):
        raise RuntimeError("complete data archive is required before training")
    if receipt.get("frozen_manifest_sha256") != sha(Path(args.stream)/"frozen_data_manifest.json"):
        raise RuntimeError("archive receipt belongs to different frozen dataset")
    lock=json.loads(Path(args.data_lock).read_text()); elig=json.loads(Path(args.eligibility).read_text())
    if any(item.get("data_revision") != REVISION_ID or item.get("stream_sha256") != EXPECTED_STREAM_SHA for item in (lock, elig)):
        raise RuntimeError("data lock/eligibility belong to different revision")
    if not lock.get("locked") or not elig.get("protocol027_data_eligible"):raise RuntimeError("invalid data lock/eligibility")
    bundle=s4.build_replay(Path(args.stream)); guard=build_guard(bundle); write_json(root/"guard_manifest.json",guard["meta"]); write_json(root/"normal_guard_audit.json",{"protocol":"027","valid":guard["meta"]["normal_rows"]>0,**guard["meta"],"birth_and_reuse_guard_implementation":"shared normal_detection_guard_report","positive_rows_zero_policy":"allowed"}); del bundle,guard
    completed=[]; failed=[]; detail={}
    for n in COMPARATORS:
        cmd=[sys.executable,str(Path(__file__).resolve()),"--child-arm",n,"--stream",args.stream,"--arm-dir",str(root/n),"--method-registration",args.method_registration,"--registration",args.registration,"--run-id",str(args.run_id)]
        so=root/f"{n}.stdout.log"; se=root/f"{n}.stderr.log"
        with so.open("w") as o,se.open("w") as e:p=subprocess.run(cmd,text=True,stdout=o,stderr=e)
        if p.returncode==0 and (root/n/"summary.json").is_file() and (root/n/"predictions.npz").is_file():completed.append(n)
        else:
            failed.append(n); detail[n]={"returncode":p.returncode,"stderr_tail":se.read_text(errors="replace")[-8000:]}; write_json(root/f"{n}.failure.json",detail[n])
    comp=compare(root,args.stream,args.run_id) if completed==list(COMPARATORS) and not failed else None
    if comp is not None:write_json(root/"comparison.json",comp); write_json(root/"cost_profile.json",comp["cost_profile"])
    status={"protocol":"027","data_revision":REVISION_ID,"frozen_data_archived":True,"frozen_data_archive":receipt,"run_id":str(args.run_id),"completed":comp is not None,"data_restored":True,"eligibility_verified":True,"normal_guard_valid":True,"completed_comparators":completed,"failed_comparators":failed,"failure_details":detail,"development_signal":None if comp is None else bool(comp["primary"]["development_signal"]),"confirmation_run":False,"test_run":False,"confirmation_seeds_used":[],"test_seeds_used":[],"automatic_followups_started":[],"blocker":None if comp is not None else "one_or_more_registered_arms_failed"}
    if comp is not None:status.update({"primary_mean_D_minus_C_fixed5_ap":comp["primary"]["equal_weight_mean_D_minus_C_fixed5_ap"],"positive_recurrence_windows":comp["primary"]["positive_windows"],"full_stream_D_minus_C_fixed5_ap":comp["full_stream"]["D_minus_C_fixed5_ap"],"D_reactivations":int((comp.get("lifecycle") or {}).get("reactivations",0)),"D_purges":int((comp.get("lifecycle") or {}).get("purges",0))})
    write_json(root/"status.json",status); print(json.dumps({"status":status,"primary":None if comp is None else comp["primary"]},indent=2,allow_nan=False))

def child(args):
    try:run_arm(args.child_arm,args.stream,args.arm_dir,args.method_registration,args.registration,args.run_id)
    except Exception:
        Path(args.arm_dir).mkdir(parents=True,exist_ok=True); write_json(Path(args.arm_dir)/"failure.json",{"protocol":"027","comparator":args.child_arm,"completed":False,"traceback":traceback.format_exc()}); raise

def main():
    p=argparse.ArgumentParser(); p.add_argument("--stream",required=True); p.add_argument("--out-root"); p.add_argument("--arm-dir"); p.add_argument("--method-registration",required=True); p.add_argument("--registration",required=True); p.add_argument("--data-lock"); p.add_argument("--eligibility"); p.add_argument("--run-id",required=True); p.add_argument("--child-arm",choices=COMPARATORS); a=p.parse_args()
    if a.child_arm:
        if not a.arm_dir:p.error("--arm-dir required")
        child(a)
    else:
        if not all((a.out_root,a.data_lock,a.eligibility)):p.error("--out-root/--data-lock/--eligibility required")
        parent(a)
if __name__=="__main__":main()
