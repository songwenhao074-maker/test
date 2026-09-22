"""Protocol-031 revision002: exactly C_fixed5 vs D_nonblocking_reuse."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, subprocess, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
try:
    import resource
except ImportError:
    resource=None

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_eval import binary_detection_metrics, positive_resource_macro_f1
from ftmoe_protocol025_session import Protocol025FixedSession
from ftmoe_protocol031_nonblocking_reuse import Protocol031DynamicSession, P031_CONFIG
import prepare_ftmoe_protocol031_stream as p31data

COMPARATORS=("C_fixed5","D_nonblocking_reuse")
RECURRENCE=("U_rec1","V_rec1","U_rec2","V_rec2","U_rec3","V_rec3")
W_BLOCKS=("W_long","W_gap1","W_gap2","W_gap3","W_gap4","W_gap5")
SNAPSHOT_POINTS={
    1900:"after_U_first",
    3500:"after_V_first",
    6700:"before_U_rec1",
    8428:"before_V_rec1",
    10156:"before_U_rec2",
    11884:"before_V_rec2",
    13612:"before_U_rec3",
    15340:"before_V_rec3",
}

def deterministic_runtime():
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[key]="1"
    os.environ["PYTHONHASHSEED"]="1"
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.use_deterministic_algorithms(True,warn_only=False)
    return {
        "OMP_NUM_THREADS":os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS":os.environ["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS":os.environ["OPENBLAS_NUM_THREADS"],
        "PYTHONHASHSEED":os.environ["PYTHONHASHSEED"],
        "torch_num_threads":int(torch.get_num_threads()),
        "torch_num_interop_threads":int(torch.get_num_interop_threads()),
        "deterministic_algorithms":bool(torch.are_deterministic_algorithms_enabled()),
    }

def json_ready(v):
    if isinstance(v,np.generic): return json_ready(v.item())
    if isinstance(v,np.ndarray): return json_ready(v.tolist())
    if torch.is_tensor(v): return json_ready(v.detach().cpu().numpy())
    if isinstance(v,dict): return {str(k):json_ready(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,set)): return [json_ready(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v): return None
    if isinstance(v,(str,int,float,bool)) or v is None: return v
    raise TypeError("Protocol031 JSON unsupported type: %s"%type(v).__name__)

def write_json(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(json_ready(value),indent=2,allow_nan=False)+"\n",encoding="utf8")

def write_jsonl(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(json_ready(row),allow_nan=False)+"\n")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def phases(manifest):
    rows=[]
    for p in manifest["timeline"]:
        rows.append({
            "name":str(p["name"]),"start":int(p["start"]),"end":int(p["end"]),
            "regime":p.get("logical_service"),
            "source_service":p.get("service"),
        })
    return rows

def phase_name(index,pdefs):
    i=int(index)
    for p in pdefs:
        if p["start"]<=i<p["end"]: return p["name"]
    return "stream_end"

def budget():
    b=dict(json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))["frozen_configuration"])
    b.update({
        "update_every_scored_intervals":16,
        "batch_size":32,
        "gradient_steps_per_opportunity":1,
        "replay_buffer_intervals":64,
        "learning_rate":1e-4,
    })
    return b

def build_guard(bundle):
    f0={p["name"]:p for p in phases(bundle["manifest"])}["F0"]
    idx=np.arange(int(f0["start"]),int(f0["end"])-1,dtype=np.int64)
    idx=idx[(idx-int(f0["start"]))%5==0]
    parts={k:[] for k in ("x","schedule","graph_x","labels","ids","before","caps")}
    raw=np.asarray(bundle["arrays"]["raw_labels"],np.int64)
    for left in range(0,len(idx),32):
        batch=idx[left:left+32].tolist()
        x,sc,g,c=s4.window_batch(bundle["replay"],batch)
        for k,v in (("x",x),("schedule",sc),("graph_x",g),
                    ("ids",c["creation_ids"]),("before",c["before_placement"]),
                    ("caps",c["capacities"])):
            parts[k].append(v)
        parts["labels"].append(torch.as_tensor(raw[np.asarray(batch)+1],dtype=torch.long))
    out={k:torch.cat(v,0) for k,v in parts.items()}
    y=out["labels"].numpy().reshape(-1)
    normal=int((y==0).sum()); positive=int((y>0).sum())
    if normal<=0: raise RuntimeError("Protocol031 F0 normal guard has no normal rows")
    out["observed_history_length"]=torch.as_tensor(np.minimum(idx+1,12),dtype=torch.long)
    out["meta"]={
        "protocol":"031","plan_revision":2,
        "role":"F0_known_normal_regression_guard","source_phase":"F0",
        "prediction_indices":idx.tolist(),"target":"same-host raw[t+1] inside F0",
        "normal_rows":normal,"positive_rows":positive,"positive_rows_required":False,
        "normal_nll_rule":"candidate <= live*1.02 + 1e-6",
        "fpr_threshold":0.5,"candidate_minus_live_fpr_max":0.01,
        "candidate_training_on_guard_rows":False,
    }
    return out

def tensor_bytes(v):
    if torch.is_tensor(v): return int(v.numel()*v.element_size())
    if isinstance(v,np.ndarray): return int(v.nbytes)
    if isinstance(v,dict): return sum(tensor_bytes(x) for x in v.values())
    if isinstance(v,(list,tuple)): return sum(tensor_bytes(x) for x in v)
    return 0

def param_bytes(s):
    return int(sum(p.numel()*p.element_size() for p in s.model.learner.parameters()))

def opt_bytes(s):
    n=tensor_bytes(s.optimizer.state_dict()) if s.optimizer is not None else 0
    n+=tensor_bytes(getattr(s,"optimizer_archive",{}))
    sh=getattr(s,"shadow_optimizer",None)
    if sh is not None: n+=tensor_bytes(sh.state_dict())
    n+=tensor_bytes(getattr(s,"shadow_optimizer_state",None))
    return int(n)

def rss_bytes():
    if resource is not None:
        scale=1 if sys.platform=="darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale)
    import psutil
    return int(psutil.Process().memory_info().rss)

def samples(s):
    return int(sum(int(r.get("batch_size",0))*int(r.get("gradient_steps",1))
                   for r in s.update_log))

def metrics(prob,cls,y):
    return {
        "detection":binary_detection_metrics(prob,y,.5),
        "resource":positive_resource_macro_f1(cls,y),
    }

def hash_tensor(h,name,t):
    x=t.detach().cpu().contiguous()
    h.update(str(name).encode()); h.update(str(tuple(x.shape)).encode())
    h.update(str(x.dtype).encode()); h.update(x.numpy().tobytes())

def shared_prefix_hash(model,count=4):
    bank=model.learner; h=hashlib.sha256()
    dynamic=hasattr(bank,"ids") and hasattr(bank,"router_weights")
    for i in range(int(count)):
        key=str(i)
        expert=bank.experts[key] if dynamic else bank.experts[i]
        for name,t in sorted(expert.state_dict().items()):
            hash_tensor(h,f"{key}|expert|{name}",t)
        if dynamic:
            hash_tensor(h,f"{key}|router_weight",bank.router_weights[key])
            hash_tensor(h,f"{key}|router_bias",bank.router_biases[key])
        else:
            hash_tensor(h,f"{key}|router_weight",bank.router.weight[i])
            hash_tensor(h,f"{key}|router_bias",bank.router.bias[i])
    return h.hexdigest()

def capture_snapshot(session,cursor,label,pdefs):
    bank=session.model.learner
    ctrl=getattr(session,"lifecycle_controller",None)
    memory=[] if ctrl is None else sorted(ctrl.specialist_memory.keys(),key=int)
    return {
        "label":label,"cursor":int(cursor),
        "phase":phase_name(max(0,int(cursor)-1),pdefs),
        "topology":bank.topology_manifest() if hasattr(bank,"topology_manifest") else None,
        "active_specialist_id":None if ctrl is None else ctrl.active_specialist_id,
        "dormant_memory_ids":memory,
        "shadow_id":None if getattr(bank,"shadow_id",None) is None else str(bank.shadow_id),
        "pending_reuse":None if ctrl is None else deepcopy_safe(getattr(ctrl,"pending_reuse",None)),
        "accepted_birth_ids":[] if ctrl is None else list(ctrl.accepted_ids),
    }

def deepcopy_safe(value):
    import copy
    return copy.deepcopy(value)

def validate_frozen_data(stream,reg,data_lock):
    root=Path(stream)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf8"))
    frozen=json.loads((root/"frozen_data_manifest.json").read_text(encoding="utf8"))
    lock=json.loads(Path(data_lock).read_text(encoding="utf8"))
    actual=sha(root/"stream.npz")
    expected=reg["generation"].get("expected_stream_sha256")
    if not expected:
        raise RuntimeError("Protocol031 expected_stream_sha256 is not frozen")
    if not (actual==expected==manifest.get("stream_sha256")==
            frozen.get("stream_sha256")==lock.get("stream_sha256")):
        raise RuntimeError("Protocol031 frozen stream identity mismatch")
    if lock.get("locked") is not True or lock.get("model_free_eligibility_passed") is not True:
        raise RuntimeError("Protocol031 data lock not eligible")
    return actual,manifest,lock,frozen

def run_arm(name,stream,arm_dir,scenario_registration,method_registration,
            data_lock,run_id):
    deterministic_runtime()
    if name not in COMPARATORS: raise ValueError(name)
    out=Path(arm_dir); out.mkdir(parents=True,exist_ok=False)
    sreg=json.loads(Path(scenario_registration).read_text(encoding="utf8"))
    mreg=json.loads(Path(method_registration).read_text(encoding="utf8"))
    if sreg.get("protocol")!="031" or int(sreg.get("plan_revision",-1))!=2:
        raise AssertionError("Protocol031 scenario registration mismatch")
    if mreg.get("protocol")!="031" or mreg.get("arms")!=list(COMPARATORS):
        raise AssertionError("Protocol031 method registration mismatch")
    stream_sha,manifest,lock,_=validate_frozen_data(stream,sreg,data_lock)
    bundle=s4.build_replay(Path(stream)); bundle["stream_dir"]=str(stream)
    if bundle["manifest"].get("stream_sha256")!=stream_sha:
        raise AssertionError("Protocol031 replay manifest hash mismatch")
    pdefs=phases(manifest)
    guard=build_guard(bundle)
    write_json(out/"guard_manifest.json",guard["meta"])
    runtime_registration={
        "protocol":"031","plan_revision":2,"comparator":name,
        "scenario_id":p31data.SCENARIO_ID,"data_revision":p31data.DATA_REVISION,
        "scenario_registration_sha256":sha(scenario_registration),
        "method_registration_sha256":sha(method_registration),
        "stream_sha256":stream_sha,"replay_seed":700,"model_seed":1,
        "confirmation_run":False,"test_run":False,
    }
    common=dict(
        seed=1,replay_bundle=bundle,budget=budget(),out_dir=out,
        run_id=f"protocol031_{run_id}_{name}",stream_dir=Path(stream),
        phase_defs=pdefs,stream_sha=stream_sha,registration=runtime_registration,
        learning_rate=1e-4,
    )
    session=(Protocol031DynamicSession(guard_anchor=guard,v2c_config=P031_CONFIG,**common)
             if name=="D_nonblocking_reuse"
             else Protocol025FixedSession("C_fixed5",**common))
    init_prefix=shared_prefix_hash(session.model,4)
    write_json(out/"initialization.json",{
        "shared_first4_expert_and_router_rows_sha256":init_prefix,
        "model_seed":1,"registered_shared_prefix_initialization":True,
    })

    snapshots=[]
    wall0=time.perf_counter(); cpu0=time.process_time()
    peak_p=param_bytes(session); peak_o=opt_bytes(session)
    peak_live=5 if name=="C_fixed5" else 4
    peak_resident=peak_live
    for i in range(session.steps):
        if name=="D_nonblocking_reuse" and i in SNAPSHOT_POINTS:
            snapshots.append(capture_snapshot(session,i,SNAPSHOT_POINTS[i],pdefs))
        session.step()
        if i%64==0 or i+1==session.steps:
            peak_p=max(peak_p,param_bytes(session)); peak_o=max(peak_o,opt_bytes(session))
        if name=="D_nonblocking_reuse":
            bank=session.model.learner
            peak_live=max(peak_live,len(bank.ids))
            peak_resident=max(peak_resident,bank.resident_count())
            if bank.resident_count()>8:
                raise AssertionError("Protocol031 resident capacity exceeded")
    if name=="D_nonblocking_reuse":
        session.lifecycle_controller.mark_stream_end()
    session.finish()
    wall=time.perf_counter()-wall0; cpu=time.process_time()-cpu0
    session.save()

    prob=session.predictions["probability"]
    cls=session.predictions["class_probability"]
    y=session.predictions["labels"]
    if (y<0).any() or (session.predictions["raw_labels"]<0).any():
        raise AssertionError("Protocol031 unsettled model outputs")
    cost={
        "wall_seconds":float(wall),"cpu_seconds":float(cpu),
        "max_rss_bytes":rss_bytes(),
        "p95_inference_seconds":float(np.percentile(session.predictions["prediction_seconds"],95)),
        "online_update_opportunities_completed":int(session.updates),
        "online_sample_draws":samples(session),
        "final_resident_parameter_bytes":param_bytes(session),
        "peak_resident_parameter_bytes":int(peak_p),
        "final_optimizer_state_bytes":opt_bytes(session),
        "peak_optimizer_state_bytes":int(peak_o),
        "peak_live_expert_count":int(peak_live),
        "peak_resident_expert_count":int(peak_resident),
        "runtime_profile":deterministic_runtime(),
        "timing_scope":{
            "included":"full scored step loop, online updates, lifecycle validation/guards, finish",
            "excluded":"session.save, parent comparison, artifact upload, git",
        },
    }
    summary={
        "protocol":"031","plan_revision":2,"scenario_id":p31data.SCENARIO_ID,
        "data_revision":p31data.DATA_REVISION,"comparator":name,
        "run_id":str(run_id),"completed":True,
        "development_only":True,"confirmation_run":False,"test_run":False,
        "stream_sha256":stream_sha,
        "manifest":session.comparator_manifest(),
        "initialization_shared_prefix_sha256":init_prefix,
        "full":metrics(prob,cls,y),
        "phases":[],
        "cost":cost,
    }
    for p in pdefs:
        a,b=p["start"],p["end"]; yy=y[a:b]
        summary["phases"].append({
            "phase":p["name"],"service":p["regime"],"intervals":[a,b],
            "positive_rows":int((yy>0).sum()),"negative_rows":int((yy==0).sum()),
            **metrics(prob[a:b],cls[a:b],yy),
        })

    if name=="D_nonblocking_reuse":
        ctrl=session.lifecycle_controller
        events=list(ctrl.events); records=list(ctrl.candidate_records.values())
        ec=Counter(e["kind"] for e in events)
        byphase=defaultdict(Counter)
        for e in events: byphase[phase_name(e.get("cursor",0),pdefs)][e["kind"]]+=1
        accepted=sum(r.get("accepted") is True and not r.get("cancelled_by_reuse") for r in records)
        cancelled=sum(bool(r.get("cancelled_by_reuse")) for r in records)
        rejected=sum(r.get("accepted") is False and not r.get("cancelled_by_reuse") for r in records)
        pending=sum(r.get("accepted") is None for r in records)
        due=ctrl.due_conservation()
        if not due["birth"]["conserved"]:
            raise AssertionError("Protocol031 birth due accounting not conserved")
        if not due["reuse_nonblocking"]["due_conserved"] or not due["reuse_nonblocking"]["started_conserved"]:
            raise AssertionError("Protocol031 reuse accounting not conserved")
        if ctrl.purges!=0:
            raise AssertionError("Protocol031 memory protection unexpectedly purged specialist")
        candidate_sources=[]
        for rec in records:
            idx=list(rec.get("training_indices") or [])
            phases_used=sorted({phase_name(i,pdefs) for i in idx})
            candidate_sources.append({
                "candidate_id":rec.get("candidate_id"),
                "created_cursor":rec.get("created_cursor"),
                "accepted":rec.get("accepted"),
                "cancelled_by_reuse":bool(rec.get("cancelled_by_reuse")),
                "training_indices_count":len(idx),
                "training_source_phases":phases_used,
                "accepted_cursor":rec.get("accepted_cursor"),
                "first_influence_cursor":rec.get("first_influence_cursor"),
                "cancelled_budget":rec.get("cancelled_budget"),
            })
        extra=dict(ctrl.extra_compute)
        cost.update({
            "shadow_train_steps":int(extra.get("shadow_train_steps",0)),
            "shadow_train_examples":int(extra.get("shadow_train_examples",0)),
            "reuse_validation_forwards":int(extra.get("p031_reuse_validation_forwards",0)),
            "reuse_guard_forwards":int(extra.get("p031_reuse_guard_forwards",0)),
            "shadow_train_seconds":float(extra.get("shadow_train_seconds",0.0)),
            "shadow_validation_seconds":float(extra.get("shadow_validation_seconds",0.0)),
            "birth_guard_seconds":float(extra.get("guard_seconds",0.0)),
            "reuse_validation_seconds":float(extra.get("p031_reuse_validation_seconds",0.0)),
            "reuse_guard_seconds":float(extra.get("p031_reuse_guard_seconds",0.0)),
        })
        lifecycle={
            "birth_candidates_created":len(records),
            "birth_candidates_accepted":accepted,
            "birth_candidates_rejected":rejected,
            "birth_candidates_cancelled":cancelled,
            "birth_candidates_pending":pending,
            "birth_conservation":len(records)==accepted+rejected+cancelled+pending,
            "birth_cancelled_by_reuse":int(ctrl.birth_cancelled_by_reuse),
            "birth_cancelled_ids":list(ctrl.birth_cancelled_ids),
            "retirements":int(ctrl.retirements),"reactivations":int(ctrl.reactivations),
            "purges":int(ctrl.purges),"capacity_preserve_skips":int(ctrl.capacity_preserve_skips),
            "accepted_birth_ids":list(ctrl.accepted_ids),
            "resident_memory_ids":sorted(ctrl.specialist_memory.keys(),key=int),
            "active_specialist_id":ctrl.active_specialist_id,
            "reuse_first_influence_cursor":ctrl.first_reuse_influence_cursor,
            "reuse_records":ctrl.reuse_records,
            "reuse_outcomes":ctrl.reuse_outcomes,
            "due_accounting":due,
            "event_counts":dict(ec),
            "event_counts_by_phase":{k:dict(v) for k,v in byphase.items()},
            "candidate_sources":candidate_sources,
            "expert_snapshots":snapshots,
            "final_topology":session.model.learner.topology_manifest(),
            "extra_compute":extra,
        }
        summary["lifecycle"]=lifecycle
        write_jsonl(out/"lifecycle.jsonl",events)
        write_json(out/"candidate_records.json",records)
        write_json(out/"reuse_records.json",ctrl.reuse_records)
        write_json(out/"opportunity_accounting.json",due)
        write_json(out/"expert_snapshots.json",snapshots)
        write_json(out/"lifecycle_summary.json",lifecycle)
    write_json(out/"summary.json",summary)
    return summary

def load_npz(path):
    with np.load(path,allow_pickle=True) as d:
        return {k:d[k].copy() for k in d.files}

def metric_window(prob,y,a,b):
    return binary_detection_metrics(prob[int(a):int(b)],y[int(a):int(b)],.5)

def compare(root,stream,run_id):
    root=Path(root)
    sums={n:json.loads((root/n/"summary.json").read_text(encoding="utf8")) for n in COMPARATORS}
    pred={n:load_npz(root/n/"predictions.npz") for n in COMPARATORS}
    c,d=pred["C_fixed5"],pred["D_nonblocking_reuse"]
    if not np.array_equal(c["labels"],d["labels"]):
        raise AssertionError("Protocol031 comparator labels differ")
    if sums["C_fixed5"]["initialization_shared_prefix_sha256"]!=sums["D_nonblocking_reuse"]["initialization_shared_prefix_sha256"]:
        raise AssertionError("Protocol031 shared first4 initialization differs")
    y=c["labels"]
    manifest=json.loads((Path(stream)/"manifest.json").read_text(encoding="utf8"))
    pmap={p["name"]:p for p in phases(manifest)}
    rows=[]; deltas=[]; positive=0; pooled=[]
    for name in RECURRENCE:
        p=pmap[name]; a,b=p["start"],p["end"]; pooled.extend(range(a,b))
        cm=metric_window(c["probability"],y,a,b); dm=metric_window(d["probability"],y,a,b)
        valid=cm["ap"] is not None and dm["ap"] is not None
        delta=float(dm["ap"]-cm["ap"]) if valid else None
        if valid: deltas.append(delta); positive+=int(delta>0)
        prefixes={}
        for width in (32,64):
            cc=metric_window(c["probability"],y,a,min(b,a+width))
            dd=metric_window(d["probability"],y,a,min(b,a+width))
            prefixes[f"first{width}"]={
                "C_fixed5":cc,"D_nonblocking_reuse":dd,
                "D_minus_C_ap":None if cc["ap"] is None or dd["ap"] is None else float(dd["ap"]-cc["ap"]),
                "D_minus_C_fpr":None if cc["fpr"] is None or dd["fpr"] is None else float(dd["fpr"]-cc["fpr"]),
            }
        rows.append({
            "phase":name,"service":p["regime"],"intervals":[a,b],
            "positive_rows":int((y[a:b]>0).sum()),"negative_rows":int((y[a:b]==0).sum()),
            "C_fixed5":cm,"D_nonblocking_reuse":dm,
            "D_minus_C_ap":delta,
            "D_minus_C_fpr":None if cm["fpr"] is None or dm["fpr"] is None else float(dm["fpr"]-cm["fpr"]),
            "valid":bool(valid),"prefixes":prefixes,
        })
    idx=np.asarray(pooled,dtype=np.int64)
    pcm=binary_detection_metrics(c["probability"][idx],y[idx],.5)
    pdm=binary_detection_metrics(d["probability"][idx],y[idx],.5)
    pooled_fpr_delta=None if pcm["fpr"] is None or pdm["fpr"] is None else float(pdm["fpr"]-pcm["fpr"])
    mean=float(np.mean(deltas)) if len(deltas)==6 else None

    wrows=[]; wdeltas=[]
    for name in W_BLOCKS:
        p=pmap[name]; a,b=p["start"],p["end"]
        cm=metric_window(c["probability"],y,a,b); dm=metric_window(d["probability"],y,a,b)
        delta=None if cm["ap"] is None or dm["ap"] is None else float(dm["ap"]-cm["ap"])
        if delta is not None: wdeltas.append(delta)
        wrows.append({"phase":name,"intervals":[a,b],"C_fixed5":cm,
                      "D_nonblocking_reuse":dm,"D_minus_C_ap":delta})
    wmean=float(np.mean(wdeltas)) if len(wdeltas)==len(W_BLOCKS) else None
    signal=bool(
        mean is not None and mean>=0.03 and positive>=4 and
        pooled_fpr_delta is not None and pooled_fpr_delta<=0.01 and
        wmean is not None and wmean>=-0.02
    )
    cf=sums["C_fixed5"]["full"]["detection"]; df=sums["D_nonblocking_reuse"]["full"]["detection"]
    comparison={
        "protocol":"031","plan_revision":2,"scenario_id":p31data.SCENARIO_ID,
        "data_revision":p31data.DATA_REVISION,"run_id":str(run_id),
        "development_only":True,"confirmation_run":False,"test_run":False,
        "stream_sha256":sums["C_fixed5"]["stream_sha256"],
        "same_frozen_data_verified":sums["C_fixed5"]["stream_sha256"]==sums["D_nonblocking_reuse"]["stream_sha256"],
        "shared_first4_initialization_verified":True,
        "recurrence128":rows,
        "primary":{
            "metric":"equal-weight mean AP(D_nonblocking_reuse)-AP(C_fixed5) across six registered U/V recurrence128 windows",
            "valid_windows":sum(r["valid"] for r in rows),"required_valid_windows":6,
            "equal_weight_mean_D_minus_C_ap":mean,
            "positive_windows":positive,
            "pooled_recurrence_normal_fpr_C_fixed5":pcm["fpr"],
            "pooled_recurrence_normal_fpr_D_nonblocking_reuse":pdm["fpr"],
            "pooled_recurrence_normal_fpr_delta_D_minus_C":pooled_fpr_delta,
            "equal_weight_mean_W_block_D_minus_C_ap":wmean,
            "development_signal":signal,
            "development_reference":{
                "mean_AP_delta_min":0.03,"positive_windows_min":4,
                "pooled_recurrence_normal_FPR_delta_max":0.01,
                "mean_W_block_AP_delta_min":-0.02,
            },
        },
        "W_blocks":wrows,
        "full_stream":{
            "C_fixed5":sums["C_fixed5"]["full"],
            "D_nonblocking_reuse":sums["D_nonblocking_reuse"]["full"],
            "D_minus_C_ap":None if cf["ap"] is None or df["ap"] is None else float(df["ap"]-cf["ap"]),
            "D_minus_C_fpr":None if cf["fpr"] is None or df["fpr"] is None else float(df["fpr"]-cf["fpr"]),
        },
        "phase_metrics":{n:sums[n]["phases"] for n in COMPARATORS},
        "lifecycle":sums["D_nonblocking_reuse"].get("lifecycle"),
        "cost_profile":{n:sums[n]["cost"] for n in COMPARATORS},
        "interpretation_limits":{
            "statistical_confirmation":False,"equal_total_cost":False,
            "D_extra_background_compute_and_memory":True,
            "single_replay_seed":700,"single_model_seed":1,
            "comparison_scope":"C_fixed5 vs D_nonblocking_reuse on registered rare-recurrence scenario only",
            "A_B_inference_allowed":False,"global_D_optimality_claim_allowed":False,
        },
    }
    return comparison

def parent(args):
    deterministic_runtime()
    root=Path(args.out_root); root.mkdir(parents=True,exist_ok=True)
    if (root/"status.json").exists() or any((root/n).exists() for n in COMPARATORS):
        raise FileExistsError("existing Protocol031 model run")
    sreg=json.loads(Path(args.scenario_registration).read_text(encoding="utf8"))
    mreg=json.loads(Path(args.method_registration).read_text(encoding="utf8"))
    validate_frozen_data(args.stream,sreg,args.data_lock)
    eligibility=json.loads(Path(args.eligibility).read_text(encoding="utf8"))
    if eligibility.get("protocol031_data_eligible") is not True:
        raise RuntimeError("Protocol031 model-free audit did not pass")
    completed=[]; failures={}
    for name in COMPARATORS:
        cmd=[
            sys.executable,str(Path(__file__).resolve()),"--child-arm",name,
            "--stream",args.stream,"--arm-dir",str(root/name),
            "--scenario-registration",args.scenario_registration,
            "--method-registration",args.method_registration,
            "--data-lock",args.data_lock,"--run-id",str(args.run_id),
        ]
        so=root/f"{name}.stdout.log"; se=root/f"{name}.stderr.log"
        with so.open("w") as o,se.open("w") as e:
            proc=subprocess.run(cmd,stdout=o,stderr=e,text=True,env=dict(os.environ))
        if proc.returncode==0 and (root/name/"summary.json").is_file() and (root/name/"predictions.npz").is_file():
            completed.append(name)
        else:
            failures[name]={"returncode":proc.returncode,"stderr_tail":se.read_text(errors="replace")[-12000:]}
            write_json(root/f"{name}.failure.json",failures[name])
            break
    comp=compare(root,args.stream,args.run_id) if completed==list(COMPARATORS) else None
    if comp is not None:
        write_json(root/"comparison.json",comp)
        write_json(root/"cost_profile.json",comp["cost_profile"])
    status={
        "protocol":"031","plan_revision":2,"scenario_id":p31data.SCENARIO_ID,
        "data_revision":p31data.DATA_REVISION,"run_id":str(args.run_id),
        "completed":comp is not None,"full_model_replays_started":len(completed)+int(bool(failures)),
        "full_model_replay_budget":2,"completed_comparators":completed,
        "failed_comparators":list(failures),"failure_details":failures,
        "data_audit_passed":True,"data_frozen_before_model_runs":True,
        "C_rerun_count":1 if "C_fixed5" in completed else 0,
        "D_rerun_count":1 if "D_nonblocking_reuse" in completed else 0,
        "A_B_runs":0,"automatic_followups_started":[],
        "development_signal":None if comp is None else bool(comp["primary"]["development_signal"]),
        "blocker":None if comp is not None else "registered_model_arm_failed",
    }
    write_json(root/"status.json",status)
    print(json.dumps(json_ready({"status":status,"primary":None if comp is None else comp["primary"]}),
                     indent=2,allow_nan=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stream",required=True)
    ap.add_argument("--out-root")
    ap.add_argument("--arm-dir")
    ap.add_argument("--scenario-registration",required=True)
    ap.add_argument("--method-registration",required=True)
    ap.add_argument("--data-lock",required=True)
    ap.add_argument("--eligibility")
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--child-arm",choices=COMPARATORS)
    a=ap.parse_args()
    try:
        if a.child_arm:
            if not a.arm_dir: ap.error("--arm-dir required")
            run_arm(a.child_arm,a.stream,a.arm_dir,a.scenario_registration,
                    a.method_registration,a.data_lock,a.run_id)
        else:
            if not all((a.out_root,a.eligibility)): ap.error("--out-root/--eligibility required")
            parent(a)
    except Exception:
        root=Path(a.out_root or a.arm_dir or "."); root.mkdir(parents=True,exist_ok=True)
        write_json(root/"runner_exception.json",{
            "protocol":"031","traceback":traceback.format_exc(),
            "no_automatic_extra_replay":True,
        })
        raise

if __name__=="__main__":
    main()
