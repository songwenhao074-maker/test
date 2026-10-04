"""Protocol-044 Stage-S frozen sequence runner: C_ref -> D_lin -> D_no_gc -> D_bounded."""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
import reference_protocol036_locked as r36
import protocol044_common as c
from protocol044_engine import Machine044
from protocol035_common import sha256_file, sha256_state_dict

N=5952
SEQ=("C_ref","D_lin","D_no_gc","D_bounded")
PLAN_SHA="a27a6cca6abde40b6395876eccd3db5f550a044e952a701fe63aa2e4306a5f51"
SOURCE_PROBE_DIGEST="sha256:1db5dd429b43759ad3e868012d8e8c850551d21e53f1cd86af0e7daf316ecf49"
ENGINEERING_CORE_DIGEST="sha256:35091be2e2e3ddb6d1c8d7da37496f85fcd6ce991d43f2e394e76d633032f27b"
REQUIRED_CASES=("birth_after","single_live_after","joint_live_after","shadow_update01_after","shadow_update15_after","shadow_update16_after",
"qualification_ready_before_decision","shadow_accept_after","shadow_reject_after","sleep_after","reuse_pending","reuse_accept_after",
"reuse_reject_after","window_expiry_after","late_old_epoch_settlement","reclaim_create_after","reclaimed_candidate_reject_after",
"terminal_first_before_second","terminal_second_before_finalize")

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def runtime():
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="1"
    os.environ["PYTHONHASHSEED"]="1"; torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.use_deterministic_algorithms(True,warn_only=False)
def plan():
    p=Path("artifacts/ftmoe_online/protocol_044/plan.json")
    if sha256_file(p)!=PLAN_SHA: raise AssertionError("Protocol044 plan hash")
    x=J(p)
    if (x.get("protocol"),x.get("revision"))!=("044",1) or x["stageS"]["sequence_order"]!=list(SEQ): raise AssertionError("Protocol044 plan identity")
    return x
def validate_input(data,input_lock):
    root=Path(data); m=J(root/"manifest.json"); lock=J(input_lock)
    if (m.get("protocol"),m.get("revision"),m.get("steps"),m.get("total_rows"))!=("044",1,N,N+1): raise AssertionError("Protocol044 stream geometry")
    digest=sha256_file(root/"stream.npz")
    if not (lock.get("protocol")=="044" and lock.get("revision")==1 and lock.get("seed")==4401 and lock.get("locked") is True): raise AssertionError("input lock")
    if digest!=lock.get("stream_sha256") or digest!=m.get("stream_sha256"): raise AssertionError("stream hash")
    return digest,m
def validate_gate(gate,input_lock,execution_sha,sequence,ledger,optimizer_calls=0):
    if optimizer_calls!=0: raise AssertionError("gate called after optimizer")
    g=J(gate) if not isinstance(gate,dict) else gate
    if (g.get("protocol"),g.get("revision"))!=("044",1): raise RuntimeError("gate identity")
    for k in ("E_gate","source_lock_pass","engineering_core_pass","crash_pass","fail_closed_pass","baseline_adapter_pass"):
        if type(g.get(k)) is not bool or not g[k]: raise RuntimeError("gate false "+k)
    if g.get("execution_sha")!=str(execution_sha): raise RuntimeError("execution SHA mismatch")
    if g.get("source_probe_artifact_digest")!=SOURCE_PROBE_DIGEST: raise RuntimeError("bad source hash")
    if g.get("engineering_core_artifact_digest")!=ENGINEERING_CORE_DIGEST: raise RuntimeError("bad engineering hash")
    if tuple(g.get("required_resume_case_ids") or ())!=REQUIRED_CASES or tuple(g.get("resume_case_ids_passed") or ())!=REQUIRED_CASES:
        raise RuntimeError("missing required case")
    lock=J(input_lock)
    if lock.get("locked") is not True or lock.get("seed")!=4401: raise RuntimeError("data lock")
    led=J(ledger); r=led["sequences"][sequence]
    if r.get("completed"): raise RuntimeError("duplicate completed sequence")
    if r.get("started"): raise RuntimeError("duplicate start exact resume required")
    if led.get("total_optimizer_steps",0)>=2360: raise RuntimeError("budget exhausted")
    if g.get("active_max")!=2 or g.get("resident_max")!=3 or g.get("shadow_max")!=1: raise RuntimeError("illegal capacity gate")
    return True
def default_ledger():
    return {"protocol":"044","revision":1,"seed":4401,"order":list(SEQ),
      "sequences":{s:{"started":False,"completed":False,"restart_from_zero":False,"resume_events":[]} for s in SEQ},
      "total_optimizer_steps":0,"restart_from_zero":False,"extra_streams":0,"extra_seeds":0,"extra_arms":0,"F_loads":0}
def load_ledger(path):
    p=Path(path)
    if not p.exists(): W(p,default_ledger())
    x=J(p)
    if (x.get("protocol"),x.get("revision"),x.get("seed"))!=("044",1,4401): raise AssertionError("ledger identity")
    return x
def start_seq(path,name,run_id,resume=None):
    x=load_ledger(path); r=x["sequences"][name]
    if r["completed"]: raise RuntimeError("sequence already completed")
    if r["started"]:
        if not resume: raise RuntimeError("sequence already started exact resume required")
        r["resume_events"].append({"run_id":str(run_id),"checkpoint":str(resume)})
    else:
        if resume: raise RuntimeError("resume before start")
        r["started"]=True; r["started_run_id"]=str(run_id)
    W(path,x)
def complete_seq(path,name,steps,extra=None):
    x=load_ledger(path); r=x["sequences"][name]
    if not r["started"]: raise AssertionError("complete unstarted")
    if r["completed"]: raise AssertionError("duplicate complete")
    r["completed"]=True; r["optimizer_steps"]=int(steps)
    if extra:r.update(extra)
    x["total_optimizer_steps"]=int(sum(int(v.get("optimizer_steps",0)) for v in x["sequences"].values()))
    if x["total_optimizer_steps"]>2360: raise AssertionError("global optimizer budget")
    W(path,x)

def make_c(data,out,run_id):
    digest,m=validate_input(data,Path(out).parent/"input_lock.json")
    b=s4.build_replay(Path(data))
    if b["steps"]!=N or b["manifest"]["stream_sha256"]!=digest: raise AssertionError("replay identity")
    reg={"protocol":"044","revision":1,"arm":"C_ref","stream_sha256":digest,"replay_seed":4401,"model_seed":1}
    sess=r36.ObservedFixedSession036("C_fixed5",1,b,p31.budget(),Path(out),run_id=str(run_id),stream_dir=Path(data),phase_defs=p31.phases(m),stream_sha=digest,registration=reg,learning_rate=1e-4)
    return sess,digest,m

def save_c_ckpt(s,out,label):
    rec=s.save_checkpoint(label,s.cursor)
    ep=Path(out)/"checkpoints"/(label+".extra.npz")
    np.savez_compressed(ep,z=s.z_tape[:s.cursor],visible=s.visible[:s.cursor],labelmax=s.labelmax[:s.cursor])
    W(Path(out)/"checkpoints"/(label+".meta.json"),{"cursor":int(s.cursor),"base":rec["path"],"base_sha256":rec["sha256"],"extra":ep.name,"extra_sha256":sha256_file(ep),"learner_hash":s.learner_state_hash()})
def restore_c_ckpt(s,out,label):
    meta=J(Path(out)/"checkpoints"/(label+".meta.json")); base=Path(meta["base"])
    if not base.exists(): base=Path(out)/"checkpoints"/base.name
    if sha256_file(base)!=meta["base_sha256"]: raise AssertionError("C checkpoint hash")
    ep=Path(out)/"checkpoints"/meta["extra"]
    if sha256_file(ep)!=meta["extra_sha256"]: raise AssertionError("C extra hash")
    s.restore_checkpoint(base)
    with np.load(ep,allow_pickle=False) as z:
        n=int(meta["cursor"]); s.z_tape[:n]=z["z"]; s.visible[:n]=z["visible"]; s.labelmax[:n]=z["labelmax"]
    if s.learner_state_hash()!=meta["learner_hash"]: raise AssertionError("C learner resume")

def cmd_c(a):
    runtime(); plan(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    validate_gate(a.gate,a.input_lock,a.execution_sha,"C_ref",a.ledger,0); start_seq(a.ledger,"C_ref",a.run_id,a.resume_label)
    s,digest,m=make_c(a.data,a.input_lock,out,a.run_id)
    if a.resume_label: restore_c_ckpt(s,out,a.resume_label)
    ini=r36.initinfo(s); t0=time.perf_counter()
    while s.cursor<N:
        s.step()
        if s.cursor%512==0 and not (out/"checkpoints"/("cursor_%04d.meta.json"%s.cursor)).exists(): save_c_ckpt(s,out,"cursor_%04d"%s.cursor)
    s.finish()
    if not (out/"checkpoints/final.meta.json").exists(): save_c_ckpt(s,out,"final")
    s.save()
    updates=[]
    for r in s.update_log:
        batch=[int(x) for x in r["buffer_indices"]]
        if batch and max(batch)+2>int(r["at_interval"]): raise AssertionError("immature C batch")
        updates.append({"at_interval":int(r["at_interval"]),"opportunity":int(r["opportunity"]),"gradient_steps":int(r["gradient_steps"]),"buffer_size":int(r["buffer_size"]),"batch_indices":batch})
    np.savez_compressed(out/"feature_tape.npz",z=s.z_tape,c_detection_logits=s.predictions["detection_logits"],c_probability=s.predictions["probability"],
      c_class_probability=s.predictions["class_probability"],labels=s.predictions["labels"],raw_labels=s.predictions["raw_labels"],
      model_version=s.predictions["model_version"],visible_input_max=s.visible,label_available_max=s.labelmax)
    W(out/"update_batches.json",{"protocol":"044","revision":1,"seed":4401,"arm":"C_ref","updates":updates})
    checks={"final_prediction_index":N-1,"all_settled":bool((s.predictions["labels"]>=0).all()),"terminal_training_steps":0,
      "visible_exact":bool(np.array_equal(s.visible,np.arange(N))),"label_cutoff_exact":bool(np.array_equal(s.labelmax,np.arange(N)-3)),
      "optimizer_steps":int(s.updates),"optimizer_budget":int(s.updates)<=372,"frozen_base_sha256":ini["frozen_base_sha256"],
      "normalization_buffers_sha256":ini["normalization_buffers_sha256"]}
    if not all(v for k,v in checks.items() if k in ("all_settled","visible_exact","label_cutoff_exact","optimizer_budget")): raise AssertionError("C checks")
    W(out/"summary.json",{"protocol":"044","revision":1,"arm":"C_ref","stream_sha256":digest,"completed":True,"initialization":ini,
      "feature_tape_sha256":sha256_file(out/"feature_tape.npz"),"predictions_sha256":sha256_file(out/"predictions.npz"),"checks":checks,"wall_seconds":time.perf_counter()-t0})
    complete_seq(a.ledger,"C_ref",s.updates,{"feature_tape_sha256":sha256_file(out/"feature_tape.npz")})

def branch_ckpt(out,cursor): return Path(out)/"checkpoints"/("cursor_%04d.pt"%int(cursor))
def save_branch(out,b,opt,cursor,ver,pred,delta,logits,versions,hashes,logs,tape_sha):
    p=branch_ckpt(out,cursor); p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"cursor":int(cursor),"version":int(ver),"model":b.state_dict(),"optimizer":opt.state_dict(),"pred":pred[:cursor],"delta":delta[:cursor],
      "logits":logits[:cursor],"versions":versions[:cursor],"hashes":list(hashes[:cursor]),"logs":logs,"torch_rng":torch.get_rng_state(),
      "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"tape_sha":tape_sha},p)
    W(str(p)+".json",{"sha256":sha256_file(p),"cursor":int(cursor),"tape_sha":tape_sha})
def restore_branch(path,b,opt,tape_sha):
    p=Path(path); meta=J(str(p)+".json")
    if sha256_file(p)!=meta["sha256"] or meta["tape_sha"]!=tape_sha: raise AssertionError("branch checkpoint")
    x=torch.load(p,map_location="cpu",weights_only=False); b.load_state_dict(x["model"]); opt.load_state_dict(x["optimizer"])
    torch.set_rng_state(x["torch_rng"]);np.random.set_state(x["numpy_rng"]);random.setstate(x["python_rng"]); return x

def cmd_lin(a):
    runtime(); plan(); validate_gate(a.gate,a.input_lock,a.execution_sha,"D_lin",a.ledger,0); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    start_seq(a.ledger,"D_lin",a.run_id,a.resume_from)
    tp=Path(a.feature_tape); tape_sha=sha256_file(tp)
    with np.load(tp,allow_pickle=False) as q:T={k:q[k].copy() for k in q.files}
    if T["z"].shape!=(N,16,73): raise AssertionError("tape geometry")
    batches=J(a.update_batches)["updates"]; by={int(x["at_interval"]):x for x in batches}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3501)
        b=r36.LinearCorrection()
    opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    pred=np.full((N,16),np.nan,np.float32);delta=np.full((N,16),np.nan,np.float32);logits=np.full((N,16,2),np.nan,np.float32)
    versions=np.zeros(N,np.int32);hashes=[];logs=[];ver=0;cursor=0
    if a.resume_from:
        x=restore_branch(a.resume_from,b,opt,tape_sha);cursor=x["cursor"];ver=x["version"];pred[:cursor]=x["pred"];delta[:cursor]=x["delta"];logits[:cursor]=x["logits"];versions[:cursor]=x["versions"];hashes=list(x["hashes"]);logs=list(x["logs"])
    for t in range(cursor,N):
        z=torch.from_numpy(T["z"][t]).float(); cl=torch.from_numpy(T["c_detection_logits"][t]).float(); m=cl[:,1]-cl[:,0]
        with torch.no_grad(): d=b.delta(z,m); oo=r36.compose(cl,d); pp=torch.softmax(oo,-1)[:,1]
        pred[t]=pp.numpy();delta[t]=d.numpy();logits[t]=oo.numpy();versions[t]=ver;hashes.append(r36.bhash(b))
        if t in by:
            ii=[int(i) for i in by[t]["batch_indices"]]
            if any(i+2>t for i in ii): raise AssertionError("immature Dlin batch")
            zt=torch.from_numpy(T["z"][ii]).float(); cm=torch.from_numpy(T["c_detection_logits"][ii,:,1]-T["c_detection_logits"][ii,:,0]).float()
            yy=torch.from_numpy((T["labels"][ii]>0).astype(np.float32)); d2=b.delta(zt,cm); ce=F.binary_cross_entropy_with_logits(cm+d2,yy);pen=.001*(d2*d2).mean();loss=ce+pen
            opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(b.parameters(),1.0));opt.step();ver+=1
            logs.append({"at_interval":t,"batch_indices":ii,"loss":float(loss.detach()),"bce":float(ce.detach()),"penalty":float(pen.detach()),"grad_norm":gn,"version_after":ver,"hash_after":r36.bhash(b)})
        if (t+1)%512==0: save_branch(out,b,opt,t+1,ver,pred,delta,logits,versions,hashes,logs,tape_sha)
    save_branch(out,b,opt,N,ver,pred,delta,logits,versions,hashes,logs,tape_sha)
    np.savez_compressed(out/"predictions.npz",probability=pred,detection_logits=logits,delta=delta,class_probability=T["c_class_probability"],
      labels=T["labels"],raw_labels=T["raw_labels"],model_version=versions)
    W(out/"update_log.json",{"updates":logs});W(out/"summary.json",{"protocol":"044","revision":1,"arm":"D_lin","completed":True,"updates":ver,
      "feature_tape_sha256":tape_sha,"predictions_sha256":sha256_file(out/"predictions.npz"),"parameter_count":sum(p.numel() for p in b.parameters()),"zero_initialized":True})
    if ver>372: raise AssertionError("Dlin optimizer budget")
    complete_seq(a.ledger,"D_lin",ver,{"predictions_sha256":sha256_file(out/"predictions.npz")})

def load_dynamic_source(feature_tape,b_pred,update_batches):
    with np.load(feature_tape,allow_pickle=False) as q:T={k:q[k].copy() for k in q.files}
    with np.load(b_pred,allow_pickle=False) as q:B={k:q[k].copy() for k in q.files}
    ups=J(update_batches)["updates"];by={int(x["at_interval"]):x for x in ups}
    return {"core":{"T":{"z":T["z"],"labels":B["labels"]},"B":B,"updates":ups,"by_t":by},"B":B,"C":B,"D_keep":B,"D_039":B,"D_040":B,"n":N,"manifest":{"timeline":[]}}

def cmd_dynamic(a):
    runtime(); plan(); arm=a.arm
    validate_gate(a.gate,a.input_lock,a.execution_sha,arm,a.ledger,0); start_seq(a.ledger,arm,a.run_id,a.resume_from)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);src=load_dynamic_source(a.feature_tape,a.b_predictions,a.update_batches)
    if a.resume_from:
        m=Machine044.restore(src,out,a.resume_from,arm=arm,allow_gradient=True,real_science=True,strict_journal=True)
    else:
        m=Machine044(src,out,arm,allow_gradient=True,real_science=True)
    m.checkpoint_dir=out/"checkpoints";m.advance();m.terminal_settle()
    arr={k:np.asarray(v).copy() for k,v in m.out.items() if k!="contribution"}
    arr.update({"class_probability":src["B"]["class_probability"].copy(),"labels":src["B"]["labels"].copy(),"raw_labels":src["B"]["raw_labels"].copy(),
      "model_version":src["B"].get("model_version",np.zeros(N,np.int32)).copy()})
    np.savez_compressed(out/"predictions.npz",**arr)
    summary={"protocol":"044","revision":1,"arm":arm,"completed":True,"first_birth_t":m.first_birth_t,"ids_created":m.next_id,
      "accepted_ids":sorted(m.experts),"active_ids_final":list(m.active_ids),"deleted_ids":sorted(m.deleted_ids),"permanent_deletions":m.permanent_deletions,
      "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,"actual_optimizer_calls":m.actual_optimizer_calls,
      "deployed_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,"shadow_preview_forwards":m.shadow_preview_forwards,
      "max_active_seen":m.max_active_seen,"max_resident_seen":m.max_resident_seen,"online_payload_bytes":m.online_payload_bytes(),
      "tracker_rebuilds":{"periodic":m.tracker.periodic_rebuild_count,"near":m.tracker.near_threshold_rebuild_count,"rank":m.tracker.rank_rebuild_count},
      "terminal_counter_delta":m.terminal_counter_delta,"predictions_sha256":sha256_file(out/"predictions.npz")}
    W(out/"summary.json",summary)
    if m.actual_optimizer_calls>808: raise AssertionError("dynamic optimizer budget")
    complete_seq(a.ledger,arm,m.actual_optimizer_calls,{"predictions_sha256":summary["predictions_sha256"],"permanent_deletions":m.permanent_deletions})

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("C_ref");p.add_argument("--data",required=True);p.add_argument("--input-lock",required=True);p.add_argument("--gate",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--ledger",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True);p.add_argument("--resume-label");p.set_defaults(fn=cmd_c)
    p=sp.add_parser("D_lin");p.add_argument("--feature-tape",required=True);p.add_argument("--update-batches",required=True);p.add_argument("--input-lock",required=True);p.add_argument("--gate",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--ledger",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True);p.add_argument("--resume-from");p.set_defaults(fn=cmd_lin)
    p=sp.add_parser("dynamic");p.add_argument("--arm",choices=("D_no_gc","D_bounded"),required=True);p.add_argument("--feature-tape",required=True);p.add_argument("--b-predictions",required=True);p.add_argument("--update-batches",required=True);p.add_argument("--input-lock",required=True);p.add_argument("--gate",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--ledger",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True);p.add_argument("--resume-from");p.set_defaults(fn=cmd_dynamic)
    a=ap.parse_args();a.fn(a)
if __name__=="__main__": main()
