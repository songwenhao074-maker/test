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
R2_PLAN=Path("artifacts/ftmoe_online/protocol_044/revision_002/plan.json")
R2_PLAN_SHA="823365226f0e418bcc5dbdc79e0e1b77f5adf72c3e404b67fb28fdffd9cc20b3"
SCENARIO=Path("artifacts/ftmoe_online/protocol_044/scenario_registration.json")
SCENARIO_SHA="6e6bd03efc0d84f5e885eacf36f1ac80e770198703463cc051216ad1a741683f"
SOURCE_PROBE_DIGEST="sha256:1db5dd429b43759ad3e868012d8e8c850551d21e53f1cd86af0e7daf316ecf49"
ENGINEERING_CORE_DIGEST="sha256:1d01c2bd2e53ba5e9e419bcd83ca5014aaf27487c18c19d440284bf8fdb3de36"
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
    if sha256_file(p)!=PLAN_SHA: raise AssertionError("Protocol044 r1 plan hash")
    if sha256_file(SCENARIO)!=SCENARIO_SHA: raise AssertionError("Protocol044 scenario hash")
    if sha256_file(R2_PLAN)!=R2_PLAN_SHA: raise AssertionError("Protocol044 r2 plan hash")
    x=J(p); r2=J(R2_PLAN)
    if (x.get("protocol"),x.get("revision"))!=("044",1) or x["stageS"]["sequence_order"]!=list(SEQ): raise AssertionError("Protocol044 r1 plan identity")
    if (r2.get("protocol"),r2.get("revision"))!=("044",2): raise AssertionError("Protocol044 r2 plan identity")
    if r2["inherits"]["plan_sha256"]!=PLAN_SHA or r2["inherits"]["scenario_sha256"]!=SCENARIO_SHA: raise AssertionError("Protocol044 dual contract")
    if r2["budget"]["science_key"]!="protocol044_revision1_stageS_seed4401_sequence": raise AssertionError("Protocol044 science key")
    return x
def validate_input(data,input_lock):
    root=Path(data); m=J(root/"manifest.json"); lock=J(input_lock)
    if (m.get("protocol"),m.get("revision"),m.get("steps"),m.get("total_rows"))!=("044",1,N,N+1): raise AssertionError("Protocol044 stream geometry")
    digest=sha256_file(root/"stream.npz")
    if not (lock.get("protocol")=="044" and lock.get("revision")==1 and lock.get("seed")==4401 and lock.get("locked") is True): raise AssertionError("input lock")
    if digest!=lock.get("stream_sha256") or digest!=m.get("stream_sha256"): raise AssertionError("stream hash")
    return digest,m
def validate_gate(gate,input_lock,execution_sha,sequence,ledger,optimizer_calls=0,resume=False):
    if optimizer_calls!=0: raise AssertionError("gate called after optimizer")
    plan()
    g=J(gate) if not isinstance(gate,dict) else gate
    if (g.get("protocol"),g.get("execution_revision"),g.get("science_config_revision"))!=("044",2,1): raise RuntimeError("r2 gate identity")
    for k in ("E_recovery_gate","inherited_E_pass","compatibility_pass","production_entrypoint_pass","generation_recovery_pass",
              "remote_transaction_pass","budget_idempotency_pass","publication_pass"):
        if type(g.get(k)) is not bool or not g[k]: raise RuntimeError("r2 gate false "+k)
    if g.get("execution_sha")!=str(execution_sha): raise RuntimeError("execution SHA mismatch")
    if g.get("r1_plan_sha256")!=PLAN_SHA or g.get("r2_plan_sha256")!=R2_PLAN_SHA or g.get("scenario_sha256")!=SCENARIO_SHA:
        raise RuntimeError("contract hash mismatch")
    old=g.get("inherited_E") or {}
    if old.get("artifact_id")!=11296492568 or old.get("artifact_digest")!="sha256:90b6bab6a149c46da0aa19e551c02f76b5665a27a83e97d7de192e21e08b6591":
        raise RuntimeError("inherited E identity")
    lock=J(input_lock)
    if lock.get("locked") is not True or lock.get("seed")!=4401: raise RuntimeError("data lock")
    led=J(ledger); r=led["sequences"][sequence]
    if led.get("execution_revision")!=2 or led.get("science_key")!="protocol044_revision1_stageS_seed4401_sequence": raise RuntimeError("ledger execution identity")
    if r.get("completed"): raise RuntimeError("duplicate completed sequence")
    if resume:
        if r.get("started") is not True: raise RuntimeError("resume requires started sequence")
        if not r.get("state_sha256") or r.get("cursor") is None: raise RuntimeError("resume state not committed")
    else:
        if r.get("started"): raise RuntimeError("duplicate start exact resume required")
    used=int(led.get("total_optimizer_steps_used",0))
    if used<0 or used>=2360: raise RuntimeError("budget exhausted")
    if int(r.get("optimizer_steps_used",0))<0: raise RuntimeError("negative sequence budget")
    return True

def default_ledger():
    return {"protocol":"044","revision":1,"execution_revision":2,"seed":4401,"order":list(SEQ),
      "science_key":"protocol044_revision1_stageS_seed4401_sequence",
      "sequences":{s:{"started":False,"completed":False,"restart_from_zero":False,"resume_events":[],
                       "optimizer_steps_used":0,"cursor":0,"state_sha256":None,"inflight":None} for s in SEQ},
      "total_optimizer_steps_used":0,"restart_from_zero":False,"extra_streams":0,"extra_seeds":0,"extra_arms":0,"F_loads":0}
def load_ledger(path):
    p=Path(path)
    if not p.exists(): W(p,default_ledger())
    x=J(p)
    if (x.get("protocol"),x.get("revision"),x.get("execution_revision"),x.get("seed"))!=("044",1,2,4401): raise AssertionError("ledger identity")
    if x.get("science_key")!="protocol044_revision1_stageS_seed4401_sequence": raise AssertionError("science key")
    total=sum(int(v.get("optimizer_steps_used",0)) for v in x["sequences"].values())
    if total!=int(x.get("total_optimizer_steps_used",-1)) or total>2360: raise AssertionError("ledger optimizer accounting")
    return x
def start_seq(path,name,run_id,resume=None):
    x=load_ledger(path); r=x["sequences"][name]
    if r["completed"]: raise RuntimeError("sequence already completed")
    if r["started"]:
        if not resume: raise RuntimeError("sequence already started exact resume required")
        r["resume_events"].append({"run_id":str(run_id),"checkpoint":str(resume),"cursor":int(r["cursor"]),"optimizer_steps_used":int(r["optimizer_steps_used"])})
    else:
        if resume: raise RuntimeError("resume before start")
        r["started"]=True; r["started_run_id"]=str(run_id); r["cursor"]=0; r["optimizer_steps_used"]=0
    W(path,x)
def record_progress(path,name,cursor,steps,state_sha256,inflight=None):
    x=load_ledger(path); r=x["sequences"][name]
    if not r["started"] or r["completed"]: raise RuntimeError("progress on illegal sequence state")
    cursor=int(cursor); steps=int(steps)
    if cursor<int(r.get("cursor",0)) or steps<int(r.get("optimizer_steps_used",0)): raise RuntimeError("nonmonotone progress")
    r["cursor"]=cursor; r["optimizer_steps_used"]=steps; r["state_sha256"]=str(state_sha256); r["inflight"]=inflight
    x["total_optimizer_steps_used"]=sum(int(v.get("optimizer_steps_used",0)) for v in x["sequences"].values())
    if x["total_optimizer_steps_used"]>2360: raise RuntimeError("global optimizer budget")
    W(path,x)
def complete_seq(path,name,steps,extra=None):
    x=load_ledger(path); r=x["sequences"][name]
    if not r["started"]: raise AssertionError("complete unstarted")
    if r["completed"]: raise AssertionError("duplicate complete")
    steps=int(steps)
    if steps!=int(r.get("optimizer_steps_used",0)): raise AssertionError("final optimizer accounting mismatch")
    r["completed"]=True; r["optimizer_steps"]=steps; r["inflight"]=None
    if extra:r.update(extra)
    x["total_optimizer_steps_used"]=sum(int(v.get("optimizer_steps_used",0)) for v in x["sequences"].values())
    if x["total_optimizer_steps_used"]>2360: raise AssertionError("global optimizer budget")
    W(path,x)

def assert_resume_binding(path,name,cursor,state_sha256):
    x=load_ledger(path); r=x["sequences"][name]
    if not r.get("started") or r.get("completed"): raise RuntimeError("resume binding illegal sequence state")
    if int(r.get("cursor",-1))!=int(cursor): raise RuntimeError("resume cursor mismatch")
    if str(r.get("state_sha256"))!=str(state_sha256): raise RuntimeError("resume state hash mismatch")
    if int(r.get("optimizer_steps_used",-1))<0: raise RuntimeError("resume budget invalid")
    return r


def make_c(data,input_lock,out,run_id):
    digest,m=validate_input(data,input_lock)
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
    resume=bool(a.resume_label)
    validate_gate(a.gate,a.input_lock,a.execution_sha,"C_ref",a.ledger,0,resume=resume)
    start_seq(a.ledger,"C_ref",a.run_id,a.resume_label)
    s,digest,m=make_c(a.data,a.input_lock,out,a.run_id)
    if a.resume_label:
        meta_path=out/"checkpoints"/(a.resume_label+".meta.json")
        if not meta_path.is_file(): raise FileNotFoundError(meta_path)
        meta=J(meta_path); assert_resume_binding(a.ledger,"C_ref",meta["cursor"],sha256_file(meta_path))
        restore_c_ckpt(s,out,a.resume_label)
    target=N if a.stop_cursor is None else min(N,int(a.stop_cursor))
    if target<int(s.cursor) or target-int(s.cursor)>512: raise RuntimeError("C segment cursor bound")
    ini=r36.initinfo(s); t0=time.perf_counter()
    while s.cursor<target: s.step()
    label="final" if target==N else "cursor_%04d"%target
    if target==N:
        s.finish()
    save_c_ckpt(s,out,label)
    meta_path=out/"checkpoints"/(label+".meta.json")
    record_progress(a.ledger,"C_ref",s.cursor,s.updates,sha256_file(meta_path),None)
    W(out/"segment_status.json",{"protocol":"044","execution_revision":2,"arm":"C_ref","cursor":int(s.cursor),
      "optimizer_steps_used":int(s.updates),"checkpoint_label":label,"checkpoint_meta_sha256":sha256_file(meta_path),
      "completed":bool(target==N)})
    if target<N: return
    s.save()
    updates=[]
    for r in s.update_log:
        batch=[int(x) for x in r["buffer_indices"]]
        if batch and max(batch)+2>int(r["at_interval"]): raise AssertionError("immature C batch")
        updates.append({"at_interval":int(r["at_interval"]),"opportunity":int(r["opportunity"]),"gradient_steps":int(r["gradient_steps"]),"buffer_size":int(r["buffer_size"]),"batch_indices":batch})
    np.savez_compressed(out/"feature_tape.npz",z=s.z_tape,c_detection_logits=s.predictions["detection_logits"],c_probability=s.predictions["probability"],
      c_class_probability=s.predictions["class_probability"],labels=s.predictions["labels"],raw_labels=s.predictions["raw_labels"],
      model_version=s.predictions["model_version"],visible_input_max=s.visible,label_available_max=s.labelmax)
    W(out/"update_batches.json",{"protocol":"044","revision":1,"execution_revision":2,"seed":4401,"arm":"C_ref","updates":updates})
    checks={"final_prediction_index":N-1,"all_settled":bool((s.predictions["labels"]>=0).all()),"terminal_training_steps":0,
      "visible_exact":bool(np.array_equal(s.visible,np.arange(N))),"label_cutoff_exact":bool(np.array_equal(s.labelmax,np.arange(N)-3)),
      "optimizer_steps":int(s.updates),"optimizer_budget":int(s.updates)<=372,"frozen_base_sha256":ini["frozen_base_sha256"],
      "normalization_buffers_sha256":ini["normalization_buffers_sha256"]}
    if not all(v for k,v in checks.items() if k in ("all_settled","visible_exact","label_cutoff_exact","optimizer_budget")): raise AssertionError("C checks")
    W(out/"summary.json",{"protocol":"044","revision":1,"execution_revision":2,"arm":"C_ref","stream_sha256":digest,"completed":True,"initialization":ini,
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
    runtime(); plan(); resume=bool(a.resume_from)
    validate_gate(a.gate,a.input_lock,a.execution_sha,"D_lin",a.ledger,0,resume=resume); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    start_seq(a.ledger,"D_lin",a.run_id,a.resume_from)
    tp=Path(a.feature_tape); tape_sha=sha256_file(tp)
    with np.load(tp,allow_pickle=False) as q:T={k:q[k].copy() for k in q.files}
    if T["z"].shape!=(N,16,73): raise AssertionError("tape geometry")
    batches=J(a.update_batches)["updates"]; by={int(x["at_interval"]):x for x in batches}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3501); b=r36.LinearCorrection()
    opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    pred=np.full((N,16),np.nan,np.float32);delta=np.full((N,16),np.nan,np.float32);logits=np.full((N,16,2),np.nan,np.float32)
    versions=np.zeros(N,np.int32);hashes=[];logs=[];ver=0;cursor=0
    if a.resume_from:
        rp=Path(a.resume_from); assert_resume_binding(a.ledger,"D_lin",J(str(rp)+".json")["cursor"],sha256_file(rp))
        x=restore_branch(rp,b,opt,tape_sha);cursor=x["cursor"];ver=x["version"];pred[:cursor]=x["pred"];delta[:cursor]=x["delta"];logits[:cursor]=x["logits"];versions[:cursor]=x["versions"];hashes=list(x["hashes"]);logs=list(x["logs"])
        if ver!=int(load_ledger(a.ledger)["sequences"]["D_lin"]["optimizer_steps_used"]): raise RuntimeError("D_lin optimizer ledger mismatch")
    target=N if a.stop_cursor is None else min(N,int(a.stop_cursor))
    if target<cursor or target-cursor>512: raise RuntimeError("D_lin segment cursor bound")
    for t in range(cursor,target):
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
    save_branch(out,b,opt,target,ver,pred,delta,logits,versions,hashes,logs,tape_sha)
    cp=branch_ckpt(out,target); record_progress(a.ledger,"D_lin",target,ver,sha256_file(cp),None)
    W(out/"segment_status.json",{"protocol":"044","execution_revision":2,"arm":"D_lin","cursor":target,"optimizer_steps_used":ver,
      "checkpoint":cp.name,"checkpoint_sha256":sha256_file(cp),"completed":bool(target==N)})
    if target<N: return
    np.savez_compressed(out/"predictions.npz",probability=pred,detection_logits=logits,delta=delta,class_probability=T["c_class_probability"],
      labels=T["labels"],raw_labels=T["raw_labels"],model_version=versions)
    W(out/"update_log.json",{"updates":logs});W(out/"summary.json",{"protocol":"044","revision":1,"execution_revision":2,"arm":"D_lin","completed":True,"updates":ver,
      "feature_tape_sha256":tape_sha,"predictions_sha256":sha256_file(out/"predictions.npz"),"parameter_count":sum(p.numel() for p in b.parameters()),"zero_initialized":True})
    if ver>372: raise AssertionError("Dlin optimizer budget")
    complete_seq(a.ledger,"D_lin",ver,{"predictions_sha256":sha256_file(out/"predictions.npz")})

def load_dynamic_source(feature_tape,b_pred,update_batches):
    with np.load(feature_tape,allow_pickle=False) as q:T={k:q[k].copy() for k in q.files}
    with np.load(b_pred,allow_pickle=False) as q:B={k:q[k].copy() for k in q.files}
    ups=J(update_batches)["updates"];by={int(x["at_interval"]):x for x in ups}
    return {"core":{"T":{"z":T["z"],"labels":B["labels"]},"B":B,"updates":ups,"by_t":by},"B":B,"C":B,"D_keep":B,"D_039":B,"D_040":B,"n":N,"manifest":{"timeline":[]}}

def cmd_dynamic(a):
    runtime(); plan(); arm=a.arm; resume=bool(a.resume_from)
    validate_gate(a.gate,a.input_lock,a.execution_sha,arm,a.ledger,0,resume=resume); start_seq(a.ledger,arm,a.run_id,a.resume_from)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);src=load_dynamic_source(a.feature_tape,a.b_predictions,a.update_batches)
    if a.resume_from:
        rp=Path(a.resume_from); assert_resume_binding(a.ledger,arm,J(str(rp)+".json")["cursor"],sha256_file(rp))
        m=Machine044.restore(src,out,rp,arm=arm,allow_gradient=True,real_science=True,strict_journal=True)
        if m.actual_optimizer_calls!=int(load_ledger(a.ledger)["sequences"][arm]["optimizer_steps_used"]): raise RuntimeError("dynamic optimizer ledger mismatch")
    else:
        m=Machine044(src,out,arm,allow_gradient=True,real_science=True)
    m.checkpoint_dir=out/"checkpoints"
    target=N if a.stop_cursor is None else min(N,int(a.stop_cursor))
    if target<int(m.cursor) or target-int(m.cursor)>512: raise RuntimeError("dynamic segment cursor bound")
    m.advance(target)
    if target<N:
        m.save_checkpoint("science_cursor_%04d"%target,True)
        cp=out/"checkpoints/latest.pt"; record_progress(a.ledger,arm,m.cursor,m.actual_optimizer_calls,sha256_file(cp),None)
        W(out/"segment_status.json",{"protocol":"044","execution_revision":2,"arm":arm,"cursor":m.cursor,
          "optimizer_steps_used":m.actual_optimizer_calls,"checkpoint_sha256":sha256_file(cp),"completed":False})
        return
    m.terminal_settle()
    cp=out/"checkpoints/latest.pt"; record_progress(a.ledger,arm,m.cursor,m.actual_optimizer_calls,sha256_file(cp),None)
    arr={k:np.asarray(v).copy() for k,v in m.out.items() if k!="contribution"}
    arr.update({"class_probability":src["B"]["class_probability"].copy(),"labels":src["B"]["labels"].copy(),"raw_labels":src["B"]["raw_labels"].copy(),
      "model_version":src["B"].get("model_version",np.zeros(N,np.int32)).copy()})
    np.savez_compressed(out/"predictions.npz",**arr)
    summary={"protocol":"044","revision":1,"execution_revision":2,"arm":arm,"completed":True,"first_birth_t":m.first_birth_t,"ids_created":m.next_id,
      "accepted_ids":sorted(m.experts),"active_ids_final":list(m.active_ids),"deleted_ids":sorted(m.deleted_ids),"permanent_deletions":m.permanent_deletions,
      "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,"actual_optimizer_calls":m.actual_optimizer_calls,
      "deployed_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,"shadow_preview_forwards":m.shadow_preview_forwards,
      "max_active_seen":m.max_active_seen,"max_resident_seen":m.max_resident_seen,"online_payload_bytes":m.online_payload_bytes(),
      "tracker_rebuilds":{"periodic":m.tracker.periodic_rebuild_count,"near":m.tracker.near_threshold_rebuild_count,"rank":m.tracker.rank_rebuild_count},
      "terminal_counter_delta":m.terminal_counter_delta,"predictions_sha256":sha256_file(out/"predictions.npz")}
    W(out/"summary.json",summary)
    if m.actual_optimizer_calls>808: raise AssertionError("dynamic optimizer budget")
    complete_seq(a.ledger,arm,m.actual_optimizer_calls,{"predictions_sha256":summary["predictions_sha256"],"permanent_deletions":m.permanent_deletions})
    W(out/"segment_status.json",{"protocol":"044","execution_revision":2,"arm":arm,"cursor":m.cursor,
      "optimizer_steps_used":m.actual_optimizer_calls,"checkpoint_sha256":sha256_file(cp),"completed":True})

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("C_ref");p.add_argument("--data",required=True);p.add_argument("--input-lock",required=True);p.add_argument("--gate",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--ledger",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True);p.add_argument("--resume-label");p.add_argument("--stop-cursor",type=int);p.set_defaults(fn=cmd_c)
    p=sp.add_parser("D_lin");p.add_argument("--feature-tape",required=True);p.add_argument("--update-batches",required=True);p.add_argument("--input-lock",required=True);p.add_argument("--gate",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--ledger",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True);p.add_argument("--resume-from");p.add_argument("--stop-cursor",type=int);p.set_defaults(fn=cmd_lin)
    p=sp.add_parser("dynamic");p.add_argument("--arm",choices=("D_no_gc","D_bounded"),required=True);p.add_argument("--feature-tape",required=True);p.add_argument("--b-predictions",required=True);p.add_argument("--update-batches",required=True);p.add_argument("--input-lock",required=True);p.add_argument("--gate",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--ledger",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True);p.add_argument("--resume-from");p.add_argument("--stop-cursor",type=int);p.set_defaults(fn=cmd_dynamic)
    a=ap.parse_args();a.fn(a)
if __name__=="__main__": main()
