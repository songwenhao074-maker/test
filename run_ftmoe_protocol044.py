"""Protocol-044 engineering closeout and dynamic science entrypoint."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,pickle,random,shutil,sys,time
from collections import deque
from pathlib import Path
import numpy as np
import torch

import protocol044_common as c
from protocol044_engine import Machine044,EngineeringCheckpointStop,semantic_digest,semantic_state

CASES=(
"birth_after","single_live_after","joint_live_after","shadow_update01_after","shadow_update15_after","shadow_update16_after",
"qualification_ready_before_decision","shadow_accept_after","shadow_reject_after","sleep_after","reuse_pending","reuse_accept_after",
"reuse_reject_after","window_expiry_after","late_old_epoch_settlement","reclaim_create_after","reclaimed_candidate_reject_after",
"terminal_first_before_second","terminal_second_before_finalize")
def J(p):return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def runtime():
    torch.set_num_threads(1);random.seed(1);np.random.seed(1);torch.manual_seed(1);np.seterr(all="raise")
def set_weight(e,scale,sign=1.0):
    v=np.zeros(c.DIM,np.float32);v[0]=float(sign)/float(scale);c.set_linear(e,v.reshape(1,-1),0.0)
def lifecycle_source(n=820,scale=20.0,switch=256):
    src=c.synthetic_source(n=n,seed=4411,feature_scale=.01)
    y=np.zeros((n,c.HOSTS),np.int64)
    for t in range(n):y[t,(np.arange(c.HOSTS)+t)%2==0]=1
    sg=(2*y-1).astype(np.float32);z=np.zeros((n,c.HOSTS,c.DIM),np.float32);z[...,0]=sg*float(scale)
    margin=2.0*sg
    if switch is not None:margin[int(switch):]=-2.0*sg[int(switch):]
    logits=np.stack((-margin/2,margin/2),axis=-1).astype(np.float32);prob=c.sigmoid_margin(margin)
    cls=np.zeros((n,c.HOSTS,4),np.float32);cls[...,0]=1
    B={"detection_logits":logits,"probability":prob,"labels":y.copy(),"raw_labels":y.copy(),"class_probability":cls.copy(),"model_version":np.zeros(n,np.int64)}
    src["core"]["T"]["z"]=z;src["core"]["T"]["labels"]=y.copy();src["core"]["B"]=B;src["B"]=B;src["C"]=B;src["D_keep"]=B;src["D_039"]=B;src["D_040"]=B
    return src
def one_active(src,work,scale=2.0,sign=1.0):
    m=Machine044(src,work,"D_bounded",True,False);e=c.new_expert(0,-1,"active");set_weight(e,scale,sign);e["first_active_t"]=0
    m.experts={0:e};m.active_ids=[0];m.next_id=1;m.tracker.reset(0,0,[0],"fixture_one_active");return m
def active_pair(src,work,harmful=False):
    m=Machine044(src,work,"D_bounded",True,False);scale=2.0
    e0=c.new_expert(0,-1,"active");set_weight(e0,scale,1);e0["first_active_t"]=0
    e1=c.new_expert(1,-1,"active");set_weight(e1,scale,-1 if harmful else 1);e1["first_active_t"]=0
    m.experts={0:e0,1:e1};m.active_ids=[0,1];m.next_id=2;m.tracker.reset(0,0,[0,1],"fixture_active_pair");return m
def active_dormant(src,work,good=True):
    m=one_active(src,work,2.0,.5);e1=c.new_expert(1,-1,"dormant");set_weight(e1,2.0,.5 if good else -1.0)
    e1["dormant_since_prediction"]=0;e1["last_active_prediction"]=-1;m.experts[1]=e1;m.next_id=2
    e1["dormant_model_hash"]=m._model_hash(e1);e1["dormant_optimizer_hash"]=m._opt_hash(e1);e1["dormant_optimizer_step"]=c.opt_step(e1["optimizer"])
    return m
def full_pool(src,work,scale=20.0,arm="D_bounded"):
    m=Machine044(src,work,arm,True,False);e0=c.new_expert(0,-1,"active");set_weight(e0,scale,1);e0["first_active_t"]=0
    e1=c.new_expert(1,-1,"dormant");set_weight(e1,scale,-1);e2=c.new_expert(2,-1,"dormant");set_weight(e2,scale,-1)
    m.experts={0:e0,1:e1,2:e2};m.active_ids=[0];m.next_id=3
    for e in (e1,e2):
        e["dormant_since_prediction"]=0;e["last_active_prediction"]=-1;e["dormant_model_hash"]=m._model_hash(e);e["dormant_optimizer_hash"]=m._opt_hash(e);e["dormant_optimizer_step"]=c.opt_step(e["optimizer"])
    m.tracker.reset(0,0,[0],"fixture_full_pool");return m
def shadow_machine(src,work):
    m=one_active(src,work,2.0,.5);s=m._new_shadow_object(1,-1);m.shadow=s;m.next_id=2;m.attempts_after_e0=1;return m
def case_spec(case):
    if case=="birth_after":return {"kind":"birth","n":500,"stop":"post_first_birth_t"}
    if case=="single_live_after":return {"kind":"single","n":96,"stop":"live_update_t"}
    if case=="joint_live_after":return {"kind":"pair","n":96,"stop":"live_update_t"}
    if case in ("shadow_update01_after","shadow_update15_after","shadow_update16_after","qualification_ready_before_decision"):
        stop={"shadow_update01_after":"shadow_01_t","shadow_update15_after":"shadow_15_t","shadow_update16_after":"shadow_16_t","qualification_ready_before_decision":"qualification_ready_before_decision"}[case]
        return {"kind":"shadow","n":400,"stop":stop}
    if case=="sleep_after":return {"kind":"harmful_pair","n":240,"stop":"post_sleep_t"}
    if case=="reuse_pending":return {"kind":"reuse_good","n":300,"stop":"reuse_start_t"}
    if case=="reuse_accept_after":return {"kind":"reuse_good","n":300,"stop":"reuse_accept_t"}
    if case=="reuse_reject_after":return {"kind":"full_constant","n":340,"stop":"reuse_reject_t"}
    if case=="window_expiry_after":return {"kind":"single_constant","n":180,"stop":"window_expiry_after"}
    if case=="late_old_epoch_settlement":return {"kind":"harmful_pair","n":240,"stop":"late_old_epoch_settlement"}
    if case in ("shadow_accept_after","reclaim_create_after"):
        return {"kind":"gc_accept","n":820,"stop":"shadow_accept_t" if case=="shadow_accept_after" else "reclaim_create_t"}
    if case in ("shadow_reject_after","reclaimed_candidate_reject_after"):
        return {"kind":"gc_reject","n":820,"stop":"shadow_reject_t"}
    if case=="terminal_first_before_second":return {"kind":"single_short","n":80,"stop":"terminal_first_settle"}
    if case=="terminal_second_before_finalize":return {"kind":"single_short","n":80,"stop":"terminal_second_settle"}
    raise KeyError(case)
def build(case,work):
    q=case_spec(case);k=q["kind"];n=q["n"]
    if k=="birth":src=lifecycle_source(n,20,256);m=Machine044(src,work,"D_bounded",True,False)
    elif k=="single":src=lifecycle_source(n,2,None);m=one_active(src,work,2,1)
    elif k=="pair":src=lifecycle_source(n,2,None);m=active_pair(src,work,False)
    elif k=="shadow":src=lifecycle_source(n,2,None);m=shadow_machine(src,work)
    elif k=="harmful_pair":src=lifecycle_source(n,2,None);m=active_pair(src,work,True)
    elif k=="reuse_good":src=lifecycle_source(n,2,None);m=active_dormant(src,work,True)
    elif k=="full_constant":src=lifecycle_source(n,20,None);m=full_pool(src,work,20)
    elif k=="single_constant":src=lifecycle_source(n,2,None);m=one_active(src,work,2,1)
    elif k=="gc_accept":src=lifecycle_source(n,20,256);m=full_pool(src,work,20)
    elif k=="gc_reject":src=lifecycle_source(n,.01,256);m=full_pool(src,work,.01)
    elif k=="single_short":src=lifecycle_source(n,2,None);m=one_active(src,work,2,1)
    else:raise KeyError(k)
    m.checkpoint_dir=Path(work)/"checkpoints";return src,m,q
def run_to_end(m):
    m.engineering_stop_prefix=None;m.advance();m.terminal_settle();return semantic_digest(m)
def cmd_cont(a):
    runtime();root=Path(a.root)/a.case/"continuous";shutil.rmtree(root,ignore_errors=True);src,m,q=build(a.case,root);d=run_to_end(m)
    W(Path(a.root)/a.case/"continuous.json",{"case":a.case,"digest":d,"state":semantic_state(m),"endpoint":"done"})
def cmd_prepare(a):
    runtime();root=Path(a.root)/a.case/"resume";shutil.rmtree(root,ignore_errors=True);src,m,q=build(a.case,root);m.engineering_stop_prefix=q["stop"]
    stopped=None
    try:
        m.advance();m.terminal_settle()
    except EngineeringCheckpointStop as e:stopped=str(e)
    if stopped is None:raise RuntimeError("target checkpoint not reached "+a.case)
    cp=root/"checkpoints/latest.pt"
    if not cp.is_file():raise RuntimeError("checkpoint absent")
    W(Path(a.root)/a.case/"prepare.json",{"case":a.case,"target":q["stop"],"stopped":stopped,"checkpoint":str(cp),"checkpoint_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),
      "optimizer_calls_at_checkpoint":m.actual_optimizer_calls,"online_payload_bytes":m.online_payload_bytes()})
def cmd_resume(a):
    runtime();case_dir=Path(a.root)/a.case;root=case_dir/"resume";q=case_spec(a.case)
    # recreate only deterministic source; all online state comes from checkpoint.
    src,_dummy,_=build(a.case,case_dir/"source_recreate_tmp");shutil.rmtree(case_dir/"source_recreate_tmp",ignore_errors=True)
    cp=root/"checkpoints/latest.pt";m=Machine044.restore(src,root,cp,arm="D_bounded",allow_gradient=True,real_science=False,strict_journal=True);m.checkpoint_dir=root/"checkpoints"
    d=run_to_end(m);ref=J(case_dir/"continuous.json")["digest"];ok=d==ref
    W(case_dir/"result.json",{"case":a.case,"pass":ok,"continuous_digest":ref,"resumed_digest":d,"fresh_process":True,
      "checkpoint_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),"optimizer_calls_final":m.actual_optimizer_calls,
      "tracker":m.tracker.online_payload(),"audit_state":m.audit_state})
    if not ok:raise RuntimeError("resume mismatch "+a.case)
def cmd_collect(a):
    rows=[]
    for case in CASES:
        p=Path(a.root)/case/"result.json"
        if not p.exists():rows.append({"case":case,"pass":False,"reason":"missing"})
        else:rows.append(J(p))
    out={"protocol":"044","revision":1,"required_cases":list(CASES),"cases":rows,"all_pass":bool(len(rows)==19 and all(r.get("pass") is True for r in rows))}
    W(a.out,out);print(json.dumps({"all_pass":out["all_pass"],"count":len(rows)},indent=2))
    if not out["all_pass"]:raise SystemExit(2)
def cmd_memory(a):
    import psutil
    rows=[]
    for n in (8192,32768,131072):
        tr=c.UtilityTracker();tr.reset(0,0,[0],"stress");pressure=deque(maxlen=320);train=deque(maxlen=66);qual=deque(maxlen=32);reuse=deque(maxlen=2)
        rng=np.random.default_rng(4411)
        for i in range(n):
            y=((np.arange(c.HOSTS)+i)%3==0);live=(rng.normal(0,1,c.HOSTS)).astype(np.float32);d=(rng.normal(0,.01,c.HOSTS)).astype(np.float32)
            tr.settle(i,0,live,{0:d},y);pressure.append(float(np.mean(np.abs(live))));train.append((i,float(live.mean())))
            if i%64<32:qual.append((i,float(d.mean())))
            if i and i%256==0:reuse.append({"decision_t":i,"score":float(d.mean())})
        payload={"tracker":tr.online_payload(),"pressure":list(pressure),"train":list(train),"qual":list(qual),"reuse":list(reuse)}
        b=len(pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL));rss=psutil.Process().memory_info().rss
        rows.append({"length":n,"online_payload_bytes":b,"rss_bytes":int(rss),"ring":max(len(x) for x in tr.rows.values()),"pressure":len(pressure),"train":len(train),"qual":len(qual),"reuse":len(reuse)})
    spread=max(x["online_payload_bytes"] for x in rows)-min(x["online_payload_bytes"] for x in rows)
    out={"protocol":"044","revision":1,"seed":4411,"lengths":rows,"payload_spread_bytes":spread,"limit":65536,"no_preloaded_full_sequence":True,"pass":spread<=65536}
    W(a.out,out);print(json.dumps(out,indent=2))
    if not out["pass"]:raise SystemExit(2)
def cmd_functional(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    live=np.ones((129,c.HOSTS),np.float32);delta=np.full_like(live,np.float32(1e-7));y=np.zeros_like(live,dtype=np.int64);y[:,::2]=1
    live[128]=-3.25;delta[128]=1.75;y[128]=1-y[128]
    tr=c.UtilityTracker();tr.reset(0,0,[0],"golden")
    for i in range(128):tr.settle(i,0,live[i],{0:delta[i]},y[i])
    x=tr.score(0,127);z=c.independent_score_from_arrays(live,delta,y,np.arange(128))
    fields=("score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta")
    exact=all(abs(float(x[k])-float(z[k]))<=1e-10 for k in fields)
    tr.settle(128,0,live[128],{0:delta[128]},y[128]);expiry=tr.score(0,128).get("first_i")==1
    rows.append({"id":"incremental_float64_window","pass":bool(exact and expiry and tr.periodic_rebuild_count>=1),"online":x,"independent":z,"periodic_rebuilds":tr.periodic_rebuild_count})
    src=lifecycle_source(240,2,None);useful=active_pair(src,out/"useful",False);useful.advance();harm=active_pair(src,out/"harm",True);harm.advance()
    us=[r for r in useful.lifecycle_events if r.get("event")=="sleep"];hs=[r for r in harm.lifecycle_events if r.get("event")=="sleep"]
    rows.append({"id":"sleep_protection","pass":bool(not us and hs),"harmful_sleeps":list(hs)})
    src2=lifecycle_source(300,2,None);r=active_dormant(src2,out/"reuse",True);before=(r._model_hash(r.experts[1]),r._opt_hash(r.experts[1]),c.opt_step(r.experts[1]["optimizer"]));r.advance()
    acts=[q for q in r.lifecycle_events if q.get("event")=="activate" and q.get("reason")=="reuse_accept"];same=bool(acts and (acts[0]["model_hash"],acts[0]["optimizer_hash"],acts[0]["optimizer_step"])==before)
    rows.append({"id":"same_id_adam_reuse","pass":same,"activation":None if not acts else acts[0]})
    src3=lifecycle_source(820,20,256);bn=full_pool(src3,out/"gc_n","D_no_gc");bb=full_pool(src3,out/"gc_b","D_bounded");bn.advance();bb.advance()
    gc=[q for q in bb.reclamation_events if q.get("event")=="permanent_reclaim_and_shadow_start"];blocks=[q for q in bn.reclamation_events if q.get("event")=="capacity_blocked_no_gc"]
    rows.append({"id":"paired_gc_control","pass":bool(gc and blocks and int(gc[0]["at_interval"])==int(blocks[0]["at_interval"])),"gc":list(gc),"block":list(blocks)[:2]})
    # crash ambiguity: actual optimizer already executed while journal remains pending.
    src4=lifecycle_source(96,2,None);cr=active_pair(src4,out/"crash",False);cr.checkpoint_dir=out/"crash_cp";cr.save_checkpoint("safe",True);safe=out/"crash_cp/latest.pt";cr.crash_probe="after_first_joint_step";blocked=False
    try:cr.advance(16)
    except RuntimeError:pass
    try:Machine044.restore(src4,out/"crash",safe,arm="D_bounded",strict_journal=True)
    except RuntimeError as e:blocked="ambiguous_state" in str(e)
    rows.append({"id":"crash_first_joint_step","pass":blocked})
    result={"protocol":"044","revision":1,"fixtures":rows,"all_pass":all(x["pass"] for x in rows),"real_model_forwards":0,"real_gradient_steps":0}
    W(out/"functional.json",result);print(json.dumps({"all_pass":result["all_pass"],"fixtures":len(rows)},indent=2))
    if not result["all_pass"]:raise SystemExit(2)
def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    for name,fn in (("continuous-case",cmd_cont),("prepare-case",cmd_prepare),("resume-case",cmd_resume)):
        p=sp.add_parser(name);p.add_argument("--case",choices=CASES,required=True);p.add_argument("--root",required=True);p.set_defaults(fn=fn)
    p=sp.add_parser("collect-cases");p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.set_defaults(fn=cmd_collect)
    p=sp.add_parser("memory-stress");p.add_argument("--out",required=True);p.set_defaults(fn=cmd_memory)
    p=sp.add_parser("functional");p.add_argument("--out",required=True);p.set_defaults(fn=cmd_functional)
    a=ap.parse_args();a.fn(a)
if __name__=="__main__":main()
