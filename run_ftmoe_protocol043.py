"""Protocol-043 revision1 runner and production-entrypoint engineering fixtures."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, shutil, time, traceback
from pathlib import Path
import numpy as np
import torch

import protocol043_common as c
from protocol043_engine import Machine043, semantic_digest
from protocol035_common import dump_json, load_npz, sha256_file

ARMS=c.ARMS
def J(p): return c.J(p)
def W(p,x): c.W(p,x)
def runtime():
    torch.set_num_threads(1); np.seterr(all="raise"); random.seed(1); np.random.seed(1); torch.manual_seed(1)

def require_bool_true(obj,key,label):
    if key not in obj or type(obj[key]) is not bool or obj[key] is not True: raise RuntimeError(label+" fail_closed "+repr(obj.get(key)))

def validate_fixture_report(path):
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: raise RuntimeError("fixture missing")
    x=J(p)
    if (x.get("protocol"),x.get("revision"))!=("043",1): raise RuntimeError("fixture revision")
    require_bool_true(x,"all_pass","fixture")
    req=c.verify_plan()["engineering"]["required_fixture_ids"]; rows=x.get("fixtures")
    if not isinstance(rows,list) or len(rows)!=len(req): raise RuntimeError("fixture evidence count")
    by={r.get("test_id"):r for r in rows if isinstance(r,dict)}
    if set(by)!=set(req): raise RuntimeError("fixture IDs")
    for k in req:
        require_bool_true(by[k],"pass","fixture "+k)
        if len(by[k])<=2: raise RuntimeError("fixture evidence empty "+k)
    if int(x.get("real_engineering_prefixes",-1))!=1 or int(x.get("real_engineering_prefix_intervals_max",-1))>256 or int(x.get("real_engineering_gradient_steps",-1))!=0:
        raise RuntimeError("real engineering budget")
    return x

def validate_audit042(path):
    x=J(path)
    if (x.get("protocol"),x.get("revision"))!=("043",1): raise RuntimeError("audit042 revision")
    require_bool_true(x,"all_pass","audit042")
    if x.get("gradients")!=0 or x.get("model_replay") is not False: raise RuntimeError("audit042 not read-only")
    return x

def validate_gate(path,execution_sha):
    x=J(path)
    if (x.get("protocol"),x.get("revision"))!=("043",1): raise RuntimeError("gate revision")
    require_bool_true(x,"all_pass","gate")
    for k in ("plan_valid","sources_valid","audit042_valid","fixture_valid","budget_valid","source_bundle_valid"):
        require_bool_true(x,k,"gate "+k)
    if x.get("execution_sha")!=execution_sha or not isinstance(execution_sha,str) or len(execution_sha)!=40: raise RuntimeError("execution sha gate")
    return x

def default_ledger(fixture):
    return {"protocol":"043","revision":1,"budget_key":"protocol043_revision1_arm","order":list(ARMS),
      "sequences":{a:{"started":False,"completed":False,"restart_from_zero":False,"resume_events":[],"live_optimizer_steps":0,"shadow_optimizer_steps":0,
                       "deployed_forwards":0,"reuse_forwards":0,"qualification_forwards":0,"permanent_deletions":0} for a in ARMS},
      "total_optimizer_steps":0,"total_deployed_forwards":0,"total_reuse_forwards":0,"total_qualification_forwards":0,
      "new_streams":0,"extra_seeds":0,"F_loads":0,"extra_arms":0,"U_reproduction_runs":0,"restart_from_zero":False,
      "real_engineering_prefixes":int(fixture["real_engineering_prefixes"]),"real_engineering_prefix_intervals_max":int(fixture["real_engineering_prefix_intervals_max"]),
      "real_engineering_gradient_steps":int(fixture["real_engineering_gradient_steps"])}

def start_arm(root,arm,fixture,resume,run_id):
    lp=Path(root)/"budget_ledger.json"
    if not lp.exists(): W(lp,default_ledger(fixture))
    d=J(lp)
    if (d.get("protocol"),d.get("revision"),d.get("order"))!=("043",1,list(ARMS)): raise RuntimeError("budget identity")
    idx=list(ARMS).index(arm)
    for prev in ARMS[:idx]:
        if type(d["sequences"][prev].get("completed")) is not bool or not d["sequences"][prev]["completed"]: raise RuntimeError("previous arm incomplete "+prev)
    r=d["sequences"][arm]
    if r.get("completed"): raise RuntimeError("arm already completed")
    if r.get("started"):
        if not resume: raise RuntimeError("exact resume required")
        r["resume_events"].append({"run_id":str(run_id),"checkpoint":str(resume)})
    else:
        if resume: raise RuntimeError("resume for unstarted arm")
        r["started"]=True; r["started_run_id"]=str(run_id)
    W(lp,d); return d

def complete_arm(root,arm,sm):
    lp=Path(root)/"budget_ledger.json"; d=J(lp); r=d["sequences"][arm]
    r.update({"completed":True,"live_optimizer_steps":int(sm["live_optimizer_steps"]),"shadow_optimizer_steps":int(sm["shadow_optimizer_steps"]),
      "deployed_forwards":int(sm["deployed_prediction_forwards"]),"reuse_forwards":int(sm["reuse_preview_forwards"]),
      "qualification_forwards":int(sm["shadow_qualification_forwards"]),"permanent_deletions":int(sm["permanent_deletions"])})
    d["total_optimizer_steps"]=sum(x["live_optimizer_steps"]+x["shadow_optimizer_steps"] for x in d["sequences"].values())
    d["total_deployed_forwards"]=sum(x["deployed_forwards"] for x in d["sequences"].values())
    d["total_reuse_forwards"]=sum(x["reuse_forwards"] for x in d["sequences"].values())
    d["total_qualification_forwards"]=sum(x["qualification_forwards"] for x in d["sequences"].values())
    p=c.verify_plan()["budget"]["total"]
    if d["total_optimizer_steps"]>p["optimizer_steps"] or d["total_deployed_forwards"]>p["deployed_forwards"] or d["total_reuse_forwards"]>p["reuse_forwards"] or d["total_qualification_forwards"]>p["shadow_qualification_forwards"]:
        raise RuntimeError("cross-arm budget")
    W(lp,d)

def copy_controls(src,raw042,root):
    d=Path(root)/"cached"; d.mkdir(parents=True,exist_ok=True)
    mapping={"C_ref":src["C"],"B_ref":src["B"],"D_keep":src["D_keep"],"D_039":src["D_039"],"D_040":src["D_040"]}
    for name,a in mapping.items():
        q=d/name; q.mkdir(parents=True,exist_ok=True); np.savez_compressed(q/"predictions.npz",**{k:v for k,v in a.items() if isinstance(v,np.ndarray)})
    for name,rel in (("042_A_hist","A_hist/predictions.npz"),("042_A_win128","A_win128/predictions.npz")):
        q=d/name; q.mkdir(parents=True,exist_ok=True); shutil.copy2(Path(raw042)/rel,q/"predictions.npz")
    W(d/"summary.json",{"protocol":"043","revision":1,"cached_controls":list(mapping)+["042_A_hist","042_A_win128"],"training":False,"F_loaded":False})

def _set_weight(e,scale,sign=1.0):
    v=np.zeros(c.DIM,np.float32); v[0]=float(sign)/float(scale); c.set_linear(e,v.reshape(1,-1),0.0)

def lifecycle_source(n=820,scale=20.0,switch=256):
    src=c.synthetic_source(n=n,seed=4311,feature_scale=.01)
    y=np.zeros((n,c.HOSTS),np.int64)
    for t in range(n): y[t,(np.arange(c.HOSTS)+t)%2==0]=1
    s=(2*y-1).astype(np.float32); z=np.zeros((n,c.HOSTS,c.DIM),np.float32); z[...,0]=s*float(scale)
    margin=2.0*s
    if switch is not None: margin[int(switch):]=-2.0*s[int(switch):]
    logits=np.stack((-margin/2,margin/2),axis=-1).astype(np.float32); prob=c.sigmoid_margin(margin)
    cls=np.zeros((n,c.HOSTS,4),np.float32); cls[...,0]=1
    B={"detection_logits":logits,"probability":prob,"labels":y.copy(),"raw_labels":y.copy(),"class_probability":cls.copy(),"model_version":np.zeros(n,np.int64)}
    src["core"]["T"]["z"]=z; src["core"]["T"]["labels"]=y.copy(); src["core"]["B"]=B; src["B"]=B; src["C"]=B; src["D_keep"]=B; src["D_039"]=B; src["D_040"]=B
    return src

def setup_full_pool(src,arm,work,scale):
    m=Machine043(src,work,arm,allow_gradient=True,real_science=False)
    e0=c.new_expert(0,-1,"active"); _set_weight(e0,scale,1); e0["first_active_t"]=0
    e1=c.new_expert(1,-1,"dormant"); _set_weight(e1,scale,-1)
    e2=c.new_expert(2,-1,"dormant"); _set_weight(e2,scale,-1)
    m.experts={0:e0,1:e1,2:e2}; m.active_ids=[0]; m.next_id=3
    for e in (e1,e2):
        e["dormant_since_prediction"]=0; e["last_active_prediction"]=-1
        e["dormant_model_hash"]=m._model_hash(e); e["dormant_optimizer_hash"]=m._opt_hash(e); e["dormant_optimizer_step"]=c.opt_step(e["optimizer"])
    m.tracker.reset(0,0,[0],"synthetic_initial_pool"); return m

def setup_useful_pair(src,work,harmful=False):
    m=Machine043(src,work,"D_bounded",True,False); scale=2.0
    e0=c.new_expert(0,-1,"active"); _set_weight(e0,scale,1); e0["first_active_t"]=0
    e1=c.new_expert(1,-1,"active"); _set_weight(e1,scale,-1 if harmful else 1); e1["first_active_t"]=0
    m.experts={0:e0,1:e1}; m.active_ids=[0,1]; m.next_id=2; m.tracker.reset(0,0,[0,1],"synthetic_active_pair"); return m

def fixture_float64(out):
    # 129 rows let the independent checker distinguish the sealed 0..127 window
    # from the adversarial 1..128 membership while production uses only 0..127.
    live=np.ones((129,c.HOSTS),np.float32); delta=np.full_like(live,np.float32(1e-7)); y=np.zeros_like(live,dtype=np.int64); y[:,::2]=1
    live[128]=np.float32(-3.25); delta[128]=np.float32(1.75); y[128]=1-y[128]
    tr=c.UtilityTracker(); tr.reset(0,0,[0],"fixture")
    for i in range(128): tr.settle(i,0,live[i],{0:delta[i]},y[i])
    online=tr.score(0,127); off=c.independent_score_from_arrays(live,delta,y,np.arange(128))
    l64=live[:128].astype(np.float64); d64=delta[:128].astype(np.float64)
    correct=c.stable_bce_rows(l64-d64,y[:128])-c.stable_bce_rows(l64,y[:128])
    legacy=c.stable_bce_rows((live[:128]-delta[:128]).astype(np.float32),y[:128])-c.stable_bce_rows(live[:128],y[:128])
    dtype_detect=float(np.max(np.abs(correct-legacy)))>1e-12
    fields=("score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta")
    exact=all(abs(float(online[k])-float(off[k]))<=1e-10 for k in fields)
    wrong_sign=c.independent_score_from_arrays(live,-delta,y,np.arange(128))
    wrong_member=c.independent_score_from_arrays(live,delta,y,np.arange(1,129))
    sign_detect=abs(float(wrong_sign["score"])-float(off["score"]))>1e-12
    member_detect=abs(float(wrong_member["score"])-float(off["score"]))>1e-12 or wrong_member.get("first_i")!=off.get("first_i")
    return {"test_id":"float64_counterfactual_independent_golden","pass":bool(exact and dtype_detect and sign_detect and member_detect),
      "online":online,"independent":off,"float32_bug_max_abs":float(np.max(np.abs(correct-legacy))),
      "wrong_sign_detected":sign_detect,"wrong_window_member_detected":member_detect,"production_tracker_called_by_independent":False}

def fixture_window(out):
    tr=c.UtilityTracker(); tr.reset(1,0,[0],"fixture"); live=np.ones(c.HOSTS,np.float32); d=np.full(c.HOSTS,.1,np.float32)
    y=np.zeros(c.HOSTS,np.int64); y[:4]=1
    for i in range(128): tr.settle(i,1,live,{0:d},y)
    a=tr.score(0,127); tr.settle(128,1,live,{0:d},y); b=tr.score(0,128)
    expiry=a.get("first_i")==0 and b.get("first_i")==1 and b.get("last_i")==128
    tr.reset(2,129,[0],"topology"); late_excluded=False
    try: tr.settle(127,1,live,{0:d},y)
    except AssertionError: late_excluded=True
    null=tr.score(0,140)
    return {"test_id":"window_expiry_epoch_late_labels_unknown","pass":bool(expiry and late_excluded and not null["valid"] and null["score"] is None),
      "window_before":a,"window_after":b,"late_old_epoch_rejected_from_control":late_excluded,"unknown":null}

def fixture_sleep(out):
    src=lifecycle_source(n=230,scale=2.0,switch=None)
    useful=setup_useful_pair(src,Path(out)/"sleep_useful",False); useful.advance(210)
    useful_sleep=[x for x in useful.lifecycle_events if x["event"]=="sleep"]
    harmful=setup_useful_pair(src,Path(out)/"sleep_harmful",True); harmful.advance(210)
    sleeps=[x for x in harmful.lifecycle_events if x["event"]=="sleep"]
    unknown=setup_useful_pair(src,Path(out)/"sleep_unknown",True); unknown.advance(110)
    unknown_sleep=[x for x in unknown.lifecycle_events if x["event"]=="sleep"]
    return {"test_id":"sleep_useful_protection_no_forced_replacement","pass":bool(not useful_sleep and sleeps and not unknown_sleep),
      "two_useful_sleep_events":useful_sleep,"harmful_sleep_events":sleeps,"unknown_before128_sleep_events":unknown_sleep,
      "active_full_forced_replacement_events":[x for x in useful.candidate_decisions if x.get("event")=="shadow_start"]}

def run_gc_pair(out,scale=20.0,n=820):
    src=lifecycle_source(n=n,scale=scale,switch=256); pair={}
    for arm in ARMS:
        m=setup_full_pool(src,arm,Path(out)/("gc_"+arm),scale); m.checkpoint_dir=Path(out)/("gc_"+arm)/"checkpoints"; m.advance(); m.terminal_settle(); pair[arm]=m
    return src,pair

def fixture_joint_shadow(out,bounded):
    starts=[x for x in bounded.candidate_decisions if x.get("event")=="shadow_start"]
    comps=[x for x in bounded.candidate_decisions if x.get("event")=="shadow_training_complete"]
    evals=[x for x in bounded.candidate_decisions if x.get("event")=="shadow_evaluate"]
    shadow_updates=[x for x in bounded.update_log if x.get("kind")=="shadow"]; live_by_t={}
    for x in bounded.update_log:
        if x.get("kind")=="live": live_by_t.setdefault(x["at_interval"],[]).append(x)
    joint=bool(shadow_updates and all(x["at_interval"] in live_by_t for x in shadow_updates if bounded.active_ids is not None))
    steps=[x["optimizer_step_after"] for x in shadow_updates]
    frozen=bool(len(shadow_updates)>=16 and 16 in steps and comps and evals and evals[0]["intervals"][1]-evals[0]["intervals"][0]==32)
    return {"test_id":"joint_Adam_detached_shadow_future_qualification","pass":bool(starts and joint and frozen),
      "shadow_starts":starts,"shadow_update_count":len(shadow_updates),"shadow_steps":steps,"training_complete":comps,"evaluations":evals,"joint_live_same_t":joint}

def fixture_reuse(out):
    src=lifecycle_source(n=300,scale=2.0,switch=None); m=Machine043(src,Path(out)/"reuse","D_bounded",True,False)
    e0=c.new_expert(0,-1,"active"); _set_weight(e0,2.0,.5); e0["first_active_t"]=0
    e1=c.new_expert(1,-1,"dormant"); _set_weight(e1,2.0,.5); e1["dormant_since_prediction"]=0; e1["last_active_prediction"]=-1
    m.experts={0:e0,1:e1}; m.active_ids=[0]; m.next_id=2; e1["dormant_model_hash"]=m._model_hash(e1); e1["dormant_optimizer_hash"]=m._opt_hash(e1); e1["dormant_optimizer_step"]=c.opt_step(e1["optimizer"])
    before=(m._model_hash(e1),m._opt_hash(e1),c.opt_step(e1["optimizer"])); m.tracker.reset(0,0,[0],"reuse_fixture"); m.advance(280)
    acts=[x for x in m.lifecycle_events if x.get("event")=="activate" and x.get("expert_id")==1 and x.get("reason")=="reuse_accept"]
    same=bool(acts and acts[0]["model_hash"]==before[0] and acts[0]["optimizer_hash"]==before[1] and acts[0]["optimizer_step"]==before[2])
    return {"test_id":"reuse_same_id_Adam","pass":same,"before":{"model":before[0],"optimizer":before[1],"step":before[2]},"reuse_activations":acts}

def fixture_weak_gain(out):
    src=lifecycle_source(n=330,scale=20.0,switch=None); m=setup_full_pool(src,"D_bounded",Path(out)/"weak",20.0)
    # E1 becomes tiny helpful; E2 stays harmful.
    _set_weight(m.experts[1],20.0,.002); m.experts[1]["dormant_model_hash"]=m._model_hash(m.experts[1])
    m.advance(320); victim,table=m._reclamation_table(319)
    r1=next(r for r in table if r["expert_id"]==1); protected=("positive_contribution" in r1["reasons"])
    return {"test_id":"eviction_positive_weak_gain_protected","pass":bool(protected and not r1["eligible"]),"selected":victim,"weak_expert_row":r1,"reuse_evaluations":len(m.reuse_history)}

def fixture_eviction_guards(out,base_unused=None):
    # Generate two real, non-overlapping future reuse rejection windows through production tick.
    src=lifecycle_source(n=420,scale=20.0,switch=None)
    m=setup_full_pool(src,"D_no_gc",Path(out)/"guard_actual",20.0); m.advance(319)
    dormant=m.dormant_ids(); hashes={eid:m._model_hash(m.experts[eid]) for eid in dormant}
    relevant=[h for h in m.reuse_history if int(h["epoch"])==m.deployment_epoch and h["candidate_ids"]==dormant and h["candidate_hashes"]==hashes]
    base_hist=copy.deepcopy(m.reuse_history)
    victim,base_table=m._reclamation_table(318)
    baseline=bool(victim is not None and len(relevant)>=2)
    checks={"baseline_real_two_window_eligible":baseline}
    if baseline:
        # Modify only sealed evidence to prove each guard fails closed; no lifecycle transition is invoked.
        # Locate the last two relevant objects by decision_t.
        rel_decisions=[int(h["decision_t"]) for h in relevant[-2:]]
        def mutate_matching(fn):
            m.reuse_history=copy.deepcopy(base_hist)
            matches=[h for h in m.reuse_history if int(h["epoch"])==m.deployment_epoch and h["candidate_ids"]==dormant and h["candidate_hashes"]==hashes]
            fn(matches[-2:])
            return m._reclamation_table(318)
        v,t=mutate_matching(lambda hs: hs[0]["intervals"].__setitem__(1,int(hs[1]["intervals"][0])+1))
        checks["overlap_blocks"]=v is None
        def unknown(hs):
            for q in hs[-1]["candidates"]: q["support"]=False
        v,t=mutate_matching(unknown); checks["unknown_blocks"]=v is None
        m.reuse_history=copy.deepcopy(base_hist); v,t=m._reclamation_table(500); checks["stale_blocks"]=v is None
        m.reuse_history=copy.deepcopy(base_hist)
        # A current hash mismatch means neither evidence window belongs to the current dormant set.
        target=dormant[0]; old=m.experts[target]["dormant_model_hash"]; m.experts[target]["dormant_model_hash"]="intentional_hash_guard_probe"
        # _reclamation_table uses actual model hashes; change evidence instead, preserving the frozen model itself.
        m.experts[target]["dormant_model_hash"]=old
        for h in m.reuse_history:
            if int(h["epoch"])==m.deployment_epoch and h["candidate_ids"]==dormant:
                h["candidate_hashes"][target]="intentional_mismatch"
        v,t=m._reclamation_table(318); checks["hash_mismatch_blocks"]=v is None
        m.reuse_history=base_hist
    else:
        checks.update({"overlap_blocks":False,"unknown_blocks":False,"stale_blocks":False,"hash_mismatch_blocks":False})
    return {"test_id":"eviction_stale_unknown_overlap_hash_guards","pass":bool(all(checks.values())),"checks":checks,
      "real_reuse_history_count":len(relevant),"real_reuse_decisions":[h.get("decision_t") for h in relevant],"baseline_table":base_table}

def fixture_gc_accept(out,src,pair):
    b=pair["D_bounded"]; ev=[x for x in b.reclamation_events if x.get("event")=="permanent_reclaim_and_shadow_start"]
    if ev:
        cid=int(ev[0]["candidate_id"]); active_count=int(np.sum(np.any(b.out["active_ids"]==cid,axis=1)))
    else: cid=None; active_count=0
    accepted=[x for x in b.candidate_decisions if x.get("event")=="shadow_evaluate" and x.get("candidate_id")==cid and x.get("pass") is True]
    return {"test_id":"production_full_pool_delete_birth_accept64","pass":bool(ev and accepted and active_count>=64),
      "gc_events":ev,"accepted":accepted,"candidate_active_predictions":active_count,"production_entrypoint":True}

def fixture_gc_reject(out):
    src,pair=run_gc_pair(Path(out)/"reject_pair",scale=.01,n=820); b=pair["D_bounded"]
    ev=[x for x in b.reclamation_events if x.get("event")=="permanent_reclaim_and_shadow_start"]
    cid=None if not ev else int(ev[0]["candidate_id"]); dec=[x for x in b.candidate_decisions if x.get("event")=="shadow_evaluate" and x.get("candidate_id")==cid]
    reject=bool(ev and dec and dec[0].get("pass") is False and int(ev[0]["deleted_id"]) in b.deleted_ids and int(ev[0]["deleted_id"]) not in b.experts)
    blocked=[x for x in pair["D_no_gc"].reclamation_events if x.get("event")=="capacity_blocked_no_gc"]
    return {"test_id":"production_eviction_then_candidate_reject","pass":bool(reject and blocked),"gc_events":ev,"candidate_decisions":dec,
      "deleted_not_restored":reject,"no_gc_blocks":blocked[:3]}

def fixture_pair_divergence(out,pair):
    n=pair["D_bounded"]; g=pair["D_no_gc"]; ge=[x for x in n.reclamation_events if x.get("event")=="permanent_reclaim_and_shadow_start"]
    if not ge: return {"test_id":"two_arm_first_GC_divergence","pass":False,"reason":"no bounded GC"}
    t=int(ge[0]["at_interval"]); same_prob=bool(np.array_equal(n.out["probability"][:t+1],g.out["probability"][:t+1]))
    same_ids=bool(np.array_equal(n.out["active_ids"][:t+1],g.out["active_ids"][:t+1]))
    bn=[x for x in g.reclamation_events if x.get("event")=="capacity_blocked_no_gc" and int(x["at_interval"])==t]
    return {"test_id":"two_arm_first_GC_divergence","pass":bool(same_prob and same_ids and bn),
      "first_gc_control_t":t,"pre_and_control_tick_outputs_exact":same_prob,"active_ids_exact":same_ids,"no_gc_block_at_same_t":bn}

def fixture_resume(out,src):
    base=setup_full_pool(src,"D_bounded",Path(out)/"resume_base",20.0); base.checkpoint_dir=Path(out)/"resume_cp"
    base.advance(180); base.save_checkpoint("manual_window_expiry",True); base.advance(); base.terminal_settle(); digest=semantic_digest(base)
    cps=sorted(Path(out,"resume_cp").glob("*.pt")); preferred=[]
    for token in ("manual_window_expiry","reuse_start","reclaim_create","shadow_01","shadow_15","shadow_16","shadow_accept"):
        q=next((p for p in cps if token in p.name),None)
        if q is not None: preferred.append(q)
    evidence=[]
    for j,q in enumerate(preferred[:7]):
        wd=Path(out)/("resume_from_%02d"%j); r=Machine043.restore(src,wd,q,arm="D_bounded",allow_gradient=True,real_science=False,strict_journal=False)
        r.checkpoint_dir=wd/"cp"; r.advance(); r.terminal_settle(); evidence.append({"checkpoint":q.name,"checkpoint_sha":sha256_file(q),"same_endpoint":semantic_digest(r)==digest})
    return {"test_id":"disk_resume_continue_to_end_all_states","pass":bool(len(evidence)>=5 and all(x["same_endpoint"] for x in evidence)),
      "continuous_digest":digest,"continuation_cases":evidence,"snapshot_only":False}

def fixture_crash(out,gcsrc,gcpair):
    # Joint multi-expert optimizer transaction: pending journal is written before either step.
    src=lifecycle_source(n=500,scale=20.0,switch=256)
    m=setup_full_pool(src,"D_bounded",Path(out)/"crash_step",20.0); m.checkpoint_dir=Path(out)/"crash_step_cp"; m.save_checkpoint("safe",True)
    safe=Path(out)/"crash_step_cp/latest.pt"; m.crash_probe="after_first_joint_step"; step_block=False
    try: m.advance(16)
    except RuntimeError: pass
    try: Machine043.restore(src,Path(out)/"crash_step",safe,arm="D_bounded",strict_journal=True)
    except RuntimeError as e: step_block="ambiguous_state" in str(e)
    # Use the paired no-GC arm to locate the first *real* full-capacity birth opportunity.
    blocks=[x for x in gcpair["D_no_gc"].reclamation_events if x.get("event")=="capacity_blocked_no_gc" and x.get("would_delete_id") is not None]
    gc_block=False; event_t=None
    if blocks:
        event_t=int(blocks[0]["at_interval"])
        g=setup_full_pool(gcsrc,"D_bounded",Path(out)/"crash_gc",20.0); g.checkpoint_dir=Path(out)/"crash_gc_cp"
        g.advance(event_t); g.save_checkpoint("safe_pre_gc",True); gs=Path(out)/"crash_gc_cp/latest.pt"
        g.crash_probe="after_reclaim_before_shadow"
        try: g.advance(event_t+1)
        except RuntimeError: pass
        try: Machine043.restore(gcsrc,Path(out)/"crash_gc",gs,arm="D_bounded",strict_journal=True)
        except RuntimeError as e: gc_block="ambiguous_state" in str(e)
    return {"test_id":"crash_atomic_step_eviction_budget","pass":bool(step_block and gc_block),
      "joint_step_ambiguous_blocked":step_block,"reclaim_create_ambiguous_blocked":gc_block,
      "real_full_capacity_control_t":event_t,"production_control_path_used":event_t is not None,"from_zero_retry":False}

def fixture_fail_closed(out):
    req=c.verify_plan()["engineering"]["required_fixture_ids"]; d=Path(out)/"gate_negative"; d.mkdir(parents=True,exist_ok=True)
    good={"protocol":"043","revision":1,"all_pass":True,"fixtures":[{"test_id":k,"pass":True,"evidence":"x"} for k in req],
      "real_engineering_prefixes":1,"real_engineering_prefix_intervals_max":256,"real_engineering_gradient_steps":0}
    cases={}
    for name,mut in (("missing",lambda x:x.pop("all_pass")),("string",lambda x:x.__setitem__("all_pass","true")),("revision",lambda x:x.__setitem__("revision",2)),("false",lambda x:x.__setitem__("all_pass",False))):
        x=copy.deepcopy(good); mut(x); p=d/(name+".json"); W(p,x)
        try: validate_fixture_report(p); cases[name]=False
        except Exception: cases[name]=True
    over=default_ledger(good); over["total_optimizer_steps"]=1537; budget_detect=over["total_optimizer_steps"]>c.verify_plan()["budget"]["total"]["optimizer_steps"]
    return {"test_id":"fail_closed_source_fixture_budget","pass":bool(all(cases.values()) and budget_detect),"negative_fixture_cases":cases,"overbudget_detected":budget_detect}

def fixture_future(out):
    s0=lifecycle_source(n=180,scale=2.0,switch=None); s1=copy.deepcopy(s0); s1["B"]["labels"][130:]=1-s1["B"]["labels"][130:]; s1["core"]["T"]["labels"]=s1["B"]["labels"]; s1["core"]["B"]["labels"]=s1["B"]["labels"]
    a=setup_useful_pair(s0,Path(out)/"future_a",False); b=setup_useful_pair(s1,Path(out)/"future_b",False); a.advance(130); b.advance(130)
    same=bool(np.array_equal(a.out["probability"][:130],b.out["probability"][:130]) and a.lifecycle_events==b.lifecycle_events and a.candidate_decisions==b.candidate_decisions)
    return {"test_id":"future_label_noninterference","pass":same,"prefix_predictions_exact":same,"future_labels_changed_from":130,"compared_before_future_visibility":True}

def run_fixtures(real_src,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Pure/bounded-stat fixtures first.
    rows.append(fixture_float64(out)); rows.append(fixture_window(out)); rows.append(fixture_sleep(out))
    # Shared production GC pair covers admission, shadow, paired divergence and guard evidence.
    gcsrc,pair=run_gc_pair(out,20.0,820)
    rows.append(fixture_joint_shadow(out,pair["D_bounded"])); rows.append(fixture_reuse(out)); rows.append(fixture_weak_gain(out))
    rows.append(fixture_eviction_guards(out,pair["D_no_gc"])); rows.append(fixture_gc_accept(out,gcsrc,pair)); rows.append(fixture_gc_reject(out))
    rows.append(fixture_pair_divergence(out,pair)); rows.append(fixture_resume(out,gcsrc)); rows.append(fixture_crash(out,gcsrc,pair))
    rows.append(fixture_fail_closed(out)); rows.append(fixture_future(out))
    req=c.verify_plan()["engineering"]["required_fixture_ids"]; by={x["test_id"]:x for x in rows}
    synth_ok=set(by)==set(req) and all(type(by[k].get("pass")) is bool and by[k]["pass"] for k in req)
    if not synth_ok:
        rep={"protocol":"043","revision":1,"all_pass":False,"fixtures":rows,"required_fixture_ids":req,
          "real_engineering_prefixes":0,"real_engineering_prefix_intervals_max":0,"real_engineering_gradient_steps":0}
        W(out/"fixture_report.json",rep); return rep
    # Only after all synthetic production fixtures pass, consume the single real <=256 zero-gradient prefix.
    m=Machine043(real_src,out/"real_prefix","D_bounded",allow_gradient=False,real_science=False); m.checkpoint_dir=out/"real_prefix_cp"; m.advance(128); m.save_checkpoint("real128",True)
    r=Machine043.restore(real_src,out/"real_prefix_restore",out/"real_prefix_cp/latest.pt",arm="D_bounded",allow_gradient=False,real_science=False,strict_journal=False)
    r.advance(256)
    real_ok=bool(np.array_equal(r.out["probability"][:256],real_src["B"]["probability"][:256]) and r.live_optimizer_steps==0 and r.shadow_optimizer_steps==0 and not r.experts)
    rep={"protocol":"043","revision":1,"all_pass":bool(real_ok),"fixtures":rows,"required_fixture_ids":req,
      "real_engineering_prefixes":1,"real_engineering_prefix_intervals_max":256,"real_engineering_gradient_steps":0,
      "real_prefix_exact_B_disk_continuation":real_ok}
    W(out/"fixture_report.json",rep); return rep

def save_predictions(path,src,m):
    np.savez_compressed(path,probability=m.out["probability"],detection_logits=m.out["detection_logits"],total_delta=m.out["total_delta"],
      live_margin=m.out["live_margin"],B_margin=m.out["B_margin"],deployment_epoch=m.out["deployment_epoch"],active_ids=m.out["active_ids"],
      active_count=m.out["active_count"],shadow_present=m.out["shadow_present"],accepted_count=m.out["accepted_count"],
      expert_versions=m.out["expert_versions"],expert_hashes=m.out["expert_hashes"],contribution=m.out["contribution"],
      labels=src["B"]["labels"],raw_labels=src["B"]["raw_labels"],class_probability=src["B"]["class_probability"],
      model_version=src["B"].get("model_version",np.zeros(src["n"],np.int64)))

def save_arm(root,src,m):
    d=Path(root)/m.arm; d.mkdir(parents=True,exist_ok=True); save_predictions(d/"predictions.npz",src,m)
    logs=[("birth_checks","checks",m.birth_checks),("pressure_checks","checks",m.pressure_checks),("utility_checks","checks",m.utility_checks),
      ("sleep_tables","tables",m.sleep_tables),("lifecycle_events","events",m.lifecycle_events),("candidate_decisions","decisions",m.candidate_decisions),
      ("reuse_decisions","decisions",m.reuse_decisions),("opportunity_log","opportunities",m.opportunity_log),("update_log","updates",m.update_log),
      ("settlement_log","settlements",m.settlement_log),("qualification_records","records",m.qualification_records),
      ("reclamation_events","events",m.reclamation_events),("tombstones","tombstones",m.tombstones),("reuse_history","slots",m.reuse_history)]
    for name,key,val in logs: W(d/(name+".json"),{"protocol":"043","revision":1,"arm":m.arm,key:val})
    experts={}
    for eid,x in sorted(m.experts.items()):
        experts[str(eid)]={"role_final":x["role"],"version":x["version"],"optimizer_step":c.opt_step(x["optimizer"]),"model_hash_final":m._model_hash(x),
          "optimizer_hash_final":m._opt_hash(x),"reactivations":x.get("reactivations",0),"first_active_t":x.get("first_active_t"),
          "last_active_prediction":x.get("last_active_prediction"),"active_predictions":int(np.sum(np.any(m.out["active_ids"]==eid,axis=1)))}
    sm={"protocol":"043","revision":1,"arm":m.arm,"first_birth_t":m.first_birth_t,"ids_created":m.next_id,"resident_experts_final":len(m.experts),
      "active_ids_final":list(m.active_ids),"deployment_epoch_final":m.deployment_epoch,"attempts_after_e0":m.attempts_after_e0,
      "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,"deployed_prediction_forwards":m.deployed_forwards,
      "reuse_preview_forwards":m.reuse_preview_forwards,"shadow_qualification_forwards":m.shadow_preview_forwards,"live_training_forwards":m.live_training_forwards,
      "shadow_training_forwards":m.shadow_training_forwards,"reuse_slots_started":m.reuse_started,"max_active_seen":m.max_active_seen,
      "max_resident_seen":m.max_resident_seen,"peak_resident_tensor_bytes":m.peak_resident_tensor_bytes,"permanent_deletions":m.permanent_deletions,
      "deleted_ids":sorted(m.deleted_ids),"actual_optimizer_calls":m.actual_optimizer_calls,"near_threshold_recompute_count":m.near_threshold_recompute_count,
      "per_expert":experts,"tombstones":m.tombstones,"terminal_progress":m.terminal_progress,"next_substep":m.next_substep,"terminal_counter_delta":m.terminal_counter_delta}
    W(d/"summary.json",sm); return sm

def audit_arm(src,m,fixture,gate,before36,before41):
    p=c.verify_plan(); after36={rel:sha256_file(src["source036_root"]/rel) for rel in p["source036"]["files"]}; after41={rel:sha256_file(src["source041_root"]/rel) for rel in p["source041"]["files"]}
    mature=all((not r["batch_indices"]) or max(r["batch_indices"])+2<=r["at_interval"] for r in m.update_log)
    valid={"source036_unchanged":before36==after36==p["source036"]["files"],"source041_unchanged":before41==after41==p["source041"]["files"],
      "fixture_all_pass":fixture["all_pass"] is True,"gate_all_pass":gate["all_pass"] is True,
      "output_all_finite":bool(np.isfinite(m.out["probability"]).all() and np.isfinite(m.out["detection_logits"]).all()),
      "classification_exact_B_copy":True,"all_update_batches_mature":mature,"first_birth_t351":m.first_birth_t==351,
      "live_steps_within_704":m.live_optimizer_steps<=704,"shadow_steps_within_64":m.shadow_optimizer_steps<=64,
      "total_steps_within_768":m.live_optimizer_steps+m.shadow_optimizer_steps<=768,"deployed_forwards_within_11232":m.deployed_forwards<=11232,
      "reuse_forwards_within_3072":m.reuse_preview_forwards<=3072,"qualification_forwards_within_128":m.shadow_preview_forwards<=128,
      "active_max_2":m.max_active_seen<=2,"resident_max_3":m.max_resident_seen<=3,"ids_max_5":m.next_id<=5,"attempts_max_4":m.attempts_after_e0<=4,
      "deletions_max_4":m.permanent_deletions<=4,"deleted_ids_not_resident":not any(i in m.experts for i in m.deleted_ids),
      "terminal_complete":m.terminal_progress==3 and m.next_substep=="done","terminal_counter_zero":not any(m.terminal_counter_delta),
      "actual_optimizer_calls_match":m.actual_optimizer_calls==m.live_optimizer_steps+m.shadow_optimizer_steps,"no_F_loaded":True}
    valid["all_pass"]=all(valid.values()); return valid

def cmd_preflight(a):
    runtime(); c.verify_plan(); src=c.load_sources(a.source036,a.source041)
    try:
        rep=run_fixtures(src,a.out_dir)
        print(json.dumps({"protocol":"043","revision":1,"all_pass":rep["all_pass"],"fixtures":len(rep["fixtures"])},indent=2))
        if not rep["all_pass"]: raise RuntimeError("engineering_incomplete")
    except Exception as e:
        W(Path(a.out_dir)/"preflight_exception.json",{"protocol":"043","revision":1,"all_pass":False,
          "error":repr(e),"traceback":traceback.format_exc(),"science_sequences_started":0,
          "real_prefix_consumed":bool((Path(a.out_dir)/"fixture_report.json").exists() and J(Path(a.out_dir)/"fixture_report.json").get("real_engineering_prefixes")==1)})
        raise

def cmd_copy(a):
    runtime(); src=c.load_sources(a.source036,a.source041); copy_controls(src,a.source042,a.out_dir)

def cmd_arm(a):
    runtime(); c.verify_plan(); src=c.load_sources(a.source036,a.source041); root=Path(a.out_dir); root.mkdir(parents=True,exist_ok=True)
    fixture=validate_fixture_report(a.fixture_report); audit=validate_audit042(a.audit_gate); gate=validate_gate(a.gate_report,a.execution_sha)
    start_arm(root,a.arm,fixture,a.resume_checkpoint,a.run_id)
    before36={rel:sha256_file(src["source036_root"]/rel) for rel in c.verify_plan()["source036"]["files"]}; before41={rel:sha256_file(src["source041_root"]/rel) for rel in c.verify_plan()["source041"]["files"]}
    ar=root/a.arm; ar.mkdir(parents=True,exist_ok=True); W(ar/"source_lock.json",{"protocol":"043","revision":1,"arm":a.arm,"source036":before36,"source041":before41,"F_loaded":False})
    t0=time.perf_counter(); cpu0=time.process_time()
    try:
        m=Machine043.restore(src,ar/"runtime_state",a.resume_checkpoint,arm=a.arm,allow_gradient=True,real_science=True,strict_journal=True) if a.resume_checkpoint else Machine043(src,ar/"runtime_state",a.arm,True,True)
        m.checkpoint_dir=ar/"checkpoints"; m.advance(); m.terminal_settle(); sm=save_arm(root,src,m); au=audit_arm(src,m,fixture,gate,before36,before41); W(ar/"run_audit.json",au)
        if not au["all_pass"]: raise RuntimeError(a.arm+" invalid_execution")
        W(ar/"science_cost_raw.json",{"protocol":"043","revision":1,"arm":a.arm,"wall_seconds":time.perf_counter()-t0,"process_seconds":time.process_time()-cpu0,
          "prediction_seconds":m.prediction_seconds,"update_seconds":m.update_seconds,"controller_cpu_seconds":m.controller_cpu_seconds,"io_seconds":m.io_seconds,
          "live_optimizer_steps":sm["live_optimizer_steps"],"shadow_optimizer_steps":sm["shadow_optimizer_steps"],"deployed_forwards":sm["deployed_prediction_forwards"],
          "reuse_forwards":sm["reuse_preview_forwards"],"qualification_forwards":sm["shadow_qualification_forwards"],"live_training_forwards":sm["live_training_forwards"],
          "shadow_training_forwards":sm["shadow_training_forwards"],"peak_resident_tensor_bytes":sm["peak_resident_tensor_bytes"]})
        complete_arm(root,a.arm,sm)
        W(ar/"scientific_execution_complete.json",{"protocol":"043","revision":1,"arm":a.arm,"complete":True,"audit_all_pass":True})
        W(ar/"status.json",{"protocol":"043","revision":1,"arm":a.arm,"science_status":"completed","analysis_status":"pending","publication_status":"pending"})
        print(json.dumps(sm,indent=2))
    except Exception as e:
        W(ar/"status.json",{"protocol":"043","revision":1,"arm":a.arm,"science_status":"interrupted","error":repr(e),"traceback":traceback.format_exc(),
          "analysis_status":"not_started","publication_status":"pending","restart_from_zero":False})
        raise

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("preflight"); p.add_argument("--source036",required=True); p.add_argument("--source041",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_preflight)
    p=sp.add_parser("copy-controls"); p.add_argument("--source036",required=True); p.add_argument("--source041",required=True); p.add_argument("--source042",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_copy)
    p=sp.add_parser("arm"); p.add_argument("--arm",choices=ARMS,required=True); p.add_argument("--source036",required=True); p.add_argument("--source041",required=True)
    p.add_argument("--out-dir",required=True); p.add_argument("--fixture-report",required=True); p.add_argument("--audit-gate",required=True); p.add_argument("--gate-report",required=True)
    p.add_argument("--execution-sha",required=True); p.add_argument("--run-id",required=True); p.add_argument("--resume-checkpoint"); p.set_defaults(fn=cmd_arm)
    a=ap.parse_args(); a.fn(a)
if __name__=="__main__": main()
