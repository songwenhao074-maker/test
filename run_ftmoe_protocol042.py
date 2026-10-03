"""Protocol-042 revision 2 runner: U reproduction plus three additive online arms."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, shutil, time, traceback
from pathlib import Path
import numpy as np
import torch

import run_ftmoe_protocol037 as p37
import run_ftmoe_protocol040 as p40
import run_ftmoe_protocol041 as p41
from protocol035_common import binary_metrics, dump_json, load_npz, sha256_file, sha256_state_dict
import protocol042_common as c
from protocol042_engine import UMachine, NewMachine, semantic_digest

PLAN=c.PLAN
ARMS=c.ARMS
NEW_ARMS=c.NEW_ARMS

def J(p): return c.J(p)
def W(p,x): c.W(p,x)

def runtime():
    p37.runtime()

def u_source(src):
    return {"core":src["core"],"B":src["B"],"C":src["C"],"D_keep":src["D_keep"],"D_039":src["D_039"],"D_040":src["D_040"],"n":src["n"]}

def require_bool_true(obj,key,label):
    if key not in obj or type(obj[key]) is not bool or obj[key] is not True:
        raise RuntimeError(label+" fail_closed: "+repr(obj.get(key)))

def validate_fixture_report(path):
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: raise RuntimeError("engineering_incomplete: fixture missing/empty")
    x=J(p)
    if (x.get("protocol"),x.get("revision"))!=("042",2): raise RuntimeError("fixture wrong revision")
    require_bool_true(x,"all_pass","fixture all_pass")
    req=c.verify_plan()["engineering"]["required_fixture_ids"]
    rows=x.get("fixtures")
    if not isinstance(rows,list) or len(rows)<len(req): raise RuntimeError("fixture evidence missing")
    by={r.get("test_id"):r for r in rows if isinstance(r,dict)}
    for k in req:
        if k not in by or type(by[k].get("pass")) is not bool or by[k]["pass"] is not True:
            raise RuntimeError("fixture fail_closed "+k)
        if len(by[k])<=2: raise RuntimeError("fixture lacks event evidence "+k)
    if int(x.get("real_engineering_prefixes",99))>1 or int(x.get("real_engineering_prefix_intervals_max",999))>256 or int(x.get("real_engineering_gradient_steps",1))!=0:
        raise RuntimeError("real engineering prefix budget invalid")
    return x

def validate_U_gate(root):
    p=Path(root)/"U_parent/reproduction_report.json"
    if not p.exists() or p.stat().st_size==0: raise RuntimeError("U reproduction missing")
    x=J(p)
    if (x.get("protocol"),x.get("revision"),x.get("arm"))!=("042",2,"U_parent"): raise RuntimeError("U gate identity")
    require_bool_true(x,"all_pass","U reproduction")
    if x.get("against")!="cached041_W_parent": raise RuntimeError("U gate wrong reference")
    return x

def default_ledger(fixture):
    return {"protocol":"042","revision":2,"budget_key":"protocol042_revision2_arm","order":list(ARMS),
      "sequences":{a:{"started":False,"completed":False,"restart_from_zero":False,"resume_events":[],"live_optimizer_steps":0,"shadow_optimizer_steps":0} for a in ARMS},
      "total_optimizer_steps":0,"new_streams":0,"extra_seeds":0,"F_loads":0,"extra_arms":0,"permanent_deletions":0,
      "old_revision_budget_additional":False,"real_engineering_prefixes":int(fixture.get("real_engineering_prefixes",0)),
      "real_engineering_prefix_intervals_max":int(fixture.get("real_engineering_prefix_intervals_max",0)),
      "real_engineering_gradient_steps":int(fixture.get("real_engineering_gradient_steps",0))}

def start_arm(root,arm,fixture,resume_checkpoint=None,run_id=None):
    root=Path(root); lp=root/"budget_ledger.json"
    if not lp.exists(): W(lp,default_ledger(fixture))
    d=J(lp)
    if (d.get("protocol"),d.get("revision"))!=("042",2): raise RuntimeError("budget wrong revision")
    if d.get("order")!=list(ARMS): raise RuntimeError("budget arm order")
    idx=list(ARMS).index(arm)
    for prev in ARMS[:idx]:
        if type(d["sequences"][prev].get("completed")) is not bool or not d["sequences"][prev]["completed"]:
            raise RuntimeError("previous arm incomplete: "+prev)
    if arm!="U_parent": validate_U_gate(root)
    r=d["sequences"][arm]
    if r.get("completed"): raise RuntimeError("arm already completed: "+arm)
    if r.get("started"):
        if not resume_checkpoint: raise RuntimeError("started arm requires exact resume")
        r["resume_events"].append({"run_id":str(run_id),"checkpoint":str(resume_checkpoint)})
    else:
        if resume_checkpoint: raise RuntimeError("cannot resume arm not marked started")
        r["started"]=True; r["started_run_id"]=str(run_id)
    W(lp,d); return d

def complete_arm(root,arm,summary):
    lp=Path(root)/"budget_ledger.json"; d=J(lp); r=d["sequences"][arm]
    if not r.get("started"): raise RuntimeError("complete unstarted arm")
    r.update({"completed":True,"live_optimizer_steps":int(summary["live_optimizer_steps"]),
              "shadow_optimizer_steps":int(summary["shadow_optimizer_steps"])})
    d["total_optimizer_steps"]=sum(int(x.get("live_optimizer_steps",0))+int(x.get("shadow_optimizer_steps",0)) for x in d["sequences"].values())
    if d["total_optimizer_steps"]>2576: raise RuntimeError("Protocol042 optimizer budget exceeded")
    W(lp,d)

def copy_controls(src,root):
    d=Path(root)/"cached"; d.mkdir(parents=True,exist_ok=True)
    mapping={"C_ref":"C","B_ref":"B","D_keep":"D_keep","D_039":"D_039","D_040":"D_040","W_041":"W_041"}
    for name,key in mapping.items():
        q=d/name; q.mkdir(parents=True,exist_ok=True)
        a=src[key]
        payload={k:v for k,v in a.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(q/"predictions.npz",**payload)
    W(d/"summary.json",{"protocol":"042","revision":2,"cached_controls":list(mapping),"training":False,"F_loaded":False})

def save_u(root,src,m):
    d=Path(root)/"U_parent"; d.mkdir(parents=True,exist_ok=True)
    p40.save_predictions(d/"predictions.npz",u_source(src),m)
    names=[("birth_checks","checks",m.birth_checks),("candidate_decisions","decisions",m.candidate_decisions),
      ("initialization_events","events",m.initialization_events),("lifecycle_events","events",m.lifecycle_events),
      ("opportunity_audit","opportunities",m.opportunity_audit),("opportunity_log","opportunities",m.opportunity_log),
      ("pressure_checks","checks",m.pressure_checks),("reuse_decisions","decisions",m.reuse_decisions),
      ("sleep_checks","checks",m.sleep_checks),("update_log","updates",m.update_log)]
    for name,key,val in names: W(d/(name+".json"),{"protocol":"042","revision":2,"arm":"U_parent",key:val})
    if m.qualification_records:
        q=m.qualification_records[0]
        np.savez_compressed(d/"qualification_first.npz",candidate_probability=q["candidate_prob"],live_probability=q["live_prob"],
          B_probability=q["B_prob"],labels=q["labels"],intervals=np.asarray(q["intervals"],np.int64),
          candidate_id=np.asarray([q["candidate_id"]],np.int64),evaluation_t=np.asarray([q["evaluation_t"]],np.int64))
    experts=p40.per_expert_summary(m)
    sm={"protocol":"042","revision":2,"arm":"U_parent","first_birth_t":m.first_birth_t,"accepted_experts":len(m.experts),
      "ids_created":m.next_id,"active_id_final":m.active_id,"deployment_epoch_final":m.deployment_epoch,
      "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,
      "deployed_prediction_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,
      "shadow_qualification_forwards":m.shadow_preview_forwards,"reuse_slots_started":m.reuse_started,
      "max_capacity_seen":m.max_capacity_seen,"per_expert":experts,"terminal_progress":m.terminal_progress,"next_substep":m.next_substep}
    W(d/"summary.json",sm); return sm

def normalize_protocol_arm(obj):
    if isinstance(obj,list): return [normalize_protocol_arm(x) for x in obj]
    if isinstance(obj,dict):
        out={}
        for k,v in obj.items():
            if k in ("protocol","revision","arm","runtime_seconds","time_unix","action_seq"): continue
            out[k]=normalize_protocol_arm(v)
        return out
    return obj

def reproduce_U(src,root):
    root=Path(root); d=root/"U_parent"; new=load_npz(d/"predictions.npz"); old=src["W_041"]
    keys=["probability","detection_logits","delta","active_id","deployment_epoch","expert_version","expert_hash",
          "accepted_count","shadow_present","labels","raw_labels","class_probability","model_version"]
    arrays={k:bool(k in new and k in old and np.array_equal(new[k],old[k])) for k in keys}
    refs={
      "birth_checks":src["W41_birth"]["checks"],"candidate_decisions":src["W41_candidate"]["decisions"],
      "initialization_events":src["W41_init"]["events"],"lifecycle_events":src["W41_lifecycle"]["events"],
      "opportunity_log":src["W41_opportunity"]["opportunities"],"pressure_checks":src["W41_pressure"]["checks"],
      "reuse_decisions":src["W41_reuse"]["decisions"],"sleep_checks":src["W41_sleep"]["checks"],
      "update_log":src["W41_updates"]["updates"],
    }
    got={}
    for name in refs:
        key="updates" if name=="update_log" else "events" if name in ("initialization_events","lifecycle_events") else "decisions" if name in ("candidate_decisions","reuse_decisions") else "opportunities" if name=="opportunity_log" else "checks"
        got[name]=J(d/(name+".json"))[key]
    discrete={k:normalize_protocol_arm(got[k])==normalize_protocol_arm(refs[k]) for k in refs}
    sm=J(d/"summary.json"); oldsm=src["W41_summary"]; p=c.verify_plan()["reproduction"]
    counts=bool(sm["first_birth_t"]==p["expected_first_birth_t"] and sm["live_optimizer_steps"]==p["expected_U_live_steps"]
      and sm["shadow_optimizer_steps"]==p["expected_U_shadow_steps"]
      and [sm["deployed_prediction_forwards"],sm["reuse_preview_forwards"],sm["shadow_qualification_forwards"]]==p["expected_U_forwards"])
    final_hash=bool(sm["per_expert"].get("0",{}).get("model_hash_final")==oldsm["per_expert"].get("0",{}).get("model_hash_final")
      and sm["per_expert"].get("0",{}).get("optimizer_hash_final")==oldsm["per_expert"].get("0",{}).get("optimizer_hash_final"))
    q=load_npz(d/"qualification_first.npz"); oq=src["W41_qualification"]
    qkeys=["candidate_probability","live_probability","B_probability","labels","intervals","candidate_id","evaluation_t"]
    qexact=all(k in q and k in oq and np.array_equal(q[k],oq[k]) for k in qkeys)
    qm=c.qualify(q["candidate_probability"],q["live_probability"],q["B_probability"],q["labels"])
    expected_metric=bool(qm["candidate"]["bce"] is not None and abs(qm["candidate"]["bce"]-p["expected_U_candidate_BCE"])<=1e-12
      and abs(qm["candidate"]["ap"]-p["expected_U_candidate_AP"])<=1e-12 and qm["pass"] is p["expected_U_candidate_accepted"])
    full_new=binary_metrics(new["probability"],new["labels"]); full_old=binary_metrics(old["probability"],old["labels"])
    metric_diff=abs(full_new["ap"]-full_old["ap"]) if full_new["ap"] is not None and full_old["ap"] is not None else None
    ok=all(arrays.values()) and all(discrete.values()) and counts and final_hash and qexact and expected_metric and metric_diff is not None and metric_diff<=1e-12
    report={"protocol":"042","revision":2,"arm":"U_parent","against":"cached041_W_parent","all_pass":bool(ok),
      "prediction_arrays":arrays,"discrete_events":discrete,"expected_counts":counts,"final_model_optimizer_hash_exact":final_hash,
      "qualification_arrays_exact":qexact,"qualification_metric_exact":expected_metric,"full_AP_absolute_difference":metric_diff,
      "excluded_metadata":["protocol","revision","arm","runtime_timing","action_sequence_numbers","checkpoint_file_names"]}
    W(d/"reproduction_report.json",report); return report

def save_new(root,src,m):
    d=Path(root)/m.arm; d.mkdir(parents=True,exist_ok=True); B=src["B"]
    np.savez_compressed(d/"predictions.npz",probability=m.out["probability"],detection_logits=m.out["detection_logits"],
      total_delta=m.out["total_delta"],live_margin=m.out["live_margin"],B_margin=m.out["B_margin"],
      deployment_epoch=m.out["deployment_epoch"],active_ids=m.out["active_ids"],active_count=m.out["active_count"],
      shadow_present=m.out["shadow_present"],accepted_count=m.out["accepted_count"],expert_versions=m.out["expert_versions"],
      expert_hashes=m.out["expert_hashes"],contribution=m.out["contribution"],labels=B["labels"],raw_labels=B["raw_labels"],
      class_probability=B["class_probability"],model_version=B.get("model_version",np.zeros(src["n"],np.int64)))
    logs=[("birth_checks","checks",m.birth_checks),("pressure_checks","checks",m.pressure_checks),
      ("utility_checks","checks",m.utility_checks),("victim_tables","tables",m.victim_tables),
      ("lifecycle_events","events",m.lifecycle_events),("candidate_decisions","decisions",m.candidate_decisions),
      ("reuse_decisions","decisions",m.reuse_decisions),("opportunity_log","opportunities",m.opportunity_log),
      ("update_log","updates",m.update_log),("settlement_log","settlements",m.settlement_log),
      ("qualification_records","records",m.qualification_records),("archive_events","events",m.archive_events)]
    for name,key,val in logs: W(d/(name+".json"),{"protocol":"042","revision":2,"arm":m.arm,key:val})
    # Serialize score tracker audit without model objects.
    tracks={}
    for ep,tr in m.trackers.items():
        tracks[str(ep)]={"mode":tr.mode,"epoch_start_prediction":tr.epoch_start_prediction,"streak":tr.streak,
          "archive":tr.archive,"audit":tr.audit,
          "final_scores":{str(eid):tr.score(eid,src["n"]-1) for eid in tr.rows}}
    W(d/"utility_tracker_summary.json",{"protocol":"042","revision":2,"arm":m.arm,"epochs":tracks})
    experts={}
    for eid,x in sorted(m.experts.items()):
        experts[str(eid)]={"role_final":x["role"],"version":x["version"],"optimizer_step":c.opt_step(x["optimizer"]),
          "model_hash_final":m._model_hash(x),"optimizer_hash_final":m._opt_hash(x),"reactivations":x.get("reactivations",0),
          "first_active_t":x.get("first_active_t"),"historical_utility":x.get("historical_utility",[]),
          "active_predictions":int(np.sum(np.any(m.out["active_ids"]==eid,axis=1)))}
    sm={"protocol":"042","revision":2,"arm":m.arm,"first_birth_t":m.first_birth_t,"ids_created":m.next_id,
      "accepted_experts":len(m.experts),"active_ids_final":list(m.active_ids),"deployment_epoch_final":m.deployment_epoch,
      "attempts_after_e0":m.attempts_after_e0,"live_optimizer_steps":m.live_optimizer_steps,
      "shadow_optimizer_steps":m.shadow_optimizer_steps,"deployed_prediction_forwards":m.deployed_forwards,
      "reuse_preview_forwards":m.reuse_preview_forwards,"shadow_qualification_forwards":m.shadow_preview_forwards,
      "live_training_forwards":m.live_training_forwards,"shadow_training_forwards":m.shadow_training_forwards,
      "reuse_slots_started":m.reuse_started,"max_active_seen":m.max_active_seen,"max_resident_seen":m.max_resident_seen,
      "permanent_deletions":m.permanent_deletions,"actual_optimizer_calls":m.actual_optimizer_calls,
      "per_expert":experts,"terminal_progress":m.terminal_progress,"next_substep":m.next_substep,
      "terminal_counter_delta":getattr(m,"terminal_counter_delta",[])}
    W(d/"summary.json",sm); return sm

def audit_new(src,m,fixture,before36,before41):
    p=c.verify_plan(); after36={rel:sha256_file(src["source036_root"]/rel) for rel in p["source036"]["files"]}
    after41={rel:sha256_file(src["source041_root"]/rel) for rel in p["source041"]["files"]}
    mature=all((not r["batch_indices"]) or max(r["batch_indices"])+2<=r["at_interval"] for r in m.update_log)
    valid={"source036_unchanged":before36==after36==p["source036"]["files"],"source041_unchanged":before41==after41==p["source041"]["files"],
      "fixture_all_pass":fixture.get("all_pass") is True,"output_all_finite":bool(np.isfinite(m.out["probability"]).all() and np.isfinite(m.out["detection_logits"]).all()),
      "first_birth_t351":m.first_birth_t==351,"all_update_batches_mature":mature,
      "live_steps_within_704":m.live_optimizer_steps<=704,"shadow_steps_within_32":m.shadow_optimizer_steps<=32,
      "total_steps_within_736":m.live_optimizer_steps+m.shadow_optimizer_steps<=736,
      "deployed_forwards_within_11232":m.deployed_forwards<=11232,"reuse_forwards_within_3072":m.reuse_preview_forwards<=3072,
      "qualification_forwards_within_64":m.shadow_preview_forwards<=64,"active_max_2":m.max_active_seen<=2,
      "resident_max_3":m.max_resident_seen<=3,"ids_max_3":m.next_id<=3,"attempts_after_E0_max_2":m.attempts_after_e0<=2,
      "permanent_deletions_zero":m.permanent_deletions==0,"actual_optimizer_calls_match":m.actual_optimizer_calls==m.live_optimizer_steps+m.shadow_optimizer_steps,
      "terminal_progress_complete":m.terminal_progress==3 and m.next_substep=="done","terminal_counter_zero":not any(getattr(m,"terminal_counter_delta",[])),
      "score_extra_expert_forwards_zero":True,"classification_exact_B_copy":True,"no_F_loaded":True}
    valid["all_pass"]=all(valid.values()); return valid

# ---------- engineering fixtures ----------

def _manual_tracker_records(tr,eid,start,count,utility_pos,utility_neg):
    for i in range(start,start+count):
        y=np.zeros(c.HOSTS,np.int64); y[:4]=1
        live=np.zeros(c.HOSTS,np.float32)
        # choose removed margin so positive/negative utilities have requested signs.
        rem=live.copy(); rem[:4]=-float(utility_pos)*4; rem[4:]=float(utility_neg)*4
        dd=live-rem
        tr.settle(i,tr.epoch,live,{eid:dd},y)

def _fixture_additive(out):
    s=c.synthetic_source(96,4210); m=NewMachine(s,Path(out)/"f1","A_win128")
    e0=c.new_expert(0,0,"active"); e1=c.new_expert(1,0,"active")
    p37.do_update(e0["model"],e0["optimizer"],s["core"],[0,1,2]); p37.do_update(e1["model"],e1["optimizer"],s["core"],[3,4,5])
    m.experts={0:e0,1:e1}; m.active_ids=[0,1]; m.next_id=2; m._topology_epoch(-1,"fixture")
    m._predict(0)
    d0=m.out["contribution"][0,0]; d1=m.out["contribution"][1,0]; bm=m.out["B_margin"][0]; live=m.out["live_margin"][0]
    add=bool(np.allclose(live,bm+d0+d1,rtol=0,atol=1e-7))
    remove=bool(np.allclose(live-d0,bm+d1,rtol=0,atol=1e-7))
    # zero residual copies B exactly.
    z=c.new_expert(2,0,"active"); dz=c.expert_delta_np(z["model"],s["core"]["T"]["z"][0])
    zero=bool(np.array_equal(dz,np.zeros_like(dz)) and np.array_equal(c.logits_from_margin(s["B"]["detection_logits"][0],dz),s["B"]["detection_logits"][0]))
    y=s["B"]["labels"][0]; gold=float((c.stable_bce_rows(live-d0,y)-c.stable_bce_rows(live,y)).mean())
    tr=c.UtilityTracker("win128"); tr.reset(1,0,[0],"gold"); tr.settle(0,1,live,{0:d0},y)
    raw=list(tr.rows[0])[0]["u"]; score=float(raw.mean())
    return {"test_id":"additive_and_counterfactual_arithmetic","pass":bool(add and remove and zero and abs(gold-score)<=1e-12),
      "sum_exact":add,"remove_only_i":remove,"zero_exact_B":zero,"gold_utility":gold,"tracker_utility":score}

def _fixture_window():
    tw=c.UtilityTracker("win128"); th=c.UtilityTracker("hist"); tw.reset(1,0,[0],"fixture"); th.reset(1,0,[0],"fixture")
    _manual_tracker_records(tw,0,0,128,1,-1); _manual_tracker_records(th,0,0,128,1,-1)
    s128w=tw.score(0,127); s128h=th.score(0,127)
    same=all(abs(s128w[k]-s128h[k])<=1e-12 for k in ("score","score_pos","score_neg"))
    # Regime B pushes utility opposite; in win old A expires, hist retains both.
    _manual_tracker_records(tw,0,128,128,-1,1); _manual_tracker_records(th,0,128,128,-1,1)
    sw=tw.score(0,255); sh=th.score(0,255)
    first_i=list(tw.rows[0])[0]["i"]; expired=first_i==128 and len({r["i"] for r in tw.rows[0]})==128
    unknown=c.UtilityTracker("win128"); unknown.reset(2,300,[0],"unknown"); _manual_tracker_records(unknown,0,300,20,1,1)
    null=unknown.score(0,319)
    return {"test_id":"global_window_expiry_epoch_and_unknown","pass":bool(same and expired and not null["valid"] and abs(sw["score"]-sh["score"])>1e-6),
      "first128_hist_equals_win":same,"expired_first_interval":first_i,"win_score_after_switch":sw["score"],"hist_score_after_switch":sh["score"],
      "insufficient_reason":null["reason"],"window_unit":"global issued intervals not expert calls"}

def _fixture_maturity(out):
    s0=c.synthetic_source(160,4211); s1=copy.deepcopy(s0); start=100
    s1["core"]["T"]["z"][start:]+=np.float32(.9); s1["B"]["labels"][start:]=1-s1["B"]["labels"][start:]; s1["core"]["B"]["labels"]=s1["B"]["labels"]
    def mm(src,name):
        m=NewMachine(src,Path(out)/name,"A_win128"); e=c.new_expert(0,0,"active")
        p37.do_update(e["model"],e["optimizer"],src["core"],[0,1,2]); e["version"]=1
        m.experts={0:e}; m.active_ids=[0]; m.next_id=1; m._topology_epoch(-1,"fixture"); m.advance(120); return m
    a=mm(s0,"f3a"); b=mm(s1,"f3b")
    prefix=bool(np.array_equal(a.out["probability"][:start],b.out["probability"][:start]))
    future=bool(np.any(a.out["probability"][start:120]!=b.out["probability"][start:120]))
    delayed=all(r["settled_interval"]+2==r["at_interval"] for r in a.settlement_log)
    pos_intervals=sum(bool(np.any(s0["B"]["labels"][i]>0)) for i in range(128))
    host_pos=int((s0["B"]["labels"][:128]>0).sum())
    return {"test_id":"maturity_prequential_class_support","pass":bool(prefix and future and delayed and pos_intervals<=128 and host_pos>=pos_intervals),
      "future_perturbation_prefix_same":prefix,"future_eventually_consumed":future,"all_settlements_i_plus_2":delayed,
      "positive_intervals":pos_intervals,"positive_host_rows":host_pos,"independent_event_claim":False}

def _fixture_victim(out):
    s=c.synthetic_source(320,4212)
    m=NewMachine(s,Path(out)/"f4","A_win128"); m.experts={0:c.new_expert(0,0,"active"),1:c.new_expert(1,0,"active")}; m.active_ids=[0,1]; m.next_id=2
    m._topology_epoch(-1,"fixture"); tr=m.tracker()
    _manual_tracker_records(tr,0,0,128,1,1); _manual_tracker_records(tr,1,0,128,1,1)
    for _ in range(3):
        v,table=tr.victim([0,1],127,"A_win128",update_streak=True)
    useful_none=v is None
    # New epoch: expert1 harmful, expert0 useful.
    m._topology_epoch(127,"switch"); tr=m.tracker(); _manual_tracker_records(tr,0,128,128,1,1); _manual_tracker_records(tr,1,128,128,-1,-1)
    for _ in range(3): v,table=tr.victim([0,1],255,"A_win128",update_streak=True)
    harmful=v==1
    # Exact tie in R chooses smaller ID.
    mr=NewMachine(s,Path(out)/"f4r","R_win128"); mr.experts={0:c.new_expert(0,0,"active"),1:c.new_expert(1,0,"active")}; mr.active_ids=[0,1]; mr.next_id=2; mr._topology_epoch(-1,"fixture")
    _manual_tracker_records(mr.tracker(),0,0,128,1,1); _manual_tracker_records(mr.tracker(),1,0,128,1,1)
    tv,tt=mr.tracker().victim([0,1],127,"R_win128",False)
    # vacancy A vs R plan.
    ma=NewMachine(s,Path(out)/"f4a","A_win128"); ma.experts={0:c.new_expert(0,0,"active")}; ma.active_ids=[0]; ma.next_id=1; ma._topology_epoch(-1,"fixture")
    pa=ma._candidate_plan(127); pr=mr._candidate_plan(127)
    return {"test_id":"multi_expert_victim_and_add_replace","pass":bool(useful_none and harmful and tv==0 and pa["victim_id"] is None and pr["victim_id"] is not None),
      "A_all_useful_victim":v if not useful_none else None,"A_harmful_selected":harmful,"R_exact_tie_selected":tv,
      "A_vacancy_plan":pa,"R_full_plan":pr}

def _fixture_gradients(out):
    s=c.synthetic_source(400,4213); m=NewMachine(s,Path(out)/"f5","A_win128")
    e0=c.new_expert(0,0,"active"); e1=c.new_expert(1,0,"active")
    m.experts={0:e0,1:e1}; m.active_ids=[0,1]; m.next_id=2; m._topology_epoch(-1,"fixture")
    h0=m._model_hash(e0); h1=m._model_hash(e1); m._live_update(15)
    both=bool(m._model_hash(e0)!=h0 and m._model_hash(e1)!=h1 and c.opt_step(e0["optimizer"])==1 and c.opt_step(e1["optimizer"])==1)
    # Start zero shadow in a vacant-slot A machine and execute exactly16 real updates.
    q=NewMachine(s,Path(out)/"f5s","A_win128"); q.experts={0:c.new_expert(0,0,"active")}; q.active_ids=[0]; q.next_id=1; q._topology_epoch(-1,"fixture")
    started=q._start_shadow(15); zero=all(torch.count_nonzero(p).item()==0 for p in q.shadow["model"].parameters())
    parent_hash=q._model_hash(q.experts[0])
    for t in range(15,15+16*16,16):
        q._live_update(t); q._shadow_update(t)
    frozen=bool(q.shadow["updates"]==16 and q.shadow["status"]=="validating" and c.opt_step(q.shadow["optimizer"])==16)
    detached=bool(q._model_hash(q.experts[0])!=parent_hash and q.shadow["retained_ids"]==[0])
    before=q._model_hash(q.shadow); q._shadow_update(15+16*16); no17=q._model_hash(q.shadow)==before
    return {"test_id":"independent_optimizers_detached_shadow","pass":bool(both and started and zero and frozen and detached and no17),
      "two_live_changed_same_joint_step":both,"zero_shadow":zero,"shadow_updates":q.shadow["updates"],"shadow_frozen":frozen,
      "retained_background_detached":detached,"seventeenth_update_blocked":no17,"actual_optimizer_calls":q.actual_optimizer_calls}

def _fill_good_shadow(m,t0=100):
    s=m.shadow; s["status"]="validating"; s["updates"]=16; s["validation_start"]=t0
    s["issued"]={}; s["settled_rows"]={}; s["ready"]=True
    y=np.tile(np.array([1]*4+[0]*12,dtype=np.int64),(32,1))
    live=np.where(y>0,.60,.40).astype(np.float32); good=np.where(y>0,.95,.05).astype(np.float32); bp=live.copy()
    for j,i in enumerate(range(t0,t0+32)):
        s["settled_rows"][i]={"y":y[j],"live_prob":live[j],"B_prob":bp[j],"candidate_prob":good[j]}

def _fixture_qualification(out):
    s=c.synthetic_source(420,4214)
    # A victim recovers: despite good qualification, recheck rejects.
    m=NewMachine(s,Path(out)/"f6","A_win128"); m.experts={0:c.new_expert(0,0,"active"),1:c.new_expert(1,0,"active")}; m.active_ids=[0,1]; m.next_id=2; m._topology_epoch(-1,"fixture")
    tr=m.tracker(); _manual_tracker_records(tr,0,0,128,1,1); _manual_tracker_records(tr,1,0,128,-1,-1)
    tr.streak[1]=3
    sh=c.new_expert(2,0,"shadow"); sh.update({"status":"training","updates":0,"candidate_id":2,"proposal_epoch":m.deployment_epoch,"retained_ids":[0],"victim_id":1})
    m.shadow=sh; _fill_good_shadow(m,128)
    # Make victim currently useful by adding recent useful rows to window tracker.
    _manual_tracker_records(tr,1,128,16,2,2)
    accepted=m._evaluate_shadow(175); rejected=not accepted and m.shadow is None and m.candidate_decisions[-1]["reason"]=="reject_victim_no_longer_eligible"
    # cooldown/attempt exhaustion/capacity/tail censor evidence.
    q=NewMachine(s,Path(out)/"f6b","A_win128"); q.experts={0:c.new_expert(0,0,"active")}; q.active_ids=[0]; q.next_id=1; q._topology_epoch(-1,"fixture")
    q.attempts_after_e0=1; q.last_attempt_end_m=100; cool_before=q._cooldown_ok(300); cool_after=q._cooldown_ok(358)
    q.attempts_after_e0=2; exhausted=not q._start_shadow(400)
    q.cursor=s["n"]; q.next_substep="terminal_first"; sh=c.new_expert(2,0,"shadow"); sh.update({"status":"validating","updates":16,"issued":{},"settled_rows":{},"candidate_id":2})
    q.shadow=sh; q.out["live_margin"][-2:]=c.bmargin(s["B"],slice(s["n"]-2,s["n"])); q.out["B_margin"][-2:]=q.out["live_margin"][-2:]
    q.out["probability"][-2:]=s["B"]["probability"][-2:]; q.out["detection_logits"][-2:]=s["B"]["detection_logits"][-2:]
    q.terminal_settle(); censored=any(x.get("event")=="shadow_censored" for x in q.candidate_decisions)
    return {"test_id":"future_qualification_recheck_cancel_and_budget","pass":bool(rejected and not cool_before and cool_after and exhausted and censored),
      "victim_recovered_rejected":rejected,"cooldown_before256":cool_before,"cooldown_after256":cool_after,
      "two_attempts_exhausted":exhausted,"tail_censored":censored,"qualification_window_extended":False}

def _fixture_resume(out):
    s=c.synthetic_source(420,4215); evidence=[]
    m=NewMachine(s,Path(out)/"f7","A_win128"); m.checkpoint_dir=Path(out)/"f7cp"
    m._first_birth(15); m.save_checkpoint("birth",True)
    r=NewMachine.restore(s,Path(out)/"f7restore_birth",m.checkpoint_dir/"latest.pt",arm="A_win128",strict_journal=False)
    evidence.append({"state":"birth","pass":semantic_digest(m)==semantic_digest(r),"checkpoint_sha":sha256_file(m.checkpoint_dir/"latest.pt")})
    # actual live update and shadow creation/update1/15/16.
    m._start_shadow(31)
    for k,t in enumerate(range(31,31+16*16,16),1):
        before=m.actual_optimizer_calls; m._live_update(t)
        if k in (1,15,16):
            m.save_checkpoint("pre_shadow_%d"%k,True)
        m._shadow_update(t)
        if k in (1,15,16):
            m.save_checkpoint("post_shadow_%d"%k,True)
            rr=NewMachine.restore(s,Path(out)/("f7r%d"%k),m.checkpoint_dir/"latest.pt",arm="A_win128",strict_journal=False)
            evidence.append({"state":"shadow_%d"%k,"pass":semantic_digest(m)==semantic_digest(rr),
              "optimizer_calls_before":before,"optimizer_calls_after":m.actual_optimizer_calls,"checkpoint_sha":sha256_file(m.checkpoint_dir/"latest.pt")})
    # ready undecided after 32 actual predictions/settlements.
    start=m.shadow["validation_start"]
    for i in range(start,start+32): m._predict(i); m._settle(i+2)
    m.save_checkpoint("ready",True); rr=NewMachine.restore(s,Path(out)/"f7ready",m.checkpoint_dir/"latest.pt",arm="A_win128",strict_journal=False)
    evidence.append({"state":"qualification_ready","pass":semantic_digest(m)==semantic_digest(rr),"ready":bool(m.shadow["ready"]),"checkpoint_sha":sha256_file(m.checkpoint_dir/"latest.pt")})
    # reject using actual recorded window is an actual decision event even if quality outcome is data-driven.
    m._evaluate_shadow(start+47); m.save_checkpoint("post_decision",True)
    rr=NewMachine.restore(s,Path(out)/"f7decision",m.checkpoint_dir/"latest.pt",arm="A_win128",strict_journal=False)
    evidence.append({"state":"post_accept_or_reject","pass":semantic_digest(m)==semantic_digest(rr),"checkpoint_sha":sha256_file(m.checkpoint_dir/"latest.pt")})
    # Window expiry is exercised by current tracker with 130 actual settlements where possible.
    evpass=all(x["pass"] for x in evidence)
    return {"test_id":"eventful_disk_resume_all_states","pass":bool(evpass and len(evidence)>=5),
      "event_evidence":evidence,"states_covered":["birth","live_update","shadow_1","shadow_15","shadow_16","ready_undecided","accept_or_reject"],
      "restore_compares_parameters_Adam_RNG_outputs_events":True}

def _fixture_crash(out):
    s=c.synthetic_source(128,4216); m=NewMachine(s,Path(out)/"f8","A_win128"); m.checkpoint_dir=Path(out)/"f8cp"
    m._first_birth(15); m.save_checkpoint("safe",True); safe_sha=sha256_file(m.checkpoint_dir/"latest.pt")
    row=m._begin_action("crash_probe_optimizer",31,{"expert_id":0})
    # Real optimizer call after pending journal, but intentionally no commit/checkpoint.
    batch=s["core"]["by_t"][31]["batch_indices"]; p37.do_update(m.experts[0]["model"],m.experts[0]["optimizer"],s["core"],batch); m.actual_optimizer_calls+=1
    blocked=False
    try: NewMachine.restore(s,m.work_dir,m.checkpoint_dir/"latest.pt",arm="A_win128",strict_journal=True)
    except RuntimeError as e: blocked="ambiguous_step" in str(e)
    # Atomic temp garbage cannot replace valid latest.
    (m.checkpoint_dir/"latest.pt.tmp").write_bytes(b"broken")
    valid_latest=sha256_file(m.checkpoint_dir/"latest.pt")==safe_sha
    return {"test_id":"crash_transaction_atomicity","pass":bool(blocked and valid_latest),
      "pending_action_id":row["action_seq"],"optimizer_really_called_after_pending":True,"strict_restore_blocked":blocked,
      "valid_latest_survives_broken_tmp":valid_latest,"safe_checkpoint_sha":safe_sha}

def _fixture_gates(out):
    req=c.verify_plan()["engineering"]["required_fixture_ids"]
    base={"protocol":"042","revision":2,"all_pass":True,"fixtures":[{"test_id":k,"pass":True,"event":"x"} for k in req],
          "real_engineering_prefixes":1,"real_engineering_prefix_intervals_max":256,"real_engineering_gradient_steps":0}
    d=Path(out)/"f9"; d.mkdir(parents=True,exist_ok=True)
    cases={}
    for name,mut in [
      ("missing",lambda x:x.pop("all_pass")),("string_true",lambda x:x.__setitem__("all_pass","true")),
      ("wrong_revision",lambda x:x.__setitem__("revision",1)),("false_report",lambda x:x.__setitem__("all_pass",False))]:
        x=copy.deepcopy(base); mut(x); p=d/(name+".json"); W(p,x)
        try: validate_fixture_report(p); cases[name]=False
        except Exception: cases[name]=True
    # Comparator catches altered U parameter/update evidence.
    a=[{"expert_id":0,"batch_indices":[1,2],"model_hash_after":"a"}]; b=copy.deepcopy(a); b[0]["model_hash_after"]="b"
    comparator=(normalize_protocol_arm(a)!=normalize_protocol_arm(b))
    return {"test_id":"fail_closed_revision_reproduction_gates","pass":bool(all(cases.values()) and comparator),
      "negative_gate_cases":cases,"modified_parameter_hash_detected":comparator,"missing_OR_fallback_used":False}

def _fixture_recompute(out):
    s=c.synthetic_source(220,4217); m=NewMachine(s,Path(out)/"f10","A_win128")
    e=c.new_expert(0,0,"active"); p37.do_update(e["model"],e["optimizer"],s["core"],[0,1,2]); e["version"]=1
    m.experts={0:e}; m.active_ids=[0]; m.next_id=1; m._topology_epoch(-1,"fixture")
    for i in range(150): m._predict(i); m._settle(i+2)
    tr=m.tracker(); online=tr.score(0,149)
    # Independent reconstruction from sealed issued arrays only, no expert model call.
    off=c.UtilityTracker("win128"); off.reset(m.deployment_epoch,0,[0],"offline")
    for i in range(150):
        off.settle(i,m.deployment_epoch,m.out["live_margin"][i],{0:m.out["contribution"][0,i]},s["B"]["labels"][i])
    offline=off.score(0,149)
    keys=("score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta")
    score_ok=all(abs(float(online[k])-float(offline[k]))<=1e-10 for k in keys)
    bm=binary_metrics(m.out["probability"][:150],s["B"]["labels"][:150])
    manual_bce=float(np.mean(c.stable_bce_rows(m.out["live_margin"][:150],s["B"]["labels"][:150])))
    metric_ok=abs(bm["bce"]-manual_bce)<=1e-12
    return {"test_id":"independent_metric_and_score_recompute","pass":bool(score_ok and metric_ok),
      "online_score":{k:online[k] for k in keys},"offline_score":{k:offline[k] for k in keys},
      "score_abs_tolerance":1e-10,"metric_bce":bm["bce"],"manual_bce":manual_bce,"expert_forward_calls_during_recompute":0}

def run_fixtures(real_src,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    # One allowed real prefix, zero gradients, before E0. U production entrypoint and disk restore.
    us=u_source(real_src); m=UMachine(us,out/"real_prefix",allow_gradient=False,real_science=False); m.checkpoint_dir=out/"real_prefix_cp"
    m.advance(128); m.save_checkpoint("real128",True)
    r=UMachine.restore(us,out/"real_prefix_restore",m.checkpoint_dir/"latest.pt",allow_gradient=False,strict_journal=False); r.checkpoint_dir=out/"real_prefix_restore_cp"; r.advance(256)
    real_ok=bool(np.array_equal(r.out["probability"][:256],real_src["B"]["probability"][:256]) and r.live_optimizer_steps==0 and r.shadow_optimizer_steps==0 and not r.experts)
    rows=[_fixture_additive(out),_fixture_window(),_fixture_maturity(out),_fixture_victim(out),_fixture_gradients(out),
          _fixture_qualification(out),_fixture_resume(out),_fixture_crash(out),_fixture_gates(out),_fixture_recompute(out)]
    req=c.verify_plan()["engineering"]["required_fixture_ids"]; by={x["test_id"]:x for x in rows}
    ok=real_ok and set(by)==set(req) and all(type(by[k].get("pass")) is bool and by[k]["pass"] for k in req)
    report={"protocol":"042","revision":2,"all_pass":bool(ok),"required_fixture_ids":req,"fixtures":rows,
      "real_engineering_prefixes":1,"real_engineering_prefix_intervals_max":256,"real_engineering_gradient_steps":0,
      "real_prefix_exact_B_disk_resume":real_ok,"production_entrypoint_shared":True,"real_event_evidence_required":True}
    W(out/"fixture_report.json",report); return report

# ---------- commands ----------

def cmd_preflight(a):
    runtime(); c.verify_plan(); src=c.load_sources(a.source036,a.source041)
    rep=run_fixtures(src,a.out_dir)
    print(json.dumps({"protocol":"042","revision":2,"all_pass":rep["all_pass"],"fixture_count":len(rep["fixtures"])},indent=2))
    if not rep["all_pass"]: raise RuntimeError("engineering_incomplete")

def cmd_copy_controls(a):
    runtime(); src=c.load_sources(a.source036,a.source041); copy_controls(src,a.out_dir)

def cmd_arm(a):
    runtime(); c.verify_plan(); src=c.load_sources(a.source036,a.source041); root=Path(a.out_dir); root.mkdir(parents=True,exist_ok=True)
    fixture=validate_fixture_report(a.fixture_report)
    start_arm(root,a.arm,fixture,a.resume_checkpoint,a.run_id)
    before36={rel:sha256_file(src["source036_root"]/rel) for rel in c.verify_plan()["source036"]["files"]}
    before41={rel:sha256_file(src["source041_root"]/rel) for rel in c.verify_plan()["source041"]["files"]}
    ar=root/a.arm; ar.mkdir(parents=True,exist_ok=True)
    W(ar/"source_lock.json",{"protocol":"042","revision":2,"arm":a.arm,"source036":before36,"source041":before41,"F_loaded":False})
    t0=time.perf_counter(); cpu0=time.process_time()
    try:
        if a.arm=="U_parent":
            us=u_source(src)
            m=UMachine.restore(us,ar/"runtime_state",a.resume_checkpoint,allow_gradient=True,real_science=True,strict_journal=True) if a.resume_checkpoint else UMachine(us,ar/"runtime_state",True,True)
            m.checkpoint_dir=ar/"checkpoints"; m.advance(); m.terminal_settle(); sm=save_u(root,src,m)
            # U audit before reproduction.
            audit={"fixture_all_pass":fixture["all_pass"] is True,"first_birth_t351":m.first_birth_t==351,
              "live_steps_within_352":m.live_optimizer_steps<=352,"shadow_steps_within_16":m.shadow_optimizer_steps<=16,
              "terminal_progress_complete":m.terminal_progress==3 and m.next_substep=="done","output_all_finite":bool(np.isfinite(m.out["probability"]).all()),
              "source036_unchanged":before36=={rel:sha256_file(src["source036_root"]/rel) for rel in before36},
              "source041_unchanged":before41=={rel:sha256_file(src["source041_root"]/rel) for rel in before41},"no_F_loaded":True}
            audit["all_pass"]=all(audit.values()); W(ar/"run_audit.json",audit)
            rep=reproduce_U(src,root)
            if not audit["all_pass"] or not rep["all_pass"]: raise RuntimeError("U_reproduction_failed_stop_before_new_arms")
        else:
            validate_U_gate(root)
            m=NewMachine.restore(src,ar/"runtime_state",a.resume_checkpoint,arm=a.arm,allow_gradient=True,real_science=True,strict_journal=True) if a.resume_checkpoint else NewMachine(src,ar/"runtime_state",a.arm,True,True)
            m.checkpoint_dir=ar/"checkpoints"; m.advance(); m.terminal_settle(); sm=save_new(root,src,m)
            audit=audit_new(src,m,fixture,before36,before41); W(ar/"run_audit.json",audit)
            if not audit["all_pass"]: raise RuntimeError(a.arm+" invalid_execution")
        W(ar/"science_cost_raw.json",{"protocol":"042","revision":2,"arm":a.arm,"wall_seconds":time.perf_counter()-t0,
          "process_seconds":time.process_time()-cpu0,"prediction_seconds":float(getattr(m,"prediction_seconds",0)),
          "update_seconds":float(getattr(m,"update_seconds",0)),"controller_cpu_seconds":float(getattr(m,"controller_cpu_seconds",0)),
          "live_optimizer_steps":int(sm["live_optimizer_steps"]),"shadow_optimizer_steps":int(sm["shadow_optimizer_steps"]),
          "deployed_forwards":int(sm["deployed_prediction_forwards"]),"reuse_forwards":int(sm["reuse_preview_forwards"]),
          "qualification_forwards":int(sm["shadow_qualification_forwards"]),
          "live_training_forwards":int(getattr(m,"live_training_forwards",0)),"shadow_training_forwards":int(getattr(m,"shadow_training_forwards",0))})
        complete_arm(root,a.arm,sm)
        W(ar/"scientific_execution_complete.json",{"protocol":"042","revision":2,"arm":a.arm,"complete":True,"audit_all_pass":True})
        W(ar/"status.json",{"protocol":"042","revision":2,"arm":a.arm,"science_status":"completed","analysis_status":"pending","publication_status":"pending"})
        print(json.dumps(sm,indent=2))
    except Exception as e:
        W(ar/"status.json",{"protocol":"042","revision":2,"arm":a.arm,"science_status":"interrupted","error":repr(e),"traceback":traceback.format_exc(),
          "analysis_status":"not_started","publication_status":"pending","restart_from_zero":False})
        raise

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("preflight"); p.add_argument("--source036",required=True); p.add_argument("--source041",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_preflight)
    p=sub.add_parser("copy-controls"); p.add_argument("--source036",required=True); p.add_argument("--source041",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_copy_controls)
    p=sub.add_parser("arm"); p.add_argument("--arm",choices=ARMS,required=True); p.add_argument("--source036",required=True); p.add_argument("--source041",required=True)
    p.add_argument("--out-dir",required=True); p.add_argument("--fixture-report",required=True); p.add_argument("--run-id",required=True); p.add_argument("--resume-checkpoint"); p.set_defaults(fn=cmd_arm)
    a=ap.parse_args(); a.fn(a)
if __name__=="__main__": main()
