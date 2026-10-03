"""Frozen Protocol-043 revision1 analysis with independent metric and score recomputation."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from protocol035_common import load_npz, dump_json, phase_bounds, W_BLOCKS

WINDOWS=[("U_rec1",3300,3428),("V_rec1",3808,3936),("U_rec2",4316,4444),("V_rec2",4824,4952),("U_rec3",5332,5460),("V_rec3",5840,5968)]
LATE={"U_rec2","V_rec2","U_rec3","V_rec3"}
def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def f(x): return "null" if x is None else "{:+.6f}".format(float(x))
def mean(xs): return None if any(x is None for x in xs) else float(np.mean(xs))
def metrics(prob,y):
    p=np.asarray(prob,dtype=np.float64).reshape(-1); yy=(np.asarray(y).reshape(-1)>0).astype(np.int64)
    if not np.isfinite(p).all(): raise AssertionError("nonfinite probabilities")
    n=len(yy); pos=int(yy.sum()); neg=n-pos; pr=p>=.5; tp=int(np.sum(pr&(yy==1))); fp=int(np.sum(pr&(yy==0))); fn=pos-tp; tn=neg-fp
    rec=None if pos==0 else tp/pos; fpr=None if neg==0 else fp/neg
    eps=np.finfo(np.float64).eps; pc=np.clip(p,eps,1-eps); bce=float(np.mean(-(yy*np.log(pc)+(1-yy)*np.log(1-pc))))
    ap=None
    if pos and neg:
        order=np.argsort(-p,kind="mergesort"); sp=p[order]; sy=yy[order]; ctp=cfp=0; acc=0.; i=0
        while i<n:
            j=i+1
            while j<n and sp[j]==sp[i]: j+=1
            gp=int(sy[i:j].sum()); gn=(j-i)-gp; ctp+=gp; cfp+=gn
            if gp: acc+=(ctp/(ctp+cfp))*(gp/pos)
            i=j
        ap=float(acc)
    return {"ap":ap,"bce":bce,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"recall":rec,"fpr":fpr,"rows":n,"positives":pos,"negatives":neg}
def md(a,b): return {k+"_delta":None if a.get(k) is None or b.get(k) is None else float(a[k]-b[k]) for k in ("ap","bce","recall","fpr")}
def pooled(a,y,ranges):
    ps=np.concatenate([a[s:e] for _,s,e in ranges],axis=0); ys=np.concatenate([y[s:e] for _,s,e in ranges],axis=0); return metrics(ps,ys)
def cmp(base,arm,bounds):
    if not np.array_equal(base["labels"],arm["labels"]): raise AssertionError("labels mismatch")
    y=base["labels"]; bf=metrics(base["probability"],y); af=metrics(arm["probability"],y); wins=[]
    for n,s,e in WINDOWS:
        b=metrics(base["probability"][s:e],y[s:e]); a=metrics(arm["probability"][s:e],y[s:e])
        b32=metrics(base["probability"][s:s+32],y[s:s+32]); a32=metrics(arm["probability"][s:s+32],y[s:s+32])
        b64=metrics(base["probability"][s:s+64],y[s:s+64]); a64=metrics(arm["probability"][s:s+64],y[s:s+64])
        wins.append({"window":n,"intervals":[s,e],"base":b,"arm":a,**md(a,b),"prefix32":{"base":b32,"arm":a32,**md(a32,b32)},"prefix64":{"base":b64,"arm":a64,**md(a64,b64)}})
    bp=pooled(base["probability"],y,WINDOWS); ap=pooled(arm["probability"],y,WINDOWS)
    wrows=[]
    for n in W_BLOCKS:
        if n in bounds:
            s,e=bounds[n]; b=metrics(base["probability"][s:e],y[s:e]); a=metrics(arm["probability"][s:e],y[s:e]); wrows.append({"phase":n,"intervals":[s,e],"base":b,"arm":a,**md(a,b)})
    wmean=mean([x["ap_delta"] for x in wrows]) if wrows else None
    local=[{"window":x["window"],"recall_delta":x["recall_delta"],"prefix32_recall_delta":x["prefix32"]["recall_delta"],
      "recall_pass":x["recall_delta"] is not None and x["recall_delta"]>=-.02,
      "prefix32_recall_pass":x["prefix32"]["recall_delta"] is not None and x["prefix32"]["recall_delta"]>=-.03} for x in wins]
    g={"full_fpr_delta":af["fpr"]-bf["fpr"],"full_recall_delta":af["recall"]-bf["recall"],
      "recurrence_pooled_fpr_delta":ap["fpr"]-bp["fpr"],"recurrence_pooled_recall_delta":ap["recall"]-bp["recall"],"W_equal_weight_AP_delta":wmean,"local_recurrence":local}
    g["common_pass"]=bool(g["full_fpr_delta"]<=.01 and g["recurrence_pooled_fpr_delta"]<=.01 and g["full_recall_delta"]>=-.01 and g["recurrence_pooled_recall_delta"]>=-.01 and wmean is not None and wmean>=-.02)
    g["local_pass"]=all(x["recall_pass"] and x["prefix32_recall_pass"] for x in local); g["all_pass"]=bool(g["common_pass"] and g["local_pass"])
    return {"base_full":bf,"arm_full":af,**md(af,bf),"windows":wins,"six_window_equal_weight_AP_delta":mean([x["ap_delta"] for x in wins]),
      "late4_equal_weight_AP_delta":mean([x["ap_delta"] for x in wins if x["window"] in LATE]),"prefix32_equal_weight_AP_delta":mean([x["prefix32"]["ap_delta"] for x in wins]),
      "prefix64_equal_weight_AP_delta":mean([x["prefix64"]["ap_delta"] for x in wins]),"positive_recurrence_windows":sum(x["ap_delta"]>0 for x in wins),
      "recurrence_pooled":{"base":bp,"arm":ap,**md(ap,bp)},"W_blocks":wrows,"guardrails":g}

def bce_rows(m,y):
    m=np.asarray(m,dtype=np.float64); yy=(np.asarray(y)>0).astype(np.float64)
    return np.maximum(m,0)-m*yy+np.log1p(np.exp(-np.abs(m)))
def score_independent(pred,eid,m):
    ep=int(pred["deployment_epoch"][m]); idx=np.flatnonzero(pred["deployment_epoch"][:m+1]==ep)
    if len(idx)==0: return {"valid":False,"reason":"no_epoch","score":None,"score_pos":None,"score_neg":None}
    start=int(idx[0]); matured=m-start+1
    if matured<128: return {"valid":False,"reason":"epoch_mature_lt128","score":None,"score_pos":None,"score_neg":None,"epoch_matured":matured}
    ii=np.arange(m-127,m+1,dtype=np.int64)
    if np.any(pred["deployment_epoch"][ii]!=ep): return {"valid":False,"reason":"window_not_full","score":None,"score_pos":None,"score_neg":None}
    d=pred["contribution"][int(eid),ii]
    if not np.isfinite(d).all(): return {"valid":False,"reason":"missing_contribution","score":None,"score_pos":None,"score_neg":None}
    live=np.asarray(pred["live_margin"][ii],dtype=np.float64); dd=np.asarray(d,dtype=np.float32).astype(np.float64); y=(pred["labels"][ii]>0)
    rem=live-dd; u=bce_rows(rem,y)-bce_rows(live,y); flat=u.reshape(-1); fy=y.reshape(-1); pos=fy; neg=~fy
    pn=int(pos.sum()); nn=int(neg.sum()); pint=int(sum(bool(np.any(row)) for row in y))
    if pn<16 or nn<16 or pint<4: return {"valid":False,"reason":"insufficient_support","score":None,"score_pos":None,"score_neg":None,
      "positive_host_rows":pn,"negative_host_rows":nn,"positive_intervals":pint}
    lm=metrics(1/(1+np.exp(-np.clip(live,-80,80))),y); rm=metrics(1/(1+np.exp(-np.clip(rem,-80,80))),y)
    return {"valid":True,"reason":"ok","score":float(flat.mean()),"score_pos":float(flat[pos].mean()),"score_neg":float(flat[neg].mean()),
      "positive_host_rows":pn,"negative_host_rows":nn,"positive_intervals":pint,"removal_fpr_delta":float(rm["fpr"]-lm["fpr"]),
      "removal_recall_delta":float(rm["recall"]-lm["recall"]),"fpr_live":lm["fpr"],"fpr_removed":rm["fpr"],"recall_live":lm["recall"],"recall_removed":rm["recall"],
      "epoch_matured":matured,"intervals":128,"first_i":int(ii[0]),"last_i":int(ii[-1])}
def score_recompute(root,arm,pred):
    rows=J(root/arm/"utility_checks.json")["checks"]; mism=[]; checked=0; streak={}; last_ep=None
    for r in rows:
        ep=int(r["epoch"]); eid=int(r["expert_id"]); m=int(r["at_interval"])-2
        if ep!=last_ep: streak={}; last_ep=ep
        off=score_independent(pred,eid,m); ok=bool(off.get("valid") and off["score"]<=0 and off["score_pos"]<=0 and off["score_neg"]<=0 and off["removal_fpr_delta"]<=.01 and off["removal_recall_delta"]>=-.02)
        before=int(streak.get(eid,0)); after=before+1 if ok else 0; streak[eid]=after
        off={**off,"eligible_now":ok,"streak_before":before,"streak_after":after,"eligible_three":bool(ok and after>=3)}; checked+=1
        same=off.get("valid")==r.get("valid") and off.get("reason")==r.get("reason") and off["eligible_now"]==r.get("eligible_now") and off["streak_before"]==r.get("streak_before") and off["streak_after"]==r.get("streak_after") and off["eligible_three"]==r.get("eligible_three")
        if same and off.get("valid"):
            for k in ("score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta"):
                if abs(float(off[k])-float(r[k]))>1e-10: same=False
        if not same: mism.append({"at_interval":r["at_interval"],"expert_id":eid,"online":r,"independent":off})
    return {"pass":len(mism)==0,"checked":checked,"mismatch_count":len(mism),"mismatches":mism,"abs_tolerance":1e-10,"model_forwards":0}

def norm(x):
    if isinstance(x,list): return [norm(v) for v in x]
    if isinstance(x,dict): return {k:norm(v) for k,v in x.items() if k not in ("arm","controller_cpu_seconds","prediction_seconds","update_seconds","io_seconds")}
    return x
def before_t(rows,t):
    return [norm(x) for x in rows if int(x.get("at_interval",-1))<int(t)]
def divergence_audit(root,A,N):
    br=J(root/"D_bounded/reclamation_events.json")["events"]; nr=J(root/"D_no_gc/reclamation_events.json")["events"]
    gc=[x for x in br if x.get("event")=="permanent_reclaim_and_shadow_start"]
    names=[("lifecycle_events","events"),("candidate_decisions","decisions"),("reuse_decisions","decisions"),("update_log","updates")]
    if gc:
        t=int(gc[0]["at_interval"]); arrays=bool(np.array_equal(A["probability"][:t+1],N["probability"][:t+1]) and np.array_equal(A["active_ids"][:t+1],N["active_ids"][:t+1]) and np.array_equal(A["expert_hashes"][:t+1],N["expert_hashes"][:t+1]))
        logs={}
        for fn,key in names:
            x=J(root/"D_bounded"/(fn+".json"))[key]; y=J(root/"D_no_gc"/(fn+".json"))[key]; logs[fn]=before_t(x,t)==before_t(y,t)
        same=arrays and all(logs.values())
        block=any(x.get("event")=="capacity_blocked_no_gc" and int(x["at_interval"])==t for x in nr)
        return {"pass":bool(same and block),"gc_exercised":True,"first_control_difference_t":t,"first_affected_prediction":t+1,
          "outputs_and_state_exact_through_control_tick":arrays,"logs_exact_before_control":logs,"no_gc_block_same_t":block}
    exact=bool(np.array_equal(A["probability"],N["probability"]) and np.array_equal(A["active_ids"],N["active_ids"]) and np.array_equal(A["contribution"],N["contribution"]))
    logs={}
    for fn,key in names:
        x=J(root/"D_bounded"/(fn+".json"))[key]; y=J(root/"D_no_gc"/(fn+".json"))[key]; logs[fn]=norm(x)==norm(y)
    return {"pass":bool(exact and all(logs.values())),"gc_exercised":False,"whole_trajectory_exact":exact,"logs_exact":logs}

def active_predictions(pred,eid): return int(np.sum(np.any(pred["active_ids"]==int(eid),axis=1)))
def dormant_span(pred,eid,t):
    s=int(t)+1; n=len(pred["active_ids"]); k=0
    for i in range(s,n):
        if np.any(pred["active_ids"][i]==int(eid)): break
        k+=1
    return k
def post_active_span(pred,eid,t):
    s=int(t)+1; n=len(pred["active_ids"]); k=0
    for i in range(s,n):
        if np.any(pred["active_ids"][i]==int(eid)): k+=1
        else: break
    return k
def lifecycle(root,pred):
    cand=J(root/"D_bounded/candidate_decisions.json")["decisions"]; life=J(root/"D_bounded/lifecycle_events.json")["events"]; rec=J(root/"D_bounded/reclamation_events.json")["events"]
    accepted=[]
    for x in cand:
        if x.get("event")=="shadow_evaluate" and x.get("pass") is True and int(x.get("candidate_id",0))>0:
            eid=int(x["candidate_id"]); accepted.append({"expert_id":eid,"at_interval":int(x["at_interval"]),"active_predictions_total":active_predictions(pred,eid),"post_accept_consecutive":post_active_span(pred,eid,int(x["at_interval"]))})
    sleeps=[]
    for x in life:
        if x.get("event")=="sleep":
            eid=int(x["expert_id"]); sleeps.append({"expert_id":eid,"at_interval":int(x["at_interval"]),"dormant_predictions":dormant_span(pred,eid,int(x["at_interval"]))})
    reuse=[]
    for x in life:
        if x.get("event")=="activate" and x.get("reason")=="reuse_accept":
            eid=int(x["expert_id"]); reuse.append({"expert_id":eid,"at_interval":int(x["at_interval"]),"dormant_before":int(x.get("dormant_predictions_before_reactivation",0)),"post_active":post_active_span(pred,eid,int(x["at_interval"]))})
    gc=[x for x in rec if x.get("event")=="permanent_reclaim_and_shadow_start"]
    gc_closed=[]
    for x in gc:
        cid=int(x["candidate_id"]); q=next((z for z in cand if z.get("event")=="shadow_evaluate" and int(z.get("candidate_id",-1))==cid),None)
        gc_closed.append({"deleted_id":int(x["deleted_id"]),"candidate_id":cid,"accepted":bool(q and q.get("pass") is True),"active_predictions":active_predictions(pred,cid)})
    core=bool(any(x["active_predictions_total"]>=64 for x in accepted) and any(x["dormant_predictions"]>=64 for x in sleeps))
    reuse_ok=bool(any(x["dormant_before"]>=64 and x["post_active"]>=64 for x in reuse))
    gc_loop=bool(any(x["accepted"] and x["active_predictions"]>=64 for x in gc_closed))
    return {"post_E0_accepts":accepted,"sleep_events":sleeps,"reuse_events":reuse,"gc_events":gc,"gc_closed_loop_rows":gc_closed,
      "core_lifecycle_exercised":core,"reuse_exercised":reuse_ok,"gc_exercised_on_stream":bool(gc),"gc_admission_closed_loop":gc_loop}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--science-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--fixture-report",required=True)
    ap.add_argument("--audit042-gate",required=True); ap.add_argument("--gate-report",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--docs-output",required=True); ap.add_argument("--run-id",required=True)
    a=ap.parse_args(); root=Path(a.science_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    C=load_npz(root/"cached/C_ref/predictions.npz"); B=load_npz(root/"cached/B_ref/predictions.npz"); K=load_npz(root/"cached/D_keep/predictions.npz")
    H=load_npz(root/"cached/042_A_hist/predictions.npz"); W=load_npz(root/"cached/042_A_win128/predictions.npz")
    N=load_npz(root/"D_no_gc/predictions.npz"); D=load_npz(root/"D_bounded/predictions.npz")
    for q in (B,K,H,W,N,D):
        for key in ("labels","raw_labels","class_probability"):
            if key not in q or not np.array_equal(C[key],q[key]): raise AssertionError("alignment "+key)
    fixture=J(a.fixture_report); audit42=J(a.audit042_gate); gate=J(a.gate_report); ledger=J(root/"budget_ledger.json")
    audits={arm:J(root/arm/"run_audit.json") for arm in ("D_no_gc","D_bounded")}
    base_valid=bool(type(fixture.get("all_pass")) is bool and fixture["all_pass"] and type(audit42.get("all_pass")) is bool and audit42["all_pass"] and
      type(gate.get("all_pass")) is bool and gate["all_pass"] and all(type(x.get("all_pass")) is bool and x["all_pass"] for x in audits.values()) and
      all(ledger["sequences"][x]["completed"] is True for x in ("D_no_gc","D_bounded")))
    score_checks={arm:score_recompute(root,arm,{"probability":q["probability"],"labels":q["labels"],"live_margin":q["live_margin"],"deployment_epoch":q["deployment_epoch"],"contribution":q["contribution"]}) for arm,q in (("D_no_gc",N),("D_bounded",D))}
    div=divergence_audit(root,D,N); independent_ok=all(x["pass"] for x in score_checks.values()) and div["pass"]
    validity=bool(base_valid and independent_ok)
    bounds=phase_bounds(a.manifest); DC=cmp(C,D,bounds); DN=cmp(N,D,bounds); DK=cmp(K,D,bounds); DH=cmp(H,D,bounds); DW=cmp(W,D,bounds); DB=cmp(B,D,bounds)
    perf=bool(DC["ap_delta"] is not None and DC["ap_delta"]>=.002 and DC["six_window_equal_weight_AP_delta"]>=.002 and DC["positive_recurrence_windows"]>=4 and
      DC["late4_equal_weight_AP_delta"]>=0 and DC["prefix32_equal_weight_AP_delta"]>=-.005 and DC["guardrails"]["all_pass"])
    life=lifecycle(root,D); req={r["test_id"]:r for r in fixture["fixtures"]}
    gceng=bool(all(req[k]["pass"] is True for k in ("production_full_pool_delete_birth_accept64","production_eviction_then_candidate_reject","two_arm_first_GC_divergence","disk_resume_continue_to_end_all_states","crash_atomic_step_eviction_budget")))
    sm=J(root/"D_bounded/summary.json"); resource=bool(sm["max_active_seen"]<=2 and sm["max_resident_seen"]<=3 and sm["attempts_after_e0"]<=4 and sm["ids_created"]<=5 and
      sm["live_optimizer_steps"]<=704 and sm["shadow_optimizer_steps"]<=64 and sm["permanent_deletions"]<=4 and audits["D_bounded"]["all_pass"])
    core=bool(validity and perf and life["core_lifecycle_exercised"] and gceng and resource); full=bool(core and life["gc_admission_closed_loop"])
    if not validity: label="invalid_execution"
    elif full: label="core_goal_supported_full_reclamation_demonstrated"
    elif core and not life["gc_exercised_on_stream"]: label="core_goal_supported_gc_unexercised_on_stream"
    elif core: label="core_goal_supported_gc_cost_observed_without_closed_loop"
    elif not perf: label="D_over_C_not_established"
    elif not life["core_lifecycle_exercised"]: label="core_lifecycle_not_exercised"
    elif not gceng: label="gc_engineering_not_verified"
    else: label="resource_or_validity_gate_failed"
    comparison={"protocol":"043","revision":1,"run_id":str(a.run_id),"validity":validity,"system_result_label":label,
      "primary":{"D_bounded_vs_C_ref":DC,"D_over_C":perf},"diagnostic":{"D_bounded_vs_D_no_gc":DN,"D_bounded_vs_D_keep":DK,"D_bounded_vs_042_A_hist":DH,"D_bounded_vs_042_A_win128":DW,"D_bounded_vs_B_ref":DB},
      "lifecycle":life,"gc_engineering_verified":gceng,"resource_bounds_pass":resource,"core_goal_supported":core,"full_reclamation_demonstrated":full,
      "first_control_divergence_audit":div,"independent_score_recompute":score_checks,
      "claims":{"independent_generalization_confirmation":False,"best_structure_or_window":False,"equal_compute_superiority":False,"deleted_expert_forever_useless":False,"end_to_end_speedup":False,"automatic_followup":False}}
    dump_json(out/"comparison.json",comparison)
    dump_json(out/"analysis_status.json",{"protocol":"043","revision":1,"run_id":str(a.run_id),"validity":validity,"system_result_label":label,"D_over_C":perf,
      "core_lifecycle_exercised":life["core_lifecycle_exercised"],"reuse_exercised":life["reuse_exercised"],"gc_engineering_verified":gceng,
      "gc_exercised_on_stream":life["gc_exercised_on_stream"],"gc_admission_closed_loop":life["gc_admission_closed_loop"],"resource_bounds_pass":resource,
      "core_goal_supported":core,"full_reclamation_demonstrated":full,"publication_status":"pending_until_main_sync","automatic_followup":False})
    dump_json(out/"independent_score_recompute.json",score_checks); dump_json(out/"first_control_divergence.json",div); dump_json(out/"lifecycle_evidence.json",life)
    costs={arm:J(root/arm/"science_cost_raw.json") for arm in ("D_no_gc","D_bounded")}; dump_json(out/"cost_profile.json",{"protocol":"043","revision":1,"arms":costs,
      "B_cache_cost_separate":True,"equal_compute_claim":False,"audit_disk_bytes":sum(p.stat().st_size for p in root.rglob("*") if p.is_file())})
    lines=["# Protocol-043 revision 1 结果","",
      "本轮仅新增两条已登记科学序列：D_no_gc → D_bounded。主臂预先固定为 D_bounded，唯一必须性能比较为冻结 C_ref；其它D仅作描述。","",
      "## 主结论","",
      "- validity={}；D_over_C={}；core_lifecycle_exercised={}；gc_engineering_verified={}；resource_bounds_pass={}。".format(validity,perf,life["core_lifecycle_exercised"],gceng,resource),
      "- core_goal_supported={}；gc_exercised_on_stream={}；gc_admission_closed_loop={}；full_reclamation_demonstrated={}。".format(core,life["gc_exercised_on_stream"],life["gc_admission_closed_loop"],full),
      "- 系统标签：**{}**。".format(label),
      "- D_bounded相对C：full ΔAP {}；six {}；late4 {}；prefix32 {}；正向窗口 {}/6；guardrails={}。".format(f(DC["ap_delta"]),f(DC["six_window_equal_weight_AP_delta"]),f(DC["late4_equal_weight_AP_delta"]),f(DC["prefix32_equal_weight_AP_delta"]),DC["positive_recurrence_windows"],DC["guardrails"]["all_pass"]),"",
      "## 生命周期与回收","",
      "- E0后接受候选：{}；窗口sleep：{}；same-ID reuse满足64/64：{}。".format(len(life["post_E0_accepts"]),len(life["sleep_events"]),life["reuse_exercised"]),
      "- 真实流永久回收次数：{}；回收创建候选闭环>=64：{}。".format(len(life["gc_events"]),life["gc_admission_closed_loop"]),
      "- 若真实流未触发GC，合成闭环仅证明工程机制可运行，不代表真实回收提高性能。","",
      "## 描述性比较（非成功门槛）","",
      "- D_bounded−D_no_gc full ΔAP {}；D_bounded−D_keep {}；D_bounded−042 A_hist {}；D_bounded−042 A_win128 {}。".format(f(DN["ap_delta"]),f(DK["ap_delta"]),f(DH["ap_delta"]),f(DW["ap_delta"])),"",
      "## 审计","",
      "- 两臂独立128-window评分全量复算：{}；首次真实控制分歧审计：{}。".format(all(x["pass"] for x in score_checks.values()),div["pass"]),
      "- 042旧报告保持原invalid_execution；043/audit042只读诊断不会覆盖旧结果。",
      "- 数据仍是已观察seed3601开发流；不声称独立泛化、最优结构/窗口、等算力胜C、永久删除永远安全或端到端加速。",
      "- Protocol-043 revision1 到此停止，不自动启动044。",""]
    Path(a.docs_output).write_text("\n".join(lines),encoding="utf8")
    print(json.dumps(J(out/"analysis_status.json"),indent=2,ensure_ascii=False))
if __name__=="__main__": main()
