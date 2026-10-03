"""Frozen Protocol-042 revision 2 analysis."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

import analyze_ftmoe_protocol039 as a39
import analyze_ftmoe_protocol040 as a40
from protocol035_common import binary_metrics, load_npz, dump_json, phase_bounds, WINDOWS
import protocol042_common as c

LATE={"U_rec2","V_rec2","U_rec3","V_rec3"}

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def f(x): return "null" if x is None else "{:+.6f}".format(float(x))
def gate(v,x): return v is not None and v>=x

def arm_cmp(base,arm,bounds):
    return a39.cmp(base,arm,bounds)

def primary_signal(comp,valid):
    return bool(valid and gate(comp["ap_delta"],.002) and gate(comp["six_window_equal_weight_AP_delta"],.002)
      and comp["positive_recurrence_windows"]>=4 and gate(comp["late4_equal_weight_AP_delta"],0)
      and gate(comp["prefix32_equal_weight_AP_delta"],-.005) and comp["guardrails"]["all_pass"])

def preservation_signal(comp,valid):
    return bool(valid and gate(comp["ap_delta"],-.001) and gate(comp["six_window_equal_weight_AP_delta"],-.001)
      and gate(comp["late4_equal_weight_AP_delta"],-.001) and gate(comp["prefix32_equal_weight_AP_delta"],-.005)
      and comp["guardrails"]["all_pass"])

def mechanism_signal(comp,valid,exercised=True):
    return bool(valid and exercised and gate(comp["ap_delta"],0) and gate(comp["six_window_equal_weight_AP_delta"],.0005)
      and comp["positive_recurrence_windows"]>=4 and gate(comp["late4_equal_weight_AP_delta"],0) and comp["guardrails"]["all_pass"])

def independent_metrics(prob,y):
    p=np.asarray(prob,dtype=np.float64).reshape(-1); yy=(np.asarray(y).reshape(-1)>0).astype(np.int64)
    if not np.isfinite(p).all(): raise AssertionError("nonfinite probability")
    n=len(yy); pos=int(yy.sum()); neg=n-pos; pred=p>=.5
    tp=int(np.sum(pred&(yy==1))); fp=int(np.sum(pred&(yy==0))); fn=pos-tp; tn=neg-fp
    recall=None if pos==0 else tp/pos; fpr=None if neg==0 else fp/neg
    eps=np.finfo(np.float64).eps; pc=np.clip(p,eps,1-eps)
    bce=float(np.mean(-(yy*np.log(pc)+(1-yy)*np.log(1-pc))))
    ap=None
    if pos and neg:
        order=np.argsort(-p,kind="mergesort"); sp=p[order]; sy=yy[order]; ctp=cfp=0; acc=0.0; i=0
        while i<n:
            j=i+1
            while j<n and sp[j]==sp[i]: j+=1
            gp=int(sy[i:j].sum()); gn=(j-i)-gp; ctp+=gp; cfp+=gn
            if gp: acc+=(ctp/(ctp+cfp))*(gp/pos)
            i=j
        ap=float(acc)
    return {"ap":ap,"bce":bce,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"recall":recall,"fpr":fpr,"rows":n,"positives":pos,"negatives":neg}

def independent_arm_check(a):
    im=independent_metrics(a["probability"],a["labels"]); bm=binary_metrics(a["probability"],a["labels"])
    ok=(abs(im["bce"]-bm["bce"])<=1e-12 and ((im["ap"] is None and bm["ap"] is None) or abs(im["ap"]-bm["ap"])<=1e-12)
        and im["tp"]==bm["tp"] and im["fp"]==bm["fp"] and im["fn"]==bm["fn"] and im["tn"]==bm["tn"])
    return {"pass":bool(ok),"independent":im,"common":bm}

def score_direct(pred,eid,m,mode):
    ep=int(pred["deployment_epoch"][m]); idx=np.flatnonzero(pred["deployment_epoch"][:m+1]==ep)
    if len(idx)==0: return {"valid":False,"reason":"no_epoch"}
    start=int(idx[0]); epoch_mature=m-start+1
    if epoch_mature<128: return {"valid":False,"reason":"epoch_mature_lt128","score":None,"score_pos":None,"score_neg":None,"epoch_matured":epoch_mature}
    if mode=="win128":
        ii=np.arange(m-127,m+1,dtype=np.int64)
        if np.any(pred["deployment_epoch"][ii]!=ep): return {"valid":False,"reason":"window_not_full","score":None,"score_pos":None,"score_neg":None}
    else:
        ii=np.arange(start,m+1,dtype=np.int64)
    delta=pred["contribution"][int(eid),ii]
    observed=np.all(np.isfinite(delta),axis=1)
    ii=ii[observed]; delta=delta[observed]
    if len(ii)==0: return {"valid":False,"reason":"not_active_or_unobserved","score":None,"score_pos":None,"score_neg":None}
    if mode=="win128" and len(ii)!=128: return {"valid":False,"reason":"window_not_full","score":None,"score_pos":None,"score_neg":None}
    live=pred["live_margin"][ii].astype(np.float64); d=delta.astype(np.float64); y=(pred["labels"][ii]>0).astype(np.int64)
    ll=c.stable_bce_rows(live,y); rl=c.stable_bce_rows(live-d,y); u=rl-ll
    flaty=y.reshape(-1); flatu=u.reshape(-1); pos=flaty>0; neg=~pos; posn=int(pos.sum()); negn=int(neg.sum())
    pint=sum(bool(np.any(row>0)) for row in y)
    if posn<16 or negn<16 or pint<4:
        return {"valid":False,"reason":"insufficient_support","score":None,"score_pos":None,"score_neg":None,
          "positive_host_rows":posn,"negative_host_rows":negn,"positive_intervals":pint}
    lm=independent_metrics(c.sigmoid_margin(live),y); rm=independent_metrics(c.sigmoid_margin(live-d),y)
    return {"valid":True,"reason":"ok","score":float(flatu.mean()),"score_pos":float(flatu[pos].mean()),"score_neg":float(flatu[neg].mean()),
      "positive_host_rows":posn,"negative_host_rows":negn,"positive_intervals":pint,
      "removal_fpr_delta":float(rm["fpr"]-lm["fpr"]),"removal_recall_delta":float(rm["recall"]-lm["recall"]),
      "fpr_live":lm["fpr"],"fpr_removed":rm["fpr"],"recall_live":lm["recall"],"recall_removed":rm["recall"],
      "epoch_matured":epoch_mature,"intervals":len(ii),"first_i":int(ii[0]),"last_i":int(ii[-1])}

def score_recompute(root,arm,pred):
    rows=J(root/arm/"utility_checks.json")["checks"]; mode="hist" if arm=="A_hist" else "win128"
    diffs=[]; checked=0
    fields=("score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta")
    for r in rows:
        m=int(r["at_interval"])-2; eid=int(r["expert_id"]); off=score_direct(pred,eid,m,mode); checked+=1
        same=bool(off.get("valid")==r.get("valid") and off.get("reason")==r.get("reason"))
        if same and off.get("valid"):
            same=all(abs(float(off[k])-float(r[k]))<=1e-10 for k in fields)
        if not same: diffs.append({"at_interval":r["at_interval"],"expert_id":eid,"online":r,"offline":off})
    return {"pass":len(diffs)==0,"checked":checked,"mismatches":diffs[:20],"tolerance":1e-10,"model_forwards":0}

def longest_true(x):
    best=cur=0
    for v in np.asarray(x).astype(bool):
        cur=cur+1 if v else 0; best=max(best,cur)
    return best

def pool_and_activity(root,arm,pred):
    ids=pred["active_ids"]; counts={}
    for eid in range(3): counts[eid]=int(np.sum(np.any(ids==eid,axis=1)))
    accepted=[eid for eid,n in counts.items() if n>0]
    life=J(root/arm/"lifecycle_events.json")["events"]
    reactivation=False
    for ev in life:
        if ev.get("event")=="activate" and int(ev.get("dormant_predictions_before_reactivation",0))>=64:
            eid=int(ev["expert_id"]); t=int(ev["at_interval"])+1
            post=0
            for i in range(t,len(ids)):
                if np.any(ids[i]==eid): post+=1
                else: break
            if post>=64: reactivation=True
    pool=bool(len([eid for eid in accepted if counts[eid]>=64])>=2 and reactivation)
    multi_long=longest_true(pred["active_count"]>=2); multi=multi_long>=64
    cand=J(root/arm/"candidate_decisions.json")["decisions"]; reuse=J(root/arm/"reuse_decisions.json")["decisions"]
    replacement=any(bool(x.get("pass")) and isinstance(x.get("plan"),dict) and x["plan"].get("victim_id") is not None for x in cand)
    replacement=replacement or any(x.get("event")=="reuse_evaluate" and x.get("winner") is not None and any(
      isinstance(q.get("plan"),dict) and q["plan"].get("victim_id") is not None and int(q.get("candidate_id",-1))==int(x["winner"]) for q in x.get("candidates",[])) for x in reuse)
    sleep=any(x.get("event")=="sleep" for x in life)
    victim=bool(replacement or any(x.get("event")=="sleep" for x in life if len(x.get("active_ids_before",[]))>=2))
    return {"pool_exercised":pool,"accepted_ids":accepted,"active_prediction_counts":counts,"same_id_reactivation_64":reactivation,
      "multi_active_exercised":multi,"multi_active_longest_run":multi_long,"victim_selection_exercised":victim,
      "sleep_selection_exercised":sleep,"replacement_selection_exercised":replacement}

def prefix_divergence(root,A,H):
    ap=A["probability"]; hp=H["probability"]; ai=A["active_ids"]; hi=H["active_ids"]
    diff=np.any(ap!=hp,axis=1)|np.any(ai!=hi,axis=1)
    idx=np.flatnonzero(diff); first=None if len(idx)==0 else int(idx[0])
    # Update/control equality before first model-output divergence. Score values/buffers are intentionally excluded.
    cutoff=len(ap) if first is None else first
    Au=J(root/"A_win128/update_log.json")["updates"]; Hu=J(root/"A_hist/update_log.json")["updates"]
    def norm_updates(rows):
        return [{k:r.get(k) for k in ("at_interval","kind","expert_id","joint_active_ids","batch_indices","model_hash_before","model_hash_after",
                                      "optimizer_hash_before","optimizer_hash_after","optimizer_step_before","optimizer_step_after","version_after")
                 } for r in rows if int(r.get("at_interval",-1))<cutoff]
    updates_equal=norm_updates(Au)==norm_updates(Hu)
    def actions(arm):
        life=J(root/arm/"lifecycle_events.json")["events"]; cand=J(root/arm/"candidate_decisions.json")["decisions"]; reuse=J(root/arm/"reuse_decisions.json")["decisions"]
        rows=[]
        for typ,xs in (("life",life),("candidate",cand),("reuse",reuse)):
            for x in xs:
                t=int(x.get("at_interval",-1))
                if t<cutoff:
                    y={k:v for k,v in x.items() if k not in ("score","score_pos","score_neg","victim_recheck")}
                    rows.append((typ,t,y))
        return rows
    actions_equal=actions("A_win128")==actions("A_hist")
    legal=None
    if first is not None:
        # A legal divergence must be preceded by a topology/control event that differs no later than prediction first-1.
        aw=actions("A_win128"); ah=actions("A_hist")
        legal=aw!=ah
    return {"first_prediction_divergence":first,"predictions_identical_if_none":first is None,
      "updates_exact_before_divergence":updates_equal,"discrete_actions_exact_before_divergence":actions_equal,
      "legal_window_caused_control_difference_observed":legal,
      "window_mechanism_exercised":bool(first is not None and updates_equal and actions_equal and legal is True)}

def diagnostic_scores(pred):
    rows=[]
    epochs=np.unique(pred["deployment_epoch"])
    for ep in epochs:
        idx=np.flatnonzero(pred["deployment_epoch"]==ep)
        if len(idx)<128: continue
        m=int(idx[-1])
        for eid in range(3):
            if not np.isfinite(pred["contribution"][eid,idx]).any(): continue
            item={"epoch":int(ep),"expert_id":eid,"at_m":m,"score128":score_direct(pred,eid,m,"win128"),
                  "score_hist":score_direct(pred,eid,m,"hist")}
            # read-only 64/256 diagnostics (support may be invalid).
            for w in (64,256):
                lo=max(int(idx[0]),m-w+1); ii=np.arange(lo,m+1)
                d=pred["contribution"][eid,ii]; ok=np.all(np.isfinite(d),axis=1)
                item["mean_abs_delta_%d"%w]=None if not np.any(ok) else float(np.mean(np.abs(d[ok])))
            rows.append(item)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--science-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--fixture-report",required=True)
    ap.add_argument("--output-dir",required=True); ap.add_argument("--docs-output",required=True); ap.add_argument("--run-id",required=True)
    a=ap.parse_args(); root=Path(a.science_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fx=J(a.fixture_report); ug=J(root/"U_parent/reproduction_report.json")
    audits={arm:J(root/arm/"run_audit.json") for arm in c.ARMS}
    valid=bool(fx.get("all_pass") is True and ug.get("all_pass") is True and all(x.get("all_pass") is True for x in audits.values()))
    controls={name:load_npz(root/"cached"/name/"predictions.npz") for name in ("C_ref","B_ref","D_keep","D_039","D_040","W_041")}
    arms={arm:load_npz(root/arm/"predictions.npz") for arm in c.ARMS}
    bounds=phase_bounds(a.manifest)
    C=controls["C_ref"]; K=controls["D_keep"]; A=arms["A_win128"]; R=arms["R_win128"]; H=arms["A_hist"]
    vsC=arm_cmp(C,A,bounds); vsK=arm_cmp(K,A,bounds); primary=primary_signal(vsC,valid); preserve=preservation_signal(vsK,valid)
    addcmp=arm_cmp(R,A,bounds); add_ex=bool(np.any(A["active_ids"]!=R["active_ids"]) or np.any(A["probability"]!=R["probability"]))
    add_signal=mechanism_signal(addcmp,valid,add_ex)
    prefix=prefix_divergence(root,A,H); wincmp=arm_cmp(H,A,bounds); win_signal=mechanism_signal(wincmp,valid,prefix["window_mechanism_exercised"])
    activity=pool_and_activity(root,"A_win128",A)
    recompute={arm:score_recompute(root,arm,arms[arm]) for arm in c.NEW_ARMS}
    metrics={arm:independent_arm_check(arms[arm]) for arm in c.ARMS}
    independent_ok=all(x["pass"] for x in recompute.values()) and all(x["pass"] for x in metrics.values())
    if not independent_ok: valid=False; primary=False; preserve=False; add_signal=False; win_signal=False
    main_success=bool(valid and primary and preserve and activity["pool_exercised"])
    if not valid: label="invalid_execution"
    elif not primary: label="D_over_C_not_established"
    elif not preserve: label="D_over_C_with_excess_pool_loss"
    elif not activity["pool_exercised"]: label="quality_retained_pool_not_exercised"
    else: label="quality_retained_pool_exercised"
    mech={"addition_policy_exercised":add_ex,"addition_policy_signal":add_signal,
          "window_policy_signal":win_signal,"window_prefix_audit":prefix,
          "window_result_label":"window_policy_signal" if win_signal else ("window_mechanism_not_exercised" if not prefix["window_mechanism_exercised"] else "window_policy_no_signal")}
    comparison={"protocol":"042","revision":2,"run_id":str(a.run_id),"validity":valid,"primary_arm":"A_win128",
      "A_win128_vs_C":vsC,"A_win128_vs_D_keep":vsK,"A_win128_D_over_C":primary,"cumulative_preservation":preserve,
      "A_win128_vs_R_win128":addcmp,"A_win128_vs_A_hist":wincmp,"mechanisms":mech,"activity":activity,
      "main_development_success":main_success,"system_result_label":label,"U_reproduction":ug,
      "independent_recompute":{"pass":independent_ok,"scores":recompute,"metrics":metrics},
      "claims":{"independent_confirmation":False,"permanent_deletion_success":False,"causal_reuse_advantage":False,
                "overall_deployment_speedup":False,"equal_compute_comparison":False,"automatic_followup":False}}
    dump_json(out/"comparison.json",comparison)
    dump_json(out/"analysis_status.json",{"protocol":"042","revision":2,"run_id":str(a.run_id),"validity":valid,
      "system_result_label":label,"main_development_success":main_success,"D_over_C":primary,"cumulative_preservation":preserve,
      "addition_policy_signal":add_signal,"window_policy_signal":win_signal,"pool_exercised":activity["pool_exercised"],
      "multi_active_exercised":activity["multi_active_exercised"],"victim_selection_exercised":activity["victim_selection_exercised"],
      "publication_status":"pending_until_main_sync","automatic_followup":False})
    dump_json(out/"score_recompute.json",recompute); dump_json(out/"mechanism_diagnostics.json",{"mechanisms":mech,"activity":activity})
    dump_json(out/"read_only_window_diagnostics.json",{arm:diagnostic_scores(arms[arm]) for arm in c.NEW_ARMS})
    costs={arm:J(root/arm/"science_cost_raw.json") for arm in c.ARMS}; dump_json(out/"cost_profile.json",{"protocol":"042","revision":2,"arms":costs,
      "note":"R/A may realize different compute. Score computation uses zero extra expert forward; reuse previews and multi-active deployment remain explicit."})
    lines=["# Protocol-042 revision 2 结果","",
      "四条预登记序列按固定顺序完成：U_parent → R_win128 → A_hist → A_win128。主臂固定为 A_win128；64/256窗口仅作为只读诊断。","",
      "## 执行与主结果","",
      "- U_parent 对041 W_parent严格复现：**{}**。".format(ug.get("all_pass")),
      "- validity={}；A_win128 D>C={}；cumulative_preservation={}；pool_exercised={}；main_development_success={}。".format(valid,primary,preserve,activity["pool_exercised"],main_success),
      "- A_win128相对C：full ΔAP {}；six {}；late4 {}；prefix32 {}。".format(f(vsC["ap_delta"]),f(vsC["six_window_equal_weight_AP_delta"]),f(vsC["late4_equal_weight_AP_delta"]),f(vsC["prefix32_equal_weight_AP_delta"])),
      "- A_win128相对D_keep：full ΔAP {}；six {}；late4 {}；prefix32 {}。".format(f(vsK["ap_delta"]),f(vsK["six_window_equal_weight_AP_delta"]),f(vsK["late4_equal_weight_AP_delta"]),f(vsK["prefix32_equal_weight_AP_delta"])),
      "- 系统标签：**{}**。".format(label),"",
      "## 机制结果","",
      "- 接入策略 A_win128-R_win128：exercised={}；signal={}；full ΔAP {}；six {}；late4 {}。".format(add_ex,add_signal,f(addcmp["ap_delta"]),f(addcmp["six_window_equal_weight_AP_delta"]),f(addcmp["late4_equal_weight_AP_delta"])),
      "- 窗口策略 A_win128-A_hist：exercised={}；signal={}；标签={}；full ΔAP {}；six {}；late4 {}。".format(prefix["window_mechanism_exercised"],win_signal,mech["window_result_label"],f(wincmp["ap_delta"]),f(wincmp["six_window_equal_weight_AP_delta"]),f(wincmp["late4_equal_weight_AP_delta"])),
      "- multi_active_exercised={}（最长连续{}区间）；victim_selection_exercised={}；sleep_selection={}；replacement_selection={}。".format(activity["multi_active_exercised"],activity["multi_active_longest_run"],activity["victim_selection_exercised"],activity["sleep_selection_exercised"],activity["replacement_selection_exercised"]),"",
      "## 审计边界","",
      "- issued margin/delta/label独立评分复算及AP/BCE/混淆矩阵复算：**{}**。".format(independent_ok),
      "- 评分本身额外专家forward为0，但多活动部署、训练和休眠专家reuse preview均单独计费；R/A不是等算力比较。",
      "- 数据仍为已观察seed3601开发证据，不构成独立确认；不证明永久删除、复用因果优势或整体部署加速。",
      "- Protocol-042 revision 2 到此停止，不自动增加窗口臂、seed、F、困难加权、永久删除或Protocol-043。",""]
    Path(a.docs_output).write_text("\n".join(lines),encoding="utf8")
    print(json.dumps(J(out/"analysis_status.json"),indent=2,ensure_ascii=False))

if __name__=="__main__": main()
