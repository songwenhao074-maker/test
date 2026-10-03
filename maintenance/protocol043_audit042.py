"""Protocol-043 read-only independent audit of sealed Protocol-042 outputs."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def L(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def bce_rows(m,y):
    m=np.asarray(m,dtype=np.float64); yy=(np.asarray(y)>0).astype(np.float64)
    return np.maximum(m,0)-m*yy+np.log1p(np.exp(-np.abs(m)))
def confusion(m,y):
    yy=(np.asarray(y)>0); pr=np.asarray(m)>=0
    tp=int(np.sum(pr&yy)); fp=int(np.sum(pr&~yy)); fn=int(np.sum(~pr&yy)); tn=int(np.sum(~pr&~yy))
    rec=None if tp+fn==0 else tp/(tp+fn); fpr=None if fp+tn==0 else fp/(fp+tn)
    return tp,fp,fn,tn,rec,fpr
def score(pred,eid,m,legacy):
    ep=int(pred["deployment_epoch"][m]); lo=m-127
    if lo<0: return {"valid":False,"reason":"window_not_full"}
    ii=np.arange(lo,m+1)
    if np.any(pred["deployment_epoch"][ii]!=ep): return {"valid":False,"reason":"window_cross_epoch"}
    d=pred["contribution"][int(eid),ii]
    if not np.isfinite(d).all(): return {"valid":False,"reason":"missing_contribution"}
    live=np.asarray(pred["live_margin"][ii],dtype=np.float32); y=(pred["labels"][ii]>0)
    if legacy:
        rem=(live-np.asarray(d,dtype=np.float32)).astype(np.float32)
        ll=bce_rows(live,y); rl=bce_rows(rem,y)
    else:
        l64=live.astype(np.float64); d64=np.asarray(d,dtype=np.float32).astype(np.float64); rem=l64-d64
        ll=bce_rows(l64,y); rl=bce_rows(rem,y)
    u=rl-ll; flat=u.reshape(-1); fy=y.reshape(-1); pos=fy; neg=~fy
    pn=int(pos.sum()); nn=int(neg.sum()); pint=int(sum(bool(np.any(r)) for r in y))
    if pn<16 or nn<16 or pint<4:
        return {"valid":False,"reason":"insufficient_support","positive_host_rows":pn,"negative_host_rows":nn,"positive_intervals":pint}
    _,_,_,_,lrec,lfpr=confusion(live,y); _,_,_,_,rrec,rfpr=confusion(rem,y)
    return {"valid":True,"reason":"ok","score":float(flat.mean()),"score_pos":float(flat[pos].mean()),"score_neg":float(flat[neg].mean()),
      "positive_host_rows":pn,"negative_host_rows":nn,"positive_intervals":pint,
      "removal_fpr_delta":float(rfpr-lfpr),"removal_recall_delta":float(rrec-lrec),
      "fpr_live":lfpr,"fpr_removed":rfpr,"recall_live":lrec,"recall_removed":rrec,
      "first_i":int(ii[0]),"last_i":int(ii[-1])}
def eligible(s): return bool(s.get("valid") and s["score"]<=0 and s["score_pos"]<=0 and s["score_neg"]<=0 and s["removal_fpr_delta"]<=.01 and s["removal_recall_delta"]>=-.02)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--raw042",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    root=Path(a.raw042); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pred=L(root/"A_win128/predictions.npz"); checks=J(root/"A_win128/utility_checks.json")["checks"]
    victims=J(root/"A_win128/victim_tables.json")["tables"]; life=J(root/"A_win128/lifecycle_events.json")["events"]
    rows=[]; legacy_mismatch=[]; numeric_diffs=[]; norm_by_t={}; streak={}; last_epoch=None
    for idx,r in enumerate(checks):
        t=int(r["at_interval"]); eid=int(r["expert_id"]); m=t-2; ep=int(r["epoch"])
        if ep!=last_epoch:
            streak={}; last_epoch=ep
        lg=score(pred,eid,m,True); rg=score(pred,eid,m,False)
        before=int(streak.get(eid,0)); ok=eligible(rg); after=before+1 if ok else 0; streak[eid]=after
        nr={**rg,"eligible_now":ok,"streak_before":before,"streak_after":after,"eligible_three":bool(ok and after>=3)}
        item={"index":idx,"at_interval":t,"epoch":ep,"expert_id":eid,"published":r,"legacy_float32":lg,"registered_float64":nr}
        rows.append(item); norm_by_t.setdefault(t,[]).append({"expert_id":eid,**nr})
        # Old runtime path should numerically match its own sealed row; do not demand exact JSON float identity.
        fields=["score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta"]
        lm=True
        for k in fields:
            x=lg.get(k); y=r.get(k)
            if (x is None)!=(y is None) or (x is not None and abs(float(x)-float(y))>1e-10): lm=False
        if not lm: legacy_mismatch.append({"index":idx,"at_interval":t,"expert_id":eid,"legacy":lg,"published":r})
        if lg.get("valid") and rg.get("valid"):
            numeric_diffs.append({"at_interval":t,"expert_id":eid,
              "score_diff":float(rg["score"]-lg["score"]),"score_pos_diff":float(rg["score_pos"]-lg["score_pos"]),"score_neg_diff":float(rg["score_neg"]-lg["score_neg"])})
    actual={int(x["at_interval"]):x.get("selected") for x in victims}
    discrete=[]; first=None
    for t in sorted(norm_by_t):
        rr=norm_by_t[t]; cand=[x for x in rr if x.get("eligible_three")]
        sel=None if not cand else int(sorted(cand,key=lambda x:(float(x["score"]),int(x["expert_id"])))[0]["expert_id"])
        act=actual.get(t)
        same=(None if act is None else int(act))==sel
        d={"at_interval":t,"registered_selected":sel,"published_selected":act,"same":bool(same),"rows":rr}; discrete.append(d)
        if not same and first is None: first=d
    sleeps=[x for x in life if x.get("event")=="sleep"]
    missing=[{"index":i,"event":x} for i,x in enumerate(sleeps) if "active_ids_before" not in x or "active_ids_after" not in x]
    boundary={}
    target=next((x for x in sleeps if int(x.get("at_interval",-1))==1615),None)
    if target is not None:
        boundary={"sleep_event":target,"active_ids_prediction_1615":pred["active_ids"][1615].astype(int).tolist(),
          "active_ids_prediction_1616":pred["active_ids"][1616].astype(int).tolist(),
          "control_at_1615_first_affected_prediction":1616,
          "control_difference_may_precede_first_output_difference_by_one_tick":True}
    maxdiff=max([abs(x["score_diff"]) for x in numeric_diffs] or [0.0])
    report={"protocol":"043","revision":1,"source_protocol":"042","source_run_id":37128097274,
      "original042_validity_remains_unchanged":True,"control_points_checked":len(rows),
      "legacy_float32_matches_published_count":len(rows)-len(legacy_mismatch),"legacy_mismatch_count":len(legacy_mismatch),
      "max_abs_registered_minus_legacy_score":maxdiff,"first_registered_discrete_difference":first,
      "subsequent_counterfactual_closed_loop_claim_forbidden":first is not None,
      "sleep_event_count":len(sleeps),"sleep_schema_missing_active_ids_count":len(missing),
      "boundary_1615_1616":boundary,"all_control_points_included":True,
      "notes":["legacy_float32 reproduces old arithmetic only","registered_float64 is a fixed-output audit, not a replayed corrected trajectory",
               "after the first discrete action difference no alternate closed-loop trajectory is inferred"]}
    W(out/"audit_summary.json",report); W(out/"all_control_points.json",{"rows":rows}); W(out/"discrete_selection_audit.json",{"rows":discrete})
    W(out/"legacy_mismatches.json",{"rows":legacy_mismatch}); W(out/"sleep_schema_gaps.json",{"rows":missing}); W(out/"numeric_dtype_differences.json",{"rows":numeric_diffs})
    # Audit completion is about source coverage/diagnosis, not making old validity true.
    gate={"protocol":"043","revision":1,"audit042_complete":bool(len(rows)>0 and len(discrete)>0 and boundary),
      "old_report_overwritten":False,"gradients":0,"model_replay":False,"all_pass":bool(len(rows)>0 and len(discrete)>0 and boundary)}
    W(out/"gate.json",gate); print(json.dumps({"audit042_complete":gate["all_pass"],"control_points":len(rows),"first_difference":first},indent=2))
    if not gate["all_pass"]: raise RuntimeError("audit042_incomplete")
if __name__=="__main__": main()
