"""Protocol-044 revision2 execution analysis; scientific thresholds remain revision1."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def JL(p):
    p=Path(p)
    return [] if not p.exists() else [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x.strip()]
def NP(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def ap(y,p):
    y=(np.asarray(y).reshape(-1)>0).astype(np.int64);p=np.asarray(p,dtype=np.float64).reshape(-1)
    pos=int(y.sum())
    if pos==0 or pos==len(y):return None
    o=np.argsort(-p,kind="mergesort");y=y[o];p=p[o];tp=fp=i=0;score=0.0
    while i<len(y):
        j=i+1
        while j<len(y) and p[j]==p[i]:j+=1
        g=int(y[i:j].sum());tp+=g;fp+=j-i-g
        if g:score+=tp/(tp+fp)*g
        i=j
    return float(score/pos)
def met(y,p):
    y=np.asarray(y).reshape(-1)>0;p=np.asarray(p,dtype=np.float64).reshape(-1);q=p>=.5
    tp=int(np.sum(q&y));fp=int(np.sum(q&~y));fn=int(np.sum(~q&y));tn=int(np.sum(~q&~y))
    return {"ap":ap(y,p),"tp":tp,"fp":fp,"fn":fn,"tn":tn,
      "recall":None if tp+fn==0 else tp/(tp+fn),"fpr":None if fp+tn==0 else fp/(fp+tn)}
def bce(m,y):
    m=np.asarray(m,dtype=np.float64);y=(np.asarray(y)>0).astype(np.float64)
    return np.maximum(m,0)-m*y+np.log1p(np.exp(-np.abs(m)))
def conf(m,y):
    q=np.asarray(m,dtype=np.float64)>=0;y=np.asarray(y)>0
    return {"tp":int(np.sum(q&y)),"fp":int(np.sum(q&~y)),"fn":int(np.sum(~q&y)),"tn":int(np.sum(~q&~y))}
def rt(a,b):return None if b==0 else a/b
def close(a,b,tol=1e-10):
    if a is None or b is None:return a is None and b is None
    return abs(float(a)-float(b))<=tol
def equal(a,b):
    return bool(np.array_equal(a,b,equal_nan=True)) if np.issubdtype(a.dtype,np.floating) else bool(np.array_equal(a,b))
def qualify(cm,lm,bm,y):
    cm=np.asarray(cm,np.float64);lm=np.asarray(lm,np.float64);bm=np.asarray(bm,np.float64);y=np.asarray(y)>0
    pos=int(y.sum());neg=int(y.size-pos);finite=bool(np.isfinite(cm).all() and np.isfinite(lm).all() and np.isfinite(bm).all());support=pos>=16 and neg>=16
    if not finite or not support:return {"pass":False,"support":support,"finite":finite}
    def M(m):
        c=conf(m,y);return {"bce":float(bce(m,y).mean()),"fpr":rt(c["fp"],c["fp"]+c["tn"]),"recall":rt(c["tp"],c["tp"]+c["fn"])}
    ca,li,ba=M(cm),M(lm),M(bm)
    checks={"bce_vs_live":ca["bce"]<=.99*li["bce"],"bce_vs_B":ca["bce"]<=ba["bce"],
      "fpr_vs_live":ca["fpr"]-li["fpr"]<=.01,"fpr_vs_B":ca["fpr"]-ba["fpr"]<=.01,
      "recall_vs_live":ca["recall"]-li["recall"]>=-.02,"recall_vs_B":ca["recall"]-ba["recall"]>=-.02}
    return {"pass":bool(all(checks.values())),"support":True,"finite":True,"candidate":ca,"live":li,"B":ba,"checks":checks}
def mm(root,name):return np.load(Path(root)/"audit_arrays"/(name+".npy"),mmap_mode="r")
def maxrun(active,eid):
    z=np.any(np.asarray(active)==int(eid),axis=1);cur=best=0
    for v in z:cur=cur+1 if v else 0;best=max(best,cur)
    return int(best)
def utility_audit(root,pred):
    root=Path(root);live=mm(root,"live_margin");dep=mm(root,"deployment_epoch");con=mm(root,"contribution");y=pred["labels"]
    rows=JL(root/"streams/utility_checks.jsonl");sleeps=JL(root/"streams/sleep_tables.jsonl");streak={};bad=[]
    for r in rows:
        t=int(r["at_interval"]);e=int(r["expert_id"]);m=t-2;ep=int(r["epoch"]);idx=np.arange(m-127,m+1) if m>=127 else np.array([],dtype=int)
        valid=len(idx)==128 and bool(np.all(np.asarray(dep[idx])==ep)) and bool(np.isfinite(np.asarray(con[e,idx])).all())
        z={"valid":False}
        if valid:
            lm=np.asarray(live[idx],np.float64);dd=np.asarray(con[e,idx],np.float64);yy=np.asarray(y[idx])>0;u=bce(lm-dd,yy)-bce(lm,yy)
            flat=u.reshape(-1);pos=yy.reshape(-1);neg=~pos;pn=int(pos.sum());nn=int(neg.sum());pi=int(sum(bool(np.any(x)) for x in yy))
            valid=pn>=16 and nn>=16 and pi>=4
            if valid:
                a=conf(lm,yy);b=conf(lm-dd,yy);lf=rt(a["fp"],a["fp"]+a["tn"]);rf=rt(b["fp"],b["fp"]+b["tn"]);lr=rt(a["tp"],a["tp"]+a["fn"]);rr=rt(b["tp"],b["tp"]+b["fn"])
                z={"valid":True,"score":float(flat.mean()),"score_pos":float(flat[pos].mean()),"score_neg":float(flat[neg].mean()),"removal_fpr_delta":rf-lf,"removal_recall_delta":rr-lr}
        ok=bool(z["valid"]==bool(r.get("valid")))
        if z["valid"]:
            ok &= all(close(z[k],r.get(k)) for k in ("score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta"))
        elig=bool(z["valid"] and z["score"]<=0 and z["score_pos"]<=0 and z["score_neg"]<=0 and z["removal_fpr_delta"]<=.01 and z["removal_recall_delta"]>=-.02)
        key=(ep,e);before=streak.get(key,0);after=before+1 if elig else 0;streak[key]=after
        ok &= bool(r.get("eligible_now"))==elig and int(r.get("streak_before",before))==before and int(r.get("streak_after",after))==after and bool(r.get("eligible_three"))==bool(elig and after>=3)
        if not ok:bad.append({"t":t,"expert_id":e})
    sleep_ok=True
    for r in sleeps:
        c=[x for x in r.get("rows",[]) if x.get("eligible_three")]
        want=None if not c else int(sorted(c,key=lambda x:(float(x["score"]),int(x["expert_id"])))[0]["expert_id"])
        got=r.get("selected");sleep_ok &= (got is None and want is None) or (got is not None and int(got)==want)
    return {"checks":len(rows),"score_pass":not bad,"sleep_selection_pass":bool(sleep_ok),"all_pass":bool(not bad and sleep_ok),"failures":bad[:20]}
def qualification_audit(root):
    root=Path(root);raw=JL(root/"streams/qualification_raw.jsonl");cand=JL(root/"streams/candidate_decisions.jsonl");reuse=JL(root/"streams/reuse_decisions.jsonl")
    cb={(int(x["at_interval"]),int(x["candidate_id"])):x for x in cand if x.get("event")=="shadow_evaluate"};rb={int(x["at_interval"]):x for x in reuse if x.get("event")=="reuse_evaluate"}
    bad=[];rows=[]
    for x in raw:
        if x["kind"]=="shadow":
            q=qualify(x["candidate_margin"],x["live_margin"],x["B_margin"],x["labels"]);r=cb.get((int(x["at_interval"]),int(x["candidate_id"])))
            ok=r is not None and bool(r.get("pass"))==q["pass"] and (not q.get("support") or (close(r["candidate"]["bce"],q["candidate"]["bce"]) and close(r["live"]["bce"],q["live"]["bce"]) and close(r["B"]["bce"],q["B"]["bce"]) and r["checks"]==q["checks"]))
            rows.append({"kind":"shadow","t":x["at_interval"],"id":x["candidate_id"],"pass":bool(ok)})
        else:
            r=rb.get(int(x["at_interval"]));calc=[];passing=[]
            for e,cm in x["candidate_margin"].items():
                q=qualify(cm,x["live_margin"],x["B_margin"],x["labels"]);lm=np.asarray(x["live_margin"],np.float64);cc=np.asarray(cm,np.float64);yy=np.asarray(x["labels"])>0;u=bce(lm,yy)-bce(cc,yy);p=yy.reshape(-1);n=~p
                q.update({"id":int(e),"Q":float(u.mean()),"Q_pos":float(u.reshape(-1)[p].mean()),"Q_neg":float(u.reshape(-1)[n].mean())});calc.append(q)
                if q["pass"]:passing.append(q)
            win=None if not passing else int(sorted(passing,key=lambda q:(q["candidate"]["bce"],q["id"]))[0]["id"])
            ok=r is not None and r.get("winner")==win and len(r.get("candidates",[]))==len(calc)
            if ok:
                old={int(q["candidate_id"]):q for q in r["candidates"]}
                for q in calc:
                    z=old[q["id"]];ok &= bool(z["pass"])==q["pass"] and close(z["Q"],q["Q"]) and close(z["Q_pos"],q["Q_pos"]) and close(z["Q_neg"],q["Q_neg"])
            rows.append({"kind":"reuse","t":x["at_interval"],"winner":win,"pass":bool(ok)})
        if not rows[-1]["pass"]:bad.append(rows[-1])
    # Any production evaluate event must have one raw record.
    eval_count=len(cb)+len(rb);coverage=len(raw)==eval_count
    return {"evaluations":len(raw),"expected_evaluations":eval_count,"coverage_pass":coverage,"all_pass":bool(coverage and not bad),"failures":bad[:20]}
def arm_audit(root,pred,bpred):
    root=Path(root);sm=J(root/"summary.json");life=JL(root/"streams/lifecycle.jsonl");cand=JL(root/"streams/candidate_decisions.jsonl");recl=JL(root/"streams/reclamation.jsonl")
    accepted=[x for x in cand if x.get("event")=="shadow_evaluate" and x.get("pass")];post=[x for x in accepted if int(x["candidate_id"])>0]
    active64=any(maxrun(pred["active_ids"],int(x["candidate_id"]))>=64 for x in post)
    sleeps=[x for x in life if x.get("event")=="sleep"];dormant64=False
    for x in sleeps:
        e=int(x["expert_id"]);start=int(x["dormant_from_prediction"]);react=next((int(z["at_interval"])+1 for z in life if z.get("event")=="activate" and int(z.get("expert_id",-1))==e and int(z["at_interval"])>int(x["at_interval"])),len(pred["probability"]))
        dormant64 |= react-start>=64
    reuse=[x for x in life if x.get("event")=="activate" and x.get("reason")=="reuse_accept"];reuseok=any(int(x.get("dormant_predictions_before_reactivation",0))>=64 and maxrun(pred["active_ids"],int(x["expert_id"]))>=64 for x in reuse)
    gc=[x for x in recl if x.get("event")=="permanent_reclaim_and_shadow_start"];closed=False
    for g in gc:
        cid=int(g["candidate_id"]);q=next((x for x in accepted if int(x["candidate_id"])==cid and x.get("created_by_reclamation")),None);closed |= q is not None and maxrun(pred["active_ids"],cid)>=64
    resource=bool(sm["max_active_seen"]<=2 and sm["max_resident_seen"]<=3 and sm["ids_created"]<=5 and sm["permanent_deletions"]<=4 and sm["actual_optimizer_calls"]<=808 and sm["deployed_forwards"]<=11904 and sm["reuse_preview_forwards"]<=3072 and sm["shadow_preview_forwards"]<=128 and not any(sm["terminal_counter_delta"]))
    return {"summary":sm,"utility":utility_audit(root,pred),"qualification":qualification_audit(root),"lifecycle_on_stream":bool(active64 and dormant64),"reuse_on_stream":bool(reuseok),"gc_on_stream":bool(gc),"gc_admission_closed_loop":bool(closed),"resource_pass":resource,"accepted":accepted,"sleeps":sleeps,"reuse_accept":reuse,"gc_events":gc}
def divergence(no,bo,A,B):
    gc=[x for x in JL(Path(bo)/"streams/reclamation.jsonl") if x.get("event")=="permanent_reclaim_and_shadow_start"]
    keys=("probability","detection_logits","total_delta","live_margin","B_margin","deployment_epoch","active_ids","active_count","shadow_present","accepted_count","expert_versions","expert_hashes")
    if not gc:
        rows={k:equal(A[k],B[k]) for k in keys};return {"gc_exercised":False,"whole_trajectory_exact":bool(all(rows.values())),"array_exact":rows,"pass":bool(all(rows.values()))}
    t=min(int(x["at_interval"]) for x in gc);rows={k:equal(A[k][:t+1],B[k][:t+1]) for k in keys}
    return {"gc_exercised":True,"first_gc_control_t":t,"pre_gc_exact":bool(all(rows.values())),"array_exact":rows,"pass":bool(all(rows.values()))}
def main():
    apx=argparse.ArgumentParser();apx.add_argument("--run-root",required=True);apx.add_argument("--data",required=True);apx.add_argument("--input-lock",required=True);apx.add_argument("--e-gate",required=True);apx.add_argument("--out",required=True);apx.add_argument("--docs",required=True);apx.add_argument("--run-id",required=True);a=apx.parse_args()
    root=Path(a.run_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);plan=J("artifacts/ftmoe_online/protocol_044/plan.json");lock=J(a.input_lock);eg=J(a.e_gate);man=J(Path(a.data)/"manifest.json");led=J(root/"budget_ledger.json")
    C=NP(root/"C_ref/predictions.npz");B=NP(root/"D_lin/predictions.npz");N=NP(root/"D_no_gc/predictions.npz");D=NP(root/"D_bounded/predictions.npz");y=C["labels"]
    windows=[(x[0],int(x[1]),int(x[2])) for x in plan["analysis"]["primary_windows"]];stages=[(x[0],int(x[1]),int(x[2])) for x in plan["analysis"]["all8_nonbaseline_stages"]]
    fullC=met(y,C["probability"]);fullD=met(y,D["probability"]);primary=[];prefix=[]
    for name,s,e in windows:
        c=met(y[s:e],C["probability"][s:e]);d=met(y[s:e],D["probability"][s:e]);primary.append({"name":name,"range":[s,e],"C":c,"D":d,"delta_ap":d["ap"]-c["ap"],"delta_recall":d["recall"]-c["recall"],"delta_fpr":d["fpr"]-c["fpr"]})
        for L in (32,64,128):
            c=met(y[s:s+L],C["probability"][s:s+L]);d=met(y[s:s+L],D["probability"][s:s+L]);prefix.append({"name":name,"length":L,"delta_ap":d["ap"]-c["ap"],"delta_recall":d["recall"]-c["recall"],"C":c,"D":d})
    stage=[]
    for name,s,e in stages:
        c=met(y[s:e],C["probability"][s:e]);d=met(y[s:e],D["probability"][s:e]);stage.append({"name":name,"delta_ap":d["ap"]-c["ap"],"C":c,"D":d})
    idx=np.concatenate([np.arange(s,e) for _,s,e in windows]);pc=met(y[idx],C["probability"][idx]);pd=met(y[idx],D["probability"][idx])
    r4=[x["delta_ap"] for x in primary];p32=[x["delta_ap"] for x in prefix if x["length"]==32]
    perf={"full_C":fullC,"full_D":fullD,"full_delta_ap":fullD["ap"]-fullC["ap"],"primary":primary,"prefix":prefix,"all8":stage,
      "return4_mean_delta_ap":float(np.mean(r4)),"positive_primary_windows":int(sum(x>0 for x in r4)),"late2_mean_delta_ap":float(np.mean(r4[2:])),
      "prefix32_mean_delta_ap":float(np.mean(p32)),"full_delta_fpr":fullD["fpr"]-fullC["fpr"],"full_delta_recall":fullD["recall"]-fullC["recall"],
      "pooled_return4_delta_fpr":pd["fpr"]-pc["fpr"],"pooled_return4_delta_recall":pd["recall"]-pc["recall"],"all8_mean_delta_ap":float(np.mean([x["delta_ap"] for x in stage]))}
    t=plan["analysis"]["D_over_C"]
    Dgt=bool(perf["full_delta_ap"]>=t["full_AP_delta_min"] and perf["return4_mean_delta_ap"]>=t["return4_mean_AP_delta_min"] and perf["positive_primary_windows"]>=t["positive_primary_windows_min"] and perf["late2_mean_delta_ap"]>=t["late2_mean_AP_delta_min"] and perf["prefix32_mean_delta_ap"]>=t["prefix32_mean_AP_delta_min"] and perf["full_delta_fpr"]<=t["full_and_return4_pooled_FPR_delta_max"] and perf["pooled_return4_delta_fpr"]<=t["full_and_return4_pooled_FPR_delta_max"] and perf["full_delta_recall"]>=t["full_and_return4_pooled_recall_delta_min"] and perf["pooled_return4_delta_recall"]>=t["full_and_return4_pooled_recall_delta_min"] and all(x["delta_recall"]>=t["each_primary_window_recall_delta_min"] for x in primary) and all(x["delta_recall"]>=t["each_prefix32_recall_delta_min"] for x in prefix if x["length"]==32) and perf["all8_mean_delta_ap"]>=t["all8_stage_mean_AP_delta_min"])
    na=arm_audit(root/"D_no_gc",N,B);da=arm_audit(root/"D_bounded",D,B);div=divergence(root/"D_no_gc",root/"D_bounded",N,D)
    budget=bool(led.get("protocol")=="044" and led.get("revision")==1 and led.get("execution_revision")==2 and
      led.get("total_optimizer_steps_used",10**9)<=2360 and
      int(led.get("total_optimizer_steps_used",-1))==sum(int(led["sequences"][x].get("optimizer_steps_used",0)) for x in ("C_ref","D_lin","D_no_gc","D_bounded")) and
      all(led["sequences"][x].get("completed") for x in ("C_ref","D_lin","D_no_gc","D_bounded")))
    inp=bool(lock.get("locked") is True and lock.get("stream_sha256")==man.get("stream_sha256") and man.get("audit_pass") is True)
    independent=bool(na["utility"]["all_pass"] and na["qualification"]["all_pass"] and da["utility"]["all_pass"] and da["qualification"]["all_pass"])
    valid=bool(eg.get("E_recovery_gate") is True and inp and budget and independent and div["pass"] and na["resource_pass"] and da["resource_pass"])
    core=bool(valid and Dgt and da["lifecycle_on_stream"] and da["resource_pass"]);full=bool(eg.get("E_recovery_gate") is True and core and da["gc_admission_closed_loop"])
    status={"protocol":"044","revision":1,"execution_revision":2,"science_config_revision":1,"run_id":str(a.run_id),"E_gate":bool(eg.get("E_recovery_gate")),"input_gate":inp,"budget_pass":budget,"independent_audit_pass":independent,"divergence_pass":div["pass"],"S_validity":valid,"D_over_C":Dgt,"lifecycle_on_stream":da["lifecycle_on_stream"],"reuse_on_stream":da["reuse_on_stream"],"gc_on_stream":da["gc_on_stream"],"gc_admission_closed_loop":da["gc_admission_closed_loop"],"resource_pass":da["resource_pass"],"new_stream_core_supported":core,"full_lifecycle_goal_completed":full,"system_result_label":"full_lifecycle_goal_completed" if full else ("core_supported_gc_unexercised_or_unclosed" if core else ("valid_but_core_not_supported" if valid else "invalid_execution"))}
    W(out/"performance.json",perf);W(out/"D_no_gc_audit.json",na);W(out/"D_bounded_audit.json",da);W(out/"divergence.json",div);W(out/"analysis_status.json",status)
    Path(a.docs).write_text("# Protocol-044 revision 1 结果\\n\\nscience run "+str(a.run_id)+"。\\n\\n## 同流性能\\n\\nD_bounded 相对 C_ref：全程 ΔAP=%+.6f；四个 return128 等权 ΔAP=%+.6f；正向窗口 %d/4；后两个 return128 平均=%+.6f；prefix32 平均=%+.6f。D_over_C=%s。\\n\\n## 生命周期\\n\\nlifecycle_on_stream=%s；reuse_on_stream=%s；gc_on_stream=%s；gc_admission_closed_loop=%s。\\n\\n## 工程与结论\\n\\nS_validity=%s；independent_audit_pass=%s；resource_pass=%s；optimizer.step=%s/2360。最终标签：%s。若本轮没有自然 GC，则只报告未覆盖，不把合成 fixture 当作真实流回收。\\n"%(
      perf["full_delta_ap"],perf["return4_mean_delta_ap"],perf["positive_primary_windows"],perf["late2_mean_delta_ap"],perf["prefix32_mean_delta_ap"],Dgt,
      da["lifecycle_on_stream"],da["reuse_on_stream"],da["gc_on_stream"],da["gc_admission_closed_loop"],valid,independent,da["resource_pass"],led.get("total_optimizer_steps_used"),status["system_result_label"]),encoding="utf8")
    print(json.dumps(status,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
