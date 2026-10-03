"""Frozen Protocol-041 analysis: candidate initialization and full-system paired comparison."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

import analyze_ftmoe_protocol039 as a39
import analyze_ftmoe_protocol040 as a40
import run_ftmoe_protocol040 as p40
from protocol035_common import binary_metrics, dump_json, load_npz, phase_bounds, WINDOWS

PLAN_SHA="32f9ea891e5b6b993c07550cc2a4f37fcc55f0df9a369580366ddc01d24110ca"
LATE={"U_rec2","V_rec2","U_rec3","V_rec3"}

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def gate(v,x): return v is not None and v>=x
def f(v): return "null" if v is None else "{:+.6f}".format(float(v))

def independent_metrics(prob,y):
    p=np.asarray(prob,dtype=np.float64).reshape(-1); yy=(np.asarray(y).reshape(-1)>0).astype(np.int64)
    if not np.isfinite(p).all(): raise AssertionError("nonfinite probability")
    n=len(yy); pos=int(yy.sum()); neg=int(n-pos)
    pred=p>=.5; tp=int(np.sum(pred & (yy==1))); fp=int(np.sum(pred & (yy==0))); fn=pos-tp; tn=neg-fp
    recall=None if pos==0 else float(tp/pos); fpr=None if neg==0 else float(fp/neg)
    eps=np.finfo(np.float64).eps; pc=np.clip(p,eps,1-eps)
    bce=float(np.mean(-(yy*np.log(pc)+(1-yy)*np.log(1-pc))))
    ap=None
    if pos>0 and neg>0:
        order=np.argsort(-p,kind="mergesort"); sp=p[order]; sy=yy[order]
        cum_tp=0; cum_fp=0; acc=0.0; i=0
        while i<n:
            j=i+1
            while j<n and sp[j]==sp[i]: j+=1
            gp=int(sy[i:j].sum()); gn=(j-i)-gp
            cum_tp+=gp; cum_fp+=gn
            if gp: acc+=(cum_tp/(cum_tp+cum_fp))*(gp/pos)
            i=j
        ap=float(acc)
    return {"ap":ap,"bce":bce,"rows":n,"positives":pos,"negatives":neg,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"recall":recall,"fpr":fpr}

def independent_bundle(prob,y):
    full=independent_metrics(prob,y); win=[]; p32=[]; p64=[]
    for name,s,e in WINDOWS:
        win.append({"window":name,"metrics":independent_metrics(prob[s:e],y[s:e])})
        p32.append({"window":name,"metrics":independent_metrics(prob[s:s+32],y[s:s+32])})
        p64.append({"window":name,"metrics":independent_metrics(prob[s:s+64],y[s:s+64])})
    def mean(rows,names=None):
        xs=[r["metrics"]["ap"] for r in rows if names is None or r["window"] in names]
        return None if any(x is None for x in xs) else float(np.mean(xs))
    return {"full":full,"windows":win,"six_window_AP":mean(win),"late4_AP":mean(win,LATE),"prefix32_AP":mean(p32),"prefix64_AP":mean(p64),
            "prefix32":p32,"prefix64":p64}

def arm_quality(C,K,D,bounds,valid):
    DC=a39.cmp(C,D,bounds); DK=a39.cmp(K,D,bounds)
    primary=bool(valid and gate(DC["ap_delta"],.002) and gate(DC["six_window_equal_weight_AP_delta"],.002)
                 and DC["positive_recurrence_windows"]>=4 and gate(DC["late4_equal_weight_AP_delta"],0)
                 and gate(DC["prefix32_equal_weight_AP_delta"],-.005) and DC["guardrails"]["all_pass"])
    preserved=bool(valid and gate(DK["ap_delta"],-.001) and gate(DK["six_window_equal_weight_AP_delta"],-.001)
                   and gate(DK["late4_equal_weight_AP_delta"],-.001) and gate(DK["prefix32_equal_weight_AP_delta"],-.005)
                   and DK["guardrails"]["all_pass"])
    return DC,DK,primary,preserved

def qmetrics(q):
    return independent_metrics(q["candidate_probability"],q["labels"])

def candidate_compare(root):
    zp=root/"Z_zero/qualification_first.npz"; wp=root/"W_parent/qualification_first.npz"
    if not zp.exists() or not wp.exists():
        return {"comparable":False,"label":"mechanism_not_observed","candidate_qualification_signal":False}
    z=load_npz(zp); w=load_npz(wp)
    same=bool(np.array_equal(z["intervals"],w["intervals"]) and np.array_equal(z["labels"],w["labels"])
              and np.array_equal(z["live_probability"],w["live_probability"]) and np.array_equal(z["B_probability"],w["B_probability"]))
    if not same:
        return {"comparable":False,"label":"mechanism_not_observed","candidate_qualification_signal":False,"paired_live_B_labels_exact":False}
    Z=qmetrics(z); W=qmetrics(w); live=independent_metrics(z["live_probability"],z["labels"]); B=independent_metrics(z["B_probability"],z["labels"])
    zq=p40.qualify(z["candidate_probability"],z["live_probability"],z["B_probability"],z["labels"])
    wq=p40.qualify(w["candidate_probability"],w["live_probability"],w["B_probability"],w["labels"])
    diff=float(W["bce"]-Z["bce"]); rel=None if Z["bce"]<=0 else float((Z["bce"]-W["bce"])/Z["bce"])
    signal=bool(wq["pass"] and W["bce"]<Z["bce"])
    if signal: label="candidate_qualification_signal"
    elif W["bce"]<Z["bce"]: label="candidate_improved_but_not_qualified"
    else: label="no_candidate_improvement"
    return {"comparable":True,"paired_live_B_labels_exact":True,"intervals":[int(z["intervals"][0]),int(z["intervals"][1])],
            "Z_candidate":Z,"W_candidate":W,"common_live":live,"B":B,"Z_original_qualification":zq,"W_original_qualification":wq,
            "L_W_minus_L_Z":diff,"relative_BCE_reduction_vs_Z":rel,"candidate_qualification_signal":signal,"label":label}

def init_signal(Z,W,bounds,valid):
    wz=a39.cmp(Z,W,bounds)
    ok=bool(valid and gate(wz["ap_delta"],0) and gate(wz["six_window_equal_weight_AP_delta"],.0005)
            and wz["positive_recurrence_windows"]>=4 and gate(wz["late4_equal_weight_AP_delta"],0)
            and wz["guardrails"]["all_pass"])
    return wz,ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--science-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--docs-output",required=True); ap.add_argument("--run-id",required=True); ap.add_argument("--fixture-report",required=True)
    a=ap.parse_args(); root=Path(a.science_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    C=load_npz(root/"cached/C_ref/predictions.npz"); B=load_npz(root/"cached/B_ref/predictions.npz")
    K=load_npz(root/"cached/D_keep/predictions.npz"); D39=load_npz(root/"cached/D_039/predictions.npz"); D40=load_npz(root/"cached/D_040/predictions.npz")
    Z=load_npz(root/"Z_zero/predictions.npz"); Wp=load_npz(root/"W_parent/predictions.npz")
    for q in (B,K,D39,D40,Z,Wp):
        for key in ("labels","raw_labels","class_probability"):
            if key not in q or not np.array_equal(C[key],q[key]): raise AssertionError("alignment mismatch "+key)
    fixture=J(a.fixture_report); za=J(root/"Z_zero/run_audit.json"); wa=J(root/"W_parent/run_audit.json"); repro=J(root/"Z_zero/reproduction_report.json")
    valid=bool(fixture.get("all_pass") and za.get("all_pass") and wa.get("all_pass") and repro.get("all_pass"))
    bounds=phase_bounds(a.manifest)

    ZC,ZK,Zprimary,Zpres=arm_quality(C,K,Z,bounds,valid)
    WC,WK,Wprimary,Wpres=arm_quality(C,K,Wp,bounds,valid)
    WZ,initsig=init_signal(Z,Wp,bounds,valid)
    W40=a39.cmp(D40,Wp,bounds); Z40=a39.cmp(D40,Z,bounds); WB=a39.cmp(B,Wp,bounds); W39=a39.cmp(D39,Wp,bounds)
    cand=candidate_compare(root)
    lifecycle=J(root/"W_parent/lifecycle_events.json")["events"]
    pool=a40.pool_exercised(Wp,lifecycle)
    locals_=[a40.local_first64(B,Wp,eid) for eid in sorted(set(int(x) for x in Wp["active_id"] if int(x)>=0))]
    local_utility=None
    if len(locals_)>=2 and all(x.get("sufficient_support") for x in locals_[:2]):
        local_utility=bool(all(x.get("issued_bce_improves_B") for x in locals_[:2]))

    flags={"validity":valid,"candidate_qualification_signal":bool(cand.get("candidate_qualification_signal")),
           "initialization_system_signal":initsig,"D_over_C":Wprimary,"cumulative_preservation":Wpres,"pool_exercised":pool["pass"]}
    joint=all(flags.values())
    if not valid: system_label="invalid_execution"
    elif not Wprimary: system_label="D_over_C_not_established"
    elif not Wpres: system_label="D_over_C_with_excess_pool_loss"
    elif not pool["pass"]: system_label="quality_retained_pool_not_exercised"
    else: system_label="quality_retained_pool_exercised"

    iz=independent_bundle(Z["probability"],Z["labels"]); iw=independent_bundle(Wp["probability"],Wp["labels"])
    commonZ=binary_metrics(Z["probability"],Z["labels"]); commonW=binary_metrics(Wp["probability"],Wp["labels"])
    recompute=bool(abs(iz["full"]["ap"]-commonZ["ap"])<=1e-12 and abs(iw["full"]["ap"]-commonW["ap"])<=1e-12)
    # independently compare aggregate AP deltas too
    independent_wz={"full_AP_delta":float(iw["full"]["ap"]-iz["full"]["ap"]),
                    "six_window_AP_delta":float(iw["six_window_AP"]-iz["six_window_AP"]),
                    "late4_AP_delta":float(iw["late4_AP"]-iz["late4_AP"]),
                    "prefix32_AP_delta":float(iw["prefix32_AP"]-iz["prefix32_AP"]),
                    "prefix64_AP_delta":float(iw["prefix64_AP"]-iz["prefix64_AP"])}
    recompute=recompute and abs(independent_wz["full_AP_delta"]-WZ["ap_delta"])<=1e-12 and abs(independent_wz["six_window_AP_delta"]-WZ["six_window_equal_weight_AP_delta"])<=1e-12
    if not recompute: raise AssertionError("independent metric recomputation mismatch")

    comp={"protocol":"041","run_id":str(a.run_id),"single_stream_development_only":True,"statistical_confirmation":False,"F_research":"DEFERRED_NOT_LOADED",
          "validity":valid,"Z_reproduction":repro,"candidate_comparison":cand,
          "Z_system":{"vs_C":ZC,"vs_D_keep":ZK,"D_over_C":Zprimary,"cumulative_preservation":Zpres},
          "W_system":{"vs_C":WC,"vs_D_keep":WK,"vs_D_039":W39,"vs_D_040":W40,"vs_B":WB,
                      "D_over_C":Wprimary,"cumulative_preservation":Wpres,"pool_exercised":pool},
          "W_minus_Z":WZ,"initialization_system_signal":initsig,"local_expert_utility":locals_,"local_two_expert_utility":local_utility,
          "separate_flags":flags,"joint_development_success":joint,"system_result_label":system_label,
          "candidate_result_label":cand.get("label"),"independent_recompute":{"pass":recompute,"Z":iz,"W":iw,"W_minus_Z":independent_wz},
          "claims":{"causal_reuse_advantage":False,"complete_dynamic_deletion_success":False,"independent_confirmation":False,
                    "overall_deployment_speedup":False,"automatic_followup":False}}
    dump_json(out/"comparison.json",comp)
    dump_json(out/"analysis_status.json",{"protocol":"041","run_id":str(a.run_id),"validity":valid,"candidate_result_label":cand.get("label"),
      "system_result_label":system_label,"separate_flags":flags,"joint_development_success":joint,"publication_status":"pending_until_main_sync",
      "automatic_followup_training":False,"stop_after_registered_two_sequences":True})
    dump_json(out/"candidate_comparison.json",cand)
    dump_json(out/"independent_recompute.json",comp["independent_recompute"])
    dump_json(out/"per_expert.json",{"pool_exercised":pool,"local_first64":locals_})
    arms={"C_ref":C,"B_ref":B,"D_keep":K,"D_039":D39,"D_040":D40,"Z_zero":Z,"W_parent":Wp}
    dump_json(out/"event_diagnostics.json",{"W_parent":a40.event_diagnostics(arms,C["labels"],lifecycle)})
    dump_json(out/"cost_profile.json",{"protocol":"041","Z":J(root/"Z_zero/science_cost_raw.json"),"W":J(root/"W_parent/science_cost_raw.json"),
      "boundary":"Historical B/C/D_keep/D_039/D_040 costs are cached controls. Parent weights are copied from already-trained live E0 inside W; no extra donor gradients are hidden."})

    lines=["# Protocol-041 结果","",
      "本轮按登记顺序完成 Z_zero 与 W_parent 两条新科学序列。Z 是041预算内的完整040零初始化复现；W 唯一处理因素是在第二候选创建时复制当前活动E0权重/偏置，并使用全新 Adam。F 未加载。","",
      "## Z复现与候选层","",
      "- **Z 对缓存040逐项复现：{}**。预测数组、更新/Adam step/批次、候选/复用/生命周期离散事件按冻结 schema 核对。".format(repro.get("all_pass")),
      "- **候选结果：{}**。W候选 BCE={}，Z候选 BCE={}，L_W-L_Z={}，相对Z BCE改善={}。".format(cand.get("label"),f(cand.get("W_candidate",{}).get("bce")),f(cand.get("Z_candidate",{}).get("bce")),f(cand.get("L_W_minus_L_Z")),f(cand.get("relative_BCE_reduction_vs_Z"))),
      "- W原资格通过={}；Z原资格通过={}；candidate_qualification_signal={}。".format(cand.get("W_original_qualification",{}).get("pass"),cand.get("Z_original_qualification",{}).get("pass"),flags["candidate_qualification_signal"]),"",
      "## 系统层","",
      "- W相对C：D_over_C={}；full ΔAP {}；六窗 {}；late4 {}；prefix32 {}。".format(Wprimary,f(WC["ap_delta"]),f(WC["six_window_equal_weight_AP_delta"]),f(WC["late4_equal_weight_AP_delta"]),f(WC["prefix32_equal_weight_AP_delta"])),
      "- W相对D_keep：累计保持={}；full ΔAP {}；六窗 {}；late4 {}。".format(Wpres,f(WK["ap_delta"]),f(WK["six_window_equal_weight_AP_delta"]),f(WK["late4_equal_weight_AP_delta"])),
      "- W-Z 初始化系统信号={}；full ΔAP {}；六窗 {}；late4 {}；正向窗 {}/6。".format(initsig,f(WZ["ap_delta"]),f(WZ["six_window_equal_weight_AP_delta"]),f(WZ["late4_equal_weight_AP_delta"]),WZ["positive_recurrence_windows"]),
      "- 双专家池 exercised={}；accepted IDs={}；active counts={}。".format(pool["pass"],pool["accepted_distinct_ids"],pool["active_prediction_counts"]),"",
      "## 六个独立判定","",
      "- validity={}；candidate_qualification_signal={}；initialization_system_signal={}；D_over_C={}；cumulative_preservation={}；pool_exercised={}。".format(flags["validity"],flags["candidate_qualification_signal"],flags["initialization_system_signal"],flags["D_over_C"],flags["cumulative_preservation"],flags["pool_exercised"]),
      "- **041_joint_development_success={}**。系统标签 **{}**。".format(joint,system_label),"",
      "## 有效性与边界","",
      "- 独立AP/BCE/混淆矩阵重算一致：{}。".format(recompute),
      "- 结果仍是已观察 seed3601 的开发证据；不构成跨流/多seed确认，不证明复用因果收益、永久删除有效或整体端到端加速。",
      "- Protocol-041 到此停止；不追加重试、第三初始化臂、训练长度、阈值、seed/新流或F研究。",""]
    Path(a.docs_output).write_text("\n".join(lines),encoding="utf8")
    print(json.dumps(J(out/"analysis_status.json"),indent=2,ensure_ascii=False))

if __name__=="__main__": main()
