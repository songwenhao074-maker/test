"""Frozen Protocol-040 analysis. No F inputs are loaded or compared."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import analyze_ftmoe_protocol039 as a39
from protocol035_common import binary_metrics, dump_json, load_npz, phase_bounds

PLAN_SHA="97839cb55ea9d00521e15529e5b5505004c69fd9587d2d22503b7b1ff99dcd35"

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def f(v): return "null" if v is None else "{:+.6f}".format(float(v))
def gate(v,x): return v is not None and v>=x

def contiguous_active_after(ids,eid,start):
    n=len(ids); j=int(start)
    while j<n and int(ids[j])==int(eid): j+=1
    return j-int(start)

def local_first64(B,D,eid):
    ids=D["active_id"]; pos=np.flatnonzero(ids==int(eid))
    if len(pos)==0: return {"expert_id":int(eid),"observed":False}
    s=int(pos[0]); e=min(len(ids),s+64)
    contiguous=bool(e-s==64 and np.all(ids[s:e]==int(eid)))
    if not contiguous: return {"expert_id":int(eid),"observed":True,"start":s,"censored":True,"contiguous_length":contiguous_active_after(ids,eid,s)}
    y=B["labels"][s:e]; support={"positive":int((y>0).sum()),"negative":int((y<=0).sum())}
    bm=binary_metrics(B["probability"][s:e],y); dm=binary_metrics(D["probability"][s:e],y)
    sufficient=support["positive"]>=16 and support["negative"]>=16
    return {"expert_id":int(eid),"observed":True,"start":s,"end":e,"censored":False,"support":support,"sufficient_support":sufficient,
            "B":bm,"expert_system":dm,"issued_bce_improves_B":bool(sufficient and dm["bce"]<bm["bce"])}

def pool_exercised(D,lifecycle):
    ids=D["active_id"]; accepted=sorted(set(int(x) for x in ids if int(x)>=0))
    active_counts={i:int(np.sum(ids==i)) for i in accepted}
    reactivations=[]
    for ev in lifecycle:
        if ev.get("event")!="activate": continue
        dormant=int(ev.get("dormant_predictions_before_reactivation",0) or 0)
        if dormant<64: continue
        eid=int(ev["expert_id"]); start=int(ev["first_affected_prediction"])
        post=contiguous_active_after(ids,eid,start)
        reactivations.append({"expert_id":eid,"at_interval":ev["at_interval"],"dormant_predictions":dormant,"post_reactivation_contiguous_active_predictions":post,
                              "passes":post>=64})
    ok=bool(len(accepted)>=2 and all(active_counts.get(i,0)>=64 for i in accepted[:2]) and any(x["passes"] for x in reactivations))
    return {"accepted_distinct_ids":accepted,"active_prediction_counts":active_counts,"qualifying_reactivations":reactivations,"pass":ok}

def event_diagnostics(arms,labels,events):
    out=[]
    for ev in events:
        if ev.get("event") not in ("sleep","activate","shadow_accept"): continue
        t=int(ev["at_interval"]); row={"event":ev.get("event"),"reason":ev.get("reason"),"expert_id":ev.get("expert_id"),"at_interval":t,
                                       "first_affected_prediction":int(ev.get("first_affected_prediction",t+1))}
        s=max(0,t-63); row["before64"]=a39.event_block(arms,labels,s,t+1)
        aft=[]
        for L in (32,64,128):
            ss=t+1; ee=min(len(labels),ss+L); aft.append({"requested_length":L,**a39.event_block(arms,labels,ss,ee)})
        row["after"]=aft; out.append(row)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--science-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--docs-output",required=True); ap.add_argument("--run-id",required=True)
    a=ap.parse_args(); root=Path(a.science_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)

    C=load_npz(root/"C_ref/predictions.npz"); B=load_npz(root/"B_ref/predictions.npz")
    K=load_npz(root/"D_keep/predictions.npz"); D39=load_npz(root/"D_039/predictions.npz"); D=load_npz(root/"D_pool2/predictions.npz")
    for q in (B,K,D39,D):
        for key in ("labels","raw_labels","class_probability"):
            if key not in q or not np.array_equal(C[key],q[key]): raise AssertionError("five-arm alignment mismatch: "+key)
    bounds=phase_bounds(a.manifest)
    audit=J(root/"run_audit.json"); fixture=J(root/"fixture_report.json"); ledger=J(root/"budget_ledger.json")
    sm=J(root/"D_pool2/summary.json"); lifecycle=J(root/"D_pool2/lifecycle_events.json")["events"]
    reuse=J(root/"D_pool2/reuse_decisions.json")["decisions"]; candidates=J(root/"D_pool2/candidate_decisions.json")["decisions"]
    prov=J(root/"implementation_manifest.json")
    validity=bool(audit.get("all_pass") and fixture.get("all_pass") and ledger.get("sequence",{}).get("completed")
                  and prov.get("protocol")=="040" and prov.get("plan_sha256")==PLAN_SHA and prov.get("checkout_sha"))

    DC=a39.cmp(C,D,bounds); DK=a39.cmp(K,D,bounds); D39c=a39.cmp(D39,D,bounds); DB=a39.cmp(B,D,bounds); BC=a39.cmp(C,B,bounds)
    primary=bool(validity and gate(DC["ap_delta"],.002) and gate(DC["six_window_equal_weight_AP_delta"],.002)
                 and DC["positive_recurrence_windows"]>=4 and gate(DC["late4_equal_weight_AP_delta"],0)
                 and gate(DC["prefix32_equal_weight_AP_delta"],-.005) and DC["guardrails"]["all_pass"])
    preserved=bool(validity and gate(DK["ap_delta"],-.001) and gate(DK["six_window_equal_weight_AP_delta"],-.001)
                   and gate(DK["late4_equal_weight_AP_delta"],-.001) and gate(DK["prefix32_equal_weight_AP_delta"],-.005)
                   and DK["guardrails"]["all_pass"])

    pool=pool_exercised(D,lifecycle)
    local=[local_first64(B,D,eid) for eid in sorted(set(int(x) for x in D["active_id"] if int(x)>=0))]
    local_utility=None
    if len(local)>=2 and all(x.get("sufficient_support") for x in local[:2]):
        local_utility=bool(all(x.get("issued_bce_improves_B") for x in local[:2]))

    if not validity: label="invalid_execution"
    elif not primary: label="D_over_C_not_established"
    elif not preserved: label="D_over_C_with_excess_pool_loss"
    elif not pool["pass"]: label="quality_retained_pool_not_exercised"
    else: label="quality_retained_pool_exercised"

    # Independent direct recomputation of the full AP delta using the common metric.
    cfull=binary_metrics(C["probability"],C["labels"]); dfull=binary_metrics(D["probability"],D["labels"])
    direct_full_ap_delta=None if cfull["ap"] is None or dfull["ap"] is None else float(dfull["ap"]-cfull["ap"])
    recompute_match=bool((direct_full_ap_delta is None and DC["ap_delta"] is None) or abs(direct_full_ap_delta-DC["ap_delta"])<1e-12)
    dump_json(out/"core_metric_recompute.json",{"C_full":cfull,"D_pool2_full":dfull,"direct_full_AP_delta":direct_full_ap_delta,
                                               "comparison_full_AP_delta":DC["ap_delta"],"exact_numeric_match":recompute_match})
    if not recompute_match: raise AssertionError("independent core metric recomputation mismatch")

    arms={"C_ref":C,"B_ref":B,"D_keep":K,"D_039":D39,"D_pool2":D}; diagnostics=event_diagnostics(arms,C["labels"],lifecycle)
    failure=[]
    if not validity: failure.append("validity_failed")
    if not primary: failure.append("primary_D_over_C_not_established")
    if primary and not preserved: failure.append("cumulative_preservation_vs_D_keep_failed")
    if primary and preserved and not pool["pass"]: failure.append("pool_not_exercised")

    comp={"protocol":"040","run_id":str(a.run_id),"single_stream_development_only":True,"statistical_confirmation":False,"F_research":"DEFERRED_NOT_LOADED",
          "primary_D_pool2_vs_C":DC,"cumulative_preservation_D_pool2_vs_D_keep":DK,"D_pool2_vs_D_039":D39c,
          "attribution":{"B_vs_C":BC,"D_pool2_vs_B":DB},"signals":{"primary_D_over_C":primary,"cumulative_preservation":preserved,
          "pool_exercised":pool["pass"],"local_two_expert_utility":local_utility},"pool_exercised":pool,"local_expert_utility":local,
          "validity":{"all_pass":validity,"science_audit":audit,"fixture_all_pass":fixture.get("all_pass"),"independent_metric_recompute":recompute_match},
          "failure_reasons":failure,"result_label":label,"event_diagnostics":diagnostics,
          "reuse_decisions":reuse,"candidate_decisions":candidates,
          "claims":{"causal_reuse_advantage":False,"complete_dynamic_deletion_success":False,"independent_confirmation":False,"overall_deployment_speedup":False,"automatic_followup":False}}
    dump_json(out/"comparison.json",comp)
    dump_json(out/"per_window.json",{"vs_C":DC["windows"],"vs_D_keep":DK["windows"],"vs_D_039":D39c["windows"]})
    dump_json(out/"per_expert.json",{"runtime":sm.get("per_expert",{}),"local_first64":local,"pool_exercised":pool})
    dump_json(out/"lifecycle_events.json",{"events":lifecycle}); dump_json(out/"reuse_decisions.json",{"decisions":reuse}); dump_json(out/"candidate_decisions.json",{"decisions":candidates})
    dump_json(out/"cost_profile.json",{"protocol":"040","science_cost_raw":J(root/"science_cost_raw.json"),
      "incremental":{"live_optimizer_steps":sm.get("live_optimizer_steps"),"shadow_optimizer_steps":sm.get("shadow_optimizer_steps"),
      "deployed_prediction_forwards":sm.get("deployed_prediction_forwards"),"reuse_preview_forwards":sm.get("reuse_preview_forwards"),
      "shadow_qualification_forwards":sm.get("shadow_qualification_forwards"),"accepted_experts":sm.get("accepted_experts")},
      "boundary":"Historical B=C+D_lin cost is separate. Dormant retained weights/Adam still consume memory; no full same-machine B end-to-end speedup claim."})
    dump_json(out/"analysis_status.json",{"protocol":"040","run_id":str(a.run_id),"result_label":label,"validity_all_pass":validity,
      "primary_D_over_C":primary,"cumulative_preservation":preserved,"pool_exercised":pool["pass"],"local_two_expert_utility":local_utility,
      "failure_reasons":failure,"science_status":"completed","analysis_status":"completed","publication_status":"pending_until_main_sync",
      "automatic_followup_training":False,"stop_after_registered_budget":True})

    lines=["# Protocol-040 结果","",
      "本轮只新增一条 **D_pool2** 科学序列；C、B=C+D_lin、D_keep、D_039 均来自冻结缓存，不重训。F 保持 DEFERRED，本轮没有加载或比较 F。","",
      "结果标签：**{}**。这是已观察 seed3601 上的开发结果，不是跨 seed/新流确认，也不能把联合结构包的效果归因到单一模块。".format(label),"",
      "## 预登记判定","",
      "- **D_pool2 相对 C：{}**。full ΔAP {}；六窗 ΔAP {}；late4 {}；prefix32 {}；正向窗 {}/6；护栏 {}。".format(primary,f(DC["ap_delta"]),f(DC["six_window_equal_weight_AP_delta"]),f(DC["late4_equal_weight_AP_delta"]),f(DC["prefix32_equal_weight_AP_delta"]),DC["positive_recurrence_windows"],DC["guardrails"]["all_pass"]),
      "- **累计保持相对 D_keep：{}**。full ΔAP {}；六窗 {}；late4 {}；prefix32 {}；护栏 {}。".format(preserved,f(DK["ap_delta"]),f(DK["six_window_equal_weight_AP_delta"]),f(DK["late4_equal_weight_AP_delta"]),f(DK["prefix32_equal_weight_AP_delta"]),DK["guardrails"]["all_pass"]),
      "- **相对上一版 D_039**：full ΔAP {}；六窗 {}；late4 {}；prefix32 {}。".format(f(D39c["ap_delta"]),f(D39c["six_window_equal_weight_AP_delta"]),f(D39c["late4_equal_weight_AP_delta"]),f(D39c["prefix32_equal_weight_AP_delta"])),
      "- **双专家池实际运行到：{}**。接受 ID={}；active 预测计数={}；满足 >=64 dormant 后同 ID 复用并再 active>=64 的事件数={}。".format(pool["pass"],pool["accepted_distinct_ids"],pool["active_prediction_counts"],sum(x["passes"] for x in pool["qualifying_reactivations"])),
      "- **两个专家局部 first64 均优于 B：{}**。该项仅作描述，不证明专业化或复用因果收益。".format(local_utility),"",
      "## 生命周期与成本","",
      "- first birth={}；最终 accepted experts={}；second attempt consumed={}；reuse slots started={}。".format(sm.get("first_birth_t"),sm.get("accepted_experts"),sm.get("second_attempt_consumed"),sm.get("reuse_slots_started")),
      "- live optimizer.step={}/352；shadow={}/16；动态部署 forward={}/5616；reuse preview={}/2048；shadow qualification={}/32。".format(sm.get("live_optimizer_steps"),sm.get("shadow_optimizer_steps"),sm.get("deployed_prediction_forwards"),sm.get("reuse_preview_forwards"),sm.get("shadow_qualification_forwards")),
      "- 休眠专家保留权重和 Adam 状态，因此不会节省该专家内存；本轮没有永久删除，也没有同机完整 B 路径重测。","",
      "## 有效性边界","",
      "- 冻结输入、因果成熟 batch、容量/梯度/preview 预算、终末零更新、无 F 加载审计：**{}**。".format(audit.get("all_pass")),
      "- 真实 <=256 前缀零梯度磁盘恢复与合成隔离/资格 fixture：**{}**。".format(fixture.get("all_pass")),
      "- 核心 full AP 独立重算一致：**{}**。".format(recompute_match),
      "- Protocol-040 到此停止；不调阈值、不补候选、不加 seed/新流、不自动启动 Protocol-041。",""]
    Path(a.docs_output).write_text("\n".join(lines),encoding="utf8")
    print(json.dumps(J(out/"analysis_status.json"),indent=2,ensure_ascii=False))

if __name__=="__main__": main()
