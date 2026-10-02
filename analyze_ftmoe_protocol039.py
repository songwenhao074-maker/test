"""Frozen Protocol-039 analysis. No F inputs are loaded or compared."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from protocol035_common import binary_metrics, dump_json, load_npz, phase_bounds, pooled_ranges, WINDOWS, W_BLOCKS

LATE={"U_rec2","V_rec2","U_rec3","V_rec3"}

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))

def mdiff(a,b):
    return {k+"_delta":None if a.get(k) is None or b.get(k) is None else float(a[k]-b[k]) for k in ("ap","bce","recall","fpr")}

def mean_defined(xs):
    return None if any(x is None for x in xs) else float(np.mean(xs))

def cmp(base,arm,bounds):
    if not np.array_equal(base["labels"],arm["labels"]): raise AssertionError("label mismatch")
    y=base["labels"]; bf=binary_metrics(base["probability"],y); af=binary_metrics(arm["probability"],y)
    wins=[]
    for n,s,e in WINDOWS:
        b=binary_metrics(base["probability"][s:e],y[s:e]); a=binary_metrics(arm["probability"][s:e],y[s:e])
        b32=binary_metrics(base["probability"][s:s+32],y[s:s+32]); a32=binary_metrics(arm["probability"][s:s+32],y[s:s+32])
        b64=binary_metrics(base["probability"][s:s+64],y[s:s+64]); a64=binary_metrics(arm["probability"][s:s+64],y[s:s+64])
        wins.append({"window":n,"intervals":[s,e],"base":b,"arm":a,**mdiff(a,b),
                     "prefix32":{"base":b32,"arm":a32,**mdiff(a32,b32)},
                     "prefix64":{"base":b64,"arm":a64,**mdiff(a64,b64)}})
    six=mean_defined([x["ap_delta"] for x in wins])
    late=mean_defined([x["ap_delta"] for x in wins if x["window"] in LATE])
    p32=mean_defined([x["prefix32"]["ap_delta"] for x in wins]); p64=mean_defined([x["prefix64"]["ap_delta"] for x in wins])
    bp=pooled_ranges(base["probability"],y,WINDOWS); ap=pooled_ranges(arm["probability"],y,WINDOWS)
    wrows=[]
    for n in W_BLOCKS:
        if n not in bounds: continue
        s,e=bounds[n]; b=binary_metrics(base["probability"][s:e],y[s:e]); a=binary_metrics(arm["probability"][s:e],y[s:e])
        wrows.append({"phase":n,"intervals":[s,e],"base":b,"arm":a,**mdiff(a,b)})
    wmean=mean_defined([x["ap_delta"] for x in wrows]) if wrows else None
    local=[]
    for x in wins:
        rd=x["recall_delta"]; r32=x["prefix32"]["recall_delta"]
        local.append({"window":x["window"],"recall_delta":rd,"prefix32_recall_delta":r32,
                      "recall_pass":rd is not None and rd>=-0.02,
                      "prefix32_recall_pass":r32 is not None and r32>=-0.03})
    g={"full_fpr_delta":mdiff(af,bf)["fpr_delta"],"full_recall_delta":mdiff(af,bf)["recall_delta"],
       "recurrence_pooled_fpr_delta":mdiff(ap,bp)["fpr_delta"],"recurrence_pooled_recall_delta":mdiff(ap,bp)["recall_delta"],
       "W_equal_weight_AP_delta":wmean,"local_recurrence":local}
    common=bool(g["full_fpr_delta"] is not None and g["full_fpr_delta"]<=0.01 and
                g["recurrence_pooled_fpr_delta"] is not None and g["recurrence_pooled_fpr_delta"]<=0.01 and
                g["full_recall_delta"] is not None and g["full_recall_delta"]>=-0.01 and
                g["recurrence_pooled_recall_delta"] is not None and g["recurrence_pooled_recall_delta"]>=-0.01 and
                wmean is not None and wmean>=-0.02)
    g["common_pass"]=common; g["local_pass"]=all(x["recall_pass"] and x["prefix32_recall_pass"] for x in local); g["all_pass"]=bool(g["common_pass"] and g["local_pass"])
    return {"base_full":bf,"arm_full":af,**mdiff(af,bf),"windows":wins,
            "six_window_equal_weight_AP_delta":six,"late4_equal_weight_AP_delta":late,
            "prefix32_equal_weight_AP_delta":p32,"prefix64_equal_weight_AP_delta":p64,
            "positive_recurrence_windows":sum(x["ap_delta"] is not None and x["ap_delta"]>0 for x in wins),
            "recurrence_pooled":{"base":bp,"arm":ap,**mdiff(ap,bp)},"W_blocks":wrows,"guardrails":g}

def event_block(arms,labels,s,e):
    return {"intervals":[int(s),int(e)],"actual_length":int(max(0,e-s)),
            **{name:binary_metrics(x["probability"][s:e],labels[s:e]) for name,x in arms.items()}}

def event_diag(name,t,arms,labels):
    if t is None: return {"event":name,"observed":False}
    t=int(t); out={"event":name,"observed":True,"event_t":t,"first_affected_prediction":t+1}
    out["before64"]=event_block(arms,labels,max(0,t-63),t+1)
    after=[]
    for L in (64,128,256):
        s=t+1; e=min(len(labels),s+L); after.append({"requested_length":L,**event_block(arms,labels,s,e)})
    out["after"]=after
    return out

def gate_ge(v,x): return v is not None and v>=x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--science-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--docs-output",required=True); ap.add_argument("--run-id",required=True)
    a=ap.parse_args(); root=Path(a.science_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    C=load_npz(root/"C_ref/predictions.npz"); B=load_npz(root/"B_ref/predictions.npz"); K=load_npz(root/"D_keep/predictions.npz"); D=load_npz(root/"D_sleepwake/predictions.npz")
    for q in (B,K,D):
        for key in ("labels","raw_labels","class_probability"):
            if key not in q or not np.array_equal(C[key],q[key]): raise AssertionError("four-arm alignment mismatch: "+key)
    bounds=phase_bounds(a.manifest); audit=J(root/"run_audit.json"); fixture=J(root/"fixture_report.json"); ledger=J(root/"budget_ledger.json"); sm=J(root/"D_sleepwake/summary.json"); prov=J(root/"implementation_manifest.json")
    validity=bool(audit.get("all_pass") and fixture.get("all_pass") and ledger.get("sequence",{}).get("completed") and
                  prov.get("protocol")=="039" and prov.get("checkout_sha") and prov.get("workflow_sha256") and
                  prov.get("plan_sha256")=="2182aa9c08a2067ea8abb89b70555a8174fc9a2809d0655c59effa4174cd9811")

    DC=cmp(C,D,bounds); DK=cmp(K,D,bounds); BC=cmp(C,B,bounds); KB=cmp(B,K,bounds); DB=cmp(B,D,bounds)
    primary=bool(gate_ge(DC["ap_delta"],.002) and gate_ge(DC["six_window_equal_weight_AP_delta"],.002) and
                 DC["positive_recurrence_windows"]>=4 and gate_ge(DC["late4_equal_weight_AP_delta"],0) and
                 gate_ge(DC["prefix32_equal_weight_AP_delta"],-.005) and DC["guardrails"]["all_pass"] and validity)
    preserved=bool(gate_ge(DK["ap_delta"],-.001) and gate_ge(DK["six_window_equal_weight_AP_delta"],-.001) and
                   gate_ge(DK["late4_equal_weight_AP_delta"],-.001) and gate_ge(DK["prefix32_equal_weight_AP_delta"],-.005) and
                   DK["guardrails"]["all_pass"] and validity)
    cycle=bool(sm.get("sleep_t") is not None and sm.get("wake_t") is not None and int(sm.get("sleep_predictions",0))>=64 and
               int(sm.get("prediction_forward_calls",10**9))<5616 and int(sm.get("optimizer_steps",10**9))<352 and
               audit.get("sleep_prediction_forward_calls_zero") and audit.get("sleep_optimizer_steps_zero") and validity)
    progress=bool(primary and preserved and cycle)

    fail=[]
    if not validity: fail.append("validity_failed")
    if not primary: fail.append("primary_D_over_C_not_established")
    if not preserved: fail.append("management_preservation_not_established")
    if not cycle: fail.append("cycle_resource_signal_not_established")
    if progress: label="progress_to_next_step"
    elif primary and preserved and not cycle: label="D_over_C_preserved_but_cycle_not_validated"
    elif primary and not preserved: label="D_over_C_with_excess_management_loss"
    elif not validity: label="invalid_execution"
    else: label="primary_D_over_C_not_established"

    arms={"C_ref":C,"B_ref":B,"D_keep":K,"D_sleepwake":D}; labels=C["labels"]
    diagnostics={"sleep":event_diag("sleep",sm.get("sleep_t"),arms,labels),"wake":event_diag("wake",sm.get("wake_t"),arms,labels)}
    resource={"cycle_status":sm.get("cycle_status"),"birth_t":sm.get("birth_t"),"sleep_t":sm.get("sleep_t"),"wake_t":sm.get("wake_t"),
              "prediction_forward_calls":sm.get("prediction_forward_calls"),"training_forward_calls":sm.get("training_forward_calls"),
              "optimizer_steps":sm.get("optimizer_steps"),"sleep_predictions":sm.get("sleep_predictions"),
              "forward_saved_vs_D_keep":None if sm.get("prediction_forward_calls") is None else int(5616-int(sm["prediction_forward_calls"])),
              "steps_saved_vs_D_keep":None if sm.get("optimizer_steps") is None else int(352-int(sm["optimizer_steps"])),
              "sleep_memory_saved":False,"sleep_preserves_weights_and_Adam":True}
    comp={"protocol":"039","run_id":str(a.run_id),"single_stream_development_only":True,"statistical_confirmation":False,
          "F_research":"DEFERRED_NOT_LOADED","primary_D_sleepwake_vs_C":DC,"management_D_sleepwake_vs_D_keep":DK,
          "attribution":{"B_vs_C":BC,"D_keep_vs_B":KB,"D_sleepwake_vs_B":DB},
          "signals":{"primary_D_over_C_signal":primary,"management_preserved_signal":preserved,"cycle_resource_signal":cycle,"progress_to_next_step_signal":progress},
          "failure_reasons":fail,"validity":{"all_pass":validity,"science_audit":audit,"fixture_pass":fixture.get("all_pass")},
          "resource_accounting":resource,"event_diagnostics":diagnostics,"result_label":label,
          "claims":{"complete_dynamic_deletion_success":False,"independent_confirmation":False,"overall_deployment_speedup":False,"automatic_followup":False}}
    dump_json(out/"comparison.json",comp)
    dump_json(out/"analysis_status.json",{"protocol":"039","run_id":str(a.run_id),"result_label":label,
              "primary_D_over_C_signal":primary,"management_preserved_signal":preserved,"cycle_resource_signal":cycle,
              "progress_to_next_step_signal":progress,"validity_all_pass":validity,"failure_reasons":fail,
              "publication_status":"pending_until_main_sync","automatic_followup_training":False,"stop_after_registered_budget":True})
    dump_json(out/"event_diagnostics.json",diagnostics)
    dump_json(out/"cost_profile.json",{"protocol":"039","incremental":resource,"science_cost_raw":J(root/"science_cost_raw.json"),
              "accounting":"Complete D includes historical B=C+D_lin plus the controller and dynamic extra expert. Historical source-run timing is not combined into a same-run speedup claim."})

    def f(v): return "null" if v is None else "{:+.6f}".format(float(v))
    lines=["# Protocol-039 结果","",
      "本轮只新增一条 D_sleepwake 科学序列；C、B=C+D_lin 与 D_keep 均为缓存控制，不重训。F 研究保持 DEFERRED，本轮没有加载或比较 F。","",
      "登记结果标签：**{}**。这是已见 seed3601 上的开发结果，不构成跨流或统计学确认。".format(label),"",
      "## 三个预登记问题","",
      "- **完整 D 相对 C：{}**。全程 ΔAP {}；六窗 ΔAP {}；late4 {}；prefix32 {}；正向复现窗 {}/6；护栏 {}。".format(primary,f(DC["ap_delta"]),f(DC["six_window_equal_weight_AP_delta"]),f(DC["late4_equal_weight_AP_delta"]),f(DC["prefix32_equal_weight_AP_delta"]),DC["positive_recurrence_windows"],DC["guardrails"]["all_pass"]),
      "- **生命周期管理相对 D_keep 保持：{}**。全程 ΔAP {}；六窗 {}；late4 {}；prefix32 {}；护栏 {}。".format(preserved,f(DK["ap_delta"]),f(DK["six_window_equal_weight_AP_delta"]),f(DK["late4_equal_weight_AP_delta"]),f(DK["prefix32_equal_weight_AP_delta"]),DK["guardrails"]["all_pass"]),
      "- **真实 cycle/resource：{}**。birth={}，sleep={}，wake={}；sleep predictions={}；expert prediction-forward={} (<5616)，optimizer.step={} (<352)。".format(cycle,sm.get("birth_t"),sm.get("sleep_t"),sm.get("wake_t"),sm.get("sleep_predictions"),sm.get("prediction_forward_calls"),sm.get("optimizer_steps")),
      "- **progress_to_next_step_signal：{}**。".format(progress),"",
      "## 贡献归因（只作解释）","",
      "- B−C：全程 ΔAP {}；六窗 {}。".format(f(BC["ap_delta"]),f(BC["six_window_equal_weight_AP_delta"])),
      "- D_keep−B：全程 ΔAP {}；六窗 {}。".format(f(KB["ap_delta"]),f(KB["six_window_equal_weight_AP_delta"])),
      "- D_sleepwake−B：全程 ΔAP {}；六窗 {}。".format(f(DB["ap_delta"]),f(DB["six_window_equal_weight_AP_delta"])),
      "因此若 D>C 主要由固定 D_lin 保留而来，应表述为“候选完整系统保住静态增益并减少新增专家计算”，不能把全部 AP 增益归因于 sleep/wake。","",
      "## 工程有效性与成本边界","",
      "- 合成真实入口磁盘恢复、未来扰动、休眠零调用和终末实计数 fixture：**{}**。".format(fixture.get("all_pass")),
      "- 科学输入/出生时刻/D_keep 前缀一致性/真实调用计数/预算审计：**{}**。".format(audit.get("all_pass")),
      "- 新科学 optimizer.step：**{} / 352 max**；相对 D_keep 少 {} 次。".format(sm.get("optimizer_steps"),resource["steps_saved_vs_D_keep"]),
      "- 新专家 prediction-forward：**{} / 5616 D_keep reference**；少 {} 次。".format(sm.get("prediction_forward_calls"),resource["forward_saved_vs_D_keep"]),
      "- sleep 保留 74 参数权重与 Adam 状态，因此**不节省该专家模型内存，也不等价于永久删除**。",
      "- 本轮没有同机完整在线端到端 B 重测，因此不宣称整体部署加速。",
      "- Protocol-039 到此停止；不自动调阈值、加种子、永久删除或恢复 F 研究。",""]
    Path(a.docs_output).write_text("\n".join(lines),encoding="utf8")
    print(json.dumps(J(out/"analysis_status.json"),indent=2,ensure_ascii=False))

if __name__=="__main__": main()
