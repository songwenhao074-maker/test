"""Frozen Protocol-038 analysis for D_bias and D_warm against cached B/F/D_zero."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import protocol037_analysis_frozen as a37
from protocol035_common import binary_metrics, dump_json, load_npz, phase_bounds, WINDOWS


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def W(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, obj)


def mean_defined(xs):
    return None if any(x is None for x in xs) else float(np.mean(xs))


def compare(q, a, bounds):
    return a37.compare(q, a, bounds)


def six_ap(arm, labels):
    vals = [binary_metrics(arm["probability"][s:e], labels[s:e])["ap"] for _, s, e in WINDOWS]
    return mean_defined(vals)


def late4_ap(arm, labels):
    late = {"U_rec2","V_rec2","U_rec3","V_rec3"}
    vals = [binary_metrics(arm["probability"][s:e], labels[s:e])["ap"] for n, s, e in WINDOWS if n in late]
    return mean_defined(vals)


def init_signal(cmp_q, cmp_b, validity):
    return bool(
        validity
        and cmp_q["six_window_equal_weight_AP_delta"] is not None
        and cmp_q["six_window_equal_weight_AP_delta"] >= 0.0005
        and cmp_q["ap_delta"] is not None and cmp_q["ap_delta"] >= 0.0
        and int(cmp_q["positive_recurrence_windows"]) >= 4
        and cmp_q["late4_equal_weight_AP_delta"] is not None and cmp_q["late4_equal_weight_AP_delta"] >= 0.0
        and cmp_q["guardrails"]["all_pass"]
        and cmp_b["guardrails"]["all_pass"]
    )


def confusion(prob, y):
    p = np.asarray(prob) >= 0.5
    t = np.asarray(y) > 0
    return {
        "tp": int((p & t).sum()), "tn": int((~p & ~t).sum()),
        "fp": int((p & ~t).sum()), "fn": int((~p & t).sum()),
    }


def delta_diag(delta, labels, s, e):
    d = np.asarray(delta[s:e], np.float64)
    y = np.asarray(labels[s:e]) > 0
    def stats(x):
        x = np.asarray(x, np.float64).reshape(-1)
        if x.size == 0:
            return {"n":0,"mean":None,"mean_abs":None,"q05":None,"q50":None,"q95":None,"abs_ge_1p98":None}
        ax = np.abs(x)
        return {
            "n": int(x.size), "mean": float(x.mean()), "mean_abs": float(ax.mean()),
            "q05": float(np.quantile(x,0.05)), "q50": float(np.quantile(x,0.50)), "q95": float(np.quantile(x,0.95)),
            "abs_ge_1p98": float(np.mean(ax >= 1.98)),
        }
    return {"all":stats(d),"positive":stats(d[y]),"negative":stats(d[~y])}


def fixed_window_diag(B, arm, s, e):
    y = B["labels"][s:e]
    bc = confusion(B["probability"][s:e], y)
    ac = confusion(arm["probability"][s:e], y)
    return {
        "intervals":[s,e],
        "B":binary_metrics(B["probability"][s:e],y),
        "A":binary_metrics(arm["probability"][s:e],y),
        "delta":delta_diag(arm["delta"], B["labels"], s, e),
        "B_confusion":bc, "A_confusion":ac,
        "false_positive_change": int(ac["fp"]-bc["fp"]),
        "false_negative_change": int(ac["fn"]-bc["fn"]),
        "corrections": a37.correction_counts(B["probability"],arm["probability"],B["labels"],s,e),
    }


def gap_closure(A, D0, F, labels):
    full = {n: binary_metrics(x["probability"], labels)["ap"] for n,x in (("A",A),("D0",D0),("F",F))}
    six = {n: six_ap(x, labels) for n,x in (("A",A),("D0",D0),("F",F))}
    def ratio(vals):
        den = vals["F"] - vals["D0"] if vals["F"] is not None and vals["D0"] is not None else None
        if den is None or den <= 0 or vals["A"] is None:
            return None
        return float((vals["A"] - vals["D0"]) / den)
    return {"full":ratio(full),"six_window_equal_weight":ratio(six),"components":{"full":full,"six":six}}


def postbirth_vs_f(A, F, labels):
    s,e=352,5968
    am=binary_metrics(A["probability"][s:e],labels[s:e])
    fm=binary_metrics(F["probability"][s:e],labels[s:e])
    return {"intervals":[s,e],"A":am,"F":fm,**a37.diff(am,fm)}


def dynamic_increment(extra_signal, cmp_f):
    return bool(
        extra_signal
        and cmp_f["six_window_equal_weight_AP_delta"] is not None and cmp_f["six_window_equal_weight_AP_delta"] >= 0.001
        and cmp_f["ap_delta"] is not None and cmp_f["ap_delta"] >= 0.0
        and cmp_f["guardrails"]["all_pass"]
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",required=True)
    ap.add_argument("--source-manifest",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--docs-output",required=True)
    ap.add_argument("--run-id",required=True)
    args=ap.parse_args()
    root=Path(args.run_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    bounds=phase_bounds(args.source_manifest)
    B=load_npz(root/"B_ref/predictions.npz")
    F=load_npz(root/"F_extra/predictions.npz")
    D0=load_npz(root/"D_zero/predictions.npz")
    Db=load_npz(root/"D_bias/predictions.npz")
    Dw=load_npz(root/"D_warm/predictions.npz")
    audit=J(root/"run_audit.json")
    ledger=J(root/"budget_ledger.json")
    stageA=J(root/"stage_A_diagnostics.json")
    validity=bool(
        audit.get("all_pass") and stageA.get("valid")
        and int(ledger.get("total_optimizer_steps",-1))==725
        and int(ledger["donor"]["optimizer_steps"])==21
        and int(ledger["D_bias"]["optimizer_steps"])==352
        and int(ledger["D_warm"]["optimizer_steps"])==352
    )

    cmp_B_F=compare(B,F,bounds)
    cmp_B_D0=compare(B,D0,bounds)
    cmp_B_Db=compare(B,Db,bounds)
    cmp_B_Dw=compare(B,Dw,bounds)
    cmp_D0_Db=compare(D0,Db,bounds)
    cmp_D0_Dw=compare(D0,Dw,bounds)
    cmp_Db_Dw=compare(Db,Dw,bounds)
    cmp_F_Db=compare(F,Db,bounds)
    cmp_F_Dw=compare(F,Dw,bounds)

    warm_signal=init_signal(cmp_D0_Dw,cmp_B_Dw,validity)
    bias_signal=init_signal(cmp_D0_Db,cmp_B_Db,validity)
    feature_signal=init_signal(cmp_Db_Dw,cmp_B_Dw,validity)
    bias_extra=a37.capability_signal(cmp_B_Db,validity)
    warm_extra=a37.capability_signal(cmp_B_Dw,validity)
    bias_dynamic=dynamic_increment(bias_extra,cmp_F_Db)
    warm_dynamic=dynamic_increment(warm_extra,cmp_F_Dw)

    post={}
    for name,arm in (("D_bias",Db),("D_warm",Dw)):
        post[name]={
            "64":fixed_window_diag(B,arm,352,416),
            "128":fixed_window_diag(B,arm,352,480),
            "256":fixed_window_diag(B,arm,352,608),
            "vs_F_postbirth":postbirth_vs_f(arm,F,B["labels"]),
            "gap_closure":gap_closure(arm,D0,F,B["labels"]),
            "parameter_snapshots":J(root/name/"parameter_snapshots.json"),
        }

    if feature_signal:
        label="warm_feature_weight_increment_signal"
    elif warm_signal and bias_signal:
        label="bias_and_warm_signal_no_detected_feature_increment"
    elif warm_signal:
        label="warm_initialization_signal_only"
    elif bias_signal:
        label="bias_initialization_signal_only"
    else:
        early = any(
            post[n][L]["A"]["ap"] is not None and post[n][L]["B"]["ap"] is not None and post[n][L]["A"]["ap"] > post[n][L]["B"]["ap"]
            for n in ("D_bias","D_warm") for L in ("64","128","256")
        )
        label="early_only" if early else "no_initialization_signal"

    comparison={
        "protocol":"038","run_id":str(args.run_id),"validity":validity,"result_label":label,
        "signals":{
            "warm_init_signal":warm_signal,
            "bias_init_signal":bias_signal,
            "feature_weight_increment":feature_signal,
            "D_bias_extra_capability":bias_extra,
            "D_warm_extra_capability":warm_extra,
            "D_bias_dynamic_increment_vs_F":bias_dynamic,
            "D_warm_dynamic_increment_vs_F":warm_dynamic,
        },
        "comparisons":{
            "B_vs_F":cmp_B_F,"B_vs_D_zero":cmp_B_D0,"B_vs_D_bias":cmp_B_Db,"B_vs_D_warm":cmp_B_Dw,
            "D_zero_vs_D_bias":cmp_D0_Db,"D_zero_vs_D_warm":cmp_D0_Dw,"D_bias_vs_D_warm":cmp_Db_Dw,
            "F_vs_D_bias":cmp_F_Db,"F_vs_D_warm":cmp_F_Dw,
        },
        "post_birth_diagnostics":post,
        "sampling_context":{
            "F_first21":stageA["F_first21_sampling"],
            "D_zero_first21":stageA["D_zero_first21_sampling"],
            "not_a_controlled_causal_comparison":True,
        },
        "interpretation_limits":{
            "single_observed_development_stream":True,
            "six_windows_not_independent_seeds":True,
            "no_statistical_significance_claim":True,
            "no_equivalence_claim_from_no_signal":True,
            "dynamic_birth_without_prefix_training_not_claimed":True,
        },
    }
    W(out/"comparison.json",comparison)

    cost={
        "protocol":"038","run_id":str(args.run_id),
        "new_scientific_optimizer_steps":{"donor_shared":21,"D_bias":352,"D_warm":352,"total":725},
        "per_deployed_method_charge":{"D_bias":373,"D_warm":373},
        "cached_baselines_new_cost":0,
        "D_bias_summary":J(root/"D_bias/summary.json"),
        "D_warm_summary":J(root/"D_warm/summary.json"),
        "science_wall_seconds":audit.get("science_wall_seconds"),
        "end_to_end_speedup":"unknown_without_same-machine_full-B-deployment-measurement",
    }
    W(out/"cost_profile.json",cost)
    status={
        "protocol":"038","run_id":str(args.run_id),"science_status":"completed","publication_status":"pending",
        "result_label":label,"validity":validity,"signals":comparison["signals"],
        "optimizer_steps":725,"stop_after_protocol":True,
    }
    W(out/"status.json",status)

    def f6(x):
        return "null" if x is None else f"{x:.6f}"
    md=[
        "# Protocol-038 结果：固定出生机制下的初始化机制对照",
        "",
        f"- GitHub Actions run：**{args.run_id}**",
        f"- 科学有效性：**{validity}**；结果标签：**{label}**。",
        "- 本轮严格复用 seed3601；B/F/D_zero 均为缓存只读对照，没有重训。",
        "- 新科学更新严格为 donor 21 + D_bias 352 + D_warm 352 = **725** 次。",
        "",
        "## 主要预登记判断",
        f"- warm_init_signal：**{warm_signal}**；D_warm−D_zero 六窗等权 AP = **{f6(cmp_D0_Dw['six_window_equal_weight_AP_delta'])}**，全程 AP = **{f6(cmp_D0_Dw['ap_delta'])}**。",
        f"- bias_init_signal：**{bias_signal}**；D_bias−D_zero 六窗等权 AP = **{f6(cmp_D0_Db['six_window_equal_weight_AP_delta'])}**，全程 AP = **{f6(cmp_D0_Db['ap_delta'])}**。",
        f"- feature_weight_increment：**{feature_signal}**；D_warm−D_bias 六窗等权 AP = **{f6(cmp_Db_Dw['six_window_equal_weight_AP_delta'])}**，全程 AP = **{f6(cmp_Db_Dw['ap_delta'])}**。",
        "",
        "## 与固定专家 F 的对照",
        f"- D_bias extra_capability：**{bias_extra}**；dynamic_increment_vs_F：**{bias_dynamic}**。",
        f"- D_warm extra_capability：**{warm_extra}**；dynamic_increment_vs_F：**{warm_dynamic}**。",
        f"- D_bias 相对 F 的出生后[352,5968) AP差：**{f6(post['D_bias']['vs_F_postbirth']['ap_delta'])}**。",
        f"- D_warm 相对 F 的出生后[352,5968) AP差：**{f6(post['D_warm']['vs_F_postbirth']['ap_delta'])}**。",
        "",
        "## 初始化成本与边界",
        "- donor 是真实的 21 步历史前缀模型，不是免费初始化；单独部署 D_bias 或 D_warm 均应计 21+352=373 步。",
        f"- 037 固定 F 前21更新采样正例比例：**{stageA['F_first21_sampling']['positive_ratio']:.6f}**；D_zero 出生后前21更新：**{stageA['D_zero_first21_sampling']['positive_ratio']:.6f}**。该差异只作历史分布背景，不是受控因果证据。",
        "- 六个复现窗来自同一条已观察开发流，不能视为六个独立随机种子，也不支持统计显著性或等价性结论。",
        "- Protocol-038 到此停止；不得自动追加新 seed、结构、休眠/唤醒/删除或下一协议。",
        "",
    ]
    Path(args.docs_output).write_text("\n".join(md),encoding="utf8")
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
