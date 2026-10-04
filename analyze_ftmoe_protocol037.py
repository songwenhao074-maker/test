"""Frozen Protocol-037 analysis: B_ref vs fixed extra expert vs one causally born expert."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from protocol035_common import binary_metrics, dump_json, load_npz, phase_bounds, pooled_ranges, WINDOWS, W_BLOCKS

LATE = {"U_rec2", "V_rec2", "U_rec3", "V_rec3"}


def J(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def mean_defined(xs):
    return None if any(x is None for x in xs) else float(np.mean(xs))


def diff(a, b):
    return {k + "_delta": None if a.get(k) is None or b.get(k) is None else float(a[k] - b[k]) for k in ("ap", "bce", "recall", "fpr")}


def block(p, y, s, e):
    return binary_metrics(p[s:e], y[s:e])


def compare(base, arm, bounds):
    if not np.array_equal(base["labels"], arm["labels"]):
        raise AssertionError("Protocol037 label mismatch")
    y = base["labels"]
    bm = binary_metrics(base["probability"], y)
    am = binary_metrics(arm["probability"], y)
    windows = []
    for name, s, e in WINDOWS:
        b = block(base["probability"], y, s, e)
        a = block(arm["probability"], y, s, e)
        b32, a32 = block(base["probability"], y, s, s + 32), block(arm["probability"], y, s, s + 32)
        b64, a64 = block(base["probability"], y, s, s + 64), block(arm["probability"], y, s, s + 64)
        windows.append({
            "window": name, "intervals": [s, e], "B": b, "A": a, **diff(a, b),
            "prefix32": {"B": b32, "A": a32, **diff(a32, b32)},
            "prefix64": {"B": b64, "A": a64, **diff(a64, b64)},
        })
    six = mean_defined([x["ap_delta"] for x in windows])
    late = mean_defined([x["ap_delta"] for x in windows if x["window"] in LATE])
    p32 = mean_defined([x["prefix32"]["ap_delta"] for x in windows])
    p64 = mean_defined([x["prefix64"]["ap_delta"] for x in windows])
    bp = pooled_ranges(base["probability"], y, WINDOWS)
    ap = pooled_ranges(arm["probability"], y, WINDOWS)
    wrows = []
    for name in W_BLOCKS:
        if name not in bounds:
            continue
        s, e = bounds[name]
        b, a = block(base["probability"], y, s, e), block(arm["probability"], y, s, e)
        wrows.append({"phase": name, "intervals": [s, e], "B": b, "A": a, **diff(a, b)})
    wmean = mean_defined([x["ap_delta"] for x in wrows]) if wrows else None
    local = []
    for x in windows:
        rd = x["recall_delta"]
        r32 = x["prefix32"]["recall_delta"]
        local.append({
            "window": x["window"], "recall_delta": rd, "prefix32_recall_delta": r32,
            "recall_pass": rd is not None and rd >= -0.02,
            "prefix32_recall_pass": r32 is not None and r32 >= -0.03,
        })
    guards = {
        "full_fpr_delta": diff(am, bm)["fpr_delta"], "full_recall_delta": diff(am, bm)["recall_delta"],
        "recurrence_pooled_fpr_delta": diff(ap, bp)["fpr_delta"], "recurrence_pooled_recall_delta": diff(ap, bp)["recall_delta"],
        "W_equal_weight_AP_delta": wmean, "local_recurrence": local,
    }
    common = bool(
        guards["full_fpr_delta"] is not None and guards["full_fpr_delta"] <= 0.01
        and guards["recurrence_pooled_fpr_delta"] is not None and guards["recurrence_pooled_fpr_delta"] <= 0.01
        and guards["full_recall_delta"] is not None and guards["full_recall_delta"] >= -0.01
        and guards["recurrence_pooled_recall_delta"] is not None and guards["recurrence_pooled_recall_delta"] >= -0.01
        and wmean is not None and wmean >= -0.02
    )
    local_pass = all(x["recall_pass"] and x["prefix32_recall_pass"] for x in local)
    guards["common_pass"] = common
    guards["local_pass"] = local_pass
    guards["all_pass"] = bool(common and local_pass)
    positive = sum(x["ap_delta"] is not None and x["ap_delta"] > 0 for x in windows)
    per_host = []
    for h in range(16):
        b, a = binary_metrics(base["probability"][:, h], y[:, h]), binary_metrics(arm["probability"][:, h], y[:, h])
        per_host.append({"host": h, "B": b, "A": a, **diff(a, b)})
    return {
        "B_full": bm, "A_full": am, **diff(am, bm),
        "windows": windows, "six_window_equal_weight_AP_delta": six,
        "late4_equal_weight_AP_delta": late, "prefix32_equal_weight_AP_delta": p32,
        "prefix64_equal_weight_AP_delta": p64, "positive_recurrence_windows": int(positive),
        "recurrence_pooled": {"B": bp, "A": ap, **diff(ap, bp)},
        "W_blocks": wrows, "guardrails": guards, "per_host": per_host,
    }


def capability_signal(x, validity):
    return bool(
        x["ap_delta"] is not None and x["ap_delta"] >= 0.002
        and x["six_window_equal_weight_AP_delta"] is not None and x["six_window_equal_weight_AP_delta"] >= 0.002
        and int(x["positive_recurrence_windows"]) >= 4
        and x["late4_equal_weight_AP_delta"] is not None and x["late4_equal_weight_AP_delta"] >= 0.0
        and x["prefix32_equal_weight_AP_delta"] is not None and x["prefix32_equal_weight_AP_delta"] >= -0.005
        and x["guardrails"]["all_pass"] and validity
    )


def direct_D_vs_F(Fp, Dp, labels, bounds):
    fakeB = {"probability": Fp, "labels": labels}
    fakeD = {"probability": Dp, "labels": labels}
    return compare(fakeB, fakeD, bounds)


def correction_counts(base_p, arm_p, labels, s=0, e=None):
    if e is None:
        e = len(base_p)
    b = np.asarray(base_p[s:e]) >= 0.5
    a = np.asarray(arm_p[s:e]) >= 0.5
    y = np.asarray(labels[s:e]) > 0
    fixed = (~(b == y)) & (a == y)
    harmed = (b == y) & (~(a == y))
    return {
        "rows": int(y.size),
        "corrected_total": int(fixed.sum()),
        "corrected_positive": int((fixed & y).sum()),
        "corrected_negative": int((fixed & ~y).sum()),
        "harmed_total": int(harmed.sum()),
        "harmed_positive": int((harmed & y).sum()),
        "harmed_negative": int((harmed & ~y).sum()),
    }


def delta_stats(delta, s, e):
    d = np.asarray(delta[s:e], dtype=np.float64).reshape(-1)
    if d.size == 0:
        return {"rows": 0, "mean_abs": None, "p95_abs": None, "saturation_abs_ge_1p9": None}
    ad = np.abs(d)
    return {
        "rows": int(d.size), "mean_abs": float(ad.mean()), "p95_abs": float(np.quantile(ad, 0.95)),
        "saturation_abs_ge_1p9": float(np.mean(ad >= 1.9)),
    }


def diagnostic(base, Fp, Dp, Fdelta, Ddelta, labels, bounds, birth_t):
    out = {"birth_t": birth_t, "first_affected_prediction": None if birth_t is None else int(birth_t) + 1}
    start = None if birth_t is None else int(birth_t) + 1
    post = []
    for L in (64, 128, 256):
        if start is None or start >= len(labels):
            post.append({"requested_length": L, "actual_length": 0})
            continue
        end = min(len(labels), start + L)
        post.append({
            "requested_length": L, "actual_length": int(end - start), "intervals": [start, end],
            "B": binary_metrics(base[start:end], labels[start:end]),
            "F": binary_metrics(Fp[start:end], labels[start:end]),
            "D": binary_metrics(Dp[start:end], labels[start:end]),
            "F_delta": delta_stats(Fdelta, start, end), "D_delta": delta_stats(Ddelta, start, end),
            "F_corrections": correction_counts(base, Fp, labels, start, end),
            "D_corrections": correction_counts(base, Dp, labels, start, end),
        })
    out["post_birth_fixed_lengths"] = post
    phases = []
    for name, (s, e) in bounds.items():
        phases.append({
            "phase": name, "intervals": [s, e],
            "B": binary_metrics(base[s:e], labels[s:e]), "F": binary_metrics(Fp[s:e], labels[s:e]), "D": binary_metrics(Dp[s:e], labels[s:e]),
            "F_delta": delta_stats(Fdelta, s, e), "D_delta": delta_stats(Ddelta, s, e),
            "F_corrections": correction_counts(base, Fp, labels, s, e), "D_corrections": correction_counts(base, Dp, labels, s, e),
        })
    out["phase_table"] = phases
    hosts = []
    for h in range(16):
        hosts.append({
            "host": h,
            "B": binary_metrics(base[:, h], labels[:, h]), "F": binary_metrics(Fp[:, h], labels[:, h]), "D": binary_metrics(Dp[:, h], labels[:, h]),
            "F_corrections": correction_counts(base[:, [h]], Fp[:, [h]], labels[:, [h]]),
            "D_corrections": correction_counts(base[:, [h]], Dp[:, [h]], labels[:, [h]]),
        })
    out["per_host"] = hosts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--docs-output", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    root = Path(a.run_root)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    B = load_npz(root / "B_ref/predictions.npz")
    Fp = load_npz(root / "F_extra/predictions.npz")
    Dp = load_npz(root / "D_birth/predictions.npz")
    bounds = phase_bounds(a.source_manifest)
    fixture = J(a.fixture)
    provenance = J(a.provenance)
    audit = J(root / "run_audit.json")
    ledger = J(root / "budget_ledger.json")
    Fsum = J(root / "F_extra/summary.json")
    Dsum = J(root / "D_birth/summary.json")

    budget_pass = bool(
        ledger["training_sequence_budget"] == 2
        and all(ledger["sequences"][x]["started"] and ledger["sequences"][x]["completed"] for x in ("F_extra", "D_birth"))
        and ledger["new_raw_streams"] == 0 and ledger["baseline_retraining"] == 0
        and ledger["other_real_stream_evaluations"] == 0 and ledger["hyperparameter_or_trigger_sweeps"] == 0
        and ledger["sleep_wake_delete_trials"] == 0
        and ledger["engineering_real_prefix_executions"] == 2 and ledger["engineering_real_prefix_max_steps"] == 256
    )
    provenance_pass = bool(
        provenance.get("protocol") == "037"
        and provenance.get("checkout_sha") not in (None, "unknown")
        and provenance.get("workflow_sha256")
        and provenance.get("run_script_sha256")
        and provenance.get("analysis_script_sha256")
        and provenance.get("plan_sha256") == "01240a944bb654d529577b397602cf67c24c2245c6fbe909564b47da2726012a"
    )
    validity = bool(fixture.get("passed") and audit.get("all_pass") and budget_pass and provenance_pass)

    Fcmp = compare(B, Fp, bounds)
    Dcmp = compare(B, Dp, bounds)
    Fsignal = capability_signal(Fcmp, validity)
    Dsignal = capability_signal(Dcmp, validity)
    direct = direct_D_vs_F(Fp["probability"], Dp["probability"], B["labels"], bounds)
    DvsF_signal = bool(
        Dsum.get("birth_occurred") and Dsignal
        and direct["six_window_equal_weight_AP_delta"] is not None and direct["six_window_equal_weight_AP_delta"] >= 0.001
        and direct["ap_delta"] is not None and direct["ap_delta"] >= 0.0
        and direct["guardrails"]["all_pass"]
    )

    if DvsF_signal:
        label = "dynamic_birth_incremental_signal"
    elif Dsignal:
        label = "dynamic_birth_extra_capability_without_incremental_advantage"
    elif Fsignal:
        label = "fixed_extra_capability_only"
    elif not Dsum.get("birth_occurred"):
        label = "trigger_not_activated"
    else:
        label = "extra_capability_not_established"

    diag = diagnostic(B["probability"], Fp["probability"], Dp["probability"], Fp["delta"], Dp["delta"], B["labels"], bounds, Dsum.get("birth_t"))
    cost = J(root / "science_cost_raw.json")
    comp = {
        "protocol": "037", "run_id": str(a.run_id), "development_only": True, "statistical_confirmation": False,
        "source_seed": 3601, "primary_baseline": "B_ref=C+D_lin",
        "F_extra_vs_B": Fcmp, "D_birth_vs_B": Dcmp, "D_birth_vs_F_extra": direct,
        "birth": {
            "occurred": bool(Dsum.get("birth_occurred")), "birth_t": Dsum.get("birth_t"),
            "first_affected_prediction": Dsum.get("first_affected_prediction"), "trigger_status": Dsum.get("trigger_status"),
        },
        "signals": {
            "F_extra_capability_signal": Fsignal,
            "D_birth_extra_capability_signal": Dsignal,
            "dynamic_birth_incremental_signal": DvsF_signal,
        },
        "validity": {
            "fixture_pass": bool(fixture.get("passed")), "science_audit_pass": bool(audit.get("all_pass")),
            "budget_pass": budget_pass, "fresh_science_provenance_pass": provenance_pass, "all_pass": validity,
        },
        "resource_accounting": cost, "diagnostics": diag, "result_label": label,
        "claims": {
            "equal_actual_training_compute": False, "independent_confirmation": False,
            "dynamic_deletion": False, "sleep_wake": False, "automatic_next_protocol": False,
        },
    }
    dump_json(out / "comparison.json", comp)
    dump_json(out / "cost_profile.json", cost)
    status = {
        "protocol": "037", "run_id": str(a.run_id), "scientific_status": "completed",
        "publication_status": "pending_until_main_sync", "training_sequences_registered": 2,
        "training_sequences_completed": 2, "new_streams_registered": 0, "new_streams_completed": 0,
        "birth_occurred": bool(Dsum.get("birth_occurred")), "F_extra_capability_signal": Fsignal,
        "D_birth_extra_capability_signal": Dsignal, "dynamic_birth_incremental_signal": DvsF_signal,
        "validity_all_pass": validity, "result_label": label, "statistical_confirmation": False,
        "automatic_followup_training": False, "stop_after_registered_budget": True,
    }
    dump_json(out / "status.json", status)

    def fmt(v):
        return "null" if v is None else f"{float(v):+.6f}"

    md = [
        "# Protocol-037 结果", "",
        "本轮只在已观察的 Protocol-036 seed3601 上进行机制开发：共同底座 B=C+D_lin 不重训；新增两条科学序列 F_extra 与 D_birth。没有生成新流、没有扫描触发阈值、没有休眠/唤醒/删除试验。",
        "",
        f"结果标签：**{label}**。本轮不是独立统计确认。",
        "",
        "## 核心结果", "",
        "| 方法 | 是否出生/存在 | 全程 ΔAP vs B | 六窗 ΔAP vs B | late4 ΔAP | prefix32 ΔAP | 护栏 | 额外能力信号 |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|",
        f"| F_extra | 从 cursor0 固定存在 | {fmt(Fcmp['ap_delta'])} | {fmt(Fcmp['six_window_equal_weight_AP_delta'])} | {fmt(Fcmp['late4_equal_weight_AP_delta'])} | {fmt(Fcmp['prefix32_equal_weight_AP_delta'])} | {Fcmp['guardrails']['all_pass']} | {Fsignal} |",
        f"| D_birth | {'t='+str(Dsum.get('birth_t')) if Dsum.get('birth_occurred') else '未触发'} | {fmt(Dcmp['ap_delta'])} | {fmt(Dcmp['six_window_equal_weight_AP_delta'])} | {fmt(Dcmp['late4_equal_weight_AP_delta'])} | {fmt(Dcmp['prefix32_equal_weight_AP_delta'])} | {Dcmp['guardrails']['all_pass']} | {Dsignal} |",
        "",
        "## 动态出生相对固定额外专家", "",
        f"- D_birth − F_extra：全程 ΔAP **{fmt(direct['ap_delta'])}**；六窗 ΔAP **{fmt(direct['six_window_equal_weight_AP_delta'])}**。",
        f"- 相对 F 的同一护栏通过：**{direct['guardrails']['all_pass']}**。",
        f"- 登记的 dynamic_birth_incremental_signal：**{DvsF_signal}**。",
        "",
        "## 触发与资源", "",
        f"- 真实出生是否发生：**{bool(Dsum.get('birth_occurred'))}**；birth_t={Dsum.get('birth_t')}；第一条可受影响预测={Dsum.get('first_affected_prediction')}。",
        f"- F_extra 优化步：**{Fsum.get('optimizer_steps')}**；D_birth 优化步：**{Dsum.get('optimizer_steps')}**。",
        f"- F_extra 推理调用：**{Fsum.get('inference_calls')}**；D_birth 推理调用：**{Dsum.get('inference_calls')}**。",
        f"- F_extra 驻留参数×interval：**{Fsum.get('resident_parameter_interval_integral')}**；D_birth：**{Dsum.get('resident_parameter_interval_integral')}**。",
        "- 两臂只具有相同新增参数上限和相同时刻更新上限；实际训练计算不同，因此不声称严格同计算。",
        "",
        "## 有效性与边界", "",
        f"- 工程恢复/未来扰动/终末结算 fixture：**{bool(fixture.get('passed'))}**。",
        f"- 科学因果、B隔离、批次成熟、一次出生上限审计：**{bool(audit.get('all_pass'))}**。",
        f"- 两条训练序列预算与零新流账本：**{budget_pass}**。",
        f"- science checkout/workflow/源码 provenance：**{provenance_pass}**。",
        "- 本轮只使用已观察 seed3601，因此不能声称跨流或跨模型初始化泛化。",
        "- 本轮没有休眠、唤醒或永久删除，不能据此宣称完整动态增删专家系统有效。",
        "- 完成 Protocol-037 后按登记停止，不自动进入下一协议。",
        "",
    ]
    Path(a.docs_output).write_text("\n".join(md), encoding="utf8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
