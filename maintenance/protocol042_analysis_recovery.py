"""Protocol-042 rev2 analysis/publication recovery.

Consumes sealed completed science only. It never imports or calls the Protocol042 arm runner.
The sole analyzer repair is to tolerate the missing U_parent science_cost_raw.json that was
not written because the original U reproduction gate raised after science completed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys

def J(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf8"))

def W(p,x):
    p=pathlib.Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf8")

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def run(cmd,cwd):
    print("+"," ".join(str(x) for x in cmd),flush=True)
    subprocess.run([str(x) for x in cmd],cwd=str(cwd),check=True)

def validate_science(root):
    root=pathlib.Path(root)
    led=J(root/"budget_ledger.json")
    assert led["protocol"]=="042" and led["revision"]==2
    assert led["order"]==["U_parent","R_win128","A_hist","A_win128"]
    assert all(type(led["sequences"][a].get("completed")) is bool and led["sequences"][a]["completed"] for a in led["order"])
    assert led["total_optimizer_steps"]<=2576
    assert led["new_streams"]==led["extra_seeds"]==led["F_loads"]==led["extra_arms"]==led["permanent_deletions"]==0
    ps=J(root/"protocol_status.json")
    assert ps["science_status"]=="completed" and ps["science_rc"]==0 and ps["restart_from_zero"] is False
    for arm in led["order"]:
        st=J(root/arm/"status.json"); assert st["science_status"]=="completed",(arm,st)
        au=J(root/arm/"run_audit.json"); assert type(au.get("all_pass")) is bool and au["all_pass"] is True,(arm,au)
        sm=J(root/arm/"summary.json"); assert sm["terminal_progress"]==3 and sm["next_substep"]=="done"
    u=J(root/"U_parent/reproduction_report.json")
    assert u["all_pass"] is True and u["science_reexecuted"] is False
    assert u["gate_recovery"]=="preregistered_float_tolerance_1e-12"
    assert (root/"U_parent/reproduction_report_gate_failure_original.json").exists()
    assert (root/"U_parent/reproduction_tolerance_audit.json").exists()
    assert not (root/"U_parent/science_cost_raw.json").exists()
    for arm in ("R_win128","A_hist","A_win128"):
        assert (root/arm/"science_cost_raw.json").exists(),arm
    raw=J(root/"scientific_raw_manifest.json")
    assert raw["protocol"]=="042" and raw["revision"]==2 and raw["file_count"]>250
    return led

def patch_analyzer(repo,analysis,science_run,recovery_run,raw_id,raw_digest):
    repo=pathlib.Path(repo); analysis=pathlib.Path(analysis); analysis.mkdir(parents=True,exist_ok=True)
    p=repo/"analyze_ftmoe_protocol042.py"
    original=p.read_bytes()
    s=original.decode("utf8")
    old='''    costs={arm:J(root/arm/"science_cost_raw.json") for arm in c.ARMS}; dump_json(out/"cost_profile.json",{"protocol":"042","revision":2,"arms":costs,
      "note":"R/A may realize different compute. Score computation uses zero extra expert forward; reuse previews and multi-active deployment remain explicit."})'''
    new='''    costs={}
    for arm in c.ARMS:
        cp=root/arm/"science_cost_raw.json"
        if cp.exists():
            costs[arm]=J(cp)
        elif arm=="U_parent":
            sm=J(root/arm/"summary.json")
            costs[arm]={"protocol":"042","revision":2,"arm":"U_parent","availability":"runtime_seconds_unavailable_due_to_original_U_gate_failure_before_cost_write","wall_seconds":None,"process_seconds":None,"prediction_seconds":None,"update_seconds":None,"controller_cpu_seconds":None,"live_optimizer_steps":int(sm["live_optimizer_steps"]),"shadow_optimizer_steps":int(sm["shadow_optimizer_steps"]),"deployed_forwards":int(sm["deployed_prediction_forwards"]),"reuse_forwards":int(sm["reuse_preview_forwards"]),"qualification_forwards":int(sm["shadow_qualification_forwards"]),"science_reexecuted":False}
        else:
            raise FileNotFoundError(str(cp))
    dump_json(out/"cost_profile.json",{"protocol":"042","revision":2,"arms":costs,
      "note":"R/A may realize different compute. Score computation uses zero extra expert forward; reuse previews and multi-active deployment remain explicit. U runtime seconds are unavailable because the original U reproduction gate stopped before science_cost_raw.json was written; U scientific counters are recovered from the sealed summary without retraining."})'''
    assert old in s,"exact analyzer recovery target not found"
    patched=s.replace(old,new).encode("utf8")
    p.write_bytes(patched)
    rec={"protocol":"042","revision":2,"science_run_id":str(science_run),"analysis_recovery_run_id":str(recovery_run),
         "raw_artifact_id":int(raw_id),"raw_artifact_digest":raw_digest,
         "frozen_execution_sha":"da5eaa7e71f44497ede538c884e7ca688e6afbf4",
         "original_analyzer_sha256":hashlib.sha256(original).hexdigest(),
         "recovered_analyzer_sha256":hashlib.sha256(patched).hexdigest(),
         "patch_scope":"cost_profile_only_allow_missing_U_runtime_file_and_recover_scientific_counters_from_sealed_summary",
         "scientific_metrics_or_thresholds_changed":False,"science_reexecuted":False,
         "missing_U_wall_process_seconds_fabricated":False}
    W(analysis/"analysis_recovery.json",rec)
    return rec

def write_pointers(repo,status,science_run,recovery_run):
    repo=pathlib.Path(repo); rid=str(science_run); rec=str(recovery_run)
    sysl=status["system_result_label"]; main=status["main_development_success"]; add=status["addition_policy_signal"]; win=status["window_policy_signal"]; pool=status["pool_exercised"]
    (repo/"AGENTS.md").write_text(
        "# Active Experiment Directive — Protocol-042 revision 2 completed\n\n"
        f"Protocol-042 revision 2 science run {rid} completed. Analysis/publication recovered in run {rec} without scientific reexecution. "
        f"System label: {sysl}. Main development success: {main}. Addition-policy signal: {add}. Window-policy signal: {win}. Pool exercised: {pool}.\n\n"
        f"Read docs/PROTOCOL042_RESULTS.md and artifacts/ftmoe_online/protocol_042/runs/run_{rid}/. Historical033–042 scientific budgets are closed.\n\n"
        "Stop after Protocol-042 revision 2. Do not retry, add 64/256 science arms, add seed/stream, enable F, add hard weighting, delete accepted memories, or automatically start Protocol-043.\n",encoding="utf8")
    (repo/"NEXT_EXPERIMENT_LATEST.md").write_text(
        "# Protocol-042 revision 2 已完成，停止等待分析\n\n"
        f"Protocol-042 revision 2 science run {rid} 已完成；分析/发布在run {rec}无重训恢复。系统标签 **{sysl}**；main development success={main}；addition-policy signal={add}；window-policy signal={win}；pool exercised={pool}。\n\n"
        f"读取 docs/PROTOCOL042_RESULTS.md 与 artifacts/ftmoe_online/protocol_042/runs/run_{rid}/。四条登记序列均已关闭。\n\n"
        "不自动重试，不把64/256窗口变成科学臂，不加seed/新流，不恢复F，不加困难样本加权，不永久删除accepted memory，不启动Protocol-043。\n",encoding="utf8")
    (repo/"PROJECT_CONTEXT_LATEST.md").write_text(
        "# Project Context Latest\n\n"
        f"Latest bounded science is Protocol-042 revision 2 science run {rid}. Analysis/publication recovery run {rec} did not reexecute science. "
        f"System label: **{sysl}**. Main development success: {main}. Addition-policy signal: {add}. Window-policy signal: {win}. Pool exercised: {pool}.\n\n"
        f"Protocol-042 tested additive multi-expert admission and current-epoch historical versus global-time window-128 utility using the same seen seed3601. Read docs/PROTOCOL042_RESULTS.md and artifacts/ftmoe_online/protocol_042/runs/run_{rid}/.\n\n"
        "Historical033–042 budgets are closed. Any retry or next protocol requires new preregistration and user handoff.\n",encoding="utf8")
    (repo/"docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
        "# 当前交接：Protocol-042 revision 2 已完成\n\n"
        f"Protocol-042 revision 2 science run {rid} 已完成；analysis/publication recovery run {rec} 仅消费封存raw，未重训。系统标签 **{sysl}**；main development success={main}；addition-policy signal={add}；window-policy signal={win}；pool exercised={pool}。\n\n"
        f"结果见 [PROTOCOL042_RESULTS.md](PROTOCOL042_RESULTS.md)，紧凑证据见 ../artifacts/ftmoe_online/protocol_042/runs/run_{rid}/。原raw artifact ID 11276470062。\n\n"
        "到此停止。不重跑042，不增加窗口科学臂/seed/流/F/困难加权/永久删除，也不自动进入043。\n",encoding="utf8")
    (repo/"README.md").write_text(
        "# PreGAN+ / FT-MoE 在线实验\n\n"
        "目标：在合理部署场景中，通过结构修改，使保留动态新增、知识保存/选择、休眠复用及容量管理思想的完整D优于持续学习C。\n\n"
        f"**最新完成：[Protocol-042 revision 2 窗口效用与加性接入](docs/PROTOCOL042_RESULTS.md)，science run {rid}。**\n"
        f"分析/发布在run {rec}无重训恢复。系统标签 **{sysl}**；main development success={main}；addition-policy signal={add}；window-policy signal={win}；pool exercised={pool}。\n\n"
        f"[执行交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [042紧凑证据](artifacts/ftmoe_online/protocol_042/runs/run_{rid}/) / [042冻结方案](docs/PROTOCOL042_WINDOWED_UTILITY_20261003.md)\n\n"
        "结果仍属于已观察seed3601开发证据，不是独立确认。Protocol-042 revision 2 到此停止。\n",encoding="utf8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True); ap.add_argument("--science",required=True); ap.add_argument("--analysis",required=True)
    ap.add_argument("--raw-info",required=True); ap.add_argument("--science-run-id",required=True); ap.add_argument("--recovery-run-id",required=True)
    ap.add_argument("--raw-artifact-id",required=True,type=int); ap.add_argument("--raw-digest",required=True)
    a=ap.parse_args()
    repo=pathlib.Path(a.repo).resolve(); science=pathlib.Path(a.science).resolve(); analysis=pathlib.Path(a.analysis).resolve()
    led=validate_science(science)
    rec=patch_analyzer(repo,analysis,a.science_run_id,a.recovery_run_id,a.raw_artifact_id,a.raw_digest)
    manifest=repo/"artifacts/ftmoe_online/protocol_036/runs/run_36831958978/seed3601/stream_manifest.json"
    assert sha(manifest)=="73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b"
    docs=repo/"docs/PROTOCOL042_RESULTS.md"
    run([sys.executable,"analyze_ftmoe_protocol042.py","--science-root",science,"--manifest",manifest,
         "--fixture-report",science/"fixture_report.json","--output-dir",analysis,"--docs-output",docs,
         "--run-id",a.science_run_id],repo)
    shutil.copy2(docs,analysis/"PROTOCOL042_RESULTS.md")
    run([sys.executable,"maintenance/protocol042_finalize.py","artifact-index","--info",a.raw_info,
         "--raw-manifest",science/"scientific_raw_manifest.json","--out",analysis/"artifact_index.json",
         "--run-id",a.science_run_id],repo)
    run([sys.executable,"maintenance/protocol042_finalize.py","manifest","--root",analysis,"--out",analysis/"analysis_manifest.json"],repo)
    status=J(analysis/"analysis_status.json")
    assert status["protocol"]=="042" and status["revision"]==2 and type(status["validity"]) is bool
    rec2=J(analysis/"analysis_recovery.json")
    assert rec2["science_reexecuted"] is False and rec2["scientific_metrics_or_thresholds_changed"] is False
    compact=repo/f"artifacts/ftmoe_online/protocol_042/runs/run_{a.science_run_id}"
    if compact.exists(): shutil.rmtree(compact)
    run([sys.executable,"maintenance/protocol042_finalize.py","compact","--science",science,"--analysis",analysis,"--dst",compact],repo)
    W(compact/"publication_status.json",{"protocol":"042","revision":2,"run_id":str(a.science_run_id),
      "analysis_recovery_run_id":str(a.recovery_run_id),"publication_status":"pending_main_sync","main_synced":False,"science_reexecuted":False})
    run([sys.executable,"maintenance/protocol042_finalize.py","manifest","--root",compact,"--out",compact/"compact_manifest.json"],repo)
    write_pointers(repo,status,a.science_run_id,a.recovery_run_id)
    print(json.dumps({"science_complete":True,"science_reexecuted":False,"total_optimizer_steps":led["total_optimizer_steps"],
      "analysis_status":status,"recovery":rec,"compact":str(compact)},indent=2,ensure_ascii=False))

if __name__=="__main__": main()
