"""Finalize Protocol-030 paired passive-diagnostic validation."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

def read_json(path,default=None):
    p=Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else default

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def fmt(v,d=9):
    return "NA" if v is None else f"{float(v):.{d}f}"

def ensure_status(root,run_id):
    root=Path(root); s=read_json(root/"status.json")
    if s is not None:return s
    s={"protocol":"030","run_id":str(run_id),"completed":False,"paired_audit_valid":None,
       "new_scientific_methods":0,"C_rerun":False,"automatic_followups_started":[],
       "blocker":"execution_stopped_before_protocol030_status"}
    root.mkdir(parents=True,exist_ok=True)
    (root/"status.json").write_text(json.dumps(s,indent=2)+"\n",encoding="utf8")
    return s

def build_results(status,pair,hist,diag,artifact_url,digest):
    lines=["# Protocol-030 Results","",
      f"GitHub Actions run: {status.get('run_id')}.",
      "Task: same-job audit_off/audit_on paired noninterference validation for the unchanged Protocol-028 D policy.",
      f"Completed: {status.get('completed',False)}. Paired audit valid: {status.get('paired_audit_valid')}.",
      "C rerun: False. New scientific methods: 0.",""]
    if not status.get("completed") or pair is None:
        lines += ["## Blocker","",str(status.get("blocker") or "paired execution incomplete"),"",
                  "No additional full replay, seed, method, threshold change or parameter search was started."]
        return lines
    lines += ["## Paired validity gate","",
      f"- Detection probability max abs error: {fmt(pair['detection_probability']['max_abs_error'],12)} (required <=1e-6).",
      f"- Class probability max abs error: {fmt(pair['class_probability']['max_abs_error'],12)} (required <=1e-6).",
      f"- Detection exact-equal fraction: {fmt(pair['detection_probability']['exact_equal_fraction'],12)}.",
      f"- Class exact-equal fraction: {fmt(pair['class_probability']['exact_equal_fraction'],12)}.",
      f"- Labels/raw labels exact: {pair['labels_exact']}/{pair['raw_labels_exact']}.",
      f"- Nine-window mean AP abs difference: {fmt(pair['mean_recurrence_ap_abs_difference'],12)} (required <=1e-6).",
      f"- Full-stream AP abs difference: {fmt(pair['full_ap_abs_difference'],12)} (required <=1e-6).",
      f"- Discrete lifecycle exact: {pair['discrete_lifecycle_events_exact']}.",
      f"- Final topology exact: {pair['final_topology_exact']}.",
      f"- State trace exact: {pair['state_trace_exact']}.",
      f"- Passive state violations: {pair['passive_state_violations']}.",""]
    if not pair["paired_audit_valid"]:
        lines += ["## Main finding","",
                  "The same-job paired noninterference gate failed. Protocol-030 therefore does not validate the passive memory-utility diagnostic.",
                  "First registered divergence: "+json.dumps(pair.get("first_divergence"),ensure_ascii=False)+".","",
                  "Historical Protocol-029 remains invalid; no diagnostic memory-utility claim is promoted from this run.",""]
    else:
        opp=(diag or {}).get("opportunities",{})
        passing=int((diag or {}).get("all_gates_passing_candidate_count",0))
        if passing:
            finding=(f"The paired noninterference gate passed. The valid passive audit found {passing} candidate-opportunity records "
                     "that satisfy all original reuse gates. They remain post_hoc_oracle_diagnostic evidence only, not an online D>C result.")
        else:
            finding=("The paired noninterference gate passed, but no dormant-memory candidate satisfied all original reuse gates "
                     "on this trajectory; reusable frozen-memory utility remains unsupported here.")
        lines += ["## Main finding","",finding,"",
                  "## Opportunity audit","",
                  f"- Reuse due opportunities: {opp.get('reuse_due_opportunities')}.",
                  f"- No-memory opportunities: {opp.get('no_memory_opportunities')}.",
                  f"- Opportunities with any all-gates-pass candidate: {opp.get('opportunities_with_any_all_gates_candidate')}.",
                  f"- Passing opportunities blocked by busy: {opp.get('passing_candidate_opportunities_blocked_by_busy')}.",
                  f"- Passing opportunities blocked by similarity/rank: {opp.get('passing_candidate_opportunities_blocked_by_similarity_or_rank')}.",
                  f"- Original selected candidate passed: {opp.get('original_selected_candidate_passed')}.",""]
    lines += ["## Nine recurrence first100 paired AP checks","",
              "| Window | audit_off AP | audit_on AP | abs diff |","|---|---:|---:|---:|"]
    for row in pair.get("recurrence_first100",[]):
        lines.append(f"| {row['phase']} | {fmt(row['audit_off']['ap'])} | {fmt(row['audit_on']['ap'])} | {fmt(row['ap_abs_difference'],12)} |")
    lines += ["","## Historical Protocol-028 comparison",""]
    for mode in ("audit_off","audit_on"):
        h=(hist or {}).get(mode,{})
        lines += [
          f"### {mode}","",
          f"- historical_028_match: {h.get('historical_028_match')}.",
          f"- Detection probability max abs error: {fmt((h.get('detection_probability') or {}).get('max_abs_error'),12)}.",
          f"- Class probability max abs error: {fmt((h.get('class_probability') or {}).get('max_abs_error'),12)}.",
          f"- Nine-window mean AP abs difference: {fmt(h.get('mean_recurrence_ap_abs_difference'),12)}.",
          f"- Full AP abs difference: {fmt(h.get('full_ap_abs_difference'),12)}.",
          f"- Lifecycle exact: {h.get('discrete_lifecycle_events_exact')}.",""]
    if diag:
        cost=diag.get("cost",{})
        lines += ["## Runtime and diagnostic overhead","",
          f"- audit_off wall seconds: {fmt((cost.get('audit_off') or {}).get('wall_seconds'),6)}.",
          f"- audit_on wall seconds: {fmt((cost.get('audit_on') or {}).get('wall_seconds'),6)}.",
          f"- Passive prediction overhead seconds: {fmt(cost.get('passive_prediction_overhead_seconds'),6)}.",
          f"- Passive guard overhead seconds: {fmt(cost.get('passive_guard_overhead_seconds'),6)}.",
          "- Protocol-030 uses a preregistered single-thread deterministic CPU profile; these times are not a new D-vs-C efficiency comparison.",""]
    lines += ["## Interpretation boundary","",
      "Passing the paired gate would establish noninterference only under the registered Protocol-030 runtime profile. Historical 028 matching is separate. Post-hoc oracle candidates and busy counterfactuals are not deployable policy results or formal D>C evidence.","",
      "## Evidence","",f"- Artifact: {artifact_url or 'see workflow run artifact'}",f"- Artifact SHA-256 digest: {digest or 'NA'}"]
    return lines

def update_entrypoints(status,pair,diag):
    valid=bool(status.get("completed") and status.get("paired_audit_valid"))
    if valid:
        n=int((diag or {}).get("all_gates_passing_candidate_count",0))
        finding=(f"030同机配对无干扰门禁通过；有效旁路诊断发现{n}条满足全部原复用门槛的候选记录。"
                 "这些记录仍是post-hoc oracle诊断，不是线上D>C证据。")
    else:
        finding=(f"030已执行但同机配对无干扰门禁未通过；blocker={status.get('blocker')}。"
                 "因此旁路记忆效用诊断仍不能升级为有效科学结论。")
    Path("AGENTS.md").write_text(
      "# Current task: Protocol-030 completed\n\nRead NEXT_EXPERIMENT_LATEST.md and docs/GITHUB_EXPERIMENT_HANDOFF.md.\n"
      "Protocol-030 paired audit_off/audit_on task has reached its registered stop point. There is no registered follow-up experiment.\n"
      "Preserve Protocol-027/028/029/030 registrations, source artifacts and failed gates. Do not automatically launch further replays, methods, seeds, "
      "ablations, threshold changes, scenario redesign or parameter search. Wait for an explicit new directive.\n",encoding="utf8")
    Path("NEXT_EXPERIMENT_LATEST.md").write_text(
      "# 当前状态：Protocol-030已结束，停止自动实验\n\n"+finding+
      "\n\n结果见[Protocol-030 Results](docs/PROTOCOL030_RESULTS.md)。当前没有已登记的下一项科学实验；等待新的明确指示。\n",encoding="utf8")
    Path("PROJECT_CONTEXT_LATEST.md").write_text(
      "# 当前项目上下文：Protocol-030已结束\n\n"+finding+
      "\n\n030没有重跑C，没有改变028线上策略、数据、阈值或seed。结果与证据边界见docs/PROTOCOL030_RESULTS.md。\n",encoding="utf8")
    Path("docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
      "# 当前交接：Protocol-030已结束\n\n以main为准。Protocol-030同机audit_off/audit_on配对任务已执行并到达停止点。\n\n"
      +finding+"\n\n结果：docs/PROTOCOL030_RESULTS.md。当前没有已登记后续实验；不要自动追加完整回放、算法修改、种子、消融、调参或场景修改。\n",encoding="utf8")
    Path("README.md").write_text(
      "# PreGAN+ / FT-MoE 在线实验\n\n目标：在业务更替与复现场景中检验动态残差专家D，同时严格保留负结果和诊断有效性边界。\n\n"
      "**当前状态：Protocol-030同机成对旁路无干扰验证已结束；暂无登记的下一实验。**\n\n"
      "## 最新结果\n\n"+finding+" 详见[Protocol-030 Results](docs/PROTOCOL030_RESULTS.md)。\n\n"
      "030只运行两次原Protocol-028 D（audit_off/audit_on），固定单线程确定性CPU配置；未重跑C/A/B，未改数据、阈值、seed或线上策略。\n\n"
      "历史有效D/C结论仍见[Protocol-028 Results](docs/PROTOCOL028_RESULTS.md)；029无效诊断和030结果均保留。当前等待新的明确实验指示。\n",encoding="utf8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",required=True); ap.add_argument("--git-dest",required=True)
    ap.add_argument("--run-id",required=True); ap.add_argument("--repository",required=True)
    ap.add_argument("--artifact-id",default=""); ap.add_argument("--artifact-url",default="")
    ap.add_argument("--artifact-digest",default=""); ap.add_argument("--status-only",action="store_true")
    a=ap.parse_args()
    root=Path(a.run_root); dest=Path(a.git_dest); dest.mkdir(parents=True,exist_ok=True)
    status=ensure_status(root,a.run_id)
    if a.status_only:return
    pair=read_json(root/"paired_consistency.json"); hist=read_json(root/"historical_comparison.json")
    diag=read_json(root/"diagnostic_summary.json")
    Path("docs/PROTOCOL030_RESULTS.md").write_text(
      "\n".join(build_results(status,pair,hist,diag,a.artifact_url,a.artifact_digest))+"\n",encoding="utf8")
    update_entrypoints(status,pair,diag)
    keep=["status.json","paired_consistency.json","historical_comparison.json","first_divergence.json",
          "diagnostic_summary.json","opportunity_audit.json","environment.json","source_artifacts.json",
          "eligibility.json","data_lock.json","workflow_execution.json"]
    for name in keep:
        src=root/name
        if src.is_file():shutil.copy2(src,dest/name)
    files={}
    for p in sorted(dest.rglob("*")):
        if p.is_file():files[str(p.relative_to(dest))]={"sha256":sha256(p),"bytes":p.stat().st_size}
    idx={"protocol":"030","github_run_id":int(a.run_id),
         "workflow_url":f"https://github.com/{a.repository}/actions/runs/{a.run_id}",
         "artifact_id":a.artifact_id or None,"artifact_url":a.artifact_url or None,
         "artifact_digest":a.artifact_digest or None,"artifact_name":f"protocol030-paired-diagnostic-{a.run_id}",
         "retention_days":90,"files_committed_to_git":files,
         "raw_evidence":"Actions artifact contains both full predictions, lifecycle logs, state traces and audit_on passive predictions."}
    (dest/"ARTIFACT_INDEX.json").write_text(json.dumps(idx,indent=2)+"\n",encoding="utf8")

if __name__=="__main__":main()
