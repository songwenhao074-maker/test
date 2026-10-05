"""Protocol-044 revision2 publication writer. Never overwrites the immutable revision1 report."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
R1_RESULTS=Path("docs/PROTOCOL044_RESULTS.md")
R2_RESULTS=Path("docs/PROTOCOL044_REVISION002_RESULTS.md")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def prepare(root,compact,status_path,run_id,r1_sha):
    root=Path(root);compact=Path(compact);st=J(status_path);rid=str(run_id)
    if (st.get("protocol"),st.get("revision"),st.get("execution_revision"))!=("044",1,2):raise RuntimeError("analysis status identity")
    if sha(root/R1_RESULTS)!=str(r1_sha):raise RuntimeError("immutable r1 report changed")
    if not (root/R2_RESULTS).is_file():raise FileNotFoundError(root/R2_RESULTS)
    label=st["system_result_label"]
    pub={"protocol":"044","revision":1,"execution_revision":2,"science_run_id":rid,"publication_status":"pending_main_sync",
      "main_synced":False,"science_reexecuted":False,"r1_results_sha256":str(r1_sha),"r2_results_sha256":sha(root/R2_RESULTS)}
    W(compact/"publication_status.json",pub)
    (root/"AGENTS.md").write_text(
      "# Active Experiment Directive — Protocol-044 revision 2 completed\n\n"
      f"Protocol-044 revision2 durable-recovery science run {rid} completed. Label: {label}. E_recovery_gate={st['E_gate']}; S_validity={st['S_validity']}; D_over_C={st['D_over_C']}; lifecycle={st['lifecycle_on_stream']}; reuse={st['reuse_on_stream']}; gc_on_stream={st['gc_on_stream']}; gc_closed_loop={st['gc_admission_closed_loop']}; new_stream_core_supported={st['new_stream_core_supported']}; full_lifecycle_goal_completed={st['full_lifecycle_goal_completed']}.\n\n"
      f"Read docs/PROTOCOL044_REVISION002_RESULTS.md and artifacts/ftmoe_online/protocol_044/revision_002/runs/run_{rid}/. The revision1 report remains immutable.\n\n"
      "Stop after Protocol-044 revision2. Do not regenerate seed4401, add streams/seeds/arms/F, change thresholds, force GC, or start Protocol045 without a new preregistered instruction and user handoff.\n",encoding="utf8")
    (root/"NEXT_EXPERIMENT_LATEST.md").write_text(
      "# Protocol-044 revision 2 已完成，停止等待分析\n\n"
      f"science run {rid} 完成。标签 **{label}**；E_recovery_gate={st['E_gate']}；S_validity={st['S_validity']}；D_over_C={st['D_over_C']}；lifecycle_on_stream={st['lifecycle_on_stream']}；reuse_on_stream={st['reuse_on_stream']}；gc_on_stream={st['gc_on_stream']}；gc_admission_closed_loop={st['gc_admission_closed_loop']}。\n\n"
      f"读取 docs/PROTOCOL044_REVISION002_RESULTS.md 与 artifacts/ftmoe_online/protocol_044/revision_002/runs/run_{rid}/。revision1 历史报告保持不变。到此停止，不自动启动045。\n",encoding="utf8")
    (root/"PROJECT_CONTEXT_LATEST.md").write_text(
      "# Project Context Latest\n\n"
      f"Latest completed execution is Protocol-044 revision2 run {rid}. Label **{label}**. E_recovery_gate={st['E_gate']}; S_validity={st['S_validity']}; D_over_C={st['D_over_C']}; lifecycle={st['lifecycle_on_stream']}; reuse={st['reuse_on_stream']}; gc_on_stream={st['gc_on_stream']}; gc_closed_loop={st['gc_admission_closed_loop']}.\n\n"
      "Protocol-044 revision1 remains immutable historical incomplete evidence. No automatic next protocol.\n",encoding="utf8")
    (root/"docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
      "# 当前交接：Protocol-044 revision 2 已完成\n\n"
      f"science run {rid} 已完成。标签 **{label}**。结果见 [PROTOCOL044_REVISION002_RESULTS.md](PROTOCOL044_REVISION002_RESULTS.md)。revision1 [PROTOCOL044_RESULTS.md](PROTOCOL044_RESULTS.md) 保持只读。\n\n"
      "到此停止，不自动补跑、重建seed4401或启动045。\n",encoding="utf8")
    (root/"README.md").write_text(
      "# PreGAN+ / FT-MoE 在线实验\n\n"
      f"**最新完成：Protocol-044 revision2 durable recovery，science run {rid}。**\n\n"
      f"结果标签 **{label}**；S_validity={st['S_validity']}；D_over_C={st['D_over_C']}；new_stream_core_supported={st['new_stream_core_supported']}；full_lifecycle_goal_completed={st['full_lifecycle_goal_completed']}。\n\n"
      "[r2结果](docs/PROTOCOL044_REVISION002_RESULTS.md) / [r1历史](docs/PROTOCOL044_RESULTS.md) / [交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)\n",encoding="utf8")
    if sha(root/R1_RESULTS)!=str(r1_sha):raise RuntimeError("r1 report changed during prepare")
def sync(compact,run_id,r1_results,r1_sha):
    p=Path(compact)/"publication_status.json";x=J(p)
    if (x.get("protocol"),x.get("execution_revision"),str(x.get("science_run_id")))!=("044",2,str(run_id)):raise RuntimeError("publication identity")
    if sha(r1_results)!=str(r1_sha):raise RuntimeError("immutable r1 report changed before sync")
    x.update({"publication_status":"synced_to_main","main_synced":True,"science_reexecuted":False});W(p,x)
def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("prepare");p.add_argument("--root",required=True);p.add_argument("--compact",required=True);p.add_argument("--analysis-status",required=True);p.add_argument("--run-id",required=True);p.add_argument("--r1-sha",required=True)
    p=sp.add_parser("sync");p.add_argument("--compact",required=True);p.add_argument("--run-id",required=True);p.add_argument("--r1-results",required=True);p.add_argument("--r1-sha",required=True)
    a=ap.parse_args()
    prepare(a.root,a.compact,a.analysis_status,a.run_id,a.r1_sha) if a.cmd=="prepare" else sync(a.compact,a.run_id,a.r1_results,a.r1_sha)
if __name__=="__main__":main()
