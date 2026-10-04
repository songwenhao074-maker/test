"""Protocol-044 terminal publication pointer writer."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def J(p):return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def prepare(root,compact,status_path,run_id):
    root=Path(root);compact=Path(compact);st=J(status_path);rid=str(run_id)
    if (st.get("protocol"),st.get("revision"))!=("044",1):raise RuntimeError("analysis status identity")
    label=st["system_result_label"]
    pub={"protocol":"044","revision":1,"science_run_id":rid,"publication_status":"pending_main_sync","main_synced":False,"science_reexecuted":False}
    W(compact/"publication_status.json",pub)
    (root/"AGENTS.md").write_text(
      "# Active Experiment Directive — Protocol-044 revision 1 completed\n\n"
      f"Protocol-044 revision1 science run {rid} completed. Label: {label}. E_gate={st['E_gate']}; S_validity={st['S_validity']}; D_over_C={st['D_over_C']}; lifecycle={st['lifecycle_on_stream']}; reuse={st['reuse_on_stream']}; gc_on_stream={st['gc_on_stream']}; gc_closed_loop={st['gc_admission_closed_loop']}; new_stream_core_supported={st['new_stream_core_supported']}; full_lifecycle_goal_completed={st['full_lifecycle_goal_completed']}.\n\n"
      f"Read docs/PROTOCOL044_RESULTS.md and artifacts/ftmoe_online/protocol_044/runs/run_{rid}/. Historical033–044 science budgets are closed.\n\n"
      "Stop after Protocol-044 revision1. Do not reroll seed4401, add streams/seeds/arms/F, change thresholds, force GC, retrain043, or start Protocol045 without a new preregistered instruction and user handoff.\n",encoding="utf8")
    (root/"NEXT_EXPERIMENT_LATEST.md").write_text(
      "# Protocol-044 revision 1 已完成，停止等待分析\n\n"
      f"science run {rid} 完成。标签 **{label}**；E_gate={st['E_gate']}；S_validity={st['S_validity']}；D_over_C={st['D_over_C']}；lifecycle_on_stream={st['lifecycle_on_stream']}；reuse_on_stream={st['reuse_on_stream']}；gc_on_stream={st['gc_on_stream']}；gc_admission_closed_loop={st['gc_admission_closed_loop']}；new_stream_core_supported={st['new_stream_core_supported']}；full_lifecycle_goal_completed={st['full_lifecycle_goal_completed']}。\n\n"
      f"读取 docs/PROTOCOL044_RESULTS.md 与 artifacts/ftmoe_online/protocol_044/runs/run_{rid}/。到此停止，不自动启动045。\n",encoding="utf8")
    (root/"PROJECT_CONTEXT_LATEST.md").write_text(
      "# Project Context Latest\n\n"
      f"Latest completed bounded science is Protocol-044 revision1 run {rid}. Label **{label}**. E_gate={st['E_gate']}; S_validity={st['S_validity']}; D_over_C={st['D_over_C']}; lifecycle={st['lifecycle_on_stream']}; reuse={st['reuse_on_stream']}; gc_on_stream={st['gc_on_stream']}; gc_closed_loop={st['gc_admission_closed_loop']}; new_stream_core_supported={st['new_stream_core_supported']}; full_lifecycle_goal_completed={st['full_lifecycle_goal_completed']}.\n\n"
      f"See docs/PROTOCOL044_RESULTS.md and artifacts/ftmoe_online/protocol_044/runs/run_{rid}/. Historical033–044 budgets are closed. No automatic next protocol.\n",encoding="utf8")
    (root/"docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
      "# 当前交接：Protocol-044 revision 1 已完成\n\n"
      f"science run {rid} 已完成。标签 **{label}**。E_gate={st['E_gate']}；S_validity={st['S_validity']}；D_over_C={st['D_over_C']}；lifecycle={st['lifecycle_on_stream']}；reuse={st['reuse_on_stream']}；gc_on_stream={st['gc_on_stream']}；gc_closed_loop={st['gc_admission_closed_loop']}。\n\n"
      f"结果见 [PROTOCOL044_RESULTS.md](PROTOCOL044_RESULTS.md)，compact 证据见 ../artifacts/ftmoe_online/protocol_044/runs/run_{rid}/。到此停止，不自动补跑或启动045。\n",encoding="utf8")
    (root/"README.md").write_text(
      "# PreGAN+ / FT-MoE 在线实验\n\n"
      f"**最新完成：Protocol-044 revision1，science run {rid}。**\n\n"
      f"结果标签 **{label}**；E_gate={st['E_gate']}；S_validity={st['S_validity']}；D_over_C={st['D_over_C']}；new_stream_core_supported={st['new_stream_core_supported']}；full_lifecycle_goal_completed={st['full_lifecycle_goal_completed']}。\n\n"
      "[结果](docs/PROTOCOL044_RESULTS.md) / [交接](docs/GITHUB_EXPERIMENT_HANDOFF.md)\n",encoding="utf8")
def sync(compact,run_id):
    p=Path(compact)/"publication_status.json";x=J(p)
    if (x.get("protocol"),x.get("revision"),str(x.get("science_run_id")))!=("044",1,str(run_id)):raise RuntimeError("publication identity")
    x.update({"publication_status":"synced_to_main","main_synced":True,"science_reexecuted":False});W(p,x)
def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("prepare");p.add_argument("--root",required=True);p.add_argument("--compact",required=True);p.add_argument("--analysis-status",required=True);p.add_argument("--run-id",required=True)
    p=sp.add_parser("sync");p.add_argument("--compact",required=True);p.add_argument("--run-id",required=True)
    a=ap.parse_args()
    prepare(a.root,a.compact,a.analysis_status,a.run_id) if a.cmd=="prepare" else sync(a.compact,a.run_id)
if __name__=="__main__":main()
