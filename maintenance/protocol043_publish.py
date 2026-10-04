"""Protocol-043 terminal pointer/publication status writer."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")

def prepare(root,compact,status_path,run_id):
    root=Path(root); compact=Path(compact); st=J(status_path)
    if (st.get("protocol"),st.get("revision"))!=("043",1): raise RuntimeError("analysis status identity")
    for k in ("validity","D_over_C","core_lifecycle_exercised","gc_engineering_verified","gc_exercised_on_stream","resource_bounds_pass","core_goal_supported","full_reclamation_demonstrated"):
        if type(st.get(k)) is not bool: raise RuntimeError("nonboolean analysis status "+k)
    label=st["system_result_label"]; rid=str(run_id)
    pub={"protocol":"043","revision":1,"run_id":rid,"publication_status":"pending_main_sync","main_synced":False,"science_reexecuted":False}
    W(compact/"publication_status.json",pub)
    (root/"AGENTS.md").write_text(
      "# Active Experiment Directive — Protocol-043 revision 1 completed\n\n"
      f"Protocol-043 revision 1 science run {rid} completed. System label: {label}. Validity: {st['validity']}. D_over_C: {st['D_over_C']}. "
      f"Core lifecycle: {st['core_lifecycle_exercised']}. GC engineering: {st['gc_engineering_verified']}. GC on stream: {st['gc_exercised_on_stream']}. "
      f"Core goal supported: {st['core_goal_supported']}. Full reclamation demonstrated: {st['full_reclamation_demonstrated']}.\n\n"
      f"Read docs/PROTOCOL043_RESULTS.md and artifacts/ftmoe_online/protocol_043/runs/run_{rid}/. Historical033–043 scientific budgets are closed.\n\n"
      "Stop after Protocol-043 revision 1. Do not retry, add arms/seeds/streams/F, tune lifecycle thresholds, force reclamation, or automatically start Protocol-044. A new preregistered instruction and user handoff are required.\n",encoding="utf8")
    (root/"NEXT_EXPERIMENT_LATEST.md").write_text(
      "# Protocol-043 revision 1 已完成，停止等待分析\n\n"
      f"science run {rid} 已完成。系统标签 **{label}**；validity={st['validity']}；D_over_C={st['D_over_C']}；core_lifecycle_exercised={st['core_lifecycle_exercised']}；"
      f"gc_engineering_verified={st['gc_engineering_verified']}；gc_exercised_on_stream={st['gc_exercised_on_stream']}；core_goal_supported={st['core_goal_supported']}；"
      f"full_reclamation_demonstrated={st['full_reclamation_demonstrated']}。\n\n"
      f"读取 docs/PROTOCOL043_RESULTS.md 与 artifacts/ftmoe_online/protocol_043/runs/run_{rid}/。两条登记科学序列已关闭。\n\n"
      "不自动重跑043，不加seed/流/额外臂/F，不调窗口或回收阈值，不强制制造GC，不启动Protocol-044。\n",encoding="utf8")
    (root/"PROJECT_CONTEXT_LATEST.md").write_text(
      "# Project Context Latest\n\n"
      f"Latest bounded science is Protocol-043 revision 1 run {rid}. System label: **{label}**. Validity={st['validity']}; D_over_C={st['D_over_C']}; "
      f"core lifecycle={st['core_lifecycle_exercised']}; GC engineering={st['gc_engineering_verified']}; GC on stream={st['gc_exercised_on_stream']}; "
      f"core_goal_supported={st['core_goal_supported']}; full_reclamation_demonstrated={st['full_reclamation_demonstrated']}.\n\n"
      f"Read docs/PROTOCOL043_RESULTS.md and artifacts/ftmoe_online/protocol_043/runs/run_{rid}/. 042 remains historical invalid_execution and was audited read-only under the 043 compact evidence.\n\n"
      "Historical033–043 budgets are closed. Any retry or next protocol requires new preregistration and user handoff.\n",encoding="utf8")
    (root/"docs/GITHUB_EXPERIMENT_HANDOFF.md").write_text(
      "# 当前交接：Protocol-043 revision 1 已完成\n\n"
      f"science run {rid} 已完成。系统标签 **{label}**；validity={st['validity']}；D_over_C={st['D_over_C']}；core lifecycle={st['core_lifecycle_exercised']}；"
      f"GC engineering={st['gc_engineering_verified']}；GC on stream={st['gc_exercised_on_stream']}；core_goal_supported={st['core_goal_supported']}；"
      f"full reclamation demonstrated={st['full_reclamation_demonstrated']}。\n\n"
      f"结果见 [PROTOCOL043_RESULTS.md](PROTOCOL043_RESULTS.md)，紧凑证据见 ../artifacts/ftmoe_online/protocol_043/runs/run_{rid}/。\n\n"
      "到此停止。不自动补跑、调阈值、增加seed/流/额外臂/F或启动044。\n",encoding="utf8")
    (root/"README.md").write_text(
      "# PreGAN+ / FT-MoE 在线实验\n\n"
      "目标：在合理部署场景中，通过结构修改，使保留动态新增、知识保存/选择、休眠复用及容量管理思想的完整D优于持续学习C。\n\n"
      f"**最新完成：[Protocol-043 revision 1 按需新增、窗口退出与有限池回收](docs/PROTOCOL043_RESULTS.md)，science run {rid}。**\n"
      f"系统标签 **{label}**；validity={st['validity']}；D_over_C={st['D_over_C']}；core_goal_supported={st['core_goal_supported']}；"
      f"full_reclamation_demonstrated={st['full_reclamation_demonstrated']}。\n\n"
      f"[执行交接](docs/GITHUB_EXPERIMENT_HANDOFF.md) / [043紧凑证据](artifacts/ftmoe_online/protocol_043/runs/run_{rid}/) / [043冻结方案](docs/PROTOCOL043_BOUNDED_LIFECYCLE_20261003.md)\n\n"
      "结果仍属于已观察seed3601开发证据，不是独立确认。Protocol-043 revision1 到此停止。\n",encoding="utf8")

def sync(compact,run_id):
    compact=Path(compact); p=compact/"publication_status.json"; x=J(p)
    if (x.get("protocol"),x.get("revision"),str(x.get("run_id")))!=("043",1,str(run_id)): raise RuntimeError("publication identity")
    x.update({"publication_status":"synced_to_main","main_synced":True,"science_reexecuted":False}); W(p,x)

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("prepare"); p.add_argument("--root",required=True); p.add_argument("--compact",required=True); p.add_argument("--analysis-status",required=True); p.add_argument("--run-id",required=True)
    p=sp.add_parser("sync"); p.add_argument("--compact",required=True); p.add_argument("--run-id",required=True)
    a=ap.parse_args()
    if a.cmd=="prepare": prepare(a.root,a.compact,a.analysis_status,a.run_id)
    else: sync(a.compact,a.run_id)
if __name__=="__main__": main()
