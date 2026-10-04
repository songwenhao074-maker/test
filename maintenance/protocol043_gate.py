"""Build Protocol-043 frozen implementation manifest and fail-closed science gate."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"artifacts/ftmoe_online/protocol_043/plan.json"
PLAN_SHA="9cd9b6f63aed0e35361f763e532d7c785a7838c19d172ea0c97bbaae5b2b0d6b"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def bt(x): return type(x) is bool and x is True
def require(x,label):
    if not x: raise RuntimeError(label)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-report",required=True); ap.add_argument("--audit-gate",required=True); ap.add_argument("--fixture-report",required=True)
    ap.add_argument("--execution-sha",required=True); ap.add_argument("--workflow",required=True); ap.add_argument("--pip-freeze",required=True)
    ap.add_argument("--implementation-out",required=True); ap.add_argument("--gate-out",required=True); ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    require(sha(PLAN)==PLAN_SHA,"plan hash")
    p=J(PLAN); require((p.get("protocol"),p.get("revision"))==("043",1),"plan identity")
    require(sha(ROOT/p["instructions"])==p["instructions_sha256"],"directive hash")
    src=J(a.source_report); audit=J(a.audit_gate); fixture=J(a.fixture_report)
    sources_valid=bool(src.get("all_locked_files_verified") is True and src.get("F_files_materialized") is False)
    audit_valid=bool((audit.get("protocol"),audit.get("revision"))==("043",1) and bt(audit.get("all_pass")) and audit.get("gradients")==0 and audit.get("model_replay") is False)
    req=p["engineering"]["required_fixture_ids"]; rows=fixture.get("fixtures")
    fixture_by={r.get("test_id"):r for r in rows} if isinstance(rows,list) else {}
    fixture_valid=bool((fixture.get("protocol"),fixture.get("revision"))==("043",1) and bt(fixture.get("all_pass")) and set(fixture_by)==set(req)
      and all(bt(fixture_by[k].get("pass")) for k in req) and fixture.get("real_engineering_prefixes")==1
      and int(fixture.get("real_engineering_prefix_intervals_max",-1))<=256 and fixture.get("real_engineering_gradient_steps")==0)
    # Immutable historical bundle locks from the registered plan.
    locked={}
    historical_ok=True
    for rel,d in p["kernel"]["files"].items():
        q=ROOT/rel; got=None if not q.exists() else sha(q); locked[rel]={"expected":d,"actual":got,"pass":got==d}; historical_ok=historical_ok and got==d
    for rel,d in p["source042"]["files"].items():
        q=ROOT/rel; got=None if not q.exists() else sha(q); locked["source042:"+rel]={"expected":d,"actual":got,"pass":got==d}; historical_ok=historical_ok and got==d
    bundle_paths=[
      "protocol043_common.py","protocol043_engine.py","run_ftmoe_protocol043.py","analyze_ftmoe_protocol043.py",
      "maintenance/validate_protocol043_plan.py","maintenance/materialize_protocol043_sources.py","maintenance/protocol043_audit042.py",
      "maintenance/protocol043_gate.py","maintenance/protocol043_finalize.py",
      "artifacts/ftmoe_online/protocol_043/plan.json","artifacts/ftmoe_online/protocol_043/plan.sha256",
      "docs/PROTOCOL043_BOUNDED_LIFECYCLE_20261003.md",a.workflow
    ]
    bundle={}
    for rel in bundle_paths:
        q=ROOT/rel; require(q.exists(),"source bundle missing "+rel); bundle[rel]=sha(q)
    execution_sha=str(a.execution_sha)
    execution_sha_valid=len(execution_sha)==40 and all(ch in "0123456789abcdef" for ch in execution_sha)
    budget=p["budget"]
    budget_valid=bool(budget["science_sequences"]==2 and budget["each"]["total_optimizer_steps"]==768 and budget["total"]["optimizer_steps"]==1536
      and budget["new_streams"]==budget["extra_seeds"]==budget["F_loads"]==budget["extra_arms"]==budget["U_reproduction_runs"]==0
      and budget["restart_from_zero"] is False)
    plan_valid=True
    source_bundle_valid=bool(historical_ok and bundle)
    all_pass=bool(plan_valid and sources_valid and audit_valid and fixture_valid and budget_valid and source_bundle_valid and execution_sha_valid)
    impl={"protocol":"043","revision":1,"run_id":str(a.run_id),"execution_sha":execution_sha,"plan_sha256":PLAN_SHA,
      "directive_sha256":p["instructions_sha256"],"workflow_sha256":sha(ROOT/a.workflow),"pip_freeze_sha256":sha(a.pip_freeze),
      "python":sys.version,"platform":platform.platform(),"source_bundle":bundle,"historical_locked_files":locked,
      "fixture_report_sha256":sha(a.fixture_report),"audit042_gate_sha256":sha(a.audit_gate),"source_report_sha256":sha(a.source_report),
      "science_started":False,"F_loaded":False}
    gate={"protocol":"043","revision":1,"run_id":str(a.run_id),"execution_sha":execution_sha,
      "plan_valid":bool(plan_valid),"sources_valid":bool(sources_valid),"audit042_valid":bool(audit_valid),"fixture_valid":bool(fixture_valid),
      "budget_valid":bool(budget_valid),"source_bundle_valid":bool(source_bundle_valid),"execution_sha_valid":bool(execution_sha_valid),
      "all_pass":bool(all_pass),"science_sequences_authorized":["D_no_gc","D_bounded"],"historical033_042_read_only":True,
      "real_engineering_gradients":0,"no_force_push":True,"restart_from_zero":False}
    W(a.implementation_out,impl); W(a.gate_out,gate)
    print(json.dumps(gate,indent=2))
    if not all_pass: raise RuntimeError("Protocol043 gate failed")
if __name__=="__main__": main()
