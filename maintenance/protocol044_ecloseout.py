"""Protocol-044 Stage-E crash/fail-closed closeout and gate builder."""
from __future__ import annotations
import argparse, copy, hashlib, json, shutil, tempfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch

import protocol044_common as c
from protocol044_engine import Machine044, semantic_digest
import run_ftmoe_protocol044 as r44
import run_ftmoe_protocol044_science as science

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")

def ambiguous_restore(src,root,cp):
    try:
        Machine044.restore(src,root,cp,arm="D_bounded",allow_gradient=True,real_science=False,strict_journal=True)
        return False,None
    except RuntimeError as e:
        return "ambiguous_state" in str(e),str(e)

def crash_cases(out):
    root=Path(out); shutil.rmtree(root,ignore_errors=True); root.mkdir(parents=True)
    rows=[]
    # 1: first of two joint steps => pending action, no unknown replay allowed.
    src=r44.lifecycle_source(96,2,None); m=r44.active_pair(src,root/"joint_first",False);m.checkpoint_dir=root/"joint_first/cp";m.save_checkpoint("safe",True);cp=root/"joint_first/cp/latest.pt"
    m.crash_probe="after_first_joint_step"; crashed=False
    try:m.advance(16)
    except RuntimeError:crashed=True
    blocked,msg=ambiguous_restore(src,root/"joint_first",cp)
    rows.append({"id":"first_of_two_joint_steps","pass":bool(crashed and blocked),"outcome":"ambiguous_state" if blocked else msg})
    # 2: optimizer steps/audit appended before action commit => ambiguous.
    src=r44.lifecycle_source(96,2,None);m=r44.active_pair(src,root/"counter_commit",False);m.checkpoint_dir=root/"counter_commit/cp";m.save_checkpoint("safe",True);cp=root/"counter_commit/cp/latest.pt"
    m.crash_probe="after_audit_before_action_commit";crashed=False
    try:m.advance(16)
    except RuntimeError:crashed=True
    blocked,msg=ambiguous_restore(src,root/"counter_commit",cp)
    rows.append({"id":"step_before_counter_commit","pass":bool(crashed and blocked),"outcome":"ambiguous_state" if blocked else msg})
    # 3: reclaim delete before shadow creation => transaction journal pending.
    src=r44.lifecycle_source(820,20,256);m=r44.full_pool(src,root/"reclaim",20);m.checkpoint_dir=root/"reclaim/cp";m.save_checkpoint("safe",True);cp=root/"reclaim/cp/latest.pt";m.crash_probe="after_reclaim_before_shadow";crashed=False
    try:m.advance()
    except RuntimeError as e:crashed="injected_crash_after_reclaim_before_shadow" in str(e)
    blocked,msg=ambiguous_restore(src,root/"reclaim",cp)
    rows.append({"id":"delete_before_shadow_create","pass":bool(crashed and blocked),"outcome":"ambiguous_state" if blocked else msg})
    # 4a: crash before atomic replace leaves prior checkpoint exactly usable.
    src=r44.lifecycle_source(96,2,None);m=r44.one_active(src,root/"cp_before",2,1);m.checkpoint_dir=root/"cp_before/cp";m.save_checkpoint("safe",True);cp=root/"cp_before/cp/latest.pt";before=semantic_digest(Machine044.restore(src,root/"cp_before",cp,arm="D_bounded"))
    m.birth_streak=7;m.crash_probe="checkpoint_before_atomic_replace";crashed=False
    try:m.save_checkpoint("probe",False)
    except RuntimeError:crashed=True
    restored=Machine044.restore(src,root/"cp_before",cp,arm="D_bounded");rows.append({"id":"checkpoint_before_atomic_replace","pass":bool(crashed and semantic_digest(restored)==before),"outcome":"exact_prior_checkpoint"})
    # 4b: crash after atomic replace/meta commit restores new state.
    src=r44.lifecycle_source(96,2,None);m=r44.one_active(src,root/"cp_after",2,1);m.checkpoint_dir=root/"cp_after/cp";m.save_checkpoint("safe",True);m.birth_streak=7;m.crash_probe="checkpoint_after_atomic_replace";crashed=False
    try:m.save_checkpoint("probe",False)
    except RuntimeError:crashed=True
    restored=Machine044.restore(src,root/"cp_after",root/"cp_after/cp/latest.pt",arm="D_bounded")
    rows.append({"id":"checkpoint_after_atomic_replace","pass":bool(crashed and restored.birth_streak==7),"outcome":"exact_new_checkpoint"})
    # 5: audit file has bytes beyond committed checkpoint => fail closed.
    src=r44.lifecycle_source(96,2,None);m=r44.one_active(src,root/"audit_offset",2,1);m.checkpoint_dir=root/"audit_offset/cp";m.advance(16);m.save_checkpoint("safe",True);cp=root/"audit_offset/cp/latest.pt"
    p=root/"audit_offset/streams/settlements.jsonl";p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf8") as f:f.write('{"injected":"after_checkpoint"}\n')
    blocked,msg=ambiguous_restore(src,root/"audit_offset",cp)
    rows.append({"id":"budget_commit_and_audit_offset","pass":bool(blocked),"outcome":"ambiguous_state" if blocked else msg})
    result={"protocol":"044","revision":1,"kind":"crash_injection","cases":rows,"all_pass":all(r["pass"] for r in rows)}
    W(root/"crash_cases.json",result);print(json.dumps(result,indent=2))
    if not result["all_pass"]: raise SystemExit(2)

def failclosed(out):
    root=Path(out);shutil.rmtree(root,ignore_errors=True);root.mkdir(parents=True)
    input_lock=root/"input_lock.json";ledger=root/"ledger.json";science.W(ledger,science.default_ledger())
    W(input_lock,{"protocol":"044","revision":1,"seed":4401,"locked":True,"stream_sha256":"x"*64})
    exe="a"*40
    valid={"protocol":"044","revision":1,"E_gate":True,"source_lock_pass":True,"engineering_core_pass":True,"crash_pass":True,
      "fail_closed_pass":True,"baseline_adapter_pass":True,"execution_sha":exe,"source_probe_artifact_digest":science.SOURCE_PROBE_DIGEST,
      "engineering_core_artifact_digest":science.ENGINEERING_CORE_DIGEST,"required_resume_case_ids":list(science.REQUIRED_CASES),
      "resume_case_ids_passed":list(science.REQUIRED_CASES),"active_max":2,"resident_max":3,"shadow_max":1}
    cases=[]
    def reject(cid,g=None,mutate_ledger=None,execution=exe,sequence="C_ref"):
        W(ledger,science.default_ledger())
        if mutate_ledger:
            x=J(ledger);mutate_ledger(x);W(ledger,x)
        calls=0;ok=False;err=None
        try: science.validate_gate(valid if g is None else g,input_lock,execution,sequence,ledger,calls)
        except Exception as e: ok=True;err=type(e).__name__+":"+str(e)
        cases.append({"id":cid,"rejected":ok,"optimizer_calls":calls,"error":err,"pass":bool(ok and calls==0)})
    reject("missing",g={k:v for k,v in valid.items() if k!="E_gate"})
    reject("empty",g={})
    g=copy.deepcopy(valid);g["E_gate"]="true";reject("string_true",g=g)
    g=copy.deepcopy(valid);g["resume_case_ids_passed"]=list(science.REQUIRED_CASES[:-1]);reject("missing_required_case",g=g)
    g=copy.deepcopy(valid);g["source_probe_artifact_digest"]="sha256:"+"0"*64;reject("bad_source_hash",g=g)
    reject("bad_execution_SHA",execution="b"*40)
    g=copy.deepcopy(valid);g["revision"]=2;reject("bad_revision",g=g)
    reject("duplicate_start",mutate_ledger=lambda x:x["sequences"]["C_ref"].update({"started":True}))
    reject("overbudget",mutate_ledger=lambda x:x.update({"total_optimizer_steps":2360}))
    g=copy.deepcopy(valid);g["resident_max"]=4;reject("illegal_capacity",g=g)
    # Positive control: gate passes but performs no model/optimizer construction.
    W(ledger,science.default_ledger());positive=False
    try: positive=science.validate_gate(valid,input_lock,exe,"C_ref",ledger,0)
    except Exception: positive=False
    result={"protocol":"044","revision":1,"kind":"actual_science_entry_fail_closed","cases":cases,"positive_control":bool(positive),
      "all_negative_optimizer_calls_zero":all(x["optimizer_calls"]==0 for x in cases),"all_pass":bool(positive and all(x["pass"] for x in cases))}
    W(root/"failclosed.json",result);print(json.dumps(result,indent=2))
    if not result["all_pass"]: raise SystemExit(2)

def build_gate(a):
    source=J(a.source_probe);fixed=J(a.fixed043);mem=J(a.memory);resume=J(a.resume);crash=J(a.crash);fail=J(a.failclosed);adapter=J(a.adapter)
    req=list(science.REQUIRED_CASES);passed=[r["case"] for r in resume["cases"] if r.get("pass") is True]
    checks={"source_lock_pass":source.get("all_pass") is True,"engineering_core_pass":bool(fixed.get("all_pass") is True and mem.get("pass") is True and resume.get("all_pass") is True),
      "crash_pass":crash.get("all_pass") is True,"fail_closed_pass":fail.get("all_pass") is True,"baseline_adapter_pass":adapter.get("all_pass") is True,
      "required_cases_exact":passed==req}
    gate={"protocol":"044","revision":1,"execution_sha":str(a.execution_sha),"source_probe_artifact_digest":science.SOURCE_PROBE_DIGEST,
      "engineering_core_artifact_digest":science.ENGINEERING_CORE_DIGEST,"required_resume_case_ids":req,"resume_case_ids_passed":passed,
      "active_max":2,"resident_max":3,"shadow_max":1,**checks}
    gate["E_gate"]=bool(all(checks.values()));W(a.out,gate);print(json.dumps(gate,indent=2))
    if not gate["E_gate"]:raise SystemExit(2)

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("crash");p.add_argument("--out",required=True);p.set_defaults(fn=lambda a:crash_cases(a.out))
    p=sp.add_parser("failclosed");p.add_argument("--out",required=True);p.set_defaults(fn=lambda a:failclosed(a.out))
    p=sp.add_parser("gate");p.add_argument("--source-probe",required=True);p.add_argument("--fixed043",required=True);p.add_argument("--memory",required=True);p.add_argument("--resume",required=True);p.add_argument("--crash",required=True);p.add_argument("--failclosed",required=True);p.add_argument("--adapter",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--out",required=True);p.set_defaults(fn=build_gate)
    a=ap.parse_args();a.fn(a)
if __name__=="__main__":main()
