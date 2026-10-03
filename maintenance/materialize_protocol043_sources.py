"""Materialize exact 036/041 inputs and verify immutable Protocol042 raw."""
from pathlib import Path
import argparse, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"artifacts/ftmoe_online/protocol_043/plan.json"; PLAN_SHA="9cd9b6f63aed0e35361f763e532d7c785a7838c19d172ea0c97bbaae5b2b0d6b"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def locate(root,rel,digest):
    root=Path(root); rel=Path(rel).as_posix()
    named=[p for p in root.rglob(Path(rel).name) if p.is_file() and (p.as_posix()==rel or p.as_posix().endswith("/"+rel))]
    hits=[p for p in named if sha(p)==digest]
    if len(hits)!=1: raise RuntimeError(f"{rel}: expected one suffix+sha match, found {len(hits)}")
    return hits[0]
def copy_set(src,dst,expected,label):
    rows=[]
    for rel,d in expected.items():
        p=locate(src,rel,d); q=Path(dst)/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
        if sha(q)!=d: raise RuntimeError(label+" copy mismatch "+rel)
        rows.append({"path":rel,"sha256":d,"source_path":str(p),"bytes":q.stat().st_size})
    return rows
def verify042(root):
    root=Path(root); man=root/"scientific_raw_manifest.json"
    if not man.exists(): raise RuntimeError("042 raw manifest missing")
    m=json.loads(man.read_text(encoding="utf8")); rows=[]; bad=[]
    for x in m.get("files",[]):
        rel=x["path"]; p=root/rel
        ok=p.exists() and p.is_file() and sha(p)==x["sha256"] and p.stat().st_size==x["size"]
        rows.append({"path":rel,"sha256":x["sha256"],"size":x["size"],"verified":bool(ok)})
        if not ok: bad.append(rel)
    if bad: raise RuntimeError("042 raw manifest mismatch: "+",".join(bad[:20]))
    # Required audit objects must exist.
    req=["A_win128/predictions.npz","A_win128/utility_checks.json","A_win128/lifecycle_events.json",
         "A_win128/victim_tables.json","A_win128/reuse_decisions.json","A_win128/candidate_decisions.json",
         "A_win128/settlement_log.json","budget_ledger.json","fixture_report.json","protocol_status.json"]
    missing=[x for x in req if not (root/x).exists()]
    if missing: raise RuntimeError("042 audit source missing "+",".join(missing))
    return {"manifest_file_count":m.get("file_count"),"verified_file_count":len(rows),"required_objects":req}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extracted036",required=True); ap.add_argument("--extracted041",required=True); ap.add_argument("--extracted042",required=True)
    ap.add_argument("--out036",required=True); ap.add_argument("--out041",required=True); ap.add_argument("--report",required=True)
    a=ap.parse_args()
    if sha(PLAN)!=PLAN_SHA: raise RuntimeError("Protocol043 plan hash")
    p=json.loads(PLAN.read_text(encoding="utf8"))
    r36=copy_set(a.extracted036,a.out036,p["source036"]["files"],"036")
    r41=copy_set(a.extracted041,a.out041,p["source041"]["files"],"041")
    r42=verify042(a.extracted042)
    obj={"protocol":"043","revision":1,"source036":r36,"source041":r41,"source042":r42,
         "all_locked_files_verified":True,"historical_checkpoints_restored":False,"F_files_materialized":False}
    q=Path(a.report); q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf8")
    print(json.dumps(obj,indent=2))
if __name__=="__main__": main()
