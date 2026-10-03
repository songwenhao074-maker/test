"""Materialize exact Protocol-036 and Protocol-041 raw inputs registered by Protocol-042 rev2."""
from pathlib import Path
import argparse, hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"artifacts/ftmoe_online/protocol_042/plan.json"
PLAN_SHA="11eaf5e89c53a6ec0f122854d513e208da6b0aa0ceda95ec38999fa4961f4be7"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def locate(root,rel,digest):
    root=Path(root); rel=Path(rel).as_posix()
    named=[p for p in root.rglob(Path(rel).name) if p.is_file() and (p.as_posix()==rel or p.as_posix().endswith("/"+rel))]
    hits=[p for p in named if sha(p)==digest]
    if len(hits)!=1:
        raise RuntimeError(f"{rel}: expected exactly one suffix+sha256 match, found {len(hits)} from {len(named)} suffix matches")
    return hits[0]

def copy_set(src,dst,expected,label):
    src,dst=Path(src),Path(dst); rows=[]
    for rel,d in expected.items():
        p=locate(src,rel,d); q=dst/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
        got=sha(q)
        if got!=d: raise RuntimeError(f"{label} copied digest mismatch {rel}: {got}")
        rows.append({"path":rel,"sha256":d,"source_path":str(p),"bytes":q.stat().st_size})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extracted036",required=True); ap.add_argument("--extracted041",required=True)
    ap.add_argument("--out036",required=True); ap.add_argument("--out041",required=True); ap.add_argument("--report",required=True)
    a=ap.parse_args()
    if sha(PLAN)!=PLAN_SHA: raise RuntimeError("Protocol042 rev2 plan hash mismatch")
    p=json.loads(PLAN.read_text(encoding="utf8"))
    if (p.get("protocol"),p.get("revision"))!=("042",2): raise RuntimeError("wrong Protocol042 registration")
    r36=copy_set(a.extracted036,a.out036,p["source036"]["files"],"036")
    r41=copy_set(a.extracted041,a.out041,p["source041"]["files"],"041")
    obj={"protocol":"042","revision":2,"source036":r36,"source041":r41,
         "all_locked_files_verified":True,"historical_checkpoints_restored":False,"F_files_materialized":False}
    q=Path(a.report); q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf8")
    print(json.dumps(obj,indent=2))
if __name__=="__main__": main()
