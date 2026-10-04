"""Protocol-044 artifact manifest and compact handoff utilities."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def manifest(root,out,protocol="044"):
    root=Path(root);rows=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.resolve()!=Path(out).resolve():
            rows.append({"path":str(p.relative_to(root)),"size":p.stat().st_size,"sha256":sha(p)})
    W(out,{"protocol":protocol,"revision":1,"file_count":len(rows),"files":rows})
def copy_if(src,dst):
    src=Path(src);dst=Path(dst)
    if src.is_file():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    elif src.is_dir():
        if dst.exists():shutil.rmtree(dst)
        shutil.copytree(src,dst)
def compact(stage,data,evidence,analysis,dst):
    stage=Path(stage);data=Path(data);evidence=Path(evidence);analysis=Path(analysis);dst=Path(dst)
    if dst.exists():shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in ("budget_ledger.json","science_status.json","implementation_manifest.json"):
        copy_if(stage/p,dst/"stageS"/p)
    for arm in ("C_ref","D_lin","D_no_gc","D_bounded"):
        for p in ("summary.json","update_batches.json","update_log.json"):
            copy_if(stage/arm/p,dst/"stageS"/arm/p)
        if arm.startswith("D_"):
            for q in ("lifecycle.jsonl","candidate_decisions.jsonl","reuse_decisions.jsonl","reclamation.jsonl","qualification_raw.jsonl","utility_checks.jsonl","sleep_tables.jsonl","pressure_checks.jsonl","birth_checks.jsonl"):
                copy_if(stage/arm/"streams"/q,dst/"stageS"/arm/"streams"/q)
    for p in ("manifest.json","generation_manifest.json","registration.json"):
        copy_if(data/p,dst/"data"/p)
    for p in ("input_lock.json","input_audit.json","data_lock.json"):
        copy_if(evidence/p,dst/"data"/p)
    copy_if(analysis,dst/"analysis")
def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("manifest");p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p=sp.add_parser("compact");p.add_argument("--stage",required=True);p.add_argument("--data",required=True);p.add_argument("--evidence",required=True);p.add_argument("--analysis",required=True);p.add_argument("--dst",required=True)
    a=ap.parse_args()
    if a.cmd=="manifest":manifest(a.root,a.out)
    else:compact(a.stage,a.data,a.evidence,a.analysis,a.dst)
if __name__=="__main__":main()
