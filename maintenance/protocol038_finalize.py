"""Protocol-038 provenance, manifest, compact-evidence and publication helpers."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

from protocol035_common import dump_json, sha256_file


def W(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    dump_json(p,x)


def file_manifest(root, out, excludes=()):
    root=Path(root); out=Path(out)
    rows=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.resolve()==out.resolve():
            continue
        rel=p.relative_to(root).as_posix()
        if any(rel==x or rel.startswith(x.rstrip("/")+"/") for x in excludes):
            continue
        rows.append({"path":rel,"sha256":sha256_file(p),"bytes":p.stat().st_size})
    obj={"root":str(root),"files":rows,"file_count":len(rows),"total_bytes":sum(x["bytes"] for x in rows)}
    W(out,obj); return obj


def verify_manifest(root, manifest):
    root=Path(root); m=json.loads(Path(manifest).read_text())
    bad=[]
    for r in m["files"]:
        p=root/r["path"]
        if not p.exists():
            bad.append({"path":r["path"],"error":"missing"})
        else:
            got=sha256_file(p)
            if got!=r["sha256"]:
                bad.append({"path":r["path"],"expected":r["sha256"],"actual":got})
    if bad:
        raise AssertionError("manifest verification failed: "+json.dumps(bad[:10]))
    return {"verified":True,"file_count":len(m["files"])}


def provenance(args):
    paths={
        "workflow":args.workflow,
        "run_script":args.run_script,
        "analysis_script":args.analysis_script,
        "frozen_runner":args.frozen_runner,
        "frozen_analysis":args.frozen_analysis,
        "plan":args.plan,
        "validator":args.validator,
        "pip_freeze":args.pip_freeze,
    }
    hashes={k+"_sha256":sha256_file(v) for k,v in paths.items()}
    W(args.out,{"protocol":"038","run_id":str(args.run_id),"checkout_sha":args.checkout_sha,**paths,**hashes})


def compact(src,dst):
    src=Path(src); dst=Path(dst)
    if dst.exists(): shutil.rmtree(dst)
    keep=[
        "stage_A_diagnostics.json","budget_ledger.json","run_audit.json","scientific_status.json",
        "implementation_manifest.json","raw_scientific_manifest.json","terminal_file_manifest.json",
        "donor/audit.json",
        "D_bias/summary.json","D_bias/initialization_audit.json","D_bias/parameter_snapshots.json",
        "D_warm/summary.json","D_warm/initialization_audit.json","D_warm/parameter_snapshots.json",
        "analysis/comparison.json","analysis/cost_profile.json","analysis/status.json",
    ]
    for rel in keep:
        p=src/rel
        if p.exists():
            q=dst/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    file_manifest(dst,dst/"compact_manifest.json")


def artifact_index(args):
    info=json.loads(Path(args.artifact_info).read_text())
    obj={
        "protocol":"038","run_id":str(args.run_id),"artifact_id":int(info["id"]),
        "name":info.get("name"),"size_in_bytes":int(info.get("size_in_bytes",0)),
        "created_at":info.get("created_at"),"expires_at":info.get("expires_at"),
        "zip_sha256":args.zip_sha256,
        "scientific_manifest_sha256":sha256_file(args.scientific_manifest),
        "terminal_manifest_sha256":sha256_file(args.terminal_manifest),
    }
    W(args.out,obj)


def publication(args):
    W(args.out,{"protocol":"038","run_id":str(args.run_id),"publication_status":args.status,"science_status":"completed"})


def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("provenance")
    for x in ("out","run_id","checkout_sha","workflow","run_script","analysis_script","frozen_runner","frozen_analysis","plan","validator","pip_freeze"):
        p.add_argument("--"+x.replace("_","-"),dest=x,required=True)
    m=sp.add_parser("manifest"); m.add_argument("--root",required=True); m.add_argument("--out",required=True); m.add_argument("--exclude",action="append",default=[])
    v=sp.add_parser("verify"); v.add_argument("--root",required=True); v.add_argument("--manifest",required=True)
    c=sp.add_parser("compact"); c.add_argument("--src",required=True); c.add_argument("--dst",required=True)
    a=sp.add_parser("artifact-index")
    for x in ("artifact_info","scientific_manifest","terminal_manifest","zip_sha256","out","run_id"):
        a.add_argument("--"+x.replace("_","-"),dest=x,required=True)
    q=sp.add_parser("publication"); q.add_argument("--out",required=True); q.add_argument("--run-id",required=True); q.add_argument("--status",required=True)
    args=ap.parse_args()
    if args.cmd=="provenance": provenance(args)
    elif args.cmd=="manifest": file_manifest(args.root,args.out,args.exclude)
    elif args.cmd=="verify": print(json.dumps(verify_manifest(args.root,args.manifest)))
    elif args.cmd=="compact": compact(args.src,args.dst)
    elif args.cmd=="artifact-index": artifact_index(args)
    else: publication(args)


if __name__=="__main__": main()
