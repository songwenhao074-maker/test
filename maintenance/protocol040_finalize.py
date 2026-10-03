"""Protocol-040 publication helpers."""
from pathlib import Path
import argparse, hashlib, json, os, platform, shutil, sys

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def dump(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")

def walk(root, skip=()):
    root=Path(root); out=[]
    skip=set(skip)
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        rel=p.relative_to(root).as_posix()
        if rel in skip: continue
        out.append((p,rel))
    return out

def build_manifest(root,out):
    root=Path(root); out=Path(out)
    rows=[]
    for p,rel in walk(root,{out.relative_to(root).as_posix()} if out.is_relative_to(root) else set()):
        rows.append({"path":rel,"sha256":sha(p),"size":p.stat().st_size})
    obj={"protocol":"040","root":str(root),"file_count":len(rows),"files":rows}
    dump(out,obj); return obj

def cmd_manifest(a):
    m=build_manifest(a.root,a.out); print(json.dumps({"file_count":m["file_count"]},indent=2))

def cmd_provenance(a):
    files={}
    for x in a.file:
        rel=x
        files[rel]=sha(rel)
    dump(a.out,{
      "protocol":"040","run_id":str(a.run_id),"checkout_sha":a.checkout_sha,
      "plan_sha256":sha(a.plan),"workflow_sha256":sha(a.workflow),
      "files":files,"python":sys.version,"platform":platform.platform(),
      "pip_freeze_sha256":sha(a.pip_freeze),
      "science_identity_not_preflight_identity":True
    })

def cmd_compact(a):
    src=Path(a.science); ana=Path(a.analysis); dst=Path(a.dst)
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True)
    keep_names={
      "source_lock.json","fixture_report.json","implementation_manifest.json","budget_ledger.json",
      "run_audit.json","science_cost_raw.json","scientific_execution_complete.json","status.json",
      "scientific_raw_manifest.json"
    }
    for p,rel in walk(src):
        if Path(rel).name in keep_names or rel.startswith("D_pool2/"):
            q=dst/"science"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    for p,rel in walk(ana):
        q=dst/"analysis"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    build_manifest(dst,dst/"compact_manifest.json")

def cmd_artifact(a):
    info=json.loads(Path(a.info).read_text()); raw=json.loads(Path(a.raw_manifest).read_text())
    dump(a.out,{"protocol":"040","run_id":str(a.run_id),"artifact_id":int(info["id"]),"artifact_name":info["name"],
                "size_in_bytes":int(info["size_in_bytes"]),"digest":info.get("digest"),"expires_at":info.get("expires_at"),
                "archive_download_url":info.get("archive_download_url"),"raw_manifest_file_count":raw["file_count"]})

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("manifest"); p.add_argument("--root",required=True); p.add_argument("--out",required=True); p.set_defaults(fn=cmd_manifest)
    p=sub.add_parser("provenance"); p.add_argument("--out",required=True); p.add_argument("--run-id",required=True); p.add_argument("--checkout-sha",required=True)
    p.add_argument("--plan",required=True); p.add_argument("--workflow",required=True); p.add_argument("--pip-freeze",required=True)
    p.add_argument("--file",action="append",default=[],required=True); p.set_defaults(fn=cmd_provenance)
    p=sub.add_parser("compact"); p.add_argument("--science",required=True); p.add_argument("--analysis",required=True); p.add_argument("--dst",required=True); p.set_defaults(fn=cmd_compact)
    p=sub.add_parser("artifact-index"); p.add_argument("--info",required=True); p.add_argument("--raw-manifest",required=True); p.add_argument("--out",required=True); p.add_argument("--run-id",required=True); p.set_defaults(fn=cmd_artifact)
    a=ap.parse_args(); a.fn(a)
if __name__=="__main__": main()
