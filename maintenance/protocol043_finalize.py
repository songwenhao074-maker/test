"""Protocol-043 revision1 provenance, manifests and compact handoff."""
from pathlib import Path
import argparse, hashlib, json, platform, shutil, sys
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def walk(root,skip=()):
    root=Path(root); skip=set(skip); out=[]
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel=p.relative_to(root).as_posix()
            if rel not in skip: out.append((p,rel))
    return out
def manifest(root,out):
    root=Path(root); out=Path(out)
    try: skip={out.relative_to(root).as_posix()}
    except ValueError: skip=set()
    rows=[{"path":rel,"sha256":sha(p),"size":p.stat().st_size} for p,rel in walk(root,skip)]
    obj={"protocol":"043","revision":1,"root":str(root),"file_count":len(rows),"files":rows}; dump(out,obj); return obj
def cmd_manifest(a): print(json.dumps({"file_count":manifest(a.root,a.out)["file_count"]},indent=2))
def cmd_provenance(a):
    dump(a.out,{"protocol":"043","revision":1,"run_id":str(a.run_id),"checkout_sha":a.checkout_sha,
      "plan_sha256":sha(a.plan),"workflow_sha256":sha(a.workflow),"source_bundle":{p:sha(p) for p in a.file},
      "python":sys.version,"platform":platform.platform(),"pip_freeze_sha256":sha(a.pip_freeze)})
def cmd_artifact(a):
    info=json.loads(Path(a.info).read_text()); raw=json.loads(Path(a.raw_manifest).read_text())
    dump(a.out,{"protocol":"043","revision":1,"run_id":str(a.run_id),"artifact_id":int(info["id"]),"artifact_name":info["name"],
      "size_in_bytes":int(info["size_in_bytes"]),"digest":info.get("digest"),"expires_at":info.get("expires_at"),
      "archive_download_url":info.get("archive_download_url"),"raw_manifest_file_count":raw["file_count"]})
def cmd_compact(a):
    src,ana,audit,dst=map(Path,[a.science,a.analysis,a.audit,a.dst])
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True)
    keep={"fixture_report.json","gate_report.json","implementation_manifest.json","budget_ledger.json","source_materialization.json",
          "scientific_raw_manifest.json","protocol_status.json","provenance.json"}
    for p,rel in walk(src):
        parts=Path(rel).parts
        if "checkpoints" in parts or "runtime_state" in parts: continue
        if rel in keep or (parts and parts[0] in ("cached","D_no_gc","D_bounded","source_bundle")):
            q=dst/"science"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    for p,rel in walk(ana):
        q=dst/"analysis"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    for p,rel in walk(audit):
        q=dst/"audit042"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    manifest(dst,dst/"compact_manifest.json")
def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("manifest"); p.add_argument("--root",required=True); p.add_argument("--out",required=True); p.set_defaults(fn=cmd_manifest)
    p=sp.add_parser("provenance"); p.add_argument("--out",required=True); p.add_argument("--run-id",required=True); p.add_argument("--checkout-sha",required=True)
    p.add_argument("--plan",required=True); p.add_argument("--workflow",required=True); p.add_argument("--pip-freeze",required=True); p.add_argument("--file",action="append",required=True); p.set_defaults(fn=cmd_provenance)
    p=sp.add_parser("artifact-index"); p.add_argument("--info",required=True); p.add_argument("--raw-manifest",required=True); p.add_argument("--out",required=True); p.add_argument("--run-id",required=True); p.set_defaults(fn=cmd_artifact)
    p=sp.add_parser("compact"); p.add_argument("--science",required=True); p.add_argument("--analysis",required=True); p.add_argument("--audit",required=True); p.add_argument("--dst",required=True); p.set_defaults(fn=cmd_compact)
    a=ap.parse_args(); a.fn(a)
if __name__=="__main__": main()
