"""Protocol-039 immutable scientific manifest, provenance and compact publication helpers."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, sys
from pathlib import Path

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def dump_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))

def files(root,skip=()):
    root=Path(root); skip=set(skip)
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        rel=p.relative_to(root).as_posix()
        if any(rel==s or rel.startswith(s.rstrip("/")+"/") for s in skip): continue
        yield p,rel

def build_manifest(root,out,skip=()):
    root,out=Path(root),Path(out); rows=[]
    for p,rel in files(root,skip):
        if p.resolve()==out.resolve(): continue
        rows.append({"path":rel,"sha256":sha256_file(p),"size":p.stat().st_size})
    obj={"protocol":"039","root":str(root),"file_count":len(rows),"files":rows}; dump_json(out,obj); return obj

def check_manifest(root,manifest):
    root=Path(root); m=J(manifest); bad=[]
    for x in m["files"]:
        p=root/x["path"]
        if not p.exists() or p.stat().st_size!=int(x["size"]) or sha256_file(p)!=x["sha256"]: bad.append(x["path"])
    return bad

def cmd_provenance(a):
    paths=[a.run_script,a.analysis_script,a.finalize_script,a.materializer,a.plan,a.validator,a.workflow,a.kernel_run,a.kernel_analysis,a.kernel_common]
    fs={str(Path(p)):{"sha256":sha256_file(p),"size":Path(p).stat().st_size} for p in paths}
    obj={"protocol":"039","run_id":str(a.run_id),"checkout_sha":a.checkout_sha,
         "plan_sha256":sha256_file(a.plan),"workflow_sha256":sha256_file(a.workflow),
         "run_script_sha256":sha256_file(a.run_script),"analysis_script_sha256":sha256_file(a.analysis_script),
         "kernel_reference_commit":"7440a3973f97599ea2ad4de45d5bacaaf8b517c8",
         "kernel_files":{"run_ftmoe_protocol037.py":sha256_file(a.kernel_run),"analyze_ftmoe_protocol037.py":sha256_file(a.kernel_analysis),"protocol035_common.py":sha256_file(a.kernel_common)},
         "files":fs,"python":sys.version,"platform":platform.platform(),"pip_freeze_sha256":sha256_file(a.pip_freeze),
         "science_identity_not_preflight_identity":True}
    dump_json(a.out,obj)

def cmd_manifest(a):
    m=build_manifest(a.root,a.out,a.skip or [])
    bad=check_manifest(a.root,a.out)
    if bad: raise RuntimeError("manifest self-check failed: "+repr(bad))
    print(json.dumps({"file_count":m["file_count"],"self_check":True},indent=2))

def cmd_compact(a):
    src,analysis,dst=Path(a.science),Path(a.analysis),Path(a.dst)
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True)
    science_names={"source_lock.json","fixture_report.json","implementation_manifest.json","budget_ledger.json","budget_ledger_events.jsonl","run_audit.json","science_cost_raw.json","scientific_execution_complete.json","status.json","scientific_raw_manifest.json"}
    leaf={"summary.json","birth_checks.json","sleep_checks.json","wake_checks.json","lifecycle_events.json","update_log.json","forward_log.json"}
    for p,rel in files(src,skip=("checkpoints","runtime_state")):
        if Path(rel).name in science_names or Path(rel).name in leaf:
            q=dst/"science"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    for p,rel in files(analysis):
        q=dst/"analysis"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    build_manifest(dst,dst/"compact_manifest.json")

def cmd_artifact(a):
    info=J(a.info); raw=J(a.raw_manifest)
    dump_json(a.out,{"protocol":"039","run_id":str(a.run_id),"artifact_id":int(info["id"]),"artifact_name":info["name"],
                     "size_in_bytes":int(info["size_in_bytes"]),"digest":info.get("digest"),"expires_at":info.get("expires_at"),
                     "archive_download_url":info.get("archive_download_url"),"raw_manifest_file_count":raw["file_count"]})

def cmd_pubstatus(a):
    dump_json(a.out,{"protocol":"039","run_id":str(a.run_id),"publication_status":a.status,"detail":a.detail or None})

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("provenance")
    for x in ("out","run-id","checkout-sha","run-script","analysis-script","finalize-script","materializer","plan","validator","workflow","kernel-run","kernel-analysis","kernel-common","pip-freeze"): p.add_argument("--"+x,required=True)
    p.set_defaults(fn=cmd_provenance)
    p=sub.add_parser("manifest"); p.add_argument("--root",required=True); p.add_argument("--out",required=True); p.add_argument("--skip",action="append"); p.set_defaults(fn=cmd_manifest)
    p=sub.add_parser("compact"); p.add_argument("--science",required=True); p.add_argument("--analysis",required=True); p.add_argument("--dst",required=True); p.set_defaults(fn=cmd_compact)
    p=sub.add_parser("artifact-index"); p.add_argument("--info",required=True); p.add_argument("--raw-manifest",required=True); p.add_argument("--out",required=True); p.add_argument("--run-id",required=True); p.set_defaults(fn=cmd_artifact)
    p=sub.add_parser("publication-status"); p.add_argument("--out",required=True); p.add_argument("--run-id",required=True); p.add_argument("--status",required=True); p.add_argument("--detail"); p.set_defaults(fn=cmd_pubstatus)
    a=ap.parse_args(); a.fn(a)
if __name__=="__main__": main()
