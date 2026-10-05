"""Protocol044 r2 publication/recovery acceptance: immutable r1, no training path."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, shutil, tempfile, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from maintenance import protocol044_r2_publish as pub

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def W(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args()
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td);(root/"docs").mkdir()
        shutil.copy2(ROOT/"docs/PROTOCOL044_RESULTS.md",root/"docs/PROTOCOL044_RESULTS.md")
        r1=sha(root/"docs/PROTOCOL044_RESULTS.md")
        (root/"docs/PROTOCOL044_REVISION002_RESULTS.md").write_text("# fixture r2 result\n",encoding="utf8")
        compact=root/"artifacts/ftmoe_online/protocol_044/revision_002/runs/run_fixture";compact.mkdir(parents=True)
        status={"protocol":"044","revision":1,"execution_revision":2,"science_config_revision":1,"system_result_label":"fixture_only",
          "E_gate":True,"S_validity":True,"D_over_C":False,"lifecycle_on_stream":False,"reuse_on_stream":False,
          "gc_on_stream":False,"gc_admission_closed_loop":False,"new_stream_core_supported":False,"full_lifecycle_goal_completed":False}
        sp=root/"status.json";W(sp,status)
        before=sha(root/"docs/PROTOCOL044_RESULTS.md")
        pub.prepare(root,compact,sp,"fixture",r1)
        mid=sha(root/"docs/PROTOCOL044_RESULTS.md")
        # prepare must update only pointers/r2 artifacts, never old report.
        if before!=mid or mid!=r1: raise RuntimeError("r1 changed during prepare")
        if not (root/"docs/PROTOCOL044_REVISION002_RESULTS.md").is_file():raise RuntimeError("r2 result removed")
        if "revision 2" not in (root/"AGENTS.md").read_text(encoding="utf8"):raise RuntimeError("r2 pointer missing")
        pub.sync(compact,"fixture",root/"docs/PROTOCOL044_RESULTS.md",r1)
        after=sha(root/"docs/PROTOCOL044_RESULTS.md")
        ps=json.loads((compact/"publication_status.json").read_text(encoding="utf8"))
        ok=bool(before==mid==after==r1 and ps["main_synced"] is True and ps["publication_status"]=="synced_to_main")
        result={"protocol":"044","execution_revision":2,"kind":"publication_recovery_acceptance","all_pass":ok,
          "r1_sha256":r1,"r1_unchanged_prepare":before==mid,"r1_unchanged_sync":mid==after,
          "r2_result_separate":True,"model_forwards":0,"gradient_steps":0,"source_code_lock_reused_as_new_execution_proof":False}
        W(out,result);print(json.dumps(result,indent=2))
        if not ok:raise SystemExit(2)
if __name__=="__main__":main()
