"""Materialize exact locked Protocol-036 and Protocol-040 inputs for Protocol-041."""
from pathlib import Path
import argparse, hashlib, json, shutil

P036={
 "stage_B/seed3601/C_ref/feature_tape.npz":"c71a2ea6d59045fe4cc8a953a776d37f48dbe1feec4aba859cd00f11c984529c",
 "stage_B/seed3601/C_ref/update_batches.json":"4d32d2ece194cb41b8bc16de1c89ddfedcd42da668b96446937fb430d6246b6f",
 "stage_B/seed3601/D_lin/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "inputs/seed3601/manifest.json":"73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b",
 "stage_B/seed3601/C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
}
P040={
 "B_ref/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
 "D_039/predictions.npz":"bc387d13721ed2e759a205ed3b4f6fa884c1ca105068a79a340148e1f5e96512",
 "D_keep/predictions.npz":"badf9005b2be9fc6436d1ff2b9377b27715e846d7d16f8e4bdfcbfa76f732e8c",
 "D_pool2/birth_checks.json":"ff08d5cb85702fed832eb0cb9c9781b9195fc93cbe7b8117e9011349e5169e2f",
 "D_pool2/candidate_decisions.json":"36c100ee1ad207d96c4e8bf4760f9ff412858b24b8759ad1b060e2b2e847ad8b",
 "D_pool2/lifecycle_events.json":"b05d76c102063002d4737f37a4b7f076ca3e1ea90176c03333aee21b82502906",
 "D_pool2/opportunity_log.json":"fac473aee5d4b8f22590355c16447dc8f43b44c62d6e79c3228e7218358fbe4a",
 "D_pool2/per_expert_runtime.json":"326a651ba5e36e67f9ab72d3bcb579127942ec51b80a5559fff2a9756780d776",
 "D_pool2/predictions.npz":"51a5ac7ec4e72f340ab1f051bb3707ad44e5540786a94bf0b10f9a8af9e7dd6a",
 "D_pool2/prefix039_audit.json":"dbba34925612a13c85b3c995bb06bb114203b4f506c325b10df93fe8728aeff9",
 "D_pool2/pressure_checks.json":"391f7ef53b9bdb75972f38bcd9a6f6f59f7a046a277ba63160a1d56c8a6230ed",
 "D_pool2/reuse_decisions.json":"671276db156b4851b54a70850cb268c9e05e05806ca68d702da3cb3fe8a5386f",
 "D_pool2/sleep_checks.json":"fa5a9f9c9bcbc29dc1338d46c5f82d65778b8ae6ace8f6d787275ac9004fd600",
 "D_pool2/summary.json":"496b165ce0ac1cd79d4bbd5073cb760f2a93ed64061d536e184f9fbf0ee72b47",
 "D_pool2/update_log.json":"8fc610dcf77087d9fcb4450133a3cfd5fe8fc3d5c9a6e70e869629c98405ab10",
}

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def locate(root,rel,digest):
    root=Path(root); suffix=Path(rel).as_posix()
    named=[p for p in root.rglob(Path(rel).name) if p.is_file() and p.as_posix().endswith("/"+suffix)]
    hits=[p for p in named if sha(p)==digest]
    if len(hits)!=1:
        raise RuntimeError("%s: expected one suffix+digest match, found %d from %d suffix matches"%(rel,len(hits),len(named)))
    return hits[0]

def copy_set(src,dst,expected,label):
    src,dst=Path(src),Path(dst); rows=[]
    for rel,d in expected.items():
        p=locate(src,rel,d); q=dst/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
        got=sha(q)
        if got!=d: raise RuntimeError(label+" copy digest mismatch "+rel)
        rows.append({"path":rel,"sha256":d,"source_path":str(p),"size":q.stat().st_size})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extracted036",required=True); ap.add_argument("--extracted040",required=True)
    ap.add_argument("--out036",required=True); ap.add_argument("--out040",required=True); ap.add_argument("--report",required=True)
    a=ap.parse_args()
    r36=copy_set(a.extracted036,a.out036,P036,"036")
    r40=copy_set(a.extracted040,a.out040,P040,"040")
    obj={"protocol":"041","source036":r36,"source040":r40,"all_locked_files_verified":True,"historical_checkpoints_restored":False,"F_files_materialized":False}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(obj,indent=2)+"\n",encoding="utf8")
    print(json.dumps(obj,indent=2))
if __name__=="__main__": main()
