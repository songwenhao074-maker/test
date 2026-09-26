"""Protocol-031 independent finalized-data audit and hash freeze."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import prepare_ftmoe_protocol031_stream as P

EXPECTED_HOSTS=16
RECURRENCE=("U_rec1","V_rec1","U_rec2","V_rec2","U_rec3","V_rec3")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def write_json(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")

def audit(data_root,output_root,freeze_registration=None):
    root=Path(data_root); out=Path(output_root); out.mkdir(parents=True,exist_ok=True)
    reg=P.registration()
    expected_steps=int(reg["scored_intervals"])
    expected_rows=expected_steps+int(reg["guard_intervals"])
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf8"))
    source_audit=json.loads((root/"data_audit.json").read_text(encoding="utf8"))
    status=json.loads((root/"generation_status.json").read_text(encoding="utf8"))
    stream=root/"stream.npz"
    actual=sha256(stream)
    with np.load(stream) as d:
        raw=np.asarray(d["raw_labels"],dtype=np.int64)
        ratio=np.asarray(d["overload_ratio"],dtype=np.float64)
        schedules=np.asarray(d["schedules"],dtype=np.float64)
        caps=np.asarray(d["capacities"],dtype=np.float64)
        physical=np.where((ratio>1.0).any(-1),ratio.argmax(-1)+1,0)
        labels_equal=bool(np.array_equal(raw,physical))
        raw_shape=list(raw.shape)
        schedules_ok=bool(np.isfinite(schedules[:expected_steps]).all() and
                          np.allclose(schedules[:expected_steps].sum(-1),1.0,atol=1e-5))
        caps_ok=bool(np.isfinite(caps[:expected_steps]).all() and
                     (caps[:expected_steps]>0).all())
    common=np.load(root/"common_observable_features.npz")["features"]
    common_ok=bool(common.shape==(expected_rows,EXPECTED_HOSTS,9) and np.isfinite(common).all())

    chunks=list(manifest.get("chunk_manifest",[]))
    contiguous=True; last=0; chunk_hashes=[]
    for rec in chunks:
        p=root/"chunks"/str(rec["file"])
        got=sha256(p) if p.is_file() else None
        ok=(int(rec["start"])==last and int(rec["end"])>last and got==rec["sha256"])
        contiguous=contiguous and ok
        chunk_hashes.append({"file":rec["file"],"start":int(rec["start"]),
                             "end":int(rec["end"]),"sha256":got,"verified":bool(ok)})
        last=int(rec["end"])
    chunk_complete=bool(contiguous and last==expected_rows)

    rec=source_audit.get("recurrence_first128_class_coverage",{})
    rec_ok=(set(rec)==set(RECURRENCE) and all(
        int(rec[n]["positive_host_steps"])>=32 and
        int(rec[n]["negative_host_steps"])>=32 and
        bool(rec[n]["ap_defined"]) for n in RECURRENCE
    ))
    source_law=root.parent.parent.parent.parent/"simulator/workload/BitbrainWorkloadProtocol025.py"
    # Repository path may not be relative to data_root in Actions; use manifest's recorded hash as
    # provenance plus current repository file from P.ROOT.
    current_law=P.ROOT/"simulator/workload/BitbrainWorkloadProtocol025.py"
    current_law_sha=sha256(current_law)
    reg_sha=P.json_sha(P.REGISTRATION_PATH)
    gates={
        "generation_complete":status.get("complete") is True and int(status.get("next_t",-1))==expected_rows,
        "manifest_identity":manifest.get("protocol")=="031" and manifest.get("plan_revision")==P.PLAN_REVISION
                            and manifest.get("scenario_id")==P.SCENARIO_ID
                            and manifest.get("data_revision")==P.DATA_REVISION,
        "stream_hash_self_consistent":actual==manifest.get("stream_sha256"),
        "shape_registered_rows_x16":raw_shape==[expected_rows,EXPECTED_HOSTS],
        "physical_label_recompute_exact":labels_equal,
        "source_model_free_audit_pass":source_audit.get("audit_pass") is True,
        "source_model_free_all_gates_pass":all(bool(v) for v in source_audit.get("gates",{}).values()),
        "six_recurrence_first128_coverage":rec_ok,
        "common_features_shape_finite":common_ok,
        "scheduler_rows_valid":schedules_ok,
        "capacities_valid":caps_ok,
        "chunk_chain_complete_and_hash_verified":chunk_complete,
        "source_law_hash_matches_manifest":current_law_sha==manifest.get("source_law_sha256"),
        "scenario_registration_hash_matches_manifest":reg_sha==manifest.get("registration_sha256"),
        "normal_guard_available":int(source_audit.get("normal_guard",{}).get("normal_rows",0))>0,
    }
    eligible=bool(all(gates.values()))
    eligibility={
        "protocol":"031","plan_revision":P.PLAN_REVISION,"scenario_id":P.SCENARIO_ID,
        "data_revision":P.DATA_REVISION,"kind":"independent_model_free_eligibility",
        "model_results_seen":False,"protocol031_data_eligible":eligible,
        "stream_sha256":actual,"shape":raw_shape,
        "source_audit_pass":bool(source_audit.get("audit_pass")),
        "recurrence_first128":rec,
        "normal_guard":source_audit.get("normal_guard"),
        "chunk_count":len(chunks),"chunk_chain_end":last,
        "source_law_sha256":current_law_sha,
        "registration_sha256_before_hash_freeze":reg_sha,
        "gates":gates,
    }
    write_json(out/"eligibility.json",eligibility)
    if not eligible:
        lock={
            "protocol":"031","plan_revision":P.PLAN_REVISION,"scenario_id":P.SCENARIO_ID,
            "data_revision":P.DATA_REVISION,"locked":False,
            "reason":"model_free_data_audit_failed","stream_sha256":actual,"gates":gates,
        }
        write_json(out/"data_lock.json",lock)
        print(json.dumps(eligibility,indent=2,allow_nan=False))
        raise SystemExit(3)

    frozen_manifest={
        "protocol":"031","plan_revision":P.PLAN_REVISION,"scenario_id":P.SCENARIO_ID,
        "data_revision":P.DATA_REVISION,"stream_sha256":actual,
        "stream_bytes":stream.stat().st_size,
        "manifest_sha256":sha256(root/"manifest.json"),
        "data_audit_sha256":sha256(root/"data_audit.json"),
        "common_features_sha256":sha256(root/"common_observable_features.npz"),
        "events_sha256":sha256(root/"events.json"),
        "registration_sha256_before_hash_freeze":reg_sha,
        "source_law_sha256":current_law_sha,
        "chunks":chunk_hashes,
        "frozen_before_model_runs":True,
    }
    write_json(root/"frozen_data_manifest.json",frozen_manifest)
    lock={
        "protocol":"031","plan_revision":P.PLAN_REVISION,"scenario_id":P.SCENARIO_ID,
        "data_revision":P.DATA_REVISION,"locked":True,
        "stream_sha256":actual,"steps":expected_steps,"guard_rows":1,
        "hosts":16,"replay_seed":700,"model_seed":1,
        "generation_budget_used":1,
        "common_features_sha256":frozen_manifest["common_features_sha256"],
        "events_sha256":frozen_manifest["events_sha256"],
        "frozen_data_manifest_sha256":sha256(root/"frozen_data_manifest.json"),
        "model_free_eligibility_passed":True,
        "six_recurrence_windows":list(RECURRENCE),
    }
    write_json(out/"data_lock.json",lock)

    if freeze_registration:
        path=Path(freeze_registration)
        payload=json.loads(path.read_text(encoding="utf8"))
        if payload.get("generation",{}).get("expected_stream_sha256") not in (None,actual):
            raise AssertionError("Protocol031 registration already frozen to a different stream")
        payload["generation"]["expected_stream_sha256"]=actual
        payload["generation"]["frozen_before_model_runs"]=True
        payload["generation"]["frozen_data_manifest_sha256"]=lock["frozen_data_manifest_sha256"]
        payload["status"]="data_frozen_before_model_runs"
        path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf8")
        lock["registration_sha256_after_hash_freeze"]=sha256(path)
        write_json(out/"data_lock.json",lock)

    print(json.dumps({"eligible":True,"stream_sha256":actual,
                      "registration_frozen":bool(freeze_registration),
                      "chunk_count":len(chunks)},indent=2))
    return eligibility,lock

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--output-root",required=True)
    ap.add_argument("--freeze-registration")
    a=ap.parse_args()
    audit(a.data_root,a.output_root,a.freeze_registration)

if __name__=="__main__":
    main()
