"""Independent R1 acceptance from recorded predictions and protected files."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parent
METHODS = ("A", "C-legacy", "C-residual-off", "C-residual-on")
FROZEN_CODE = {
    "run_ftmoe_protocol020_r1.py": "bd3c0f72d8cef4a763a995830277758ebd6b83dfd9f55e797d60738e0453f8e8",
    "recovery/PreGANSrc/src/ftmoe_online_r1.py": "72ae878eebaf8a5abca76123c1127a9c1910b49c39dd60aa8d1d04b92acd80cb",
}
PROTECTED = {
    "artifacts/ftmoe_end_to_end/runs/physical_lr0003_e30/v4_seed1/checkpoints_by_epoch/epoch021.pt": "e3513575ec26fc94a9f3c1d877a7119b21bacc97d7001fa02d18278aa316720b",
    "artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/best.pt": "670c56fe94e738bd8a0f65dcacb0ca836960c1ce86dce6c1601a8c74f5078e3b",
    "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt": "10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b",
}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def metrics(p, c, y):
    p, c, y = p.reshape(-1), c.reshape(-1, 3), y.reshape(-1)
    positive, predicted = y > 0, p >= .5
    tp = int((positive & predicted).sum())
    fp = int((~positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    classes = c.argmax(-1) + 1
    diagnosis = np.where(predicted, classes, 0)
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "pr_auc": float(average_precision_score(positive, p)) if positive.any() else None,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn,
        "conditional_resource_f1": float(f1_score(y[positive], classes[positive], labels=[1, 2, 3], average="macro", zero_division=0)),
        "end_to_end_resource_f1": float(f1_score(y, diagnosis, labels=[1, 2, 3], average="macro", zero_division=0)),
    }


def verify(run_root, output):
    checks, runs, arrays = [], {}, {}

    def check(name, ok):
        checks.append({"name": name, "passed": bool(ok)})

    for relative, expected in PROTECTED.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        check("protected:" + relative, actual == expected)
    for path in sorted(run_root.rglob("summary.json")):
        summary = read(path)
        config = read(path.parent / "configuration.json")
        key = (config["method"], int(config["replay_seed"]))
        if key in arrays:
            raise ValueError("Duplicate run: %s" % (key,))
        with np.load(path.parent / "predictions.npz") as archive:
            p = {name: archive[name] for name in archive.files}
        arrays[key] = (p, path.parent)
        name = "%s/dev%d" % key
        for source, expected in FROZEN_CODE.items():
            if source.endswith("ftmoe_online_r1.py") and not key[0].startswith("C-residual"):
                continue
            check(name + ":registered_code:" + source, config["code_sha256"].get(source) == expected and hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected)
        check(name + ":complete", summary["status"] == "complete" and p["probability"].shape == (2000, 16) and (p["labels"] >= 0).all())
        check(name + ":finite", all(np.isfinite(p[k]).all() for k in ("probability", "class_probability")))
        m = metrics(p["probability"], p["class_probability"], p["labels"])
        for field in ("f1", "pr_auc", "recall", "fp", "fn"):
            check(name + ":metric:" + field, np.isclose(m[field], summary["metrics"]["full"][field], rtol=0, atol=1e-12))
        check(name + ":frozen", summary["frozen_parameters_unchanged"])
        runs[name] = {"folder": str(path.parent.resolve()), "metrics": m,
                      "time_blocks_400": [metrics(p["probability"][s:s+400], p["class_probability"][s:s+400], p["labels"][s:s+400]) for s in range(0, 2000, 400)]}
    check("exact_grid", set(arrays) == {(m, s) for m in METHODS for s in (500, 501)})
    for seed in (500, 501):
        base = arrays[("A", seed)][0]
        for method in METHODS[1:]:
            p, folder = arrays[(method, seed)]
            name = "%s/dev%d" % (method, seed)
            check(name + ":paired_labels", np.array_equal(p["labels"], base["labels"]) and np.array_equal(p["raw_labels"], base["raw_labels"]))
            if not method.startswith("C-residual"):
                continue
            ledger = rows(folder / "r1_ledger.jsonl")
            check(name + ":ledger_complete", [r["window_index"] for r in ledger] == list(range(2000)))
            check(name + ":maturity", all(r["label_matured"] and r["matured_step"] == r["prediction_step"] + 1 for r in ledger))
            bp = np.asarray([r["base"]["probability"] for r in ledger], dtype=np.float32).reshape(2000, 16)
            bc = np.asarray([r["base"]["class_probability"] for r in ledger], dtype=np.float32).reshape(2000, 16, 3)
            check(name + ":full_base_matches_A", np.array_equal(bp, base["probability"]) and np.array_equal(bc, base["class_probability"]))
            alpha = np.asarray([r["deployment_alpha"] for r in ledger])
            check(name + ":fallback_matches_A", np.array_equal(p["probability"][alpha == 0], base["probability"][alpha == 0]))
            events = rows(folder / "r1_events.jsonl")
            assessments = [e for e in events if e["event"] == "assessment_evaluated"]
            check(name + ":fixed_assessment_evidence", all(len(e["window_indices"]) == 100 and all(ledger[i]["assessment_snapshot_id"] == e["assessment_snapshot_id"] for i in e["window_indices"]) for e in assessments))
            runs[name]["protection"] = {
                "active_steps": int((alpha > 0).sum()),
                "active_ratio": float((alpha > 0).mean()),
                "accepted": sum(e.get("decision") == "accepted_live_replaced" for e in events),
                "rejected": sum(e.get("decision") == "rejected_live_unchanged" for e in events),
                "rollback": sum(e.get("decision") == "rollback_to_frozen_A" for e in events),
            }
    result = {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL",
              "scope": "implementation_and_recorded_result_integrity_only_not_performance_success",
              "checks": checks, "runs": runs}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [c for c in checks if not c["passed"]]}))
    return result["status"] == "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.runs, args.output) else 1)
