"""Protocol-037 provenance, raw/compact manifests, artifact index and publication status helpers."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from protocol035_common import dump_json, sha256_file


def J(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def iter_files(root, excludes=()):
    root = Path(root)
    ex = [str(x) for x in excludes]
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(rel == x or rel.startswith(x.rstrip("/") + "/") for x in ex):
            continue
        yield p, rel


def manifest(root, out, excludes=()):
    root, out = Path(root), Path(out)
    rows = []
    for p, rel in iter_files(root, excludes):
        if p.resolve() == out.resolve():
            continue
        rows.append({"path": rel, "sha256": sha256_file(p), "size": p.stat().st_size})
    obj = {"protocol": "037", "root": str(root), "files": rows, "file_count": len(rows)}
    dump_json(out, obj)
    return obj


def cmd_provenance(a):
    files = {}
    for p in [a.run_script, a.analysis_script, a.finalize_script, a.plan, a.validator, a.workflow]:
        q = Path(p)
        files[q.name] = {"path": str(q), "sha256": sha256_file(q), "size": q.stat().st_size}
    obj = {
        "protocol": "037", "run_id": str(a.run_id), "checkout_sha": a.checkout_sha,
        "workflow_path": a.workflow, "workflow_sha256": sha256_file(a.workflow),
        "run_script_sha256": sha256_file(a.run_script), "analysis_script_sha256": sha256_file(a.analysis_script),
        "finalize_script_sha256": sha256_file(a.finalize_script),
        "plan_sha256": sha256_file(a.plan), "validator_sha256": sha256_file(a.validator),
        "python": sys.version, "platform": platform.platform(), "files": files,
        "pip_freeze_sha256": sha256_file(a.pip_freeze), "pip_freeze_path": a.pip_freeze,
        "preflight_identity_is_not_science_identity": True,
    }
    dump_json(a.out, obj)


COMPACT_NAMES = {
    "source_lock.json", "budget_ledger.json", "budget_ledger_events.jsonl", "run_audit.json",
    "science_cost_raw.json", "scientific_execution_complete.json", "status.json",
    "raw_scientific_manifest.json", "implementation_manifest.json", "fixture_report.json",
    "artifact_index.json", "comparison.json", "cost_profile.json",
}


def should_compact(rel):
    name = Path(rel).name
    if name in COMPACT_NAMES:
        return True
    if name in {"summary.json", "causality_audit.json", "isolation_audit.json", "birth_checks.json", "lifecycle_events.json", "update_log.json"}:
        return True
    return False


def make_compact(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p, rel in iter_files(src, excludes=("checkpoints",)):
        if should_compact(rel):
            q = dst / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    manifest(dst, dst / "compact_manifest.json")
    return dst


def cmd_manifest(a):
    manifest(a.root, a.out, a.exclude or [])


def cmd_compact(a):
    make_compact(a.src, a.dst)


def cmd_artifact_index(a):
    info = J(a.artifact_info)
    raw = J(a.raw_manifest)
    obj = {
        "protocol": "037", "run_id": str(a.run_id),
        "artifact_id": int(info["id"]), "artifact_name": info["name"],
        "artifact_url": info.get("archive_download_url"), "artifact_size_bytes": int(info["size_in_bytes"]),
        "artifact_digest": info.get("digest"), "expires_at": info.get("expires_at"),
        "raw_file_count": raw.get("file_count"), "raw_files": raw.get("files"),
    }
    dump_json(a.out, obj)


def cmd_status(a):
    p = Path(a.status)
    d = J(p)
    d["publication_status"] = a.value
    dump_json(p, d)
    if a.compact_root:
        manifest(a.compact_root, Path(a.compact_root) / "compact_manifest.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("provenance")
    for x in ("out", "run-id", "checkout-sha", "workflow", "run-script", "analysis-script", "finalize-script", "plan", "validator", "pip-freeze"):
        p.add_argument("--" + x, required=True)
    p.set_defaults(fn=cmd_provenance)
    p = sub.add_parser("manifest")
    p.add_argument("--root", required=True); p.add_argument("--out", required=True); p.add_argument("--exclude", action="append")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("compact")
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.set_defaults(fn=cmd_compact)
    p = sub.add_parser("artifact-index")
    p.add_argument("--artifact-info", required=True); p.add_argument("--raw-manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--run-id", required=True)
    p.set_defaults(fn=cmd_artifact_index)
    p = sub.add_parser("status")
    p.add_argument("--status", required=True); p.add_argument("--value", required=True); p.add_argument("--compact-root")
    p.set_defaults(fn=cmd_status)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
