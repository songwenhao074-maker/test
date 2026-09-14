"""Verify published experiment inputs using only the Python standard library."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "maintenance/github_sync_20260914/required_assets.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    failures = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
        elif path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            failures.append({"path": row["path"], "reason": "bytes_or_sha256_mismatch"})
    result = {"passed": not failures, "checked_files": len(manifest["files"]),
              "checked_bytes": sum(row["bytes"] for row in manifest["files"]),
              "failures": failures,
              "scope": "File completeness only; run regression tests separately. Known legacy metric bugs remain."}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
