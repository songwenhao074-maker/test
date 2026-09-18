"""Extract reviewable v2c or Protocol025 audit JSON from GitHub job stdout."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def extract(text, kind):
    clean = re.sub(r"^\ufeff?\d{4}-\d\d-\d\dT\S+[ \t]?", "", text, flags=re.M)
    clean = "\n".join(line for line in clean.splitlines()
        if not (line.startswith("/opt/") and "UserWarning: enable_nested_tensor is True" in line)
        and not line.strip().startswith('warnings.warn(f"enable_nested_tensor is True'))
    decoder = json.JSONDecoder()
    matches = []
    for match in re.finditer(r"^\{", clean, flags=re.M):
        try:
            value, _ = decoder.raw_decode(clean[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if kind == "v2c" and value.get("status", {}).get("round") == "next_round_v2c" and "recurrence" in value:
            matches.append(value)
        if kind == "data_audit" and {"audit_pass", "gates", "F0_guard", "stream_sha256"} <= set(value):
            matches.append(value)
    if len(matches) != 1:
        raise ValueError("expected exactly one matching result, found %d" % len(matches))
    return matches[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["v2c", "data_audit"])
    parser.add_argument("job_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.job_log.read_bytes()
    value = extract(raw.decode("utf-8-sig"), args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        output.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"kind": args.kind, "job_log_sha256": hashlib.sha256(raw).hexdigest()}))

if __name__ == "__main__":
    main()
