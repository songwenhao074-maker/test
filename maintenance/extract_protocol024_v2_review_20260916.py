"""Extract complete top-level v2 result objects from GitHub job stdout."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def extract(text):
    clean = re.sub(r"^\ufeff?\d{4}-\d\d-\d\dT\S+[ \t]?", "", text, flags=re.M)
    clean = "\n".join(line for line in clean.splitlines()
        if not (line.startswith("/opt/") and "UserWarning: enable_nested_tensor is True" in line)
        and not line.strip().startswith('warnings.warn(f"enable_nested_tensor is True'))
    decoder = json.JSONDecoder()
    results = []
    for match in re.finditer(r"^\{", clean, flags=re.M):
        try:
            value, _ = decoder.raw_decode(clean[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if ("status" in value and "recurrence" in value and "lifecycle" in value):
            results.append(value)
        elif value.get("round") == "next_round_v2" and "first_budget" in value:
            results.append(value)
    if not results:
        raise ValueError("no complete v2 result objects found")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.job_log.read_bytes()
    values = extract(raw.decode("utf-8-sig"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        output.write(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"objects": len(values), "job_log_sha256": hashlib.sha256(raw).hexdigest()}))

if __name__ == "__main__":
    main()
