#!/usr/bin/env python3
"""Extract Protocol-024 comparison from downloaded GitHub job stdout.

Usage: python maintenance/extract_protocol024_review_20260915.py JOB_LOG OUTPUT
Only timestamp prefixes and the two known interleaved PyTorch warning lines
are removed. Numeric values and scientific conclusions are never rewritten.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

def extract(text):
    clean = re.sub(r"^\ufeff?\d{4}-\d\d-\d\dT\S+[ \t]?", "", text, flags=re.M)
    lines = []
    for line in clean.splitlines():
        if line.startswith("/opt/") and "UserWarning: enable_nested_tensor is True" in line:
            continue
        if line.strip().startswith('warnings.warn(f"enable_nested_tensor is True'):
            continue
        lines.append(line)
    clean = "\n".join(lines)
    decoder = json.JSONDecoder()
    found = []
    for match in re.finditer(r"^\{", clean, flags=re.M):
        try:
            value, _ = decoder.raw_decode(clean[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"full_detection", "switch_first100", "lifecycle", "development_signal"} <= set(value):
            found.append(value)
    if len(found) != 1:
        raise ValueError("expected exactly one complete comparison; found %d" % len(found))
    return found[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.job_log.read_bytes()
    data = extract(raw.decode("utf-8-sig"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(json.dumps({"job_log_sha256": hashlib.sha256(raw).hexdigest(),
                      "output": str(args.output), "development_signal": data["development_signal"]}))

if __name__ == "__main__":
    main()
