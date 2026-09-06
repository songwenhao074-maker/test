"""Protocol 020 S3 — build the source-disjoint VM cohort split (vm_split.json).

Data-only stage: NO model outputs / fault labels / D-C anywhere in this file.

Pool: every local Bitbrain rnd VM CSV (simulator/workload/datasets/bitbrain/rnd/).
Per-VM statistics are computed from the RAW trace only (documented units):

- cpu_usage_mhz -> cpu_ips = cpu_usage_mhz * IPS_MULT (IPS_MULT = 2054/(2*600))
- ram_units = mem_usage_kb / 4000            (BWGD2/016 demand-adapter base unit)
- disk_read_kbs / disk_write_kbs             (descriptive only)

Pre-registered eligibility criteria (fixed before any scan result was seen):
1. rows >= MIN_ROWS (400)
2. nan_ratio < 0.2      (fraction of NaN cells among numeric cols 3..8)
3. cpu active ratio (cpu_usage_mhz > 0 among valid rows) >= 0.05

Pre-registered split rule (deterministic, source-disjoint):
1. eligible VMs are stratified by (cpu_ips_mean tercile x ram_units_mean tercile)
2. inside each stratum: bucket = sha256("p20:v1:<vm_id>") mod 10
3. bucket 0-5 -> train, 6-7 -> dev, 8-9 -> online

Run: python build_ftmoe_protocol020_vm_split.py [--out ...]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "simulator/workload/datasets/bitbrain/rnd"
DEFAULT_OUT = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"

IPS_MULT = 2054.0 / (2 * 600)
MIN_ROWS = 400
MAX_NAN_RATIO = 0.2
MIN_CPU_ACTIVE_RATIO = 0.05
BUCKET_RULE = {"train": (0, 5), "dev": (6, 7), "online": (8, 9)}


def parse_vm(path):
    """Return (rows, columns) with columns as float numpy arrays (NaN for missing).

    CSV schema (0-based cols): 0 ts, 1 cores, 2 cpu cap prov, 3 cpu usage mhz,
    4 cpu usage pct, 5 mem cap prov kb, 6 mem usage kb, 7 disk read kb/s,
    8 disk write kb/s, 9 net recv kb/s, 10 net send kb/s.
    """
    numeric = {3, 6, 7, 8}
    values = {i: [] for i in numeric}
    rows = 0
    with Path(path).open("r", encoding="utf8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if not row:
                continue
            rows += 1
            if rows == 1:
                continue  # header
            for i in numeric:
                raw = row[i].strip() if i < len(row) else ""
                try:
                    values[i].append(float(raw))
                except ValueError:
                    values[i].append(float("nan"))
    arrays = {i: np.asarray(values[i], dtype=np.float64) for i in numeric}
    return rows - 1, arrays


def stats_for(vm_id):
    path = DATA / f"{vm_id}.csv"
    rows, arrays = parse_vm(path)
    cpu = arrays[3]
    mem = arrays[6]
    valid_cpu = cpu[np.isfinite(cpu)]
    valid_mem = mem[np.isfinite(mem)]
    nan_ratio = 1.0 - np.isfinite(cpu).mean()
    cpu_zero = float((valid_cpu == 0).mean()) if valid_cpu.size else float("nan")
    mem_zero = float((valid_mem == 0).mean()) if valid_mem.size else float("nan")
    cpu_active = float((valid_cpu > 0).mean()) if valid_cpu.size else 0.0
    cpu_ips = valid_cpu * IPS_MULT
    ram_units = valid_mem / 4000.0
    pct = lambda x: (np.percentile(x, 50), np.percentile(x, 95))  # noqa: E731
    d50, d95 = pct(arrays[7][np.isfinite(arrays[7])]) if np.isfinite(arrays[7]).any() else (0.0, 0.0)
    w50, w95 = pct(arrays[8][np.isfinite(arrays[8])]) if np.isfinite(arrays[8]).any() else (0.0, 0.0)
    return {
        "vm_id": vm_id,
        "rows": rows,
        "nan_ratio": float(nan_ratio),
        "cpu_usage_mhz_mean": float(valid_cpu.mean()) if valid_cpu.size else 0.0,
        "cpu_ips_mean": float(cpu_ips.mean()) if cpu_ips.size else 0.0,
        "cpu_ips_p95": float(pct(cpu_ips)[1]) if cpu_ips.size else 0.0,
        "cpu_zero_ratio": cpu_zero,
        "cpu_active_ratio": cpu_active,
        "mem_usage_kb_mean": float(valid_mem.mean()) if valid_mem.size else 0.0,
        "ram_units_mean": float(ram_units.mean()) if ram_units.size else 0.0,
        "ram_units_p95": float(pct(ram_units)[1]) if ram_units.size else 0.0,
        "mem_zero_ratio": mem_zero,
        "disk_read_kbs_mean": float(d50),
        "disk_read_kbs_p95": float(d95),
        "disk_write_kbs_mean": float(w50),
        "disk_write_kbs_p95": float(w95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    vm_ids = sorted(int(p.stem) for p in DATA.glob("*.csv") if p.stem.isdigit())
    if not vm_ids:
        raise RuntimeError("No Bitbrain rnd CSV files found under %s" % DATA)
    print("scanning %d VM traces ..." % len(vm_ids), flush=True)
    all_stats = [stats_for(i) for i in vm_ids]

    eligible = [s for s in all_stats
                if s["rows"] >= MIN_ROWS
                and s["nan_ratio"] < MAX_NAN_RATIO
                and s["cpu_active_ratio"] >= MIN_CPU_ACTIVE_RATIO]
    pool_size = len(eligible)
    if pool_size < 30:
        raise RuntimeError("Eligible pool too small: %d (criteria too strict?)" % pool_size)

    # Stratification: terciles of cpu_ips_mean and ram_units_mean over eligible VMs.
    def tercile(values, key):
        col = np.asarray([v[key] for v in values], dtype=np.float64)
        cut = np.percentile(col, [33.3333, 66.6667])
        return np.searchsorted(cut, col, side="right")

    cpu_ter = tercile(eligible, "cpu_ips_mean")
    ram_ter = tercile(eligible, "ram_units_mean")
    for s, ct, rt in zip(eligible, cpu_ter, ram_ter):
        digest = hashlib.sha256(f"p20:v1:{s['vm_id']}".encode("utf8")).hexdigest()
        bucket = int(digest, 16) % 10
        s["cpu_tercile"] = int(ct)
        s["ram_tercile"] = int(rt)
        s["bucket"] = bucket
        for cohort, (lo, hi) in BUCKET_RULE.items():
            if lo <= bucket <= hi:
                s["cohort"] = cohort
                break

    cohorts = {name: sorted(s["vm_id"] for s in eligible if s["cohort"] == name)
               for name in BUCKET_RULE}
    # Test 9 — source split disjointness (asserted here and re-checked by consumers).
    a, b, c = (set(v) for v in cohorts.values())
    assert not (a & b) and not (a & c) and not (b & c), "cohorts overlap"
    assert a | b | c == set(s["vm_id"] for s in eligible), "cohorts do not partition pool"
    for name, ids in cohorts.items():
        print("cohort %-6s %3d VMs  cpu_ips_mean %.1f..%.1f  ram_units_mean %.1f..%.1f" % (
            name, len(ids),
            min(s["cpu_ips_mean"] for s in eligible if s["cohort"] == name),
            max(s["cpu_ips_mean"] for s in eligible if s["cohort"] == name),
            min(s["ram_units_mean"] for s in eligible if s["cohort"] == name),
            max(s["ram_units_mean"] for s in eligible if s["cohort"] == name)), flush=True)

    out = {
        "schema_version": 1,
        "protocol": "020",
        "name": "vm source-disjoint split",
        "pool": "bitbrain rnd local CSV",
        "scanned_vm_count": len(vm_ids),
        "eligible_vm_count": pool_size,
        "criteria": {"min_rows": MIN_ROWS, "max_nan_ratio": MAX_NAN_RATIO,
                     "min_cpu_active_ratio": MIN_CPU_ACTIVE_RATIO,
                     "units": {"cpu_ips": "cpu_usage_mhz*2054/1200",
                               "ram_units": "mem_usage_kb/4000"}},
        "split_rule": "stratum=(cpu_ips_mean_tercile,ram_units_mean_tercile); "
                      "bucket=sha256('p20:v1:<vm_id>') mod 10; "
                      "bucket 0-5 train / 6-7 dev / 8-9 online",
        "bucket_rule": {k: {"min": v[0], "max": v[1]} for k, v in BUCKET_RULE.items()},
        "cohorts": cohorts,
        "per_vm": {str(s["vm_id"]): {k: s[k] for k in
                   ("rows", "nan_ratio", "cpu_ips_mean", "cpu_ips_p95",
                    "cpu_usage_mhz_mean", "cpu_zero_ratio", "cpu_active_ratio",
                    "mem_usage_kb_mean", "ram_units_mean", "ram_units_p95",
                    "mem_zero_ratio", "disk_read_kbs_mean", "disk_read_kbs_p95",
                    "disk_write_kbs_mean", "disk_write_kbs_p95",
                    "cpu_tercile", "ram_tercile", "bucket", "cohort")}
                   for s in eligible},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.out) + ".tmp")
    temporary.write_text(json.dumps(out, indent=1) + "\n", encoding="utf8")
    temporary.replace(args.out)
    print("wrote %s (eligible %d VMs)" % (args.out, pool_size), flush=True)


if __name__ == "__main__":
    main()
