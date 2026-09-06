"""Protocol 019 — unified input contract for FT-MoE online experiments.

Defines the physical meaning, units and generation policy of every model
input column, and validates stream manifests against the contract.

Contract version: 2
- Every model input feature is declared here.  A column whose training-time
  semantics differ from online-time semantics must be declared as such, never
  silently reused with a different physical meaning.
- The two hardware groups are part of the contract because normalization v2
  falls back to same-group statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

# Feature order shared by every (T,16,7) array in the pipeline
# (host_features, demands / container_demand_series, time_series rows).
HOST_FEATURES = [
    "cpu_demand",
    "ram_space",
    "ram_read_or_network_rx",
    "ram_write_or_network_tx",
    "disk_space",
    "disk_read",
    "disk_write",
]

# CPU / RAM / Disk resource columns in capacity arrays (T,16,3) and in
# graph_host_capacity normalization rows.
RESOURCE_COLUMNS = ["cpu", "ram", "disk"]

# feature index of cpu_demand, ram_space, disk_space (used wherever a raw
# resource ratio is computed, e.g. EAGate raw_resources or capacity ratios).
RESOURCE_FEATURE_INDEX = {"cpu": 0, "ram": 1, "disk": 4}

# Hardware groups used by normalization-v2 fallback (same group => same Pi
# model; hosts 0-7 are Raspberry Pi 4GB, hosts 8-15 Raspberry Pi 8GB).
HARDWARE_GROUPS = {
    "rpi4gb": list(range(0, 8)),
    "rpi8gb": list(range(8, 16)),
}


def host_group(host: int) -> str:
    if not 0 <= host < 16:
        raise ValueError(f"host index out of contract range: {host}")
    return "rpi4gb" if host < 8 else "rpi8gb"


def validate_feature_columns(features: list | tuple) -> None:
    if list(features) != HOST_FEATURES:
        raise ValueError(
            f"feature columns violate input contract v2: {list(features)}"
        )


def validate_stream_manifest(manifest: dict, stream_dir: Path | None = None) -> dict:
    """Structural validation of a Protocol 019 stream manifest.

    Returns a report dict; raises ValueError on contract violations that make
    the stream unusable for Protocol 019.
    """
    report: dict = {"contract_version": 2, "violations": []}
    for key in ("seed", "steps", "workload", "interval_seconds", "hosts"):
        if key not in manifest:
            report["violations"].append(f"missing manifest key: {key}")
    if "capacity_scales" not in manifest:
        report["violations"].append("missing manifest key: capacity_scales")
    if manifest.get("hosts") != 16:
        report["violations"].append(f"hosts != 16: {manifest.get('hosts')}")
    if manifest.get("steps", 0) < 300:
        report["violations"].append("development streams must have >= 300 steps")
    if stream_dir is not None and "stream_sha256" in manifest:
        import hashlib

        def sha(path: Path) -> str:
            result = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    result.update(block)
            return result.hexdigest()

        actual = sha(stream_dir / "stream.npz")
        if manifest["stream_sha256"] != actual:
            report["violations"].append(
                f"stream_sha256 mismatch: manifest {manifest['stream_sha256'][:12]}... "
                f"actual {actual[:12]}..."
            )
    return report


def write_contract_report(path: Path | str, report: dict) -> None:
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
