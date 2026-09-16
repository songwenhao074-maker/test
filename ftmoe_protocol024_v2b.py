"""Readable loader for the Protocol-024 v2b implementation.

The implementation is stored in line-preserving fragments under
maintenance/protocol024_v2b_parts/ to keep GitHub connector writes auditable.
The Actions workflow concatenates those fragments before execution; direct
imports use the same fragments through this loader.
"""
from pathlib import Path

_root = Path(__file__).resolve().parent
_parts = sorted((_root / "maintenance" / "protocol024_v2b_parts").glob("part*.pyfrag"))
if len(_parts) != 4:
    raise RuntimeError("expected four Protocol024 v2b source fragments")
_source = "".join(p.read_text(encoding="utf8") for p in _parts)
exec(compile(_source, str(_root / "ftmoe_protocol024_v2b_assembled.py"), "exec"), globals(), globals())
