"""Integrated Protocol-024 strict-prequential C/D equivalence smoke.

Unlike the earlier container smoke, this exercises ``Protocol024Session``
directly on the tracked seed-700 development stream. Lifecycle operations stay
disabled. Passing proves the final D runner starts from the same online behavior
as C before lifecycle logic is enabled; it is not a superiority result.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_session import Protocol024Session

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = (ROOT / "artifacts/ftmoe_online/protocol_024/continuation_20260914"
                    / "integrated_session_smoke.json")


def run(intervals=24, seed=1, atol=2e-6, rtol=2e-6):
    budget = json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))
    frozen = budget["frozen_configuration"]
    bundle_c = s4.build_replay(s4.DEV_STREAM)
    bundle_d = s4.build_replay(s4.DEV_STREAM)
    anchor_c = s4.load_anchor_pool(bundle_c["replay"].time_scale)
    anchor_d = s4.load_anchor_pool(bundle_d["replay"].time_scale)

    with tempfile.TemporaryDirectory(prefix="p24_session_") as temp:
        temp = Path(temp)
        fixed = Protocol024Session(
            "C", seed, bundle_c, frozen, temp / "C", anchor=anchor_c,
            learning_rate=1e-4)
        dynamic = Protocol024Session(
            "D", seed, bundle_d, frozen, temp / "D", anchor=anchor_d,
            learning_rate=1e-4, max_experts=8, ramp_updates=10)

        p_diff = 0.0
        c_diff = 0.0
        for _ in range(int(intervals)):
            p_c, cls_c = fixed.step()
            p_d, cls_d = dynamic.step()
            p_diff = max(p_diff, float(np.max(np.abs(p_c - p_d))))
            c_diff = max(c_diff, float(np.max(np.abs(cls_c - cls_d))))
            if not np.allclose(p_c, p_d, atol=atol, rtol=rtol):
                raise AssertionError("C/D detection diverged at interval %d" % fixed.cursor)
            if not np.allclose(cls_c, cls_d, atol=atol, rtol=rtol):
                raise AssertionError("C/D diagnosis diverged at interval %d" % fixed.cursor)

        topology = dynamic.model.learner.topology_manifest()
        lifecycle_off = (
            topology["active_ids"] == ["0", "1", "2", "3"]
            and not topology["dormant_ids"]
            and topology["shadow_id"] is None
            and all(abs(v - 1.0) <= 1e-12 for v in topology["ramp"].values())
        )
        checks = {
            "updates_equal": fixed.updates == dynamic.updates,
            "frozen_C_unchanged": fixed.model.frozen_hash() == fixed.frozen_hash,
            "frozen_D_unchanged": dynamic.model.frozen_hash() == dynamic.frozen_hash,
            "lifecycle_disabled": lifecycle_off,
        }
        passed = all(checks.values())
        return {
            "protocol": "024",
            "kind": "integrated_strict_prequential_C_D_equivalence_smoke",
            "formal_model_result": False,
            "seed": int(seed),
            "intervals": int(intervals),
            "updates_C": int(fixed.updates),
            "updates_D": int(dynamic.updates),
            "probability_max_abs": p_diff,
            "class_probability_max_abs": c_diff,
            "D_topology": topology,
            "checks": checks,
            "passed": bool(passed),
            "interpretation": (
                "Protocol024Session D matches C while lifecycle is disabled. "
                "This is an integration/fairness gate, not evidence that D improves C."
            ),
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = run(args.intervals, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
