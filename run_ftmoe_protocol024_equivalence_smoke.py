"""Protocol-024 engineering smoke: fixed-C vs dynamic-bank with lifecycle disabled.

This is NOT a performance experiment.  It proves that replacing the fixed
residual bank with ``DynamicResidualBank`` does not itself change the online
learner when birth/retire/reactivate are disabled.

Both sessions use the Protocol-023 calibrated seed-700 stream, the same frozen
checkpoint, replay/anchor data, strict-prequential loop, update budget, seed,
loss and optimizer settings.  The only implementation difference is the bank
container.  The smoke compares every prediction and the mapped residual
parameters after several real online updates.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_dynamic_residual import DynamicResidualBank


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = (ROOT / "artifacts/ftmoe_online/protocol_024/continuation_20260914"
                    / "equivalence_smoke.json")


def _make_session(dynamic: bool, seed: int, root: Path):
    budget = json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))
    frozen = budget["frozen_configuration"]
    bundle = s4.build_replay(s4.DEV_STREAM)
    anchor = s4.load_anchor_pool(bundle["replay"].time_scale)
    out = root / ("dynamic" if dynamic else "fixed")
    session = s4.PrequentialS4(
        "C", seed, bundle, frozen, out, probe_paths=None,
        anchor=anchor, learning_rate=1e-4,
    )
    if not dynamic:
        return session

    # Convert ONLY the learner bank.  The frozen base, data, loss, optimizer
    # family, update opportunities and deployment alpha remain unchanged.
    source = session.model.learner
    dynamic_bank = DynamicResidualBank(source, max_experts=8, ramp_updates=4)
    session.model.learner = dynamic_bank
    session.model.set_trainability()
    session.model.set_deployment("learner", 1.0)
    trainable = [p for p in session.model.parameters() if p.requires_grad]
    session.optimizer = torch.optim.AdamW(
        trainable, lr=session.learning_rate, weight_decay=1e-4)
    session.learner_hash = session.model.learner_state_hash()
    session.protocol024_dynamic_container = True
    return session


def _parameter_differences(fixed, dynamic):
    result = {}
    fixed_bank = fixed.model.learner
    dynamic_bank = dynamic.model.learner

    fixed_w = fixed_bank.router.weight.detach()
    dynamic_w = torch.stack(
        [dynamic_bank.router_weights[key] for key in dynamic_bank.ids], dim=0).detach()
    fixed_b = fixed_bank.router.bias.detach()
    dynamic_b = torch.stack(
        [dynamic_bank.router_biases[key] for key in dynamic_bank.ids], dim=0).detach()
    result["router_weight_max_abs"] = float((fixed_w - dynamic_w).abs().max())
    result["router_bias_max_abs"] = float((fixed_b - dynamic_b).abs().max())

    expert_max = 0.0
    for index, key in enumerate(dynamic_bank.ids):
        left = fixed_bank.experts[index].state_dict()
        right = dynamic_bank.experts[key].state_dict()
        if set(left) != set(right):
            raise AssertionError("expert state keys differ for %s" % key)
        for name in left:
            expert_max = max(
                expert_max,
                float((left[name].detach() - right[name].detach()).abs().max()))
    result["expert_tensor_max_abs"] = expert_max
    return result


def run(intervals=24, seed=1, atol=2e-6, rtol=2e-6):
    with tempfile.TemporaryDirectory(prefix="p24_equivalence_") as temp:
        temp = Path(temp)
        fixed = _make_session(False, seed, temp)
        dynamic = _make_session(True, seed, temp)

        initial_fixed_count = sum(
            p.numel() for p in fixed.model.learner.parameters())
        initial_dynamic_count = dynamic.model.learner.active_parameter_count()
        if initial_fixed_count != initial_dynamic_count:
            raise AssertionError("initial residual parameter counts differ")

        probability_max_abs = 0.0
        class_probability_max_abs = 0.0
        for _ in range(int(intervals)):
            p_fixed, c_fixed = fixed.step()
            p_dynamic, c_dynamic = dynamic.step()
            probability_max_abs = max(
                probability_max_abs, float(np.max(np.abs(p_fixed - p_dynamic))))
            class_probability_max_abs = max(
                class_probability_max_abs,
                float(np.max(np.abs(c_fixed - c_dynamic))))
            if not np.allclose(p_fixed, p_dynamic, atol=atol, rtol=rtol):
                raise AssertionError("detection probability diverged at interval %d"
                                     % fixed.cursor)
            if not np.allclose(c_fixed, c_dynamic, atol=atol, rtol=rtol):
                raise AssertionError("class probability diverged at interval %d"
                                     % fixed.cursor)

        differences = _parameter_differences(fixed, dynamic)
        parameter_tol = 5e-6
        parameter_pass = all(v <= parameter_tol for v in differences.values())
        topology = dynamic.model.learner.topology_manifest()
        topology_pass = (
            topology["active_ids"] == ["0", "1", "2", "3"]
            and not topology["dormant_ids"]
            and topology["shadow_id"] is None
            and all(abs(v - 1.0) <= 1e-12 for v in topology["ramp"].values())
        )
        updates_pass = fixed.updates == dynamic.updates
        frozen_pass = (
            fixed.model.frozen_hash() == fixed.frozen_hash
            and dynamic.model.frozen_hash() == dynamic.frozen_hash)
        passed = bool(parameter_pass and topology_pass and updates_pass and frozen_pass)

        return {
            "protocol": "024",
            "kind": "fixed_vs_dynamic_container_equivalence_smoke",
            "formal_model_result": False,
            "stream": str(s4.DEV_STREAM.relative_to(ROOT)),
            "seed": int(seed),
            "intervals": int(intervals),
            "updates_fixed": int(fixed.updates),
            "updates_dynamic": int(dynamic.updates),
            "initial_residual_parameter_count": int(initial_fixed_count),
            "probability_max_abs": probability_max_abs,
            "class_probability_max_abs": class_probability_max_abs,
            "parameter_differences": differences,
            "parameter_tolerance": parameter_tol,
            "topology": topology,
            "checks": {
                "predictions_close": True,
                "mapped_parameters_close": bool(parameter_pass),
                "update_count_equal": bool(updates_pass),
                "frozen_base_unchanged": bool(frozen_pass),
                "lifecycle_stayed_disabled": bool(topology_pass),
            },
            "passed": passed,
            "interpretation": (
                "Container-equivalence smoke only. Passing does not show D is better "
                "than C; it shows the dynamic residual representation is a fair "
                "starting point before lifecycle operations are enabled."
            ),
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = run(intervals=args.intervals, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
