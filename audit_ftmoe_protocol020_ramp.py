"""Synthetic Protocol-020 R0 diagnostic for the S8 ramp boundary.

This constructs only an ``EAGateMoE`` source and the production
``OnlineEAGateV3`` wrapper.  It does not load a checkpoint, instantiate the
full model, train, or run the simulator.  The fifth expert is activated beside
four original experts whose scores are all above their thresholds.  The
diagnostic compares a ramp just below one with ramp equal to one, where V3
changes the hard active set from five entries to top-four.
"""

from pathlib import Path
import ctypes
import json

import torch

from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, EAGateMoE
from recovery.PreGANSrc.src.ftmoe_online_s8 import OnlineEAGateV3


SEED = 20260909
RAM_GUARD_GIB = 3.0
EPSILONS = (1e-3, 1e-5, 1e-7)


def _as_list(tensor):
    return tensor.detach().cpu().tolist()


def _set_zero_linear(linear, bias):
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.as_tensor(bias, dtype=linear.bias.dtype))


def build_gate():
    # Four source experts are copied into the V3 keyed gate.  A small hidden
    # size keeps this a pure, bounded structural probe.
    config = AblationConfig(
        hosts=1, hidden=8, classes=3, experts=4, dropout=0.0,
        eagate_residual_initial=1.0,
    )
    torch.manual_seed(SEED)
    source = EAGateMoE(config)
    gate = OnlineEAGateV3(source, seed=SEED)
    gate.eval()
    gate.record_enabled = False

    x = torch.tensor(
        [[[1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]]],
        dtype=torch.float32,
    )
    resources = torch.zeros((1, 1, 3), dtype=torch.float32)
    basis = torch.eye(config.hidden, dtype=torch.float32)

    # Make routing state exactly x, and put the four original keys on the
    # first four basis vectors.  All raw thresholds are -3 (tanh ~= -0.995),
    # so every original expert is eligible with distinct scores.
    with torch.no_grad():
        gate.resource_proj.weight.zero_()
        gate.resource_proj.bias.zero_()
        gate.temperature.fill_(1.0)
        gate.residual_gain.fill_(1.0)
        for index, key in enumerate(gate.ids):
            gate.key_rows[key].copy_(basis[index])
            gate.threshold_rows[key].fill_(-3.0)
            final = gate.experts[key][-1]
            final.weight.zero_()
            final.bias.zero_()
            final.bias[index] = float(index + 1)
            _set_zero_linear(
                gate.expert_detection_heads[key],
                (float(index + 1), -float(index + 1)),
            )
            _set_zero_linear(
                gate.expert_class_heads[key],
                (float(index + 1), 0.0, -float(index + 1)),
            )

    # create_shadow clones a real keyed expert and gives the new expert an ID;
    # its key and all heads are then fixed manually for this probe.
    candidate, parent = gate.create_shadow(basis[4])
    with torch.no_grad():
        gate.key_rows[candidate].copy_(basis[4])
        gate.threshold_rows[candidate].fill_(-3.0)
        final = gate.experts[candidate][-1]
        final.weight.zero_()
        final.bias.zero_()
        final.bias[4] = 5.0
        _set_zero_linear(gate.expert_detection_heads[candidate], (20.0, -20.0))
        _set_zero_linear(gate.expert_class_heads[candidate], (5.0, 0.0, -5.0))

    # Make the fifth expert active before changing its ramp.  This exposes the
    # actual ramp boundary in the production forward path.
    if gate.activate_shadow() != candidate or gate.shadow_id is not None:
        raise AssertionError("Failed to activate the fifth synthetic expert")
    with torch.no_grad():
        for key in gate.ids:
            if key != candidate:
                gate.ramp_steps_done[key] = 10

    return config, gate, x, resources, candidate, parent


def available_ram_gib():
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys / (1024 ** 3)


def run_diagnostic():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    available_gib = available_ram_gib()
    if available_gib < RAM_GUARD_GIB:
        raise RuntimeError(
            f"RAM guard failed: {available_gib:.3f} GiB available, "
            f"need at least {RAM_GUARD_GIB:.1f} GiB"
        )

    config, gate, x, resources, candidate, parent = build_gate()
    state = x + gate.resource_proj(resources)
    ids = list(gate.ids)
    if ids != ["0", "1", "2", "3", candidate]:
        raise AssertionError(f"Unexpected active IDs: {ids}")
    raw_scores = (torch.nn.functional.normalize(state, dim=-1) @
                  torch.nn.functional.normalize(
                      torch.stack([gate.key_rows[key] for key in ids]), dim=-1
                  ).T)
    raw_thresholds = torch.stack([gate.threshold_rows[key] for key in ids])
    thresholds = raw_thresholds.tanh()
    eligible = raw_scores >= thresholds
    if not bool(eligible.all()):
        raise AssertionError("Synthetic setup did not make all five experts eligible")

    with torch.no_grad():
        expert_outputs = torch.stack(
            [gate.experts[key](state)[0, 0] for key in ids]
        )
        expert_detection = torch.stack(
            [gate.expert_detection_heads[key](resources)[0, 0] for key in ids]
        )
    output_pairwise_min = min(
        float((expert_outputs[i] - expert_outputs[j]).abs().max())
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    )
    detection_pairwise_min = min(
        float((expert_detection[i] - expert_detection[j]).abs().max())
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    )
    if output_pairwise_min <= 0.0 or detection_pairwise_min <= 0.0:
        raise AssertionError("Synthetic expert outputs must be pairwise distinct")

    results = []
    for epsilon in EPSILONS:
        near_steps = 10.0 * (1.0 - epsilon)
        gate.ramp_steps_done[candidate] = near_steps
        with torch.no_grad():
            near = gate(x, resources)
        gate.ramp_steps_done[candidate] = 10
        with torch.no_grad():
            exact = gate(x, resources)

        near_output, near_weights, near_active_count, near_detection, _ = near
        exact_output, exact_weights, exact_active_count, exact_detection, _ = exact
        near_active = [ids[i] for i, value in enumerate(near_weights[0, 0])
                       if float(value) > 1e-12]
        exact_active = [ids[i] for i, value in enumerate(exact_weights[0, 0])
                        if float(value) > 1e-12]
        if near_active != ids:
            raise AssertionError(f"Expected five active experts below one: {near_active}")
        if exact_active != ids[:4]:
            raise AssertionError(f"Expected mature top-four at one: {exact_active}")
        symmetric_difference = sorted(
            set(near_active).symmetric_difference(exact_active)
        )
        results.append({
            "epsilon": epsilon,
            "candidate_ramp_steps_done": near_steps,
            "candidate_ramp_near_one": near_steps / 10.0,
            "candidate_ramp_equal_one": 1.0,
            "eligible_all_five": bool(eligible.all()),
            "near_one": {
                "weights": _as_list(near_weights[0, 0]),
                "active_ids": near_active,
                "active_count": float(near_active_count[0, 0]),
                "output": _as_list(near_output[0, 0]),
                "detection_logits": _as_list(near_detection[0, 0]),
            },
            "equal_one": {
                "weights": _as_list(exact_weights[0, 0]),
                "active_ids": exact_active,
                "active_count": float(exact_active_count[0, 0]),
                "output": _as_list(exact_output[0, 0]),
                "detection_logits": _as_list(exact_detection[0, 0]),
            },
            "max_abs_output_diff": float(
                (near_output - exact_output).abs().max()
            ),
            "max_abs_detection_logit_diff": float(
                (near_detection - exact_detection).abs().max()
            ),
            "active_set_changed": near_active != exact_active,
            "active_set_symmetric_difference": symmetric_difference,
        })

    root = Path(__file__).resolve().parent
    output_dir = root / "artifacts" / "ftmoe_online" / "protocol_020" / \
        "revision_20260909" / "r0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ramp_diagnostic.json"
    suffix = 2
    while output_path.exists():
        output_path = output_dir / f"ramp_diagnostic_{suffix}.json"
        suffix += 1

    report = {
        "diagnostic": "synthetic_s8_ramp_boundary",
        "interpretation": "合成结构诊断，不是 D 性能实验；不加载 checkpoint、不训练、不运行模拟器。",
        "seed": SEED,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "ram_guard_gib": RAM_GUARD_GIB,
        "available_ram_gib_at_start": available_gib,
        "config": {
            "hosts": config.hosts,
            "hidden": config.hidden,
            "classes": config.classes,
            "experts": config.experts,
            "temperature": 1.0,
        },
        "input": {
            "x": _as_list(x),
            "resources": _as_list(resources),
            "resource_projection_zeroed": True,
        },
        "expert_ids": ids,
        "candidate_id": candidate,
        "candidate_parent": parent,
        "raw_scores": _as_list(raw_scores[0, 0]),
        "raw_thresholds": _as_list(raw_thresholds),
        "thresholds_after_tanh": _as_list(thresholds),
        "eligible": _as_list(eligible[0, 0]),
        "expert_output_pairwise_min_abs_diff": output_pairwise_min,
        "expert_detection_pairwise_min_abs_diff": detection_pairwise_min,
        "expert_output_vectors": _as_list(expert_outputs),
        "expert_detection_logits": _as_list(expert_detection),
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({
        "output": str(output_path),
        "candidate": candidate,
        "parent": parent,
        "all_five_eligible": bool(eligible.all()),
        "results": [
            {
                "epsilon": row["epsilon"],
                "output_diff": row["max_abs_output_diff"],
                "detection_logit_diff": row["max_abs_detection_logit_diff"],
                "near_active": row["near_one"]["active_ids"],
                "equal_active": row["equal_one"]["active_ids"],
            }
            for row in results
        ],
    }, indent=2))


if __name__ == "__main__":
    run_diagnostic()
