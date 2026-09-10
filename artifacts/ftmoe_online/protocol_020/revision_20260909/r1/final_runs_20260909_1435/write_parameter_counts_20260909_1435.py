from __future__ import annotations

import json
from pathlib import Path

import torch

from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE


REPO = Path(r"F:\PreGANPlus-master")
CHECKPOINT_PATH = REPO / "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt"
OUTPUT_PATH = REPO / "artifacts/ftmoe_online/protocol_020/revision_20260909/r1/final_runs_20260909_1435/parameter_counts.json"


def model_counts(model):
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    return {
        "total_numel": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_numel": int(sum(parameter.numel() for parameter in trainable)),
        "total_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in parameters)),
        "trainable_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in trainable)),
    }


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    online_a = OnlineFTMoE(checkpoint, "A", 1)
    online_c = OnlineFTMoE(checkpoint, "C", 1)
    frozen_residual_c = FrozenResidualFTMoE(checkpoint, "C", 1)

    correction_numel = frozen_residual_c.correction_parameter_counts()
    correction_bytes = frozen_residual_c.correction_memory_bytes()
    payload = {
        "schema_version": 1,
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_variant": checkpoint.get("variant"),
        "checkpoint_seed": checkpoint.get("seed"),
        "model_seed": 1,
        "forward_called": False,
        "training_called": False,
        "models": {
            "OnlineFTMoE_A": model_counts(online_a),
            "OnlineFTMoE_C": model_counts(online_c),
            "FrozenResidualFTMoE_C": model_counts(frozen_residual_c),
        },
        "frozen_residual_correction_snapshots": {
            "numel": {name: int(value) for name, value in correction_numel.items()},
            "bytes": {name: int(value) for name, value in correction_bytes.items()},
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
