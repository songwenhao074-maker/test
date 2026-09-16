"""Reproduce the v2b preview/deployment mismatch using the reviewed bank.
Synthetic code-path evidence, not a measured performance result.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from ftmoe_protocol024_dynamic_residual import DynamicResidualBank

torch.set_num_threads(1)
source = nn.Module()
source.router = nn.Linear(1, 4)
source.experts = nn.ModuleList([nn.Linear(1, 5) for _ in range(4)])
with torch.no_grad():
    source.router.weight.zero_()
    source.router.bias.zero_()
    for expert in source.experts:
        expert.weight.zero_()
        expert.bias.zero_()
        expert.bias[1] = -2.0
bank = DynamicResidualBank(source, max_experts=8)
z = torch.zeros(1, 1)
old = bank.create_shadow("0")
with torch.no_grad():
    bank.shadow_experts[old].bias[1] = 10.0
bank.activate_shadow()
bank.set_ramp(old, 1.0)
live, _ = bank(z)
new = bank.create_shadow("0")
with torch.no_grad():
    bank.shadow_experts[new].bias[1] = 1.0
preview, _ = bank.preview_with_shadow(z, shadow_ramp=1.0)
bank.activate_shadow()
bank.set_ramp(new, 1.0)
bank.set_ramp(old, 0.0)
bank.retire(old)
deployed, _ = bank(z)
def loss(output):
    return float(nn.functional.cross_entropy(output[..., :2], torch.ones(1, dtype=torch.long)))
result = {
    "kind": "synthetic_reviewed_bank_reproduction",
    "source_commit": "c2ada4446199c0c9a660ac096d2a92cdb224262e",
    "live_anomaly_logit": float(live[0, 1]),
    "qualification_anomaly_logit": float(preview[0, 1]),
    "deployed_anomaly_logit": float(deployed[0, 1]),
    "live_positive_CE": loss(live),
    "qualification_positive_CE": loss(preview),
    "deployed_positive_CE": loss(deployed),
    "preview_deployment_max_abs": float((preview-deployed).abs().max()),
    "is_seed700_performance_evidence": False,
}
result["qualification_CE_relative_improvement"] = (
    result["live_positive_CE"]-result["qualification_positive_CE"])/result["live_positive_CE"]
assert result["qualification_CE_relative_improvement"] > .01
assert result["deployed_positive_CE"] > result["live_positive_CE"]
assert result["preview_deployment_max_abs"] > 1.0
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path)
args = parser.parse_args()
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as output:
        output.write(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
