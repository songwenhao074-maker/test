"""Synthetic semantics checks for an optional, currently inactive graph candidate."""
import json
import torch
from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, ScheduleGraphEncoder
from recovery.PreGANSrc.src.ftmoe_capacity_graph import CapacityRatioGraphEncoder


def main():
    torch.set_num_threads(1)
    torch.manual_seed(918)
    cfg = AblationConfig(experts=4)
    baseline = ScheduleGraphEncoder(cfg).eval()
    rng_before = torch.get_rng_state().clone()
    candidate = CapacityRatioGraphEncoder.from_existing(baseline)
    assert torch.equal(rng_before, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    demands = torch.rand(2, 16, 12, 7)
    schedule = torch.eye(16).reshape(1,1,16,16).expand(2,12,-1,-1).clone()
    schedule[:,6:,0] = 0
    schedule[:,6:,0,3] = 1
    with torch.no_grad():
        torch.testing.assert_close(candidate(demands,schedule), baseline(demands,schedule), rtol=0, atol=0)
    example = torch.rand(2,12,16,13) + .1
    example[..., -3:] = torch.tensor([2.,4.,8.])
    converted_units = example.clone()
    factors = torch.tensor([1000., .001, 100.])
    converted_units[..., [0,1,4]] *= factors
    converted_units[..., -3:] *= factors
    torch.testing.assert_close(candidate.resource_ratios(example), candidate.resource_ratios(converted_units))
    with torch.no_grad():
        candidate.ratio_projection.weight.fill_(.1)
    # The branch must respond to physical capacity changes, not just demand.
    smaller_capacity = example.clone()
    smaller_capacity[..., -3:] *= .5
    assert torch.all(candidate.ratio_projection(candidate.resource_ratios(smaller_capacity)) >
                     candidate.ratio_projection(candidate.resource_ratios(example)))
    candidate.zero_grad()
    output = candidate(demands,schedule)
    (output * torch.randn_like(output)).sum().backward()
    assert candidate.ratio_projection.weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in baseline.parameters()) == 192
    print(json.dumps({'identical_initial_function':True, 'old_parameters_and_rng_preserved':True,
                      'resource_unit_invariance':True, 'capacity_sensitivity':True,
                      'new_branch_receives_gradient':True, 'extra_parameters':192}))


if __name__=='__main__':
    main()
