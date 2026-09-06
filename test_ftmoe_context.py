"""Synthetic checks for the inactive local/global MoE context candidates."""
import copy
import json
import torch
from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, SoftmaxMoE
from recovery.PreGANSrc.src.ftmoe_context import enable_moe_context


def main():
    torch.set_num_threads(1); torch.manual_seed(701)
    baseline = SoftmaxMoE(AblationConfig(experts=4)).eval()
    state = torch.get_rng_state().clone()
    local = enable_moe_context(copy.deepcopy(baseline),'local')
    global_model = enable_moe_context(copy.deepcopy(baseline),'global')
    assert torch.equal(state,torch.get_rng_state())
    tokens = torch.randn(2,16,64)
    resources = torch.rand(2,16,3)
    for candidate in [local,global_model]:
        for old,new in zip(baseline(tokens,resources),candidate(tokens,resources)):
            torch.testing.assert_close(old,new,rtol=0,atol=0)
        assert sum(p.numel() for p in candidate.parameters())-sum(p.numel() for p in baseline.parameters()) == 4096
    with torch.no_grad():
        local.context_projection.weight.copy_(.2*torch.eye(64))
        global_model.context_projection.weight.copy_(.2*torch.eye(64))
    changed = tokens.clone(); changed[0,1] += torch.arange(64)/10
    old_local = local(tokens,resources)[1]
    new_local = local(changed,resources)[1]
    torch.testing.assert_close(old_local[0,0],new_local[0,0],rtol=0,atol=0)
    old_global = global_model(tokens,resources)[1]
    new_global = global_model(changed,resources)[1]
    assert not torch.allclose(old_global[0,0],new_global[0,0])
    torch.testing.assert_close(old_global[1],new_global[1],rtol=0,atol=0)
    permutation = torch.randperm(16)
    for before,after in zip(global_model(tokens,resources),global_model(tokens[:,permutation],resources[:,permutation])):
        torch.testing.assert_close(before[:,permutation],after)
    global_model.zero_grad()
    routing = global_model(tokens,resources)[1]
    (routing*torch.randn_like(routing)).sum().backward()
    assert global_model.context_projection.weight.grad.abs().sum() > 0
    print(json.dumps({'identical_initial_function':True,'rng_preserved':True,'matched_extra_parameters':4096,
                      'global_context_responds_to_other_hosts':True,'no_cross_sample_information':True,
                      'host_permutation_equivariance':True,'context_gradient':True}))


if __name__=='__main__':
    main()
