"""Check the inactive expert-dropout candidate's train/eval semantics."""
import copy
import json
import torch
from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, SoftmaxMoE, EAGateMoE
from recovery.PreGANSrc.src.ftmoe_expert_regularization import enable_expert_dropout


def main():
    torch.set_num_threads(1); torch.manual_seed(908)
    for cls in [SoftmaxMoE,EAGateMoE]:
        base = cls(AblationConfig(experts=4)).eval()
        if cls is EAGateMoE:
            with torch.no_grad():
                for expert in base.experts:
                    expert[-1].weight.normal_(std=.1)
        rng = torch.get_rng_state().clone()
        candidate = enable_expert_dropout(copy.deepcopy(base),.3)
        assert torch.equal(rng,torch.get_rng_state())
        assert base.state_dict().keys() == candidate.state_dict().keys()
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,candidate.state_dict()[key],rtol=0,atol=0)
        x = torch.randn(2,16,64); resources = torch.rand(2,16,3)
        for a,b in zip(base(x,resources),candidate(x,resources)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        candidate.train()
        torch.manual_seed(19); first = candidate(x,resources)[0]
        torch.manual_seed(19); repeated = candidate(x,resources)[0]
        torch.testing.assert_close(first,repeated,rtol=0,atol=0)
        assert not torch.equal(first,candidate(x,resources)[0])
        first.square().sum().backward()
        assert candidate.experts[0][0].weight.grad.abs().sum()>0
    print(json.dumps({'parameters_and_rng_preserved':True,'eval_exact':True,
                      'seed_reproducibility':True,'training_stochastic':True,'gradients':True}))


if __name__=='__main__': main()
