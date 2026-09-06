"""Structural tests for an inactive controlled gate-replacement family."""
import json
import torch
from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig
from recovery.PreGANSrc.src.ftmoe_matched_routing import FTMoEMatchedRouting
from train_ftmoe_end_to_end import loss_fn


def main():
    torch.set_num_threads(1)
    cfg = AblationConfig(experts=4,moe_residual_initial=0.,graph_residual_initial=0.,cmha_residual_initial=0.)
    models = {}; shared = {}
    for variant in ['v0','v1','v2','v3','v4']:
        torch.manual_seed(1); model = FTMoEMatchedRouting(variant,cfg)
        assert model.eagate is None
        for name,p in model.named_parameters():
            if name in shared: torch.testing.assert_close(p,shared[name],rtol=0,atol=0)
            else: shared[name] = p.detach().clone()
        models[variant] = model
    assert sum(p.numel() for p in models['v1'].parameters())==sum(p.numel() for p in models['v2'].parameters())
    x = torch.rand(2,16,12,7); g = torch.rand_like(x)
    schedule = torch.eye(16)[None,None].expand(2,12,-1,-1)
    changed_schedule = schedule.roll(1,dims=-1)
    labels = (torch.arange(32)%4).reshape(2,16)
    for variant,model in models.items():
        model.eval()
        with torch.no_grad():
            # Schedule independence is also checked after adapters have trained.
            if model.moe is not None:
                model.moe.detection_adapter.weight.normal_(std=.1)
                model.moe.class_adapter.weight.normal_(std=.1)
            first = model(x,schedule,g)
            if variant in ['v0','v1','v2']:
                second = model(x,changed_schedule,g*3)
                for key in ['detection_logits','class_logits']:
                    torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
            else:
                assert not torch.allclose(model.graph_encoder(g,schedule),model.graph_encoder(g,changed_schedule))
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        for _ in range(4):
            model.train(); optimizer.zero_grad()
            output = model(x,schedule,g)
            loss = loss_fn(model,output,labels,.7,.3,0.,.01,2.,.5)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
        changed = [n for n,p in model.named_parameters() if not torch.equal(p,before[n])]
        for prefix in ['encoder','moe','graph_encoder','cmha']:
            if getattr(model,prefix,None) is not None:
                assert any(n.startswith(prefix+'.') for n in changed),(variant,prefix)
        if variant!='v0':
            assert 'moe.resource_proj.weight' in changed and 'moe.router.bias' in changed
            assert list(model.routing_outputs)==['moe']
            torch.testing.assert_close(output['router_probabilities'].sum(-1),torch.ones(2,16))
            assert (output['active_experts']>=1).all() and (output['active_experts']<=4).all()
    # Extreme thresholds exercise the mandatory top-1 fallback and 4-expert cap.
    gate = models['v2'].moe
    with torch.no_grad(): gate.router.bias.fill_(100.)
    probabilities = gate.probabilities(torch.rand(2,16,64))
    assert (gate.active_experts==1).all() and torch.isfinite(probabilities).all()
    with torch.no_grad(): gate.router.bias.fill_(-100.)
    gate.probabilities(torch.rand(2,16,64)); assert (gate.active_experts==4).all()
    print(json.dumps({'matched_parameters':True,'common_initial_parameters':True,
                      'schedule_only_enters_graph_variants':True,'joint_training':True,
                      'router_gradient_and_normalization':True,'top_any_bounds':True}))


if __name__=='__main__': main()
