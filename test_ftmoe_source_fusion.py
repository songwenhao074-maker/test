"""Validate the inactive host-aligned fusion candidate without training data."""
import copy
import io
import json
import torch
from torch import nn
from recovery.PreGANSrc.src.ftmoe_source_fusion import SourceAttentionFusion,SourceGatedFusion
from train_ftmoe_end_to_end import create_model


def main():
    torch.set_num_threads(1); torch.manual_seed(512)
    base = nn.MultiheadAttention(64,4,dropout=.1,batch_first=True).eval()
    state = torch.get_rng_state().clone()
    candidates = [SourceAttentionFusion(copy.deepcopy(base)),SourceGatedFusion(copy.deepcopy(base))]
    assert torch.equal(state,torch.get_rng_state())
    time = torch.randn(2,16,64); graph = torch.randn_like(time)
    altered = graph.clone(); altered[0,1]+=3
    for module in candidates:
        assert sum(p.numel() for p in module.parameters())==16640
        first = module(time,graph,graph)[0]; second = module(time,altered,altered)[0]
        torch.testing.assert_close(first[0,0],second[0,0],rtol=0,atol=0)
        torch.testing.assert_close(first[1],second[1],rtol=0,atol=0)
        assert not torch.allclose(first[0,1],second[0,1])
        permutation = torch.randperm(16)
        permuted = module(time[:,permutation],graph[:,permutation],graph[:,permutation])[0]
        torch.testing.assert_close(first[:,permutation],permuted)
        module.zero_grad(); (first*torch.randn_like(first)).sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
        if isinstance(module,SourceAttentionFusion):
            for projection in module.attention.in_proj_weight.grad.split(64):
                assert projection.abs().sum()>0
    inputs = torch.rand(2,16,12,7); schedule = torch.eye(16)[None,None].expand(2,12,-1,-1)
    original = create_model('v4',1,expert_dropout=.1).eval()
    for fusion in ['source_attention','source_gated']:
        options = {'expert_dropout':.1,'fusion':fusion}
        model = create_model('v4',1,**options).eval()
        with torch.no_grad():
            for key in ['detection_logits','class_logits']:
                torch.testing.assert_close(model(inputs,schedule,inputs)[key],original(inputs,schedule,inputs)[key],rtol=0,atol=0)
            model.cmha_detection_adapter.weight.normal_(std=.1)
        memory = io.BytesIO(); torch.save({'model':model.state_dict(),'model_options':options},memory); memory.seek(0)
        state = torch.load(memory,weights_only=False)
        restored = create_model('v4',1,**state['model_options']).eval(); restored.load_state_dict(state['model'])
        with torch.no_grad():
            torch.testing.assert_close(model(inputs,schedule,inputs)['detection_logits'],restored(inputs,schedule,inputs)['detection_logits'],rtol=0,atol=0)
    print(json.dumps({'matched_16640_parameters':True,'rng_preserved':True,'host_alignment':True,
                      'initial_predictions_and_checkpoint_roundtrip':True,
                      'no_batch_mixing':True,'host_permutation_equivariance':True,'all_projection_gradients':True}))


if __name__=='__main__': main()
