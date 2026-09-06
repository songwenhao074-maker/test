"""Validate matched fusion capacity, locality, shared initialization and restoration."""
import io
import json
import torch
from torch import nn
from recovery.PreGANSrc.src.ftmoe_fusion_controls import GatedCrossFusion
from train_ftmoe_end_to_end import create_model


def main():
    torch.set_num_threads(1); torch.manual_seed(501)
    attention = nn.MultiheadAttention(64,4,dropout=.1,batch_first=True).eval()
    rng = torch.get_rng_state().clone(); gated = GatedCrossFusion(attention)
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in attention.parameters())==sum(p.numel() for p in gated.parameters())==16640
    x = torch.randn(2,16,64); g = torch.randn_like(x)
    expected = gated(x,g,g)[0]; changed = g.clone(); changed[0,1]+=2
    actual = gated(x,changed,changed)[0]
    torch.testing.assert_close(expected[0,0],actual[0,0],rtol=0,atol=0)
    torch.testing.assert_close(expected[1],actual[1],rtol=0,atol=0)
    assert not torch.allclose(expected[0,1],actual[0,1])
    gate_model = create_model('v4',1,expert_dropout=.1,fusion='gated').eval()
    full_model = create_model('v4',1,expert_dropout=.1).eval()
    assert sum(p.numel() for p in gate_model.parameters())==sum(p.numel() for p in full_model.parameters())
    original = dict(full_model.named_parameters())
    for name,p in gate_model.named_parameters():
        if not name.startswith('cmha.'):
            torch.testing.assert_close(p,original[name],rtol=0,atol=0)
    data = torch.rand(2,16,12,7); schedule = torch.eye(16)[None,None].expand(2,12,-1,-1)
    with torch.no_grad():
        for key in ['detection_logits','class_logits']:
            torch.testing.assert_close(gate_model(data,schedule,data)[key],full_model(data,schedule,data)[key],rtol=0,atol=0)
        gate_model.cmha_detection_adapter.weight.normal_(std=.1)
    state = {'model':gate_model.state_dict(),'model_options':{'expert_dropout':.1,'fusion':'gated'}}
    memory = io.BytesIO(); torch.save(state,memory); memory.seek(0)
    loaded = torch.load(memory,weights_only=False)
    restored = create_model('v4',1,**loaded['model_options']).eval(); restored.load_state_dict(loaded['model'])
    with torch.no_grad():
        torch.testing.assert_close(restored(data,schedule,data)['detection_logits'],gate_model(data,schedule,data)['detection_logits'],rtol=0,atol=0)
    print(json.dumps({'matched_16640_parameters':True,'rng_preserved':True,'same_host_fusion':True,
                      'unchanged_other_branches':True,'initial_predictions_identical':True,'checkpoint_roundtrip':True}))


if __name__=='__main__': main()
